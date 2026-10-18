"""房价预测 v2：面向 Kaggle 评分（对数 RMSE）的优化版流水线。

相比 v1（单次 70/30 + 随机森林）的改进：

1. 目标做 log1p 变换 —— Kaggle 的评分就是对 log(SalePrice) 算 RMSE，
   直接在对数空间优化，顺便解决原始分布右偏的问题。
2. 领域特征工程 —— 总面积、总卫浴、房龄、有无地下室/车库/泳池等。
3. 缺失值语义化 —— NA 在原始数据里大多表示"没有"，填 "None" 而不是众数。
4. 有序质量等级按语义排序 —— Po<Fa<TA<Gd<Ex，而不是字母序。
5. 三棵梯度提升树 + 岭回归，10 折交叉验证 × 多随机种子。
6. 用 OOF 预测做非负最小二乘融合，权重由数据决定。

用法：
    python house_price_v2.py                # 完整跑（3 seeds × 10 folds）
    python house_price_v2.py --seeds 42     # 快速跑（1 seed）
    python house_price_v2.py --models lgbm,xgb
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd()

DATA_DIR = PROJECT_DIR / "data"
ART_DIR = PROJECT_DIR / "artifacts"
ART_DIR.mkdir(exist_ok=True)

VERSION = "v2.2"

# ---------------------------------------------------------------- 数据与特征

# 这些列的缺失表示"根本没有这个东西"，填 "None" 是正确语义
ABSENT_AS_NONE = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]

# 有序质量等级：必须按真实顺序编码，字母序会打乱语义
ORDINAL_MAPS = {
    "ExterQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "HeatingQC": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "PoolQC": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """在 train+test 合并后的表上做特征工程，保证两边列完全对齐。"""
    df = df.copy()

    for col in ABSENT_AS_NONE:
        if col in df.columns:
            df[col] = df[col].astype("object").where(df[col].notna(), "None")

    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype(float)

    # ---- 领域特征 ----
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (
        df["FullBath"] + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"]
        + df["ScreenPorch"] + df["WoodDeckSF"]
    )
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
    df["IsRemod"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasDeck"] = (df["WoodDeckSF"] > 0).astype(int)

    df["QualSF"] = df["OverallQual"] * df["TotalSF"]
    df["QualArea"] = df["OverallQual"] * df["GrLivArea"]
    df["AreaRatio"] = df["GrLivArea"] / df["LotArea"].replace(0, np.nan)
    df["BsmtFinRatio"] = df["BsmtFinSF1"] / df["TotalBsmtSF"].replace(0, np.nan)
    df["GarageScore"] = df["GarageCars"] * df["GarageArea"]

    return df


def load_data(drop_outliers: bool = True):
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    y = train["SalePrice"].astype(float)
    ids = test["Id"].copy()
    train = train.drop(columns=["Id", "SalePrice"])
    test = test.drop(columns=["Id"])

    if drop_outliers:
        # 训练集里两个著名的异常点：4000+ 平方英尺却只卖十几万
        mask = (train["GrLivArea"] > 4000) & (y < 300_000)
        train = train[~mask]
        y = y[~mask]
        print(f"[数据] 去掉 {int(mask.sum())} 个异常样本")

    n_train = len(train)
    all_df = pd.concat([train, test], axis=0, ignore_index=True)
    all_df = build_features(all_df)

    X = all_df.iloc[:n_train].reset_index(drop=True)
    X_test = all_df.iloc[n_train:].reset_index(drop=True)
    return X, y.reset_index(drop=True), X_test, ids.reset_index(drop=True)


def split_types(X: pd.DataFrame):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def make_category_dtype(X: pd.DataFrame, X_test: pd.DataFrame, cat_cols):
    for col in cat_cols:
        # 统一成字符串并显式给出缺失类别，避免 pandas 3 的 NA 与 str 混排
        s_tr = X[col].fillna("__NA__").astype(str)
        s_te = X_test[col].fillna("__NA__").astype(str)
        X[col] = s_tr
        X_test[col] = s_te
        categories = pd.Index(sorted(set(s_tr) | set(s_te)))
        X[col] = pd.Categorical(s_tr, categories=categories)
        X_test[col] = pd.Categorical(s_te, categories=categories)
    return X, X_test


# ---------------------------------------------------------------- 各模型

def fit_lgbm(X_tr, y_tr, X_va, y_va, X_te, seed, cat_cols):
    # 用列索引而不是列名：LightGBM 在 eval_X 接口下按名字解析会失效
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=8000,
        learning_rate=0.01,
        num_leaves=16,
        min_child_samples=8,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(300, verbose=False)],
        categorical_feature=cat_idx,
    )
    return model.predict(X_va), model.predict(X_te)


def fit_xgb(X_tr, y_tr, X_va, y_va, X_te, seed, cat_cols):
    model = xgb.XGBRegressor(
        n_estimators=8000,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        random_state=seed,
        n_jobs=4,
        early_stopping_rounds=300,
        eval_metric="rmse",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model.predict(X_va), model.predict(X_te)


def fit_cat(X_tr, y_tr, X_va, y_va, X_te, seed, cat_cols):
    def as_str(frame):
        frame = frame.copy()
        for col in cat_cols:
            frame[col] = frame[col].astype(str)
        return frame

    model = cb.CatBoostRegressor(
        iterations=6000,
        learning_rate=0.02,
        depth=5,
        l2_leaf_reg=5.0,
        loss_function="RMSE",
        random_seed=seed,
        od_type="Iter",
        od_wait=250,
        verbose=0,
        thread_count=8,
        cat_features=cat_cols,
    )
    model.fit(as_str(X_tr), y_tr, eval_set=(as_str(X_va), y_va), use_best_model=True)
    return model.predict(as_str(X_va)), model.predict(as_str(X_te))


def build_ridge(num_cols, cat_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=3), cat_cols),
        ]
    )
    return Pipeline([("pre", pre), ("model", Ridge(alpha=30.0))])


def fit_ridge(X_tr, y_tr, X_va, y_va, X_te, seed, cat_cols, num_cols):
    model = build_ridge(num_cols, cat_cols)
    model.fit(X_tr, y_tr)
    return model.predict(X_va), model.predict(X_te)


def fit_enet(X_tr, y_tr, X_va, y_va, X_te, seed, cat_cols, num_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=3), cat_cols),
    ])
    model = Pipeline([("pre", pre),
                      ("model", ElasticNet(alpha=5e-4, l1_ratio=0.5, max_iter=20000))])
    model.fit(X_tr, y_tr)
    return model.predict(X_va), model.predict(X_te)


# ---------------------------------------------------------------- 交叉验证

def run_cv(name, X, y_log, X_test, folds, seeds, fitter, **kwargs):
    cache_path = ART_DIR / f"{VERSION}_{name}.npz"
    meta_path = ART_DIR / f"{VERSION}_{name}.json"
    if cache_path.exists() and meta_path.exists():
        data = np.load(cache_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("folds") == folds and meta.get("seeds") == list(seeds):
            print(f"[缓存] {name:<10} rmse={meta['rmse']:.6f}")
            return data["oof"], data["test"], meta["rmse"]

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    n_fit = 0
    t0 = time.time()

    for seed in seeds:
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        for tr_idx, va_idx in kf.split(X):
            p_va, p_te = fitter(
                X.iloc[tr_idx], y_log[tr_idx],
                X.iloc[va_idx], y_log[va_idx],
                X_test, seed, **kwargs,
            )
            oof[va_idx] += p_va
            test_pred += p_te
            n_fit += 1

    oof /= len(seeds)
    test_pred /= n_fit
    rmse = float(np.sqrt(mean_squared_error(y_log, oof)))
    np.savez_compressed(cache_path, oof=oof, test=test_pred)
    meta_path.write_text(json.dumps({
        "name": name, "rmse": rmse, "folds": folds,
        "seeds": list(seeds), "n_fit": n_fit,
        "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] {name:<10} rmse={rmse:.6f}  ({n_fit} 次拟合, {time.time()-t0:.0f}s)")
    return oof, test_pred, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,2024,7")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--models", default="lgbm,xgb,cat,ridge,enet")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.no_cache:
        for p in ART_DIR.glob(f"{VERSION}_*"):
            p.unlink()

    X, y, X_test, ids = load_data(drop_outliers=True)
    y_log = np.log1p(y.values)
    np.save(ART_DIR / "y_log.npy", y_log)

    num_cols, cat_cols = split_types(X)
    X, X_test = make_category_dtype(X, X_test, cat_cols)
    print(f"[特征] 数值 {len(num_cols)} 个，类别 {len(cat_cols)} 个，总计 {X.shape[1]} 列")
    print(f"[CV] {args.folds} 折 × {len(seeds)} 个种子 = {args.folds * len(seeds)} 次拟合/模型\n")

    oofs, tests, scores = {}, {}, {}

    if "lgbm" in models:
        oofs["lgbm"], tests["lgbm"], scores["lgbm"] = run_cv(
            "lgbm", X, y_log, X_test, args.folds, seeds, fit_lgbm, cat_cols=cat_cols)
    if "xgb" in models:
        oofs["xgb"], tests["xgb"], scores["xgb"] = run_cv(
            "xgb", X, y_log, X_test, args.folds, seeds, fit_xgb, cat_cols=cat_cols)
    if "cat" in models:
        oofs["cat"], tests["cat"], scores["cat"] = run_cv(
            "cat", X, y_log, X_test, args.folds, seeds, fit_cat, cat_cols=cat_cols)
    if "ridge" in models:
        oofs["ridge"], tests["ridge"], scores["ridge"] = run_cv(
            "ridge", X, y_log, X_test, args.folds, seeds, fit_ridge,
            cat_cols=cat_cols, num_cols=num_cols)
    if "enet" in models:
        oofs["enet"], tests["enet"], scores["enet"] = run_cv(
            "enet", X, y_log, X_test, args.folds, seeds, fit_enet,
            cat_cols=cat_cols, num_cols=num_cols)

    names = list(oofs)
    print("\n单模型（对数空间 RMSE，越小越好）:")
    for n in sorted(names, key=lambda k: scores[k]):
        print(f"  {n:<8} {scores[n]:.6f}")

    A = np.column_stack([oofs[n] for n in names])
    B = np.column_stack([tests[n] for n in names])

    # 简单平均
    avg_rmse = float(np.sqrt(mean_squared_error(y_log, A.mean(axis=1))))

    # 非负最小二乘：权重 >= 0，由 OOF 决定
    weights, _ = nnls(A, y_log)
    blend_rmse = float(np.sqrt(mean_squared_error(y_log, A @ weights)))

    # 归一化权重（让预测的量级不偏）
    w_norm = weights / weights.sum()
    blend_norm_rmse = float(np.sqrt(mean_squared_error(y_log, A @ w_norm)))

    candidates = {
        "平均": (avg_rmse, B.mean(axis=1)),
        "NNLS": (blend_rmse, B @ weights),
        "NNLS归一": (blend_norm_rmse, B @ w_norm),
    }
    best_name = min(candidates, key=lambda k: candidates[k][0])
    best_rmse, best_log_pred = candidates[best_name]

    print("\n融合:")
    for k, (v, _) in candidates.items():
        print(f"  {k:<8} {v:.6f}")
    print(f"  → 采用 {best_name}，权重 {dict(zip(names, np.round(weights, 4)))}")

    pred = np.clip(np.expm1(best_log_pred), 1.0, None)
    sub = pd.DataFrame({"Id": ids, "SalePrice": pred})
    sub.to_csv(PROJECT_DIR / "submission_v2.csv", index=False)

    print(f"\n对比：")
    print(f"  单模型最好   RMSE(log) = {min(scores.values()):.6f}")
    print(f"  融合后       RMSE(log) = {best_rmse:.6f}")
    print(f"  预测价格范围 {pred.min():,.0f} ~ {pred.max():,.0f}")
    print(f"  已写出 submission_v2.csv（{len(sub)} 行）")


if __name__ == "__main__":
    main()


