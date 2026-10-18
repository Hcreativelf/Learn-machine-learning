"""超参对比实验：5 折单种子，快速筛掉明显不好的配置。

运行：python experiments.py
结果存在 artifacts/exp_*.json，不写 submission。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

from house_price_v2 import (
    load_data, split_types, make_category_dtype, run_cv,
)

VERSION = "exp"


def lgbm_fit(params):
    def _fit(X_tr, y_tr, X_va, y_va, X_te, seed, cat_cols):
        cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
        model = lgb.LGBMRegressor(random_state=seed, n_jobs=8, verbose=-1, **params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(300, verbose=False)],
                  categorical_feature=cat_idx)
        return model.predict(X_va), model.predict(X_te)
    return _fit


def xgb_fit(params):
    def _fit(X_tr, y_tr, X_va, y_va, X_te, seed, cat_cols):
        model = xgb.XGBRegressor(random_state=seed, n_jobs=8, tree_method="hist",
                                 enable_categorical=True, early_stopping_rounds=300,
                                 eval_metric="rmse", **params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return model.predict(X_va), model.predict(X_te)
    return _fit


def cat_fit(params):
    def _fit(X_tr, y_tr, X_va, y_va, X_te, seed, cat_cols):
        def s(f):
            f = f.copy()
            for c in cat_cols:
                f[c] = f[c].astype(str)
            return f
        model = cb.CatBoostRegressor(random_seed=seed, verbose=0, thread_count=8,
                                     od_type="Iter", cat_features=cat_cols, **params)
        model.fit(s(X_tr), y_tr, eval_set=(s(X_va), y_va), use_best_model=True)
        return model.predict(s(X_va)), model.predict(s(X_te))
    return _fit


def main():
    X, y, X_test, _ = load_data(drop_outliers=True)
    y_log = np.log1p(y.values)
    num_cols, cat_cols = split_types(X)
    X, X_test = make_category_dtype(X, X_test, cat_cols)

    experiments = {
        "lgbm_a": (lgbm_fit(dict(n_estimators=8000, learning_rate=0.02, num_leaves=31,
                                 min_child_samples=5, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0)), {}),
        "lgbm_b": (lgbm_fit(dict(n_estimators=8000, learning_rate=0.02, num_leaves=8,
                                 min_child_samples=10, subsample=0.9, subsample_freq=1,
                                 colsample_bytree=0.7, reg_alpha=0.0, reg_lambda=1.0)), {}),
        "lgbm_c": (lgbm_fit(dict(n_estimators=8000, learning_rate=0.01, num_leaves=63,
                                 max_depth=7, min_child_samples=3, subsample=0.7,
                                 subsample_freq=1, colsample_bytree=0.4,
                                 reg_alpha=0.5, reg_lambda=2.0)), {}),
        "xgb_a": (xgb_fit(dict(n_estimators=8000, learning_rate=0.02, max_depth=5,
                               min_child_weight=2, subsample=0.8, colsample_bytree=0.7,
                               reg_alpha=0.1, reg_lambda=1.0)), {}),
        "xgb_b": (xgb_fit(dict(n_estimators=8000, learning_rate=0.01, max_depth=6,
                               min_child_weight=1, subsample=0.8, colsample_bytree=0.4,
                               reg_alpha=0.5, reg_lambda=2.0)), {}),
        "cat_a": (cat_fit(dict(iterations=4000, learning_rate=0.03, depth=6,
                               l2_leaf_reg=3.0, loss_function="RMSE", od_wait=200)), {}),
        "cat_b": (cat_fit(dict(iterations=6000, learning_rate=0.02, depth=5,
                               l2_leaf_reg=5.0, loss_function="RMSE", od_wait=250)), {}),
    }

    print(f"{'模型':<10}{'RMSE(log)':>12}")
    results = {}
    for name, (fitter, kwargs) in experiments.items():
        _, _, rmse = run_cv(name, X, y_log, X_test, 5, (42,), fitter,
                            cat_cols=cat_cols, **kwargs)
        results[name] = rmse

    print("\n排序:")
    for name, rmse in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {name:<10}{rmse:.6f}")


if __name__ == "__main__":
    main()
