# %% [markdown]
# # 房价预测：Windows 原生版（scikit-learn）
#
# 这个文件把 `house-prices-prediction-using-tfdf-中文版.md` 的流程原样走了一遍，
# 只是把 TF-DF 换成了 scikit-learn —— 因为 TF-DF 没有 Windows 安装包。
#
# 推荐用法（不用装 Jupyter Notebook 也能有 notebook 的体验）：
# 1. 在 VSCode 里选中 `.venv\Scripts\python.exe` 解释器；
# 2. 点击每个 `# %%` 上方的「Run Cell」，或把光标放在单元内按 `Shift+Enter`；
# 3. 结果会出现在右侧的 Interactive Window（交互窗口）里。
#
# 也可以一次性整篇运行：在 VSCode 终端里执行
#   .\.venv\Scripts\python.exe house_price_sklearn.py

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# 交互窗口里没有 __file__，退回到当前工作目录
try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd()

DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42  # 固定随机种子，保证每次跑出来的结果可复现

# 终端和交互窗口统一用 UTF-8，避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 让 matplotlib 能正常显示中文（Windows 自带微软雅黑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

train_path = DATA_DIR / "train.csv"
if not train_path.exists():
    raise FileNotFoundError(
        f"没找到 {train_path}\n"
        "请到 Kaggle 竞赛页 Data 标签下载并解压，把 train.csv / test.csv / "
        "sample_submission.csv 放进 data 目录。"
    )

# %% [markdown]
# ## 1. 加载数据
#
# 和 notebook 一样：81 列、1460 行，去掉与预测无关的 `Id` 列。

# %%
dataset_df = pd.read_csv(train_path)
print(f"训练集形状: {dataset_df.shape}")  # 应为 (1460, 81)
dataset_df = dataset_df.drop("Id", axis=1)
print(f"去掉 Id 后: {dataset_df.shape}")

dataset_df.head(3)

# %%
dataset_df.info()

# %% [markdown]
# ## 2. 房价分布

# %%
print(dataset_df["SalePrice"].describe())
plt.figure(figsize=(9, 5))
sns.histplot(dataset_df["SalePrice"], color="g", bins=100, alpha=0.4, kde=True)
plt.title("SalePrice 分布")
plt.xlabel("SalePrice")
plt.savefig(FIG_DIR / "01_saleprice_dist.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. 数值型特征的分布

# %%
df_num = dataset_df.select_dtypes(include="number")
print(f"数值型特征数量: {df_num.shape[1]}")

df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_numeric_hist.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. 划分训练集 / 验证集
#
# 与 notebook 一致：留 30% 做验证。区别是这里固定了 `RANDOM_STATE`，
# 方便复现和调参对比（notebook 每次运行划分都不一样）。

# %%
label = "SalePrice"
rng = np.random.default_rng(RANDOM_STATE)
is_valid = rng.random(len(dataset_df)) < 0.30
train_ds_pd, valid_ds_pd = dataset_df[~is_valid], dataset_df[is_valid]
print(f"训练 {len(train_ds_pd)} 条，验证 {len(valid_ds_pd)} 条")

X_train, y_train = train_ds_pd.drop(columns=label), train_ds_pd[label]
X_valid, y_valid = valid_ds_pd.drop(columns=label), valid_ds_pd[label]

# %% [markdown]
# ### 与 TF-DF 最大的差别：类别型特征要自己处理
#
# TF-DF 能直接吃字符串列和缺失值，scikit-learn 不行。所以这里用
# `ColumnTransformer` 搭一条预处理流水线：
#
# * 数值列 → 用中位数填补缺失值；
# * 类别列（字符串）→ 用众数填补，再按出现顺序编码成整数。
#
# 编码器统一用 `handle_unknown="use_encoded_value"`，这样测试集里出现
# 训练集没见过的类别也不会报错。

# %%
numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
print(f"数值特征 {len(numeric_cols)} 个，类别特征 {len(categorical_cols)} 个")

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        (
            "cat",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "encode",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                ]
            ),
            categorical_cols,
        ),
    ],
    remainder="drop",
)

# %% [markdown]
# ## 5. 训练随机森林
#
# 对应 notebook 里的 `tfdf.keras.RandomForestModel()`。`n_estimators=300`
# 与 TF-DF 默认的 300 棵树对齐；`oob_score=True` 打开袋外评估，
# 正好对应 notebook 讲的 OOB 得分。

# %%
rf = Pipeline(
    steps=[
        ("prep", preprocess),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                oob_score=True,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)
rf.fit(X_train, y_train)
print("训练完成。袋外 R² =", round(rf.named_steps["model"].oob_score_, 4))

# %% [markdown]
# ## 6. 评估：RMSE
#
# notebook 用 MSE，这里换成 RMSE（单位是美元，比 MSE 好解释）。
# 注意验证集 RMSE 通常比 OOB 略差，这是正常的。

# %%
pred_valid = rf.predict(X_valid)
rmse = float(np.sqrt(mean_squared_error(y_valid, pred_valid)))
mae = float(mean_absolute_error(y_valid, pred_valid))
print(f"验证集 RMSE: {rmse:,.0f}")
print(f"验证集 MAE : {mae:,.0f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_valid, pred_valid, s=12, alpha=0.6)
lims = [0, max(y_valid.max(), pred_valid.max()) * 1.05]
plt.plot(lims, lims, "r--", linewidth=1, label="理想预测线")
plt.xlabel("真实售价")
plt.ylabel("预测售价")
plt.title(f"验证集预测 vs 真实（RMSE = {rmse:,.0f}）")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "03_pred_vs_true.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. 变量重要性
#
# 对应 notebook 的 `NUM_AS_ROOT`。随机森林这里用的是"不纯度下降"，
# 含义不同但作用类似：看哪些特征对模型贡献最大。

# %%
feature_names = numeric_cols + categorical_cols
importances = rf.named_steps["model"].feature_importances_
order = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(10, 6))
plt.barh(range(len(order)), importances[order][::-1], color="steelblue")
plt.yticks(range(len(order)), [feature_names[i] for i in order][::-1])
plt.xlabel("重要性（不纯度下降）")
plt.title("前 20 个重要特征")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_importance.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. 对比一下梯度提升树
#
# notebook 提到 `GradientBoostedTreesModel` 也是一个选项。
# scikit-learn 的 `HistGradientBoostingRegressor` 有一个很香的特性和 TF-DF 类似：
# **原生支持类别型特征和缺失值**，不用做上面的编码和填补。
# 只要把字符串列转成 `category` 类型，并设置 `categorical_features="from_dtype"`。

# %%
X_train_cat = X_train.copy()
X_valid_cat = X_valid.copy()
for col in categorical_cols:
    X_train_cat[col] = X_train_cat[col].astype("category")
    X_valid_cat[col] = X_valid_cat[col].astype("category")

gbt = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=500,
    categorical_features="from_dtype",
    random_state=RANDOM_STATE,
)
gbt.fit(X_train_cat, y_train)
gbt_rmse = float(np.sqrt(mean_squared_error(y_valid, gbt.predict(X_valid_cat))))
print(f"随机森林 RMSE      : {rmse:,.0f}")
print(f"梯度提升树 RMSE    : {gbt_rmse:,.0f}")

# %% [markdown]
# ## 9. 生成提交文件
#
# 训练集重新拟合后预测 test.csv，写出的 `submission.csv` 可以直接上传 Kaggle。
# 记得 `Id` 必须保留，列名必须是 `Id,SalePrice`。

# %%
test_path = DATA_DIR / "test.csv"
if not test_path.exists():
    raise FileNotFoundError(f"没找到 {test_path}，请把 test.csv 放进 data 目录")

test_df = pd.read_csv(test_path)
test_ids = test_df.pop("Id")

final_model = Pipeline(
    steps=[
        ("prep", preprocess),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
            ),
        ),
    ]
)
final_model.fit(dataset_df.drop(columns=label), dataset_df[label])

submission = pd.DataFrame(
    {"Id": test_ids, "SalePrice": final_model.predict(test_df)}
)
submission.to_csv(PROJECT_DIR / "submission.csv", index=False)
print(submission.head())
print(f"已写出 {PROJECT_DIR / 'submission.csv'}，共 {len(submission)} 行")
