# House Prices：房价预测

Kaggle [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
的本地完整实现。从原 notebook 的 TF-DF 方案出发，做了两件事：

- 一条**教学线**：用 scikit-learn 复刻原 notebook 的每一步，看懂整个流程；
- 一条**提分线**：特征工程 + 多模型融合 + 交叉验证，把验证误差从 0.1327 降到 **0.1065**（对数 RMSE）。

最终提交文件：[`submission_v2.csv`](submission_v2.csv)

## 结果

| 版本 | 做法 | CV 对数 RMSE | 美元 RMSE |
| --- | --- | --- | --- |
| v1 教学版 | 随机森林 300 棵，单次 70/30 划分 | — | 29,601 |
| v1 同口径 | 随机森林 + log 目标，10 折 CV | 0.1327 | 25,327 |
| **v2 最终** | **5 个模型融合，10 折 × 3 种子** | **0.1065** | **19,805** |

> 评测口径见 [下文](#评测口径)。0.1065 是对数空间的 RMSE，也就是排行榜上会看到的分数；
> 19,805 只是乘以房价后便于理解的美元误差。

v2 的单模型表现（10 折 × 3 种子）：

| 模型 | 对数 RMSE | 融合权重 | 训练耗时 |
| --- | --- | --- | --- |
| ElasticNet（独热编码） | 0.1110 | 0.458 | 4 s |
| CatBoost | 0.1112 | 0.222 | 764 s |
| XGBoost | 0.1114 | 0.320 | 52 s |
| Ridge | 0.1126 | 0 | 2 s |
| LightGBM | 0.1141 | 0 | 25 s |

## 快速开始

### 1. 环境

项目已带 `.venv`（Python 3.11，uv 创建），直接用即可。换电脑时重建：

```powershell
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
```

### 2. 数据

到 [竞赛 Data 页](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
点 `Download All`，把三个文件放进 `data\`：

```
data\train.csv
data\test.csv
data\sample_submission.csv
```

### 3. 运行

```powershell
# 环境自检（版本、依赖、数据是否齐全）
.\.venv\Scripts\python.exe check_env.py

# 教学版：单次划分 + 随机森林，约 1 分钟
.\.venv\Scripts\python.exe house_price_sklearn.py

# 提分版：完整融合，约 15 分钟（主要花在 CatBoost）
.\.venv\Scripts\python.exe house_price_v2.py --seeds 42,2024,7 --folds 10
```

提分版跑完会写出 `submission_v2.csv`，上传到 Kaggle 的 **Submit Predictions** 即可。

## 目录结构

```
大作业/
├─ data/                          # Kaggle 数据 + 字段说明
├─ figures/                       # 教学版生成的图
├─ artifacts/                     # v2 的 OOF 缓存（删掉会重训）
├─ .venv/                         # Python 3.11 环境
├─ .vscode/                       # 解释器、调试、任务配置
│
├─ house_price_sklearn.py         # 教学版：单次划分 + 随机森林（# %% 分块）
├─ house_price_v2.py              # 提分版：特征工程 + 5 模型 + 10 折 CV + 融合
├─ check_env.py                   # 环境自检
│
├─ experiments.py                 # 超参对比实验（LGBM / XGB / CatBoost）
├─ ridge_sweep.py                 # 线性模型正则强度扫描
├─ blend.py                       # 融合方式对比，写最终提交
├─ baseline_v1.py                 # v1 同口径基线（证明改进幅度）
│
├─ 优化记录.md                    # 实验记录、结论、复现命令
├─ house-prices-...-中文版.md     # 原 notebook 的中文翻译
├─ requirements.txt
└─ submission_v2.csv              # 最终提交文件
```

## 两个版本

### 教学版 `house_price_sklearn.py`

按原 notebook 的顺序走一遍：加载 → 看分布 → 划分 → 预处理 → 随机森林 → 评估 → 变量重要性 → 梯度提升对比 → 提交。

- 用 `# %%` 分块，VSCode 里按 `Shift+Enter` 逐段运行，结果在交互窗口；
- 预处理器用 `Pipeline` 封装，保证"训练集 fit、验证集 transform"，不泄漏；
- 固定 `RANDOM_STATE = 42`，结果可复现；
- 图上标了 RMSE，生成的图存在 `figures/`。

### 提分版 `house_price_v2.py`

```
log1p 目标 → 特征工程 → 缺失值语义化 → 有序等级编码
          → 5 个模型 × 10 折 × 3 种子 → OOF 融合 → expm1 → 提交
```

七项改进：

| # | 改动 | 作用 |
| --- | --- | --- |
| 1 | 目标取 `log1p` | 对齐评分口径；压掉右偏 |
| 2 | 融合线性模型 | 房价在 log 空间近似线性，独热编码≈街区固定效应，与树互补 |
| 3 | 18 个领域特征 | 总面积、总卫浴、房龄、有无车库/地下室/泳池等 |
| 4 | 缺失值语义化 | `BsmtQual=NA` 表示"没有地下室"，填 `"None"` 而非众数 |
| 5 | 有序等级按语义编码 | `Po<Fa<TA<Gd<Ex`，而不是字母序 |
| 6 | 原生类别模型 | LightGBM / XGBoost / CatBoost 直接吃 category 和 NaN |
| 7 | 10 折 × 3 种子 | 把评估噪声从 ±3% 压到 1% 以内 |

**融合是在 OOF 预测上做的**：每个模型对每条训练数据都有一个"折外"预测，用这些预测拟合正系数权重，避免用验证集本身调权重。

`artifacts/` 缓存了每个模型的 OOF 和测试集预测，所以改融合方式不需要重新训练：

```powershell
.\.venv\Scripts\python.exe blend.py     # 秒级，重算权重并写 submission_v2.csv
```

快速模式（1 个种子、5 折，约 5 分钟）：

```powershell
.\.venv\Scripts\python.exe house_price_v2.py --seeds 42 --folds 5
```

## 评测口径

Kaggle 这个竞赛用的是**对数价格的 RMSE**，不是美元 RMSE：

$$
\text{score} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\log \hat y_i - \log y_i\right)^2}
$$

所以训练目标用 `log1p(SalePrice)`，预测后再 `expm1` 还原。在美元空间直接优化 MSE 会被几套豪宅主导，方向是错的。

## 实验脚本

| 脚本 | 做什么 | 命令 |
| --- | --- | --- |
| `experiments.py` | 三套 GBDT 的超参对比（5 折单种子） | `python experiments.py` |
| `ridge_sweep.py` | Ridge / ElasticNet 正则扫描 | `python ridge_sweep.py` |
| `baseline_v1.py` | v1 同口径基线 | `python baseline_v1.py` |
| `blend.py` | 融合方式对比（平均 / NNLS / 爬山 / 正系数+截距） | `python blend.py` |

详细数据与结论见 [`优化记录.md`](优化记录.md)。

## 环境说明

### 为什么用 `.venv` 而不是 Anaconda 的 base

本机 Anaconda 的 base 是 Python 3.14，而 TensorFlow / scikit-learn 生态一般只跟到 3.12 左右，装在 base 里容易版本冲突、把 Anaconda 搞坏。项目用 `uv` 建了独立环境：不污染系统，删掉文件夹就彻底卸载。

习惯 conda 的话等价做法：

```powershell
conda create -n houseprice python=3.11 -y
conda activate houseprice
pip install -r requirements.txt
```

### 为什么没有 `tensorflow_decision_forests`

PyPI 上 TF-DF 从 1.0.0 到 1.12.0 **只有 Linux 和 macOS 的轮子**，Windows 一个都没有，conda-forge 也没有这个包。所以在 Windows 上 `pip install tensorflow-decision-forests` 必然失败——不是环境问题。

替代方案就是本项目的两条线：教学版用 scikit-learn 复刻同样的算法（随机森林、梯度提升树），提分版用 LightGBM / XGBoost / CatBoost——它们和 TF-DF 一样原生支持类别列和缺失值。

如果确实要用原版 TF-DF，可以走 WSL2：

```powershell
wsl --install -d Ubuntu          # 管理员 PowerShell，需要重启
```

```bash
# Ubuntu 里
sudo apt update && sudo apt install -y python3-venv
cd /mnt/d/1-博士/课程学习/机器学习/作业/大作业
python3 -m venv .venv-wsl && source .venv-wsl/bin/activate
pip install tensorflow-cpu==2.19.0 tensorflow-decision-forests==1.12.0
```

然后在 VSCode 里 `WSL: Reopen Folder in WSL`，选 `.venv-wsl/bin/python`。

## 在 VSCode 里用

| 方式 | 操作 | 场景 |
| --- | --- | --- |
| 交互窗口 | 光标放在 `# %%` 单元内按 `Shift+Enter` | 边写边看、画图、调参 |
| 直接运行 | 右上角 ▶ 或 `Ctrl+F5` | 一次跑完 |
| 调试 | 打红点断点 → `F5` | 看变量、单步跟踪 |
| 终端 | `.\.venv\Scripts\Activate.ps1` 后 `python xxx.py` | 命令行、批量跑 |

首次打开需要选解释器：`Ctrl+Shift+P` → `Python: Select Interpreter` → `.venv\Scripts\python.exe`。
配好之后这个文件夹会自动使用它（配置在 `.vscode/settings.json`）。

## 常见问题

**中文显示成方块**
脚本已设 `Microsoft YaHei`；不行就换成 `SimHei`。

**终端报"禁止运行脚本"（Activate.ps1）**

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**`ModuleNotFoundError: No module named 'sklearn'`**
解释器选错了，重选 `.venv\Scripts\python.exe`。

**CatBoost 太慢**
单折约 25 秒，完整 30 次训练约 12 分钟，属正常。想快就用 `--seeds 42 --folds 5`，或者 `--models enet,ridge` 只跑线性模型。

**提交报格式错误**
文件必须是两列 `Id,SalePrice`、1459 行、无索引列。`to_csv(..., index=False)` 不能省。

## 参考

- 教材：周志华《机器学习》第 4、8 章；李航《统计学习方法》第 5、8 章；*The Elements of Statistical Learning* 第 9、10、15 章
- 视频：StatQuest 的 Decision Trees / Random Forests / Gradient Boost 系列；李宏毅《机器学习》Ensemble 一讲
- 论文：Breiman (2001) *Random Forests*；Friedman (2001) *Greedy Function Approximation*；Chen & Guestrin (2016) *XGBoost*
- 竞赛讨论区：搜 "House Prices top solution"，看别人的特征工程思路

