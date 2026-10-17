# 房价预测（Kaggle House Prices）—— 在本机跑起来

## 0. 先说清楚一件事：TF-DF 装了不上去

原 notebook 的核心是 `tensorflow_decision_forests`（TF-DF）。我查了 PyPI 上
1.0.0 到 1.12.0 的**全部**发行文件，只有：

```
...-cp311-cp311-manylinux_2_17_x86_64.whl     ← Linux
...-cp311-cp311-macosx_12_0_arm64.whl         ← macOS
```

Windows 一个都没有，conda-forge 也没有这个包。所以在这台 Windows 上
`pip install tensorflow-decision-forests` 一定会失败，`import tensorflow_decision_forests`
也一定报 `ModuleNotFoundError`。这不是你环境没配好。

于是有两条路：

| 路线 | 代价 | 适合 |
| --- | --- | --- |
| **A. 本机 Windows + scikit-learn**（已给你搭好，现在就能跑） | 换框架，方法论完全一样 | 想马上动手、写作业、交报告 |
| **B. WSL2 里跑原版 TF-DF** | 要装 WSL、重启、下载约 1GB | 老师要求必须用 TF-DF |

---

## 1. 已经准备好的东西

| 文件 | 作用 |
| --- | --- |
| `.venv\` | 项目专用 Python 3.11 环境（uv 创建，已装好依赖） |
| `requirements.txt` | 依赖清单，换电脑时 `uv pip install -r requirements.txt` 即可重建 |
| `check_env.py` | 环境自检：解释器、版本、数据是否齐全 |
| `house_price_sklearn.py` | 按 notebook 分节的完整流程（`# %%` 单元格，可逐段运行） |
| `figures\` | 脚本跑出来的图会自动存这里 |
| `.vscode\` | 解释器、调试、任务配置，打开文件夹就生效 |

> 你的 `Housing-priceforecast.py` 原样保留了，它在本机会停在
> `import tensorflow_decision_forests` 这一行——原因见第 0 节。
> 想跑同样的东西，用 `house_price_sklearn.py`。

---

## 2. 三步跑起来

**第一步：确认解释器**

打开文件夹后按 `Ctrl+Shift+P` → 输入 `Python: Select Interpreter` →
选 `.venv\Scripts\python.exe`（路径里带 `.venv` 的那个）。
选一次就记住了，以后打开这个文件夹自动用它。

**第二步：放数据**

到 [竞赛 Data 页](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
点 `Download All`，解压后把这三个文件放进 `data\`：

```
data\train.csv
data\test.csv
data\sample_submission.csv
```

**第三步：运行**

打开 `house_price_sklearn.py`，按 `Shift+Enter` 一段一段跑（右侧出现交互窗口），
或直接点右上角 ▶ 一次跑完。跑完会生成 `submission.csv`，可以上传 Kaggle。

跑之前想先确认环境没问题？打开 `check_env.py` 按 `Ctrl+F5`。

---

## 3. VSCode 里运行 Python 的四种方式

| 方式 | 操作 | 适合场景 |
| --- | --- | --- |
| **交互窗口**（最像 notebook） | 光标放在 `# %%` 单元里按 `Shift+Enter` | 边写边看结果、画图、调参 |
| **直接运行** | 右上角 ▶ 或 `Ctrl+F5` | 脚本已经写好了，一次跑完 |
| **调试** | 打红点断点 → `F5` | 想看某个变量的值、单步跟踪 |
| **终端** | `.\.venv\Scripts\Activate.ps1` 后 `python xxx.py` | 命令行、批量跑 |

`Shift+Enter` 需要 Jupyter 扩展（已装好 `ms-toolsai.jupyter`）。第一次运行会让你选内核，
选 `.venv (Python 3.11.16)` 那个。

调试配置在 `.vscode\launch.json`，按 `F5` 后左上角下拉框可以选「当前文件」或「房价预测（sklearn）」。

---

## 4. 为什么不直接用 Anaconda

你机器上 Anaconda 的 base 是 **Python 3.14**，而 TensorFlow / scikit-learn 这类库
通常只跟到 3.12 左右，直接在 base 里装很容易踩版本冲突，把 Anaconda 本身搞坏。

所以这里用 `uv` 在项目里建了独立的 `.venv`：不污染 Anaconda，删掉文件夹就彻底卸载，
而且 uv 解析和下载都比 conda 快得多。

如果你更习惯 conda，等价做法是：

```powershell
conda create -n houseprice python=3.11 -y
conda activate houseprice
pip install -r requirements.txt
```

然后在 VSCode 里选 `houseprice` 这个环境即可。两种方式选一种就行。

---

## 5. 概念对照：notebook 里的 TF-DF → sklearn

| notebook（TF-DF） | 本脚本（scikit-learn） | 说明 |
| --- | --- | --- |
| `tfdf.keras.RandomForestModel()` | `RandomForestRegressor(n_estimators=300)` | 都是 300 棵树 |
| 自动处理类别列和缺失值 | `ColumnTransformer` + `SimpleImputer` + `OrdinalEncoder` | **最大的差别**，sklearn 要自己写预处理 |
| `pd_dataframe_to_tf_dataset()` | 直接传 DataFrame | sklearn 不需要转换 |
| OOB 得分 `inspector.evaluation()` | `oob_score=True` | 一样是袋外评估 |
| `NUM_AS_ROOT` 变量重要性 | `feature_importances_` | 一个是"当根节点次数"，一个是"不纯度下降"，都用于看特征重要性 |
| `GradientBoostedTreesModel` | `HistGradientBoostingRegressor` | 后者也原生支持类别列和缺失值，最接近 TF-DF 的体验 |
| `rf.predict()` 写 `submission.csv` | 同 | 提交格式必须 `Id,SalePrice` |

几个值得注意的细节：

- 原 notebook 划分训练/验证集**没有固定随机种子**，每次结果都不一样；脚本里用
  `RANDOM_STATE = 42` 固定住，方便对比调参效果。
- 原 notebook 用 MSE，脚本换成 **RMSE**（开方后单位是美元，好解释）。
- 原始数据里 `SalePrice` 是右偏分布，进阶做法是对它取对数再训练
  （`np.log1p`/`np.expm1`），Kaggle 上能明显提分，可以自己试。

---

## 6. 路线 B：在 WSL2 里跑原版 TF-DF

需要管理员权限和一次重启，**不影响**上面已经能用的 Windows 环境。

```powershell
# 1) 管理员 PowerShell 里安装 WSL（默认 Ubuntu）
wsl --install -d Ubuntu
# 重启电脑，按提示设置 Linux 用户名和密码
```

装好后，在 Ubuntu 终端里：

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
cd /mnt/d/1-博士/课程学习/机器学习/作业/大作业     # 直接进入 Windows 上的这个文件夹
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
pip install tensorflow-cpu==2.19.0 tensorflow-decision-forests==1.12.0
```

然后 VSCode 装 `WSL` 扩展，`Ctrl+Shift+P` → `WSL: Reopen Folder in WSL`，
再选 `.venv-wsl/bin/python` 作为解释器，就能原样运行 notebook 了。

> 不想折腾 WSL 的话，[Kaggle Notebook](https://www.kaggle.com/code) 或
> Google Colab 里 TF-DF 是预装好的，直接把 notebook 传上去点 Run All 即可。

---

## 7. 常见问题

**中文显示成方块？**
脚本里已经设了 `Microsoft YaHei`；如果还不行，改成 `SimHei`。

**`plt.show()` 弹窗一堆？**
在交互窗口里跑不会弹窗。整篇运行时可以用 `Ctrl+C` 中断，图已经存到 `figures\` 了。

**终端提示 `无法加载文件 Activate.ps1，因为在此系统上禁止运行脚本`？**
执行一次：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

**`ModuleNotFoundError: No module named 'sklearn'`？**
解释器选错了，回到第 2 步重选 `.venv\Scripts\python.exe`。

**内存不够 / 跑得慢？**
`n_jobs=-1` 会用满所有核心；想限制就改成 `n_jobs=4`。
