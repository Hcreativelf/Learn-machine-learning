# 使用 TensorFlow Decision Forests 进行房价预测

本 notebook 将带你一步步了解如何使用 TensorFlow Decision Forests，在本竞赛提供的房价（House Prices）数据集上训练一个基线随机森林（Random Forest）模型。

大致来说，代码结构如下：

```python
import tensorflow_decision_forests as tfdf
import pandas as pd

dataset = pd.read_csv("project/dataset.csv")
tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")

model = tfdf.keras.RandomForestModel()
model.fit(tf_dataset)

print(model.summary())
```

决策森林（Decision Forests）是一类基于树的模型，包括随机森林（Random Forests）和梯度提升树（Gradient Boosted Trees）。在处理表格型数据时，它们是入手的最佳选择，并且在你开始尝试神经网络之前，它们往往能取得更好的效果（或提供一个很强的基线）。

## 导入库

```python
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Comment this if the data visualisations doesn't work on your side
%matplotlib inline
```

```python
print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)
```

```
TensorFlow v2.11.0
TensorFlow Decision Forests v1.2.0
```

## 加载数据集

```python
train_file_path = "../input/house-prices-advanced-regression-techniques/train.csv"
dataset_df = pd.read_csv(train_file_path)
print("Full train dataset shape is {}".format(dataset_df.shape))
```

```
Full train dataset shape is (1460, 81)
```

数据由 81 列和 1460 条记录组成。我们可以通过以下代码打印前 3 条记录，来查看数据集的全部 81 个维度：

```python
dataset_df.head(3)
```

```
   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
0   1          60       RL         65.0     8450   Pave   NaN      Reg
1   2          20       RL         80.0     9600   Pave   NaN      Reg
2   3          60       RL         68.0    11250   Pave   NaN      IR1

  LandContour Utilities  ... PoolArea PoolQC Fence MiscFeature MiscVal MoSold  \
0         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2
1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      5
2         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      9

  YrSold  SaleType  SaleCondition  SalePrice
0   2008        WD         Normal     208500
1   2007        WD         Normal     181500
2   2008        WD         Normal     223500

[3 rows x 81 columns]
```

* 共有 79 个特征列。你的模型需要使用这些特征来预测房屋售价，即标签列 `SalePrice` 所表示的数值。

我们将删除 `Id` 列，因为它对模型训练不是必需的。

```python
dataset_df = dataset_df.drop('Id', axis=1)
dataset_df.head(3)
```

```
   MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
0          60       RL         65.0     8450   Pave   NaN      Reg
1          20       RL         80.0     9600   Pave   NaN      Reg
2          60       RL         68.0    11250   Pave   NaN      IR1

  LandContour Utilities LotConfig  ... PoolArea PoolQC Fence MiscFeature  \
0         Lvl    AllPub    Inside  ...        0    NaN   NaN         NaN
1         Lvl    AllPub       FR2  ...        0    NaN   NaN         NaN
2         Lvl    AllPub    Inside  ...        0    NaN   NaN         NaN

  MiscVal MoSold  YrSold  SaleType  SaleCondition  SalePrice
0       0      2    2008        WD         Normal     208500
1       0      5    2007        WD         Normal     181500
2       0      9    2008        WD         Normal     223500

[3 rows x 80 columns]
```

我们可以使用以下代码查看各特征列的类型：

```python
dataset_df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 80 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   MSSubClass     1460 non-null   int64
 1   MSZoning       1460 non-null   object
 2   LotFrontage    1201 non-null   float64
 3   LotArea        1460 non-null   int64
 4   Street         1460 non-null   object
 5   Alley          91 non-null     object
 6   LotShape       1460 non-null   object
 7   LandContour    1460 non-null   object
 8   Utilities      1460 non-null   object
 9   LotConfig      1460 non-null   object
 10  LandSlope      1460 non-null   object
 11  Neighborhood   1460 non-null   object
 12  Condition1     1460 non-null   object
 13  Condition2     1460 non-null   object
 14  BldgType       1460 non-null   object
 15  HouseStyle     1460 non-null   object
 16  OverallQual    1460 non-null   int64
 17  OverallCond    1460 non-null   int64
 18  YearBuilt      1460 non-null   int64
 19  YearRemodAdd   1460 non-null   int64
 20  RoofStyle      1460 non-null   object
 21  RoofMatl       1460 non-null   object
 22  Exterior1st    1460 non-null   object
 23  Exterior2nd    1460 non-null   object
 24  MasVnrType     1452 non-null   object
 25  MasVnrArea     1452 non-null   float64
 26  ExterQual      1460 non-null   object
 27  ExterCond      1460 non-null   object
 28  Foundation     1460 non-null   object
 29  BsmtQual       1423 non-null   object
 30  BsmtCond       1423 non-null   object
 31  BsmtExposure   1422 non-null   object
 32  BsmtFinType1   1423 non-null   object
 33  BsmtFinSF1     1460 non-null   int64
 34  BsmtFinType2   1422 non-null   object
 35  BsmtFinSF2     1460 non-null   int64
 36  BsmtUnfSF      1460 non-null   int64
 37  TotalBsmtSF    1460 non-null   int64
 38  Heating        1460 non-null   object
 39  HeatingQC      1460 non-null   object
 40  CentralAir     1460 non-null   object
 41  Electrical     1459 non-null   object
 42  1stFlrSF       1460 non-null   int64
 43  2ndFlrSF       1460 non-null   int64
 44  LowQualFinSF   1460 non-null   int64
 45  GrLivArea      1460 non-null   int64
 46  BsmtFullBath   1460 non-null   int64
 47  BsmtHalfBath   1460 non-null   int64
 48  FullBath       1460 non-null   int64
 49  HalfBath       1460 non-null   int64
 50  BedroomAbvGr   1460 non-null   int64
 51  KitchenAbvGr   1460 non-null   int64
 52  KitchenQual    1460 non-null   object
 53  TotRmsAbvGrd   1460 non-null   int64
 54  Functional     1460 non-null   object
 55  Fireplaces     1460 non-null   int64
 56  FireplaceQu    770 non-null    object
 57  GarageType     1379 non-null   object
 58  GarageYrBlt    1379 non-null   float64
 59  GarageFinish   1379 non-null   object
 60  GarageCars     1460 non-null   int64
 61  GarageArea     1460 non-null   int64
 62  GarageQual     1379 non-null   object
 63  GarageCond     1379 non-null   object
 64  PavedDrive     1460 non-null   object
 65  WoodDeckSF     1460 non-null   int64
 66  OpenPorchSF    1460 non-null   int64
 67  EnclosedPorch  1460 non-null   int64
 68  3SsnPorch      1460 non-null   int64
 69  ScreenPorch    1460 non-null   int64
 70  PoolArea       1460 non-null   int64
 71  PoolQC         7 non-null      object
 72  Fence          281 non-null    object
 73  MiscFeature    54 non-null     object
 74  MiscVal        1460 non-null   int64
 75  MoSold         1460 non-null   int64
 76  YrSold         1460 non-null   int64
 77  SaleType       1460 non-null   object
 78  SaleCondition  1460 non-null   object
 79  SalePrice      1460 non-null   int64
dtypes: float64(3), int64(34), object(43)
memory usage: 912.6+ KB
```

## 房价分布

现在让我们来看一下房价是如何分布的。

```python
print(dataset_df['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(dataset_df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});
```

（输出为统计描述信息与分布直方图，内容过大，此处省略。）

## 数值型数据分布

现在，我们来看一下数值型特征是如何分布的。为此，让我们先列出数据集中所有数据的类型，并只选择数值型的特征。

```python
list(set(dataset_df.dtypes.tolist()))
```

```
[dtype('O'), dtype('int64'), dtype('float64')]
```

```python
df_num = dataset_df.select_dtypes(include = ['float64', 'int64'])
df_num.head()
```

```
   MSSubClass  LotFrontage  LotArea  OverallQual  OverallCond  YearBuilt  \
0          60         65.0     8450            7            5       2003
1          20         80.0     9600            6            8       1976
2          60         68.0    11250            7            5       2001
3          70         60.0     9550            7            5       1915
4          60         84.0    14260            8            5       2000

   YearRemodAdd  MasVnrArea  BsmtFinSF1  BsmtFinSF2  ...  WoodDeckSF  \
0          2003       196.0         706           0  ...           0
1          1976         0.0         978           0  ...         298
2          2002       162.0         486           0  ...           0
3          1970         0.0         216           0  ...           0
4          2000       350.0         655           0  ...         192

   OpenPorchSF  EnclosedPorch  3SsnPorch  ScreenPorch  PoolArea  MiscVal  \
0           61              0          0            0         0        0
1            0              0          0            0         0        0
2           42              0          0            0         0        0
3           35            272          0            0         0        0
4           84              0          0            0         0        0

   MoSold  YrSold  SalePrice
0       2    2008     208500
1       5    2007     181500
2       9    2008     223500
3       2    2006     140000
4      12    2008     250000

[5 rows x 37 columns]
```

现在让我们绘制所有数值型特征的分布图。

```python
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);
```

（输出为各数值型特征的分布直方图，内容过大，此处省略。）

## 准备数据集

该数据集同时包含数值型特征、类别型（categorical）特征以及缺失值。TF-DF 原生支持所有这些特征类型，因此无需进行任何预处理。这是基于树的模型的一大优势，也使其成为入门 TensorFlow 和机器学习的绝佳起点。

现在，让我们把数据集划分为训练集和测试集：

```python
import numpy as np

def split_dataset(dataset, test_ratio=0.30):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))
```

```
1010 examples in training, 450 examples in testing.
```

在训练模型之前，还有一个步骤要做：我们需要把数据集从 Pandas 格式（`pd.DataFrame`）转换成 TensorFlow Datasets 格式（`tf.data.Dataset`）。

[TensorFlow Datasets](https://www.tensorflow.org/datasets/overview) 是一个高性能的数据加载库，在使用 GPU、TPU 等加速器训练神经网络时非常有用。

默认情况下，随机森林模型被配置为用于训练分类任务。由于这是一个回归问题，我们在这里将任务类型（`tfdf.keras.Task.REGRESSION`）作为参数指定。

```python
label = 'SalePrice'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)
```

## 选择模型

你可以从以下几种基于树的模型中进行选择：

* RandomForestModel
* GradientBoostedTreesModel
* CartModel
* DistributedGradientBoostedTreesModel

首先，我们将使用随机森林（Random Forest）。这是决策森林训练算法中最广为人知的一种。

随机森林是若干棵决策树的集合，每棵树都是在训练数据集的随机子集上（有放回抽样）独立训练得到的。该算法的独特之处在于它对过拟合具有很强的鲁棒性，而且易于使用。

我们可以使用以下代码列出 TensorFlow Decision Forests 中所有可用的模型：

```python
tfdf.keras.get_all_models()
```

```
[tensorflow_decision_forests.keras.RandomForestModel,
 tensorflow_decision_forests.keras.GradientBoostedTreesModel,
 tensorflow_decision_forests.keras.CartModel,
 tensorflow_decision_forests.keras.DistributedGradientBoostedTreesModel]
```

## 如何配置它们？

TensorFlow Decision Forests 为你提供了良好的默认配置（例如，我们在基准测试中排名靠前的超参数，并稍作调整以在合理的时间内运行）。如果你想配置学习算法，你会发现有许多可探索的选项，以便获得尽可能高的准确率。

你可以选择一个模板和/或如下设置参数：

```rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1", task=tfdf.keras.Task.REGRESSION)```

点击[这里](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel)了解更多。

## 创建随机森林模型

今天，我们将使用默认配置来创建随机森林模型，同时将任务类型指定为 `tfdf.keras.Task.REGRESSION`。

```python
rf = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)
rf.compile(metrics=["mse"]) # Optional, you can use this to include a list of eval metrics
```

```
Use /tmp/tmp_z_fwys1 as temporary training directory
```

## 训练模型

我们将用一行代码来训练模型。

注意：你可能会看到关于 Autograph 的警告。你可以安全地忽略它，这个问题将在下一个版本中修复。

```python
rf.fit(x=train_ds)
```

```
Reading training dataset...
Training dataset read in 0:00:07.344780. Found 1010 examples.
Training model...


[INFO 2023-04-03T10:17:06.624549063+00:00 kernel.cc:1214] Loading model from path /tmp/tmp_z_fwys1/model/ with prefix 82e90639875e465c


Model trained in 0:00:02.232848
Compiling model...


[INFO 2023-04-03T10:17:07.034502053+00:00 decision_forest.cc:661] Model loaded with 300 root(s), 96054 node(s), and 75 input feature(s).
[INFO 2023-04-03T10:17:07.034588334+00:00 abstract_model.cc:1311] Engine "RandomForestOptPred" built
[INFO 2023-04-03T10:17:07.034670992+00:00 kernel.cc:1046] Use fast generic engine


WARNING: AutoGraph could not transform <function simple_ml_inference_op_with_handle at 0x7fd14c8e70e0> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: could not get source code
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
Model compiled.


<keras.callbacks.History at 0x7fd13c11a690>
```

## 可视化模型

基于树的模型的一大好处是你可以轻松地将它们可视化。随机森林默认使用的树的数量是 300。我们可以在下面选择一棵树进行展示。

```python
tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)
```

```
<IPython.core.display.HTML object>
```

## 在袋外（OOB）数据和验证数据集上评估模型

在训练之前，我们已经手动将 20% 的数据集划分出来作为验证集，命名为 `valid_ds`。

我们也可以使用袋外（Out of bag，OOB）得分来验证我们的随机森林模型。
为了训练随机森林模型，算法会从训练集中选择一组随机样本，而其余样本则用于微调模型。未被选中的那部分数据就被称为袋外数据（OOB）。
OOB 得分是在 OOB 数据上计算得到的。

点击[这里](https://developers.google.com/machine-learning/decision-forests/out-of-bag)了解更多关于 OOB 数据的内容。

训练日志根据模型中树的数量，展示了在袋外数据集上评估得到的均方根误差（Root Mean Squared Error，RMSE）。让我们把它绘制出来。

注意：对于这个指标，数值越小越好。

```python
import matplotlib.pyplot as plt
logs = rf.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("RMSE (out-of-bag)")
plt.show()
```

（输出为 RMSE（袋外）随树数量变化的曲线图，内容过大，此处省略。）

我们还可以在 OOB 数据集上查看一些总体统计信息：

```python
inspector = rf.make_inspector()
inspector.evaluation()
```

```
Evaluation(num_examples=1010, accuracy=None, loss=None, rmse=29660.363022492173, ndcg=None, aucs=None, auuc=None, qini=None)
```

现在，让我们使用验证数据集进行一次评估。

```python
evaluation = rf.evaluate(x=valid_ds,return_dict=True)

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")
```

```
1/1 [==============================] - 1s 842ms/step - loss: 0.0000e+00 - mse: 712569024.0000
loss: 0.0000
mse: 712569024.0000
```

## 变量重要性

变量重要性通常表示某个特征对模型预测结果或模型质量的贡献程度。使用 TensorFlow Decision Forests，有多种方法可以识别重要特征。
让我们列出决策树可用的 `变量重要性（Variable Importances）` 指标：

```python
print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
  print("\t", importance)
```

```
Available variable importances:
	 INV_MEAN_MIN_DEPTH
	 NUM_AS_ROOT
	 NUM_NODES
	 SUM_SCORE
```

作为示例，让我们展示 `NUM_AS_ROOT` 这一变量重要性指标下的重要特征。

`NUM_AS_ROOT` 的重要性得分越大，它对模型结果的影响就越大。

默认情况下，列表按从最重要到最不重要的顺序排序。从输出中你可以推断出，位于列表顶部的特征，在随机森林的大多数树中被用作根节点的次数比其他任何特征都多。

```python
inspector.variable_importances()["NUM_AS_ROOT"]
```

```
[("OverallQual" (1; #62), 121.0),
 ("GarageCars" (1; #32), 49.0),
 ("ExterQual" (4; #22), 40.0),
 ("Neighborhood" (4; #59), 35.0),
 ("GrLivArea" (1; #38), 21.0),
 ("GarageArea" (1; #31), 15.0),
 ("BsmtQual" (4; #14), 7.0),
 ("YearBuilt" (1; #76), 5.0),
 ("KitchenQual" (4; #44), 4.0),
 ("TotalBsmtSF" (1; #73), 3.0)]
```

使用 Matplotlib 将检查器（inspector）中的变量重要性绘制出来

```python
plt.figure(figsize=(12, 4))

# Mean decrease in AUC of the class 1 vs the others.
variable_importance_metric = "NUM_AS_ROOT"
variable_importances = inspector.variable_importances()[variable_importance_metric]

# Extract the feature name and importance values.
#
# `variable_importances` is a list of <feature, importance> tuples.
feature_names = [vi[0].name for vi in variable_importances]
feature_importances = [vi[1] for vi in variable_importances]
# The feature are ordered in decreasing importance value.
feature_ranks = range(len(feature_names))

bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
plt.yticks(feature_ranks, feature_names)
plt.gca().invert_yaxis()

# TODO: Replace with "plt.bar_label()" when available.
# Label each bar with values
for importance, patch in zip(feature_importances, bar.patches):
  plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.4f}", va="top")

plt.xlabel(variable_importance_metric)
plt.title("NUM AS ROOT of the class 1 vs the others")
plt.tight_layout()
plt.show()
```

（输出为变量重要性（NUM_AS_ROOT）的水平条形图，内容过大，此处省略。）

# 提交结果

最后，使用模型对竞赛测试数据进行预测。

```python
test_file_path = "../input/house-prices-advanced-regression-techniques/test.csv"
test_data = pd.read_csv(test_file_path)
ids = test_data.pop('Id')

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    test_data,
    task = tfdf.keras.Task.REGRESSION)

preds = rf.predict(test_ds)
output = pd.DataFrame({'Id': ids,
                       'SalePrice': preds.squeeze()})

output.head()
```

```
2/2 [==============================] - 1s 24ms/step


     Id      SalePrice
0  1461  123554.718750
1  1462  153939.062500
2  1463  176793.765625
3  1464  183828.296875
4  1465  193644.484375
```

```python
sample_submission_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
sample_submission_df['SalePrice'] = rf.predict(test_ds)
sample_submission_df.to_csv('/kaggle/working/submission.csv', index=False)
sample_submission_df.head()
```

```
2/2 [==============================] - 0s 24ms/step


     Id      SalePrice
0  1461  123554.718750
1  1462  153939.062500
2  1463  176793.765625
3  1464  183828.296875
4  1465  193644.484375
```
