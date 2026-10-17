# 把 Kaggle 数据放在这个文件夹

从竞赛页下载：<https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data>

点 **Download All**，解压 `house-prices-advanced-regression-techniques.zip`，把其中这三个文件复制到本目录：

- `train.csv`（1460 行 × 81 列，带 SalePrice）
- `test.csv`（1459 行，没有 SalePrice，用来预测）
- `sample_submission.csv`（提交格式样例）

```
大作业/
├─ data/
│  ├─ train.csv
│  ├─ test.csv
│  └─ sample_submission.csv
├─ house_price_sklearn.py
└─ ...
```

放好后运行 `check_env.py`，三项都显示 `[有]` 就说明数据到位了。
