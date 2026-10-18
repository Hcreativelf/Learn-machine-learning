"""v1 基线的对数空间成绩：随机森林 300 棵 + 中位数/众数填补 + 序数编码。

用途：和 v2 的梯度提升融合做同口径对比（都用 10 折 CV、log1p 目标）。
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from house_price_v2 import load_data, split_types, make_category_dtype, run_cv


def fit_rf(X_tr, y_tr, X_va, y_va, X_te, seed, cat_cols, num_cols):
    def prep(df):
        df = df.copy()
        for c in cat_cols:
            df[c] = df[c].astype(str)
        return df

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("model", RandomForestRegressor(n_estimators=300, n_jobs=1, random_state=seed)),
    ])
    pipe.fit(prep(X_tr), y_tr)
    return pipe.predict(prep(X_va)), pipe.predict(prep(X_te))


def main():
    X, y, X_test, _ = load_data(drop_outliers=True)
    y_log = np.log1p(y.values)
    num_cols, cat_cols = split_types(X)
    X, X_test = make_category_dtype(X, X_test, cat_cols)

    _, _, rmse = run_cv("rf_base", X, y_log, X_test, 10, (42,), fit_rf,
                        cat_cols=cat_cols, num_cols=num_cols)
    print(f"\nv1 等价基线（随机森林 + Log 目标）: RMSE(log) = {rmse:.6f}")


if __name__ == "__main__":
    main()
