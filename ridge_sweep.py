"""Ridge / ElasticNet 的正则强度扫描（5 折单种子）。"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_price_v2 import load_data, split_types, make_category_dtype, run_cv

VERSION = "exp"


def make_fitter(kind, alpha, l1_ratio=None):
    def _fit(X_tr, y_tr, X_va, y_va, X_te, seed, cat_cols, num_cols):
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=3), cat_cols),
        ])
        if kind == "ridge":
            model = Ridge(alpha=alpha)
        else:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000)
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_tr, y_tr)
        return pipe.predict(X_va), pipe.predict(X_te)
    return _fit


def main():
    X, y, X_test, _ = load_data(drop_outliers=True)
    y_log = np.log1p(y.values)
    num_cols, cat_cols = split_types(X)
    X, X_test = make_category_dtype(X, X_test, cat_cols)

    configs = {
        "ridge_1": ("ridge", 1.0),
        "ridge_3": ("ridge", 3.0),
        "ridge_30": ("ridge", 30.0),
        "ridge_100": ("ridge", 100.0),
        "ridge_300": ("ridge", 300.0),
        "enet_5e-4": ("enet", 5e-4),
        "enet_2e-3": ("enet", 2e-3),
    }

    results = {}
    for name, (kind, alpha) in configs.items():
        fitter = make_fitter(kind, alpha, 0.5 if kind == "enet" else None)
        _, _, r = run_cv(name, X, y_log, X_test, 5, (42,), fitter,
                         cat_cols=cat_cols, num_cols=num_cols)
        results[name] = r

    print("\n排序:")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {k:<12}{v:.6f}")


if __name__ == "__main__":
    main()
