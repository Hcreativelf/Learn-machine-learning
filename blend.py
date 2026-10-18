"""融合实验：读取 artifacts 里各模型的 OOF / 测试集预测，比较几种加权方式。

运行：python blend.py
输出：submission_v2.csv（用 OOF 上最优的融合方式）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parent
ART_DIR = PROJECT_DIR / "artifacts"
VERSION = "v2.2"


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def hill_climb(A, y, n_iter=300, step=0.02):
    m = A.shape[1]
    scores = [rmse(y, A[:, j]) for j in range(m)]
    w = np.zeros(m)
    w[int(np.argmin(scores))] = 1.0
    best = min(scores)
    for _ in range(n_iter):
        improved = False
        for j in range(m):
            for s in (step, -step):
                w2 = w.copy()
                w2[j] = max(0.0, w2[j] + s)
                if w2.sum() == 0:
                    continue
                w2 /= w2.sum()
                r = rmse(y, A @ w2)
                if r < best - 1e-9:
                    best, w, improved = r, w2, True
        if not improved:
            break
    return w, best


def main():
    meta = {}
    oofs, tests = [], []
    for path in sorted(ART_DIR.glob(f"{VERSION}_*.npz")):
        name = path.stem.split("_", 1)[1]
        data = np.load(path)
        oofs.append(data["oof"])
        tests.append(data["test"])
        m = json.loads((ART_DIR / f"{VERSION}_{name}.json").read_text(encoding="utf-8"))
        meta[name] = m["rmse"]

    names = list(meta)
    A = np.column_stack(oofs)
    B = np.column_stack(tests)
    y = np.load(ART_DIR / "y_log.npy") if (ART_DIR / "y_log.npy").exists() else None
    if y is None:
        raise SystemExit("缺少 artifacts/y_log.npy")

    print("单模型:")
    for n in sorted(names, key=lambda k: meta[k]):
        print(f"  {n:<8} {meta[n]:.6f}")

    print("\nOOF 误差相关性:")
    err = A - y[:, None]
    corr = np.corrcoef(err.T)
    print(pd.DataFrame(corr, index=names, columns=names).round(3).to_string())

    results = {}
    results["最优单模型"] = (A[:, int(np.argmin([meta[n] for n in names]))].copy(),
                        B[:, int(np.argmin([meta[n] for n in names]))].copy(),
                        min(meta.values()))
    results["简单平均"] = (A.mean(1), B.mean(1), rmse(y, A.mean(1)))

    w_nnls, _ = nnls(A, y)
    results["NNLS"] = (A @ w_nnls, B @ w_nnls, rmse(y, A @ w_nnls))

    w_hc, hc_score = hill_climb(A, y)
    results["爬山法"] = (A @ w_hc, B @ w_hc, hc_score)

    lin = LinearRegression(positive=True).fit(A, y)
    results["正系数+截距"] = (lin.predict(A), lin.predict(B), rmse(y, lin.predict(A)))

    print("\n融合方式:")
    for k, (_, _, s) in sorted(results.items(), key=lambda kv: kv[1][2]):
        print(f"  {k:<10} {s:.6f}")

    print(f"\nNNLS 权重: {dict(zip(names, np.round(w_nnls, 3)))}")
    print(f"爬山权重: {dict(zip(names, np.round(w_hc, 3)))}")

    best_name = min(results, key=lambda k: results[k][2])
    print(f"\n采用: {best_name}")

    log_pred = results[best_name][1]
    ids = pd.read_csv(PROJECT_DIR / "data" / "test.csv")["Id"]
    pred = np.clip(np.expm1(log_pred), 1.0, None)
    sub = pd.DataFrame({"Id": ids, "SalePrice": pred})
    sub.to_csv(PROJECT_DIR / "submission_v2.csv", index=False)
    print(f"已写出 submission_v2.csv：{len(sub)} 行，"
          f"价格 {pred.min():,.0f} ~ {pred.max():,.0f}")
    print(f"OOF RMSE(log) = {results[best_name][2]:.6f}")


if __name__ == "__main__":
    main()

