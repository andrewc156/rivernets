"""Paired comparison of two benchmark runs on the same rows and outer folds.

usage:
  python compare_runs.py BASE_DIR NEW_DIR [--base-model M] [--new-model M] [--out CSV]

Compares model --base-model in BASE_DIR (default Graph-Laplacian) with
--new-model in NEW_DIR (default: the same name) for every target in both.
R2 uses equal station weights, as in water_quality_model_benchmark.py. The 95%
interval resamples stations with the predictions held fixed, so it leaves out
the variability of the inner-loop choices. Also reports how often the inner
loop changed its choice across outer folds (distinct feature sets and distinct
(weighting, lambda) pairs), as a measure of stability.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 2000


def station_means(df):
    df = df.assign(
        y2=df["y_true"] ** 2,
        e_base=(df["y_true"] - df["base"]) ** 2,
        e_new=(df["y_true"] - df["new"]) ** 2,
    )
    return df.groupby("node").agg(
        my=("y_true", "mean"), my2=("y2", "mean"),
        e_base=("e_base", "mean"), e_new=("e_new", "mean"),
    )


def r2_equal(s, col):
    mu = s["my"].mean()
    return 1 - s[col].mean() / (s["my2"].mean() - mu ** 2)


def choices(run_dir, model):
    t = pd.read_pickle(Path(run_dir) / "benchmark_tuning.pkl")
    t = t[t["model"] == model]
    out = {}
    for target, g in t.groupby("target"):
        pairs = g["best_params"].apply(
            lambda p: (p.get("scheme"), p.get("lambda")) if isinstance(p, dict) else None
        )
        out[target] = (g["features"].nunique(), pairs.nunique(), len(g))
    return out


def compare(base_dir, new_dir, base_model, new_model):
    key = ["target", "node", "row_index"]
    b = pd.read_pickle(Path(base_dir) / "benchmark_predictions.pkl")
    o = pd.read_pickle(Path(new_dir) / "benchmark_predictions.pkl")
    b = b[b["model"] == base_model].rename(columns={"y_pred": "base"})
    o = o[o["model"] == new_model].rename(columns={"y_pred": "new", "y_true": "y_true_new"})
    m = b.merge(o[[*key, "new", "y_true_new"]], on=key)
    for target in m["target"].unique():
        nb, no, nm = (len(d[d["target"] == target]) for d in (b, o, m))
        if not nb == no == nm:
            raise ValueError(f"{target}: rows differ ({nb}, {no}, matched {nm})")
    if not np.allclose(m["y_true"], m["y_true_new"]):
        raise ValueError("y_true differs between the runs")

    cb, cn = choices(base_dir, base_model), choices(new_dir, new_model)
    rng = np.random.default_rng(0)
    rows = []
    for target, g in m.groupby("target"):
        s = station_means(g)
        boot = []
        for _ in range(N_BOOT):
            r = s.iloc[rng.integers(0, len(s), len(s))]
            boot.append(r2_equal(r, "e_new") - r2_equal(r, "e_base"))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        d = s["e_new"] - s["e_base"]
        tol = 1e-12 * s["e_base"]
        gain = -d.sum()
        rows.append({
            "target": target, "stations": len(s),
            "R2_base": r2_equal(s, "e_base"), "R2_new": r2_equal(s, "e_new"),
            "dR2": r2_equal(s, "e_new") - r2_equal(s, "e_base"), "ci_lo": lo, "ci_hi": hi,
            "stations_better": int((d < -tol).sum()), "stations_worse": int((d > tol).sum()),
            "top_station_share_of_gain": (-d.min() / gain) if gain > 0 else np.nan,
            "feature_sets_base": cb.get(target, (np.nan,) * 3)[0],
            "feature_sets_new": cn.get(target, (np.nan,) * 3)[0],
            "weighting_lambda_base": cb.get(target, (np.nan,) * 3)[1],
            "weighting_lambda_new": cn.get(target, (np.nan,) * 3)[1],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir")
    ap.add_argument("new_dir")
    ap.add_argument("--base-model", default="Graph-Laplacian")
    ap.add_argument("--new-model", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = compare(a.base_dir, a.new_dir, a.base_model, a.new_model or a.base_model)
    pd.set_option("display.width", 250)
    print(r.round(4).to_string(index=False))
    if a.out:
        r.to_csv(a.out, index=False)
