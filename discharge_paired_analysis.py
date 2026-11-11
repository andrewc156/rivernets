"""Paired comparison of the discharge benchmark runs.

Compares results/benchmark_results_discharge_baseline (discharge rows, no
discharge feature) with results/benchmark_results_discharge_on (same rows,
"+ Q" feature sets offered in the inner loop). Both runs use the same rows and
outer folds, so the difference is the effect of offering the discharge anomaly.

R2 uses equal station weights, as in water_quality_model_benchmark.py.
The 95% interval resamples stations with the predictions held fixed, so it does
not include the variability of the inner-loop choices.

Also computes a subset diagnostic: R2 of the main inner-FS runs restricted to
the stations that have discharge (indicative only: those models were trained
on all stations).

usage: python discharge_paired_analysis.py
writes results/discharge_paired_comparison.csv and
       results/discharge_subset_diagnostic.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("results/benchmark_results_discharge_baseline")
ON = Path("results/benchmark_results_discharge_on")
MAIN = {  # main inner-FS runs (Table "final" in the report)
    "Graph-Laplacian": Path("results/benchmark_results_inner_fs_graph"),
    "other": Path("results/benchmark_results_inner_fs"),
}
N_BOOT = 2000


def station_means(df, pred_cols):
    df = df.assign(y2=df["y_true"] ** 2, **{
        f"e_{c}": (df["y_true"] - df[c]) ** 2 for c in pred_cols
    })
    return df.groupby("node").agg(
        my=("y_true", "mean"), my2=("y2", "mean"),
        **{f"e_{c}": (f"e_{c}", "mean") for c in pred_cols},
    )


def r2_equal(s, col):
    mu = s["my"].mean()
    sst = s["my2"].mean() - mu ** 2
    return 1 - s[col].mean() / sst


def paired():
    b = pd.read_pickle(BASE / "benchmark_predictions.pkl")
    o = pd.read_pickle(ON / "benchmark_predictions.pkl")
    key = ["target", "model", "node", "row_index"]
    m = b.merge(o, on=key, suffixes=("", "_on"))
    assert len(m) == len(b) == len(o)
    assert np.allclose(m["y_true"], m["y_true_on"])
    m = m.rename(columns={"y_pred": "base", "y_pred_on": "q"})

    sel = pd.read_csv(ON / "feature_selection_counts.csv")
    sel["q"] = sel["features"].str.endswith("+ Q")
    q_folds = sel[sel["q"]].groupby(["target", "model"])["outer_folds"].sum()
    all_folds = sel.groupby(["target", "model"])["outer_folds"].sum()

    rng = np.random.default_rng(0)
    rows = []
    for (target, model), g in m.groupby(["target", "model"]):
        s = station_means(g, ["base", "q"])
        r_b, r_q = r2_equal(s, "e_base"), r2_equal(s, "e_q")
        boot = []
        for _ in range(N_BOOT):
            r = s.iloc[rng.integers(0, len(s), len(s))]
            boot.append(r2_equal(r, "e_q") - r2_equal(r, "e_base"))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        d = s["e_q"] - s["e_base"]
        tol = 1e-12 * s["e_base"]
        gain = -d.sum()
        rows.append({
            "target": target, "model": model, "stations": len(s),
            "R2_baseline": r_b, "R2_with_Q": r_q, "dR2": r_q - r_b,
            "ci_lo": lo, "ci_hi": hi,
            "stations_better": int((d < -tol).sum()),
            "stations_worse": int((d > tol).sum()),
            "top_station_share_of_gain": (-d.min() / gain) if gain > 0 else np.nan,
            "Q_chosen_folds": int(q_folds.get((target, model), 0)),
            "folds": int(all_folds[(target, model)]),
        })
    return pd.DataFrame(rows)


def subset_diagnostic():
    """Main-run R2 on all stations vs on the discharge stations only."""
    nodes = pd.read_pickle(BASE / "benchmark_predictions.pkl").groupby("target")["node"].unique()
    rows = []
    for which, path in MAIN.items():
        p = pd.read_pickle(path / "benchmark_predictions.pkl")
        if which == "other":
            p = p[p["model"] != "Graph-Laplacian"]
        for (target, model), g in p.groupby(["target", "model"]):
            if target not in nodes:
                continue
            sub = g[g["node"].isin(nodes[target])]
            rows.append({
                "target": target, "model": model,
                "R2_all_stations": r2_equal(station_means(g, ["y_pred"]), "e_y_pred"),
                "stations_all": g["node"].nunique(),
                "R2_discharge_stations": r2_equal(station_means(sub, ["y_pred"]), "e_y_pred"),
                "stations_discharge": sub["node"].nunique(),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    r = paired()
    r.to_csv("results/discharge_paired_comparison.csv", index=False)
    print(r.round(4).to_string(index=False))
    d = subset_diagnostic()
    d.to_csv("results/discharge_subset_diagnostic.csv", index=False)
    print(d.round(4).to_string(index=False))
