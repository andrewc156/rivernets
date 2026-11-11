"""R2 of every model under each outer split scheme (BENCHMARK_SPLIT runs),
next to the leave-one-station-out benchmark.
For the leave-one-subbasin-out split the R2 of every held-out subbasin is
also written (results/split_subbasin_folds.csv).

usage: python split_analysis.py

R2 uses equal station weights, as in the benchmark. For station_repeated the
pooled R2 is over all draws (a station tested in several draws counts once,
with all its test rows), and the spread across draws is given as the mean and
standard deviation of the per-draw R2 (each draw has only 3-5 test stations).

writes results/split_comparison.csv and results/split_repeated_draws.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

LOO = [  # leave-one-station-out runs of Table 14 and Section 12; the first run
    # holding a model wins (inner_fs also holds an older Graph-Laplacian)
    Path("results/benchmark_results_inner_fs_graph"),
    Path("results/idea_graph_gb"),
    Path("results/benchmark_results_inner_fs"),
]
SPLITS = ["station_repeated", "station_single", "time_within", "random_rows", "subbasin"]


def r2_equal(df):
    """Equal-station-weight R2 of y_pred against y_true."""
    s = df.assign(e=(df["y_true"] - df["y_pred"]) ** 2, y2=df["y_true"] ** 2).groupby("node").agg(
        my=("y_true", "mean"), my2=("y2", "mean"), e=("e", "mean")
    )
    mu = s["my"].mean()
    return 1 - s["e"].mean() / (s["my2"].mean() - mu ** 2)


def loo_predictions():
    parts = []
    for path in LOO:
        p = pd.read_pickle(path / "benchmark_predictions.pkl")
        if parts:
            p = p[~p["model"].isin(pd.concat(parts)["model"].unique())]
        parts.append(p)
    return pd.concat(parts)


def main():
    rows, draws, basins = [], [], []
    runs = {"loo": loo_predictions()}
    for split in SPLITS:
        path = Path(f"results/benchmark_results_split_{split}/benchmark_predictions.pkl")
        if path.exists():
            runs[split] = pd.read_pickle(path)
        else:
            print("missing:", path)

    cfg = Path("results/benchmark_results_split_subbasin/benchmark_config.json")
    subbasin_of = json.load(open(cfg))["graph_laplacian"]["subbasins"] if cfg.exists() else {}
    for split, p in runs.items():
        for (target, model), g in p.groupby(["target", "model"]):
            rows.append({"split": split, "target": target, "model": model,
                         "stations": g["node"].nunique(), "R2": r2_equal(g)})
            if split == "station_repeated":
                per = [r2_equal(d) for _, d in g.groupby("fold")]
                draws.append({"target": target, "model": model, "draws": len(per),
                              "R2_mean": np.mean(per), "R2_sd": np.std(per, ddof=1),
                              "R2_min": np.min(per), "R2_max": np.max(per)})
            if split == "subbasin":
                for fold, d in g.groupby("fold"):
                    basins.append({"target": target, "model": model,
                                   "fold": subbasin_of[str(int(d["node"].iloc[0]))],
                                   "stations": d["node"].nunique(), "R2": r2_equal(d)})

    table = pd.DataFrame(rows)
    wide = table.pivot_table(index=["target", "model"], columns="split", values="R2")
    wide = wide[[c for c in ["loo", *SPLITS] if c in wide.columns]]
    pd.set_option("display.width", 250)
    print(wide.round(4).to_string())
    table.to_csv("results/split_comparison.csv", index=False)
    if draws:
        d = pd.DataFrame(draws)
        print(d.round(4).to_string(index=False))
        d.to_csv("results/split_repeated_draws.csv", index=False)
    if basins:
        b = pd.DataFrame(basins)
        print(b.pivot_table(index=["target", "model"], columns="fold", values="R2").round(3).to_string())
        b.to_csv("results/split_subbasin_folds.csv", index=False)


if __name__ == "__main__":
    main()
