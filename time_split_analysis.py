"""Analysis of the held-out time period run (BENCHMARK_TIME_SPLIT).

usage: python time_split_analysis.py [TIME_SPLIT_DIR]   (default results/benchmark_results_time_split)

1. Paired comparisons inside the time-split run, on identical test rows:
   Graph-Laplacian + GB vs Graph-Laplacian, vs Gradient Boosting, and
   Graph-Laplacian vs Gradient Boosting (equal station weights, station
   bootstrap with predictions held fixed).
2. Reference: the all-time runs (main graph, main gradient boosting, the
   idea_graph_gb hybrid) restricted to the same test stations and the rows on
   or after the cutoff. Those models were trained on all periods, so the
   difference shows the cost of not seeing the future; R2 levels are only
   comparable within a column, not with Table 14 (different season and rows).

writes <dir>/time_split_paired.csv and <dir>/time_split_reference.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_results_time_split")
REFERENCE = {
    "Graph-Laplacian": Path("results/benchmark_results_inner_fs_graph"),
    "Gradient Boosting": Path("results/benchmark_results_inner_fs"),
    "Graph-Laplacian + GB": Path("results/idea_graph_gb"),
}
PAIRS = [
    ("Graph-Laplacian", "Graph-Laplacian + GB"),
    ("Gradient Boosting", "Graph-Laplacian + GB"),
    ("Gradient Boosting", "Graph-Laplacian"),
]
N_BOOT = 2000


def station_stats(df, cols):
    df = df.assign(y2=df["y_true"] ** 2, **{f"e_{c}": (df["y_true"] - df[c]) ** 2 for c in cols})
    return df.groupby("node").agg(
        my=("y_true", "mean"), my2=("y2", "mean"), **{f"e_{c}": (f"e_{c}", "mean") for c in cols}
    )


def r2(s, col):
    mu = s["my"].mean()
    return 1 - s[col].mean() / (s["my2"].mean() - mu ** 2)


def main():
    pred = pd.read_pickle(RUN / "benchmark_predictions.pkl")
    wide = pred.pivot_table(index=["target", "node", "row_index", "y_true"],
                            columns="model", values="y_pred").reset_index()
    rng = np.random.default_rng(0)

    rows = []
    for target, g in wide.groupby("target"):
        models = [m for m in pred["model"].unique() if m in g and g[m].notna().all()]
        s = station_stats(g, models)
        levels = {m: r2(s, f"e_{m}") for m in models}
        for base, new in PAIRS:
            if base not in models or new not in models:
                continue
            boot = []
            for _ in range(N_BOOT):
                r = s.iloc[rng.integers(0, len(s), len(s))]
                boot.append(r2(r, f"e_{new}") - r2(r, f"e_{base}"))
            d = s[f"e_{new}"] - s[f"e_{base}"]
            rows.append({
                "target": target, "stations": len(s), "base": base, "new": new,
                "R2_base": levels[base], "R2_new": levels[new], "dR2": levels[new] - levels[base],
                "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
                "stations_better": int((d < 0).sum()), "stations_worse": int((d > 0).sum()),
            })
        rows.append({"target": target, "stations": len(s), "base": "levels",
                     **{f"R2[{m}]": v for m, v in levels.items()}})
    paired = pd.DataFrame(rows)

    cutoff = pd.Timestamp(json.load(open(RUN / "benchmark_config.json"))
                          ["graph_laplacian"]["time_split"], tz="UTC")
    times = pd.to_datetime(pd.read_pickle("data/benchmark_data/all_static_df.pkl")["time"], utc=True)
    test_nodes = pred.groupby("target")["node"].unique()
    ref_rows = []
    for model, path in REFERENCE.items():
        p = pd.read_pickle(path / "benchmark_predictions.pkl")
        p = p[p["model"] == model]
        p = p[times.loc[p["row_index"]].to_numpy() >= cutoff]
        for target, g in p.groupby("target"):
            if target not in test_nodes:
                continue
            g = g[g["node"].isin(test_nodes[target])].rename(columns={"y_pred": "m"})
            here = pred[(pred["target"] == target) & (pred["model"] == model)]
            ref_rows.append({
                "target": target, "model": model, "stations": g["node"].nunique(),
                "R2_all_time_training": r2(station_stats(g, ["m"]), "e_m"),
                "R2_time_split": (r2(station_stats(here.rename(columns={"y_pred": "m"}), ["m"]), "e_m")
                                  if len(here) else np.nan),
                "rows_all_time": len(g), "rows_time_split": len(here),
            })
    reference = pd.DataFrame(ref_rows)

    pd.set_option("display.width", 250)
    print(paired.round(4).to_string(index=False))
    print(reference.round(4).to_string(index=False))
    paired.to_csv(RUN / "time_split_paired.csv", index=False)
    reference.to_csv(RUN / "time_split_reference.csv", index=False)


if __name__ == "__main__":
    main()
