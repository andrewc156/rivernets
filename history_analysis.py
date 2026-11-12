"""Diagnostics of the longer-history run (BENCHMARK_HISTORY) against the
leave-one-station-out benchmark, on identical test rows.

usage: python history_analysis.py        (seconds)

For each model in both runs:
  * bias: the mean over stations of the signed station-mean error (y_pred - y_true),
    and the mean squared station bias and within-station error variance;
  * water temperature R2 after subtracting one common offset (the mean signed
    station bias over all stations but 09429490). This is a descriptive check of
    how much of the loss is a shift common to all stations, not a fair model;
  * R2 on region subsets (Upper Basin, without San Juan, headwaters only). These
    are the same LOO predictions scored on fewer stations (trained on all of them).
Also the water temperature anomaly per water year (each hour against the mean of
its station and day of year over the history tables).

writes results/history_diagnostics.csv, results/history_vs_base_by_region.csv
and results/history_wt_anomaly_by_wy.csv
"""
import json

import pandas as pd

BASE = ["results/benchmark_results_inner_fs_graph", "results/benchmark_results_inner_fs"]
HIST = "results/benchmark_results_history"
OUTLIER = 203177183  # USGS-09429490, above Imperial Dam
TARGETS = ["Water temperature", "DO concentration", "Specific conductance", "pH", "Turbidity"]


def r2(df):
    s = df.assign(e=(df.y_true - df.y_pred) ** 2, y2=df.y_true ** 2).groupby("node").agg(
        my=("y_true", "mean"), my2=("y2", "mean"), e=("e", "mean"))
    return 1 - s.e.mean() / (s.my2.mean() - s.my.mean() ** 2)


def main():
    g = pd.read_pickle(f"{BASE[0]}/benchmark_predictions.pkl")
    o = pd.read_pickle(f"{BASE[1]}/benchmark_predictions.pkl")
    base = pd.concat([g, o[~o.model.isin(g.model.unique())]])
    hist = pd.read_pickle(f"{HIST}/benchmark_predictions.pkl")
    sub = {int(k): v for k, v in
           json.load(open(f"{HIST}/benchmark_config.json"))["graph_laplacian"]["subbasins"].items()}
    lower = [n for n, v in sub.items() if v.startswith("Lower")]
    sj = [n for n, v in sub.items() if v.startswith("San Juan")]
    hw = [n for n, v in sub.items() if v.startswith("Colorado head")]
    subsets = {"Upper Basin (excl. Lower)": lambda n: ~n.isin(lower),
               "Excl. San Juan + Lower": lambda n: ~n.isin(lower + sj),
               "Headwaters only": lambda n: n.isin(hw)}
    models = [m for m in hist.model.unique() if m != "Dummy"]

    diag, region = [], []
    for t in TARGETS:
        for m in models:
            x = base[(base.target == t) & (base.model == m)]
            y = hist[(hist.target == t) & (hist.model == m)]
            row = {"target": t, "model": m, "R2_base": r2(x), "R2_history": r2(y)}
            for name, d in [("base", x), ("history", y)]:
                e = (d.y_pred - d.y_true).groupby(d.node)
                b = e.mean().drop(OUTLIER, errors="ignore")
                row[f"signed_bias_{name}"] = b.mean()
                row[f"bias2_{name}"] = (e.mean() ** 2).mean()
                row[f"within_var_{name}"] = e.var(ddof=0).mean()
            if t == "Water temperature":
                off = row["signed_bias_history"]
                row["R2_history_minus_offset"] = r2(y.assign(y_pred=y.y_pred - off))
            diag.append(row)
            for s, f in subsets.items():
                xs, ys = x[f(x.node)], y[f(y.node)]
                region.append({"subset": s, "target": t, "model": m, "stations": xs.node.nunique(),
                               "R2_base": r2(xs), "R2_history": r2(ys)})

    pd.set_option("display.width", 250)
    d = pd.DataFrame(diag)
    print(d.round(3).to_string(index=False))
    d.to_csv("results/history_diagnostics.csv", index=False)
    r = pd.DataFrame(region)
    print(r.round(3).to_string(index=False))
    r.to_csv("results/history_vs_base_by_region.csv", index=False)

    h = pd.read_pickle("data/benchmark_data_history/fourier_interaction_df.pkl")
    h = h[["node", "time", "Water temperature"]].dropna()
    t = pd.to_datetime(h.time)
    h = h.assign(wy=t.dt.year + (t.dt.month >= 10), doy=t.dt.dayofyear)
    h["anom"] = h["Water temperature"] - h.groupby(["node", "doy"])["Water temperature"].transform("mean")
    a = h.groupby("wy").agg(hours=("anom", "size"), anomaly=("anom", "mean"))
    print(a.round(2).to_string())
    a.to_csv("results/history_wt_anomaly_by_wy.csv")


if __name__ == "__main__":
    main()
