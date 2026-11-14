"""Diagnostics of the water-year runs (BENCHMARK_YEAR_EFFECT=wy) next to the
leave-one-station-out benchmark and the longer-history run, on identical test rows.

usage: python year_effect_analysis.py        (seconds)

For every run (base, history, wy, history_wy), target and model:
  signed_bias  mean over stations of the station-mean error (y_pred - y_true),
               without 09429490 (above Imperial Dam, one outlying fold)
  bias2        mean over stations of the squared station-mean error
  within_var   mean over stations of the variance of the error about its mean
  R2           equal-station-weight R2, and R2_excl without 09429490
Also the days of data per station and water year in the history tables
(how unbalanced the panel is).

writes results/year_effect_diagnostics.csv and results/history_coverage_by_wy.csv
"""
import pandas as pd

OUTLIER = 203177183  # USGS-09429490
RUNS = {
    "base": ["benchmark_results_inner_fs_graph", "benchmark_results_inner_fs"],
    "history": ["benchmark_results_history"],
    "wy": ["benchmark_results_wy"],
    "history_wy": ["benchmark_results_history_wy"],
}


def load(dirs):
    parts = []
    for d in dirs:
        p = pd.read_pickle(f"results/{d}/benchmark_predictions.pkl")
        if parts:
            p = p[~p.model.isin(pd.concat(parts).model.unique())]
        parts.append(p)
    return pd.concat(parts)


def r2(df):
    s = df.assign(e=(df.y_true - df.y_pred) ** 2, y2=df.y_true ** 2).groupby("node").agg(
        my=("y_true", "mean"), my2=("y2", "mean"), e=("e", "mean"))
    return 1 - s.e.mean() / (s.my2.mean() - s.my.mean() ** 2)


def main():
    models = ["Graph-Laplacian", "OLS", "Ridge", "Elastic Net", "Spline GAM"]
    rows = []
    for run, dirs in RUNS.items():
        p = load(dirs)
        for (t, m), g in p[p.model.isin(models)].groupby(["target", "model"]):
            e = (g.y_pred - g.y_true).groupby(g.node)
            b = e.mean()
            rows.append({"run": run, "target": t, "model": m,
                         "signed_bias": b.drop(OUTLIER, errors="ignore").mean(),
                         "bias2": (b ** 2).mean(), "within_var": e.var(ddof=0).mean(),
                         "R2": r2(g), "R2_excl": r2(g[g.node != OUTLIER])})
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(d.round(4).to_string(index=False))
    d.to_csv("results/year_effect_diagnostics.csv", index=False)

    h = pd.read_pickle("data/benchmark_data_history/fourier_interaction_df.pkl")
    t = pd.to_datetime(h.time)
    h["wy"] = (t.dt.year + (t.dt.month >= 10)).clip(2021, 2026)
    cov = []
    for v in ["Water temperature", "DO concentration", "Specific conductance", "pH", "Turbidity"]:
        x = h.dropna(subset=[v])
        c = (x.groupby(["node", "wy"]).size() / 24).round().unstack(fill_value=0).astype(int)
        c.columns = [f"days_WY{c}" for c in c.columns]
        c["mean"] = x.groupby("node")[v].mean()
        cov.append(c.reset_index().assign(variable=v))
    cov = pd.concat(cov)
    cov.to_csv("results/history_coverage_by_wy.csv", index=False)
    print(cov[cov.variable == "Specific conductance"].round(0).to_string(index=False))


if __name__ == "__main__":
    main()
