"""Runoff signal for turbidity from the station's own conductance sensor
(BENCHMARK_RUNOFF in water_quality_model_benchmark.py).

usage: python build_runoff_features.py        (a few seconds)

Storm runoff dilutes river water, so specific conductance (SC) drops while
sediment, and with it turbidity, rises. The features use only the station's
own SC record (a sensor the benchmark already uses as a predictor for other
targets), never turbidity values. Fixed in advance from a quick linear screen:
  sc_anom      log(SC) - log(trailing 7-day rolling median of SC); time-based
               window ending at the row's hour, at least 24 readings
  sc_d6        log SC(t) - log SC(t - 6 h), with the reading exactly 6 hours
               earlier (NaN if that hour has no reading; no shift by row count)
  sc_mean_log  the node's mean log SC over its own rows (a station level)
SC is clipped at >= 1 uS/cm before the logs. Rows without SC, or whose window
has fewer than 24 readings, or without a reading 6 hours earlier, get NaN; the
benchmark (BENCHMARK_RUNOFF=baseline or on) keeps only rows with all three.

writes data/benchmark_data/runoff_features.pkl  (node, time, sc_anom, sc_d6, sc_mean_log)
"""
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK = Path("data/benchmark_data/all_static_df.pkl")
OUT = Path("data/benchmark_data/runoff_features.pkl")
COLS = ["sc_anom", "sc_d6", "sc_mean_log"]


def node_features(d):
    s = np.log(d.set_index("time")["Specific conductance"].clip(lower=1)).sort_index()
    med = s.rolling("7D", min_periods=24).median()
    lag6 = s.reindex(s.index - pd.Timedelta(hours=6)).to_numpy()
    return pd.DataFrame({
        "time": s.index,
        "sc_anom": (s - med).to_numpy(),
        "sc_d6": s.to_numpy() - lag6,
        "sc_mean_log": s.mean(),
    })


def main():
    bench = pd.read_pickle(BENCHMARK)
    sc = bench.dropna(subset=["Specific conductance"])[["node", "time", "Specific conductance"]]
    if sc.duplicated(["node", "time"]).any():
        raise ValueError("duplicate node-hours in the benchmark data")
    feats = sc.groupby("node", group_keys=True).apply(node_features).reset_index(level=0)
    feats = feats.reset_index(drop=True)

    keys = bench[["node", "time"]].drop_duplicates()
    out = keys.merge(feats, on=["node", "time"], how="left", validate="one_to_one")
    assert len(out) == len(keys)
    out = out[["node", "time", *COLS]]
    out.to_pickle(OUT)

    ok = out[COLS].notna().all(axis=1)
    turb = bench.dropna(subset=["Turbidity"])[["node", "time"]].merge(out, on=["node", "time"], how="left")
    tok = turb[COLS].notna().all(axis=1)
    print(f"{len(out):,} node-hours, {ok.sum():,} ({ok.mean():.1%}) with all runoff features, "
          f"{out.loc[ok, 'node'].nunique()} nodes")
    print(f"turbidity rows: {len(turb):,} at {turb['node'].nunique()} stations -> with runoff features "
          f"{tok.sum():,} at {turb.loc[tok, 'node'].nunique()} stations")
    print(out.loc[ok, COLS].describe().round(3).to_string())


if __name__ == "__main__":
    main()
