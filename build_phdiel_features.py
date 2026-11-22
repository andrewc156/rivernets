"""Diel part of water temperature at each station, a predictor for pH
(BENCHMARK_PHDIEL in water_quality_model_benchmark.py).

usage: python build_phdiel_features.py        (a few seconds)

Feature (fixed in advance from a quick linear screen, where it added +0.023 to
pH while raw water temperature lowered it):
  wt_diel   the node's hourly water temperature minus its centred 24-hour rolling
            mean (time-based window of 24 hours centred on the row, at least 12
            hourly observations, otherwise NaN)
The day's heating and cooling at the station tracks sunlight, which drives
photosynthesis and so the daily pH cycle. Water temperature together with
wt_diel determines the 24-hour mean, so that mean is not a separate column.

wt_diel uses only the station's own water-temperature sensor, never pH or any
other target value. The centred window uses up to 12 hours AFTER the row: that
is fine for filling in a station's record after the fact, but not for real-time
prediction. Rows without water temperature, or with fewer than 12 hourly values
in the window, get NaN; the benchmark (BENCHMARK_PHDIEL=baseline|on) keeps only
pH rows with wt_diel in both arms, so the two are paired.

The definition matches the DO saturation option's wt_diel (agent 1) so the two
can be unified.

writes data/benchmark_data/phdiel_features.pkl  (node, time, wt_diel)
"""
from pathlib import Path

import pandas as pd

BENCHMARK = Path("data/benchmark_data/all_static_df.pkl")
OUT = Path("data/benchmark_data/phdiel_features.pkl")
WINDOW, MIN_OBS = "24h", 12


def diel(d):
    s = d.set_index("time")["Water temperature"]
    m = s.rolling(WINDOW, center=True, min_periods=MIN_OBS).mean()
    out = d[["node", "time"]].copy()
    out["wt_diel"] = (s - m).to_numpy()
    return out


def main():
    df = pd.read_pickle(BENCHMARK)[["node", "time", "Water temperature", "pH", "Specific conductance"]]
    if df.duplicated(["node", "time"]).any():
        raise ValueError("duplicate node-hours in all_static_df")
    df = df.sort_values(["node", "time"]).reset_index(drop=True)
    out = pd.concat([diel(d) for _, d in df.groupby("node", sort=False)], ignore_index=True)
    assert len(out) == len(df) and out["time"].dtype == df["time"].dtype
    out.to_pickle(OUT)

    ok = out["wt_diel"].notna()
    print(f"{len(out):,} node-hours, {ok.sum():,} ({ok.mean():.1%}) with wt_diel, "
          f"{out.loc[ok, 'node'].nunique()} nodes; wt_diel sd {out.loc[ok, 'wt_diel'].std():.2f} C")
    ph = df.dropna(subset=["pH", "Specific conductance"])
    ph_ok = ph.merge(out, on=["node", "time"], how="left")["wt_diel"].notna().to_numpy()
    print(f"pH rows with conductance: {len(ph):,} at {ph['node'].nunique()} stations; "
          f"with wt_diel: {ph_ok.sum():,} at {ph.loc[ph_ok, 'node'].nunique()} stations")


if __name__ == "__main__":
    main()
