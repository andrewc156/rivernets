"""At-station daily precipitation features from Daymet (a second precipitation
source next to the MRMS radar of build_rain_features.py).

usage: python build_daymet_features.py        (a few seconds)

Input: data/rain/daymet_features_2025.csv, the user's cleaned Daymet file: one row
per water-quality station and day of calendar 2025, Daymet precipitation at the
station's own coordinates (not over the river upstream, unlike MRMS). Its window
columns hold only days BEFORE the row's date:
  rain_kd_mm               mm summed over the k days before the date
  antecedent_precip_index  API(d) = 0.8 API(d-1) + 0.2 P(d-1)

Features (fixed in advance, five like the MRMS set):
  dm_1d, dm_3d, dm_7d, dm_30d   log(1 + rain_kd_mm), k = 1, 3, 7, 30
  dm_api                        log(1 + antecedent_precip_index)
Each benchmark row (hour t in UTC) takes the features of its local date, t - 7 h
(Mountain Standard Time, fixed offset), so every window ends at local midnight,
before the measurement hour. Two nodes hold two stations each (a cluster of
nearby stations); they take the mean of the two.

Rows outside 2025-01-31 .. 2025-12-31 (local) get NaN: the file starts on
1 January, so the 30-day window is complete only from 31 January. The benchmark
(BENCHMARK_DAYMET) keeps only rows with all five features, for every arm.

writes data/benchmark_data/daymet_features.pkl  (node, time, dm_1d ... dm_api)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/rain/daymet_features_2025.csv")
COMPRESSED = Path("graphs/colorado/master_compressed_river_graph.pkl")
BENCHMARK = Path("data/benchmark_data/all_static_df.pkl")
OUT = Path("data/benchmark_data/daymet_features.pkl")
SRC_COLS = {"dm_1d": "rain_1d_mm", "dm_3d": "rain_3d_mm", "dm_7d": "rain_7d_mm",
            "dm_30d": "rain_30d_mm", "dm_api": "antecedent_precip_index"}
COLS = list(SRC_COLS)
FIRST, LAST = pd.Timestamp("2025-01-31"), pd.Timestamp("2025-12-31")


def main():
    Gc = pickle.load(open(COMPRESSED, "rb"))
    st2node = {s: n for n, d in Gc.nodes(data=True)
               for c in d.get("usgs_clusters") or [] for s in c["member_stations"]}
    f = pd.read_csv(SRC, parse_dates=["date"])
    f["node"] = f["wq_station"].map(st2node)
    if f["node"].isna().any():
        raise ValueError(f"stations not on the graph: {sorted(f.loc[f['node'].isna(), 'wq_station'].unique())}")
    f = f[(f["date"] >= FIRST) & (f["date"] <= LAST)]
    if f[list(SRC_COLS.values())].isna().any().any():
        raise ValueError("missing Daymet values inside the window")
    for new, old in SRC_COLS.items():
        f[new] = np.log1p(f[old])
    daily = f.groupby(["node", "date"], as_index=False)[COLS].mean()
    daily["node"] = daily["node"].astype(int)

    bench = pd.read_pickle(BENCHMARK)[["node", "time"]].drop_duplicates()
    bench["date"] = (bench["time"] - pd.Timedelta(hours=7)).dt.tz_localize(None).dt.normalize()
    out = bench.merge(daily, on=["node", "date"], how="left", validate="many_to_one")
    assert len(out) == len(bench)
    out = out.drop(columns="date")
    out.to_pickle(OUT)
    ok = out[COLS].notna().all(axis=1)
    print(f"{len(out):,} node-hours, {ok.sum():,} ({ok.mean():.1%}) with Daymet features, "
          f"{out.loc[ok, 'node'].nunique()} nodes; share with dm_1d > 0: {(out.loc[ok, 'dm_1d'] > 0).mean():.3f}")


if __name__ == "__main__":
    main()
