"""Air-temperature features at every benchmark node-hour (BENCHMARK_AIRTEMP).

usage: python build_airtemp_features.py      (a few seconds; run download_airtemp.py first)

Input: data/airtemp/node_<node>.csv from download_airtemp.py, hourly 2 m air
temperature from ERA5-Land (Munoz-Sabater et al. 2021, Earth System Science
Data 13, 4349-4383, doi:10.5194/essd-13-4349-2021) served by the Open-Meteo
historical API (Zippenfenig 2023, doi:10.5281/zenodo.7970649), in UTC, at each
water-quality node's coordinates.

Features (degrees C, fixed in advance):
  at_1h     air temperature at the row's UTC hour
  at_24h    mean over the 24 hours ending at the row's hour (that hour included)
  at_72h    mean over the 72 hours ending at the row's hour
  at_168h   mean over the 168 hours (7 days) ending at the row's hour
Water lags the air, so the trailing means let the model weigh recent and older
warming. The windows need every hour present (min_periods = window length);
the download starts 8 days before each node's first row, so they are complete.

Air temperature exists at any point of the basin, monitored or not, so it fits
the task of predicting at an unmonitored station: no water-quality value of the
held-out station enters these features.

writes data/benchmark_data/airtemp_features.pkl  (node, time, at_1h ... at_168h)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/airtemp")
BENCHMARK = Path("data/benchmark_data/all_static_df.pkl")
OUT = Path("data/benchmark_data/airtemp_features.pkl")
WINDOWS = {"at_24h": 24, "at_72h": 72, "at_168h": 168}
COLS = ["at_1h", *WINDOWS]


def node_features(node):
    a = pd.read_csv(SRC / f"node_{node}.csv")
    s = pd.Series(a["temperature_2m"].to_numpy(float),
                  index=pd.to_datetime(a["time"]).dt.tz_localize("UTC")).sort_index()
    if s.index.duplicated().any():
        raise ValueError(f"node {node}: duplicate hours")
    s = s.asfreq("h")  # makes any gap explicit (NaN), so a window never spans it silently
    f = pd.DataFrame({"at_1h": s})
    for col, hours in WINDOWS.items():
        f[col] = s.rolling(hours, min_periods=hours).mean()
    return f.rename_axis("time").reset_index().assign(node=node)


def main():
    bench = pd.read_pickle(BENCHMARK)[["node", "time"]].drop_duplicates()
    at = pd.concat([node_features(n) for n in sorted(bench["node"].unique())], ignore_index=True)
    out = bench.merge(at, on=["node", "time"], how="left", validate="many_to_one")
    assert len(out) == len(bench)
    out[COLS] = out[COLS].astype(np.float32)
    out = out[["node", "time", *COLS]].reset_index(drop=True)
    out.to_pickle(OUT)
    ok = out[COLS].notna().all(axis=1)
    print(f"{len(out):,} node-hours, {ok.sum():,} ({ok.mean():.2%}) with all air-temperature "
          f"features, {out.loc[ok, 'node'].nunique()} of {out['node'].nunique()} nodes; "
          f"at_1h range {out['at_1h'].min():.1f} .. {out['at_1h'].max():.1f} C")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
