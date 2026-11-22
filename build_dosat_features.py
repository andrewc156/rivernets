"""DO-saturation features for the DO target (BENCHMARK_DOSAT).

usage: python build_dosat_features.py        (a few seconds)

Two columns per node-hour, from the station's own water-temperature sensor and
its static elevation only (never DO or any other target value):

  do_sat   dissolved-oxygen saturation (mg/L) of fresh water at the measured
           water temperature and the station's barometric pressure:
             ln(DO_sat,1atm) = -139.34411 + 1.575701e5/T - 6.642308e7/T^2
                               + 1.243800e10/T^3 - 8.621949e11/T^4,  T in K
           (Benson & Krause 1984, Limnology and Oceanography 29(3):620-632,
           as used in the USGS DOTABLES and APHA Standard Methods 4500-O),
           times the standard-atmosphere pressure ratio at elevation h (m),
             P/P0 = (1 - 2.25577e-5 h)^5.25588.
           Water holds less oxygen when warm and at altitude; a linear model in
           water temperature and elevation can only approximate this curve.
  wt_diel  water temperature minus its centred 24-hour rolling mean at the same
           station (time-based window, at least 12 hourly values): the day's
           heating and cooling, a sunlight / photosynthesis signal. Water
           temperature itself is already a DO predictor, so water temperature
           and wt_diel together also give the 24-hour mean. The centred window
           uses up to 12 hours after the row, which is fine for filling in a
           station's record but not for real-time prediction.

Rows: every node-hour of data/benchmark_data/all_static_df.pkl with a water
temperature. wt_diel is NaN where the 24-hour window holds fewer than 12 values;
the benchmark (BENCHMARK_DOSAT=baseline or on) keeps only rows with both
columns, in both arms, so the two runs are paired. The same wt_diel definition
is used by the pH option (BENCHMARK_PHDIEL).

writes data/benchmark_data/dosat_features.pkl  (node, time, do_sat, wt_diel)
"""
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK = Path("data/benchmark_data/all_static_df.pkl")
OUT = Path("data/benchmark_data/dosat_features.pkl")
COLS = ["do_sat", "wt_diel"]


def do_saturation(water_temp_c, elevation_m):
    """Benson & Krause (1984) fresh-water DO saturation (mg/L) at station pressure."""
    T = np.asarray(water_temp_c, dtype=float) + 273.15
    ln_do = (-139.34411 + 1.575701e5 / T - 6.642308e7 / T**2
             + 1.243800e10 / T**3 - 8.621949e11 / T**4)
    pressure = (1 - 2.25577e-5 * np.asarray(elevation_m, dtype=float)) ** 5.25588
    return np.exp(ln_do) * pressure


def diel_part(group):
    """Water temperature minus its centred 24-hour rolling mean (min 12 values)."""
    s = group.set_index("time")["Water temperature"]
    mean_24h = s.rolling("24h", center=True, min_periods=12).mean()
    return pd.Series((s - mean_24h).values, index=group.index)


def main():
    df = pd.read_pickle(BENCHMARK)[["node", "time", "Water temperature", "elevation_m"]]
    df = df.dropna(subset=["Water temperature"]).sort_values(["node", "time"])
    if df.duplicated(["node", "time"]).any():
        raise ValueError("duplicate node-hours in all_static_df")
    df["do_sat"] = do_saturation(df["Water temperature"], df["elevation_m"])
    df["wt_diel"] = df.groupby("node", group_keys=False)[["time", "Water temperature"]].apply(diel_part)
    out = df[["node", "time", *COLS]].reset_index(drop=True)
    out.to_pickle(OUT)
    ok = out[COLS].notna().all(axis=1)
    print(f"{len(out):,} node-hours with water temperature, {ok.sum():,} ({ok.mean():.2%}) with both "
          f"features, {out.loc[ok, 'node'].nunique()} nodes")
    print(f"do_sat range {out['do_sat'].min():.2f} .. {out['do_sat'].max():.2f} mg/L; "
          f"wt_diel sd {out['wt_diel'].std():.2f} C")


if __name__ == "__main__":
    main()
