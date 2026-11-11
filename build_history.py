"""Build the benchmark tables with the longer history (BENCHMARK_HISTORY).

usage:
  python build_history.py --check   rebuild the CURRENT tables from
                                    data/usgs/usgs_continuous_download and the
                                    current windows, and compare them with
                                    data/benchmark_data (must match exactly)
  python build_history.py           build the extended tables from
                                    data/usgs/history_raw (download_history.py)

The extended tables keep every current row unchanged, with the same index, and
append the added node-hours after them; the column orig is 1 for current rows
and 0 for added ones. Added rows are the node-hours outside each node's current
rows, from water year 2021 on. They go through the same steps as the current
data (hourly median of the instantaneous values, median over the stations of a
node, the same seasonal and static columns), with one difference in cleaning:
the current pipeline keeps only approved values when a series has any, which
over several years would drop the recent provisional months, so the added rows
prefer the approved value per timestamp instead.

writes data/benchmark_data_history/all_static_df.pkl and fourier_interaction_df.pkl
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WQ = ["DO concentration", "Water temperature", "Specific conductance", "Turbidity", "pH"]
CURRENT = Path("data/benchmark_data")
OUT = Path("data/benchmark_data_history")
STATIC = ["elevation_m", "log_upa", "log_width", "hand_m", "local_slope"]


def safe(variable):
    return variable.lower().replace(" ", "_").replace("/", "_")


def station_to_node():
    with open("graphs/colorado/master_compressed_river_graph.pkl", "rb") as f:
        G = pickle.load(f)
    out = {}
    for node, data in G.nodes(data=True):
        for cluster in data.get("usgs_clusters", []):
            stations = cluster.get("member_stations", []) or (
                [cluster["anchor_station"]] if cluster.get("anchor_station") is not None else []
            )
            for s in stations:
                out[s] = node
    return out


def clean_current(df, variable):
    """clean_raw_series of USGS_Data_Cleaning.ipynb."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time", "value"])
    if "approval_status" in df.columns:
        approved = df["approval_status"].fillna("").astype(str).str.strip().str.lower().eq("approved")
        if approved.any():
            df = df[approved].copy()
    if "last_modified" in df.columns:
        df["last_modified"] = pd.to_datetime(df["last_modified"], errors="coerce", utc=True)
        df = df.sort_values(["time", "last_modified"])
    else:
        df = df.sort_values("time")
    return (df.drop_duplicates(subset=["time"], keep="last")[["time", "value"]]
            .rename(columns={"value": variable}).sort_values("time").reset_index(drop=True))


def clean_added(df, variable):
    """As clean_current, but the approved value is preferred per timestamp."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time", "value"])
    df["approved"] = df["approval_status"].fillna("").astype(str).str.strip().str.lower().eq("approved")
    df["last_modified"] = pd.to_datetime(df["last_modified"], errors="coerce", utc=True)
    df = df.sort_values(["time", "approved", "last_modified"])
    return (df.drop_duplicates(subset=["time"], keep="last")[["time", "value"]]
            .rename(columns={"value": variable}).sort_values("time").reset_index(drop=True))


def hourly(clean, variable, start=None, end=None):
    h = clean.set_index("time")[[variable]].resample("1h").median()
    if start is not None:  # current pipeline: the planned window, all hours
        idx = pd.date_range(start.floor("h"), end.ceil("h"), freq="1h", tz="UTC")
        h = h.reindex(idx)
    else:
        h = h.dropna()
    h.index.name = "time"
    return h.reset_index()


def node_table(station_frames, mapping):
    frames = []
    for station, parts in station_frames.items():
        merged = parts[0]
        for p in parts[1:]:
            merged = merged.merge(p, on="time", how="outer")
        merged = merged.sort_values("time").drop_duplicates("time")
        merged["node"] = mapping.get(station)
        frames.append(merged)
    allh = pd.concat(frames, ignore_index=True, sort=False)
    allh = allh[allh["node"].notna()].copy()
    allh["node"] = allh["node"].astype(int)
    for v in WQ:
        if v not in allh:
            allh[v] = np.nan
    return allh.groupby(["node", "time"], as_index=False)[WQ].median()


def add_features(df, statics):
    """Seasonal columns of largest_uncompressed_laplacian.ipynb (K = 1..5) and the
    daily x annual interactions; returns (all_static rows, fourier_interaction rows)."""
    dt = pd.to_datetime(df["time"])
    hour = dt.dt.hour + dt.dt.minute / 60.0
    doy = dt.dt.dayofyear + hour / 24.0
    f = df[["node", "time", *WQ]].copy()
    for k in range(1, 6):
        f[f"hour_sin_{k}"] = np.sin(2 * np.pi * k * hour / 24.0)
        f[f"hour_cos_{k}"] = np.cos(2 * np.pi * k * hour / 24.0)
        f[f"doy_sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        f[f"doy_cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    fi = f.copy()
    for k in range(1, 5):
        for dl, dc in (("hsin", f"hour_sin_{k}"), ("hcos", f"hour_cos_{k}")):
            for yl, yc in (("ysin", "doy_sin_1"), ("ycos", "doy_cos_1")):
                fi[f"interaction_{dl}{k}_{yl}"] = f[dc] * f[yc]
    st = f.merge(statics, on="node", how="left")
    st.index = f.index
    return st, fi


def compare(new, old, name):
    key = ["node", "time"]
    a = old.sort_values(key).reset_index(drop=True)
    b = new.sort_values(key).reset_index(drop=True)
    assert list(a.columns) == list(b.columns), (name, list(a.columns), list(b.columns))
    assert len(a) == len(b), (name, len(a), len(b))
    assert (a["node"].to_numpy() == b["node"].to_numpy()).all(), name
    assert (a["time"] == b["time"]).all(), name
    worst = 0.0
    for c in a.columns[2:]:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        assert (np.isnan(x) == np.isnan(y)).all(), (name, c)
        ok = ~np.isnan(x)
        worst = max(worst, float(np.max(np.abs(x[ok] - y[ok]), initial=0.0)))
    print(f"{name}: {len(a):,} rows match, largest difference {worst:.2e}")
    return worst


def check():
    mapping = station_to_node()
    plan = pd.read_csv("data/usgs/usgs_download_plan.csv")
    plan["start"] = pd.to_datetime(plan["start"], utc=True)
    plan["end"] = pd.to_datetime(plan["end"], utc=True)
    frames = {}
    for _, row in plan.iterrows():
        path = Path("data/usgs/usgs_continuous_download") / row["station_id"] / f"{safe(row['variable'])}.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            raw = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if raw.empty or not {"time", "value"} <= set(raw.columns):
            continue
        c = clean_current(raw, row["variable"])
        if c.empty:
            continue
        frames.setdefault(row["station_id"], []).append(
            hourly(c, row["variable"], row["start"], row["end"]))
    nodes = node_table(frames, mapping)
    old_st = pd.read_pickle(CURRENT / "all_static_df.pkl")
    old_fi = pd.read_pickle(CURRENT / "fourier_interaction_df.pkl")
    statics = old_st.groupby("node", as_index=False)[STATIC].first()
    st, fi = add_features(nodes, statics)
    w1 = compare(st, old_st, "all_static_df")
    w2 = compare(fi, old_fi, "fourier_interaction_df")
    assert max(w1, w2) < 1e-9


def build():
    mapping = station_to_node()
    old_st = pd.read_pickle(CURRENT / "all_static_df.pkl")
    old_fi = pd.read_pickle(CURRENT / "fourier_interaction_df.pkl")
    assert old_st.index.equals(old_fi.index)
    assert (old_st[["node", "time"]].to_numpy() == old_fi[["node", "time"]].to_numpy()).all()

    frames = {}
    for station_dir in sorted(Path("data/usgs/history_raw").iterdir()):
        for v in WQ:
            files = sorted(station_dir.glob(f"{safe(v)}_WY*.csv.gz"))
            parts = []
            for f in files:
                try:
                    parts.append(pd.read_csv(f))
                except pd.errors.EmptyDataError:
                    pass
            parts = [p for p in parts if len(p) and "value" in p]
            if not parts:
                continue
            c = clean_added(pd.concat(parts, ignore_index=True), v)
            if len(c):
                frames.setdefault(station_dir.name, []).append(hourly(c, v))
    nodes = node_table(frames, mapping)
    nodes = nodes[nodes[WQ].notna().any(axis=1)]
    nodes = nodes[nodes["node"].isin(old_st["node"].unique())]

    current = pd.MultiIndex.from_frame(old_st[["node", "time"]])
    added = nodes[~pd.MultiIndex.from_frame(nodes[["node", "time"]]).isin(current)]
    added = added.sort_values(["node", "time"]).reset_index(drop=True)
    added.index = added.index + (old_st.index.max() + 1)

    statics = old_st.groupby("node", as_index=False)[STATIC].first()
    st, fi = add_features(added, statics)
    st = pd.concat([old_st.assign(orig=1), st.assign(orig=0)])
    fi = pd.concat([old_fi.assign(orig=1), fi.assign(orig=0)])
    assert st.index.is_unique and st.index.equals(fi.index)

    OUT.mkdir(parents=True, exist_ok=True)
    st.to_pickle(OUT / "all_static_df.pkl")
    fi.to_pickle(OUT / "fourier_interaction_df.pkl")

    rows = []
    for v in WQ:
        for orig, g in st[st[v].notna()].groupby("orig"):
            t = pd.to_datetime(g["time"])
            rows.append({"variable": v, "rows": "current" if orig else "added", "stations": g["node"].nunique(),
                         "node_hours": len(g), "first": t.min().date(), "last": t.max().date()})
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(OUT / "history_summary.csv", index=False)


if __name__ == "__main__":
    check() if "--check" in sys.argv else build()
