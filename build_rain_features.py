"""Upstream rainfall features for the water-quality stations, from the MRMS
download (download_mrms.py), on the uncompressed and on the compressed graph.

usage: python build_rain_features.py        (about a minute)

For every water-quality node i and hour t, the rain over the river upstream of
i (its ancestors on the directed graph, edges pointing downstream, plus i
itself), as the equally weighted mean over those nodes of the MRMS cell each
node lies in:
  uncompressed  every node of graphs/colorado/largest_river_component.pkl
                (67,401 river cells of 90 m, so the mean weights each cell by
                the length of upstream channel in it): rain everywhere along the
                upstream river
  compressed    only the 67 nodes of master_compressed_river_graph.pkl
                (stations, gauges, confluences and channel ends): rain at a few
                points upstream; a station with no compressed node upstream
                uses its own cell
The two differ only in where the rain is sampled, so a paired run tells
whether rain at every node adds anything over rain at the graph's nodes.

Features (fixed in advance): rain_1h, rain_6h, rain_24h, rain_72h, rain_168h,
log(1 + mm summed over the last 1, 6, 24, 72 and 168 hours). The benchmark row
of hour t holds the median of [t, t+1h); an MRMS file stamped t holds the rain
of (t-1h, t], so rain_1h at row t is that file and every window ends before the
measurement hour (no rain after the reading is used). A missing MRMS hour
(83 in all, none in the main 2025-26 window) is left out of the sums; a window
with no hour at all (only at the very start of the archive) counts as 0.

writes data/benchmark_data/rain_features_uncompressed.pkl and
       data/benchmark_data/rain_features_compressed.pkl
       (node, time, rain_1h ... rain_168h) for the benchmark node-hours
"""
import glob
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

MRMS = Path("data/rain/mrms")
GRAPH = Path("graphs/colorado/largest_river_component.pkl")
COMPRESSED = Path("graphs/colorado/master_compressed_river_graph.pkl")
BENCHMARK = Path("data/benchmark_data/all_static_df.pkl")
WINDOWS = [1, 6, 24, 72, 168]
COLS = [f"rain_{w}h" for w in WINDOWS]


def weights(G, nodes, wq, cell_of, n_cells):
    """(stations x cells) matrix: row i averages the cells of i's upstream nodes in `nodes`."""
    W = np.zeros((len(wq), n_cells))
    for k, i in enumerate(wq):
        up = [n for n in nx.ancestors(G, i) if n in nodes] + [i]
        np.add.at(W[k], [cell_of[n] for n in up], 1.0)
        W[k] /= len(up)
    return W


def main():
    G = pickle.load(open(GRAPH, "rb"))
    Gc = pickle.load(open(COMPRESSED, "rb"))
    wq = sorted(n for n, d in Gc.nodes(data=True) if d.get("usgs_clusters"))
    cell_of = pd.read_csv(MRMS / "node_cells.csv").set_index("node")["cell"].to_dict()
    n_cells = len(pd.read_csv(MRMS / "cells.csv"))
    schemes = {"uncompressed": weights(G, set(G.nodes), wq, cell_of, n_cells),
               "compressed": weights(G, set(Gc.nodes), wq, cell_of, n_cells)}
    for name, W in schemes.items():
        print(f"{name}: upstream nodes per station "
              f"min {int((W > 0).sum(1).min())} cells, median {int(np.median((W > 0).sum(1)))} cells")

    # hourly upstream-mean rain at every WQ node (NaN where the hour is missing)
    times, series = [], {k: [] for k in schemes}
    for f in sorted(glob.glob(str(MRMS / "days" / "*.npz"))):
        z = np.load(f)
        rain = z["rain"].astype(float)                     # hours x cells
        ok = ~np.isnan(rain).all(1)
        times.append(pd.to_datetime(z["times"]).tz_localize("UTC"))
        for k, W in schemes.items():
            v = np.full((len(rain), len(wq)), np.nan)
            v[ok] = np.nan_to_num(rain[ok]) @ W.T
            series[k].append(v)
    times = pd.DatetimeIndex(np.concatenate(times))
    full = pd.date_range(times.min(), times.max(), freq="h")

    bench = pd.read_pickle(BENCHMARK)[["node", "time"]].drop_duplicates()
    for k in schemes:
        hourly = pd.DataFrame(np.vstack(series[k]), index=times, columns=wq).reindex(full)
        feats = []
        for w, col in zip(WINDOWS, COLS):
            s = hourly.rolling(w, min_periods=1).sum().fillna(0.0)   # mm over the last w hours
            feats.append(np.log1p(s).stack().rename(col))
        out = pd.concat(feats, axis=1).rename_axis(["time", "node"]).reset_index()
        out = bench.merge(out, on=["node", "time"], how="left")
        missing = out[COLS].isna().any(axis=1).mean()
        out[COLS] = out[COLS].fillna(0.0)
        out.to_pickle(f"data/benchmark_data/rain_features_{k}.pkl")
        print(f"{k}: {len(out):,} node-hours, {missing:.4%} outside the rain record "
              f"(set to 0), share of hours with rain_24h > 0: {(out['rain_24h'] > 0).mean():.3f}")


if __name__ == "__main__":
    main()
