"""River distance from the nearest major dam upstream of every water-quality node
(for BENCHMARK_TAILWATER in water_quality_model_benchmark.py).

usage: python build_tailwater_features.py        (under a minute)

Why: water released from the bottom of a large reservoir is cold in summer and
mild in winter, so just below a dam the annual cycle of water temperature is
damped and shifted. The benchmark's annual Fourier terms are the same at every
station, and the three Lower Colorado / Verde tailwater stations carry a large
share of the water-temperature and DO error. The dam option of the main
benchmark (MAJOR_DAMS, the distance_dams weighting) treated dams only as
high-resistance graph edges and, because dams were snapped to the graph within
2 km, it missed:
  - Bartlett Dam on the Verde, directly above USGS-09510000 (not in the list);
  - Blue Mesa, Navajo, McPhee, Granby and Dillon, which lie 18-116 km from the
    compressed graph (the graph starts below them), so they were skipped.
MAJOR_DAMS itself is left unchanged so that weighting stays reproducible; this
script uses TAILWATER_DAMS = MAJOR_DAMS + Bartlett.

How each dam is attached to the compressed river graph (edges point downstream,
every edge keeps the MERIT cells it replaced in original_path):
  snapped  the dam lies within 2 km of a graph cell (as in the benchmark); the
           river distance starts at that cell.
  traced   otherwise: start at the cell of largest upstream area within 1.5 km
           of the dam (the river channel at the dam), follow the MERIT Hydro D8
           flow directions (data/colorado/merit_colorado_dir_native.tif, ESRI
           codes 1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE, same 3-arcsecond grid
           as the graph's node ids) downstream until the trace reaches a graph
           cell; distance = traced length + distance onward along the graph.
           A traced dam is kept only if it reaches the graph within
           MAX_TRACE_KM and the first water-quality station below it is on the
           expected river (EXPECTED_PREFIX: the station's 4-digit USGS prefix);
           otherwise it is dropped and reported.
From the attachment point the distance is accumulated downstream, node by node;
every water-quality node met on the way gets that distance, and each node keeps
the smallest distance over all dams (the nearest major dam upstream).

Features used by the benchmark (fixed in advance, as in the screen):
  tw_tail                          exp(-d / 100 km), 0 with no dam upstream
  tw_tail_doy_{sin,cos}_{1,2}      tw_tail x the annual Fourier terms
(the products are formed in the benchmark, per row).

writes data/benchmark_data/tailwater_features.csv   one row per WQ node:
           node, stations, dam, dam_km, attach, tw_tail
       data/benchmark_data/tailwater_dams.csv       one row per dam:
           dam, attach, offset_km, trace_km, join_lat, join_lon,
           first_station, first_station_km, kept, note
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

GRAPH = Path("graphs/colorado/master_compressed_river_graph.pkl")
DIR_TIF = Path("data/colorado/merit_colorado_dir_native.tif")
FLOAT_TIF = Path("data/colorado/merit_colorado_float_native.tif")  # band 1 = upa (km2)
OUT_NODES = Path("data/benchmark_data/tailwater_features.csv")
OUT_DAMS = Path("data/benchmark_data/tailwater_dams.csv")

# The benchmark's MAJOR_DAMS (water_quality_model_benchmark.py), copied unchanged,
# plus Bartlett Dam (Verde River, above USGS-09510000).
MAJOR_DAMS = {
    "Glen Canyon": (36.9375, -111.4844),
    "Hoover": (36.0161, -114.7377),
    "Davis": (35.1972, -114.5697),
    "Parker": (34.2964, -114.1394),
    "Flaming Gorge": (40.9147, -109.4217),
    "Fontenelle": (42.0275, -110.0675),
    "Blue Mesa": (38.4528, -107.3358),
    "Navajo": (36.8006, -107.6125),
    "McPhee": (37.5761, -108.5747),
    "Granby": (40.1483, -105.8664),
    "Dillon": (39.6247, -106.0647),
    "Theodore Roosevelt": (33.6717, -111.1617),
    "Coolidge": (33.1736, -110.5250),
}
TAILWATER_DAMS = {**MAJOR_DAMS, "Bartlett": (33.8186, -111.6322)}

SNAP_KM = 2.0          # as DAM_SNAP_KM in the benchmark
CHANNEL_KM = 1.5       # search radius for the river channel at a traced dam
MAX_TRACE_KM = 400.0
SCALE_KM = 100.0       # tw_tail = exp(-d / SCALE_KM)
# expected first water-quality station below an off-network dam (USGS 4-digit prefix range)
EXPECTED_PREFIX = {
    "Blue Mesa": ("0911", "0915"),   # Gunnison
    "Navajo": ("0934", "0937"),      # San Juan
    "McPhee": ("0916", "0918"),      # Dolores
    "Granby": ("0900", "0910"),      # upper Colorado
    "Dillon": ("0900", "0910"),      # Blue River -> upper Colorado
}

COLS = 15001                         # MERIT node id = row * COLS + col
CELL_KM = 111.32 / 1200.0            # one 3-arcsecond cell, north-south
# ESRI D8 code -> (d_row, d_col)
D8 = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}


class BlockReader:
    """Reads one band of a large raster in 512 x 512 blocks, on demand."""

    def __init__(self, path, band=1, block=512):
        self.src = rasterio.open(path)
        self.band, self.block, self.cache = band, block, {}

    def __call__(self, r, c):
        if not (0 <= r < self.src.height and 0 <= c < self.src.width):
            return None
        key = (r // self.block, c // self.block)
        if key not in self.cache:
            r0, c0 = key[0] * self.block, key[1] * self.block
            h, w = min(self.block, self.src.height - r0), min(self.block, self.src.width - c0)
            self.cache[key] = self.src.read(self.band, window=Window(c0, r0, w, h))
        return self.cache[key][r % self.block, c % self.block]


def main():
    G = pickle.load(open(GRAPH, "rb"))
    any_d = next(iter(G.nodes(data=True)))[1]
    lat0 = any_d["latitude"] + any_d["row"] / 1200.0      # latitude of row 0 (cell centre)
    lon0 = any_d["longitude"] - any_d["col"] / 1200.0     # longitude of col 0
    for n, d in G.nodes(data=True):
        if divmod(int(n), COLS) != (d["row"], d["col"]):
            raise ValueError("MERIT node id does not decode to its row/col")

    stations = {n: [s for c in d.get("usgs_clusters") or [] for s in c["member_stations"]]
                for n, d in G.nodes(data=True)}
    wq_nodes = {n for n, s in stations.items() if s}

    # every graph cell -> (edge, remaining length to the edge's downstream node, m)
    cell_rows = []
    for u, v, e in G.edges(data=True):
        p = e["original_path"]
        for k, cell in enumerate(p):
            cell_rows.append((int(cell), u, v, e["length_m"] * (1 - k / len(p))))
    cells = pd.DataFrame(cell_rows, columns=["cell", "u", "v", "rem_m"])
    cells["row"], cells["col"] = np.divmod(cells["cell"].to_numpy(), COLS)
    cells["lat"] = lat0 - cells["row"] / 1200.0
    cells["lon"] = lon0 + cells["col"] / 1200.0
    on_graph = cells.groupby("cell")["rem_m"].idxmin()  # one entry per cell (shortest remaining)

    def walk_down(node, d_km):
        """Distances of the WQ nodes from node (at d_km) to the outlet."""
        out = {}
        while True:
            if node in wq_nodes:
                out[node] = d_km
            nxt = list(G.successors(node))
            if not nxt:
                return out
            d_km += G.edges[node, nxt[0]]["length_m"] / 1000.0
            node = nxt[0]

    def km_between(la, lo, la2, lo2):
        return np.hypot((la2 - la) * 111.0, (lo2 - lo) * 111.0 * np.cos(np.radians(la)))

    dirs = BlockReader(DIR_TIF)
    upa = BlockReader(FLOAT_TIF, band=1)

    best, dam_rows = {}, []
    for name, (la, lo) in TAILWATER_DAMS.items():
        km = km_between(la, lo, cells["lat"].to_numpy(), cells["lon"].to_numpy())
        i = int(np.argmin(km))
        row = dict(dam=name, offset_km=round(float(km[i]), 1), trace_km=0.0, kept=True, note="")
        if km[i] <= SNAP_KM:
            join = cells.iloc[i]
            row["attach"] = "snapped"
        else:
            row["attach"] = "traced"
            # river channel at the dam: largest upstream area within CHANNEL_KM
            r0 = int(round((lat0 - la) * 1200)); c0 = int(round((lo - lon0) * 1200))
            rad = int(np.ceil(CHANNEL_KM / CELL_KM))
            cand = [(upa(r, c), r, c) for r in range(r0 - rad, r0 + rad + 1)
                    for c in range(c0 - rad, c0 + rad + 1)
                    if km_between(la, lo, lat0 - r / 1200.0, lon0 + c / 1200.0) <= CHANNEL_KM]
            _, r, c = max(cand)
            trace_km, join = 0.0, None
            while trace_km <= MAX_TRACE_KM:
                cell = r * COLS + c
                if cell in on_graph.index:
                    join = cells.loc[on_graph[cell]]
                    break
                code = dirs(r, c)
                if code not in D8:
                    break                                  # outlet, depression or nodata
                dr, dc = D8[int(code)]
                trace_km += CELL_KM * np.hypot(dr, dc * np.cos(np.radians(lat0 - r / 1200.0)))
                r, c = r + dr, c + dc
            row["trace_km"] = round(trace_km, 1)
            if join is None:
                row.update(kept=False, note=f"trace did not reach the graph within {MAX_TRACE_KM:.0f} km")
        if join is not None:
            row["join_lat"], row["join_lon"] = round(float(join["lat"]), 4), round(float(join["lon"]), 4)
            reached = walk_down(join["v"], row["trace_km"] + join["rem_m"] / 1000.0)
            if reached:
                first = min(reached, key=reached.get)
                st = stations[first][0]
                row["first_station"], row["first_station_km"] = st, round(reached[first], 1)
                if name in EXPECTED_PREFIX:
                    lo_p, hi_p = EXPECTED_PREFIX[name]
                    if not lo_p <= st.split("-")[1][:4] <= hi_p:
                        row.update(kept=False, note=f"first station {st} is not on the expected river")
            else:
                row["note"] = "no water-quality station downstream"
            if row["kept"]:
                for n, d in reached.items():
                    if d < best.get(n, (np.inf, None))[0]:
                        best[n] = (d, name, row["attach"])
        dam_rows.append(row)

    dams = pd.DataFrame(dam_rows)[["dam", "attach", "offset_km", "trace_km", "join_lat", "join_lon",
                                   "first_station", "first_station_km", "kept", "note"]]
    nodes = pd.DataFrame([
        dict(node=n, stations=",".join(stations[n]),
             dam=best[n][1] if n in best else "", dam_km=round(best[n][0], 1) if n in best else np.inf,
             attach=best[n][2] if n in best else "")
        for n in sorted(wq_nodes)
    ])
    nodes["tw_tail"] = np.exp(-nodes["dam_km"] / SCALE_KM)
    nodes.to_csv(OUT_NODES, index=False)
    dams.to_csv(OUT_DAMS, index=False)

    pd.set_option("display.width", 200)
    print("Dams:")
    print(dams.fillna("").to_string(index=False))
    near = nodes[nodes["dam_km"] <= 150].sort_values("dam_km")
    print(f"\nWater-quality nodes within 150 km below a major dam ({len(near)} of {len(nodes)}):")
    print(near[["stations", "dam", "attach", "dam_km", "tw_tail"]].round(3).to_string(index=False))
    print(f"\nnodes with a dam upstream at any distance: {np.isfinite(nodes['dam_km']).sum()}; "
          f"wrote {OUT_NODES} and {OUT_DAMS}")


if __name__ == "__main__":
    main()
