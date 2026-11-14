"""Download hourly MRMS rainfall (MultiSensor QPE 01H Pass 2) at the river-graph nodes.

usage:
  python download_mrms.py                 every station's benchmark window (resumable;
                                          run on the server: 1,783 days, about 3 hours)
  python download_mrms.py --main-only     only the main window (most stations)
  python download_mrms.py --days 2026-08-01   one day only (a quick test)

Source: NOAA MRMS on AWS (s3://noaa-mrms-pds, public, no key),
  CONUS/MultiSensor_QPE_01H_Pass2_00.00/YYYYMMDD/
      MRMS_MultiSensor_QPE_01H_Pass2_00.00_YYYYMMDD-HH0000.grib2.gz
a 0.01 degree grid (7000 x 3500, west edge -130, north edge 55) of the radar +
gauge precipitation (mm) accumulated over the hour ENDING at the file time.
The archive starts on 2020-10-14.

Windows: for every station of data/usgs/usgs_download_plan.csv the benchmark
window (one year per station), extended LAG_DAYS earlier so that rainfall sums
over the previous days exist at the first hour; overlapping windows are merged.
17 stations share 2025-10-02 to 2026-10-02 (373 days with the lag); the other 11
have their own windows (2020-21, 2022-23 twice, 2023-24, five within 2024-25, and
two shorter ones ending 2026-10-02). --main-only keeps the windows that end with
the main one (2025-09-25 to 2026-10-02).

Only the grid cells under the nodes of the uncompressed river graph
(graphs/colorado/largest_river_component.pkl) are kept (each 90 m node lies in
one 1 km cell; about 7,300 distinct cells for 67,401 nodes). Decoding: dry cells
come out of the GRIB packing as about 4e-8 instead of 0, so values are rounded
to MRMS's 0.1 mm; -3 (no radar coverage) and missing files become NaN.

writes
  data/rain/mrms/cells.csv          cell index, grid row/col, cell-centre lon/lat
  data/rain/mrms/node_cells.csv     graph node -> cell index
  data/rain/mrms/days/YYYYMMDD.npz  times (24 hour-end timestamps, UTC) and
                                    rain (24 x cells, float32 mm, NaN = missing)
"""
import argparse
import gzip
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.io import MemoryFile

BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"
PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00"
GRAPH = Path("graphs/colorado/largest_river_component.pkl")
PLAN = Path("data/usgs/usgs_download_plan.csv")
OUT = Path("data/rain/mrms")
ARCHIVE_START = pd.Timestamp("2020-10-14", tz="UTC")
LAG_DAYS = 7
WEST, NORTH, STEP, WIDTH, HEIGHT = -130.0, 55.0, 0.01, 7000, 3500
WORKERS = 8


def node_cells():
    """Cells under the graph nodes; writes cells.csv and node_cells.csv once."""
    if (OUT / "cells.csv").exists() and (OUT / "node_cells.csv").exists():
        cells = pd.read_csv(OUT / "cells.csv")
        return cells["row"].to_numpy(), cells["col"].to_numpy()
    G = pickle.load(open(GRAPH, "rb"))
    nodes = pd.DataFrame(
        [(n, d["longitude"], d["latitude"]) for n, d in G.nodes(data=True)],
        columns=["node", "lon", "lat"])
    nodes["row"] = np.floor((NORTH - nodes["lat"]) / STEP).astype(int)
    nodes["col"] = np.floor((nodes["lon"] - WEST) / STEP).astype(int)
    if not (nodes["row"].between(0, HEIGHT - 1).all() and nodes["col"].between(0, WIDTH - 1).all()):
        raise ValueError("a node lies outside the MRMS grid")
    cells = nodes[["row", "col"]].drop_duplicates().sort_values(["row", "col"]).reset_index(drop=True)
    cells["cell"] = cells.index
    cells["lon"] = WEST + (cells["col"] + 0.5) * STEP
    cells["lat"] = NORTH - (cells["row"] + 0.5) * STEP
    nodes = nodes.merge(cells[["row", "col", "cell"]], on=["row", "col"])
    OUT.mkdir(parents=True, exist_ok=True)
    cells[["cell", "row", "col", "lon", "lat"]].to_csv(OUT / "cells.csv", index=False)
    nodes[["node", "cell"]].to_csv(OUT / "node_cells.csv", index=False)
    print(f"{len(nodes):,} nodes in {len(cells):,} MRMS cells", flush=True)
    return cells["row"].to_numpy(), cells["col"].to_numpy()


def windows(main_only):
    plan = pd.read_csv(PLAN)
    plan["start"] = pd.to_datetime(plan["start"], utc=True)
    plan["end"] = pd.to_datetime(plan["end"], utc=True)
    w = plan.groupby("station_id").agg(start=("start", "min"), end=("end", "max"))
    if main_only:
        main = w["end"].mode()[0]
        w = w[w["end"] == main]
    days = set()
    for start, end in zip(w["start"], w["end"]):
        first = max((start - pd.Timedelta(days=LAG_DAYS)).floor("D"), ARCHIVE_START)
        days.update(pd.date_range(first, end.floor("D"), freq="D"))
    skipped = (w["start"] - pd.Timedelta(days=LAG_DAYS) < ARCHIVE_START).sum()
    if skipped:
        print(f"{skipped} station window(s) start before the archive (2020-10-14); "
              "their first days have no rainfall", flush=True)
    return sorted(days)


def fetch_hour(session, t, rows, cols):
    name = f"MRMS_{PRODUCT}_{t:%Y%m%d-%H}0000.grib2.gz"
    url = f"{BUCKET}/CONUS/{PRODUCT}/{t:%Y%m%d}/{name}"
    for attempt in range(5):
        try:
            r = session.get(url, timeout=120)
        except requests.RequestException:
            continue
        if r.status_code == 404:
            return None
        if r.ok:
            with MemoryFile(gzip.decompress(r.content)) as mem, mem.open() as src:
                if src.width != WIDTH or src.height != HEIGHT:
                    raise ValueError(f"unexpected grid {src.width}x{src.height} in {name}")
                v = src.read(1)[rows, cols].astype(np.float32)
            v = np.round(v, 1)
            v[v < 0] = np.nan
            return v
    raise RuntimeError(f"giving up on {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-only", action="store_true")
    ap.add_argument("--days", nargs="*", help="YYYY-MM-DD days instead of the windows")
    args = ap.parse_args()

    rows, cols = node_cells()
    days = ([pd.Timestamp(d, tz="UTC") for d in args.days] if args.days
            else windows(args.main_only))
    (OUT / "days").mkdir(parents=True, exist_ok=True)
    print(f"{len(days)} days, {len(rows):,} cells", flush=True)

    session = requests.Session()
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, day in enumerate(days, 1):
            path = OUT / "days" / f"{day:%Y%m%d}.npz"
            if path.exists():
                continue
            # file times 01:00 ... 24:00 (next day 00:00): the hours ending within this day
            times = pd.date_range(day + pd.Timedelta(hours=1), periods=24, freq="h")
            vals = list(pool.map(lambda t: fetch_hour(session, t, rows, cols), times))
            missing = sum(v is None for v in vals)
            rain = np.stack([np.full(len(rows), np.nan, np.float32) if v is None else v for v in vals])
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez_compressed(tmp, times=times.tz_convert(None).to_numpy(), rain=rain)
            tmp.replace(path)
            print(f"[{i}/{len(days)}] {day:%Y-%m-%d}: {24 - missing} hours"
                  + (f", {missing} missing" if missing else "")
                  + f", max {np.nanmax(rain) if np.isfinite(rain).any() else float('nan'):.1f} mm",
                  flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
