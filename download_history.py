"""Download the longer water-quality history for the benchmark stations.

usage: python download_history.py          (resumable; run on the server)

For every station and water-quality variable of the current benchmark
(data/usgs/usgs_download_plan.csv) it downloads the instantaneous values from
HISTORY_START (water year 2021, fixed before any results) to the end of the
series in data/usgs/usgs_28stations_metadata.csv, one file per water year:
    data/usgs/history_raw/<station>/<variable>_WY<year>.csv.gz
The current benchmark window is downloaded again with the rest; build_history.py
keeps the current rows unchanged and only adds rows outside them.

The USGS API key is optional and read from the environment (USGS_API_KEY);
without it the public rate limit applies and the script waits when limited.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

HISTORY_START = pd.Timestamp("2020-10-01", tz="UTC")
URL = "https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/items"
OUT = Path("data/usgs/history_raw")
KEEP = ["time", "value", "approval_status", "last_modified"]
VARIABLES = ["DO concentration", "Water temperature", "Specific conductance", "Turbidity", "pH"]
PAGE_LIMIT = 50000


def safe(variable):
    return variable.lower().replace(" ", "_").replace("/", "_")


def get(session, url, params=None):
    for attempt in range(20):
        try:
            r = session.get(url, params=params, timeout=180)
        except requests.RequestException as e:
            print(f"    network error ({e}); retrying in 30 s", flush=True)
            time.sleep(30)
            continue
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 60))
            print(f"    rate limited; waiting {wait} s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code in (500, 502, 503, 504):
            print(f"    server error {r.status_code}; retrying in 30 s", flush=True)
            time.sleep(30)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"giving up on {url}")


def download(session, station, pcode, start, end):
    params = {
        "f": "json", "limit": PAGE_LIMIT, "monitoring_location_id": station,
        "parameter_code": pcode, "time": f"{start.isoformat()}/{end.isoformat()}",
    }
    rows, url = [], URL
    while url:
        payload = get(session, url, params)
        rows += [f.get("properties", {}) for f in payload.get("features", [])]
        url = next((l["href"] for l in payload.get("links", []) if l.get("rel") == "next"), None)
        params = None
        time.sleep(0.1)
    df = pd.DataFrame(rows)
    return df.reindex(columns=KEEP)


def main():
    plan = pd.read_csv("data/usgs/usgs_download_plan.csv", dtype={"parameter_code": str})
    plan = plan[plan["variable"].isin(VARIABLES)]
    meta = pd.read_csv("data/usgs/usgs_28stations_metadata.csv", low_memory=False)
    meta = meta[meta["variable"].isin(VARIABLES)]
    meta["begin"] = pd.to_datetime(meta["begin"], utc=True, errors="coerce")
    meta["end"] = pd.to_datetime(meta["end"], utc=True, errors="coerce")
    span = meta.groupby(["monitoring_location_id", "variable"]).agg(begin=("begin", "min"), end=("end", "max"))

    session = requests.Session()
    if os.environ.get("USGS_API_KEY"):
        session.headers["X-Api-Key"] = os.environ["USGS_API_KEY"]
    else:
        print("USGS_API_KEY not set: using the public rate limit", flush=True)

    jobs = []
    for _, row in plan.iterrows():
        key = (row["station_id"], row["variable"])
        if key not in span.index:
            continue
        begin = max(HISTORY_START, span.loc[key, "begin"])
        end = span.loc[key, "end"]
        year = begin.year + (begin.month >= 10)
        while True:
            wy_start = pd.Timestamp(f"{year - 1}-10-01", tz="UTC")
            wy_end = pd.Timestamp(f"{year}-10-01", tz="UTC")
            if wy_start >= end:
                break
            jobs.append((row["station_id"], row["variable"], str(row["parameter_code"]).zfill(5),
                         year, max(wy_start, begin), min(wy_end, end)))
            year += 1

    print(f"{len(jobs)} station-variable-water-year files", flush=True)
    for i, (station, variable, pcode, year, start, end) in enumerate(jobs, 1):
        path = OUT / station / f"{safe(variable)}_WY{year}.csv.gz"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        df = download(session, station, pcode, start, end)
        tmp = path.with_suffix(".tmp")
        df.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, path)
        print(f"[{i}/{len(jobs)}] {station} {variable} WY{year}: {len(df):,} rows", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
