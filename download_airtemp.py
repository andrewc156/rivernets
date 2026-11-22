"""Hourly 2 m air temperature (ERA5-Land) at every water-quality node, for the
air-temperature experiment (BENCHMARK_AIRTEMP, build_airtemp_features.py).

usage: python download_airtemp.py        (26 requests, about two minutes)

Source: the Open-Meteo historical weather API (archive-api.open-meteo.com),
model ERA5-Land (models=era5_land): ECMWF's land-surface reanalysis, hourly on a
0.1 degree grid, 1950 to about five days ago. Open-Meteo serves the nearest grid
cell and adjusts the temperature to the elevation of the requested point (its
90 m digital elevation model); both the grid cell and that elevation are saved
in the request log. Times are UTC (timezone=GMT), the same as the benchmark rows.

  Munoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global
    reanalysis dataset for land applications. Earth System Science Data 13,
    4349-4383. doi:10.5194/essd-13-4349-2021
  Zippenfenig, P. (2023). Open-Meteo.com Weather API. Zenodo.
    doi:10.5281/zenodo.7970649

One request per WQ node of the compressed graph, at the node's latitude and
longitude, covering the node's time range in data/benchmark_data/all_static_df.pkl
plus the 8 days before it (so the trailing 7-day mean is complete on the first
row). Only the node coordinates and dates are sent (public USGS station
locations); no personal or project information, and the default urllib
User-Agent. Requests are spaced out and retried with backoff on HTTP 429/5xx.

Responses are cached as data/airtemp/node_<node>.csv (time, temperature_2m in
degrees C); a node whose file exists and covers its window is not downloaded
again. data/airtemp/requests.json records what was requested and returned.
"""
import json
import pickle
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://archive-api.open-meteo.com/v1/archive"
MODEL = "era5_land"
COMPRESSED = Path("graphs/colorado/master_compressed_river_graph.pkl")
BENCHMARK = Path("data/benchmark_data/all_static_df.pkl")
OUT = Path("data/airtemp")
PAD_DAYS = 8
PAUSE_S = 3.0
BACKOFF_S = (10, 30, 60, 120, 300)


def fetch(params):
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt, wait in enumerate((0, *BACKOFF_S)):
        if wait:
            print(f"    retry {attempt} in {wait} s")
            time.sleep(wait)
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
            print(f"    HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"    {e}")
    raise RuntimeError(f"giving up on {url}")


def main():
    G = pickle.load(open(COMPRESSED, "rb"))
    bench = pd.read_pickle(BENCHMARK)[["node", "time"]]
    span = bench.groupby("node")["time"].agg(["min", "max"])
    OUT.mkdir(parents=True, exist_ok=True)
    log_file = OUT / "requests.json"
    log = json.load(open(log_file)) if log_file.exists() else {}

    for i, (node, row) in enumerate(span.iterrows(), 1):
        start = (row["min"] - pd.Timedelta(days=PAD_DAYS)).strftime("%Y-%m-%d")
        end = row["max"].strftime("%Y-%m-%d")
        lat, lon = G.nodes[node]["latitude"], G.nodes[node]["longitude"]
        csv = OUT / f"node_{node}.csv"
        if csv.exists():
            t = pd.read_csv(csv, usecols=["time"])["time"]
            if t.iloc[0] <= f"{start}T00:00" and t.iloc[-1] >= f"{end}T23:00":
                print(f"[{i:2d}/{len(span)}] node {node}: cached")
                continue
        params = {
            "latitude": round(lat, 5), "longitude": round(lon, 5),
            "start_date": start, "end_date": end,
            "hourly": "temperature_2m", "timezone": "GMT", "models": MODEL,
        }
        print(f"[{i:2d}/{len(span)}] node {node}: {start} .. {end}")
        d = fetch(params)
        h = pd.DataFrame({"time": d["hourly"]["time"], "temperature_2m": d["hourly"]["temperature_2m"]})
        missing = int(h["temperature_2m"].isna().sum())
        h.to_csv(csv, index=False)
        log[str(node)] = {
            "requested": params, "grid_latitude": d.get("latitude"),
            "grid_longitude": d.get("longitude"), "elevation_m": d.get("elevation"),
            "hours": len(h), "missing_hours": missing,
            "units": d.get("hourly_units", {}).get("temperature_2m"),
        }
        json.dump(log, open(log_file, "w"), indent=1)
        print(f"    {len(h):,} hours, {missing} missing, grid cell "
              f"({d.get('latitude')}, {d.get('longitude')}), elevation {d.get('elevation')} m")
        time.sleep(PAUSE_S)


if __name__ == "__main__":
    main()
