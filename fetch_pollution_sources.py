"""Possible pollution sources upstream of the water-quality stations (public data).

usage: python fetch_pollution_sources.py        (a few minutes; public APIs, no keys)

Writes four files to data/pollution/:

  echo_npdes.csv           every active Clean Water Act discharge permit (NPDES and
                           related) in the Upper and Lower Colorado regions (2-digit
                           HUC 14 and 15), from EPA ECHO (echodata.epa.gov,
                           cwa_rest_services): name, city, state, location, SIC and
                           NAICS codes, permit type and status, major/minor, permit
                           components (POTW, stormwater, ...), design and actual
                           average flow (MGD), TRI surface-water releases (lb/yr),
                           receiving water, 8-digit HUC.
  osm_data_centers.csv     data centres mapped in OpenStreetMap (telecom=data_center,
                           building=data_center or data_centre) in a box around the
                           basin, via the Overpass API. Crowd-sourced and incomplete.
  station_comids.csv       each USGS station's NHDPlus reach (COMID) from the USGS
                           NLDI, and the reach whose StreamCat values are used (the
                           first one downstream within 10 km if its own has none).
  station_names.csv        USGS station names (NWIS site service).
  streamcat_pressures.csv  EPA StreamCat metrics for each station's whole upstream
                           watershed (WS): densities of NPDES sites, wastewater plants
                           (all, major), TRI and Superfund sites, mines, coal mines,
                           dams and canals; population, housing, roads, septic
                           systems; impervious, urban, crop, hay and barren land;
                           fertilizer, manure, anthropogenic and wastewater nitrogen;
                           irrigation water input; area burned 2014-2018; and the
                           water-chemistry component of EPA's Index of Watershed
                           Integrity (1 = no human pressure on water chemistry).

The requests carry only HUC codes, a bounding box and NHDPlus reach ids: no
water-quality value and nothing about this project.
"""
import io
import json
import pickle
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/pollution")
UA = {"User-Agent": "river-water-quality-research-script"}

ECHO = "https://echodata.epa.gov/echo/cwa_rest_services."
# ECHO column ids (cwa_rest_services.metadata)
ECHO_COLS = [1, 2, 4, 5, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 53, 56, 62, 63, 103, 156, 161, 163, 175]

OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]
BOX = "30.5,-116.5,43.6,-104.8"   # south, west, north, east: the Colorado basin and surroundings
OSM_QUERY = (f'[out:json][timeout:150];(nwr["telecom"="data_center"]({BOX});'
             f'nwr["building"="data_center"]({BOX});nwr["building"="data_centre"]({BOX}););'
             "out center tags;")

NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite/{}"
DOWNSTREAM = "https://api.water.usgs.gov/nldi/linked-data/comid/{}/navigation/DM/flowlines"
STREAMCAT = "https://api.epa.gov/StreamCat/streams/metrics"
SC_METRICS = ["npdesdens", "wwtpalldens", "wwtpmajordens", "tridens", "superfunddens", "minedens",
              "coalminedens", "damdens", "canaldens", "popden2010", "huden2010", "rddens", "septic",
              "pctimp2019", "pcturbhi2019", "pcturbmd2019", "pcturblo2019", "pcturbop2019",
              "pctcrop2019", "pcthay2019", "pctbl2019", "fert", "manure", "nani", "n_usgsww_2012",
              "waterinput", "chem", *[f"pctburnarea{y}" for y in range(2014, 2019)]]


def get(url, tries=4, **kw):
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=kw.pop("timeout", 300), **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if k == tries - 1:
                raise
            print(f"  retry {url[:60]}... ({e})")
            time.sleep(10 * (k + 1))


def fetch_echo():
    parts = []
    for huc in ("14", "15"):
        res = get(ECHO + "get_facilities", params={"output": "JSON", "p_huc": huc, "p_act": "Y"}).json()["Results"]
        csv = get(ECHO + "get_download", params={"output": "CSV", "qid": res["QueryID"],
                                                 "qcolumns": ",".join(map(str, ECHO_COLS))}).content
        d = pd.read_csv(io.BytesIO(csv), dtype=str, encoding="latin-1")
        print(f"ECHO HUC {huc}: {len(d):,} active permits (query said {res['QueryRows']})")
        parts.append(d.assign(query_huc2=huc))
    e = pd.concat(parts, ignore_index=True).drop_duplicates("SourceID")
    e.to_csv(OUT / "echo_npdes.csv", index=False)
    return e


def fetch_osm():
    for k in range(6):
        url = OVERPASS[k % len(OVERPASS)]
        try:
            r = requests.post(url, data={"data": OSM_QUERY}, headers=UA, timeout=240)
            elements = r.json()["elements"]
            break
        except (requests.RequestException, ValueError, KeyError):
            print(f"  Overpass busy at {url}, retrying")
            time.sleep(20)
    else:
        raise RuntimeError("Overpass API unavailable; try again later")
    rows = []
    for el in elements:
        t = el.get("tags", {})
        c = el.get("center", {})
        rows.append({"osm_id": f"{el['type']}/{el['id']}", "lat": el.get("lat", c.get("lat")),
                     "lon": el.get("lon", c.get("lon")), "name": t.get("name"), "operator": t.get("operator"),
                     "city": t.get("addr:city"), "tag": "telecom" if t.get("telecom") == "data_center"
                     else "building"})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "osm_data_centers.csv", index=False)
    print(f"OpenStreetMap: {len(d)} data-centre features")
    return d


def streamcat(comids):
    r = get(STREAMCAT, params={"name": ",".join(SC_METRICS), "aoi": "ws",
                               "comid": ",".join(str(int(c)) for c in comids)}, timeout=180)
    return pd.DataFrame(r.json()["items"])


def fetch_streamcat():
    with open("graphs/colorado/master_compressed_river_graph.pkl", "rb") as f:
        G = pickle.load(f)
    rows = []
    for node, data in G.nodes(data=True):
        for cl in data.get("usgs_clusters") or []:
            for st in cl.get("member_stations") or []:
                r = requests.get(NLDI.format(st), headers=UA, timeout=60)
                comid = int(r.json()["features"][0]["properties"]["comid"]) if r.status_code == 200 else None
                rows.append({"node": int(node), "station": st, "comid": comid})
                time.sleep(0.2)
    st = pd.DataFrame(rows)
    sc = streamcat(st["comid"].dropna().unique())
    st["streamcat_comid"] = st["comid"]
    for i in st.index[~st["comid"].isin(sc["comid"]) & st["comid"].notna()]:
        r = get(DOWNSTREAM.format(int(st.at[i, "comid"])), params={"distance": 10}, timeout=60)
        for c in [f["properties"]["nhdplus_comid"] for f in r.json()["features"]][1:]:
            found = streamcat([c])
            if len(found):
                st.at[i, "streamcat_comid"] = int(c)
                sc = pd.concat([sc, found])
                break
    st.to_csv(OUT / "station_comids.csv", index=False)
    out = st.merge(sc.rename(columns={"comid": "streamcat_comid"}), on="streamcat_comid", how="left")
    out.to_csv(OUT / "streamcat_pressures.csv", index=False)
    print(f"StreamCat: {out.shape[1] - 4} metrics for {out['station'].nunique()} stations "
          f"({out.iloc[:, 4:].notna().all(axis=1).sum()} complete)")
    return out


def fetch_station_names():
    st = pd.read_csv(OUT / "station_comids.csv")
    sites = ",".join(sorted({s.split("-")[1] for s in st["station"]}))
    txt = get("https://waterservices.usgs.gov/nwis/site/", params={"format": "rdb", "sites": sites,
                                                                   "siteOutput": "basic"}).text
    lines = [l for l in txt.splitlines() if not l.startswith("#")]
    d = pd.read_csv(io.StringIO("\n".join([lines[0], *lines[2:]])), sep="\t", dtype=str)
    d = d.rename(columns={"site_no": "site", "station_nm": "name"})[["site", "name"]]
    d.to_csv(OUT / "station_names.csv", index=False)
    print(f"USGS names for {len(d)} stations")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fetch_echo()
    fetch_osm()
    fetch_streamcat()
    fetch_station_names()
    json.dump({"downloaded": date.today().isoformat(), "echo_huc2": ["14", "15"], "osm_box": BOX,
               "streamcat_metrics": SC_METRICS}, open(OUT / "sources_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
