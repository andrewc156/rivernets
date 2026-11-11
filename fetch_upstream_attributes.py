"""Upstream-watershed attributes of the WQ nodes (BENCHMARK_FS_STATIC=...,upstream).

usage: python fetch_upstream_attributes.py      (about a minute, public APIs)

For every USGS station of a WQ node: its NHDPlus reach (COMID) from the USGS
NLDI, then three StreamCat watershed metrics (accumulated over the whole
upstream watershed), chosen before any run for specific conductance:
    ws_sulfur   mean lithological sulfur, %        (S)    gypsum, evaporites
    ws_cao      mean lithological calcium oxide, % (CaO)  carbonates, gypsum
    ws_pctcrop  row-crop land, % (NLCD 2019)              irrigation return flow
A node with several stations gets the mean over its stations. If StreamCat has
no values for a station's reach, the first reach downstream (within 10 km) that
has them is used (only USGS-09217000, whose reach is missing).

writes data/benchmark_data/upstream_attributes.csv
"""
import pickle
import time

import pandas as pd
import requests

NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite/{}"
STREAMCAT = "https://api.epa.gov/StreamCat/streams/metrics"
METRICS = {"sws": "ws_sulfur", "caows": "ws_cao", "pctcrop2019ws": "ws_pctcrop"}
DOWNSTREAM = "https://api.water.usgs.gov/nldi/linked-data/comid/{}/navigation/DM/flowlines"


def streamcat(comids):
    r = requests.get(STREAMCAT, params={"name": "s,cao,pctcrop2019", "aoi": "ws",
                                        "comid": ",".join(str(int(c)) for c in comids)}, timeout=120)
    r.raise_for_status()
    return pd.DataFrame(r.json()["items"], columns=["comid", *METRICS]).rename(columns=METRICS)


def main():
    with open("graphs/colorado/master_compressed_river_graph.pkl", "rb") as f:
        G = pickle.load(f)
    rows = []
    for node, data in G.nodes(data=True):
        for cluster in data.get("usgs_clusters", []):
            for station in cluster.get("member_stations", []) or (
                [cluster["anchor_station"]] if cluster.get("anchor_station") is not None else []
            ):
                r = requests.get(NLDI.format(station), timeout=60)
                comid = r.json()["features"][0]["properties"]["comid"] if r.status_code == 200 else None
                rows.append({"node": int(node), "station": station, "comid": comid})
                time.sleep(0.2)
    st = pd.DataFrame(rows)
    sc = streamcat(st["comid"].dropna().unique())
    st["streamcat_comid"] = st["comid"]
    for i in st.index[~st["comid"].isin(sc["comid"]) & st["comid"].notna()]:
        r = requests.get(DOWNSTREAM.format(int(st.at[i, "comid"])), params={"distance": 10}, timeout=60)
        for c in [f["properties"]["nhdplus_comid"] for f in r.json()["features"]][1:]:
            found = streamcat([c])
            if len(found):
                st.at[i, "streamcat_comid"] = int(c)
                sc = pd.concat([sc, found])
                break
    st = st.merge(sc.rename(columns={"comid": "streamcat_comid"}), on="streamcat_comid", how="left")
    print(st.to_string(index=False))
    missing = st[st[list(METRICS.values())].isna().any(axis=1)]
    if len(missing):
        print("\nstations without attributes (left out of their node's mean):")
        print(missing[["node", "station", "comid"]].to_string(index=False))
    node = st.groupby("node", as_index=False)[list(METRICS.values())].mean()
    node["stations"] = st.groupby("node")["station"].apply(";".join).to_numpy()
    node.to_csv("data/benchmark_data/upstream_attributes.csv", index=False)
    print(f"\n{node[list(METRICS.values())].notna().all(axis=1).sum()} of {len(node)} nodes complete")


if __name__ == "__main__":
    main()
