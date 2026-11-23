"""Which possible pollution sources drain past which water-quality station.

usage: python build_pollution_sources.py     (a minute or two; run fetch_pollution_sources.py first)

Every discharge permit (EPA ECHO, data/pollution/echo_npdes.csv) and every mapped
data centre (OpenStreetMap, data/pollution/osm_data_centers.csv) is followed
downstream from its own 3-arcsecond MERIT Hydro cell along the D8 flow
directions (data/colorado/merit_colorado_dir_native.tif, ESRI codes, as in
build_tailwater_features.py) until the path reaches a cell of the compressed
river graph, and then along the graph to the outlet. Every water-quality node
passed on the way lies downstream of the source, at the river distance
travelled (the D8 path plus the graph). A source whose path never reaches the
graph drains away from every station (another basin, a closed depression, or a
river joining below the lowest station, such as the Gila and Phoenix).
Facility coordinates are the permit's address or centroid, not the outfall.

Source categories, from the permit's facility type, components and SIC codes
(first match):
  Municipal wastewater      POTW, or SIC 4952 (sewerage systems)
  Power plant               SIC 4911, 4931, 4939
  Coal mine                 SIC 1221-1241
  Metal or uranium mine     SIC 1000-1099
  Other mining              SIC 1400-1499 (sand and gravel, potash, ...)
  Oil and gas               SIC 1311-1389, 2911, 4922-4925
  Manufacturing             SIC 2000-3999 except 2911 (factories)
  Fish hatchery             SIC 0921
  Drinking-water plant      SIC 4941
  Industrial stormwater     any other permit whose components include industrial stormwater
  Other individual permit   other individual permits (hotels, campgrounds, ski areas, ...)
  Other general permit      general-permit coverage with none of the above: mostly
                            construction sites, dewatering and hydrostatic tests that
                            ECHO does not label as construction stormwater; counted
                            separately and left out of the permit totals
Permits labelled construction stormwater (temporary building sites) are left out.
Records with the same name and location (to 0.001 degree) and category are counted
once. Data centres are grouped into campuses (buildings within about 1 km, same
operator).

writes data/pollution/sources_traced.csv   one row per source: category, name, location,
                                           first water-quality node below it and km
       data/pollution/source_links.csv     one row per source and downstream WQ node, with km
       data/pollution/station_sources.csv  one row per WQ node: for each category the number
                                           of sources upstream (any distance), within 100 km,
                                           and "local" (this is the first station below them,
                                           i.e. between it and the stations above), municipal
                                           design flow upstream / local (MGD), TRI surface-water
                                           releases upstream (lb/yr), major permits, data-centre
                                           campuses upstream and the nearest one (km)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from build_tailwater_features import BlockReader, COLS, CELL_KM, D8, DIR_TIF, GRAPH

SRC = Path("data/pollution")
NEAR_KM = 100.0
MAX_STEPS = 200_000

CATS = ["Municipal wastewater", "Power plant", "Coal mine", "Metal or uranium mine", "Other mining",
        "Oil and gas", "Manufacturing", "Fish hatchery", "Drinking-water plant", "Industrial stormwater",
        "Other individual permit", "Other general permit", "Data centre"]
SHORT = {"Municipal wastewater": "wwtp", "Power plant": "power", "Coal mine": "coal",
         "Metal or uranium mine": "metal", "Other mining": "minother", "Oil and gas": "oilgas",
         "Manufacturing": "manuf", "Fish hatchery": "hatchery", "Drinking-water plant": "dwtp",
         "Industrial stormwater": "indsw", "Other individual permit": "otherind",
         "Other general permit": "othergp", "Data centre": "datacentre"}
NOT_PERMIT_TOTAL = ("Other general permit", "Data centre")


def sic_list(s):
    return [int(x) for x in str(s).replace(" ", "").split(",") if x.isdigit()] if isinstance(s, str) else []


def category(row):
    comp = str(row["PermitComponents"]) if isinstance(row["PermitComponents"], str) else ""
    sics = sic_list(row["CWPSICCodes"])
    if row["CWPFacilityTypeIndicator"] == "POTW" or "POTW" in comp or 4952 in sics:
        return "Municipal wastewater"
    for s in sics:
        if s in (4911, 4931, 4939):
            return "Power plant"
    for s in sics:
        if 1221 <= s <= 1241:
            return "Coal mine"
        if 1000 <= s <= 1099:
            return "Metal or uranium mine"
        if 1400 <= s <= 1499:
            return "Other mining"
        if 1311 <= s <= 1389 or s == 2911 or 4922 <= s <= 4925:
            return "Oil and gas"
    for s in sics:
        if 2000 <= s <= 3999:
            return "Manufacturing"
        if s == 921:
            return "Fish hatchery"
        if s == 4941:
            return "Drinking-water plant"
    if "Industrial Stormwater" in comp:
        return "Industrial stormwater"
    if row["CWPPermitTypeDesc"] != "General Permit Covered Facility":
        return "Other individual permit"
    return "Other general permit"


def load_sources():
    e = pd.read_csv(SRC / "echo_npdes.csv", dtype=str)
    comp = e["PermitComponents"].fillna("")
    e = e[comp != "Construction Stormwater"].copy()
    e["category"] = e.apply(category, axis=1)
    # one record per facility: same name, location and category (individual permits first)
    e["prio"] = e["CWPPermitTypeDesc"].map({"NPDES Individual Permit": 0}).fillna(1)
    e["key"] = (e["CWPName"].str.upper().str.strip() + "|" + e["category"] + "|"
                + pd.to_numeric(e["FacLat"], errors="coerce").round(3).astype(str) + "|"
                + pd.to_numeric(e["FacLong"], errors="coerce").round(3).astype(str))
    e = e.sort_values(["prio"]).drop_duplicates("key").sort_index()
    num = lambda c: pd.to_numeric(e[c], errors="coerce")
    src = pd.DataFrame({
        "source_id": e["SourceID"], "category": e["category"], "name": e["CWPName"],
        "city": e["CWPCity"], "state": e["CWPState"], "lat": num("FacLat"), "lon": num("FacLong"),
        "major": e["CWPMajorMinorStatusFlag"].eq("M"), "permit_type": e["CWPPermitTypeDesc"],
        "sic": e["CWPSICDescs"], "design_mgd": num("CWPTotalDesignFlowNmbr"),
        "actual_mgd": num("CWPActualAverageFlowNmbr"), "tri_sw_lb": num("FacTRISwRel"),
        "receiving_water": e["CWPStateWaterBodyName"],
    })
    dc = pd.read_csv(SRC / "osm_data_centers.csv").dropna(subset=["lat", "lon"])
    # campuses: same operator (or name) and within ~1 km
    dc["key"] = dc["operator"].fillna(dc["name"]).fillna("unnamed")
    dc["cell"] = (dc["lat"] / 0.01).round().astype(int).astype(str) + "_" + (dc["lon"] / 0.01).round().astype(int).astype(str)
    camp = (dc.groupby(["key", "cell"])
              .agg(lat=("lat", "mean"), lon=("lon", "mean"), name=("name", lambda s: "; ".join(sorted({x for x in s if isinstance(x, str)}))[:120]),
                   city=("city", "first"), buildings=("osm_id", "size"))
              .reset_index())
    dcs = pd.DataFrame({"source_id": "OSM-" + camp.index.astype(str), "category": "Data centre",
                        "name": camp["name"].where(camp["name"] != "", camp["key"]), "city": camp["city"],
                        "state": "", "lat": camp["lat"], "lon": camp["lon"], "major": False,
                        "permit_type": "OpenStreetMap", "sic": "", "design_mgd": np.nan, "actual_mgd": np.nan,
                        "tri_sw_lb": np.nan, "receiving_water": "", "buildings": camp["buildings"]})
    out = pd.concat([src, dcs], ignore_index=True)
    return out.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def main():
    G = pickle.load(open(GRAPH, "rb"))
    a = next(iter(G.nodes(data=True)))[1]
    lat0, lon0 = a["latitude"] + a["row"] / 1200.0, a["longitude"] - a["col"] / 1200.0
    stations = {n: [s for c in d.get("usgs_clusters") or [] for s in c["member_stations"]]
                for n, d in G.nodes(data=True)}
    wq_nodes = {n for n, s in stations.items() if s}

    # graph cell -> (downstream node v, remaining metres to v); node cells map to themselves
    join = {}
    for u, v, e in G.edges(data=True):
        p = e["original_path"]
        for k, cell in enumerate(p):
            rem = e["length_m"] * (1 - k / len(p))
            if int(cell) not in join or rem < join[int(cell)][1]:
                join[int(cell)] = (v, rem)
    for n in G.nodes:
        join[int(n)] = (n, 0.0)

    def walk_down(node, d_km):
        out = {}
        while True:
            if node in wq_nodes:
                out[node] = d_km
            nxt = list(G.successors(node))
            if not nxt:
                return out
            d_km += G.edges[node, nxt[0]]["length_m"] / 1000.0
            node = nxt[0]

    dirs = BlockReader(DIR_TIF)
    memo = {}   # cell -> (graph cell reached or None, km from this cell to it)

    def trace(r, c):
        path, km = [], 0.0
        for _ in range(MAX_STEPS):
            cell = r * COLS + c
            if cell in memo:
                hit, rest = memo[cell]
                break
            if cell in join:
                hit, rest = cell, 0.0
                break
            code = dirs(r, c)
            if code is None or int(code) not in D8:
                hit, rest = None, 0.0
                break
            path.append((cell, km))
            dr, dc = D8[int(code)]
            km += CELL_KM * np.hypot(dr, dc * np.cos(np.radians(lat0 - r / 1200.0)))
            r, c = r + dr, c + dc
        else:
            hit, rest = None, 0.0
        for cell, k in path:
            memo[cell] = (hit, None if hit is None else km - k + rest)
        start = path[0][0] if path else r * COLS + c
        return (hit, 0.0 if not path else memo[start][1]) if path else (hit, rest)

    src = load_sources()
    links, first = [], []
    for i, s in src.iterrows():
        r = int(round((lat0 - s["lat"]) * 1200)); c = int(round((s["lon"] - lon0) * 1200))
        hit, km = trace(r, c)
        reached = {}
        if hit is not None:
            v, rem = join[hit]
            reached = walk_down(v, km + rem / 1000.0)
        for n, d in reached.items():
            links.append((s["source_id"], n, round(d, 1)))
        if reached:
            n0 = min(reached, key=reached.get)
            first.append((n0, round(reached[n0], 1), len(reached)))
        else:
            first.append((None, np.nan, 0))
    src["first_node"] = [f[0] for f in first]
    src["first_station"] = [",".join(stations[f[0]]) if f[0] is not None else "" for f in first]
    src["first_km"] = [f[1] for f in first]
    src["n_downstream_nodes"] = [f[2] for f in first]
    src.to_csv(SRC / "sources_traced.csv", index=False)
    L = pd.DataFrame(links, columns=["source_id", "node", "river_km"]).merge(
        src[["source_id", "category", "major", "design_mgd", "tri_sw_lb", "first_node"]], on="source_id")
    L["local"] = L["node"] == L["first_node"]
    L.to_csv(SRC / "source_links.csv", index=False)

    rows = []
    for n in sorted(wq_nodes):
        g = L[L["node"] == n]
        row = {"node": n, "stations": ",".join(stations[n])}
        for cat in CATS:
            k = SHORT[cat]
            gc = g[g["category"] == cat]
            row[f"n_{k}"] = len(gc)
            row[f"n_{k}_100km"] = int((gc["river_km"] <= NEAR_KM).sum())
            row[f"n_{k}_local"] = int(gc["local"].sum())
        perm = g[~g["category"].isin(NOT_PERMIT_TOTAL)]
        row["n_permits"] = len(perm)
        row["n_permits_local"] = int(perm["local"].sum())
        row["n_major"] = int(perm["major"].sum())
        row["n_major_local"] = int((perm["major"] & perm["local"]).sum())
        w = g[g["category"] == "Municipal wastewater"]
        row["wwtp_design_mgd"] = round(w["design_mgd"].sum(), 2)
        row["wwtp_design_mgd_local"] = round(w.loc[w["local"], "design_mgd"].sum(), 2)
        row["tri_sw_lb"] = round(perm["tri_sw_lb"].sum(), 1)
        d = g[g["category"] == "Data centre"]
        row["datacentre_nearest_km"] = d["river_km"].min() if len(d) else np.nan
        rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv(SRC / "station_sources.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    perm = ~src["category"].isin(NOT_PERMIT_TOTAL)
    print(f"{perm.sum():,} permitted dischargers, {(src['category'] == 'Other general permit').sum():,} other "
          f"general permits, {(src['category'] == 'Data centre').sum()} data-centre campuses; of the "
          f"dischargers {src.loc[perm, 'first_node'].notna().sum():,} drain past at least one station")
    print(src.groupby("category").agg(sources=("source_id", "size"),
                                      upstream_of_a_station=("first_node", lambda x: x.notna().sum()))
          .reindex(CATS).to_string())
    print(S[["stations", "n_permits", "n_permits_local", "n_major", "n_wwtp", "wwtp_design_mgd",
             "n_coal", "n_metal", "n_oilgas", "n_power", "n_manuf", "n_datacentre", "datacentre_nearest_km"]]
          .to_string(index=False))
    print(f"memoised cells: {len(memo):,}")


if __name__ == "__main__":
    main()
