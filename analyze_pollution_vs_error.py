"""Do the stations where the benchmark does worst have more pollution sources upstream?

usage: python analyze_pollution_vs_error.py
       (seconds; after fetch_pollution_sources.py and build_pollution_sources.py)

Joins each station's error in the main benchmark (graph model:
results/benchmark_results_inner_fs_graph; Ridge and gradient boosting:
results/benchmark_results_inner_fs) with what lies upstream of it:
  - traced sources (data/pollution/station_sources.csv): EPA ECHO discharge permits by
    category and OpenStreetMap data centres whose flow path passes the station
    (build_pollution_sources.py), upstream and "local" (this station is the first one
    below them);
  - EPA StreamCat watershed metrics (data/pollution/streamcat_pressures.csv);
  - the share of the river's mean flow that is municipal wastewater: design flows of the
    traced wastewater plants (MGD x 1.547 = cfs) over the station's
    mean discharge in data/usgs/usgs_all_stations_hourly.csv (22 of the 26 nodes have
    discharge; design flows overstate actual discharges, and reservoirs and diversions
    make this an indicator, not a mixing calculation).

Per station and target (equal station weights, as in the benchmark):
  share       the station's share of the model's total squared error (shares add to 1;
              R2 = 1 - mean station MSE / pooled variance)
  bias        mean of prediction - observation (positive: the model predicts too high)
  level_frac  bias^2 / MSE, the part of the station's error that is a constant offset
  r2_station  1 - MSE / the station's own variance

Spearman rank correlations between each pressure metric and the graph model's error
share, across the stations of each target, with permutation p-values (10,000
permutations), Benjamini-Hochberg false-discovery rates within each target, and a
partial correlation that removes the drainage area (both ranks regressed on log area).
With 17-24 stations only |rho| above about 0.45-0.5 differs from zero at p = 0.05, and
about 1 in 20 metrics will pass that by chance, so the table is exploratory.

writes results/pollution/station_errors.csv          per target, model and station
       results/pollution/station_pressures.csv       per node: the metrics used
       results/pollution/pressure_error_spearman.csv per target and metric
       results/pollution/top_error_stations.csv      stations carrying half of each target's error
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/pollution")
GRAPH_RUN = Path("results/benchmark_results_inner_fs_graph")
CONV_RUN = Path("results/benchmark_results_inner_fs")
TARGETS = ["Water temperature", "DO concentration", "Specific conductance", "pH", "Turbidity"]
N_PERM = 10_000

# metric -> (theme, label); traced metrics come from station_sources.csv, the rest from StreamCat
METRICS = {
    "permits_per_1000km2": ("Wastewater and urban", "Discharge permits per 1000 km2 (traced)"),
    "wwtp_mgd": ("Wastewater and urban", "Municipal wastewater design flow, MGD (traced)"),
    "wwtpalldensws": ("Wastewater and urban", "Wastewater plants per km2 (StreamCat)"),
    "n_usgsww_2012ws": ("Wastewater and urban", "Wastewater nitrogen, kg/km2 (StreamCat)"),
    "popden2010ws": ("Wastewater and urban", "Population per km2 (2010)"),
    "pctimp2019ws": ("Wastewater and urban", "Impervious surface, %"),
    "pcturb": ("Wastewater and urban", "Urban land, %"),
    "septicws": ("Wastewater and urban", "Septic systems per km2"),
    "ww_share_pct": ("Wastewater and urban", "Wastewater share of river flow, %"),
    "n_manuf": ("Industry and toxics", "Factories with permits (traced)"),
    "n_indsw": ("Industry and toxics", "Industrial-stormwater permits (traced)"),
    "tridensws": ("Industry and toxics", "Toxic Release Inventory sites per km2"),
    "superfunddensws": ("Industry and toxics", "Superfund sites per km2"),
    "n_coal": ("Mining and energy", "Coal mines with permits (traced)"),
    "n_metal": ("Mining and energy", "Metal and uranium mines with permits (traced)"),
    "n_oilgas": ("Mining and energy", "Oil and gas permits (traced)"),
    "n_power": ("Mining and energy", "Power plants with permits (traced)"),
    "minedensws": ("Mining and energy", "Mines per km2 (StreamCat, all)"),
    "pctcrop2019ws": ("Agriculture and irrigation", "Cropland, %"),
    "pcthay2019ws": ("Agriculture and irrigation", "Hay and pasture, %"),
    "canaldensws": ("Agriculture and irrigation", "Canals and ditches, km/km2"),
    "waterinputws": ("Agriculture and irrigation", "Irrigated land / precipitation"),
    "fertws": ("Agriculture and irrigation", "Fertilizer N, kg/ha/yr"),
    "manurews": ("Agriculture and irrigation", "Manure N, kg/ha/yr"),
    "pctbl2019ws": ("Land disturbance", "Barren land, %"),
    "burn_2014_2018": ("Land disturbance", "Area burned 2014-2018, %"),
    "rddensws": ("Land disturbance", "Road density, km/km2"),
    "damdensws": ("Land disturbance", "Dams per km2"),
    "chem_pressure": ("Overall", "Water-chemistry pressure (1 - IWI chem)"),
    "n_permits_local": ("Between this station and the next one up", "Discharge permits, local (traced)"),
    "wwtp_mgd_local": ("Between this station and the next one up", "Wastewater design flow, local, MGD"),
    "ww_share_local_pct": ("Between this station and the next one up", "Wastewater share of flow, local, %"),
    "n_datacentre": ("Data centres", "Data-centre campuses upstream (OSM)"),
}


def station_errors(pred, model, target):
    p = pred[(pred["model"] == model) & (pred["target"] == target)]
    g = p.assign(err=p["y_pred"] - p["y_true"], y2=p["y_true"] ** 2).groupby("node")
    s = pd.DataFrame({"rows": g.size(), "mean_y": g["y_true"].mean(), "var_y": g["y_true"].var(ddof=0),
                      "mse": g["err"].apply(lambda e: np.mean(e ** 2)), "bias": g["err"].mean(),
                      "my2": g["y2"].mean()})
    V = s["my2"].mean() - s["mean_y"].mean() ** 2
    s["share"] = s["mse"] / s["mse"].sum()
    s["level_frac"] = s["bias"] ** 2 / s["mse"]
    s["r2_station"] = 1 - s["mse"] / s["var_y"]
    s["r2_model"] = 1 - s["mse"].mean() / V
    return s.drop(columns="my2").reset_index().assign(model=model, target=target)


def ranks(x):
    return pd.Series(x).rank().to_numpy(float)


def spearman_perm(x, y, rng):
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    rho = (rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry))
    perm = np.array([rx @ rng.permutation(ry) for _ in range(N_PERM)]) / np.sqrt((rx @ rx) * (ry @ ry))
    return rho, (np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (N_PERM + 1)


def partial_spearman(x, y, z):
    rx, ry, rz = ranks(x), ranks(y), ranks(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    a, b = res(rx), res(ry)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def pressures():
    ss = pd.read_csv("data/pollution/station_sources.csv")
    sc = pd.read_csv("data/pollution/streamcat_pressures.csv")
    sc["pcturb"] = sc[[f"pcturb{k}2019ws" for k in ("hi", "md", "lo", "op")]].sum(axis=1)
    sc["burn_2014_2018"] = sc[[f"pctburnarea{y}ws" for y in range(2014, 2019)]].sum(axis=1)
    sc["chem_pressure"] = 1 - sc["chemws"]
    scn = sc.groupby("node").mean(numeric_only=True).drop(columns=["comid", "streamcat_comid"])
    area = pd.read_pickle("data/benchmark_data/static_node_df.pkl").set_index("node")["upa_km2"]
    P = ss.set_index("node").join(scn)
    P["upa_km2"] = area.reindex(P.index).to_numpy()
    P["permits_per_1000km2"] = P["n_permits"] / P["upa_km2"] * 1000
    P["wwtp_mgd"] = P["wwtp_design_mgd"]
    P["wwtp_mgd_local"] = P["wwtp_design_mgd_local"]
    # mean discharge per node (median over the node's stations per hour, as in the benchmark)
    import pickle
    G = pickle.load(open("graphs/colorado/master_compressed_river_graph.pkl", "rb"))
    s2n = {st: n for n, d in G.nodes(data=True) for c in d.get("usgs_clusters") or []
           for st in (c.get("member_stations") or [])}
    q = pd.read_csv("data/usgs/usgs_all_stations_hourly.csv", usecols=["station_id", "time", "Discharge"])
    q["node"] = q["station_id"].map(s2n)
    q = q[q["node"].notna() & (q["Discharge"] > 0)]
    qm = q.groupby(["node", "time"])["Discharge"].median().groupby("node").mean()
    P["q_mean_cfs"] = qm.reindex(P.index).to_numpy()
    CFS = 1.547   # cubic feet per second in one million gallons per day
    P["ww_share_pct"] = 100 * P["wwtp_design_mgd"] * CFS / P["q_mean_cfs"]
    P["ww_share_local_pct"] = 100 * P["wwtp_design_mgd_local"] * CFS / P["q_mean_cfs"]
    return P


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = pd.read_csv("data/pollution/station_names.csv", dtype=str).set_index("site")["name"]
    P = pressures()
    P["name"] = [names.get(s.split(",")[0].replace("USGS-", ""), s) for s in P["stations"]]
    P.to_csv(OUT / "station_pressures.csv")

    pg = pd.read_pickle(GRAPH_RUN / "benchmark_predictions.pkl")
    pc = pd.read_pickle(CONV_RUN / "benchmark_predictions.pkl")
    E = pd.concat([station_errors(pg, "Graph-Laplacian", t) for t in TARGETS]
                  + [station_errors(pc, m, t) for t in TARGETS for m in ("Ridge", "Gradient Boosting")])
    E["name"] = E["node"].map(P["name"])
    E.to_csv(OUT / "station_errors.csv", index=False)

    rng = np.random.default_rng(0)
    rows = []
    for t in TARGETS:
        e = E[(E["model"] == "Graph-Laplacian") & (E["target"] == t)].set_index("node")
        r = E[(E["model"] == "Ridge") & (E["target"] == t)].set_index("node")
        x = P.loc[e.index]
        rel = np.log2(e["mse"] / r["mse"].reindex(e.index))      # > 0: graph worse than Ridge
        for m, (theme, label) in METRICS.items():
            ok = x[m].notna().to_numpy()
            v = x[m].to_numpy(float)[ok]
            if ok.sum() < 8 or np.nanstd(v) == 0:
                rows.append(dict(target=t, metric=m, theme=theme, label=label, n=len(v)))
                continue
            rho, p = spearman_perm(v, e["share"].to_numpy()[ok], rng)
            rho_b, p_b = spearman_perm(v, np.abs(e["bias"]).to_numpy()[ok], rng)
            rho_rel, p_rel = spearman_perm(v, rel.to_numpy()[ok], rng)
            rows.append(dict(target=t, metric=m, theme=theme, label=label, n=len(v), rho=rho, p=p,
                             rho_partial_area=partial_spearman(v, e["share"].to_numpy()[ok],
                                                               x["upa_km2"].to_numpy()[ok]),
                             rho_absbias=rho_b, p_absbias=p_b, rho_graph_vs_ridge=rho_rel, p_graph_vs_ridge=p_rel))
    R = pd.DataFrame(rows)
    R["q"] = R["q_graph_vs_ridge"] = np.nan
    for t in TARGETS:
        i = R.index[(R["target"] == t) & R["p"].notna()]
        R.loc[i, "q"] = bh(R.loc[i, "p"])
        R.loc[i, "q_graph_vs_ridge"] = bh(R.loc[i, "p_graph_vs_ridge"])
    R.to_csv(OUT / "pressure_error_spearman.csv", index=False)

    top = []
    for t in TARGETS:
        e = E[(E["model"] == "Graph-Laplacian") & (E["target"] == t)].sort_values("share", ascending=False)
        k = int(np.searchsorted(e["share"].cumsum().to_numpy(), 0.5) + 1)
        top.append(e.head(k))
    T = pd.concat(top)
    T.to_csv(OUT / "top_error_stations.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_colwidth", 48)
    print(T[["target", "name", "rows", "share", "bias", "level_frac", "r2_station"]].round(3).to_string(index=False))
    show = R[R["p"] < 0.05].sort_values(["target", "p"])
    print(f"\n{len(show)} of {R['p'].notna().sum()} metric-target pairs with p < 0.05 "
          f"(about {0.05 * R['p'].notna().sum():.0f} expected by chance); smallest q per target:")
    print(R.groupby("target")["q"].min().round(3).to_string())
    print(show[["target", "label", "n", "rho", "p", "q", "rho_partial_area", "rho_absbias", "rho_graph_vs_ridge"]]
          .round(3).to_string(index=False))
    print("\nflow shares:")
    print(R[R["metric"].str.contains("share")][["target", "label", "n", "rho", "p", "q", "rho_graph_vs_ridge",
                                                 "p_graph_vs_ridge"]].round(3).to_string(index=False))
    gv = R[R["p_graph_vs_ridge"] < 0.05].sort_values(["target", "p_graph_vs_ridge"])
    print("\ngraph worse than Ridge where (p < 0.05):")
    print(gv[["target", "label", "rho_graph_vs_ridge", "p_graph_vs_ridge", "q_graph_vs_ridge"]].round(3)
          .to_string(index=False))
    print(P[["name", "q_mean_cfs", "wwtp_design_mgd", "ww_share_pct", "ww_share_local_pct"]]
          .round(2).sort_values("ww_share_pct", ascending=False).to_string())


if __name__ == "__main__":
    main()
