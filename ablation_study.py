"""
Ablation and reverse ablation of the graph-Laplacian water-quality model
========================================================================

For each target, every predictor group AND the graph term are treated as
"variables":

    chemistry       each other WQ variable used as a predictor for the target
                    (fixed per target, as in the benchmark, so rows never change)
    daily cycle     hour_sin_k, hour_cos_k, k = 1..K
    annual cycle    doy_sin_k,  doy_cos_k,  k = 1..K
    elevation_m, log_upa, log_width, local_slope     (static MERIT, one each)
    graph term      the river-network effect g, smoothed by a Kron Laplacian

Ablation          full model (all variables)   minus one variable at a time
Reverse ablation  base model (intercept only)  plus  one variable at a time

Every configuration is scored by its own nested leave-one-node-out run, on the
benchmark's rows (target + its chemistry predictors present). Inside each outer
fold the inner LOO tunes the ridge penalty lambda_beta and, when the graph term
is on, the Laplacian weighting and lambda, using the six weightings and the
extended lambda grid of water_quality_model_benchmark.py. Without the graph term
the model is ridge regression on the same features (a common intercept).

Model (graph term on):
    y_it = b0 + z_it' b + g_i
    minimise  sum_i (1/n_i) sum_t (y_it - b0 - z_it' b - g_i)^2
              + lambda_beta |b|^2 + lambda g' L g,   sum_i g_i = 0

R2 / RMSE are node-weighted (each station counts equally), as in the benchmark.
hand_m is not used (it is 0 at every WQ station). The water-temperature
interaction terms of the benchmark are not used.

Run from this folder:

    python ablation_study.py                      # all targets
    python ablation_study.py --targets pH --jobs 4
    python ablation_study.py --targets pH --max-outer-folds 2   # smoke test

Outputs (results/ablation/):
    ablation_results.csv      one row per (target, study, variable)
    ablation_tuning.csv       chosen settings per outer fold
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = BASE_DIR / "data/benchmark_data"
MASTER_GRAPH_FILE = BASE_DIR / "graphs/colorado/master_compressed_river_graph.pkl"
OUT_DIR = BASE_DIR / "results/ablation"

TARGET_CHEMISTRY = {
    "Water temperature": ["Specific conductance"],
    "DO concentration": ["Water temperature", "Specific conductance"],
    "Specific conductance": ["Water temperature"],
    "pH": ["Specific conductance"],
    "Turbidity": [],
}
STATIC = ["elevation_m", "log_upa", "log_width", "local_slope"]
GRAPH = "graph term"

EXTRA_LAMBDAS = [0.0001, 0.001, 0.003]

# ---------------------------------------------------------------------------
# data and Laplacians (same files and construction as the benchmark)
# ---------------------------------------------------------------------------


def load_settings():
    with open(BENCHMARK_DIR / "graph_settings.json") as f:
        s = json.load(f)
    lam = sorted({*EXTRA_LAMBDAS, *map(float, s["GRAPH_LAMBDA_GRID"])})
    beta = [float(x) for x in s["GRAPH_BETA_GRID"]]
    return lam, beta


def load_laplacians():
    with open(BENCHMARK_DIR / "effective_laplacians_clean.pkl", "rb") as f:
        laps = {k: np.asarray(v, dtype=float) for k, v in pickle.load(f).items()}
    with open(MASTER_GRAPH_FILE, "rb") as f:
        G = pickle.load(f)
    nodes = sorted(
        n for n, d in G.nodes(data=True)
        if d.get("usgs_station_count", 0) > 0 or len(d.get("usgs_clusters", [])) > 0
    )

    def kron(edge_weight):
        order = list(G.nodes())
        pos = {n: i for i, n in enumerate(order)}
        L = np.zeros((len(order), len(order)))
        for u, v, data in G.to_undirected(as_view=True).edges(data=True):
            w = edge_weight(data)
            i, j = pos[u], pos[v]
            L[i, j] -= w
            L[j, i] -= w
            L[i, i] += w
            L[j, j] += w
        b = np.array([pos[n] for n in nodes])
        it = np.setdiff1d(np.arange(len(order)), b)
        Lk = L[np.ix_(b, b)] - L[np.ix_(b, it)] @ np.linalg.solve(L[np.ix_(it, it)], L[np.ix_(it, b)])
        Lk = (Lk + Lk.T) / 2.0
        np.fill_diagonal(Lk, 0.0)
        np.fill_diagonal(Lk, -Lk.sum(axis=1))
        d = np.diag(Lk)
        return Lk / d[d > 1e-12].mean()

    if not np.allclose(kron(lambda d: 1000.0 / float(d["length_m"])), laps["distance_only"],
                       rtol=1e-8, atol=1e-10):
        raise ValueError("Rebuilt distance_only Laplacian does not match the saved one.")
    laps["unweighted_compressed"] = kron(lambda d: 1.0)
    return laps, {int(n): i for i, n in enumerate(nodes)}


def variables_for(target, K):
    v = {c: [c] for c in TARGET_CHEMISTRY[target]}
    v["daily cycle"] = [f"hour_{t}_{k}" for k in range(1, K + 1) for t in ("sin", "cos")]
    v["annual cycle"] = [f"doy_{t}_{k}" for k in range(1, K + 1) for t in ("sin", "cos")]
    v.update({s: [s] for s in STATIC})
    v[GRAPH] = []
    return v


# ---------------------------------------------------------------------------
# exact fits from per-node moments (same algebra as the benchmark)
# ---------------------------------------------------------------------------


def node_stats(df, target, features):
    stats = {}
    for node, g in df.groupby("node"):
        X = g[features].to_numpy(float)
        y = g[target].to_numpy(float)
        n = len(g)
        stats[int(node)] = dict(mean_x=X.mean(0), exx=X.T @ X / n, exy=X.T @ y / n,
                                mean_y=y.mean(), mean_y2=np.mean(y ** 2))
    return stats


def slice_stats(stats, idx):
    idx = np.asarray(idx, dtype=int)
    return {n: dict(mean_x=s["mean_x"][idx], exx=s["exx"][np.ix_(idx, idx)], exy=s["exy"][idx],
                    mean_y=s["mean_y"], mean_y2=s["mean_y2"]) for n, s in stats.items()}


def std_moments(stat, mean_x, std_x):
    mx, exx, exy, my = stat["mean_x"], stat["exx"], stat["exy"], stat["mean_y"]
    zmean = (mx - mean_x) / std_x
    zexx = (exx - np.outer(mx, mean_x) - np.outer(mean_x, mx) + np.outer(mean_x, mean_x)) \
        / np.outer(std_x, std_x)
    zexy = (exy - mean_x * my) / std_x
    p = len(mean_x)
    q2 = np.empty((p + 1, p + 1))
    q2[0, 0] = 1.0
    q2[0, 1:] = q2[1:, 0] = zmean
    q2[1:, 1:] = zexx
    return np.concatenate(([1.0], zmean)), q2, np.concatenate(([my], zexy))


def train_moments(stats, train):
    p = len(next(iter(stats.values()))["mean_x"])
    if p:
        mean_x = np.mean([stats[n]["mean_x"] for n in train], axis=0)
        ex2 = np.mean([np.diag(stats[n]["exx"]) for n in train], axis=0)
        std_x = np.sqrt(np.maximum(ex2 - mean_x ** 2, 1e-12))
    else:
        mean_x = std_x = np.zeros(0)
    q = p + 1
    A11, rb, qmean = np.zeros((q, q)), np.zeros(q), {}
    for n in train:
        qm, q2, qy = std_moments(stats[n], mean_x, std_x)
        A11 += q2
        rb += qy
        qmean[n] = qm
    return dict(mean_x=mean_x, std_x=std_x, q=q, A11=A11, rb=rb, qmean=qmean,
                mean_y={n: stats[n]["mean_y"] for n in train})


def ridge(q):
    R = np.eye(q)
    R[0, 0] = 0.0                                    # intercept not penalised
    return R


def solve_ridge(tm, lb):
    beta = np.linalg.solve(tm["A11"] + lb * ridge(tm["q"]), tm["rb"])
    return dict(beta=beta, g=None, mean_x=tm["mean_x"], std_x=tm["std_x"])


def prepare_graph(tm, L, lam, node_index):
    N, q = L.shape[0], tm["q"]
    Bt, rg, mask = np.zeros((N, q)), np.zeros(N), np.zeros(N)
    for n, qm in tm["qmean"].items():
        j = node_index[n]
        Bt[j] += qm
        mask[j] = 1.0
        rg[j] = tm["mean_y"][n]
    c = np.ones(N)                                   # gauge: sum of g = 0
    sol = np.linalg.solve(np.diag(mask) + lam * L, np.column_stack((Bt, rg, c)))
    X, u, v = sol[:, :q], sol[:, q], sol[:, q + 1]
    return dict(tm=tm, X=X, u=u, v=v, BtX=Bt.T @ X, Btu=Bt.T @ u, Btv=Bt.T @ v,
                cX=c @ X, cu=c @ u, cv=c @ v)


def solve_graph(prep, lb):
    tm, q = prep["tm"], prep["tm"]["q"]
    S = np.empty((q + 1, q + 1))
    S[:q, :q] = tm["A11"] + lb * ridge(q) - prep["BtX"]
    S[:q, q] = -prep["Btv"]
    S[q, :q] = prep["cX"]
    S[q, q] = prep["cv"]
    z = np.linalg.solve(S, np.concatenate((tm["rb"] - prep["Btu"], [prep["cu"]])))
    beta, mu = z[:q], z[q]
    return dict(beta=beta, g=prep["u"] - prep["X"] @ beta - prep["v"] * mu,
                mean_x=tm["mean_x"], std_x=tm["std_x"])


def coef(model, node, node_index):
    c = model["beta"].copy()
    if model["g"] is not None:
        c[0] += model["g"][node_index[node]]
    return c


def node_mse(stat, model, node, node_index):
    _, q2, qy = std_moments(stat, model["mean_x"], model["std_x"])
    c = coef(model, node, node_index)
    return max(float(stat["mean_y2"] - 2.0 * c @ qy + c @ q2 @ c), 0.0)


# ---------------------------------------------------------------------------
# one configuration = one nested LOO
# ---------------------------------------------------------------------------


def nested_loo(job):
    """job: dict(target, study, variable, features, graph, ...). Returns (row, tuning rows)."""
    t0 = time.time()
    stats, df, features, use_graph = job["stats"], job["df"], job["features"], job["graph"]
    laps, node_index, lam_grid, beta_grid = job["laps"], job["node_index"], job["lam"], job["beta"]
    target = job["target"]
    nodes = sorted(stats)
    outer = nodes[:job["max_outer"]] if job["max_outer"] else nodes
    preds, tuning = [], []

    for test in outer:
        train = [n for n in nodes if n != test]
        totals = {}
        for val in train:
            tm = train_moments(stats, [n for n in train if n != val])
            if use_graph:
                for scheme, L in laps.items():
                    for lam in lam_grid:
                        prep = prepare_graph(tm, L, lam, node_index)
                        for lb in beta_grid:
                            k = (scheme, lam, lb)
                            totals[k] = totals.get(k, 0.0) + node_mse(
                                stats[val], solve_graph(prep, lb), val, node_index)
            else:
                for lb in beta_grid:
                    k = (None, None, lb)
                    totals[k] = totals.get(k, 0.0) + node_mse(
                        stats[val], solve_ridge(tm, lb), val, node_index)
        scheme, lam, lb = min(totals, key=totals.get)
        tm = train_moments(stats, train)
        model = (solve_graph(prepare_graph(tm, laps[scheme], lam, node_index), lb)
                 if use_graph else solve_ridge(tm, lb))

        test_df = df[df["node"] == test]
        Z = (test_df[features].to_numpy(float) - model["mean_x"]) / np.where(
            model["std_x"] > 0, model["std_x"], 1.0) if features else np.zeros((len(test_df), 0))
        pred = np.column_stack((np.ones(len(test_df)), Z)) @ coef(model, test, node_index)
        preds.append(pd.DataFrame(dict(node=test, y=test_df[target].to_numpy(float), p=pred)))
        tuning.append(dict(target=target, study=job["study"], variable=job["variable"],
                           outer_test_node=test, scheme=scheme, lam=lam, lambda_beta=lb))

    p = pd.concat(preds, ignore_index=True)
    w = p["node"].map(1.0 / p.groupby("node").size()).to_numpy()
    sse = np.sum(w * (p["y"] - p["p"]) ** 2)
    ybar = np.sum(w * p["y"]) / w.sum()
    r2 = 1.0 - sse / np.sum(w * (p["y"] - ybar) ** 2)
    rmse = np.sqrt(sse / w.sum())
    row = dict(target=target, study=job["study"], variable=job["variable"],
               n_features=len(features), graph=use_graph, R2=r2, RMSE=rmse,
               seconds=time.time() - t0)
    return row, tuning


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", choices=list(TARGET_CHEMISTRY), default=list(TARGET_CHEMISTRY))
    ap.add_argument("--K", type=int, default=4, help="Fourier harmonics in the daily and annual cycles")
    ap.add_argument("--jobs", type=int, default=1, help="configurations run in parallel")
    ap.add_argument("--max-outer-folds", type=int, default=None,
                    help="smoke test: only the first N outer folds (results not saved)")
    args = ap.parse_args()

    t0 = time.time()
    lam_grid, beta_grid = load_settings()
    laps, node_index = load_laplacians()
    data = pd.read_pickle(BENCHMARK_DIR / "all_static_df.pkl")
    print(f"Weightings: {list(laps)}\nlambda grid: {lam_grid}\nlambda_beta grid: {beta_grid}")

    jobs = []
    for target in args.targets:
        variables = variables_for(target, args.K)
        allf = [f for v, cols in variables.items() for f in cols]
        df = data[["node", target, *allf]].dropna(subset=[target, *TARGET_CHEMISTRY[target]]).copy()
        full_stats = node_stats(df, target, allf)
        print(f"{target}: {df['node'].nunique()} stations, {len(df):,} rows, "
              f"variables: {list(variables)}")

        def config(study, variable, keep):
            feats = [f for v in keep if v != GRAPH for f in variables[v]]
            jobs.append(dict(
                target=target, study=study, variable=variable, features=feats, graph=GRAPH in keep,
                stats=slice_stats(full_stats, [allf.index(f) for f in feats]), df=df,
                laps=laps, node_index=node_index, lam=lam_grid, beta=beta_grid,
                max_outer=args.max_outer_folds))

        names = list(variables)
        config("full", "(all variables)", names)
        config("base", "(intercept only)", [])
        for v in names:
            config("ablation", v, [n for n in names if n != v])
            config("reverse", v, [v])

    print(f"{len(jobs)} configurations\n")
    rows, tunings = [], []
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            results = ex.map(nested_loo, jobs)
            for row, tun in results:
                rows.append(row)
                tunings += tun
                print(f"  {row['target']:<21} {row['study']:<9} {row['variable']:<21} "
                      f"R2={row['R2']:.4f} ({row['seconds']:.0f}s)", flush=True)
    else:
        for job in jobs:
            row, tun = nested_loo(job)
            rows.append(row)
            tunings += tun
            print(f"  {row['target']:<21} {row['study']:<9} {row['variable']:<21} "
                  f"R2={row['R2']:.4f} ({row['seconds']:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    ref = res[res["study"].isin(["full", "base"])].set_index(["target", "study"])["R2"]
    res["delta_R2"] = np.nan
    abl = res["study"] == "ablation"
    rev = res["study"] == "reverse"
    res.loc[abl, "delta_R2"] = res.loc[abl, "R2"].to_numpy() - ref.loc[
        list(zip(res.loc[abl, "target"], ["full"] * abl.sum()))].to_numpy()
    res.loc[rev, "delta_R2"] = res.loc[rev, "R2"].to_numpy() - ref.loc[
        list(zip(res.loc[rev, "target"], ["base"] * rev.sum()))].to_numpy()

    pd.set_option("display.width", 200)
    for target, g in res.groupby("target", sort=False):
        full = g.loc[g["study"] == "full", "R2"].iloc[0]
        base = g.loc[g["study"] == "base", "R2"].iloc[0]
        t = (g[g["study"].isin(["ablation", "reverse"])]
             .pivot(index="variable", columns="study", values="delta_R2")
             .rename(columns={"ablation": "drop from full", "reverse": "add to base"})
             .sort_values("drop from full"))
        print(f"\n{target}   full R2 = {full:.4f}   base R2 = {base:.4f}")
        print(t.to_string(float_format=lambda x: f"{x:+.4f}"))

    if args.max_outer_folds is not None:
        print("\n(--max-outer-folds was set: smoke test only, results not saved)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.drop(columns="seconds").to_csv(OUT_DIR / "ablation_results.csv", index=False)
    pd.DataFrame(tunings).to_csv(OUT_DIR / "ablation_tuning.csv", index=False)
    print(f"\nSaved results/ablation/ablation_results.csv and ablation_tuning.csv "
          f"(total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
