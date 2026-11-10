"""
Effect of edge weights on the graph-Laplacian model
===================================================

Colorado basin, original (uncompressed, 67,401 nodes) and compressed (67 nodes)
river graphs. Builds Laplacians with and without edge weights, Kron-reduces each
onto the 26 WQ nodes, compares the matrices, and reruns the nested
leave-one-node-out graph-Laplacian regression with each one.

Edge weightings
    distance     w = 1 / length_km   (what every model so far has used)
    unweighted   w = 1 on every edge of the graph being used
    hops         compressed graph only: w = 1 / original_edge_count, i.e. the
                 unweighted original graph's segments in series. Included to show
                 why "unweighted" differs between the two graphs.

Laplacians compared (all Kron-reduced to 26x26 and divided by their mean
positive diagonal, as in the notebook; lambda is tuned, so only the RELATIVE
edge weights matter):
    distance_uncompressed     distance_compressed
    unweighted_uncompressed   unweighted_compressed
    hops_compressed           (should equal unweighted_uncompressed)

Both graphs are trees, so the effective resistance between two WQ nodes is just
the sum of 1/w along the river path: km of river for "distance", the number of
original segments for "unweighted" on the original graph, and the number of
compressed reaches for "unweighted" on the compressed graph.

Predictors and grids are the same as compare_kron_vs_full_laplacian.py
(chemistry only), so the distance_* rows reproduce kron_vs_full_laplacian_r2.csv.

Run from this folder (about a minute; every fit uses a 26x26 matrix):

    python compare_edge_weights.py
    python compare_edge_weights.py --targets pH --max-outer-folds 2   # smoke test

Outputs (results/):
    edge_weight_matrix_checks.csv
    edge_weight_r2.csv
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy import sparse

from compare_kron_vs_full_laplacian import (
    BASE_DIR,
    TARGET_FEATURES,
    GraphSpace,
    haversine_m,
    effective_resistance_small,
    kron_reduce,
    kron_scale,
    load_inputs,
    nested_loo,
    rel_fro,
)

OUT_MATRIX = BASE_DIR / "results/edge_weight_matrix_checks.csv"
OUT_R2 = BASE_DIR / "results/edge_weight_r2.csv"

VARIANTS = [
    ("distance", "uncompressed"),
    ("distance", "compressed"),
    ("unweighted", "uncompressed"),
    ("unweighted", "compressed"),
    ("hops", "compressed"),
]


def edge_weight(G, u, v, data, weighting):
    if weighting == "unweighted":
        return 1.0
    if weighting == "hops":
        return 1.0 / max(float(data.get("original_edge_count", 1)), 1.0)
    length_m = data.get("length_m")
    if length_m is None or not np.isfinite(float(length_m)) or float(length_m) <= 0:
        a, b = G.nodes[u], G.nodes[v]
        length_m = haversine_m(a["longitude"], a["latitude"], b["longitude"], b["latitude"])
    return 1.0 / max(float(length_m) / 1000.0, 1e-8)


def laplacian(G, weighting):
    """L = D - W on the undirected graph."""
    order = list(G.nodes())
    index = {n: i for i, n in enumerate(order)}
    rows, cols, vals = [], [], []
    for u, v, data in G.to_undirected(as_view=True).edges(data=True):
        w = edge_weight(G, u, v, data, weighting)
        i, j = index[u], index[v]
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    n = len(order)
    W = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    L = (sparse.diags(np.asarray(W.sum(axis=1)).ravel()) - W).tocsc()
    return L, index


def off_diagonal(M):
    return M[~np.eye(len(M), dtype=bool)]


def compare(name_a, a, name_b, b):
    """How different two Kron Laplacians are, after the model's scaling."""
    Sa, Sb = a["L_kron"] / a["scale"], b["L_kron"] / b["scale"]
    # resistance of the scaled matrices (raw units differ: km vs segment counts)
    Ra, Rb = off_diagonal(a["R"]) * a["scale"], off_diagonal(b["R"]) * b["scale"]
    ea, eb = np.linalg.eigvalsh(Sa)[1:], np.linalg.eigvalsh(Sb)[1:]
    return {
        "comparison": f"{name_a} vs {name_b}",
        "rel_fro_scaled": rel_fro(Sa, Sb),
        "eigenvalue_max_rel_diff_scaled": np.max(np.abs(ea - eb) / np.abs(ea)),
        "resistance_max_rel_diff": np.max(np.abs(Ra - Rb) / Ra),
        "resistance_corr": np.corrcoef(Ra, Rb)[0, 1],
        "resistance_rank_corr": pd.Series(Ra).corr(pd.Series(Rb), method="spearman"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", choices=list(TARGET_FEATURES), default=list(TARGET_FEATURES))
    ap.add_argument("--max-outer-folds", type=int, default=None,
                    help="smoke test: only the first N outer folds (results not saved)")
    args = ap.parse_args()

    t0 = time.time()
    G_full, G_master, wq_nodes, node_hourly = load_inputs()
    graphs = {"uncompressed": G_full, "compressed": G_master}
    print(f"Loaded inputs in {time.time() - t0:.1f}s  |  WQ nodes: {len(wq_nodes)}")
    for name, G in graphs.items():
        print(f"  {name:<13} {G.number_of_nodes():>6} nodes  "
              f"{G.to_undirected(as_view=True).number_of_edges():>6} edges")

    # ---------------- Laplacians ----------------
    lap = {}
    for weighting, graph in VARIANTS:
        L, index = laplacian(graphs[graph], weighting)
        b_idx = np.array([index[n] for n in wq_nodes])
        Lk, _, _ = kron_reduce(L, b_idx)
        key = f"{weighting}_{graph}"
        lap[key] = dict(L_kron=Lk, scale=kron_scale(Lk), R=effective_resistance_small(Lk))
        R = off_diagonal(lap[key]["R"])
        print(f"  {key:<24} Kron scale={lap[key]['scale']:.4g}  "
              f"pairwise resistance: median {np.median(R):.4g}, range {R.min():.4g}-{R.max():.4g}")

    # ---------------- matrix checks ----------------
    pairs = [
        ("distance_uncompressed", "distance_compressed"),      # compression, weighted
        ("unweighted_uncompressed", "unweighted_compressed"),  # compression, unweighted
        ("unweighted_uncompressed", "hops_compressed"),        # compression with segment counts
        ("distance_uncompressed", "unweighted_uncompressed"),  # weights, original graph
        ("distance_compressed", "unweighted_compressed"),      # weights, compressed graph
    ]
    checks = pd.DataFrame([compare(a, lap[a], b, lap[b]) for a, b in pairs])
    pd.set_option("display.width", 250)
    print("\nHOW DIFFERENT ARE THE LAPLACIANS? (scaled, as used in the model)")
    print(checks.to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    # ---------------- nested LOO ----------------
    kron_pos = {n: i for i, n in enumerate(wq_nodes)}
    spaces = [GraphSpace(k, v["L_kron"] / v["scale"], kron_pos, np.arange(len(wq_nodes)))
              for k, v in lap.items()]

    results, tunings = [], []
    for target in args.targets:
        features = TARGET_FEATURES[target]
        df = node_hourly[["node", "time", target] + features].dropna().copy()
        print(f"\nTARGET: {target}  (predictors: {features}, nodes: {df['node'].nunique()})")
        row, _, tuning = nested_loo(df, target, features, spaces, args.max_outer_folds)
        results.append(row)
        tunings.append(tuning)

    res = pd.DataFrame(results)
    tun = pd.concat(tunings, ignore_index=True)

    r2_cols = ["Target", "Nodes", "linear R2"] + [f"{s.name} R2" for s in spaces]
    print("\nR^2 by edge weighting (linear = no graph term):")
    print(res[r2_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nChosen lambda (outer folds) per Laplacian:")
    for s in spaces:
        counts = tun.groupby("target")[f"{s.name}_lambda"].agg(lambda x: x.value_counts().to_dict())
        print(f"  {s.name}")
        for target, c in counts.items():
            print(f"    {target:<22} {c}")

    if args.max_outer_folds is not None:
        print("\n(--max-outer-folds was set: smoke test only, results not saved)")
        return

    checks.to_csv(OUT_MATRIX, index=False)
    res.to_csv(OUT_R2, index=False)
    print(f"\nSaved {OUT_MATRIX.relative_to(BASE_DIR)} and {OUT_R2.relative_to(BASE_DIR)}"
          f"   (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
