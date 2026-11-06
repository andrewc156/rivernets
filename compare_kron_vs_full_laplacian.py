"""
Kron (Schur-complement) Laplacian vs. the full ("normal") Laplacian
====================================================================

Colorado basin, 1/distance_km edge weights -- the same set-up as cells 1-13 of
largest_uncompressed_laplacian.ipynb.

(a) Builds, for BOTH the uncompressed largest component (67,401 nodes) and the
    master compressed graph (67 nodes):
      * the full combinatorial Laplacian  L = D - W
      * the Kron-reduced Laplacian onto the 26 WQ (boundary) nodes
            L_kron = L_BB - L_BI L_II^{-1} L_IB

(b) Measures how different they are:
      * Kron(uncompressed) vs Kron(compressed)       -- direct matrix difference
      * Kron vs full Laplacian, per graph            -- the two are different
        sizes, so we compare what Kron is supposed to preserve exactly:
          - effective resistance between every pair of WQ nodes
          - Dirichlet energy  x^T L x  of the harmonic extension of random
            boundary values  ==  x_B^T L_kron x_B
      * the naive sub-matrix L_BB (just dropping interior nodes) as a reference
        for what a WRONG reduction looks like

(c) Re-runs the nested leave-one-node-out graph-Laplacian regression with four
    Laplacians and compares the R^2 / RMSE:
      kron_uncompressed    26x26 Kron of the 67k graph   (what the notebook used)
      kron_compressed      26x26 Kron of the 67-node graph
      full_uncompressed    the 67,401x67,401 Laplacian, a graph effect g on EVERY node
      full_compressed      the 67x67 Laplacian, a graph effect g on every node
    The full_* fits never form a Schur complement: they solve the complete
    (beta, g_all_nodes) normal equations with a sparse factorisation.
    Each full Laplacian is divided by the same scale factor as its Kron
    counterpart (mean positive diagonal of the Kron matrix, as in the notebook),
    so lambda means the same thing in both and the comparison is apples to apples.

Run from this folder:

    python compare_kron_vs_full_laplacian.py              # everything (~15 min, the
                                                          #  67k-node fits dominate)
    python compare_kron_vs_full_laplacian.py --skip-r2    # (a)+(b) only, seconds
    python compare_kron_vs_full_laplacian.py --targets pH --max-outer-folds 1   # smoke test

Outputs (results/):
    kron_vs_full_laplacian_matrix_checks.csv
    kron_vs_full_laplacian_r2.csv
"""

from __future__ import annotations

import argparse
import pickle
import time
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
FULL_GRAPH_FILE = BASE_DIR / "graphs/colorado/largest_river_component.pkl"
MASTER_GRAPH_FILE = BASE_DIR / "graphs/colorado/master_compressed_river_graph.pkl"
HOURLY_FILE = BASE_DIR / "data/usgs/usgs_all_stations_hourly.csv"
PREVIOUS_RESULTS_FILE = BASE_DIR / "results/full_graph_nested_loo_results.csv"
OUT_MATRIX = BASE_DIR / "results/kron_vs_full_laplacian_matrix_checks.csv"
OUT_R2 = BASE_DIR / "results/kron_vs_full_laplacian_r2.csv"

WQ_VARIABLES = ["DO concentration", "Water temperature", "Specific conductance", "Turbidity", "pH"]

# Same predictor choices / grids as the notebook.
TARGET_FEATURES = {
    "DO concentration": ["Water temperature", "Specific conductance"],
    "Water temperature": ["Specific conductance"],
    "pH": ["Specific conductance"],
    "Specific conductance": ["Water temperature"],
    "Turbidity": ["Specific conductance", "pH"],
}
LINEAR_ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
GRAPH_LAMBDA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
GRAPH_BETA_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


# ============================================================================
# Loading (mirrors notebook cell 1 / cell 3)
# ============================================================================

def load_inputs():
    with open(FULL_GRAPH_FILE, "rb") as f:
        G_full = pickle.load(f)
    with open(MASTER_GRAPH_FILE, "rb") as f:
        G_master = pickle.load(f)

    station_to_node = {}
    for node, data in G_master.nodes(data=True):
        for cluster in data.get("usgs_clusters", []):
            stations = cluster.get("member_stations", [])
            if not stations and cluster.get("anchor_station") is not None:
                stations = [cluster["anchor_station"]]
            for station in stations:
                station_to_node[station] = node

    wq_nodes = sorted(
        node for node, data in G_master.nodes(data=True)
        if data.get("usgs_station_count", 0) > 0 or len(data.get("usgs_clusters", [])) > 0
    )

    hourly = pd.read_csv(HOURLY_FILE)
    hourly["time"] = pd.to_datetime(hourly["time"], utc=True, errors="coerce")
    hourly["node"] = hourly["station_id"].map(station_to_node)
    hourly = hourly[hourly["node"].notna()].copy()
    hourly["node"] = hourly["node"].astype(int)

    node_hourly = (
        hourly[["node", "time"] + WQ_VARIABLES]
        .groupby(["node", "time"], as_index=False)[WQ_VARIABLES]
        .median()
    )
    return G_full, G_master, wq_nodes, node_hourly


# ============================================================================
# (a) Laplacians
# ============================================================================

def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(lambda v: radians(float(v)), (lon1, lat1, lon2, lat2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2.0 * 6371000.0 * asin(sqrt(a))


def distance_laplacian(G):
    """L = D - W on the undirected graph, w_ij = 1 / distance_km (notebook cell 2/9)."""
    order = list(G.nodes())
    index = {n: i for i, n in enumerate(order)}
    rows, cols, vals = [], [], []
    for u, v, data in G.to_undirected(as_view=True).edges(data=True):
        length_m = data.get("length_m")
        if length_m is None or not np.isfinite(float(length_m)) or float(length_m) <= 0:
            a, b = G.nodes[u], G.nodes[v]
            length_m = haversine_m(a["longitude"], a["latitude"], b["longitude"], b["latitude"])
        w = 1.0 / max(float(length_m) / 1000.0, 1e-8)
        i, j = index[u], index[v]
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    n = len(order)
    W = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    L = (sparse.diags(np.asarray(W.sum(axis=1)).ravel()) - W).tocsc()
    return L, order, index


def kron_reduce(L, boundary_idx):
    """Schur complement onto boundary_idx, with the notebook's clean-up
    (symmetrise + force zero row sums). Returns (L_kron_clean, L_kron_raw, parts)."""
    n = L.shape[0]
    interior_mask = np.ones(n, dtype=bool)
    interior_mask[boundary_idx] = False
    interior_idx = np.flatnonzero(interior_mask)

    L = L.tocsr()
    L_BB = L[boundary_idx][:, boundary_idx].toarray()
    L_BI = L[boundary_idx][:, interior_idx].tocsc()
    L_IB = L[interior_idx][:, boundary_idx].toarray()
    L_II = L[interior_idx][:, interior_idx].tocsc()

    lu_II = splu(L_II)
    raw = np.asarray(L_BB - L_BI @ lu_II.solve(L_IB))

    clean = (raw + raw.T) / 2.0
    np.fill_diagonal(clean, 0.0)
    np.fill_diagonal(clean, -clean.sum(axis=1))
    parts = dict(interior_idx=interior_idx, L_BB=L_BB, L_IB=L_IB, lu_II=lu_II)
    return clean, raw, parts


def kron_scale(L_kron):
    d = np.diag(L_kron)
    return d[d > 1e-12].mean()


# ============================================================================
# (b) How different are they?
# ============================================================================

def rel_fro(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(A)


def effective_resistance_full(L, boundary_idx):
    """Pairwise effective resistance among boundary nodes, computed on the FULL
    Laplacian by grounding boundary node 0 and solving sparse systems."""
    ground = boundary_idx[0]
    keep = np.setdiff1d(np.arange(L.shape[0]), [ground])
    pos = {g: k for k, g in enumerate(keep)}
    lu = splu(L.tocsr()[keep][:, keep].tocsc())
    others = boundary_idx[1:]
    rhs = np.zeros((len(keep), len(others)))
    for c, b in enumerate(others):
        rhs[pos[b], c] = 1.0
    sol = lu.solve(rhs)
    Gram = np.zeros((len(boundary_idx), len(boundary_idx)))
    Gram[1:, 1:] = sol[[pos[b] for b in others], :]
    return _resistance_from_gram(Gram)


def effective_resistance_small(Lk):
    Gram = np.zeros_like(Lk)
    Gram[1:, 1:] = np.linalg.inv(Lk[1:, 1:])
    return _resistance_from_gram(Gram)


def _resistance_from_gram(Gram):
    d = np.diag(Gram)
    return d[:, None] + d[None, :] - 2.0 * Gram


def energy_check(L, boundary_idx, L_kron, parts, n_trials=5, seed=0):
    """max relative |x^T L x - x_B^T L_kron x_B| over random boundary vectors,
    with x the harmonic extension (interior solves L_II x_I = -L_IB x_B)."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    n = L.shape[0]
    for _ in range(n_trials):
        xB = rng.standard_normal(len(boundary_idx))
        x = np.zeros(n)
        x[boundary_idx] = xB
        x[parts["interior_idx"]] = -parts["lu_II"].solve(parts["L_IB"] @ xB)
        e_full = float(x @ (L @ x))
        e_kron = float(xB @ L_kron @ xB)
        worst = max(worst, abs(e_full - e_kron) / abs(e_full))
    return worst


def matrix_checks(graphs, wq_nodes):
    rows = []

    def add(comparison, metric, value):
        rows.append(dict(comparison=comparison, metric=metric, value=value))
        print(f"  {comparison:<48} {metric:<34} {value:.3e}")

    print("\n" + "=" * 100)
    print("(b) HOW DIFFERENT ARE THE LAPLACIANS?")
    print("=" * 100)

    R_full = {}
    for name, g in graphs.items():
        L, b_idx, Lk, Lk_raw, parts = g["L"], g["b_idx"], g["L_kron"], g["L_kron_raw"], g["parts"]
        label = f"Kron vs full  [{name}]"
        add(label, "notebook clean-up change (rel fro)", rel_fro(Lk_raw, Lk))
        add(label, "Kron symmetry error (max abs)", np.abs(Lk_raw - Lk_raw.T).max())
        add(label, "Kron row-sum error (max abs)", np.abs(Lk.sum(axis=1)).max())

        R_f = effective_resistance_full(L, b_idx)
        R_k = effective_resistance_small(Lk)
        off = ~np.eye(len(b_idx), dtype=bool)
        R_full[name] = R_f
        add(label, "eff. resistance max rel error", (np.abs(R_f - R_k)[off] / R_f[off]).max())
        add(label, "Dirichlet energy max rel error", energy_check(L, b_idx, Lk, parts))
        add(f"naive L_BB vs Kron  [{name}]", "rel fro (reference: NOT equal)",
            rel_fro(Lk, parts["L_BB"]))

    ku, kc = graphs["uncompressed"]["L_kron"], graphs["compressed"]["L_kron"]
    su, sc = graphs["uncompressed"]["scale"], graphs["compressed"]["scale"]
    add("Kron(uncompressed) vs Kron(compressed)", "max abs diff (unscaled)", np.abs(ku - kc).max())
    add("Kron(uncompressed) vs Kron(compressed)", "rel fro diff (unscaled)", rel_fro(ku, kc))
    add("Kron(uncompressed) vs Kron(compressed)", "rel fro diff (scaled, as in model)",
        rel_fro(ku / su, kc / sc))
    add("Kron(uncompressed) vs Kron(compressed)", "eigenvalue max rel diff (scaled)",
        _eig_rel_diff(ku / su, kc / sc))
    off = ~np.eye(len(wq_nodes), dtype=bool)
    Ru, Rc = R_full["uncompressed"], R_full["compressed"]
    add("full(uncompressed) vs full(compressed)", "eff. resistance max rel diff",
        (np.abs(Ru - Rc)[off] / Ru[off]).max())

    return pd.DataFrame(rows)


def _eig_rel_diff(A, B):
    ea, eb = np.linalg.eigvalsh(A)[1:], np.linalg.eigvalsh(B)[1:]  # drop the 0 eigenvalue
    return np.max(np.abs(ea - eb) / np.abs(ea))


# ============================================================================
# (c) Nested-LOO graph-Laplacian regression (notebook cell 7, generalised to
#     an arbitrary Laplacian over any node set that contains the WQ nodes)
# ============================================================================

def make_node_stats(df, target, features):
    stats = {}
    for node, group in df.groupby("node"):
        X = group[features].to_numpy(float)
        y = group[target].to_numpy(float)
        n = len(group)
        stats[int(node)] = dict(n=n, mean_x=X.mean(axis=0), exx=X.T @ X / n,
                                exy=X.T @ y / n, mean_y=y.mean(), mean_y2=np.mean(y ** 2))
    return stats


def training_scaling(stats, train_nodes):
    mean_x = np.mean([stats[n]["mean_x"] for n in train_nodes], axis=0)
    ex2 = np.mean([np.diag(stats[n]["exx"]) for n in train_nodes], axis=0)
    return mean_x, np.sqrt(np.maximum(ex2 - mean_x ** 2, 1e-12))


def standard_moments(stat, mean_x, std_x):
    mx, exx, exy, my = stat["mean_x"], stat["exx"], stat["exy"], stat["mean_y"]
    zmean = (mx - mean_x) / std_x
    zexx = (exx - np.outer(mx, mean_x) - np.outer(mean_x, mx) + np.outer(mean_x, mean_x)) \
        / np.outer(std_x, std_x)
    zexy = (exy - mean_x * my) / std_x
    p = len(mean_x)
    q2 = np.zeros((p + 1, p + 1))
    q2[0, 0] = 1.0
    q2[0, 1:] = q2[1:, 0] = zmean
    q2[1:, 1:] = zexx
    return np.concatenate(([1.0], zmean)), q2, np.concatenate(([my], zexy))


def ridge_matrix(q):
    R = np.eye(q)
    R[0, 0] = 0.0
    return R


def fit_linear(stats, train_nodes, alpha):
    mean_x, std_x = training_scaling(stats, train_nodes)
    q = len(mean_x) + 1
    A, b = np.zeros((q, q)), np.zeros(q)
    for n in train_nodes:
        _, q2, qy = standard_moments(stats[n], mean_x, std_x)
        A += q2
        b += qy
    beta = np.linalg.solve(A + alpha * ridge_matrix(q), b)
    return dict(beta=beta, mean_x=mean_x, std_x=std_x, g=None)


class GraphSpace:
    """A (scaled) Laplacian plus where the WQ nodes sit inside it."""

    def __init__(self, name, L_scaled, node_pos, constraint_idx):
        self.name = name
        self.L = sparse.csc_matrix(L_scaled)
        self.N = self.L.shape[0]
        self.dense = self.N <= 2000
        if self.dense:
            self.L_dense = self.L.toarray()
        self.node_pos = node_pos            # WQ node id -> row in L
        self.c = np.zeros(self.N)            # gauge: sum of g over WQ nodes = 0
        self.c[constraint_idx] = 1.0         # (same as the notebook for Kron)


def prepare_graph(stats, train_nodes, lambda_lap, space):
    """Everything that does not depend on lambda_beta, so one factorisation of
    K = diag(mask) + lambda * L is shared across the beta grid.

    Full system (unknowns beta, g in R^N, mu):
        [A11+rb*R   B      0 ] [beta]   [r_beta]
        [B^T        K      c ] [g   ] = [r_g   ]
        [0          c^T    0 ] [mu  ]   [0     ]
    solved exactly by block elimination of g (K is SPD for a connected graph
    with >=1 observed node). No Kron reduction is involved here."""
    mean_x, std_x = training_scaling(stats, train_nodes)
    q = len(mean_x) + 1
    A11 = np.zeros((q, q))
    rb = np.zeros(q)
    Bt = np.zeros((space.N, q))
    rg = np.zeros(space.N)
    mask = np.zeros(space.N)
    for n in train_nodes:
        qmean, q2, qy = standard_moments(stats[n], mean_x, std_x)
        j = space.node_pos[n]
        A11 += q2
        rb += qy
        Bt[j] += qmean
        mask[j] = 1.0
        rg[j] = stats[n]["mean_y"]

    rhs = np.column_stack((Bt, rg, space.c))
    if space.dense:
        sol = np.linalg.solve(np.diag(mask) + lambda_lap * space.L_dense, rhs)
    else:
        K = (sparse.diags(mask) + lambda_lap * space.L).tocsc()
        sol = splu(K, permc_spec="MMD_AT_PLUS_A").solve(rhs)
    X, u, v = sol[:, :q], sol[:, q], sol[:, q + 1]
    return dict(A11=A11, rb=rb, Bt=Bt, X=X, u=u, v=v, c=space.c, q=q,
                mean_x=mean_x, std_x=std_x)


def solve_graph(prep, lambda_beta):
    q, Bt, X, u, v, c = prep["q"], prep["Bt"], prep["X"], prep["u"], prep["v"], prep["c"]
    S = np.zeros((q + 1, q + 1))
    S[:q, :q] = prep["A11"] + lambda_beta * ridge_matrix(q) - Bt.T @ X
    S[:q, q] = -(Bt.T @ v)
    S[q, :q] = c @ X
    S[q, q] = c @ v
    r = np.concatenate((prep["rb"] - Bt.T @ u, [c @ u]))
    z = np.linalg.solve(S, r)
    beta, mu = z[:q], z[q]
    g = u - X @ beta - v * mu
    return dict(beta=beta, g=g, mean_x=prep["mean_x"], std_x=prep["std_x"])


def node_mse(node, stat, model, space=None):
    _, q2, qy = standard_moments(stat, model["mean_x"], model["std_x"])
    coef = model["beta"].copy()
    if model["g"] is not None:
        coef[0] += model["g"][space.node_pos[node]]
    return max(float(stat["mean_y2"] - 2.0 * coef @ qy + coef @ q2 @ coef), 0.0)


def predict(df, features, model, space=None):
    Z = (df[features].to_numpy(float) - model["mean_x"]) / model["std_x"]
    pred = np.column_stack((np.ones(len(df)), Z)) @ model["beta"]
    if model["g"] is not None:
        pred += np.array([model["g"][space.node_pos[int(n)]] for n in df["node"]])
    return pred


def tune_linear(stats, outer_train):
    scores = {}
    for alpha in LINEAR_ALPHA_GRID:
        scores[alpha] = np.mean([
            node_mse(v, stats[v], fit_linear(stats, [n for n in outer_train if n != v], alpha))
            for v in outer_train
        ])
    return min(scores, key=scores.get)


def tune_graph(stats, outer_train, space):
    totals = {(lam, lb): 0.0 for lam in GRAPH_LAMBDA_GRID for lb in GRAPH_BETA_GRID}
    for v in outer_train:
        train = [n for n in outer_train if n != v]
        for lam in GRAPH_LAMBDA_GRID:
            prep = prepare_graph(stats, train, lam, space)
            for lb in GRAPH_BETA_GRID:
                totals[(lam, lb)] += node_mse(v, stats[v], solve_graph(prep, lb), space)
    scores = {k: t / len(outer_train) for k, t in totals.items()}
    return min(scores, key=scores.get)


def weighted_scores(pred_df, col):
    w = pred_df["node"].map(1.0 / pred_df.groupby("node").size()).to_numpy()
    y, p = pred_df["y_true"].to_numpy(), pred_df[col].to_numpy()
    return r2_score(y, p, sample_weight=w), np.sqrt(mean_squared_error(y, p, sample_weight=w))


def nested_loo(df, target, features, spaces, max_outer_folds=None):
    stats = make_node_stats(df, target, features)
    nodes = sorted(stats)
    outer = nodes if max_outer_folds is None else nodes[:max_outer_folds]
    parts, tuning = [], []

    for fold, test_node in enumerate(outer, 1):
        train = [n for n in nodes if n != test_node]
        test_df = df[df["node"] == test_node]
        fold_df = pd.DataFrame(dict(node=test_df["node"].to_numpy(),
                                    y_true=test_df[target].to_numpy()))

        alpha = tune_linear(stats, train)
        fold_df["linear"] = predict(test_df, features, fit_linear(stats, train, alpha))
        trow = dict(target=target, heldout_node=test_node, linear_alpha=alpha)

        for space in spaces:
            t0 = time.time()
            lam, lb = tune_graph(stats, train, space)
            model = solve_graph(prepare_graph(stats, train, lam, space), lb)
            fold_df[space.name] = predict(test_df, features, model, space)
            trow[f"{space.name}_lambda"] = lam
            trow[f"{space.name}_beta"] = lb
            print(f"    fold {fold:>2}/{len(outer)}  {space.name:<18} "
                  f"lambda={lam:<6} beta={lb:<5} ({time.time() - t0:.1f}s)", flush=True)

        parts.append(fold_df)
        tuning.append(trow)

    preds = pd.concat(parts, ignore_index=True)
    row = dict(Target=target, Nodes=len(nodes), OuterFoldsRun=len(outer))
    row["linear R2"], row["linear RMSE"] = weighted_scores(preds, "linear")
    for space in spaces:
        row[f"{space.name} R2"], row[f"{space.name} RMSE"] = weighted_scores(preds, space.name)
    return row, preds, pd.DataFrame(tuning)


# ============================================================================
# main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-r2", action="store_true", help="only do parts (a) and (b)")
    ap.add_argument("--targets", nargs="+", choices=list(TARGET_FEATURES), default=list(TARGET_FEATURES))
    ap.add_argument("--variants", nargs="+",
                    choices=["kron_uncompressed", "kron_compressed", "full_uncompressed", "full_compressed"],
                    default=["kron_uncompressed", "kron_compressed", "full_uncompressed", "full_compressed"])
    ap.add_argument("--max-outer-folds", type=int, default=None,
                    help="debug/smoke test: only run the first N outer folds per target "
                         "(R2 is then NOT comparable to the notebook)")
    args = ap.parse_args()

    t0 = time.time()
    G_full, G_master, wq_nodes, node_hourly = load_inputs()
    print(f"Loaded inputs in {time.time() - t0:.1f}s  |  WQ (boundary) nodes: {len(wq_nodes)}")

    # ---------------- (a) ----------------
    print("\n" + "=" * 100)
    print("(a) LAPLACIANS")
    print("=" * 100)
    graphs = {}
    for name, G in (("uncompressed", G_full), ("compressed", G_master)):
        L, order, index = distance_laplacian(G)
        missing = [n for n in wq_nodes if n not in index]
        if missing:
            raise ValueError(f"{name}: WQ nodes missing from graph: {missing}")
        b_idx = np.array([index[n] for n in wq_nodes])
        Lk, Lk_raw, parts = kron_reduce(L, b_idx)
        scale = kron_scale(Lk)
        graphs[name] = dict(L=L, index=index, b_idx=b_idx, L_kron=Lk,
                            L_kron_raw=Lk_raw, parts=parts, scale=scale)
        print(f"  {name:<13} full L: {L.shape[0]:>6} x {L.shape[1]:<6} nnz={L.nnz:<7} "
              f"| Kron L: {Lk.shape[0]} x {Lk.shape[1]}  scale={scale:.6g}")

    # ---------------- (b) ----------------
    checks = matrix_checks(graphs, wq_nodes)
    OUT_MATRIX.parent.mkdir(exist_ok=True)
    checks.to_csv(OUT_MATRIX, index=False)
    print(f"\nSaved {OUT_MATRIX.relative_to(BASE_DIR)}")

    if args.skip_r2:
        return

    # ---------------- (c) ----------------
    print("\n" + "=" * 100)
    print("(c) NESTED LOO R^2 WITH EACH LAPLACIAN")
    print("=" * 100)

    kron_pos = {n: i for i, n in enumerate(wq_nodes)}
    all_spaces = {}
    for name in ("uncompressed", "compressed"):
        g = graphs[name]
        all_spaces[f"kron_{name}"] = GraphSpace(
            f"kron_{name}", g["L_kron"] / g["scale"], kron_pos, np.arange(len(wq_nodes)))
        # full Laplacian divided by the SAME scale as its Kron reduction
        all_spaces[f"full_{name}"] = GraphSpace(
            f"full_{name}", g["L"] / g["scale"], {n: g["index"][n] for n in wq_nodes}, g["b_idx"])
    spaces = [all_spaces[v] for v in args.variants]

    results, tunings = [], []
    for target in args.targets:
        features = TARGET_FEATURES[target]
        df = node_hourly[["node", "time", target] + features].dropna().copy()
        print(f"\n  TARGET: {target}  (predictors: {features}, nodes: {df['node'].nunique()})")
        t1 = time.time()
        row, _, tuning = nested_loo(df, target, features, spaces, args.max_outer_folds)
        print(f"  done in {time.time() - t1:.1f}s")
        results.append(row)
        tunings.append(tuning)

    res = pd.DataFrame(results)
    tun = pd.concat(tunings, ignore_index=True)

    ref = spaces[0].name
    for s in spaces[1:]:
        res[f"{s.name} - {ref} R2"] = res[f"{s.name} R2"] - res[f"{ref} R2"]
        agree = tun.groupby("target").apply(
            lambda t, s=s: ((t[f"{s.name}_lambda"] == t[f"{ref}_lambda"])
                            & (t[f"{s.name}_beta"] == t[f"{ref}_beta"])).mean(),
            include_groups=False)
        res[f"{s.name} same hyperparams as {ref}"] = res["Target"].map(agree)

    if PREVIOUS_RESULTS_FILE.exists() and args.max_outer_folds is None:
        prev = pd.read_csv(PREVIOUS_RESULTS_FILE).set_index("Target")
        res["notebook Graph R2"] = res["Target"].map(prev["Graph R2"])
        res["notebook Linear R2"] = res["Target"].map(prev["Linear R2"])

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    r2_cols = ["Target", "Nodes", "linear R2"] + [f"{s.name} R2" for s in spaces]
    r2_cols += [c for c in ("notebook Linear R2", "notebook Graph R2") if c in res]
    print("\nR^2 by Laplacian:")
    print(res[r2_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    diff_cols = ["Target"] + [c for c in res if " - " in c or "same hyperparams" in c]
    if len(diff_cols) > 1:
        print("\nDifferences vs", ref)
        print(res[diff_cols].to_string(index=False, float_format=lambda x: f"{x:.2e}"))

    if args.max_outer_folds is not None:
        print("\n(--max-outer-folds was set: smoke test only, results not saved)")
        return
    res.to_csv(OUT_R2, index=False)
    print(f"\nSaved {OUT_R2.relative_to(BASE_DIR)}   (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
