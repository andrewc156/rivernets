# Imports
from pathlib import Path
import json
import os
import pickle
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42

BENCHMARK_DIR = Path("data/benchmark_data")
# Feature selection
#   inner (default): in every outer fold, each model picks its feature set
#       from FEATURE_CANDIDATES (see "Feature Candidates") by inner-LOO MSE
#       only, so the held-out node never influences which features are used.
#   fixed: every model uses the hand-picked TARGET_CONFIGS features
#       (the Nov 5 benchmark).
# BENCHMARK_FS_K / BENCHMARK_FS_STATIC set the candidate grid.
# BENCHMARK_FS_SEARCH:
#   two_stage (default): pick the feature set at a reference setting of each
#       model, then tune the model's settings on that set. Cost is about
#       (candidates + settings) inner-LOO passes instead of their product.
#   joint: search every (feature set, settings) pair.
# BENCHMARK_INNER_STRIDE=k fits the inner-loop sklearn models on every k-th
#   hour of each node (outer refits always use all rows). 1 = exact.
FEATURE_SELECTION = os.environ.get("BENCHMARK_FEATURE_SELECTION", "inner")
FS_K_VALUES = [int(k) for k in os.environ.get("BENCHMARK_FS_K", "1,2,4").split(",") if k]
FS_STATIC_OPTIONS = [
    s for s in os.environ.get("BENCHMARK_FS_STATIC", "none,local_slope,reduced").split(",") if s
]
FS_SEARCH = os.environ.get("BENCHMARK_FS_SEARCH", "two_stage")
INNER_STRIDE = int(os.environ.get("BENCHMARK_INNER_STRIDE", "1"))

if FEATURE_SELECTION not in ("inner", "fixed"):
    raise ValueError("BENCHMARK_FEATURE_SELECTION must be 'inner' or 'fixed'")
if FS_SEARCH not in ("two_stage", "joint"):
    raise ValueError("BENCHMARK_FS_SEARCH must be 'two_stage' or 'joint'")
if INNER_STRIDE < 1:
    raise ValueError("BENCHMARK_INNER_STRIDE must be >= 1")

RESULTS_DIR = Path(
    os.environ.get(
        "BENCHMARK_RESULTS_DIR",
        "results/benchmark_results"
        if FEATURE_SELECTION == "fixed"
        else "results/benchmark_results_inner_fs"
    )
)

MASTER_GRAPH_FILE = Path("graphs/colorado/master_compressed_river_graph.pkl")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Optional run subsets for quick checks, e.g.
#   BENCHMARK_TARGETS="pH" BENCHMARK_MODELS="Graph-Laplacian,OLS" \
#   BENCHMARK_RESULTS_DIR=/tmp/bench python water_quality_model_benchmark.py
# Leave them unset for the full benchmark.
RUN_TARGETS = [t for t in os.environ.get("BENCHMARK_TARGETS", "").split(",") if t]
RUN_MODELS = [m for m in os.environ.get("BENCHMARK_MODELS", "").split(",") if m]

print("Benchmark data:", BENCHMARK_DIR.resolve())
print("Results:", RESULTS_DIR.resolve())
print(
    "Feature selection:", FEATURE_SELECTION,
    "| search:", FS_SEARCH,
    "| K:", FS_K_VALUES,
    "| static:", FS_STATIC_OPTIONS,
    "| inner stride:", INNER_STRIDE
)


# Check Files
required_files = [
    "all_static_df.pkl",
    "fourier_interaction_df.pkl",
    "static_node_df.pkl"
]

missing_files = []

for filename in required_files:
    path = BENCHMARK_DIR / filename

    if path.exists():
        print(filename, "FOUND")
    else:
        print(filename, "MISSING")
        missing_files.append(filename)

if missing_files:
    raise FileNotFoundError(
        "Missing files: " + ", ".join(missing_files)
    )


# Load Data
all_static_df = pd.read_pickle(
    BENCHMARK_DIR / "all_static_df.pkl"
)

fourier_interaction_df = pd.read_pickle(
    BENCHMARK_DIR / "fourier_interaction_df.pkl"
)

static_node_df = pd.read_pickle(
    BENCHMARK_DIR / "static_node_df.pkl"
)

print("all_static_df:", all_static_df.shape)
print("fourier_interaction_df:", fourier_interaction_df.shape)
print("static_node_df:", static_node_df.shape)


# Graph Settings
GRAPH_LAMBDA_GRID = None
GRAPH_BETA_GRID = None

settings_path = BENCHMARK_DIR / "graph_settings.json"

if settings_path.exists():
    with open(settings_path, "r") as f:
        graph_settings = json.load(f)

    GRAPH_LAMBDA_GRID = graph_settings.get("GRAPH_LAMBDA_GRID")
    GRAPH_BETA_GRID = graph_settings.get("GRAPH_BETA_GRID")

    print("Graph settings loaded.")
else:
    print("No graph settings file found.")


# Graph-Laplacian Model
#
# Fitted inside benchmark_target_nested_loo, on exactly the same rows,
# predictors and outer/inner folds as every other model.
#
#   y_it = b0 + z_it' b + g_i
#   minimise  sum_i (1/n_i) sum_t (y_it - b0 - z_it' b - g_i)^2
#             + lambda_beta * |b|^2 + lambda * g' L_eff g,   sum_i g_i = 0
#
# L_eff is one of the five scaled Kron-reduced Laplacians saved by
# largest_uncompressed_laplacian.ipynb (effective_laplacians_clean.pkl), or the
# unweighted compressed-graph Laplacian built below.
# The weighting scheme, lambda and lambda_beta are tuned by the inner LOO.
# Kron reduction is exact: refitting with the full 67,401-node Laplacian
# gives the same R2 to 1e-10 (reports/kron_vs_full_laplacian_report.tex).
GRAPH_MODEL_NAME = "Graph-Laplacian"

if GRAPH_LAMBDA_GRID is None or GRAPH_BETA_GRID is None:
    raise FileNotFoundError(
        "graph_settings.json is needed for the Graph-Laplacian model."
    )

with open(BENCHMARK_DIR / "effective_laplacians_clean.pkl", "rb") as f:
    EFFECTIVE_LAPLACIANS = {
        name: np.asarray(L, dtype=float)
        for name, L in pickle.load(f).items()
    }

# Rows/columns of every effective Laplacian follow the sorted WQ node ids
# (boundary_nodes in largest_uncompressed_laplacian.ipynb).
with open(MASTER_GRAPH_FILE, "rb") as f:
    _G_master = pickle.load(f)

GRAPH_NODES = sorted(
    node
    for node, data in _G_master.nodes(data=True)
    if data.get("usgs_station_count", 0) > 0
    or len(data.get("usgs_clusters", [])) > 0
)
GRAPH_NODE_INDEX = {int(node): i for i, node in enumerate(GRAPH_NODES)}

for _name, _L in EFFECTIVE_LAPLACIANS.items():
    if _L.shape != (len(GRAPH_NODES), len(GRAPH_NODES)):
        raise ValueError(f"Laplacian {_name} does not match the {len(GRAPH_NODES)} WQ nodes.")


def compressed_kron_laplacian(edge_weight):
    """Kron reduction of the 67-node compressed graph onto the WQ nodes, with the
    notebook's clean-up (symmetrise, zero row sums) and scaling (mean positive
    diagonal = 1). edge_weight(data) gives each compressed edge's weight."""
    order = list(_G_master.nodes())
    pos = {n: i for i, n in enumerate(order)}
    L = np.zeros((len(order), len(order)))
    for u, v, data in _G_master.to_undirected(as_view=True).edges(data=True):
        w = edge_weight(data)
        i, j = pos[u], pos[v]
        L[i, j] -= w
        L[j, i] -= w
        L[i, i] += w
        L[j, j] += w
    b = np.array([pos[n] for n in GRAPH_NODES])
    interior = np.setdiff1d(np.arange(len(order)), b)
    Lk = L[np.ix_(b, b)] - L[np.ix_(b, interior)] @ np.linalg.solve(
        L[np.ix_(interior, interior)], L[np.ix_(interior, b)]
    )
    Lk = (Lk + Lk.T) / 2.0
    np.fill_diagonal(Lk, 0.0)
    np.fill_diagonal(Lk, -Lk.sum(axis=1))
    d = np.diag(Lk)
    return Lk / d[d > 1e-12].mean()


# Check: rebuilding distance_only this way reproduces the saved matrix, so the
# node order and scaling of the new Laplacian below match the saved ones.
_check = compressed_kron_laplacian(lambda data: 1000.0 / float(data["length_m"]))
if not np.allclose(_check, EFFECTIVE_LAPLACIANS["distance_only"], rtol=1e-8, atol=1e-10):
    raise ValueError("Rebuilt distance_only Laplacian does not match the saved one.")

# unweighted_compressed: every compressed reach has weight 1, whatever its
# length or number of original segments, so the graph distance between two
# stations is the number of reaches between them. Without weights the original
# and compressed graphs are NOT equivalent; unweighted on the original 67k graph
# is almost identical to distance_only (compare_edge_weights.py).
EFFECTIVE_LAPLACIANS["unweighted_compressed"] = compressed_kron_laplacian(lambda data: 1.0)

# BENCHMARK_GRAPH_WEIGHTINGS restricts which weightings the inner loop may pick.
_weightings = [w for w in os.environ.get("BENCHMARK_GRAPH_WEIGHTINGS", "").split(",") if w]
if _weightings:
    _unknown = [w for w in _weightings if w not in EFFECTIVE_LAPLACIANS]
    if _unknown:
        raise ValueError(f"Unknown BENCHMARK_GRAPH_WEIGHTINGS: {_unknown}")
    EFFECTIVE_LAPLACIANS = {w: EFFECTIVE_LAPLACIANS[w] for w in _weightings}

# Lambda grid: the saved grid plus smaller values. With the unweighted
# compressed Laplacian the inner loop chose the smallest saved value (0.01) in
# most folds, so the grid is extended downward. BENCHMARK_GRAPH_LAMBDA_GRID
# overrides it, e.g. "0.01,0.03,0.1,0.3,1,3,10,30,100" for the saved grid only.
_lambda_env = os.environ.get("BENCHMARK_GRAPH_LAMBDA_GRID", "")
if _lambda_env:
    GRAPH_LAMBDA_GRID = [float(x) for x in _lambda_env.split(",") if x]
else:
    GRAPH_LAMBDA_GRID = sorted({0.0001, 0.001, 0.003, *map(float, GRAPH_LAMBDA_GRID)})
print("Graph lambda grid:", GRAPH_LAMBDA_GRID)

print(
    "Graph-Laplacian:",
    len(EFFECTIVE_LAPLACIANS), "weightings", list(EFFECTIVE_LAPLACIANS), "|",
    len(GRAPH_NODES), "WQ nodes"
)


def graph_node_stats(df, target, features):
    """Per-node moments; every graph fit and MSE below is exact from these."""
    stats = {}
    for node, group in df.groupby("node"):
        X = group[features].to_numpy(dtype=float)
        y = group[target].to_numpy(dtype=float)
        n = len(group)
        stats[int(node)] = {
            "mean_x": X.mean(axis=0),
            "exx": X.T @ X / n,
            "exy": X.T @ y / n,
            "mean_y": y.mean(),
            "mean_y2": np.mean(y ** 2),
        }
    return stats


def _graph_standard_moments(stat, mean_x, std_x):
    mx, exx, exy, my = stat["mean_x"], stat["exx"], stat["exy"], stat["mean_y"]
    zmean = (mx - mean_x) / std_x
    zexx = (
        exx - np.outer(mx, mean_x) - np.outer(mean_x, mx) + np.outer(mean_x, mean_x)
    ) / np.outer(std_x, std_x)
    zexy = (exy - mean_x * my) / std_x
    p = len(mean_x)
    q2 = np.empty((p + 1, p + 1))
    q2[0, 0] = 1.0
    q2[0, 1:] = q2[1:, 0] = zmean
    q2[1:, 1:] = zexx
    return np.concatenate(([1.0], zmean)), q2, np.concatenate(([my], zexy))


def _graph_train_moments(stats, train_nodes):
    # standardise with equal weight per training node
    mean_x = np.mean([stats[n]["mean_x"] for n in train_nodes], axis=0)
    ex2 = np.mean([np.diag(stats[n]["exx"]) for n in train_nodes], axis=0)
    std_x = np.sqrt(np.maximum(ex2 - mean_x ** 2, 1e-12))
    q = len(mean_x) + 1
    A11, rb, qmean = np.zeros((q, q)), np.zeros(q), {}
    for n in train_nodes:
        qm, q2, qy = _graph_standard_moments(stats[n], mean_x, std_x)
        A11 += q2
        rb += qy
        qmean[n] = qm
    return {
        "mean_x": mean_x, "std_x": std_x, "q": q, "A11": A11, "rb": rb,
        "qmean": qmean, "mean_y": {n: stats[n]["mean_y"] for n in train_nodes},
    }


def _graph_prepare(tm, L, lam):
    """Eliminate g from the full system with one solve of diag(mask) + lam*L,
    shared across the lambda_beta grid."""
    N, q = L.shape[0], tm["q"]
    Bt, rg, mask = np.zeros((N, q)), np.zeros(N), np.zeros(N)
    for n, qm in tm["qmean"].items():
        j = GRAPH_NODE_INDEX[n]
        Bt[j] += qm
        mask[j] = 1.0
        rg[j] = tm["mean_y"][n]
    c = np.ones(N)                                   # gauge: sum of g = 0
    sol = np.linalg.solve(np.diag(mask) + lam * L, np.column_stack((Bt, rg, c)))
    X, u, v = sol[:, :q], sol[:, q], sol[:, q + 1]
    return {"tm": tm, "X": X, "u": u, "v": v, "BtX": Bt.T @ X, "Btu": Bt.T @ u,
            "Btv": Bt.T @ v, "cX": c @ X, "cu": c @ u, "cv": c @ v}


def _graph_solve(prep, lambda_beta):
    tm, q = prep["tm"], prep["tm"]["q"]
    ridge = np.eye(q)
    ridge[0, 0] = 0.0                                # intercept not penalised
    S = np.empty((q + 1, q + 1))
    S[:q, :q] = tm["A11"] + lambda_beta * ridge - prep["BtX"]
    S[:q, q] = -prep["Btv"]
    S[q, :q] = prep["cX"]
    S[q, q] = prep["cv"]
    z = np.linalg.solve(S, np.concatenate((tm["rb"] - prep["Btu"], [prep["cu"]])))
    beta, mu = z[:q], z[q]
    return {"beta": beta, "g": prep["u"] - prep["X"] @ beta - prep["v"] * mu,
            "mean_x": tm["mean_x"], "std_x": tm["std_x"]}


def _graph_coef(model, node):
    coef = model["beta"].copy()
    coef[0] += model["g"][GRAPH_NODE_INDEX[int(node)]]
    return coef


def _graph_node_mse(stat, model, node):
    _, q2, qy = _graph_standard_moments(stat, model["mean_x"], model["std_x"])
    coef = _graph_coef(model, node)
    return max(float(stat["mean_y2"] - 2.0 * coef @ qy + coef @ q2 @ coef), 0.0)


def _graph_inner_scores(stats):
    """Mean per-node inner-LOO MSE for every (scheme, lambda, lambda_beta)."""
    train_nodes = sorted(stats)
    totals = defaultdict(float)
    for val_node in train_nodes:
        tm = _graph_train_moments(stats, [n for n in train_nodes if n != val_node])
        for scheme, L in EFFECTIVE_LAPLACIANS.items():
            for lam in GRAPH_LAMBDA_GRID:
                prep = _graph_prepare(tm, L, float(lam))
                for lb in GRAPH_BETA_GRID:
                    model = _graph_solve(prep, float(lb))
                    totals[(scheme, float(lam), float(lb))] += _graph_node_mse(
                        stats[val_node], model, val_node
                    )
    return {k: t / len(train_nodes) for k, t in totals.items()}


def fit_predict_graph_laplacian(outer_train, outer_test, target, candidates):
    """Inner-LOO choice of (feature set, weighting, lambda, lambda_beta), refit
    on the outer training nodes, predict the held-out node. Same selection
    criterion as tune_model_inner_loo: the mean of the per-node validation MSE.
    The search is always joint: the graph model is solved exactly from per-node
    moments, so it is cheap."""
    best = None
    for name, features in candidates.items():
        stats = graph_node_stats(outer_train, target, features)
        scores = _graph_inner_scores(stats)
        key = min(scores, key=scores.get)
        if best is None or scores[key] < best[0]:
            best = (scores[key], name, key, stats)
    inner_mse, name, (scheme, lam, lb), stats = best
    features = candidates[name]
    model = _graph_solve(
        _graph_prepare(_graph_train_moments(stats, sorted(stats)), EFFECTIVE_LAPLACIANS[scheme], lam), lb
    )
    test_node = outer_test["node"].iloc[0]
    Z = (outer_test[features].to_numpy(dtype=float) - model["mean_x"]) / model["std_x"]
    pred = np.column_stack((np.ones(len(Z)), Z)) @ _graph_coef(model, test_node)
    best_params = {"scheme": scheme, "lambda": lam, "lambda_beta": lb}
    return name, best_params, inner_mse, pred


# Fourier Helpers
def fourier_features(K):
    cols = []

    for k in range(1, K + 1):
        cols.extend([
            f"hour_sin_{k}",
            f"hour_cos_{k}",
            f"doy_sin_{k}",
            f"doy_cos_{k}"
        ])

    return cols


def separate_fourier_features(K_day, K_year):
    cols = []

    for k in range(1, K_day + 1):
        cols.extend([
            f"hour_sin_{k}",
            f"hour_cos_{k}"
        ])

    for k in range(1, K_year + 1):
        cols.extend([
            f"doy_sin_{k}",
            f"doy_cos_{k}"
        ])

    return cols


# Temperature Interactions
TEMP_K4_MAIN = separate_fourier_features(
    K_day=4,
    K_year=4
)

base_cols = set(all_static_df.columns)

extra_cols = [
    c
    for c in fourier_interaction_df.columns
    if c not in base_cols
    and c not in [
        "node",
        "Water temperature"
    ]
]

TEMP_K4_INTERACTION = [
    c
    for c in extra_cols
    if (
        (
            "hour" in c.lower()
            and (
                "doy" in c.lower()
                or "year" in c.lower()
            )
        )
        or "interaction" in c.lower()
    )
]

if len(TEMP_K4_INTERACTION) == 0:
    TEMP_K4_INTERACTION = extra_cols

print(
    "Detected temperature interaction terms:",
    len(TEMP_K4_INTERACTION)
)

for c in TEMP_K4_INTERACTION:
    print(c)


# Temperature Data
REDUCED_STATIC = [
    "elevation_m",
    "log_upa",
    "local_slope"
]

temp_benchmark_df = fourier_interaction_df.copy()

ALL_STATIC = [
    "elevation_m",
    "log_upa",
    "log_width",
    "local_slope"
]

missing_static = [
    c
    for c in ALL_STATIC
    if c not in temp_benchmark_df.columns
]

if missing_static:
    temp_benchmark_df = temp_benchmark_df.merge(
        static_node_df[
            ["node", *missing_static]
        ],
        on="node",
        how="left"
    )

print("Temperature dataframe:", temp_benchmark_df.shape)


# Features
TEMP_FEATURES = [
    "Specific conductance",
    *TEMP_K4_MAIN,
    *TEMP_K4_INTERACTION,
    "elevation_m",
    "log_upa",
    "local_slope"
]

DO_FEATURES = [
    "Water temperature",
    "Specific conductance",
    *fourier_features(2),
    "elevation_m",
    "log_upa",
    "local_slope"
]

COND_FEATURES = [
    "Water temperature",
    "local_slope",
    *separate_fourier_features(
        K_day=1,
        K_year=4
    )
]

PH_FEATURES = [
    "Specific conductance",
    *fourier_features(1)
]

TURB_FEATURES = [
    "local_slope",
    *fourier_features(1)
]


# Target Configs
TARGET_CONFIGS = {
    "Water temperature": {
        "df": temp_benchmark_df,
        "features": TEMP_FEATURES
    },

    "DO concentration": {
        "df": all_static_df,
        "features": DO_FEATURES
    },

    "Specific conductance": {
        "df": all_static_df,
        "features": COND_FEATURES
    },

    "pH": {
        "df": all_static_df,
        "features": PH_FEATURES
    },

    "Turbidity": {
        "df": all_static_df,
        "features": TURB_FEATURES
    }
}


# Feature Candidates
# The chemistry predictors stay fixed per target: they are the only columns
# with missing values, so every candidate uses exactly the same rows as the
# Nov 5 benchmark. The seasonal (Fourier K) and static MERIT parts are chosen
# inside the inner loop. "configured" (the TARGET_CONFIGS set) is always a
# candidate. hand_m is never used: it is 0 at every WQ node.
TARGET_CHEMISTRY = {
    "Water temperature": ["Specific conductance"],
    "DO concentration": ["Water temperature", "Specific conductance"],
    "Specific conductance": ["Water temperature"],
    "pH": ["Specific conductance"],
    "Turbidity": []
}

STATIC_SETS = {
    "none": [],
    "local_slope": ["local_slope"],
    "reduced": REDUCED_STATIC,
    "all": ALL_STATIC
}

unknown_static = [s for s in FS_STATIC_OPTIONS if s not in STATIC_SETS]
if unknown_static:
    raise ValueError(f"Unknown BENCHMARK_FS_STATIC options: {unknown_static}")


def feature_candidates(target, cfg):
    candidates = {"configured": list(cfg["features"])}

    if FEATURE_SELECTION == "fixed":
        return candidates

    chemistry = TARGET_CHEMISTRY[target]
    if not set(chemistry) <= set(cfg["features"]):
        raise ValueError(f"{target}: chemistry predictors not in TARGET_CONFIGS")

    for K in FS_K_VALUES:
        for static in FS_STATIC_OPTIONS:
            features = [
                *chemistry,
                *fourier_features(K),
                *STATIC_SETS[static]
            ]
            if any(set(features) == set(f) for f in candidates.values()):
                continue
            candidates[f"K={K} | {static}"] = features

    return candidates


for target, cfg in TARGET_CONFIGS.items():
    cfg["candidates"] = feature_candidates(target, cfg)


# Verify Features
all_ok = True

for target, cfg in TARGET_CONFIGS.items():
    union = sorted({
        f
        for features in cfg["candidates"].values()
        for f in features
    })

    required = [
        "node",
        target,
        *union
    ]

    missing = [
        c
        for c in required
        if c not in cfg["df"].columns
    ]

    if missing:
        all_ok = False
        print(
            target,
            "MISSING:",
            missing
        )
    else:
        print(
            target,
            "OK",
            "| feature candidates:",
            len(cfg["candidates"]),
            "| predictors in union:",
            len(union)
        )

if not all_ok:
    raise ValueError(
        "Fix missing columns before continuing."
    )


# Model Grids
if GRAPH_BETA_GRID is not None:
    RIDGE_ALPHAS = [
        float(x)
        for x in GRAPH_BETA_GRID
        if float(x) > 0
    ]
else:
    RIDGE_ALPHAS = [
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0
    ]

MODEL_CANDIDATES = {
    "Dummy": [
        {}
    ],

    "OLS": [
        {}
    ],

    "Ridge": [
        {
            "alpha": alpha
        }
        for alpha in RIDGE_ALPHAS
    ],

    "Elastic Net": [
        {
            "alpha": 0.001,
            "l1_ratio": 0.25
        },
        {
            "alpha": 0.01,
            "l1_ratio": 0.50
        },
        {
            "alpha": 0.10,
            "l1_ratio": 0.50
        },
        {
            "alpha": 1.00,
            "l1_ratio": 0.75
        }
    ],

    "Spline GAM": [
        {
            "n_knots": 4,
            "alpha": 0.1
        },
        {
            "n_knots": 4,
            "alpha": 1.0
        },
        {
            "n_knots": 6,
            "alpha": 1.0
        },
        {
            "n_knots": 6,
            "alpha": 10.0
        }
    ],

    "Random Forest": [
        {
            "n_estimators": 250,
            "max_depth": 8,
            "min_samples_leaf": 5
        },
        {
            "n_estimators": 250,
            "max_depth": 16,
            "min_samples_leaf": 5
        },
        {
            "n_estimators": 250,
            "max_depth": None,
            "min_samples_leaf": 10
        },
        {
            "n_estimators": 250,
            "max_depth": None,
            "min_samples_leaf": 20
        }
    ],

    "Gradient Boosting": [
        {
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 15,
            "l2_regularization": 0.0
        },
        {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0
        },
        {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0
        },
        {
            "learning_rate": 0.10,
            "max_iter": 200,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0
        }
    ]
}

print("Ridge grid:", RIDGE_ALPHAS)


# Model Builder
def build_benchmark_model(
    model_name,
    params
):

    if model_name == "Dummy":
        return Pipeline([
            (
                "model",
                DummyRegressor(
                    strategy="mean"
                )
            )
        ])

    if model_name == "OLS":
        return Pipeline([
            (
                "scale",
                StandardScaler()
            ),
            (
                "model",
                LinearRegression()
            )
        ])

    if model_name == "Ridge":
        return Pipeline([
            (
                "scale",
                StandardScaler()
            ),
            (
                "model",
                Ridge(
                    alpha=params["alpha"]
                )
            )
        ])

    if model_name == "Elastic Net":
        return Pipeline([
            (
                "scale",
                StandardScaler()
            ),
            (
                "model",
                ElasticNet(
                    alpha=params["alpha"],
                    l1_ratio=params["l1_ratio"],
                    max_iter=20000,
                    random_state=RANDOM_STATE
                )
            )
        ])

    if model_name == "Spline GAM":
        return Pipeline([
            (
                "splines",
                SplineTransformer(
                    n_knots=params["n_knots"],
                    degree=3,
                    include_bias=False
                )
            ),
            (
                "scale",
                StandardScaler()
            ),
            (
                "model",
                Ridge(
                    alpha=params["alpha"]
                )
            )
        ])

    if model_name == "Random Forest":
        return Pipeline([
            (
                "model",
                RandomForestRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ])

    if model_name == "Gradient Boosting":
        return Pipeline([
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=params["learning_rate"],
                    max_iter=params["max_iter"],
                    max_leaf_nodes=params["max_leaf_nodes"],
                    l2_regularization=params["l2_regularization"],
                    random_state=RANDOM_STATE
                )
            )
        ])

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# Equal Node Weights
def equal_node_weights(
    df,
    node_col="node"
):

    counts = (
        df
        .groupby(node_col)[node_col]
        .transform("size")
        .astype(float)
    )

    return (
        1.0 / counts
    ).to_numpy()


# Inner LOO Tuning
def tune_model_inner_loo(
    outer_train,
    target,
    features,
    model_name,
    node_col="node",
    params_list=None
):

    if params_list is None:
        params_list = MODEL_CANDIDATES[model_name]

    train_nodes = (
        outer_train[node_col]
        .drop_duplicates()
        .tolist()
    )

    best_score = np.inf
    best_params = None

    for params in params_list:

        node_mse = []

        for val_node in train_nodes:

            inner_train = outer_train[
                outer_train[node_col]
                != val_node
            ]

            inner_val = outer_train[
                outer_train[node_col]
                == val_node
            ]

            X_train = (
                inner_train[
                    features
                ]
                .to_numpy()
            )

            y_train = (
                inner_train[
                    target
                ]
                .to_numpy()
            )

            X_val = (
                inner_val[
                    features
                ]
                .to_numpy()
            )

            y_val = (
                inner_val[
                    target
                ]
                .to_numpy()
            )

            train_weights = (
                equal_node_weights(
                    inner_train,
                    node_col=node_col
                )
            )

            model = (
                build_benchmark_model(
                    model_name,
                    params
                )
            )

            model.fit(
                X_train,
                y_train,
                model__sample_weight=
                    train_weights
            )

            pred = (
                model.predict(
                    X_val
                )
            )

            mse = (
                np.mean(
                    (
                        y_val
                        - pred
                    ) ** 2
                )
            )

            node_mse.append(
                mse
            )

        mean_node_mse = (
            np.mean(
                node_mse
            )
        )

        if (
            mean_node_mse
            < best_score
        ):

            best_score = (
                mean_node_mse
            )

            best_params = (
                params.copy()
            )

    return (
        best_params,
        best_score
    )


# Inner LOO Feature Selection
def thin_rows(df, stride, node_col="node"):
    """Every stride-th row of each node (rows are in time order per node)."""
    if stride == 1:
        return df
    return df[df.groupby(node_col).cumcount() % stride == 0]


def select_and_tune_inner_loo(
    outer_train,
    target,
    candidates,
    model_name,
    node_col="node"
):
    """Choose (feature set, settings) by inner-LOO node MSE only.
    Returns (feature set name, settings, inner MSE)."""

    tune_df = thin_rows(outer_train, INNER_STRIDE, node_col)
    grid = MODEL_CANDIDATES[model_name]

    # The dummy model ignores its inputs, so the feature set is irrelevant.
    if model_name == "Dummy":
        candidates = {"configured": candidates["configured"]}

    def tune(name, params_list):
        return tune_model_inner_loo(
            outer_train=tune_df,
            target=target,
            features=candidates[name],
            model_name=model_name,
            node_col=node_col,
            params_list=params_list
        )

    if len(candidates) == 1 or len(grid) == 1 or FS_SEARCH == "joint":
        best = None
        for name in candidates:
            params, score = tune(name, grid)
            if best is None or score < best[2]:
                best = (name, params, score)
        return best

    # two_stage: features at the middle setting of the grid, then settings
    reference = grid[len(grid) // 2]
    stage1 = {name: tune(name, [reference])[1] for name in candidates}
    name = min(stage1, key=stage1.get)
    others = [p for p in grid if p is not reference]
    params, score = tune(name, others)
    if stage1[name] <= score:
        params, score = reference.copy(), stage1[name]
    return name, params, score


# Nested Outer LOO
def benchmark_target_nested_loo(
    df,
    target,
    candidates,
    node_col="node",
    model_names=None
):

    if model_names is None:

        model_names = [
            "Dummy",
            "OLS",
            "Ridge",
            "Elastic Net",
            "Spline GAM",
            "Random Forest",
            "Gradient Boosting",
            GRAPH_MODEL_NAME
        ]

    # every candidate is fitted on the same rows
    features_union = sorted({
        f
        for features in candidates.values()
        for f in features
    })

    required = [
        node_col,
        target,
        *features_union
    ]

    work = (
        df[
            required
        ]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .dropna()
        .copy()
    )

    nodes = (
        work[
            node_col
        ]
        .drop_duplicates()
        .tolist()
    )

    print(
        "\nTarget:",
        target
    )

    print(
        "Nodes:",
        len(nodes)
    )

    print(
        "Rows:",
        f"{len(work):,}"
    )

    print(
        "Feature candidates:",
        len(candidates)
    )

    prediction_rows = []
    tuning_rows = []

    for (
        fold_number,
        test_node
    ) in enumerate(
        nodes,
        start=1
    ):

        print(
            f"\rOuter fold "
            f"{fold_number}/"
            f"{len(nodes)} "
            f"| node "
            f"{test_node}",
            end=""
        )

        outer_train = (
            work[
                work[
                    node_col
                ]
                != test_node
            ]
        )

        outer_test = (
            work[
                work[
                    node_col
                ]
                == test_node
            ]
        )

        y_test = (
            outer_test[
                target
            ]
            .to_numpy()
        )

        for model_name in model_names:

            if model_name == GRAPH_MODEL_NAME:

                (
                    feature_set,
                    best_params,
                    inner_mse,
                    pred
                ) = fit_predict_graph_laplacian(
                    outer_train,
                    outer_test,
                    target,
                    candidates
                )

            else:

                (
                    feature_set,
                    best_params,
                    inner_mse
                ) = (
                    select_and_tune_inner_loo(
                        outer_train=
                            outer_train,
                        target=
                            target,
                        candidates=
                            candidates,
                        model_name=
                            model_name,
                        node_col=
                            node_col
                    )
                )

                features = candidates[
                    feature_set
                ]

                X_test = (
                    outer_test[
                        features
                    ]
                    .to_numpy()
                )

                X_train = (
                    outer_train[
                        features
                    ]
                    .to_numpy()
                )

                y_train = (
                    outer_train[
                        target
                    ]
                    .to_numpy()
                )

                train_weights = (
                    equal_node_weights(
                        outer_train,
                        node_col=
                            node_col
                    )
                )

                model = (
                    build_benchmark_model(
                        model_name,
                        best_params
                    )
                )

                model.fit(
                    X_train,
                    y_train,
                    model__sample_weight=
                        train_weights
                )

                pred = (
                    model.predict(
                        X_test
                    )
                )

            for (
                row_index,
                y_true,
                y_pred
            ) in zip(
                outer_test.index,
                y_test,
                pred
            ):

                prediction_rows.append({
                    "target":
                        target,
                    "model":
                        model_name,
                    "node":
                        test_node,
                    "row_index":
                        row_index,
                    "y_true":
                        y_true,
                    "y_pred":
                        y_pred
                })

            tuning_rows.append({
                "target":
                    target,
                "outer_test_node":
                    test_node,
                "model":
                    model_name,
                "features":
                    feature_set,
                "best_params":
                    best_params,
                "inner_node_mse":
                    inner_mse
            })

    print()

    predictions = (
        pd.DataFrame(
            prediction_rows
        )
    )

    tuning = (
        pd.DataFrame(
            tuning_rows
        )
    )

    metric_rows = []

    for model_name in model_names:

        p = (
            predictions[
                predictions[
                    "model"
                ]
                == model_name
            ]
            .copy()
        )

        weights = (
            equal_node_weights(
                p,
                node_col="node"
            )
        )

        r2 = (
            r2_score(
                p["y_true"],
                p["y_pred"],
                sample_weight=
                    weights
            )
        )

        rmse = (
            np.sqrt(
                mean_squared_error(
                    p["y_true"],
                    p["y_pred"],
                    sample_weight=
                        weights
                )
            )
        )

        metric_rows.append({
            "Target":
                target,
            "Model":
                model_name,
            "Nodes":
                p[
                    "node"
                ].nunique(),
            "Rows":
                len(p),
            "R2":
                r2,
            "RMSE":
                rmse
        })

    results = (
        pd.DataFrame(
            metric_rows
        )
        .sort_values(
            "R2",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return (
        results,
        predictions,
        tuning
    )


# Run Benchmarks
if RUN_TARGETS:
    TARGET_CONFIGS = {t: TARGET_CONFIGS[t] for t in RUN_TARGETS}

MODEL_NAMES = RUN_MODELS or [*MODEL_CANDIDATES, GRAPH_MODEL_NAME]

print("Targets:", list(TARGET_CONFIGS))
print("Models:", MODEL_NAMES)

all_benchmark_results = []
benchmark_predictions = {}
benchmark_tuning = {}

for target, cfg in TARGET_CONFIGS.items():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STARTING:",
        target
    )

    print(
        "=" * 80
    )

    (
        results,
        predictions,
        tuning
    ) = (
        benchmark_target_nested_loo(
            df=
                cfg["df"],
            target=
                target,
            candidates=
                cfg["candidates"],
            model_names=
                MODEL_NAMES
        )
    )

    all_benchmark_results.append(
        results
    )

    benchmark_predictions[
        target
    ] = predictions

    benchmark_tuning[
        target
    ] = tuning

    safe_name = (
        target
        .lower()
        .replace(
            " ",
            "_"
        )
    )

    results.to_csv(
        RESULTS_DIR
        / f"{safe_name}_metrics.csv",
        index=False
    )

    predictions.to_pickle(
        RESULTS_DIR
        / f"{safe_name}_predictions.pkl"
    )

    tuning.to_pickle(
        RESULTS_DIR
        / f"{safe_name}_tuning.pkl"
    )

    print(
        "\nSaved:",
        target
    )

    print(
        results.to_string(
            index=False
        )
    )


# Combine Results
benchmark_results = (
    pd.concat(
        all_benchmark_results,
        ignore_index=True
    )
)

benchmark_results = (
    benchmark_results
    .sort_values(
        [
            "Target",
            "R2"
        ],
        ascending=[
            True,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)

print(
    benchmark_results.to_string(
        index=False
    )
)


# Save Metrics
benchmark_results.to_csv(
    RESULTS_DIR
    / "benchmark_metrics.csv",
    index=False
)

benchmark_results.to_pickle(
    RESULTS_DIR
    / "benchmark_metrics.pkl"
)

print("Saved master metrics.")


# Save Predictions
all_predictions = (
    pd.concat(
        benchmark_predictions.values(),
        ignore_index=True
    )
)

all_predictions.to_csv(
    RESULTS_DIR
    / "benchmark_predictions.csv",
    index=False
)

all_predictions.to_pickle(
    RESULTS_DIR
    / "benchmark_predictions.pkl"
)

print(
    "Prediction rows:",
    f"{len(all_predictions):,}"
)


# Save Tuning
all_tuning = (
    pd.concat(
        benchmark_tuning.values(),
        ignore_index=True
    )
)

all_tuning.to_pickle(
    RESULTS_DIR
    / "benchmark_tuning.pkl"
)

tuning_csv = (
    all_tuning.copy()
)

tuning_csv[
    "best_params"
] = (
    tuning_csv[
        "best_params"
    ]
    .astype(str)
)

tuning_csv.to_csv(
    RESULTS_DIR
    / "benchmark_tuning.csv",
    index=False
)

print("Saved tuning results.")

# How often each feature set was chosen (one choice per outer fold)
feature_selection_counts = (
    all_tuning
    .groupby(["target", "model", "features"])
    .size()
    .rename("outer_folds")
    .reset_index()
    .sort_values(["target", "model", "outer_folds"], ascending=[True, True, False])
)

feature_selection_counts.to_csv(
    RESULTS_DIR
    / "feature_selection_counts.csv",
    index=False
)

print("\nFEATURE SETS CHOSEN (outer folds)\n")
print(feature_selection_counts.to_string(index=False))


# Save Configs
FEATURE_CONFIGS = {
    target:
        cfg["candidates"]
    for target, cfg
    in TARGET_CONFIGS.items()
}

with open(
    RESULTS_DIR
    / "feature_configs.json",
    "w"
) as f:

    json.dump(
        FEATURE_CONFIGS,
        f,
        indent=2
    )

benchmark_config = {
    "random_state":
        RANDOM_STATE,
    "ridge_alphas":
        RIDGE_ALPHAS,
    "models":
        MODEL_NAMES,
    "graph_laplacian": {
        "laplacians": "data/benchmark_data/effective_laplacians_clean.pkl (Kron-reduced, scaled)"
                      " + unweighted_compressed (built from the compressed graph, same Kron/scaling)",
        "weightings": list(EFFECTIVE_LAPLACIANS),
        "lambda_grid": GRAPH_LAMBDA_GRID,
        "lambda_beta_grid": GRAPH_BETA_GRID,
        "tuned_by": "inner leave-one-node-out, same folds and rows as the other models",
    },
    "feature_selection": {
        "mode": FEATURE_SELECTION,
        "search": FS_SEARCH,
        "fourier_K": FS_K_VALUES,
        "static_options": FS_STATIC_OPTIONS,
        "inner_row_stride": INNER_STRIDE,
        "chemistry": TARGET_CHEMISTRY,
        "selected_by": "inner leave-one-node-out MSE; candidates in feature_configs.json",
    },
    "validation":
        "Nested leave-one-node-out",
    "training_weighting":
        "Equal total weight per monitoring node"
}

with open(
    RESULTS_DIR
    / "benchmark_config.json",
    "w"
) as f:

    json.dump(
        benchmark_config,
        f,
        indent=2
    )

print("Saved configs.")


# R2 Table
r2_table = (
    benchmark_results
    .pivot(
        index="Target",
        columns="Model",
        values="R2"
    )
)

print(
    "\nR2 TABLE\n"
)

print(
    r2_table.to_string()
)

r2_table.to_csv(
    RESULTS_DIR
    / "r2_comparison_table.csv"
)


# RMSE Table
rmse_table = (
    benchmark_results
    .pivot(
        index="Target",
        columns="Model",
        values="RMSE"
    )
)

print(
    "\nRMSE TABLE\n"
)

print(
    rmse_table.to_string()
)

rmse_table.to_csv(
    RESULTS_DIR
    / "rmse_comparison_table.csv"
)


# Best Conventional Models
best_conventional = (
    benchmark_results[
        benchmark_results["Model"] != GRAPH_MODEL_NAME
    ]
    .sort_values(
        "R2",
        ascending=False
    )
    .groupby(
        "Target",
        as_index=False
    )
    .first()
)

print(
    "\nBEST CONVENTIONAL MODELS\n"
)

print(
    best_conventional.to_string(
        index=False
    )
)

best_conventional.to_csv(
    RESULTS_DIR
    / "best_conventional_models.csv",
    index=False
)


# Tuning Summary
TUNED_MODELS = [
    "Ridge",
    "Elastic Net",
    "Spline GAM",
    "Random Forest",
    "Gradient Boosting"
]

for target in TARGET_CONFIGS:

    print(
        "\n"
        + "=" * 80
    )

    print(target)

    print(
        "=" * 80
    )

    target_tuning = (
        all_tuning[
            all_tuning[
                "target"
            ]
            == target
        ]
    )

    for model_name in TUNED_MODELS:

        print(
            "\n",
            model_name
        )

        summary = (
            target_tuning[
                target_tuning[
                    "model"
                ]
                == model_name
            ][
                "best_params"
            ]
            .astype(str)
            .value_counts()
            .rename_axis(
                "Parameters"
            )
            .reset_index(
                name="Outer folds"
            )
        )

        print(
            summary.to_string(
                index=False
            )
        )


# Graph Results
# The Graph-Laplacian model is now fitted inside the benchmark loop above,
# so it is already in benchmark_results. (Previously its row was read from a
# hand-made graph_laplacian_metrics.csv, fitted on different rows/features.)
final_results = (
    benchmark_results.copy()
)


# Final Table
final_results = (
    final_results
    .sort_values(
        [
            "Target",
            "R2"
        ],
        ascending=[
            True,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)

print(
    "\nFINAL TABLE\n"
)

print(
    final_results.to_string(
        index=False
    )
)


# Save Final
final_results.to_csv(
    RESULTS_DIR
    / "final_model_comparison.csv",
    index=False
)

final_results.to_pickle(
    RESULTS_DIR
    / "final_model_comparison.pkl"
)

print("Finished.")