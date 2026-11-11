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

# Discharge (opt-in), see "Discharge" below:
#   ""          off (default)
#   "baseline"  keep only rows where discharge is measured, but do not use it,
#               so it is the paired baseline for "on"
#   "on"        same rows; every feature candidate is offered with and without
#               the discharge anomaly, chosen in the inner loop
DISCHARGE_MODE = os.environ.get("BENCHMARK_DISCHARGE", "")
if DISCHARGE_MODE not in ("", "baseline", "on"):
    raise ValueError("BENCHMARK_DISCHARGE must be '', 'baseline' or 'on'")

# Held-out time period (opt-in): BENCHMARK_TIME_SPLIT=YYYY-MM-DD keeps the
# leave-one-station-out folds but also splits in time: every model (and its
# inner loop) is trained on the other stations' rows BEFORE the cutoff, and is
# tested on the held-out station's rows ON OR AFTER it. A station needs at least
# TIME_SPLIT_MIN_HOURS rows on a side to train or to be tested. The cutoff
# 2026-01-01 was fixed from station coverage alone, before any results: most
# records start in October 2025, so a water-year split would leave 6-9 training
# stations. The test is the 2026 spring-summer season at stations never seen.
TIME_SPLIT = os.environ.get("BENCHMARK_TIME_SPLIT", "")
TIME_SPLIT_MIN_HOURS = int(os.environ.get("BENCHMARK_TIME_SPLIT_MIN_HOURS", "720"))
TIME_CUTOFF = pd.Timestamp(TIME_SPLIT, tz="UTC") if TIME_SPLIT else None

# Outer split (opt-in), BENCHMARK_SPLIT:
#   loo (default)     leave one station out: each station is predicted by a
#                     model trained on all the others (the main benchmark)
#   station_repeated  BENCHMARK_SPLIT_REPEATS random draws (default 20) of 20%
#                     of the stations as test stations, trained on the other 80%
#   station_single    the first of those draws only
#   time_within       every station: first 80% of its record (by time) trains,
#                     last 20% tests; the test stations are seen in training
#   random_rows       a random 20% of all rows tests (leaky: neighbouring hours
#                     of the same station fall on both sides; an upper bound)
# In every scheme the models' settings and feature sets are still chosen by the
# inner leave-one-station-out loop on the training rows only.
SPLIT = os.environ.get("BENCHMARK_SPLIT", "loo")
SPLIT_REPEATS = int(os.environ.get("BENCHMARK_SPLIT_REPEATS", "20"))
SPLIT_TEST_FRACTION = 0.2
if SPLIT not in ("loo", "station_repeated", "station_single", "time_within", "random_rows"):
    raise ValueError(f"Unknown BENCHMARK_SPLIT: {SPLIT}")
if SPLIT != "loo" and TIME_SPLIT:
    raise ValueError("BENCHMARK_SPLIT and BENCHMARK_TIME_SPLIT cannot be combined")

if SPLIT != "loo":
    _default_results = f"results/benchmark_results_split_{SPLIT}"
elif TIME_SPLIT:
    _default_results = "results/benchmark_results_time_split"
elif DISCHARGE_MODE:
    _default_results = f"results/benchmark_results_discharge_{DISCHARGE_MODE}"
elif FEATURE_SELECTION == "fixed":
    _default_results = "results/benchmark_results"
else:
    _default_results = "results/benchmark_results_inner_fs"

RESULTS_DIR = Path(
    os.environ.get(
        "BENCHMARK_RESULTS_DIR",
        _default_results
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
    diagonal = 1). edge_weight(u, v, data) gives each compressed edge's weight."""
    order = list(_G_master.nodes())
    pos = {n: i for i, n in enumerate(order)}
    L = np.zeros((len(order), len(order)))
    for u, v, data in _G_master.to_undirected(as_view=True).edges(data=True):
        w = edge_weight(u, v, data)
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
_check = compressed_kron_laplacian(lambda u, v, data: 1000.0 / float(data["length_m"]))
if not np.allclose(_check, EFFECTIVE_LAPLACIANS["distance_only"], rtol=1e-8, atol=1e-10):
    raise ValueError("Rebuilt distance_only Laplacian does not match the saved one.")

# unweighted_compressed: every compressed reach has weight 1, whatever its
# length or number of original segments, so the graph distance between two
# stations is the number of reaches between them. Without weights the original
# and compressed graphs are NOT equivalent; unweighted on the original 67k graph
# is almost identical to distance_only (compare_edge_weights.py).
EFFECTIVE_LAPLACIANS["unweighted_compressed"] = compressed_kron_laplacian(lambda u, v, data: 1.0)

# length_pow_a: w = length_km^(-a), a continuous family between the two above
# (a = 0 is unweighted_compressed, a = 1 is distance_only). The inner loop picks a.
for _a in (0.25, 0.5, 0.75):
    EFFECTIVE_LAPLACIANS[f"length_pow_{_a}"] = compressed_kron_laplacian(
        lambda u, v, data, a=_a: (float(data["length_m"]) / 1000.0) ** (-a)
    )

# confluence_compressed: the graph distance between two stations counts the
# confluences (nodes of degree >= 3) between them. Each edge has resistance
# 0.5 per confluence at its ends plus CONFLUENCE_EPS, so a node the path merely
# passes through (another station or gauge) adds only CONFLUENCE_EPS and nearby
# stations with no confluence between them are not fused into one.
CONFLUENCE_EPS = 0.1
_degree = dict(_G_master.to_undirected(as_view=True).degree())
EFFECTIVE_LAPLACIANS["confluence_compressed"] = compressed_kron_laplacian(
    lambda u, v, data: 1.0 / (
        0.5 * (_degree[u] >= 3) + 0.5 * (_degree[v] >= 3) + CONFLUENCE_EPS
    )
)

# The ten weightings of BENCHMARK_GRAPH_EXTENDED=1 (report Table 19). The
# matrices added below are opt-in only, through BENCHMARK_GRAPH_WEIGHTINGS.
EXTENDED_GRAPH_WEIGHTINGS = list(EFFECTIVE_LAPLACIANS)

# Dams (opt-in weightings distance_dams, unweighted_dams)
# A large reservoir breaks the river's continuity (cold, clear bottom releases),
# so smoothing across a dam should be weak. Each dam adds a resistance of
# DAM_RESISTANCE x the median reach resistance to the compressed reach it sits
# on, in series with the reach:  w' = 1 / (1/w + n_dams * R_dam).
# MAJOR_DAMS is a hand-compiled list of the basin's large storage dams
# (reservoir storage of about 0.3 km^3 or more); a dam counts only if it lies
# within DAM_SNAP_KM of a reach's MERIT path, so dams off the network are skipped.
DAM_RESISTANCE = float(os.environ.get("BENCHMARK_DAM_RESISTANCE", "10"))
DAM_SNAP_KM = 2.0
MAJOR_DAMS = {
    "Glen Canyon": (36.9375, -111.4844),
    "Hoover": (36.0161, -114.7377),
    "Davis": (35.1972, -114.5697),
    "Parker": (34.2964, -114.1394),
    "Flaming Gorge": (40.9147, -109.4217),
    "Fontenelle": (42.0275, -110.0675),
    "Blue Mesa": (38.4528, -107.3358),
    "Navajo": (36.8006, -107.6125),
    "McPhee": (37.5761, -108.5747),
    "Granby": (40.1483, -105.8664),
    "Dillon": (39.6247, -106.0647),
    "Theodore Roosevelt": (33.6717, -111.1617),
    "Coolidge": (33.1736, -110.5250),
}

# MERIT node ids are row * 15001 + col on the 3-arcsecond (1/1200 degree) grid;
# the grid origin is recovered from the compressed nodes' row/col/lat/lon.
_MERIT_COLS = 15001
_any = next(iter(_G_master.nodes(data=True)))[1]
_LAT0 = _any["latitude"] + _any["row"] / 1200.0
_LON0 = _any["longitude"] - _any["col"] / 1200.0
for _n, _d in _G_master.nodes(data=True):
    _r, _c = divmod(int(_n), _MERIT_COLS)
    if (_r, _c) != (_d["row"], _d["col"]):
        raise ValueError("MERIT node id does not decode to its row/col")

_path_ll, _path_edge = [], []
for _u, _v, _d in _G_master.edges(data=True):
    for _n in _d["original_path"]:
        _r, _c = divmod(int(_n), _MERIT_COLS)
        _path_ll.append((_LAT0 - _r / 1200.0, _LON0 + _c / 1200.0))
        _path_edge.append(frozenset((_u, _v)))
_path_ll = np.array(_path_ll)

DAMS_ON_EDGE = {}
for _name, (_la, _lo) in MAJOR_DAMS.items():
    _km = np.hypot(
        (_path_ll[:, 0] - _la) * 111.0,
        (_path_ll[:, 1] - _lo) * 111.0 * np.cos(np.radians(_la)),
    )
    _i = int(_km.argmin())
    if _km[_i] <= DAM_SNAP_KM:
        DAMS_ON_EDGE[_path_edge[_i]] = DAMS_ON_EDGE.get(_path_edge[_i], 0) + 1
        print(f"Dam {_name}: on the network ({_km[_i]:.2f} km)")
    else:
        print(f"Dam {_name}: off the network ({_km[_i]:.0f} km), skipped")


def _with_dams(edge_weight):
    edges = list(_G_master.to_undirected(as_view=True).edges(data=True))
    r_dam = DAM_RESISTANCE * np.median([1.0 / edge_weight(u, v, d) for u, v, d in edges])
    return lambda u, v, data: 1.0 / (
        1.0 / edge_weight(u, v, data) + DAMS_ON_EDGE.get(frozenset((u, v)), 0) * r_dam
    )


EFFECTIVE_LAPLACIANS["distance_dams"] = compressed_kron_laplacian(
    _with_dams(lambda u, v, data: 1000.0 / float(data["length_m"]))
)
EFFECTIVE_LAPLACIANS["unweighted_dams"] = compressed_kron_laplacian(
    _with_dams(lambda u, v, data: 1.0)
)

# tailup_<km> (opt-in): flow-directed covariance instead of a Laplacian, the
# exponential tail-up model of Ver Hoef, Peterson & Theobald (2006). For station
# i upstream of station j
#     C_ij = sqrt(A_i / A_j) * exp(-d_ij / range),
# with A the upstream area (the additive weights telescope to this ratio) and
# d_ij the river distance; stations on different branches are uncorrelated a
# priori. g ~ N(0, C / lambda) gives the penalty lambda * g' C^-1 g, so C^-1
# (scaled like the Laplacians) takes the place of L_eff. C^-1 has no constant
# null vector, so the sum-to-zero constraint on g is not applied to it
# (_graph_prepare): a held-out station with no flow-connected neighbour then
# gets g = 0, the intercept.
TAILUP_RANGES_KM = (100, 300, 1000)
_upa = static_node_df.drop_duplicates("node").set_index("node")["upa_km2"]
_C_flow = np.eye(len(GRAPH_NODES))
_D_flow = np.full((len(GRAPH_NODES), len(GRAPH_NODES)), np.inf)
for _i, _n in enumerate(GRAPH_NODES):
    _dist, _cur = 0.0, _n
    while _G_master.out_degree(_cur) > 0:
        (_nxt,) = _G_master.successors(_cur)
        _dist += float(_G_master.edges[_cur, _nxt]["length_m"]) / 1000.0
        _cur = _nxt
        if _cur in GRAPH_NODE_INDEX:
            _j = GRAPH_NODE_INDEX[_cur]
            _D_flow[_i, _j] = _D_flow[_j, _i] = _dist
            _ratio = min(float(_upa[_n]) / float(_upa[_cur]), 1.0)
            _C_flow[_i, _j] = _C_flow[_j, _i] = np.sqrt(_ratio)
for _range in TAILUP_RANGES_KM:
    _C = np.where(np.isfinite(_D_flow), _C_flow * np.exp(-np.nan_to_num(_D_flow, posinf=0.0) / _range), 0.0)
    np.fill_diagonal(_C, 1.0)
    if np.linalg.eigvalsh(_C).min() <= 0:
        raise ValueError(f"tail-up covariance with range {_range} km is not positive definite")
    _Q = np.linalg.inv(_C)
    _Q = (_Q + _Q.T) / 2.0
    EFFECTIVE_LAPLACIANS[f"tailup_{_range}"] = _Q / np.diag(_Q).mean()

# Graph model settings
#
# Default: the main graph model of the report (Tables 13 and 14): six
# weightings, lambda grid 1e-4 ... 100, raw targets only.
#
# BENCHMARK_GRAPH_EXTENDED=1 turns on the variants tested in the report's
# "Further graph-model variants" section: all ten weightings (adds
# length_pow_{0.25,0.5,0.75} and confluence_compressed), the wide lambda grid
# 1e-6 ... 3000, and raw vs log(1 + y) for turbidity and specific conductance.
# They did not improve the model.
#
# BENCHMARK_GRAPH_WEIGHTINGS, BENCHMARK_GRAPH_LAMBDA_GRID and
# BENCHMARK_GRAPH_LOG_TARGETS override each part individually.
GRAPH_EXTENDED = os.environ.get("BENCHMARK_GRAPH_EXTENDED", "0") == "1"

DEFAULT_GRAPH_WEIGHTINGS = [
    "distance_only", "area", "width", "slope", "area_width", "unweighted_compressed"
]
_weightings = [w for w in os.environ.get("BENCHMARK_GRAPH_WEIGHTINGS", "").split(",") if w]
if not _weightings:
    _weightings = EXTENDED_GRAPH_WEIGHTINGS if GRAPH_EXTENDED else DEFAULT_GRAPH_WEIGHTINGS
_unknown = [w for w in _weightings if w not in EFFECTIVE_LAPLACIANS]
if _unknown:
    raise ValueError(f"Unknown BENCHMARK_GRAPH_WEIGHTINGS: {_unknown}")
EFFECTIVE_LAPLACIANS = {w: EFFECTIVE_LAPLACIANS[w] for w in _weightings}

# Lambda grid: the saved grid (0.01 ... 100) extended down to 1e-4, or with
# BENCHMARK_GRAPH_EXTENDED=1 widened to 1e-6 ... 3000.
_lambda_env = os.environ.get("BENCHMARK_GRAPH_LAMBDA_GRID", "")
if _lambda_env:
    GRAPH_LAMBDA_GRID = [float(x) for x in _lambda_env.split(",") if x]
elif GRAPH_EXTENDED:
    GRAPH_LAMBDA_GRID = sorted({
        1e-6, 1e-5, 0.0001, 0.001, 0.003,
        *map(float, GRAPH_LAMBDA_GRID),
        300.0, 1000.0, 3000.0
    })
else:
    GRAPH_LAMBDA_GRID = sorted({0.0001, 0.001, 0.003, *map(float, GRAPH_LAMBDA_GRID)})
print("Graph lambda grid:", GRAPH_LAMBDA_GRID)

# Log targets: for these targets the graph model may also be fitted on
# log(1 + y) (turbidity has a few zeros). Predictions are transformed back with
# a node-weighted smearing correction, and raw vs log is chosen by the inner-LOO
# MSE on the original scale. Off by default.
GRAPH_LOG_TARGETS = [
    t for t in os.environ.get(
        "BENCHMARK_GRAPH_LOG_TARGETS",
        "Turbidity,Specific conductance" if GRAPH_EXTENDED else ""
    ).split(",") if t
]

# Averaging: with BENCHMARK_GRAPH_AVERAGE_TOP=k > 1, the graph model takes the
# best (lambda, lambda_beta) within every (feature set, weighting) pair, ranks
# those pairs by inner-LOO MSE, and averages the predictions of the top k with
# equal weights. k = 1 (default) is the single best setting. Applies to raw
# targets; a log-target choice still uses its single best setting.
GRAPH_AVERAGE_TOP = int(os.environ.get("BENCHMARK_GRAPH_AVERAGE_TOP", "1"))
if GRAPH_AVERAGE_TOP < 1:
    raise ValueError("BENCHMARK_GRAPH_AVERAGE_TOP must be >= 1")
print("Graph averaging: top", GRAPH_AVERAGE_TOP)

# Hyperparameter choice (opt-in): BENCHMARK_GRAPH_SELECT=reml chooses the
# weighting and lambda by restricted (REML) likelihood instead of inner-LOO MSE.
# The penalised fit is the posterior mode of
#     node-mean residual r_i = F_i' a + g_i + e_i,   e ~ N(0, s2 I),
#     g ~ N(0, s2 * L_eff^+ / lambda),
# where F holds the intercept and the node-constant (static MERIT) predictors
# and r_i is the node mean minus the time-varying part of the fit. Feature set
# and lambda_beta are still chosen by inner-LOO MSE, with the weighting and
# lambda inside each inner fold set by REML on that fold's training stations.
# The aim is more stable choices: the likelihood uses all training stations at
# once instead of scoring one held-out station at a time.
GRAPH_SELECT = os.environ.get("BENCHMARK_GRAPH_SELECT", "loo")
if GRAPH_SELECT not in ("loo", "reml"):
    raise ValueError("BENCHMARK_GRAPH_SELECT must be 'loo' or 'reml'")

# Network-varying seasonal cycle (opt-in): BENCHMARK_GRAPH_VARYING=annual (or
# "annual,daily") offers every feature set also with station-specific
# coefficients on the first annual (doy_sin_1, doy_cos_1) and/or daily
# harmonic, smoothed over the graph like g:
#     y_it = b0 + z_it' b + g_i + sum_k gamma_ik s_k(t),
#     penalty  lambda * g' L g + lambda_v * sum_k gamma_k' L gamma_k,
# so a station's seasonal amplitude and phase are borrowed from its neighbours.
# lambda_v has its own grid. Such feature sets are named "<set> | vary".
VARYING_SETS = {
    "annual": ["doy_sin_1", "doy_cos_1"],
    "daily": ["hour_sin_1", "hour_cos_1"],
}
GRAPH_VARYING_COLS = [
    c for s in os.environ.get("BENCHMARK_GRAPH_VARYING", "").split(",") if s
    for c in VARYING_SETS[s]
]
GRAPH_VARY_SUFFIX = " | vary"
GRAPH_VARYING_LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
if GRAPH_VARYING_COLS and GRAPH_SELECT == "reml":
    raise ValueError("BENCHMARK_GRAPH_VARYING is not supported with BENCHMARK_GRAPH_SELECT=reml")
print("Graph hyperparameter choice:", GRAPH_SELECT, "| varying columns:", GRAPH_VARYING_COLS)

# Graph + gradient boosting (opt-in model, BENCHMARK_MODELS="Graph-Laplacian + GB"):
# the graph model as configured above, then HistGradientBoosting fitted to its
# out-of-station residuals (each training station predicted by the graph model
# refitted without it), with the time-varying predictors only (no static MERIT
# columns, which made plain boosting overfit between stations). Prediction =
# graph prediction + boosting correction. The boosting setting is fixed in
# advance (GRAPH_GB_PARAMS), not tuned.
GRAPH_GB_NAME = "Graph-Laplacian + GB"
GRAPH_GB_PARAMS = {
    "learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 15, "l2_regularization": 1.0
}
print("Graph log-target candidates:", GRAPH_LOG_TARGETS)

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
    A11, rb, qmean, q2s, qys = np.zeros((q, q)), np.zeros(q), {}, {}, {}
    for n in train_nodes:
        qm, q2, qy = _graph_standard_moments(stats[n], mean_x, std_x)
        A11 += q2
        rb += qy
        qmean[n], q2s[n], qys[n] = qm, q2, qy
    return {
        "mean_x": mean_x, "std_x": std_x, "q": q, "A11": A11, "rb": rb,
        "qmean": qmean, "q2": q2s, "qy": qys,
        "mean_y": {n: stats[n]["mean_y"] for n in train_nodes},
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
            "Btv": Bt.T @ v, "cX": c @ X, "cu": c @ u, "cv": c @ v,
            "gauge": _is_laplacian(L)}


def _is_laplacian(L):
    """A Laplacian has the constant vector in its null space, so g is only
    defined up to a constant and needs the sum-to-zero gauge. A positive
    definite precision (tail-up) does not."""
    return bool(np.allclose(L.sum(axis=1), 0.0, atol=1e-8))


def _graph_solve(prep, lambda_beta):
    tm, q = prep["tm"], prep["tm"]["q"]
    ridge = np.eye(q)
    ridge[0, 0] = 0.0                                # intercept not penalised
    if not prep["gauge"]:
        beta = np.linalg.solve(
            tm["A11"] + lambda_beta * ridge - prep["BtX"], tm["rb"] - prep["Btu"]
        )
        return {"beta": beta, "g": prep["u"] - prep["X"] @ beta,
                "mean_x": tm["mean_x"], "std_x": tm["std_x"]}
    S = np.empty((q + 1, q + 1))
    S[:q, :q] = tm["A11"] + lambda_beta * ridge - prep["BtX"]
    S[:q, q] = -prep["Btv"]
    S[q, :q] = prep["cX"]
    S[q, q] = prep["cv"]
    z = np.linalg.solve(S, np.concatenate((tm["rb"] - prep["Btu"], [prep["cu"]])))
    beta, mu = z[:q], z[q]
    return {"beta": beta, "g": prep["u"] - prep["X"] @ beta - prep["v"] * mu,
            "mean_x": tm["mean_x"], "std_x": tm["std_x"]}


def _graph_prepare_varying(tm, L, lam, lam_v, vidx):
    """Like _graph_prepare, with station-specific coefficients on the columns
    vidx of [1, z] as well as the intercept: G (N x p) holds g and the gammas,
    column k penalised by lam_k * G_k' L G_k (lam for g, lam_v for the rest),
    each with its own sum-to-zero gauge when L is a Laplacian."""
    N, q = L.shape[0], tm["q"]
    cols = [0, *vidx]
    p = len(cols)
    M = np.kron(np.diag([lam] + [lam_v] * (p - 1)), L)
    B, rg = np.zeros((N * p, q)), np.zeros(N * p)
    for n, q2 in tm["q2"].items():
        idx = GRAPH_NODE_INDEX[n] + N * np.arange(p)
        M[np.ix_(idx, idx)] += q2[np.ix_(cols, cols)]
        B[idx] = q2[cols]
        rg[idx] = tm["qy"][n][cols]
    C = np.kron(np.eye(p), np.ones((N, 1))) if _is_laplacian(L) else np.zeros((N * p, 0))
    sol = np.linalg.solve(M, np.column_stack((B, rg, C)))
    X, u, V = sol[:, :q], sol[:, q], sol[:, q + 1:]
    return {"tm": tm, "X": X, "u": u, "V": V, "cols": cols, "N": N,
            "BtX": B.T @ X, "Btu": B.T @ u, "BtV": B.T @ V,
            "CX": C.T @ X, "Cu": C.T @ u, "CV": C.T @ V}


def _graph_solve_varying(prep, lambda_beta):
    tm, q = prep["tm"], prep["tm"]["q"]
    k = prep["V"].shape[1]
    ridge = np.eye(q)
    ridge[0, 0] = 0.0
    S = np.empty((q + k, q + k))
    S[:q, :q] = tm["A11"] + lambda_beta * ridge - prep["BtX"]
    S[:q, q:] = -prep["BtV"]
    S[q:, :q] = prep["CX"]
    S[q:, q:] = prep["CV"]
    z = np.linalg.solve(S, np.concatenate((tm["rb"] - prep["Btu"], prep["Cu"])))
    beta, mu = z[:q], z[q:]
    G = (prep["u"] - prep["X"] @ beta - prep["V"] @ mu).reshape(len(prep["cols"]), prep["N"]).T
    return {"beta": beta, "g": G[:, 0], "G": G, "cols": prep["cols"],
            "mean_x": tm["mean_x"], "std_x": tm["std_x"]}


def _graph_coef(model, node):
    coef = model["beta"].copy()
    j = GRAPH_NODE_INDEX[int(node)]
    if "G" in model:
        coef[model["cols"]] += model["G"][j]
    else:
        coef[0] += model["g"][j]
    return coef


# Covariance of g up to scale, for the REML criterion: L_eff^+ (the inverse for
# the positive definite tail-up precisions).
GRAPH_COVARIANCE = {name: np.linalg.pinv(L) for name, L in EFFECTIVE_LAPLACIANS.items()}


def _graph_reml(model, tm, nodes, cov, lam, fixed_idx):
    """Restricted log-likelihood (s2 profiled out) of the training stations'
    node-mean residuals r_i = mean y_i - (time-varying part of the fit) under
    r = F a + g + e, Var = s2 (I + cov / lam), F = [1, static columns]."""
    beta = model["beta"]
    tv = np.setdiff1d(np.arange(len(beta)), fixed_idx)
    Qm = np.array([tm["qmean"][n] for n in nodes])
    r = np.array([tm["mean_y"][n] for n in nodes]) - Qm[:, tv] @ beta[tv]
    F = Qm[:, fixed_idx]
    idx = [GRAPH_NODE_INDEX[n] for n in nodes]
    V = np.eye(len(nodes)) + cov[np.ix_(idx, idx)] / lam
    Vi = np.linalg.inv(V)
    FtViF = F.T @ Vi @ F
    P = Vi - Vi @ F @ np.linalg.solve(FtViF, F.T @ Vi)
    m, k = len(nodes), F.shape[1]
    return -0.5 * (
        np.linalg.slogdet(V)[1] + np.linalg.slogdet(FtViF)[1] + (m - k) * np.log(r @ P @ r)
    )


def _graph_fixed_idx(features):
    """Indices in [1, z] of the intercept and the node-constant predictors."""
    return np.array([0, *[1 + i for i, f in enumerate(features) if f in ALL_STATIC]])


def _graph_node_mse(stat, model, node):
    _, q2, qy = _graph_standard_moments(stat, model["mean_x"], model["std_x"])
    coef = _graph_coef(model, node)
    return max(float(stat["mean_y2"] - 2.0 * coef @ qy + coef @ q2 @ coef), 0.0)


def _graph_settings(tm, vidx=()):
    """Yield ((scheme, lambda, lambda_beta[, lambda_v]), fitted model) over the grid."""
    for scheme, L in EFFECTIVE_LAPLACIANS.items():
        for lam in GRAPH_LAMBDA_GRID:
            if vidx:
                for lam_v in GRAPH_VARYING_LAMBDA_GRID:
                    prep = _graph_prepare_varying(tm, L, float(lam), float(lam_v), vidx)
                    for lb in GRAPH_BETA_GRID:
                        yield ((scheme, float(lam), float(lb), float(lam_v)),
                               _graph_solve_varying(prep, float(lb)))
            else:
                prep = _graph_prepare(tm, L, float(lam))
                for lb in GRAPH_BETA_GRID:
                    yield (scheme, float(lam), float(lb)), _graph_solve(prep, float(lb))


def _graph_inner_scores(stats, vidx=()):
    """Mean per-node inner-LOO MSE for every (scheme, lambda, lambda_beta[, lambda_v])."""
    train_nodes = sorted(stats)
    totals = defaultdict(float)
    for val_node in train_nodes:
        tm = _graph_train_moments(stats, [n for n in train_nodes if n != val_node])
        for key, model in _graph_settings(tm, vidx):
            totals[key] += _graph_node_mse(stats[val_node], model, val_node)
    return {k: t / len(train_nodes) for k, t in totals.items()}


def _graph_reml_choices(tm, nodes, fixed_idx):
    """For every lambda_beta, the (scheme, lambda) with the largest REML
    criterion on these training stations, and its fitted model."""
    best = {}
    for scheme, L in EFFECTIVE_LAPLACIANS.items():
        for lam in GRAPH_LAMBDA_GRID:
            prep = _graph_prepare(tm, L, float(lam))
            for lb in GRAPH_BETA_GRID:
                model = _graph_solve(prep, float(lb))
                ll = _graph_reml(model, tm, nodes, GRAPH_COVARIANCE[scheme], float(lam), fixed_idx)
                if float(lb) not in best or ll > best[float(lb)][0]:
                    best[float(lb)] = (ll, (scheme, float(lam), float(lb)), model)
    return best


def _graph_inner_scores_reml(stats, fixed_idx):
    """Mean per-node inner-LOO MSE for every lambda_beta, with the weighting
    and lambda chosen by REML on each inner fold's training stations."""
    train_nodes = sorted(stats)
    totals = defaultdict(float)
    for val_node in train_nodes:
        nodes = [n for n in train_nodes if n != val_node]
        tm = _graph_train_moments(stats, nodes)
        for lb, (_, _, model) in _graph_reml_choices(tm, nodes, fixed_idx).items():
            totals[lb] += _graph_node_mse(stats[val_node], model, val_node)
    return {k: t / len(train_nodes) for k, t in totals.items()}


def _graph_candidates(candidates):
    """The feature sets the graph model searches: every candidate, plus a
    "<set> | vary" copy when BENCHMARK_GRAPH_VARYING is set."""
    out = dict(candidates)
    if GRAPH_VARYING_COLS:
        for name, features in candidates.items():
            if any(c in features for c in GRAPH_VARYING_COLS):
                out[name + GRAPH_VARY_SUFFIX] = features
    return out


def _graph_vidx(name, features):
    if not name.endswith(GRAPH_VARY_SUFFIX):
        return ()
    return tuple(1 + features.index(c) for c in GRAPH_VARYING_COLS if c in features)


def _graph_search_all(train_df, target, candidates):
    """For every (feature set, weighting) pair, its best (lambda, lambda_beta):
    a list of (inner MSE, feature set, (scheme, lambda, lambda_beta), stats)."""
    out, cache = [], {}
    for name, features in candidates.items():
        if tuple(features) not in cache:
            cache[tuple(features)] = graph_node_stats(train_df, target, features)
        stats = cache[tuple(features)]
        if GRAPH_SELECT == "reml":
            fixed_idx = _graph_fixed_idx(features)
            scores = _graph_inner_scores_reml(stats, fixed_idx)
            lb = min(scores, key=scores.get)
            nodes = sorted(stats)
            key = _graph_reml_choices(_graph_train_moments(stats, nodes), nodes, fixed_idx)[lb][1]
            out.append((scores[lb], name, key, stats))
            continue
        scores = _graph_inner_scores(stats, _graph_vidx(name, features))
        for scheme in EFFECTIVE_LAPLACIANS:
            key = min((k for k in scores if k[0] == scheme), key=scores.get)
            out.append((scores[key], name, key, stats))
    return out


def _graph_search(train_df, target, candidates):
    """Best (inner MSE, feature set, (scheme, lambda, lambda_beta), stats)."""
    return min(_graph_search_all(train_df, target, candidates), key=lambda r: r[0])


def _graph_fit(stats, nodes, scheme, lam, lb, lam_v=None, vidx=()):
    tm = _graph_train_moments(stats, nodes)
    if vidx:
        return _graph_solve_varying(
            _graph_prepare_varying(tm, EFFECTIVE_LAPLACIANS[scheme], lam, lam_v, vidx), lb
        )
    return _graph_solve(_graph_prepare(tm, EFFECTIVE_LAPLACIANS[scheme], lam), lb)


def _graph_predict(model, df, features, node):
    Z = (df[features].to_numpy(dtype=float) - model["mean_x"]) / model["std_x"]
    return np.column_stack((np.ones(len(Z)), Z)) @ _graph_coef(model, node)


def _graph_log_predict(by_node, stats, features, key, train_nodes, test_df, test_node, target):
    """Fit on log(1 + y) of train_nodes, predict test_df on the original scale.
    Back-transform with Duan's smearing factor, averaged with equal weight per
    training node: y_hat = exp(f) * mean(exp(residual)) - 1."""
    scheme, lam, lb = key
    model = _graph_fit(stats, train_nodes, scheme, lam, lb)
    smear = np.mean([
        np.mean(np.exp(
            by_node[n][target].to_numpy(dtype=float) - _graph_predict(model, by_node[n], features, n)
        ))
        for n in train_nodes
    ])
    return np.exp(_graph_predict_nodes(model, test_df, features)) * smear - 1.0


def _graph_predict_nodes(model, df, features):
    """Predict rows that may belong to several stations, each with its own g."""
    out = np.empty(len(df))
    for node, pos in df.groupby("node").indices.items():
        out[pos] = _graph_predict(model, df.iloc[pos], features, node)
    return out


def _fold_id(target, outer_test):
    return (target, len(outer_test), outer_test.index[0], outer_test.index[-1])


def fit_predict_graph_laplacian(outer_train, outer_test, target, candidates):
    """Inner-LOO choice of (feature set, weighting, lambda, lambda_beta), and for
    GRAPH_LOG_TARGETS also raw vs log(1 + y), then refit on the outer training
    nodes and predict the held-out node. Same selection criterion as
    tune_model_inner_loo: the mean of the per-node validation MSE, always on the
    original scale. The search is joint: the graph model is solved exactly from
    per-node moments, so it is cheap. For the log option the settings are tuned
    in log space and then scored on the original scale over the same inner folds."""
    test_node = outer_test["node"].iloc[0]
    candidates = _graph_candidates(candidates)
    ranked = sorted(_graph_search_all(outer_train, target, candidates), key=lambda r: r[0])
    choice = ("none", *ranked[0])

    if target in GRAPH_LOG_TARGETS:
        log_train = outer_train.copy()
        log_train[target] = np.log1p(log_train[target])
        _, name, key, stats = _graph_search(log_train, target, candidates)
        by_node = {int(n): g for n, g in log_train.groupby("node")}
        train_nodes = sorted(stats)
        node_mse = []
        for val in train_nodes:
            pred = _graph_log_predict(
                by_node, stats, candidates[name], key,
                [n for n in train_nodes if n != val], by_node[val], val, target
            )
            y = np.expm1(by_node[val][target].to_numpy(dtype=float))
            node_mse.append(np.mean((y - pred) ** 2))
        log_mse = float(np.mean(node_mse))
        if log_mse < choice[1]:
            choice = ("log1p", log_mse, name, key, stats)

    transform, inner_mse, name, key, stats = choice
    features = candidates[name]
    members = [(name, *key)]
    if transform == "none":
        top = ranked[:GRAPH_AVERAGE_TOP]
        pred = np.mean([
            _graph_predict_nodes(
                _graph_fit(st, sorted(st), *k, vidx=_graph_vidx(n, candidates[n])),
                outer_test, candidates[n]
            )
            for _, n, k, st in top
        ], axis=0)
        members = [(n, *k) for _, n, k, _ in top]
    else:
        pred = _graph_log_predict(
            by_node, stats, features, key, sorted(stats), outer_test, test_node, target
        )
    scheme, lam, lb = key[:3]
    best_params = {"scheme": scheme, "lambda": lam, "lambda_beta": lb, "transform": transform}
    if len(key) > 3:
        best_params["lambda_vary"] = key[3]
    if GRAPH_AVERAGE_TOP > 1:
        best_params["members"] = members
    _GRAPH_LAST.clear()
    _GRAPH_LAST.update({
        "id": _fold_id(target, outer_test), "name": name, "key": key, "stats": stats,
        "features": features, "transform": transform, "pred": pred,
        "best_params": best_params, "inner_mse": inner_mse,
    })
    return name, best_params, inner_mse, pred


_GRAPH_LAST = {}


def fit_predict_graph_gb(outer_train, outer_test, target, candidates):
    """Graph model, then gradient boosting on its out-of-station residuals."""
    if _GRAPH_LAST.get("id") != _fold_id(target, outer_test):
        fit_predict_graph_laplacian(outer_train, outer_test, target, candidates)
    g = dict(_GRAPH_LAST)
    if g["transform"] != "none" or GRAPH_AVERAGE_TOP > 1:
        raise ValueError(f"{GRAPH_GB_NAME} needs raw targets and GRAPH_AVERAGE_TOP=1")
    name, key, stats, features = g["name"], g["key"], g["stats"], g["features"]
    vidx = _graph_vidx(name, features)
    train_nodes = sorted(stats)
    graph_train = np.empty(len(outer_train))
    for n, pos in outer_train.groupby("node").indices.items():
        model = _graph_fit(stats, [m for m in train_nodes if m != int(n)], *key, vidx=vidx)
        graph_train[pos] = _graph_predict(model, outer_train.iloc[pos], features, n)
    gb_features = sorted({
        f for fs in candidates.values() for f in fs if f not in ALL_STATIC
    })
    gb = build_benchmark_model("Gradient Boosting", GRAPH_GB_PARAMS)
    gb.fit(
        outer_train[gb_features].to_numpy(),
        outer_train[target].to_numpy() - graph_train,
        model__sample_weight=equal_node_weights(outer_train, node_col="node"),
    )
    pred = g["pred"] + gb.predict(outer_test[gb_features].to_numpy())
    best_params = {**g["best_params"], "gb": GRAPH_GB_PARAMS, "gb_features": len(gb_features)}
    return name, best_params, g["inner_mse"], pred


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


# Discharge
# Hourly discharge from usgs_all_stations_hourly.csv, mapped to WQ nodes as the
# median over each node's stations, like the WQ variables. The feature is the
# discharge anomaly  q_anom = log(Q) - (that station's mean log Q):  raw log Q
# mostly measures river size (it overlaps log upstream area), while the
# dilution / sediment effects are within-station, over time. It uses only the
# station's own discharge record, never water-quality values.
# Rows without discharge are dropped (complete case); "baseline" uses the same
# rows without the feature, so the two runs are a paired comparison. The task
# becomes "predict water quality at a flow-gauged station with no WQ sensor".
HOURLY_FILE = Path("data/usgs/usgs_all_stations_hourly.csv")

if DISCHARGE_MODE:
    _station_to_node = {}
    for _node, _data in _G_master.nodes(data=True):
        for _cluster in _data.get("usgs_clusters", []):
            _stations = _cluster.get("member_stations", []) or (
                [_cluster["anchor_station"]] if _cluster.get("anchor_station") is not None else []
            )
            for _station in _stations:
                _station_to_node[_station] = _node

    _q = pd.read_csv(HOURLY_FILE, usecols=["station_id", "time", "Discharge"])
    _q["time"] = pd.to_datetime(_q["time"], utc=True, errors="coerce")
    _q["node"] = _q["station_id"].map(_station_to_node)
    _q = _q[_q["node"].notna() & (_q["Discharge"] > 0)].copy()
    _q["node"] = _q["node"].astype(int)
    _q = _q.groupby(["node", "time"], as_index=False)["Discharge"].median()
    _q["log_q"] = np.log(_q["Discharge"])
    _q["q_anom"] = _q["log_q"] - _q.groupby("node")["log_q"].transform("mean")

    print(f"\nDischarge ({DISCHARGE_MODE}): {_q['node'].nunique()} WQ nodes with discharge")
    for target, cfg in TARGET_CONFIGS.items():
        df = cfg["df"].drop(columns=["q_anom"], errors="ignore").merge(
            _q[["node", "time", "q_anom"]], on=["node", "time"], how="left"
        )
        df = df[df["q_anom"].notna()].copy()
        cfg["df"] = df
        used = df.dropna(subset=[target, *TARGET_CHEMISTRY[target]])
        per_node = used.groupby("node").size()
        print(
            f"  {target:<22} stations {len(per_node):>2} | rows {len(used):>7,} | "
            f"rows per station min {per_node.min():,}, median {int(per_node.median()):,}"
        )
        if DISCHARGE_MODE == "on":
            cfg["candidates"] = {
                **cfg["candidates"],
                **{f"{name} + Q": [*features, "q_anom"]
                   for name, features in cfg["candidates"].items()}
            }


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
def outer_fold_masks(work, nodes, node_col):
    """(label, train mask, test mask) for every outer fold of BENCHMARK_SPLIT."""
    node = work[node_col].to_numpy()
    if SPLIT == "loo":
        return [(n, node != n, node == n) for n in nodes]
    rng = np.random.default_rng(RANDOM_STATE)
    if SPLIT in ("station_repeated", "station_single"):
        k = max(1, int(round(SPLIT_TEST_FRACTION * len(nodes))))
        out = []
        for r in range(1 if SPLIT == "station_single" else SPLIT_REPEATS):
            test = rng.choice(np.array(nodes), size=k, replace=False)
            m = np.isin(node, test)
            out.append((f"draw {r + 1}: " + ",".join(map(str, sorted(test))), ~m, m))
        return out
    if SPLIT == "time_within":
        rank = (
            pd.to_datetime(work["time"], utc=True)
            .groupby(work[node_col]).rank(method="first", pct=True).to_numpy()
        )
        m = rank > 1 - SPLIT_TEST_FRACTION
        return [("last 20% of every station", ~m, m)]
    m = rng.random(len(work)) < SPLIT_TEST_FRACTION
    return [("random 20% of rows", ~m, m)]


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
        *features_union,
        *(["time"] if TIME_SPLIT or SPLIT == "time_within" else [])
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

    if TIME_SPLIT:
        before = pd.to_datetime(work["time"], utc=True) < TIME_CUTOFF
        n_before = work[before].groupby(node_col).size()
        n_after = work[~before].groupby(node_col).size()
        train_nodes_ok = set(n_before[n_before >= TIME_SPLIT_MIN_HOURS].index)
        nodes = [n for n in nodes if n_after.get(n, 0) >= TIME_SPLIT_MIN_HOURS]
        print(
            f"Time split {TIME_SPLIT}: {len(train_nodes_ok)} training stations, "
            f"{len(nodes)} test stations"
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

    folds = outer_fold_masks(work, nodes, node_col)

    for (
        fold_number,
        (test_node, train_mask, test_mask)
    ) in enumerate(
        folds,
        start=1
    ):

        print(
            f"\rOuter fold "
            f"{fold_number}/"
            f"{len(folds)} "
            f"| test "
            f"{test_node}",
            end=""
        )

        outer_train = work[train_mask]
        outer_test = work[test_mask]

        if TIME_SPLIT:
            outer_train = outer_train[
                before.loc[outer_train.index]
                & outer_train[node_col].isin(train_nodes_ok)
            ]
            outer_test = outer_test[~before.loc[outer_test.index]]

        y_test = (
            outer_test[
                target
            ]
            .to_numpy()
        )

        for model_name in model_names:

            if model_name in (GRAPH_MODEL_NAME, GRAPH_GB_NAME):

                (
                    feature_set,
                    best_params,
                    inner_mse,
                    pred
                ) = (
                    fit_predict_graph_laplacian
                    if model_name == GRAPH_MODEL_NAME
                    else fit_predict_graph_gb
                )(
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
                row_node,
                y_true,
                y_pred
            ) in zip(
                outer_test.index,
                outer_test[node_col],
                y_test,
                pred
            ):

                prediction_rows.append({
                    "target":
                        target,
                    "model":
                        model_name,
                    "fold":
                        fold_number,
                    "node":
                        row_node,
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
                      " + unweighted_compressed, length_pow_{0.25,0.5,0.75} and confluence_compressed"
                      " (built from the compressed graph, same Kron/scaling)",
        "extended": GRAPH_EXTENDED,
        "average_top": GRAPH_AVERAGE_TOP,
        "select": GRAPH_SELECT,
        "varying": GRAPH_VARYING_COLS,
        "dam_resistance": DAM_RESISTANCE,
        "dams_on_network": len(DAMS_ON_EDGE),
        "tailup_ranges_km": list(TAILUP_RANGES_KM),
        "graph_gb_params": GRAPH_GB_PARAMS,
        "discharge": DISCHARGE_MODE or "off",
        "time_split": TIME_SPLIT or "off",
        "split": SPLIT,
        "split_repeats": SPLIT_REPEATS,
        "time_split_min_hours": TIME_SPLIT_MIN_HOURS,
        "confluence_eps": CONFLUENCE_EPS,
        "log_target_candidates": GRAPH_LOG_TARGETS,
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
        ~benchmark_results["Model"].isin([GRAPH_MODEL_NAME, GRAPH_GB_NAME])
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