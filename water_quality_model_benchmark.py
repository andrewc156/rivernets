# Imports
from pathlib import Path
import json
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

BENCHMARK_DIR = Path("benchmark_data")
RESULTS_DIR = Path("benchmark_results")

RESULTS_DIR.mkdir(exist_ok=True)

print("Benchmark data:", BENCHMARK_DIR.resolve())
print("Results:", RESULTS_DIR.resolve())


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

missing_static = [
    c
    for c in REDUCED_STATIC
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


# Verify Features
all_ok = True

for target, cfg in TARGET_CONFIGS.items():
    required = [
        "node",
        target,
        *cfg["features"]
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
            "| predictors:",
            len(cfg["features"])
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
    node_col="node"
):

    train_nodes = (
        outer_train[node_col]
        .drop_duplicates()
        .tolist()
    )

    best_score = np.inf
    best_params = None

    for params in MODEL_CANDIDATES[
        model_name
    ]:

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


# Nested Outer LOO
def benchmark_target_nested_loo(
    df,
    target,
    features,
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
            "Gradient Boosting"
        ]

    required = [
        node_col,
        target,
        *features
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
        "Predictors:",
        len(features)
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

        X_test = (
            outer_test[
                features
            ]
            .to_numpy()
        )

        y_test = (
            outer_test[
                target
            ]
            .to_numpy()
        )

        for model_name in model_names:

            (
                best_params,
                inner_mse
            ) = (
                tune_model_inner_loo(
                    outer_train=
                        outer_train,
                    target=
                        target,
                    features=
                        features,
                    model_name=
                        model_name,
                    node_col=
                        node_col
                )
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
            features=
                cfg["features"]
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


# Save Configs
FEATURE_CONFIGS = {
    target:
        cfg["features"]
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
        list(
            MODEL_CANDIDATES.keys()
        ),
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
    benchmark_results
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
graph_path = (
    RESULTS_DIR
    / "graph_laplacian_metrics.csv"
)

if graph_path.exists():

    graph_results = (
        pd.read_csv(
            graph_path
        )
    )

    final_results = (
        pd.concat(
            [
                benchmark_results,
                graph_results
            ],
            ignore_index=True
        )
    )

else:

    final_results = (
        benchmark_results.copy()
    )

    print(
        "Graph-Laplacian benchmark "
        "has not been added yet."
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