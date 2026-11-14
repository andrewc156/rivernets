"""Merge per-target benchmark runs into one results folder, matching what a single
full run of water_quality_model_benchmark.py writes.

usage: merge_target_parts.py OUT_DIR PARTS_DIR
PARTS_DIR holds one sub-folder per target (water_temperature, do_concentration, ...),
each written by the benchmark with BENCHMARK_TARGETS set to that one target.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ORDER = ["Water temperature", "DO concentration", "Specific conductance", "pH", "Turbidity"]
GRAPH_NAMES = ["Graph-Laplacian", "Graph-Laplacian + GB"]

out, parts = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
dirs = [parts / t.lower().replace(" ", "_") for t in ORDER]
for d in dirs:
    if not (d / "final_model_comparison.csv").exists():
        sys.exit(f"missing finished part: {d}")

# per-target files are already what a full run writes
for d in dirs:
    for f in d.glob("*"):
        if f.name.startswith(d.name + "_"):
            shutil.copy2(f, out / f.name)


def cat(name):
    return pd.concat([pd.read_pickle(d / name) for d in dirs], ignore_index=True)


metrics = cat("benchmark_metrics.pkl").sort_values(["Target", "R2"], ascending=[True, False]).reset_index(drop=True)
metrics.to_csv(out / "benchmark_metrics.csv", index=False)
metrics.to_pickle(out / "benchmark_metrics.pkl")

preds = cat("benchmark_predictions.pkl")
preds.to_csv(out / "benchmark_predictions.csv", index=False)
preds.to_pickle(out / "benchmark_predictions.pkl")

tuning = cat("benchmark_tuning.pkl")
tuning.to_pickle(out / "benchmark_tuning.pkl")
tcsv = tuning.copy()
tcsv["best_params"] = tcsv["best_params"].astype(str)
tcsv.to_csv(out / "benchmark_tuning.csv", index=False)

(tuning.groupby(["target", "model", "features"]).size().rename("outer_folds").reset_index()
 .sort_values(["target", "model", "outer_folds"], ascending=[True, True, False])
 .to_csv(out / "feature_selection_counts.csv", index=False))

fc = {}
for d in dirs:
    fc.update(json.load(open(d / "feature_configs.json")))
json.dump({t: fc[t] for t in ORDER if t in fc}, open(out / "feature_configs.json", "w"), indent=2)
shutil.copy2(dirs[0] / "benchmark_config.json", out / "benchmark_config.json")

metrics.pivot(index="Target", columns="Model", values="R2").to_csv(out / "r2_comparison_table.csv")
metrics.pivot(index="Target", columns="Model", values="RMSE").to_csv(out / "rmse_comparison_table.csv")
(metrics[~metrics["Model"].isin(GRAPH_NAMES)].sort_values("R2", ascending=False)
 .groupby("Target", as_index=False).first().to_csv(out / "best_conventional_models.csv", index=False))

metrics.to_csv(out / "final_model_comparison.csv", index=False)
metrics.to_pickle(out / "final_model_comparison.pkl")
print(f"merged {len(dirs)} targets into {out}: {len(metrics)} metric rows, {len(preds):,} prediction rows")
