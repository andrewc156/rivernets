"""Merge per-target benchmark runs into one results folder, matching what a single
full run of water_quality_model_benchmark.py writes.

usage: merge_target_parts.py OUT_DIR PARTS_DIR [--targets "T1,T2"]
PARTS_DIR holds one sub-folder per target (water_temperature, do_concentration, ...),
each written by the benchmark with BENCHMARK_TARGETS set to that one target.
--targets limits the merge to those targets (default: all five).
A target folder may instead hold model-group sub-folders (e.g. turbidity/linear and
turbidity/gb, each run with its own BENCHMARK_MODELS); their outputs are concatenated.
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ORDER = ["Water temperature", "DO concentration", "Specific conductance", "pH", "Turbidity"]
GRAPH_NAMES = ["Graph-Laplacian", "Graph-Laplacian + GB", "Graph-GB blend"]

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("parts")
ap.add_argument("--targets", default=",".join(ORDER))
args = ap.parse_args()
targets = [t for t in ORDER if t in args.targets.split(",")]
if len(targets) != len([t for t in args.targets.split(",") if t]):
    raise SystemExit(f"unknown target in --targets: {args.targets}")

out, parts = Path(args.out), Path(args.parts)
out.mkdir(parents=True, exist_ok=True)
dirs = [parts / t.lower().replace(" ", "_") for t in targets]


def groups(d):
    """The finished run folders of one target: the folder itself, or its model groups."""
    if (d / "final_model_comparison.csv").exists():
        return [d]
    subs = sorted(s for s in d.glob("*") if s.is_dir())
    if subs and all((s / "final_model_comparison.csv").exists() for s in subs):
        return subs
    raise SystemExit(f"missing finished part: {d}")


runs = {d: groups(d) for d in dirs}

# per-target files are already what a full run writes (concatenated over model groups)
for d, gs in runs.items():
    for f in gs[0].glob(d.name + "_*"):
        if len(gs) == 1:
            shutil.copy2(f, out / f.name)
        elif f.suffix == ".pkl":
            pd.concat([pd.read_pickle(g / f.name) for g in gs], ignore_index=True).to_pickle(out / f.name)
        elif f.suffix == ".csv":
            pd.concat([pd.read_csv(g / f.name) for g in gs], ignore_index=True).to_csv(out / f.name, index=False)


def cat(name):
    return pd.concat([pd.read_pickle(g / name) for gs in runs.values() for g in gs], ignore_index=True)


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
for gs in runs.values():
    for g in gs:
        fc.update(json.load(open(g / "feature_configs.json")))
json.dump({t: fc[t] for t in ORDER if t in fc}, open(out / "feature_configs.json", "w"), indent=2)
first = runs[dirs[0]][0]
config = json.load(open(first / "benchmark_config.json"))
if any(len(gs) > 1 for gs in runs.values()) and isinstance(config.get("models"), list):
    models = []
    for gs in runs.values():
        for g in gs:
            models += [m for m in json.load(open(g / "benchmark_config.json")).get("models", []) if m not in models]
    config["models"] = models
    json.dump(config, open(out / "benchmark_config.json", "w"), indent=2)
else:
    shutil.copy2(first / "benchmark_config.json", out / "benchmark_config.json")

metrics.pivot(index="Target", columns="Model", values="R2").to_csv(out / "r2_comparison_table.csv")
metrics.pivot(index="Target", columns="Model", values="RMSE").to_csv(out / "rmse_comparison_table.csv")
(metrics[~metrics["Model"].isin(GRAPH_NAMES)].sort_values("R2", ascending=False)
 .groupby("Target", as_index=False).first().to_csv(out / "best_conventional_models.csv", index=False))

metrics.to_csv(out / "final_model_comparison.csv", index=False)
metrics.to_pickle(out / "final_model_comparison.pkl")
n_parts = sum(len(gs) for gs in runs.values())
print(f"merged {len(dirs)} targets ({n_parts} parts) into {out}: "
      f"{len(metrics)} metric rows, {len(preds):,} prediction rows")
