#!/bin/bash
# usage: tmux new -d -s blend "PYTHON=../.venv/bin/python ./run_blend_parallel.sh 2>&1 | tee blend_driver.log"
# Graph-GB blend: w * Graph-Laplacian + (1 - w) * Gradient Boosting, with w chosen in
# every outer fold by an inner leave-one-station-out on the training stations only
# (see "option: blend" in water_quality_model_benchmark.py). The same run writes the
# plain Graph-Laplacian and Gradient Boosting predictions on identical rows and folds,
# so the blend is compared with each component inside one results folder.
# One process per target (5), THREADS threads each (default 4: the six option
# drivers run at the same time), under nice. Parts go to blend_parts/<target>, are
# merged into results/benchmark_results_blend, and logged in blend.log.
# Cost: dominated by the Gradient Boosting tuning, as in any GB run; the blend adds
# one GB refit per training station and outer fold (about 8% more GB fits).
#
# Afterwards (paired, same rows):
#   python compare_runs.py results/benchmark_results_blend results/benchmark_results_blend \
#     --base-model "Gradient Boosting" --new-model "Graph-GB blend" --out results/blend_vs_gradient_boosting.csv
#   python compare_runs.py results/benchmark_results_blend results/benchmark_results_blend \
#     --base-model "Graph-Laplacian" --new-model "Graph-GB blend" --out results/blend_vs_graph_laplacian.csv
# The chosen weights: column best_params ("w_graph") of
#   results/benchmark_results_blend/benchmark_tuning.csv, model "Graph-GB blend".
cd "$(dirname "$(readlink -f "$0")")" || exit 1
P=${PYTHON:-python}
THREADS=${THREADS:-4}
MODELS="Graph-Laplacian,Gradient Boosting,Graph-GB blend"
TARGETS=("Water temperature" "DO concentration" "Specific conductance" "pH" "Turbidity")
mkdir -p blend_parts/logs
date -u +%s > blend_parts/start

for T in "${TARGETS[@]}"; do
  safe=$(echo "$T" | tr 'A-Z ' 'a-z_')
  nice -n 10 env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
    MKL_NUM_THREADS=$THREADS LOKY_MAX_CPU_COUNT=$THREADS \
    BENCHMARK_TARGETS="$T" BENCHMARK_MODELS="$MODELS" \
    BENCHMARK_RESULTS_DIR=blend_parts/$safe \
    $P water_quality_model_benchmark.py > blend_parts/logs/$safe.log 2>&1 &
done
wait

log=blend.log ok=1
: > $log
for T in "${TARGETS[@]}"; do
  safe=$(echo "$T" | tr 'A-Z ' 'a-z_')
  echo "===== part: $T =====" >> $log
  cat blend_parts/logs/$safe.log >> $log
  tail -1 blend_parts/logs/$safe.log | grep -q '^Finished\.' || ok=0
done
if [ $ok = 1 ] && $P merge_target_parts.py results/benchmark_results_blend blend_parts >> $log 2>&1; then
  echo "Finished." >> $log
  rm -rf blend_parts/water_temperature blend_parts/do_concentration \
         blend_parts/specific_conductance blend_parts/ph blend_parts/turbidity
  echo "blend ok" >> blend_parts/finished
else
  echo "blend error" >> blend_parts/finished
fi
echo "all done" >> blend_parts/finished
