#!/bin/bash
# usage: tmux new -d -s phdiel "PYTHON=../.venv/bin/python ./run_phdiel_parallel.sh 2>&1 | tee phdiel_driver.log"
# pH with the diel part of water temperature, two paired arms on the same rows
# (pH node-hours with wt_diel from build_phdiel_features.py; 18 of the 19 pH
# stations, the one without water temperature drops out):
#   base    BENCHMARK_PHDIEL=baseline     no new feature
#   on      BENCHMARK_PHDIEL=on           every pH candidate also with wt_diel ("+ WD")
# Linear + graph models only (no Gradient Boosting / Random Forest). One process
# per arm (pH only), THREADS threads each (default 4: six drivers share the
# server), under nice. A single-target run already writes a complete results
# folder, so each arm is moved to results/benchmark_results_phdiel_<arm>
# (merge_target_parts.py needs all five targets) and logged in phdiel_<arm>.log.
# Needs data/benchmark_data/phdiel_features.pkl (python build_phdiel_features.py).
#
# afterwards (paired, same rows and folds):
#   for m in "Graph-Laplacian" "OLS" "Ridge" "Elastic Net" "Spline GAM"; do
#     s=$(echo "$m" | tr 'A-Z ' 'a-z_')
#     $PYTHON compare_runs.py results/benchmark_results_phdiel_base results/benchmark_results_phdiel_on \
#       --base-model "$m" --new-model "$m" --out results/phdiel_on_vs_base_$s.csv
#   done
cd "$(dirname "$(readlink -f "$0")")" || exit 1
P=${PYTHON:-python}
T=${THREADS:-4}
MODELS="Dummy,OLS,Ridge,Elastic Net,Spline GAM,Graph-Laplacian"
TARGET="pH"
safe=ph
ARMS=(base on)
[ -f data/benchmark_data/phdiel_features.pkl ] || $P build_phdiel_features.py || exit 1
mkdir -p phdiel_parts/logs
date -u +%s > phdiel_parts/start

for A in "${ARMS[@]}"; do
  nice -n 10 env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T \
    MKL_NUM_THREADS=$T LOKY_MAX_CPU_COUNT=$T BENCHMARK_PHDIEL=$([ $A = base ] && echo baseline || echo on) \
    BENCHMARK_TARGETS="$TARGET" BENCHMARK_MODELS="$MODELS" \
    BENCHMARK_RESULTS_DIR=phdiel_parts/$A/$safe \
    $P water_quality_model_benchmark.py > phdiel_parts/logs/${A}_$safe.log 2>&1 &
done
wait

for A in "${ARMS[@]}"; do
  log=phdiel_$A.log
  cp phdiel_parts/logs/${A}_$safe.log $log
  if tail -1 $log | grep -q '^Finished\.' && [ -f phdiel_parts/$A/$safe/final_model_comparison.csv ]; then
    rm -rf results/benchmark_results_phdiel_$A
    mv phdiel_parts/$A/$safe results/benchmark_results_phdiel_$A
    rm -rf phdiel_parts/$A
    echo "$A ok" >> phdiel_parts/finished
  else
    echo "$A error" >> phdiel_parts/finished
  fi
done
echo "all done" >> phdiel_parts/finished
