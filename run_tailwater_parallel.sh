#!/bin/bash
# usage: tmux new -d -s tailwater "PYTHON=../.venv/bin/python ./run_tailwater_parallel.sh 2>&1 | tee tailwater_driver.log"
# Dam-tailwater experiment (build_tailwater_features.py must have written
# data/benchmark_data/tailwater_features.csv; it is committed). Two paired arms on
# the same rows (the tailwater terms are static per station, so no rows change):
#   base    BENCHMARK_TAILWATER=baseline   main benchmark candidates
#   on      BENCHMARK_TAILWATER=on         every candidate also offered "+ TW":
#           tw_tail = exp(-d/100 km) below the nearest major dam, and tw_tail x
#           doy_sin_1, doy_cos_1, doy_sin_2, doy_cos_2
# Targets: water temperature, specific conductance, DO, pH (the inner loop may
# decline the terms). Linear + graph models only (no Gradient Boosting / Random
# Forest). One process per arm and target (8 processes x THREADS threads, default
# 4, under nice). Each arm is merged into results/benchmark_results_tailwater_<arm>
# and logged in tailwater_<arm>.log.
#
# Afterwards, paired comparisons (same rows and folds):
#   for m in "OLS" "Ridge" "Elastic Net" "Spline GAM" "Graph-Laplacian"; do
#     s=$(echo "$m" | tr 'A-Z -' 'a-z__')
#     ../.venv/bin/python compare_runs.py results/benchmark_results_tailwater_base \
#       results/benchmark_results_tailwater_on --base-model "$m" --new-model "$m" \
#       --out results/tailwater_on_vs_base_$s.csv
#   done
cd "$(dirname "$(readlink -f "$0")")" || exit 1
P=${PYTHON:-python}
T_N=${THREADS:-4}
MODELS="Dummy,OLS,Ridge,Elastic Net,Spline GAM,Graph-Laplacian"
TARGETS=("Water temperature" "Specific conductance" "DO concentration" "pH")
declare -A ENVS=([base]="BENCHMARK_TAILWATER=baseline"
                 [on]="BENCHMARK_TAILWATER=on")
ARMS=(base on)
[ -f data/benchmark_data/tailwater_features.csv ] || { echo "run build_tailwater_features.py first"; exit 1; }
mkdir -p tailwater_parts/logs
date -u +%s > tailwater_parts/start

for A in "${ARMS[@]}"; do
  for T in "${TARGETS[@]}"; do
    safe=$(echo "$T" | tr 'A-Z ' 'a-z_')
    nice -n 10 env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$T_N OPENBLAS_NUM_THREADS=$T_N \
      MKL_NUM_THREADS=$T_N LOKY_MAX_CPU_COUNT=$T_N ${ENVS[$A]} \
      BENCHMARK_TARGETS="$T" BENCHMARK_MODELS="$MODELS" \
      BENCHMARK_RESULTS_DIR=tailwater_parts/$A/$safe \
      $P water_quality_model_benchmark.py > tailwater_parts/logs/${A}_$safe.log 2>&1 &
  done
done
wait

for A in "${ARMS[@]}"; do
  log=tailwater_$A.log ok=1
  : > $log
  for T in "${TARGETS[@]}"; do
    safe=$(echo "$T" | tr 'A-Z ' 'a-z_')
    echo "===== part: $T =====" >> $log
    cat tailwater_parts/logs/${A}_$safe.log >> $log
    tail -1 tailwater_parts/logs/${A}_$safe.log | grep -q '^Finished\.' || ok=0
  done
  if [ $ok = 1 ] && $P merge_target_parts.py results/benchmark_results_tailwater_$A tailwater_parts/$A --targets "Water temperature,Specific conductance,DO concentration,pH" >> $log 2>&1; then
    echo "Finished." >> $log
    rm -rf tailwater_parts/$A
    echo "$A ok" >> tailwater_parts/finished
  else
    echo "$A error" >> tailwater_parts/finished
  fi
done
echo "all done" >> tailwater_parts/finished
