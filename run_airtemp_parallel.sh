#!/bin/bash
# usage: tmux new -d -s airtemp "PYTHON=../.venv/bin/python ./run_airtemp_parallel.sh 2>&1 | tee airtemp_driver.log"
# Air-temperature experiment (ERA5-Land, download_airtemp.py + build_airtemp_features.py;
# data/benchmark_data/airtemp_features.pkl is committed, so nothing is downloaded here).
# Two paired arms on the same rows (every benchmark node-hour has air temperature):
#   base    BENCHMARK_AIRTEMP=baseline   no air temperature
#   on      BENCHMARK_AIRTEMP=on         every candidate also offered + at_1h, at_24h, at_72h, at_168h
# Targets: water temperature and DO. Linear + graph models only (no Gradient Boosting /
# Random Forest). One process per arm and target (4 x THREADS threads, default 4), under
# nice. Each arm is merged into results/benchmark_results_airtemp_<arm> and logged in
# airtemp_<arm>.log.
#
# Afterwards (paired comparisons, same rows and folds):
#   for m in "Graph-Laplacian" "OLS" "Ridge" "Elastic Net" "Spline GAM"; do
#     f=$(echo "$m" | tr 'A-Z ' 'a-z_')
#     python compare_runs.py results/benchmark_results_airtemp_base results/benchmark_results_airtemp_on \
#       --base-model "$m" --out results/airtemp_on_vs_base_$f.csv
#   done
cd "$(dirname "$(readlink -f "$0")")" || exit 1
P=${PYTHON:-python}
T=${THREADS:-4}
MODELS="Dummy,OLS,Ridge,Elastic Net,Spline GAM,Graph-Laplacian"
TARGETS=("Water temperature" "DO concentration")
TARGET_LIST="Water temperature,DO concentration"
declare -A ENVS=([base]="BENCHMARK_AIRTEMP=baseline"
                 [on]="BENCHMARK_AIRTEMP=on")
ARMS=(base on)
[ -f data/benchmark_data/airtemp_features.pkl ] || { echo "missing data/benchmark_data/airtemp_features.pkl"; exit 1; }
mkdir -p airtemp_parts/logs
date -u +%s > airtemp_parts/start

for A in "${ARMS[@]}"; do
  for TG in "${TARGETS[@]}"; do
    safe=$(echo "$TG" | tr 'A-Z ' 'a-z_')
    nice -n 10 env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T \
      MKL_NUM_THREADS=$T LOKY_MAX_CPU_COUNT=$T ${ENVS[$A]} \
      BENCHMARK_TARGETS="$TG" BENCHMARK_MODELS="$MODELS" \
      BENCHMARK_RESULTS_DIR=airtemp_parts/$A/$safe \
      $P water_quality_model_benchmark.py > airtemp_parts/logs/${A}_$safe.log 2>&1 &
  done
done
wait

for A in "${ARMS[@]}"; do
  log=airtemp_$A.log ok=1
  : > $log
  for TG in "${TARGETS[@]}"; do
    safe=$(echo "$TG" | tr 'A-Z ' 'a-z_')
    echo "===== part: $TG =====" >> $log
    cat airtemp_parts/logs/${A}_$safe.log >> $log
    tail -1 airtemp_parts/logs/${A}_$safe.log | grep -q '^Finished\.' || ok=0
  done
  if [ $ok = 1 ] && $P merge_target_parts.py results/benchmark_results_airtemp_$A airtemp_parts/$A --targets "$TARGET_LIST" >> $log 2>&1; then
    echo "Finished." >> $log
    rm -rf airtemp_parts/$A
    echo "$A ok" >> airtemp_parts/finished
  else
    echo "$A error" >> airtemp_parts/finished
  fi
done
echo "all done" >> airtemp_parts/finished
