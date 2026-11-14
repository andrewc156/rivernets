#!/bin/bash
# usage: tmux new -d -s rain "PYTHON=../.venv/bin/python ./run_rain_parallel.sh 2>&1 | tee rain_driver.log"
# Rain benchmarks, one process per target so the work spreads over many cores.
# Phase 1: linear + graph models (fast). Phase 2: Gradient Boosting and Random Forest.
# Each phase waits for all its parts, then merges them into the usual results folder
# (results/benchmark_results_rain_<graph>[_gb|_rf]) and log (rain_<graph>[_gb|_rf].log).
# All processes run under nice so other users' jobs keep priority.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
P=${PYTHON:-python}   # e.g. PYTHON=../.venv/bin/python
TARGETS=("Water temperature" "DO concentration" "Specific conductance" "pH" "Turbidity")
mkdir -p rain_parts/logs

run_phase () {   # $1 phase name, $2 threads per process, then "suffix|models" specs
  local phase=$1 threads=$2; shift 2
  date -u +%s > rain_parts/start_$phase
  for spec in "$@"; do
    local suf=${spec%%|*} models=${spec#*|}
    for R in uncompressed compressed; do
      for T in "${TARGETS[@]}"; do
        local safe=$(echo "$T" | tr 'A-Z ' 'a-z_')
        nice -n 10 env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads \
          MKL_NUM_THREADS=$threads LOKY_MAX_CPU_COUNT=$threads \
          BENCHMARK_RAIN=$R BENCHMARK_TARGETS="$T" BENCHMARK_MODELS="$models" \
          BENCHMARK_RESULTS_DIR=rain_parts/${R}${suf}/$safe \
          $P water_quality_model_benchmark.py > rain_parts/logs/${R}${suf}_$safe.log 2>&1 &
      done
    done
  done
  wait
  for spec in "$@"; do
    local suf=${spec%%|*}
    for R in uncompressed compressed; do
      local log=rain_${R}${suf}.log ok=1
      : > $log
      for T in "${TARGETS[@]}"; do
        local safe=$(echo "$T" | tr 'A-Z ' 'a-z_')
        echo "===== part: $T =====" >> $log
        cat rain_parts/logs/${R}${suf}_$safe.log >> $log
        tail -1 rain_parts/logs/${R}${suf}_$safe.log | grep -q '^Finished\.' || ok=0
      done
      if [ $ok = 1 ] && $P merge_target_parts.py results/benchmark_results_rain_${R}${suf} rain_parts/${R}${suf} >> $log 2>&1; then
        echo "Finished." >> $log
        rm -rf rain_parts/${R}${suf}
        echo "${R}${suf} ok" >> rain_parts/finished
      else
        echo "${R}${suf} error" >> rain_parts/finished
      fi
    done
  done
}

run_phase fast 8 "|Dummy,OLS,Ridge,Elastic Net,Spline GAM,Graph-Laplacian"
run_phase slow 4 "_gb|Gradient Boosting" "_rf|Random Forest"
echo "all done" >> rain_parts/finished
