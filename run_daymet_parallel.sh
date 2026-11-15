#!/bin/bash
# usage: tmux new -d -s daymet "PYTHON=../.venv/bin/python ./run_daymet_parallel.sh 2>&1 | tee daymet_driver.log"
# Daymet precipitation experiment, three paired arms on the same rows (node-hours
# with Daymet features, local dates 2025-01-31 .. 2025-12-31):
#   base    BENCHMARK_DAYMET=baseline                        no precipitation
#   mrms    BENCHMARK_DAYMET=baseline BENCHMARK_RAIN=uncompressed   MRMS upstream rain
#   on      BENCHMARK_DAYMET=on                              Daymet at the station
# Linear + graph models only (no Gradient Boosting / Random Forest). One process
# per arm and target (15 x 8 threads), under nice. Each arm is merged into
# results/benchmark_results_daymet_<arm> and logged in daymet_<arm>.log.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
P=${PYTHON:-python}
MODELS="Dummy,OLS,Ridge,Elastic Net,Spline GAM,Graph-Laplacian"
TARGETS=("Water temperature" "DO concentration" "Specific conductance" "pH" "Turbidity")
declare -A ENVS=([base]="BENCHMARK_DAYMET=baseline"
                 [mrms]="BENCHMARK_DAYMET=baseline BENCHMARK_RAIN=uncompressed"
                 [on]="BENCHMARK_DAYMET=on")
ARMS=(base mrms on)
mkdir -p daymet_parts/logs
date -u +%s > daymet_parts/start

for A in "${ARMS[@]}"; do
  for T in "${TARGETS[@]}"; do
    safe=$(echo "$T" | tr 'A-Z ' 'a-z_')
    nice -n 10 env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
      MKL_NUM_THREADS=8 LOKY_MAX_CPU_COUNT=8 ${ENVS[$A]} \
      BENCHMARK_TARGETS="$T" BENCHMARK_MODELS="$MODELS" \
      BENCHMARK_RESULTS_DIR=daymet_parts/$A/$safe \
      $P water_quality_model_benchmark.py > daymet_parts/logs/${A}_$safe.log 2>&1 &
  done
done
wait

for A in "${ARMS[@]}"; do
  log=daymet_$A.log ok=1
  : > $log
  for T in "${TARGETS[@]}"; do
    safe=$(echo "$T" | tr 'A-Z ' 'a-z_')
    echo "===== part: $T =====" >> $log
    cat daymet_parts/logs/${A}_$safe.log >> $log
    tail -1 daymet_parts/logs/${A}_$safe.log | grep -q '^Finished\.' || ok=0
  done
  if [ $ok = 1 ] && $P merge_target_parts.py results/benchmark_results_daymet_$A daymet_parts/$A >> $log 2>&1; then
    echo "Finished." >> $log
    rm -rf daymet_parts/$A
    echo "$A ok" >> daymet_parts/finished
  else
    echo "$A error" >> daymet_parts/finished
  fi
done
echo "all done" >> daymet_parts/finished
