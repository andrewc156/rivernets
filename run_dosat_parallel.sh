#!/bin/bash
# usage: tmux new -d -s dosat "PYTHON=../.venv/bin/python ./run_dosat_parallel.sh 2>&1 | tee dosat_driver.log"
# DO-saturation experiment (DO concentration only), two paired arms on the same
# rows (DO node-hours with do_sat and wt_diel from build_dosat_features.py):
#   base    BENCHMARK_DOSAT=baseline   the main DO feature candidates
#   on      BENCHMARK_DOSAT=on         every candidate also with + do_sat (" + S")
#                                      and + do_sat, wt_diel (" + S + WD")
# Linear + graph models only (no Gradient Boosting / Random Forest). One process
# per arm, THREADS threads each (default 4, as six drivers share the server),
# under nice. Each arm's folder is moved to results/benchmark_results_dosat_<arm> and
# logged in dosat_<arm>.log.
#
# Afterwards (paired, same rows and folds):
#   for m in "Graph-Laplacian" OLS Ridge "Elastic Net" "Spline GAM"; do
#     f=$(echo "$m" | tr 'A-Z -' 'a-z__')
#     python compare_runs.py results/benchmark_results_dosat_base results/benchmark_results_dosat_on \
#       --base-model "$m" --new-model "$m" --out results/dosat_on_vs_base_$f.csv
#   done
cd "$(dirname "$(readlink -f "$0")")" || exit 1
P=${PYTHON:-python}
THREADS=${THREADS:-4}
MODELS="Dummy,OLS,Ridge,Elastic Net,Spline GAM,Graph-Laplacian"
TARGETS=("DO concentration")
declare -A ENVS=([base]="BENCHMARK_DOSAT=baseline" [on]="BENCHMARK_DOSAT=on")
ARMS=(base on)
mkdir -p dosat_parts/logs
date -u +%s > dosat_parts/start

# features (seconds; deterministic, rebuilt so they always match this checkout)
$P build_dosat_features.py > dosat_parts/logs/build.log 2>&1 || { echo "build failed" >> dosat_parts/finished; exit 1; }

for A in "${ARMS[@]}"; do
  for T in "${TARGETS[@]}"; do
    safe=$(echo "$T" | tr 'A-Z ' 'a-z_')
    nice -n 10 env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
      MKL_NUM_THREADS=$THREADS LOKY_MAX_CPU_COUNT=$THREADS ${ENVS[$A]} \
      BENCHMARK_TARGETS="$T" BENCHMARK_MODELS="$MODELS" \
      BENCHMARK_RESULTS_DIR=dosat_parts/$A/$safe \
      $P water_quality_model_benchmark.py > dosat_parts/logs/${A}_$safe.log 2>&1 &
  done
done
wait

for A in "${ARMS[@]}"; do
  log=dosat_$A.log ok=1
  : > $log
  cat dosat_parts/logs/build.log >> $log
  for T in "${TARGETS[@]}"; do
    safe=$(echo "$T" | tr 'A-Z ' 'a-z_')
    echo "===== part: $T =====" >> $log
    cat dosat_parts/logs/${A}_$safe.log >> $log
    tail -1 dosat_parts/logs/${A}_$safe.log | grep -q '^Finished\.' || ok=0
  done
  # one target only: the part folder is already a complete results folder
  # (merge_target_parts.py expects all five targets), so it is moved into place
  if [ $ok = 1 ] && [ -f dosat_parts/$A/do_concentration/final_model_comparison.csv ] && \
     rm -rf results/benchmark_results_dosat_$A && mkdir -p results && \
     mv dosat_parts/$A/do_concentration results/benchmark_results_dosat_$A; then
    echo "Finished." >> $log
    rm -rf dosat_parts/$A
    echo "$A ok" >> dosat_parts/finished
  else
    echo "$A error" >> dosat_parts/finished
  fi
done
echo "all done" >> dosat_parts/finished
