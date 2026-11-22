#!/bin/bash
# usage: tmux new -d -s runoff "PYTHON=../.venv/bin/python ./run_runoff_parallel.sh 2>&1 | tee runoff_driver.log"
# Turbidity runoff signal from the station's own conductance (BENCHMARK_RUNOFF),
# two paired arms on the same rows (turbidity node-hours with runoff features):
#   base   BENCHMARK_RUNOFF=baseline   no runoff features
#   on     BENCHMARK_RUNOFF=on         candidates also offered + [sc_anom, sc_d6] ("+ R")
#                                      and + [sc_anom, sc_d6, sc_mean_log] ("+ RM")
# Turbidity only. Per arm two processes: linear + graph models, and Gradient
# Boosting on its own (slower; turbidity's best model). THREADS per process
# (default 4, since several drivers run at once), under nice. Each arm is merged
# into results/benchmark_results_runoff_<arm> and logged in runoff_<arm>.log.
# Builds data/benchmark_data/runoff_features.pkl first if it is missing.
#
# Afterwards (paired comparisons on identical rows, one per model):
#   for m in "Graph-Laplacian" "OLS" "Ridge" "Elastic Net" "Spline GAM" "Gradient Boosting"; do
#     python compare_runs.py results/benchmark_results_runoff_base results/benchmark_results_runoff_on \
#       --base-model "$m" --new-model "$m" \
#       --out "results/runoff_on_vs_base_$(echo "$m" | tr 'A-Z ' 'a-z_').csv"
#   done
cd "$(dirname "$(readlink -f "$0")")" || exit 1
P=${PYTHON:-python}
THREADS=${THREADS:-4}
T="Turbidity"
safe=turbidity
ARMS=(base on)

models_for () {   # model group -> BENCHMARK_MODELS
  case $1 in
    linear) echo "Dummy,OLS,Ridge,Elastic Net,Spline GAM,Graph-Laplacian" ;;
    gb)     echo "Gradient Boosting" ;;
  esac
}
mode_for () { [ "$1" = base ] && echo baseline || echo on; }   # arm -> BENCHMARK_RUNOFF

[ -f data/benchmark_data/runoff_features.pkl ] || $P build_runoff_features.py || exit 1
mkdir -p runoff_parts/logs
date -u +%s > runoff_parts/start

for A in "${ARMS[@]}"; do
  for G in linear gb; do
    nice -n 10 env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
      MKL_NUM_THREADS=$THREADS LOKY_MAX_CPU_COUNT=$THREADS BENCHMARK_RUNOFF=$(mode_for $A) \
      BENCHMARK_TARGETS="$T" BENCHMARK_MODELS="$(models_for $G)" \
      BENCHMARK_RESULTS_DIR=runoff_parts/$A/$safe/$G \
      $P water_quality_model_benchmark.py > runoff_parts/logs/${A}_$G.log 2>&1 &
  done
done
wait

for A in "${ARMS[@]}"; do
  log=runoff_$A.log ok=1
  : > $log
  for G in linear gb; do
    echo "===== part: $T, $G =====" >> $log
    cat runoff_parts/logs/${A}_$G.log >> $log
    tail -1 runoff_parts/logs/${A}_$G.log | grep -q '^Finished\.' || ok=0
  done
  if [ $ok = 1 ] && $P merge_target_parts.py results/benchmark_results_runoff_$A runoff_parts/$A \
       --targets "$T" >> $log 2>&1; then
    echo "Finished." >> $log
    rm -rf runoff_parts/$A
    echo "$A ok" >> runoff_parts/finished
  else
    echo "$A error" >> runoff_parts/finished
  fi
done
echo "all done" >> runoff_parts/finished
