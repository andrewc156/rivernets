#!/bin/bash
# usage: PYTHON=../.venv/bin/python ./run_all_options.sh            (all six)
#        PYTHON=../.venv/bin/python ./run_all_options.sh dosat blend (some of them)
# Starts each R2-improvement experiment in its own detached tmux session, so all
# run at the same time and independently:
#   dosat      DO saturation + diel water temperature, DO            2 processes
#   tailwater  dam-tailwater seasonal terms, corrected dam list,
#              water temp / spec. cond. / DO / pH                     8 processes
#   runoff     conductance runoff signal, turbidity (incl. GB)        4 processes
#   phdiel     diel water temperature, pH                             2 processes
#   blend      Graph-GB blend, all five targets (incl. GB)            5 processes
#   airtemp    ERA5-Land air temperature, water temp / DO             4 processes
# 25 processes in all, each with THREADS threads (default 4, so 100 threads);
# lower THREADS if the server has fewer cores. Watch with `tmux ls`, attach with
# `tmux attach -t <name>`; each experiment writes <name>_driver.log and appends
# "<arm> ok" / "all done" to <name>_parts/finished (blend: blend_parts/finished).
# Every driver's header lists the compare_runs.py commands for its results.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
P=${PYTHON:-python}
T=${THREADS:-4}
NAMES=("$@")
[ ${#NAMES[@]} -eq 0 ] && NAMES=(dosat tailwater runoff phdiel blend airtemp)

for n in "${NAMES[@]}"; do
  if [ ! -x "run_${n}_parallel.sh" ]; then
    echo "unknown experiment: $n" >&2; exit 1
  fi
  if tmux has-session -t "$n" 2>/dev/null; then
    echo "tmux session '$n' already exists, skipped" >&2; continue
  fi
  tmux new -d -s "$n" "PYTHON=$P THREADS=$T ./run_${n}_parallel.sh 2>&1 | tee ${n}_driver.log"
  echo "started $n (tmux session '$n')"
done
