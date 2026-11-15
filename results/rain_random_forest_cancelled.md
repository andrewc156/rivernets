# Upstream rainfall runs: Random Forest cancelled

The Random Forest arm of the upstream MRMS rainfall experiment
(`BENCHMARK_RAIN=uncompressed` and `BENCHMARK_RAIN=compressed`,
`BENCHMARK_MODELS="Random Forest"`) was stopped on 2026-11-15 at 01:27 UTC
before it finished, because it was taking too long.

- It ran as one process per target and graph (10 processes, 8 threads each, nice 15)
  on the shared server, restarted 2026-11-14 05:50 UTC.
- After about 19.5 hours it was only about half done (outer folds reached: water
  temperature 7/24, DO concentration 14/17, specific conductance 8/24, pH 12/19,
  turbidity finished), with roughly 58 more hours estimated, driven mostly by the
  water temperature and specific conductance folds (about 3 h each).
- No Random Forest results were merged or committed for these runs
  (`results/benchmark_results_rain_*_rf` does not exist).

The other arms are unaffected: linear + graph models
(`results/benchmark_results_rain_uncompressed`, `..._compressed`) are committed, and
Gradient Boosting (`..._uncompressed_gb`, `..._compressed_gb`) is committed when it finishes.

To rerun Random Forest later, consider fewer outer folds or trees, or making its
`n_jobs` configurable so more cores can be given to it when the server is free.
