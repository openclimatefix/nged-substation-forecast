# Radiation resampler benchmark

**The revised interval-aware resampler uses less time and peak memory than the pre-review
implementation in these two workloads.**
The comparison uses identical shuffled synthetic inputs for the European Centre for
Medium-Range Weather Forecasts ensemble (ECMWF ENS), with all 51 members and the
complete 15-day horizon (85 native steps and 721 output timestamps per trajectory).
Each variant has three fresh-process repetitions, with execution order rotated between
variants. Eight Polars threads are fixed for every worker. No tests ran concurrently.

Host CPU: AMD EPYC 9124 16-Core Processor. Python 3.14.7; Polars 2.0.0.

**Timing includes query construction and collection of every output column.** Input
construction is excluded. Peak resident set size (RSS) covers the whole process,
including imports, input/output frames, and result assertions; it is not incremental
resampler memory. The inputs contain short-wave, long-wave, and temperature.
Archive I/O, forecast joins, lags, rolling
features, and forecasting-model fitting are excluded. No end-to-end cross-validation
timing or forecast skill is claimed.

| Sites × runs | Output rows | Variant | Median time (s) | Range (s) | Median peak RSS (MiB) |
|---|---|---|---|---|---|
| 10 × 2 | 735,420 | point | 0.058 | 0.052–0.061 | 469.4 |
| 10 × 2 | 735,420 | before | 0.562 | 0.549–0.581 | 944.2 |
| 10 × 2 | 735,420 | after | 0.407 | 0.403–0.421 | 693.4 |
| 50 × 8 | 14,708,400 | point | 1.055 | 1.019–1.070 | 3748.1 |
| 50 × 8 | 14,708,400 | before | 9.924 | 9.875–10.122 | 6636.1 |
| 50 × 8 | 14,708,400 | after | 6.706 | 6.652–6.719 | 5289.7 |

**The variants compare different processing policies on the same native means.**
`point` uses main at `ce7285e4`; `before` uses the radiation PR at `99a40cf5` with
full-target sorts and as-of joins. `after` uses native-window neighbour fills, the
20 W/m² conditioning floor, and lazy schedule/coordinate validation. All helper hashes
and individual measurements are retained in the accompanying JSON. The comparison does
not isolate the lookup change from the other review fixes. Results at larger scales
must be measured; these two workloads do not establish linear scaling or a
memory cap.
