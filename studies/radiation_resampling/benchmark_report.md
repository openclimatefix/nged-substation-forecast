# Radiation resampler benchmark

**Native-window neighbour lookup reduces the cost of interval-aware reconstruction.**
The comparison uses identical shuffled synthetic ECMWF inputs, all 51 members, and
the complete 15-day horizon (85 native steps and 721 output timestamps per trajectory).
Each variant has three fresh-process repetitions, with execution order rotated between
variants. Eight Polars threads are fixed for every worker. No tests ran concurrently.

Host CPU: AMD EPYC 9124 16-Core Processor. Python 3.14.7; Polars 2.0.0.

**Timing includes query construction and collection of every output column.** Input
construction is excluded. Peak RSS is the whole process, including imports, input/output
frames, and result assertions; it is not incremental resampler memory. The inputs contain
short-wave, long-wave, and temperature. Archive I/O, forecast joins, lags, rolling
features, and forecasting-model fitting are excluded. No full-CV timing or forecast
skill is claimed.

| Sites × runs | Output rows | Variant | Median time (s) | Range (s) | Median peak RSS (MiB) |
|---|---|---|---|---|---|
| 10 × 2 | 735,420 | point | 0.061 | 0.059–0.061 | 474.4 |
| 10 × 2 | 735,420 | before | 0.576 | 0.548–0.584 | 936.1 |
| 10 × 2 | 735,420 | after | 0.397 | 0.396–0.456 | 687.4 |
| 50 × 8 | 14,708,400 | point | 1.106 | 1.099–1.138 | 3663.4 |
| 50 × 8 | 14,708,400 | before | 9.821 | 9.692–10.457 | 6703.4 |
| 50 × 8 | 14,708,400 | after | 6.657 | 6.613–6.698 | 5240.5 |

**The variants compare different processing policies on the same native means.**
`point` uses main at `ce7285e4`; `before` uses the radiation PR at `99a40cf5` with
full-target sorts and as-of joins. `after` uses native-window neighbour fills, the
20 W/m² conditioning floor, and lazy schedule validation. All helper hashes and
individual measurements are retained in the accompanying JSON. The comparison does
not isolate the lookup change from the other review fixes. Results at larger scales
must be measured; these two workloads do not establish linear scaling or a
memory cap.
