# Radiation resampling illustration

The synthetic examples call the public production feature pipeline and compare it with point
interpolation on identical native interval means. The references use independent pvlib solar
geometry, constant attenuation, a sunny-morning/dull-afternoon transition, and an Ineichen
clear-sky reference whose shape differs from production, plus an invented
long-wave curve. The comparison checks timing and short-wave conservation,
not forecast skill or recovery of unknown sub-window changes. The [architecture
explanation](https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/#radiation-resampling)
includes the figure and its limitations.

Run from the repository root:

```bash
uv run python studies/radiation_resampling/plot_radiation_resampling.py
npx svgo@4 --multipass --precision=1 --final-newline docs/architecture/assets/radiation-resampling.svg
```

The script writes the SVG and PNG in `docs/architecture/assets/` and the numerical checks in
`studies/radiation_resampling/report.md`. Both native step widths use a complete shared middle day;
centroid shifts include all 48 half-hours. Float64 reductions compare the reconstructed values with
the supplied Float32 native means. Long-wave uses piecewise cubic Hermite interpolating polynomial
(PCHIP) interpolation at half-hour
midpoints without enforcing interval energy.
Assertions stop generation if short-wave conservation or constant-attenuation timing checks fail.

## Bulk resampler benchmark

**The benchmark measures in-memory resampling time and whole-process peak memory.** The
[generated report](benchmark_report.md) and [raw measurements](benchmark_report.json) compare
point interpolation, the PR before the external-review fixes, and native-window neighbour lookup.
The workload includes both radiation columns and temperature, all 51 members, and the full
15-day horizon. Archive reads and the rest of feature engineering are excluded.

Create detached source checkouts, sharing the current environment's dependencies:

```bash
git worktree add --detach .claude/worktrees/radiation-benchmark-point ce7285e4
git worktree add --detach .claude/worktrees/radiation-benchmark-before 99a40cf5
```

Run each workload without concurrent tests or other CPU-heavy work. The script rotates variant
order across three fresh-process repetitions and verifies each worker's imported source path:

```bash
uv run python studies/radiation_resampling/benchmark_radiation_resampling.py \
  --sites 10 --runs 2 --threads 8 --label small \
  --compare-sources \
    point=.claude/worktrees/radiation-benchmark-point/packages/ml_core/src \
    before=.claude/worktrees/radiation-benchmark-before/packages/ml_core/src \
    after=packages/ml_core/src \
  --output /tmp/radiation-benchmark/small.json
uv run python studies/radiation_resampling/benchmark_radiation_resampling.py \
  --sites 50 --runs 8 --threads 8 --label large \
  --compare-sources \
    point=.claude/worktrees/radiation-benchmark-point/packages/ml_core/src \
    before=.claude/worktrees/radiation-benchmark-before/packages/ml_core/src \
    after=packages/ml_core/src \
  --output /tmp/radiation-benchmark/large.json
uv run python studies/radiation_resampling/benchmark_radiation_resampling.py \
  --summarize-directory /tmp/radiation-benchmark
```

The installed environment is shared so dependency changes cannot explain a variant difference.
The JSON records Python and Polars versions, thread count, and implementation hashes. These two
synthetic workloads do not establish full-CV throughput or a bound on memory at larger scales.
