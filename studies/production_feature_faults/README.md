# Production feature faults

This study investigates Part 1 of issue 1156 through aggregate feature diagnostics and GPU
forecast experiments. The report is
[Production feature faults](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/production-feature-faults/).
All corrections are study-only; production files are unchanged.

## Inputs and scope

Use the quick-wins engineered cache under
`data/studies/per_study/quick_wins/frames/`. The forward fold trains on April 2024–June 2025
control-member rows and evaluates all 51 members of 53 weekly runs in July 2025–June 2026.
The backwards stress test trains on July 2025–June 2026 and evaluates 66 weekly runs in April
2024–June 2025. That chronology is deliberately reversed and is not prospective validation.

Set `DATA_PATH_INTERNAL` to the local data root. The audit and forecasts use
`studies.power.scan_power`. Metadata supplies solar
geometry; the delivery capacity table supplies each series’ error denominator. No downloads,
licences, external service writes, or production changes are required.

The audit reads member 0 only. The forecast baseline uses production’s 24 features, 500 fixed
boosting rounds, and the other settings in `conf/model/xgboost.yaml`. Controlled comparisons
change column subsampling to 1. The `exact_production` CLI setting means a cached-data refit with
production’s column subsampling 0.8; it does not reproduce the operational pipeline. A second
regularised setting comes from `studies.cross_validation.SENSITIVITY_HYPER_PARAMETERS`.
Every model is trained on CUDA, serially, with eight CPU threads, and seeds 0, 1, and 2.

The combined correction changes short-wave reconstruction, interpolated wind directions at both
heights, and windchill outside its domain. Leave-one-correction-out comparisons measure
conditional contributions. Precipitation and weekday masking retain their column positions.
The report explains the cached-native-value and boundary limitations.

## Reproduce

**Run this historical audit from the pinned polar checkout.** Use an isolated worktree at commit
`5ac2e16a7088d13f2f7103f0385a58f54388bfba`, its dependencies, and the matching polar archive and
engineered caches. The raw speed/bearing probes exercise that historical contract and resampler;
they do not exercise the component pipeline. From the current repository root:

```bash
git worktree add --detach .claude/worktrees/historical-feature-faults \
  5ac2e16a7088d13f2f7103f0385a58f54388bfba
cd .claude/worktrees/historical-feature-faults
uv sync
```

Run all commands below from the root of that isolated checkout:

```bash
export DATA_PATH_INTERNAL=/path/to/local/data
export POLARS_MAX_THREADS=8
export MLFLOW_DISABLE_AGENT_HINT=1

uv run python studies/production_feature_faults/audit_production_features.py \
  --frames "$DATA_PATH_INTERNAL/studies/per_study/quick_wins/frames/mid_2025_to_mid_2026" \
  --data-root "$DATA_PATH_INTERNAL" --output data/studies/production_feature_faults

uv run python studies/production_feature_faults/feature_fault_experiments.py \
  --frames "$DATA_PATH_INTERNAL/studies/per_study/quick_wins/frames/mid_2025_to_mid_2026" \
  --output data/studies/production_feature_faults/forward

uv run python studies/production_feature_faults/feature_fault_experiments.py \
  --frames "$DATA_PATH_INTERNAL/studies/per_study/quick_wins/frames/mid_2025_to_mid_2026" \
  --output data/studies/production_feature_faults/forward \
  --settings exact_production --arms baseline all_fixed

uv run python studies/production_feature_faults/feature_fault_experiments.py \
  --frames "$DATA_PATH_INTERNAL/studies/per_study/quick_wins/frames/backwards_2025_07_to_2026_06" \
  --output data/studies/production_feature_faults/backwards --arms baseline all_fixed

uv run python studies/production_feature_faults/report_feature_fault_scores.py \
  --input data/studies/production_feature_faults/forward \
  --output data/studies/production_feature_faults/forward/report

uv run python studies/production_feature_faults/report_feature_fault_scores.py \
  --input data/studies/production_feature_faults/backwards \
  --output data/studies/production_feature_faults/backwards/report --combined-only
```

Use `--prepare-only` to build correction values without fitting. Prepared files contain row
positions and corrected weather only. Their stamps bind source, script, metadata, and helper
hashes. Fit manifests also pin the observation and capacity snapshots. Reusing a changed input
raises rather than silently combining incompatible outputs; choose a fresh output directory.
The runner validates fold dates, series populations, native-grid reconstruction, prediction keys,
complete member labels, and finite predictions. Its known-answer probes cover corrected
radiation, wind wrap, guarded windchill, missing radiation, and truncated boundaries.

Saved scores use a study-specific anonymous 31-series permutation and capacity-normalised losses.
No source identifiers, coordinates, or unnormalised power series are written. Case pairing is
checked again before reporting. The primary estimator averages each series’ row losses, gives
series equal weight, then averages fitting seeds. Bootstrap intervals resample whole valid-time
months and one paired seed, recomputing series means from monthly sums and counts. The additional
month t-interval instead gives months equal weight; its different estimand is explicit.
These are sampled-run study scores, not official leaderboard submissions.

## Figures and evidence

Run the plotting scripts with `--text-only` for prose review before rendering. Then omit that flag:

```bash
uv run python studies/production_feature_faults/plot_production_features.py \
  --input data/studies/production_feature_faults --output docs/studies/assets \
  --png-dir data/studies/production_feature_faults/figures

uv run python studies/production_feature_faults/plot_feature_fault_scores.py \
  --report data/studies/production_feature_faults/forward/report \
  --fits data/studies/production_feature_faults/forward --output docs/studies/assets \
  --png-dir data/studies/production_feature_faults/figures
```

Inspect the PNGs and optimise the SVGs with the repository’s SVGO recipe. The generated
`report.md` files contain every quoted table, all feature lists, case counts, monthly differences,
and correction-support diagnostics. Total-gain importance is descriptive; its plotted variability
uses series-averaged shares across the three seeds, not an uncertainty interval for benefit.

The scripts receive code review before execution, followed by two scientific-validity reviews,
chart inspection, prose/persona/evidence review, and the studies tests. The prescribed Claude
models were unavailable; fresh available reviewers were substituted and the PR records triage.
No shared package changed, so no mutation pass applies.
