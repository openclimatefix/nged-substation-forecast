# Choosing how to interpolate wind components

Compare linear, shape-preserving cubic Hermite (PCHIP), and modified Akima (Makima)
interpolation of eastward u and northward v. Derive speed and meteorological bearing after
interpolation. This focused experiment leaves production interpolation unchanged.

## Planned comparison

- Read a pinned component NWP snapshot, ECMWF ENS control member 0, at the three wind farms'
  existing H3 cells, retaining only 00:00 UTC runs. Use instantaneous anchors at `valid_time`, without extrapolation.
- Use every complete native run for each site. Interpolate the full 85-step forecast, then score
  half-hourly leads 24–47.5 h and 168–191.5 h separately, covering 3-hour and 6-hour spacing.
- Give each per-farm XGBoost model exactly four columns: `wind_speed_10m`,
  `wind_direction_10m`, `wind_speed_100m`, and `wind_direction_100m`. Bearings stay in degrees,
  as requested. There are no calendar, lag, solar, demand, or extra wind inputs.
- Use the registered chronological fold: nominal training starts 2024-04-01 and validation spans
  2025-07-01 through 2026-06-30. Purge training targets at or after 2025-06-24 00:00 UTC in both
  bands, before the earliest seven-day validation initialisation. Fit a separate model per farm
  and lead band; preserve `(site, init_time, valid_time, band)` keys throughout.
- Read cleaned power through `studies.power.scan_power`, retain zero readings, and normalise
  each farm by its maximum cleaned training power before the embargo (a capacity proxy, not a
  verified physical capacity). Record missing
  targets and complete-run exclusions. Use exactly the same rows and row order for all methods.
- Fit on the RTX A6000 GPU with the shared primary and sensitivity XGBoost settings, all columns
  available to each tree, and seeds 0, 1, and 2. No tuning or early stopping. Clip negative power
  predictions to zero, but do not impose an observed validation maximum.
- The four planned contrasts are PCHIP minus linear and Makima minus linear at each lead band.
  Report absolute mean absolute error (MAE) as percentage points of training capacity, plus paired
  differences and 95% intervals resampling whole validation months and one fitting seed (2,000
  resamples). Pool the farms on matched forecast keys within each band; intervals describe these three farms,
  weather months and fitting noise, not a population of wind farms. The four intervals are nominal,
  with additional 98.75% intervals correcting the four contrasts by Bonferroni. Prefer linear
  unless both settings support a consistent reduction; do not infer equivalence from a null result. Report each farm separately as descriptive context.
- As an independent interpolation check, hide the 3-hour samples between 6-hour native anchors
  through lead 144 h and compare reconstructed u/v with those withheld native values. This tests
  6-to-3-hour reconstruction, not actual half-hourly wind truth or power accuracy.
- Verify known affine vectors, native-anchor reproduction, and finite values. Save anonymised
  inputs, held-out predictions, losses, feature gains, provenance, and report under
  `data/studies/per_study/wind_interpolation/`, with no overwrite of published results.

## Deliverable and checks

Write a short docs explanation with plots of interpolation behaviour and the paired power-error
comparison, whatever the outcome. Keep script and results reproducible; no production-policy change
is implied. The user explicitly requested a quick, focused experiment, so omit the full study
persona suite and long paper format. Keep two independent scientific reviews and script/diff
review. Do not submit these restricted three-farm, four-feature predictions as leaderboard scores.

Process size is complex because results are stored and several methods could satisfy the question.
Production serving and degradation rules are untouched; no shared package logic changes. No
mutation pass is needed for isolated study scripts. Run study-boundary/tests, lint, typing, Markdown
and strict docs/link checks before pushing. Review figure text before rendering, inspect PNGs,
optimise SVGs, and check each published number against the script-generated report.
