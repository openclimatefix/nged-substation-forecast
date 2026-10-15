---
id: "00033"
slug: best-guess-architecture-vs-production
status: planned
size: complex
hypothesis: >-
  The best-guess architecture that the first 32 experiments point to lowers the quantile score
  below the deployed production recipe by at least 1.0 point of capacity for Primary substation
  demand, solar farms, and wind farms, each scored on its own rows, and is level with or better
  than N1, `manual_heuristic`, and `climatology` in each of the three.
prediction: >-
  Primary demand: the best guess scores 5.3, production 7.6, `manual_heuristic` 5.9, N1 6.3
  (experiments 13 and 19). Solar: the best guess scores 5.0 and production at least 6.0. Wind: the
  best guess scores 11.0 and production at least 14. Production is worse than `manual_heuristic`
  for demand (experiment 13) and the ERA5-trained generator models beat the ENS-trained
  production-like model (experiments 1 and 2). The twCRPS at `historical_p99` and the Brier score
  at the same threshold rank the arms as the quantile score does, with the exceedance rate of the
  0.95 quantile closer to 5% for the best guess than for production.
decision_rule: >-
  Three decisive contrasts, the best guess minus production on the quantile score over all leads,
  for Primary demand, solar, and wind, each read at 98.33% (Bonferroni across the three) with
  whole months and seeds resampled. The experiment is supported when all three intervals lie
  wholly below -1.0; it is partly supported when all three lie wholly below zero and at least one
  estimate is above -1.0; it is refuted when any interval lies wholly above zero; otherwise it is
  inconclusive. The threshold of 1.0 point (about 15% of production's Primary score) pays for the
  work of porting a new training recipe, ERA5 training for generators, and the error correction.
  Every other contrast is exploratory within a family, with its own Bonferroni share (see
  "Planned contrasts").
benchmark: >-
  The production recipe (stored `xgboost_baseline`, fold `mid_2025_to_mid_2026`) on the same rows.
row_set_id: fold_a_every5d_v1 (demand, 20 series); fold_a_tuesdays_v1 (solar and wind, 9 series)
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels, the estimate of the
  CRPS), in points of capacity; MAE of the median; twCRPS at each series' `historical_p99`; Brier
  score at the same threshold; exceedance rates of the 0.90 and 0.95 quantiles
advice: []
production_status: not-for-production
adopted_in: none
code: ml-experiments/00033-best-guess-architecture-vs-production/
---

# Experiment 33: does the best-guess architecture beat production, N1, and `manual_heuristic`?

**This page is the plan, committed before any code runs.** The results replace the sections below
when the experiment finishes.

One experiment compares, on one validation year, the architecture that experiments 1 to 32
recommend against the deployed recipe, three baselines, six notable ideas, and a perfect-weather
ceiling. Units are points of capacity (1 point is 1% of a series' effective capacity). The
validation rows are the 73 runs of `fold_a_every5d_v1` for the 20 demand series (16 Primary
substations, 2 BSP, and 2 GSP) and the 53 Tuesday runs of `fold_a_tuesdays_v1` for the 6 solar
farms and 3 wind farms. No valid time on or after 2026-07-01 is scored.

## Arms

**An arm whose recipe, rows, and seeds an earlier experiment already ran is read from that
experiment's stored predictions, not refitted.** `experiment_store.read_predictions` refuses sets on
a different row set, and every reused set has three seeds. The table names the source of each
reused arm. Every fit below is new code in this experiment's folder.

| Family | Arm | Source |
|---|---|---|
| Primary demand | **Best guess:** N1 + 14-day correction + spread 1.15 | reused: 00019 `stack_minus_h` |
| Primary demand | N1 (no correction), N1 + correction | reused: 00013 `n1`, 00019 `n1_c` |
| BSP, GSP demand | **Best guess:** N1, no correction; N1 + correction and N1 + correction + spread as arms | reused: 00013 `n1`; correction and spread computed here |
| All demand | Production (`xgboost_baseline`), `manual_heuristic`, `climatology`, each with and without the 14-day correction | reused: 00013 `production`, `manual_heuristic`, `climatology`; the correction is computed here with 00019's code |
| All demand | Notable 1: weekly analogue mean and median, with and without correction | reused: 00030 `a7`, `a7_c` |
| All demand | Notable 2: one model per substation type with the series identifier, with and without the correction | reused: 00024 `type_id`, `type_id_stack` (the stack adds the correction and spread) |
| Primary demand | Notable 3: N1 retrained monthly with a 0.5-year half-life, plus correction | **new fit**, below |
| Solar | **Best guess:** ERA5-trained mixture with spread factor | reused: 00020 `base-spread` |
| Wind | **Best guess:** ERA5-trained mixture with power-curve margin and spread factor | reused: 00020 `full` |
| Solar, wind | Production (`xgboost_baseline`), `manual_heuristic`, `climatology` | stored forecasts; read for the Tuesday runs, quantiles of the stored members at the 9 levels |
| Solar, wind | N1 for generators | **new fit**, below |
| Solar | Notable 4: CAMS-trained mixture | reused: 00020 `cams` |
| Solar | Notable 6: PV physics fitted with SciPy, and the physics output as a column in the XGBoost model | reused: 00029 `scipy` and `torch-hybrid` (see below) |
| Wind | Notable 5: best guess plus `power_to_wind3_q75_28d` | reused: 00026 `a4` |

**Two substitutions are named here so that the page cannot hide them.** Notable 5 is 00026's arm
A4, which adds the `top_decile_28d` column as well as `power_to_wind3_q75_28d`; no stored arm adds
the second column alone. Notable 6 uses the stored PyTorch-fitted hybrid, because 00029 found the
SciPy and PyTorch fits level on score (C3, -0.005, -0.020 to +0.010) and stored no SciPy hybrid.
Both are re-run only if the maintainer asks.

**Production for the generators is read, not refitted.** The `power_forecasts` Delta table holds
`xgboost_baseline` forecasts for all 31 series, including the 9 generators, on fold
`mid_2025_to_mid_2026`. Where a stored production forecast lacks a Tuesday run, the arm is scored
on the cases all arms hold, and `result.md` states the count of dropped cases.

**N1 for generators** is the 00013 recipe on each generator: one XGBoost model per farm with the
9-level quantile loss, trained on the ENS control member of every daily run with valid times from
2024-04-01 to 2025-06-30, recency weights with a half-life of 2 years anchored at 2025-06-30, the
technology's common columns (nine for solar, six for wind), `lead_hours`, and three power lags from
the forecast origin (newest usable observation at origin minus 3.5 hours, latest earlier same
half-hour, latest earlier same weekday and half-hour). It has no calendar columns, because experiment
1 found calendar columns let an ENS-trained generator model memorise valid times (+3.40 wind, +1.13
solar). The 51 member forecasts are pooled as in 00013. No correction is applied.

**Notable 3 (monthly retraining)** refits each Primary substation's N1 model on the first day of
each month from 2025-07 to 2026-06 with a 0.5-year half-life anchored at the refit date, using the
ENS control member of every run whose valid time is at least 3.5 hours before the refit. After
2025-06-30 the only ENS runs on disk are the 73 validation runs, so refits from 2025-08 onward add
at most 73 runs' control members, a sparser sample than the daily runs before. The arm is scored
with the 14-day correction. It uses seed 0 only (16 series x 12 refits is about 1 hour); the
interval for it resamples months only, and the page says so.

## Perfect-weather arm

**The perfect-weather arms measure the headroom from better weather, not a deployable model.** ERA5
reaches us about 5 days late and CAMS is satellite-derived after the fact, so no serving system
could use either at the target time. The page labels every perfect-weather result "ceiling".

- **Demand:** the best-guess model (N1 + correction + spread for Primary; N1 for BSP and GSP),
  refitted as N1 with a 2-year half-life, replaces the 7 forecast weather columns with ERA5 at the
  valid time (same code path as 00022, N1 and not N1H); the 51 members collapse to one. Variants:
  `shortwave` from CAMS at the series' own location with ERA5 for the rest, and the model retrained
  on ERA5 over the same 2024-04 to 2025-06 window and fed ERA5. Single-variable swaps, keeping ENS
  members for the other columns: `shortwave` (ERA5), `temperature_2m`, `wind_speed_10m`. The
  correction history comes from the arm's own past forecasts.
- **Solar:** the ERA5-trained model of the best guess is fed ERA5 at the valid time (one input per
  valid time, so the forecast is the model's 9 quantiles); a CAMS-trained variant is fed CAMS
  `shortwave` with ERA5 for the other eight columns. Single-variable swap: `shortwave` only.
- **Wind:** ERA5 10 m and 100 m wind speed and direction at the valid time, with the power-curve
  margin reading ERA5 100 m speed; single-variable swap: 100 m wind speed alone.
- **Spread factor:** for solar and wind, each perfect-weather arm is reported with and without the
  best guess's spread factor, refitted monthly on the arm's own earlier cases as 00020 did. A
  single-input band is narrower than a pooled mixture, so the unrefitted factor may mislead.
- **Lead bands:** 0 to 24 h, 1 to 2 d, 3 to 6 d, 7 to 10 d, 11 to 15 d. For solar and wind the
  perfect-weather arm does not depend on lead, so the by-lead gap is the best guess's degradation.
- **Weather available at forecast time:** none of ERA5 or CAMS. Lagged power uses the observation
  at least 3.5 hours old, as in 00013; no other lag-time weather enters any arm.

## Metrics

All arms store 9 quantiles at 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, and 0.95. For
`manual_heuristic`, `production`, and `climatology`, the quantiles are those of the stored members at
the 9 levels, as 00013 took them.

- **MAE of the median:** the absolute error of `q50`.
- **Quantile score (the CRPS estimate):** as in 00019.
- **twCRPS at `historical_p99`:** replace every quantile and the observation by `max(z, r)` and take
  the quantile score. `r` is the 99th percentile of the series' power over capacity in the constraint
  direction (high positive power for demand and for generation) over the training years, which end
  2025-06-30, so the threshold uses no validation observation. `historical_p95` is the exploratory
  second line.
- **Brier score at the same threshold:** the exceedance probability `p(r)` comes from the quantile
  function: linear interpolation of the level against the quantile between `q05` and `q95`; below
  `q05` the probability is 1; above `q95` the probability falls by half for every width `q95 - q90`
  beyond `q95`, starting from 0.05 (an exponential tail), and is 0 beyond `q95` when `q95 = q90`.
  This tail is an approximation, so the Brier score is read alongside the exceedance rates below.
- **Exceedance rates of the 0.90 and 0.95 quantiles:** the share of cases where the observation
  exceeds that quantile; the aim is 10% and 5%. These need no model of the tail.
- **Thresholds and the flexibility use case:** NGED buys flexibility ahead of an expected
  constraint, so the exceedance metrics are also reported for leads of 1 to 2 days and 3 to 6 days,
  the horizons at which procurement decisions are made. Metrics are never ranked on half-hours chosen
  by the observed load.
- **Absolute scores for every arm** are reported, not only contrasts. BSP, GSP, solar, and wind are
  reported per type and never averaged into Primary.
- **Intervals:** month-block bootstrap, whole validation months and the three seeds resampled, with
  `studies.bootstrap.bootstrap_difference_at_level` and `n_resamples` raised for the Bonferroni
  level.

## Planned contrasts

**Bonferroni shares are set within five families**, each family's shares being 5% divided by its
contrast count.

1. **Decisive (3):** best guess minus production for Primary demand, solar, and wind.
2. **Primary demand (7):** best guess minus `manual_heuristic`, N1, and `climatology`; N1 + correction
   minus N1; production + correction minus production; and the notable arms 1, 2, and 3, each minus
   the same arm with the N1 + correction base.
3. **BSP and GSP (4, pooled over the 4 series):** N1 minus production, `manual_heuristic`, and
   `climatology`; N1 + correction minus N1.
4. **Generators (8):** for solar and wind separately, the best guess minus N1, `manual_heuristic`, and
   `climatology`; solar notable 4 minus the best guess; wind notable 5 minus the best guess.
5. **Perfect weather (6):** for Primary demand, solar, and wind, the all-variable ERA5 (and CAMS)
   arm minus the best guess on the quantile score; and the same for the demand arm retrained on
   ERA5.

Scores by lead band, by variable, at `historical_p95`, and for the exceedance
rates are exploratory and carry no interval claim beyond a stated interval.

## Figures

- **Headline:** quantile score of every arm per family with 98.33% intervals on the decisive
  contrasts, at the top of the page.
- **Metrics:** twCRPS, Brier score, and exceedance rates of the 0.90 and 0.95 quantiles per arm.
- **By lead band:** best guess, production, N1, `manual_heuristic`.
- **Perfect weather:** gap to the best guess by lead band and by variable.
- **Fan charts (the 9 quantiles and the actual):** several series per family on good and bad days,
  sunny and cloudy, windy and calm, a cold snap; the best guess beside production. Generators carry
  anonymised labels ("Solar farm A", "Wind farm B") from `studies.anonymise`, never a name or an
  identifier; substations may be named.

## Plan, written before any code ran

- **Leakage checks.** Every refit stops at its stated training end; lags and the correction use
  observations at least 3.5 hours before the origin; ERA5 and CAMS values are those of the valid
  time only and are used in the perfect-weather arms alone; the thresholds use training-year data.
  A reviewer checks the monthly-retraining code for leakage.
- **Reproduction checks.** The reused demand arms must reproduce 00019's stored scores to 3 decimal
  places when rescored with this experiment's scorer, and the refitted N1 must match 00013's `n1`
  within 0.02 points.
- **Weather source at lag times.** No arm uses weather at lag times; only the power lags read the
  past, and they read observed power available at the forecast origin.
- **Compute.** About 2.5 hours in total on slots A and B (22 cores, one fit at a time on the GPU):
  generator N1 (9 series x 3 seeds, minutes); demand perfect-weather fits (N1 refit, 20 series x 3
  seeds, about 20 minutes, and the ERA5-retrained variant, about 20 minutes); monthly retraining
  (about 1 hour, seed 0); generator perfect-weather predictions (minutes). If the fits would run
  more than 5 hours in total, the plan proposes a reduction to the manager instead of cutting arms
  silently.

### Sizing

- **Stored data:** prediction sets under `data/ml-experiments/00033-.../predictions/`; no table.
- **Production serving path:** not touched; the production arm is read from stored forecasts.
- **Degradation rule:** not touched.
- **Defensible designs:** more than one could compare the architectures (for example the choice
  of thresholds, the tail of the exceedance probability, and the reused-versus-refitted arms), so
  the plan is reviewed.
- **Callers:** the scripts copy the helpers of experiments 13, 19, 20, and 22; the span of code is
  wide, so the size is complex.
