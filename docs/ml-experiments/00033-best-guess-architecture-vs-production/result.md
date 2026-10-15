---
id: "00033"
slug: best-guess-architecture-vs-production
status: planned
size: complex
hypothesis: >-
  The best-guess architecture that the first 32 experiments point to lowers the quantile score
  below the deployed production recipe by more than the threshold set for each technology (1.0
  point of capacity for Primary substation demand, 0.5 for solar farms, 1.0 for wind farms), each
  scored on its own rows.
prediction: >-
  Primary demand: the best guess scores 5.3 and production 9.52 (experiments 13 and 19), with
  `manual_heuristic` at 5.9 and N1 at 6.3. Solar: the best guess scores 5.0 and production at least
  5.7, so the expected verdict for solar is a small gain at best. Wind: the best guess scores 11.0
  and production at least 14. Production is worse than `manual_heuristic` for demand (experiment
  13), and the ERA5-trained generator models beat ENS-trained models (experiments 1 and 2). The
  twCRPS and the Brier score at `historical_p99` rank the arms as the quantile score does, and the
  exceedance rate of the 0.95 quantile is closer to 5% for the best guess than for production.
decision_rule: >-
  Three decisive contrasts, the best guess minus production on the quantile score over all leads,
  for Primary demand, solar, and wind, each read at 98.33% (Bonferroni across the three) with whole
  months and seeds resampled. Per technology the verdict is supported when the interval lies wholly
  below minus the threshold; small gain when it lies wholly below zero but not wholly below minus
  the threshold; refuted when it lies wholly above zero; otherwise inconclusive. The experiment is
  supported when all three are supported. The thresholds justify the effort of porting a new
  training recipe, ERA5 training for generators, and the error correction; Primary's 1.0 point is
  about 10% of production's 9.52. Every other contrast is exploratory (see "Planned contrasts").
benchmark: >-
  The production recipe (stored `xgboost_baseline`, fold `mid_2025_to_mid_2026`) on the same rows.
row_set_id:
  - fold_a_every5d_v1
  - fold_a_tuesdays_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
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
| Primary demand | **Best guess:** N1 + 14-day correction + spread factor 1.15 | reused: 00019 `stack_minus_h` |
| Primary demand | N1 (no correction), N1 + correction | reused: 00013 `n1`, 00019 `n1_c` |
| BSP, GSP demand | **Best guess:** N1, no correction; N1 + correction and N1 + correction + spread as arms | reused: 00013 `n1`; correction and spread computed here |
| All demand | Production (`xgboost_baseline`), `manual_heuristic`, `climatology`, each with and without the 14-day correction | reused: 00013 `production`, `manual_heuristic`, `climatology`; the correction is computed here with 00019's code |
| All demand | Notable 1: weekly analogue mean and median, with and without correction | reused: 00030 `a7`, `a7_c` |
| All demand | Notable 2: one model per substation type with the series identifier, with and without correction | reused: 00024 `type_id`, `type_id_stack`; contrast bases `own`, `own_stack` |
| Primary demand | Notable 3: N1 retrained monthly with a 0.5-year half-life, plus correction | **new fit**, below |
| Solar | **Best guess:** ERA5-trained mixture with spread factor | reused: 00020 `base-spread` |
| Wind | **Best guess:** ERA5-trained mixture with power-curve margin and spread factor | reused: 00020 `full` |
| Solar, wind | Production (`xgboost_baseline`), `manual_heuristic`, `climatology` | stored forecasts; read for the Tuesday runs, quantiles of the stored members at the 9 levels |
| Solar, wind | N1 for generators | **new fit**, below |
| Solar | Notable 4: CAMS-trained mixture with spread factor | reused: 00020 `full-no-s0` |
| Solar | Notable 6: PV physics, and the physics output as a column in the XGBoost model | reused: 00029 `scipy`, `torch-hybrid`; contrast base 00020 `base` |
| Wind | Notable 5: margin plus `top_decile_28d` and `power_to_wind3_q75_28d` | reused: 00026 `a4`, `a0` |

**Four substitutions are named here.** The stored arms differ from the notable ideas in four ways.

- Notable 4 is 00020 `full-no-s0` (CAMS-trained plus the spread factor), so that it shares the best
  guess's spread factor.
- Notable 5 is 00026's arm A4 (wind), which adds `top_decile_28d` as well as
  `power_to_wind3_q75_28d` and has the power-curve margin but no spread factor. Its contrast is A4
  minus A0 within 00026, where A0 equals 00020 `margin`.
- Notable 6 is the stored PyTorch-fitted hybrid `torch-hybrid`, because 00029 found the SciPy and
  PyTorch fits level on score (C3, -0.005, -0.020 to +0.010) and stored no SciPy hybrid. The hybrid
  has no spread factor, so its contrast is against 00020 `base`.
- Notable 2's stored arms use a half-life of 0.5 years, and `type_id_stack` adds the correction and
  the spread factor. Its contrasts are against 00024's own per-series arms (`own`, `own_stack`).

The maintainer can ask for any of these to be re-run in the exact form.

**Production for the generators is read, not refitted.** The `power_forecasts` Delta table holds
`xgboost_baseline` forecasts for all 31 series, including the 9 generators, on fold
`mid_2025_to_mid_2026`. The arm uses `experiment_name` `xgboost_baseline`, the recipe experiment 13
scored as production for demand. The table also holds `xgboost_baseline_retrain_638` and
`xgboost_baseline_retrain_790`, reported as exploratory rows. Where a stored production forecast
lacks a Tuesday run, the arm is scored on the cases all arms hold, and `result.md` states the count
of dropped cases.

**N1 for generators uses the 00013 recipe with no calendar columns.** The recipe, applied to each
generator, is one XGBoost model per farm with the nine-level quantile loss, trained on the ENS control
member of every daily run with valid times from 2024-04-01 to 2025-06-30, recency weights with a
half-life of 2 years anchored at 2025-06-30, the technology's common columns (nine for solar, six
for wind), `lead_hours`, and three power lags from the forecast origin (newest usable observation at
origin minus 3.5 hours, latest earlier same half-hour, latest earlier same weekday and half-hour).
The calendar columns are left out, because experiment 1 found calendar columns let an ENS-trained
generator model memorise valid times (+3.40 wind, +1.13 solar). The 51 member forecasts are pooled
as in 00013. No correction is applied to N1 for generators.

**Monthly retraining refits each Primary substation's N1 model on the first day of each month.**
The refits run from 2025-07 to 2026-06 with a half-life of 0.5 years anchored at the refit date,
on the ENS control member of every daily run (the NWP store holds a daily run from 2024-04-01) with
valid times at least 3.5 hours before the refit. A run is forecast by the latest refit on or before
its origin, so the refit is chosen by origin and not by valid time. The 2025-07-01 refit equals the
half-life-0.5-year fit trained to 2025-06-30. The 14-day correction history spans a change of model
at each month boundary, and the page says so. The arm is reported with and without the correction.
Building control-member features for the daily runs after 2025-06-30 is new code; if that costs
more than about 2 hours, the arm falls back to the 73 validation runs' control members and the page
says so. The arm uses seed 0 only (about 1 hour for 16 series by 12 refits), so it is contrasted
with the seed-0 rows of the reference arms, and its interval resamples months only.

## Perfect-weather arms

**The perfect-weather arms measure the headroom from better weather, not a deployable model.** ERA5
and CAMS are estimates of past weather, not ground truth, so figures label the arms "ERA5-weather
ceiling". ERA5 reaches us about 5 days late and CAMS is satellite-derived after the fact, so no
serving system could use either at the target time. The page labels every perfect-weather result
"ceiling".

- **Demand:** the arms `ens`, `era5_all`, `era5_shortwave`, `era5_temperature_2m`, and
  `era5_wind_speed_10m` of 00022 are reused for all 20 series (N1H; 00019 C5 found the half-life
  neutral). Their contrasts are against 00022's own `ens`, never the 00019 set. Each is reported
  with and without the correction, and the correction and spread factor are applied here wherever
  00022's stored set lacks them. Two variants are new fits: `shortwave` from CAMS at the series'
  own location with ERA5 for the other six columns, and the model retrained on ERA5 over the same
  2024-04 to 2025-06 window and fed ERA5. For BSP and GSP the arms are N1 without correction.
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
- **Weather available at forecast time:** neither ERA5 nor CAMS is available at forecast time.
  Lagged power uses the observation at least 3.5 hours old, as in 00013; no other lag-time weather
  enters any arm.

## Metrics

All arms store 9 quantiles at 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, and 0.95. For
`manual_heuristic`, `production`, and `climatology`, the quantiles are those of the stored members
at the 9 levels, as 00013 took them.

- **MAE of the median:** the absolute error of `q50`.
- **Quantile score (the CRPS estimate):** as in 00019.
- **twCRPS at `historical_p99`:** replace every quantile and the observation by `max(z, r)` and take
  the quantile score. `r` is the 99th percentile of the series' power over capacity in the
  constraint direction (high positive power for demand and for generation) over the series' full
  history to 2026-06-30, as `evaluation-metrics.md` defines it. The threshold is a fixed scoring
  constant that no model sees. The page reports, per series, the count of half-hours and of distinct
  days above the threshold. For the generators the threshold sits near capacity (capacity is itself
  a P99), so `historical_p95` is a co-primary line for solar and wind and an exploratory line for
  demand.
- **Brier score at the same threshold:** the exceedance probability `p(r)` comes from the quantile
  function: linear interpolation of the level against the quantile between `q05` and `q95`; above
  `q95` the probability falls by half for every width `q95 - q90` beyond `q95`, starting from 0.05
  (an exponential tail), and is 0 beyond `q95` when `q95 = q90`; below `q05` the exceedance
  probability is `1 - 0.05 * 2^(-(q05 - r) / (q10 - q05))`. The tails are approximations, and at
  `historical_p99` most cases lie in the upper tail, so the Brier score there mainly measures the
  tail assumption. The page therefore also reports the Brier score at `historical_p95`, which is
  interpolated, and the hit rate and false-alarm count at the threshold for the 1-to-2-day and
  3-to-6-day bands. The twCRPS ignores all mass above `q95`.
- **Exceedance rates of the 0.90 and 0.95 quantiles:** the share of cases where the observation
  exceeds that quantile; the aim is 10% and 5%. The rates need no model of the tail, but they are
  calibration checks and not tail-skill rankings. The page reports each stored arm's member count;
  cases of `manual_heuristic` with fewer than 13 members are flagged, and its rates are read against
  a finite-ensemble reference.
- **Thresholds and the flexibility use case:** NGED buys flexibility ahead of an expected
  constraint, so the exceedance metrics are also reported for leads of 1 to 2 days and 3 to 6 days,
  the horizons at which procurement decisions are made. Metrics are never ranked on half-hours
  chosen by the observed load.
- **Absolute scores for every arm** are reported, not only contrasts. BSP, GSP, solar, and wind are
  reported per type and never averaged into Primary.
- **Intervals:** month-block bootstrap, whole validation months and the three seeds resampled, with
  `studies.bootstrap.bootstrap_difference_at_level` and `n_resamples` raised for the Bonferroni
  level. Intervals resample months, not series, so with three wind farms they understate
  between-farm uncertainty.

## Planned contrasts

**Only C1 to C3 decide the verdict, so only they carry the Bonferroni correction across the three
(98.33%).** The other contrasts are exploratory; each family below reads at 5% divided by its
contrast count, so a reader sees which intervals survive a correction within the family.

1. **Decisive (3):** C1 to C3, the best guess minus production for Primary demand, solar, and wind.
2. **Primary demand (9):** C4 to C6, the best guess minus `manual_heuristic`, N1, and `climatology`;
   C7, N1 + correction minus N1; C8, production + correction minus production; C9 to C11, notable
   arms 1, 2, and 3 (with correction) minus N1 + correction, `own_stack`, and N1 + correction at
   seed 0; C12, monthly retraining without correction minus N1 at seed 0.
3. **BSP and GSP (4, pooled over 4 series):** C13 to C15, N1 minus production, `manual_heuristic`,
   and `climatology`; C16, N1 + correction minus N1.
4. **Generators (10):** for solar and wind separately, C17 to C22, the best guess minus N1,
   `manual_heuristic`, and `climatology`; C23, solar notable 4 minus the best guess; C24, wind
   notable 5 (A4) minus A0; C25, solar `torch-hybrid` minus 00020 `base`; C26, solar `scipy` minus
   00020 `base`.
5. **Weather ceiling (6):** C27, the all-variable ERA5 arm minus 00022 `ens` for Primary demand
   (with correction); C28 and C29, the ERA5 (solar: and CAMS) arm minus the best guess for solar and
   wind; C30, the demand arm retrained on ERA5 minus its ENS counterpart; C31, the CAMS-shortwave
   demand arm minus `ens`; C32, the BSP and GSP all-variable ERA5 arm minus N1.

**Several contrasts are expected to be inconclusive.** Notable 1 (-0.05 in 00030), notable 4
(-0.009, -0.127 to +0.072 in 00020), and notable 5 (-0.048 in 00026) will not resolve at a share of
one tenth or less, and the BSP and GSP correction (+0.36, -0.19 to +0.98 in 00021) is unresolved
with 4 series. Scores by lead band, by variable, at `historical_p95`, and for the exceedance rates
are exploratory.

## Advice followed and departed from

- **Followed:** use ERA5-trained generator models with no calendar columns and no stage 2 or
  correction (00001, 00023); keep the demand correction at 14 days, additive (00019, 00021); leave
  the correction out of the best guess for BSP and GSP (00021).
- **Departed from:** 00019 C5 shows a 0.5-year half-life on a frozen model changes nothing; notable
  2 uses it because 00024 ran it, and notable 3 tests the half-life with monthly retraining, which
  00019 never did.

## Figures

- **Headline:** quantile score of every arm per family with 98.33% intervals on the decisive
  contrasts, at the top of the page.
- **Metrics:** twCRPS, Brier score, and exceedance rates of the 0.90 and 0.95 quantiles per arm.
- **By lead band:** best guess, production, N1, `manual_heuristic`.
- **Perfect weather:** gap to the best guess by lead band and by variable.
- **Fan charts (the 9 quantiles and the actual):** several series per family on good and bad days,
  sunny and cloudy, windy and calm, a cold snap; the best guess beside production. Generators carry
  anonymised labels ("Solar farm A", "Wind farm B") from `studies.anonymise`, never a name or an
  identifier; substations may be named. Generator charts follow the `dataviz` skill: the axis shows
  days 1 to 7 and no calendar date, the text gives month and year only, `aria` is off, values are
  fractions of capacity and not megawatts, and no caption gives the date of a "windy" or "cold" day.

## Plan, written before any code ran

- **Leakage checks.** Every refit stops at its stated training end; lags and the correction use
  observations at least 3.5 hours before the origin; ERA5 and CAMS values are those of the valid
  time only and are used in the perfect-weather arms alone; the thresholds use training-year data.
  A reviewer checks the monthly-retraining code for leakage.
- **Reproduction checks.** The reused demand arms must reproduce 00019's stored scores to 3 decimal
  places on 00019's exact case set when rescored with this experiment's scorer; each family is then
  scored on its own case intersection and the page reports the dropped counts, and the refitted N1
  must match 00013's `n1` within 0.02 points.
- **Weather source at lag times.** No arm uses weather at lag times; only the power lags read the
  past, and they read observed power available at the forecast origin.
- **Compute.** About 3 hours in total on slots A and B (22 cores, one fit at a time on the GPU):
  generator N1 (9 series x 3 seeds, minutes); demand perfect-weather fits (N1 refit, 20 series x 3
  seeds, about 20 minutes, and the ERA5-retrained variant, about 20 minutes); monthly retraining
  (about 1 hour, seed 0, plus the feature build); generator perfect-weather predictions (minutes).
  If the fits would run more than 5 hours in total, the plan proposes a reduction to the manager
  instead of cutting arms silently.

### Sizing

- **Stored data:** prediction sets under `data/ml-experiments/00033-.../predictions/`; no table.
- **Production serving path:** not touched; the production arm is read from stored forecasts.
- **Degradation rule:** not touched.
- **Defensible designs:** more than one could compare the architectures (for example the choice
  of thresholds, the tail of the exceedance probability, and the reused-versus-refitted arms), so
  the plan is reviewed.
- **Callers:** the scripts copy the helpers of experiments 13, 19, 20, and 22; the span of code is
  wide, so the size is complex.
