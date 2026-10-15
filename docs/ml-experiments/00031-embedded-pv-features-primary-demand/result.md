---
id: "00031"
slug: embedded-pv-features-primary-demand
status: planned
size: complex
hypothesis: >-
  For the 16 Primary substation demand series, giving the single-stage ENS-trained quantile model N1
  trailing estimates of each substation's embedded (unmetered) solar capacity, built from the
  Embedded Capacity Register (ECR) as a lower bound plus a non-negative amount learned from recent
  power and irradiance, lowers the quantile score on sunny days, and a linear PV offset or a
  sunshine-shaped correction does so more reliably than extra tree columns.
prediction: >-
  All-lead Primary gains against the 00019 stack of 0 to -0.15 points of capacity, concentrated in
  daylight with a clear-sky index of 0.7 or more from April to September (-0.3 to -1.0 there) at
  the Primaries with the most registered solar, and near zero at night. The sunshine-shaped
  correction (A5) is the likeliest winner, because the 14-day correction already removes level
  drift. The shuffled control (A6) gains nothing.
decision_rule: >-
  Six planned contrasts on the 16 Primary substations, quantile score over all leads, read at
  99.17% (Bonferroni across six), whole months and the three seeds resampled. C1 is A1 minus B0, C2
  is A2 minus B0, C3 is A3 minus B0, C4 is A4 minus B0, C5 is A5 minus B0, and C6 is A3 minus A6.
  A4 is refitted on the 8 directional Primaries only, and the 8 non-directional Primaries reuse
  B0's rows, so C4 is pooled over all 16 and diluted by half; the 8-series contrast is reported
  beside it and decides nothing. Each arm has one outcome. It is "port" when its interval is wholly
  below zero and its estimate meets its threshold, "small gain" when the interval is wholly below
  zero and the estimate does not, "harm" when the interval is wholly above zero, and "null"
  otherwise. The thresholds are -0.02 points for A1 and A2 (extra columns in the existing
  pipeline), -0.05 for A5 (a new pipeline stage), and -0.10 for A3 and A4 (a new data source, the
  ECR, in serving); the -0.02 tier does not apply to A3. A2 and A3 also need C6 wholly below zero
  to be "port"; if C6 is not, the arm is "small gain" at best, because the gain then comes from the
  physics shape or a date fingerprint. The verdict is `supported` when at least one arm is "port",
  `small gain` when none is but at least one is "small gain", `refuted` when no arm is "port" or
  "small gain" and every null interval has a half-width under 0.05 (so a gain of that size is
  excluded), and `inconclusive` otherwise. The prediction (0 to -0.15 all-lead) mostly sits below
  the
  bars, so `small gain` is the expected verdict. A verdict never rests on a breakdown. The mean
  absolute error of the median decides nothing, nor do A7 and the no-fit baseline.
benchmark: >-
  B0, the 00019 stack (N1 with the 14-day correction and the 1.15 spread factor); also
  `manual_heuristic`,
  `climatology`, and N1 (stored, 00013)
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels) and mean absolute error
  of the median, in points of capacity (1 point is 1% of the series' effective capacity)
production_status: not-for-production
adopted_in: none
code: ml-experiments/00031-embedded-pv-features-primary-demand/
---

# Experiment 31: do embedded-solar capacity features lower the error of Primary demand forecasts?

**Planned.** This page is the plan, committed before any code runs. The design came from an Opus
design brief of 2026-10-15; the departures from it are listed under "Departures from the design".

## Plan: sizing

- **Stored data:** prediction sets and a daily feature table under `data/ml-experiments/`.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Defensible designs:** more than one (feature, offset, or correction; several estimators). The
  arms below test each family once.
- **Callers:** the scripts copy 00019's helpers and read its stored N1 fit, which the author could
  name without searching.

Size: complex, because more than one design defensibly tests the hypothesis and the experiment adds
a data source (the ECR). Plan, code, and write-up each get one Sonnet review.

## Earlier advice followed and departed from

- **Followed:** compare against N1 with the 14-day correction and the 00019 stack, not N1 alone
  (00030); keep `colsample` at 1.0; report absolute scores beside contrasts; guard against one
  substation setting the pooled figure (00007).
- **Departed from:** 00030's weekly mean and median lags are not in B0, because 00030 has not
merged;
  a follow-up can add them to every arm.

## Method

**Capacity enters as a trailing estimate that changes over time, or as a linear offset.** N1 fits
one
model per series, so a per-series constant gives the trees nothing. Every feature is computed at
each
forecast origin from power and irradiance observed up to 48 hours before the origin, so the same
function serves training rows and validation rows.

- **Per-unit PV, `p_unit`:** AC output of a generic fleet per MW of AC capacity
  (`studies.pv_physics.plant_power_mw`, tilt 35 degrees, azimuth 180 degrees, DC 1.15 MW, AC 1.0 MW,
  times a 0.86 system derate; untuned). Fitted constants (the shrinkage weights and prior variance)
  use training origins only, which are those before 2025-07-01; validation origins run from
  2025-07-01 to 2026-06-30. At lag times the irradiance is CAMS GHI at the substation's
  location, which ECMWF publishes with a delay of at most 2 days (an ECMWF notice of February 2026
  says "up to yesterday"; older pages say 2 days), so it exists before the 48-hour cut-off. The
  latency was not measured; if it exceeds 48 hours, the features would need the ENS analysis proxy,
  and the write-up says so. At valid
  times the irradiance is the ENS member's `shortwave`.
- **Floor:** the ECR rows on the `Register Part 1 50kW - <1MW` sheet whose `Primary` name matches
the
  substation exactly after `studies.name_matching.normalise` (no fuzzy matching), plus any `>=1MW`
  solar row whose customer name matches no metered generator. The decision per row is printed by the
  script. The Primaries are "Disaggregated Demand", so metered generation is already subtracted. The
  floor is zero where no row matches. The August 2026 snapshot is used at every date, a small
  look-ahead that can only inflate the ECR arms. BSP and GSP series have no floor (data-only
  estimates) and are descriptive.
- **Features, 28-day windows ending at the origin minus 48 hours, daylight half-hours (sun above 15
  degrees):**
    - F1 `cap_slope_28d`: minus the slope of power on `p_unit`, with fixed effects for local hour by
      day type (weekday, or weekend or bank holiday). Null with fewer than 8 days.
    - F2 `sunny_minus_cloudy_28d`: mean power on sunny days (daily clear-sky index at least 0.7)
      minus cloudy days (at most 0.3), matched by hour and day type, over the matching difference
      in mean `p_unit`, sign flipped.
    - F3 `cap_change_yoy`: F1 minus F1 for the window 364 days earlier.
    - F4 `unreg_cap`: `max(0, c_fit - floor)`, where `c_fit` is F1 shrunk towards its trailing
      365-day mean by inverse-variance weights fitted on training origins only.
    - F5 `cap_total`: `c = floor + u` (data only: `c_fit`).
    - F6 `pv_hat`: `c * p_unit_ENS(valid)`. F7 `lag_origin_gross`: N1's `lag_origin` plus
      `c * p_unit_CAMS(lag time)`. F8 `pv_floor_hat`: `floor * p_unit_ENS(valid)`.
- **Known-answer checks before real data:** (a) inject `-c(t) * p_unit_CAMS(t)` with `c` rising
  linearly from 0 to 20% of capacity into two directional Primaries and check that F1, F2, and F4
  recover `c(t)` within one standard error; (b) at the two GSPs compare `c` with PV_Live's installed
  capacity over the same months (trend only).

## Arms

Rows `fold_a_every5d_v1`, ENS weather, seeds 0 to 2, N1's 00013 configuration. Every arm carries
the 14-day correction and the 1.15 spread factor, as B0 does.

| Arm | Change from B0 | Extra columns |
|---|---|---|
| B0 | The 00019 stack | 0 |
| A1 | Capacity statistics F1, F2, F3 | 3 |
| A2 | Data-only PV products F6, F7 (`c = c_fit`) | 2 |
| A3 | PV products with the ECR floor: F6, F7, F8 | 3 |
| A4 | Linear offset: `base_margin = -c * p_unit_ENS(valid)`, `c = floor + u`; directional Primaries only | 0 |
| A5 | The flat 14-day mean replaced by `a + b * p_unit` fitted on the same residuals; `b = 0` with fewer than 3 distinct sunny origin-run dates (daily clear-sky index at least 0.7) in the window, and `b` not sign-clipped; no refit | 0 |
| A6 | Shuffled control of A3: `c` permuted across origin dates within series and calendar month, with F6 and F7 recomputed from the shuffled `c`; F8 keeps the true floor | 3 |
| A7 | Oracle at the 2 GSPs: B0 plus PV_Live at the valid time | 1 |

A5 tests a sunshine-shaped correction of the median's remaining error, not capacity: the residuals
come from runs whose forecasts already contain some PV signal. A4 reuses B0 on the 8 non-directional
Primaries, which fold when PV exceeds demand.

Plus a no-fit baseline: `manual_heuristic` on gross demand minus `c * p_unit_ENS(valid)`.

## Departures from the design

- **No temperature covariate in F1.** An archive of ENS analysis temperature at lag times would be
  daily for training runs and 5-daily for validation runs, so training and validation would differ.
  F3 cancels most of the seasonal confound instead.
- **No outcome-based half-hour selection on non-directional MVA series.** Selecting half-hours by
  observed power biases the slope, so every daylight half-hour is used in A1 to A3 as well; the
  folded readings are a stated limit, not a result, and the write-up splits directional from
  non-directional series. The 99.17% interval's half-width is the one the refuted rule reads.

## Evaluation

Primary first (16 series), absolute scores beside contrasts. BSP (2) and GSP (2) are descriptive and
never averaged into the Primary figure. Breakdowns, all exploratory at 95%: clear-sky-index bins at
the valid time (night, under 0.3, 0.3 to 0.7, at least 0.7); sunny against cloudy days; April to
September against October to March; lead bands 0 to 48 h, 3 to 6 d, 7 to 15 d; per substation
ordered by registered floor over capacity, split by directional MW and non-directional MVA; the
pooled contrast without the series of largest B0 error and without D8, D12, and D22. The clear-sky
index is used to bin only, never as an input.

## Limits to expect

One validation year gives 12 month blocks, so a 0.05-point effect may be undetectable. The ECR
floor is wrong in both directions at Primaries (metered rows, future connections). The ENS forecast
irradiance is biased against CAMS, so a capacity estimated on CAMS mis-scales when applied to ENS.
