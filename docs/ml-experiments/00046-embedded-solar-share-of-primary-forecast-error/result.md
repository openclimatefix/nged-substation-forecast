---
id: "00046"
slug: embedded-solar-share-of-primary-forecast-error
status: planned
size: complex
hypothesis: >-
  At the five Primary substations metered in MW that have a clean 2023/24 capacity estimate, an
  explicit solar term lowers the error of the demand forecast that experiment 00033's best-guess
  recipe produces without its shortwave and longwave radiation inputs (arm `d1l`) by at least 0.35
  points of capacity on daylight April-to-September rows. The solar term shifts every forecast by
  minus the capacity of experiment 00040 times the calibrated ENS forecast of the solar anomaly,
  with the coefficient fixed at 1. The contrast is against `d1l` and not against the best guess.
prediction: >-
  The fixed-coefficient shift built from CAMS at the valid time (the oracle) lowers the quantile
  score by 1.2 points (plausible range 0.7 to 2.0); the same shift built from calibrated ENS lowers
  it by 0.3 (0.0 to 0.7), so the expected verdict is `small gain`, and `supported` is unlikely. The
  precedent is arm A4 of experiment 00031, a linear offset built from ENS solar output, whose
  contrast with its baseline was +0.012 (-0.051 to +0.081). The calibrated ENS shift, laid over
  `d0` (the best guess), is expected to leave the score within 0.3 points either way (C6 of -0.05,
  -0.4 to +0.3). The scratch preview behind the oracle figure was not blind: the stored 00033
  forecast, shifted by the oracle with coefficient 1, scored 0.87 points lower on seed 0.
decision_rule: >-
  Four planned contrasts (C3 to C6 below; the ids of the original six are kept, and C1 and C2 move
  to a later experiment). Two decide the verdict, C3 and C4, and are read at 97.5% (Bonferroni across
  two). Every other read is exploratory, labelled so, and read at 95%. Intervals resample whole
  months and seeds, 50,000 resamples, quantile score on the scored rows. The threshold is 0.35
  points of capacity on those rows. It is a judgement about the cost of porting the solar term into
  production, which needs a capacity table to be read and an ENS-to-CAMS calibration to be kept
  current, and the maintainer may change it before the contrasts are read. A validity gate comes
  first, and the verdict rests on C3. Gate: C4 (the oracle) lies wholly below -0.35, otherwise the
  verdict is `inconclusive` and says that the fixed-coefficient shift is not a usable measure. If
  the gate passes, `supported` when C3 lies wholly below -0.35, `small gain` when C3 lies wholly
  below zero but not wholly below -0.35, `refuted` when C3 lies wholly above -0.35 and not wholly
  below zero, and `inconclusive` otherwise. Series D9 (bound 0.90) is in the verdict set, and the
  write-up reports the verdict with and without D9; series D5 is exploratory only. PVDIS-5 is a
  gain test only if C6 (`d1l_ens_shift` minus `d0`) also lies wholly below zero. Otherwise PVDIS-5
  is a non-inferiority and explanation test.
benchmark: >-
  Arm D1L, experiment 00033's best-guess recipe (N1 with the 14-day correction and the 1.15 spread
  factor) refitted without `shortwave` and `longwave`, with three matched seeds. Arm D0, the
  recipe with those columns, is refitted on the same seeds and is the benchmark of C6 only. The
  stored `n1_refit_cs` of experiment
  00033 is a self-test of `d0`, not a benchmark.
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS) in points of capacity, after the 14-day correction and the spread factor, on daylight
  April-to-September rows with a CAMS unit PV shape and an observed power
code: ml-experiments/00046-embedded-solar-share-of-primary-forecast-error/
---

# Experiment 46: how much of the best guess's daylight error is embedded solar?

We asked how much of the error in the demand forecast for a Primary substation comes from solar
panels behind the substation's meter. The meter shows demand minus the panels' output, so a sunnier
day than the weather forecast expected lowers the metered power. We retrain the demand forecast
without its two radiation inputs. We then add a solar term: a shift that subtracts the fitted solar
capacity times the sunshine that the weather forecast did not expect. We score every version on the
same daylight half-hours of July to September 2025 and April to June 2026, and we separate the gain
from knowing the true sunshine from the gain from the weather forecast of it. The planned verdict
rests on one contrast: whether the solar term built from the weather forecast lowers the error of the
demand forecast without radiation inputs (`d1l`) by at least 0.35 points of capacity. That contrast
does not test whether the solar term beats the best guess, which still has its radiation inputs. The
contrast of the solar term with the best guess is C6, and it decides what PVDIS-5 tests.

**The experiment tests whether PVDIS-5 should be a gain test or only an explanation test.** If the
solar term built from the ENS weather forecast clears the threshold (C3) and also lies wholly below
the best guess (C6), PVDIS-5 compares its two forecasts against the best guess for gain. If C3 clears
the threshold but C6 does not lie wholly below zero, PVDIS-5 runs only as a non-inferiority test of
the explanation. If the solar term built from CAMS (C4) gains a lot but the ENS version gains little,
the weather forecast is the bottleneck, and PVDIS-5 also runs only as a non-inferiority test. If the
CAMS version gains little, the fixed-coefficient measure is not usable, and PVDIS-5 waits for the
coefficient diagnostic below. Supported is unlikely. Experiment 00031's arm A4, a linear offset
built from ENS solar output, scored +0.012 (-0.051 to +0.081) against its baseline, and the
prediction above puts the ENS version just short of the threshold. Experiment 00022 found that ERA5
weather in place of ENS weather lowered the score by 0.10 points over all rows, and experiment 00009
found that the stage 1a model gained 0.88 points of daylight error from observed irradiance against
0.04 from ENS. This experiment divides that gap between the sunshine itself and the weather forecast
of it.

## Size

**The experiment is complex because more than one design would test the hypothesis, and the other
four triggers are absent.**

| Trigger | Answer |
|---|---|
| What gets stored | Four private prediction sets, and two more for the headline-capacity sensitivity read, under `data/ml-experiments/`, written with `experiment_store.write_predictions`. No production table changes. |
| The production serving path | Not touched. |
| A degradation rule | Not touched. |
| More than one defensible design | Yes. The shift could sit on `d0` or on `d1l`, the ENS anomaly could be calibrated or raw, and the coefficient could be fixed or fitted. The sections below fix one choice for each and give the reason. |
| Code whose callers could not be named without searching | No. The scripts copy named helpers of experiment 00033 and read three earlier experiments' outputs and `studies.pv_physics`. |

The full process applies: a Sonnet plan, an Opus plan review, a Sonnet implementation, a Sonnet code
review for leakage, a Sonnet run, and a Sonnet write-up with a prose review.

## Arms

All arms use the 6 Primaries metered in MW that have a 2023/24 capacity (D5, D9, D10, D11, D14, and
D16), three seeds, and the 73 runs of `fold_a_every5d_v1`. The two refitted arms share seeds, so
XGBoost's row subsampling is identical across them. Experiment 00033's recipe uses a column sampling
rate of 1.0, so the arm with fewer columns does not also change which columns each tree sees.

| Arm | What it is | Fitted? |
|---|---|---|
| `d0` | Experiment 00033's N1 recipe: 13 weather and calendar columns, `lead_hours`, and 3 power lags, trained on the ENS control member, with the 14-day correction and the 1.15 spread factor | yes |
| `d1l` | `d0` without `shortwave` and `longwave` | yes |
| `d1l_ens_shift` | `d1l` with each ENS member's quantiles shifted by minus the capacity times the calibrated ENS solar anomaly, divided by the series' effective capacity, before the members are pooled | no, post hoc |
| `d1l_cams_shift` | `d1l` with every quantile shifted by minus the capacity times the CAMS solar anomaly at the valid time (the oracle) | no, post hoc |

**Two arms from the first proposal move to a later experiment.** Arm `d1` (`d0` without `shortwave`
alone) and arm `d0_obs_sw` (`d0` with observed CAMS shortwave at serving) would answer what the ENS
shortwave column adds (C1) and how much observed shortwave would add through the model's own route
(C2). Neither answer changes the verdict or the design of PVDIS-5, so both contrasts wait. The
contrast ids C3 to C6 are kept so that earlier reviews still point at the right contrast.

**`d1l` removes both radiation columns, so `d0` minus `d1l` reads what the radiation inputs add and
never the solar share of the error.** The frozen column list (experiment 00006's, as experiment 00005
froze it) holds `shortwave`, `longwave`, `temperature_2m`, `dew_point_temperature_2m`,
`pressure_surface`, `precipitation`, and `wind_speed_10m`, then `solar_zenith`, `solar_azimuth`,
`hour_local`, `day_of_week_local`, `covid_indicator`, and `days_to_christmas`. Experiment 00033 adds
`lead_hours` and the three power lags. Four routes to solar stay in `d1l`: the sun's position
(`solar_zenith` and `solar_azimuth`) gives the clear-sky shape, the three lags carry yesterday's and
last week's solar, the dew point, pressure, and precipitation are cloud proxies, and the 14-day
correction absorbs a persistent level error.

**The explicit shift sits on `d1l`, not on `d0`, because `d0` already contains part of the
anomaly.** A preview on the stored `n1_refit_cs` forecast (seed 0, five series, daylight April to
September rows, not blind) regressed the median's residual on the oracle shift, capacity times CAMS
anomaly over effective capacity. The slope was -0.52 at leads under 1 day, -0.57 at 1 to 2 days,
-0.67 at 2 to 6 days, and -0.81 at 6 to 10 days, against -1 for a forecast that held no solar
information beyond climatology. A shift on `d0` would partly count the anomaly twice, by a
lead-dependent amount that no fixed coefficient can state. The preview slopes are for the
`d0`-equivalent forecast. No slope has been measured on `d1l`, so the coefficient of 1 is untested
there. `d1l` holds less solar information than `d0`, which makes a slope nearer -1 plausible but
does not show it. The write-up therefore reports the slope on `d1l` before it reports any verdict
(see the coefficient diagnostic below).

**The shift is well defined, in capacity units.** Experiment 00040's C is in MW. It is the
no-confound capacity, the least-squares combination of the six arm estimates against the six recovery
ratios R of the injected farms (`C = (R . c) / (R . R)` in `e00040_detector.py`). C is the 99th
percentile of the fitted solar output of a generic fleet whose unit shape `u` has a 99th percentile
of 1 over the half-hours with a solar zenith under 75 degrees, which experiment 00039's validation
prints as 1.0 for both the farm and the substation frames, and for each of the 20 sites. The shift
of one forecast value is `-C x (u(t) - u_clim(month, half-hour)) / effective_capacity_mw`, in the same
units as the forecast, which is power over effective capacity. The preview gave standard deviations
of the shift of 6.2 to 9.1 points of capacity at the five series, which a self-test checks.

**C is not the headline estimate multiplied by a recovery ratio, and the two differ by up to 22%.**
Experiment 00040's `real_scores.parquet` also holds `headline`, which is the estimate of its single
`M1D` arm divided by that arm's recovery ratio. For 2023/24 the ratio C over `headline` is 1.22 at
D10 (2.42 against 1.98 MW), 1.10 at D11, 1.03 at D14, 1.06 at D16, 1.17 at D9, and 0.77 at D5. A
reviewer's reading of D10 as a conflict between C and the headline is therefore correct, and the
cause is the choice of estimator. The plan keeps C because experiment 00040's advice is to report C,
and adds a sensitivity read, in the exploratory section, that repeats the two shift arms with
`headline` in place of C.

## Rows, scores, and who is in

**The scored rows are the daylight half-hours of April to September in the validation year.** Months
are local calendar months (Europe/London) throughout this plan, in the scored rows, the climatology,
and the folds. Daylight means that the month-by-half-hour mean of `u`, over the training window,
exceeds 0.02. A row is scored only when the series has an observed power and a CAMS `u` (CAMS
reliability at least 0.9), which drops about 10% of April to September half-hours. Every arm is
scored on the same rows. In the preview the rows made up 0.277 of all half-hours of the five series.
The validation months in the scored rows are July to September 2025 and April to June 2026, six
months.

**Five series carry the verdicts: D9, D10, D11, D14, and D16.** These are the Primaries metered in MW
with an experiment 00040 capacity for 2023/24, the latest season-year (September to August) that ends
before the July 2025 validation start, and that are not flagged. A later capacity would use
validation-period power in its fit.

| Series | Primary or status | C for 2023/24 (MW) | Bound B | Role |
|---|---|---|---|---|
| D10, D11, D14 | Spilsby, Horncastle, Mount Bridge Boston (in the order of experiment 00040's table, not matched here) | 2.42, 3.38, 3.71 | 0.19 to 0.21 | clean, in the verdicts |
| D16 | Donington | 2.80 | 0.18 | clean, in the verdicts |
| D9 | Trusthorpe | 2.96 | 0.90 | low power, in the verdicts; the write-up reports the verdict with and without it |
| D5 | Kirton | 2.54, flagged; library range 1.99 to 3.97 | | exploratory only, at both ends of the range |
| D15 | Wrangle T1 | none | | left out: no 2023/24 capacity, no detector power |
| D18 | Wrangle T2 | none | | left out: no 2023/24 capacity (its record starts in 2024/25) |

## Planned contrasts

Each contrast is the first arm's quantile score minus the second's, so a negative value means the
first arm is better. C3 and C4 decide the verdict and are read at 97.5%. C5 and C6 are exploratory,
read at 95%, with no correction for multiple comparisons. C6 does not decide the verdict, but it
decides what PVDIS-5 tests.

| Id | Contrast | Question | Predicted |
|---|---|---|---|
| C3 | `d1l_ens_shift` minus `d1l` | Does the solar term built from the ENS weather forecast lower the error? (decides the verdict) | -0.3 (-0.7 to 0.0) |
| C4 | `d1l_cams_shift` minus `d1l` | How much does knowing the true sunshine lower the error? (validity gate, decides the verdict) | -1.2 (-2.0 to -0.7) |
| C5 | `d1l_cams_shift` minus `d1l_ens_shift` | How much of the CAMS gain is lost to the ENS weather forecast? (exploratory) | -0.9 (-1.5 to -0.4) |
| C6 | `d1l_ens_shift` minus `d0` | Does the solar term beat the model's own use of its radiation inputs? (exploratory; sets the PVDIS-5 test) | -0.05 (-0.4 to +0.3) |

**The calibrated ENS anomaly is built from the training window only.** For each series and lead band
(the five bands of experiment 00033), a slope through the origin of the CAMS anomaly on the ENS
anomaly is fitted on rows up to 2025-06-30. The ENS anomaly is the ENS `u`, computed from
`shortwave` by the same chain as CAMS (`studies.pv_physics`), minus that source's own
month-by-half-hour mean of `u`. Subtracting each source's own mean removes the mean bias between the
products, and the slope removes the rest by lead.

**The ENS chain gives a `u` for every half-hour, including those between ENS steps.** ENS `shortwave`
is a step value, with 3-hour steps early in a run and 6-hour steps later. The step's clear-sky index,
the ratio of the step value to the clear-sky irradiance at the step's midpoint, is held across every
half-hour of the step. The half-hour's `u` is then the clear-sky `u` of that half-hour times a fitted
function of the held clear-sky index (the fallback route in the implementation outline), unless the
direct route passes check 5. The month-by-half-hour climatology of the ENS `u` comes from the control
member's training-window values at the same half-hours, so a half-hour between steps has a
climatology and therefore an anomaly. The implementation confirms the step convention (mean over the
preceding or following interval) against the served data before check 5 is read.

**Two statistical weaknesses of the calibration are handled in the design, not left to the
write-up.** First, a slope fitted on the control member alone suffers regression dilution: the
control member's own sampling noise lowers the fitted slope, and a lowered slope applied to each
member shrinks the spread of the pooled shift. The slope is therefore fitted on the anomalies of all
51 members pooled, which uses 51 times the data and averages out the control member's noise, and an
exploratory read repeats C3 with the shift built from the ensemble-mean anomaly alone. Second, the
month-by-half-hour climatology has few days per cell on a short training window, so each
cell mixes few independent weather days. The self-test prints the smallest number of days in any
cell, and if any cell holds fewer than 30 days, the climatology pools the neighbouring local month's
half-hours with the same weights for every source.

**The 14-day correction is computed on the shifted forecasts.** The correction takes the mean error
of the previous 14 days' median forecasts, so a shifted arm corrects its own past errors, as
PVDIS-5's Arm B would. The spread factor of 1.15 follows.

## Exploratory reads

These are post hoc, read at 95% with no correction for multiple comparisons, labelled exploratory in
the write-up, and decide nothing. Each is a table or a figure.

- **The coefficient diagnostic, reported before the verdict.** The residual of the median of `d1l`,
  regressed on the oracle shift, fitted on odd months and scored on even months (cross-fitted by
  month, then swapped), by lead band, clear-sky-index bin, and series. A slope far from -1 says the
  capacity, or the assumption that demand falls one for one with the anomaly, is wrong. The write-up
  prints these slopes before it states the verdict, because they say whether the coefficient of 1
  was reasonable on `d1l`, where it is untested. The slope also absorbs any demand that rises or
  falls with sunshine, which the capacity fit cannot tell from solar, so the oracle gain is an upper
  bound on the solar share and not an estimate of it.
- **The headline-capacity sensitivity.** C3 and C4 repeated with `headline` of experiment 00040 in
  place of C, and the arms stored as `d1l_ens_shift_headline` and `d1l_cams_shift_headline`.
- **The ensemble-mean shift.** C3 repeated with the shift built from the ensemble-mean anomaly alone.
- **Splits of C3 to C6** by lead band, by clear-sky-index bin, by series, and by month, and the same
  contrasts with D9 left out.
- **Kirton (D5)** at the two ends of its library range.
- **`d0` minus `d1l`**, and all four contrasts on the daylight rows of all 12 validation months.
- **A week-block bootstrap** in place of month blocks, for the contrasts that month blocks cannot
  resolve (see the power note below).

## Checks that must pass before any contrast is read

1. **The refit `d0` matches `n1_refit_cs` of experiment 00033.** The mean absolute difference of the
   median over the scored rows is under 0.05 points. GPU fits at a fixed seed may differ in the last
   bits, and a larger difference means the code path differs.
2. **The column lists are exactly 13 + 1 + 3 for `d0` and two fewer for `d1l`.**
3. **A shift of zero returns the unshifted quantiles bit for bit.**
4. **The standard deviation of the oracle shift on the scored rows is 5 to 10 points of capacity at
   every one of the five series.** This tests the units.
5. **The ENS-derived `u` agrees with the CAMS `u` on the training window, as anomalies.** The check
   has two parts. First, the ENS chain is applied to CAMS irradiance (CAMS hourly values presented
   at the ENS valid times), and the resulting anomaly must correlate at 0.95 or more with the
   CAMS-route anomaly: this tests the chain. Second, the real ENS control member's anomaly is
   correlated with the CAMS anomaly at the same valid times, by lead band, and the correlations are
   printed: this tests the product, and no threshold is set because the values are the forecast
   skill itself. Correlating `u` and not the anomalies is not accepted, because both share the same
   daily cycle and would correlate near 1. If the first part fails, the experiment uses the
   clear-sky-index route in the implementation outline (`u_clear` times a fitted function of the
   ratio of irradiance to clear-sky irradiance).
6. **Every climatology, calibration slope, and the daylight mask use valid times up to 2025-06-30
   only,** asserted in code.
7. **All arms hold the same scored keys,** which `experiment_store.losses_for_bootstrap` enforces.

## Leakage and bias risks

- **Capacity for 2023/24 is a year and a half old at the validation midpoint.** The fleet grew by
  about 5% to 15% a year at the clean Primaries in experiment 00040's headline estimator, so the
  shift is probably 10% to 20% too small. The result is therefore conservative for the solar term,
  and the write-up says so.
- **Capacity is a point estimate with a bound.** Experiment 00040's bound B of 0.18 to 0.21 (0.90 at
  D9) is the undetected-load error of the capacity, and the shift inherits it. The write-up quotes B
  beside C3 and C4.
- **The scored rows overlap the CAMS record that fitted C, but not in time.** C used power from
  September 2023 to August 2024. The validation rows are July 2025 to June 2026.
- **Month blocks leave six blocks.** With six months, the 97.5% interval nearly spans the range of
  the month means. A contrast therefore counts only if every month has the same sign, which the
  by-month figure shows. The week-block read is the sensitivity check.

## Compute

**The refits are cheap enough for 4 cores, because experiment 00033's logs put one fit and its
prediction at about 19 seconds on the GPU.** Experiment 00033's `n1_refit` arm fitted 60 models (20
series, three seeds) at 16 to 21 seconds each, 18.8 on average, and its monthly refits took 11 seconds
each for the fit and a 73-run prediction. Six series and three seeds give 18 fits per arm, about 6
minutes, and the two arms about 12 minutes on one GPU stream. The log does not record CPU-only
times, so a CPU fallback is unmeasured; the implementation outline asks for one timed fit on 4 threads
before any fallback is chosen, and the plan does not rely on a fallback until that time is in hand.
The 73 validation runs of the shift variants add prediction passes but no fits, because the member
quantiles of one prediction pass serve both shifts. GPU memory stays inside the 8 GB share if one
worker uses under 2.5 GB, which the first fit's `nvidia-smi` reading confirms.

## Stop rule

The experiment stops without a verdict, and the result says why, if any of checks 1 to 4 or 6 fails
and the cause is not fixed within one revision of the code. It also stops if a fit fails twice.
Nothing is tuned after the contrasts are read: the capacity, the coefficient of 1, the threshold of
0.35, the daylight mask, and the calibration bands are fixed here. If the first part of check 5
fails, the clear-sky-index route replaces the physical route before any contrast is read, and the
write-up says so.

## Advice items followed and departed from

**Followed:**

- Report C and keep the daily-fit capacity as a cross-check only (experiments 00040 and its Opus
  review); this experiment uses C and never the daily fit.
- Leave Wrangle T1 out of any claim that a Primary is free of a cloud load, report Kirton only as a
  range, and leave out the 8 Primaries metered in MVA (experiments 00040 and 00039).
- Calibrate ENS irradiance to CAMS before applying a fitted PV capacity to ENS forecasts (experiment
  00031), by lead band and on the training window.
- Score any irradiance-input change by lead band as well as in total (experiment 00009).
- Take a threshold's resolution from a contrast between arms trained alike (experiment 00009): the
  two refitted arms share seeds and rows.
- Do not add the capacity as a column to N1 (experiment 00031): the capacity enters only as a fixed
  shift.

**Departed from:**

- Experiment 00031 compared every capacity arm with a shuffled-capacity twin, because trained
  columns can act as a proxy for calendar position. The shift here has no fitted coefficient and no
  trained column, so a shuffled twin could only add noise, and it is left out. PVDIS-5 keeps its
  shuffled twins, where a column is trained.
- The research plan's proposal shifts every arm. This experiment shifts only `d1l`, for the double
  counting shown above.
- The research plan's proposal tests shortwave first (experiment 00022). This experiment removes both
  radiation columns in one arm and defers the single-column arm (`d1`) and the observed-shortwave arm
  (`d0_obs_sw`).
