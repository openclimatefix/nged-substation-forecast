---
id: "00074"
slug: an1-same-time-analogues-vs-explicit-solar-forecast-demand
status: planned
size: complex
hypothesis: >-
  At the 5 Primary substations (substations that feed 11 kV circuits from the 33 kV network) with a
  clean solar-capacity estimate, when the XGBoost model is refitted every month on the latest 12
  months of data (P-roll), an explicit forecast of the unmetered solar output, given to the model as
  a starting offset (experiment 64's arm `am`), lowers the daylight April-to-September quantile
  score by more than same-time-of-day analogues do. An analogue is a recent day, at the same
  half-hour of the day and of the same day type (working day, weekend, or bank holiday), whose
  satellite-based sunshine was closest to the sunshine forecast for the target. The model receives
  the observed power on the K nearest such days (K is 1, 3, or 5) as input columns. Once the
  explicit offset is in the model, the analogue columns add less than 0.05 points of capacity.
prediction: >-
  Against the P-roll reference without radiation inputs (`R0`), the explicit forecast (`E`) scores
  -0.10 to -0.25 points, because experiment 64's `am` scored -0.256 against the frozen `d1l` and
  retraining already absorbs part of the capacity growth. The analogue arm with K = 3 (`A3`) scores
  -0.02 to -0.10 against `R0`: it shows the model the net power on days with similar sunshine, which
  contains the panels' output implicitly, but it is blind to the sunshine forecast's skill at the
  target and sees only days at least 3 days old. `E` beats `A3` by 0.03 to 0.20 points. The arm with
  both (`A3E`) scores within -0.03 to +0.02 of `E`. The placebo twin of `E` scores 0.15 to 0.35
  worse than `E` (experiment 64: 0.300). The analogue arm with candidates drawn at random (`A3S`)
  scores 0.01 to 0.06 worse than `A3`. At night, `A3` and `A3S` score within 0.02 of each other.
decision_rule: >-
  Six planned contrasts on the quantile score in points of capacity, seed 0, over the 5 series'
  daylight April-to-September rows of the `fold_a_every5d_v1` validation runs, every one a paired
  difference with whole ISO weeks of forecast start resampled (about 29 blocks) and 50,000
  resamples. The family of six is Bonferroni-corrected, so every interval is 99.17%. C1 is `E`
  minus `R0`. C2 is `A3` minus `R0`. C3 is `A3` minus `E`, the head-to-head contrast. C4 is `A3E`
  minus `E`, which asks whether the analogues add anything once the explicit offset is in. C5 is `E`
  minus its placebo twin `E_twin`, which takes the forecast sunshine from another date (the
  construction of experiment 64). C6 is `A3` minus `A3S`, which asks whether matching on sunshine
  and day type together matters, with the pool of candidate days held equal. C6 does not separate
  the two kinds of matching. The verdict is `refuted`
  when C3's upper end is below zero (the analogues beat the explicit forecast) or C4's upper end is
  below -0.05 (the analogues add a porting-worthy gain on top of it). Otherwise, when C1's upper
  end is below zero and C3's lower end is above zero, the verdict is `small gain` if C4's upper end
  is below zero (the analogues help a little on top of the offset, and the gain does not clear
  -0.05 with certainty), and `supported` if C4's lower end is at or above -0.05 (a gain of 0.05
  points is ruled out). The verdict is `inconclusive` in every other case, including when C1 does
  not exclude zero and when C4's interval runs from below -0.05 to above zero. The threshold of
  -0.05 on C4 is set by the
  cost of porting: an analogue builder needs the last 3 to 16 days of telemetry, an irradiance
  estimate of those days, and a day-type calendar at every forecast, whereas the explicit offset
  needs a table of capacities and weights. In experiment 64 the paired contrasts had half-widths of
  0.044 (`am` against `d0`) to 0.105 points (`am` against its placebo twin) at 98.75%, so the
  threshold of -0.05 may be unresolvable for C4. When C4's half-width exceeds 0.05, the write-up
  says that the experiment could not rule out a gain of that size and does not claim a null. C5 and
  C6 decide no
  verdict. Every other read is exploratory, labelled so, at 95%: `A1` and `A5` against `A3`, the
  night rows (April-to-September rows after dark) for `A3` against `A3S`, the leads under 48 hours,
  the sunniest tenth of rows, each valid month, and each series. Absolute scores are reported for
  every arm with a 95% interval. Experiment 64's frozen-model arms (`d0`, `d1l`, `am`, the
  climatology, and the manual heuristic) are context rows only: they are scored on the same rows
  but the frozen arms carry the 14-day correction, and no contrast uses them.
benchmark: >-
  `R0`, experiment 49's P-roll refit (monthly, latest 12 months, 150 rounds, learning rate 0.15,
  half-life 0.5 years, spread factor 1.15, no 14-day correction) of experiment 64's `d1l` inputs,
  which are experiment 33's best-guess inputs without `shortwave` and `longwave`. Experiment 49's
  P-roll reference of 5.150 points covers 16 Primaries on all rows, so it is not this experiment's
  benchmark; `R0` is scored on this experiment's rows.
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS) in points of capacity (1 point is 1% of the series' effective capacity), after the spread
  factor 1.15, on daylight April-to-September rows with a CAMS unit PV shape and an observed power,
  as in experiments 46 and 64.
code: ml-experiments/00074-an1-same-time-analogues-vs-explicit-solar-forecast-demand/
---

# Experiment 74: do same-time analogues add anything once the solar forecast is explicit? (AN1)

**The plan on this page was committed before any code ran.** The write-up replaces the plan when the
experiment finishes.

We ask two questions about the forecast of the power that a Primary substation draws. Substations
with solar panels behind the meter draw less power on a sunny day than on a cloudy day. First, does
telling the forecast model about that effect explicitly, by forecasting the panels' output, help
more than showing the model what the substation drew on recent days with similar sunshine? Second,
if the explicit forecast is already in the model, do the recent similar days add anything?

## Design

**Both ideas are tested on the five Primary substations of experiment 64, with a model that is
refitted every month.** Experiment 64 (PVDIS-5) found that giving the XGBoost model a forecast of the
solar output as a starting offset beat the best guess. The maintainer expects the explicit forecast
to beat analogues. A model refitted monthly on the latest year (P-roll, from experiment 49) is the
reference, because a frozen model overstates the benefit of any input that follows the recent level
of power, and analogue columns follow that level.

**The eight arms share one recipe and differ in what the solar information is.**

| Arm | What the XGBoost model receives beyond `d1l`'s 15 input columns |
|---|---|
| `R0` | Nothing (the P-roll reference). |
| `E` | A starting offset (base margin): the weight times the capacity times the calibrated sunshine anomaly of the ENS ensemble mean. The weights and capacities are experiment 64's, fixed on data to 2025-06-30. |
| `E_twin` | The same offset with the sunshine anomaly taken from another forecast run (a seeded cyclic shift within series, month, half-hour, and lead band). Training and scoring both use the shifted anomaly, with `E`'s weights. |
| `A1` | 3 columns built from the nearest analogue (K = 1): the power on that day, the gap in sunshine, and the number of analogues found (0 or 1). |
| `A3` | 4 columns built from the 3 nearest analogues: the mean power, the standard deviation of power, the mean gap in sunshine, and the number found. |
| `A5` | The same 4 columns from the 5 nearest analogues. |
| `A3E` | `A3`'s columns and `E`'s offset together. |
| `A3S` | `A3`'s columns, built from 3 candidate days drawn at random from the same window, ignoring sunshine and day type. |

**An analogue is a past day, at the same UTC half-hour as the target, whose sunshine was closest to
the forecast sunshine at the target.** Candidates are the 14 days from 3 to 16 days before the day
of the forecast origin, so every candidate's telemetry is at least 57 hours old. This gap exceeds the
48 hours that experiment 21 found sufficient for lagged power, and it is longer than CAMS's
delivery delay, which has been assumed (under 48 hours) and not measured. A candidate must have the
same day type as the target (working day, weekend, or bank holiday in England, from the
`holidays` package, applied to the Europe/London date) and an observed power. The distance is the
absolute difference between the candidate's CAMS unit PV shape and the target's calibrated ENS unit
PV shape (the climatology plus the calibrated anomaly of experiment 46), both on the CAMS scale.
Ties, which occur at night when both shapes are zero, go to the most recent day. A target with
fewer than K candidates receives fewer analogues, and a target with none receives missing values
that XGBoost handles natively. In training the target shape uses the ensemble mean, as experiment 64's
arm `a` does, and at inference each of the 51 members uses its own calibrated shape.

**Leakage rules.** A candidate's power and CAMS shape are at least 57 hours old at the forecast
origin, given the 9-hour publication delay of the 00:00 UTC run, whose earliest origin on a day is
09:00. The target shape comes from the ENS run that issued the forecast. The refit at time `t` sees
only valid times up to `t` minus 3.5 hours and runs initialised by `t`, as in experiment 49. Climatology,
calibration slopes, weights, and capacities are fixed on data to 2025-06-30, which lies before every
validation row. The refit window slides, so the training rows after 2025-06-30 need the ensemble-mean
sunshine of their ENS runs: these are built from the archive with experiment 46's chain, and the
climatology and slopes stay at their 2025-06-30 values. No calendar column is added; the frozen
column set holds hour and weekday, as every earlier experiment's `d1l` does, and each arm shares it.

**Every arm starts from the same level.** Each arm's training and inference matrices carry a base
margin. The margin of `R0` and of the analogue arms is the median of the refit's training labels.
The margin of `E`, `E_twin`, and `A3E` is that median plus the offset. Without this, the arms with
an offset would start from zero and the others from XGBoost's estimated base score, a confound that
experiment 64 listed. All arms use `colsample` 1.0, so the wider matrices of the analogue arms
gain nothing from column subsampling. `A3S` keeps the gap column, so it still carries a signal about
how far the sunshine of its candidates differs from the target's.

**Rebuilt inputs.** The P-roll window slides past 2025-06-30, so four inputs of experiment 64 are
rebuilt for every training row of the window: the ensemble-mean sunshine, the calibrated anomaly,
the CAMS shape of the valid time, and the placebo twin. The twin is a cyclic shift drawn over the
whole window's rows, a different pool from experiment 64's. Experiment 64's guard against rows after
2025-06-30 is replaced by a check that no refit sees a valid time later than its own cutoff.
The climatology, calibration slopes, and weights stay at their 2025-06-30 values.

**Costs and limits, stated now.** The capacity table is experiment 64's, which is 5% to 19% below
experiment 45's 2025/26 values. The shortfall works against `E` and `A3E`, and the analogue arms use
no capacity. `E`'s weights were fitted for a model without analogues, so in `A3E` the offset and the
analogues may count the same sunshine twice. The weights and the calibration are not refitted
monthly. Only seed 0 decides the verdict;
if time remains, seeds 1 and 2 are added and labelled post hoc.

## Size

**The size is complex, from one trigger: more than one design would defensibly test the question.**
The other four triggers are absent.

1. What gets stored: private prediction sets under `data/ml-experiments/`, nothing in production.
2. Production serving path: not touched.
3. Degradation rule: not touched.
4. More than one defensible design: yes. The analogue's matching space, candidate window, and K
   could each be chosen differently.
5. Code whose callers cannot be named without searching: no, the scripts are new and are copied
   from experiments 46, 49, and 64.

The maintainer's brief asks for Sonnet to plan, review, implement, run, and review the prose, so
the plan review that the full process gives to Opus goes to Sonnet.

## Earlier advice

Followed: fit the solar weight on the training window instead of fixing it at 1 (experiment 46);
check the sign of each contrast in every valid month (46); report coverage of the 0.05 and 0.95
quantiles (46); attach the solar term as a starting offset (64); set the weight to zero beyond 10
days (64); take experiment 49's P-roll as the screening reference. Departed from: refit with a
current capacity table (64), so that `E` stays comparable with experiment 64's `am`.
