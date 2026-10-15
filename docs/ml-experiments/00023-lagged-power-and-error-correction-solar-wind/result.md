---
id: "00023"
slug: lagged-power-and-error-correction-solar-wind
status: planned
size: medium
hypothesis: >-
  For the 6 solar farms and 3 wind farms, giving the stage 2 calibrator the farm's recent observed
  power (lagged power, or summaries of the latest day), or shifting the forecast by its own recent
  error (the 14-day correction of experiments 00017 and 00019), lowers the continuous ranked
  probability score (CRPS) below that of the same calibrator without recent power.
prediction: >-
  Lagged power lowers the CRPS of the calibrator by 0.10 points of capacity for wind (plausible
  range 0.0 to -0.30) and by 0.05 for solar (+0.03 to -0.15), mostly at leads under 2 days, because
  the weather explains most of a generator's output and recent power carries only curtailment and
  capacity changes. Summaries add less than 0.03 to the raw lags. The 14-day correction moves the
  pooled mixture's CRPS by less than 0.05 for both technologies, because the farms' error has no
  slow drift like the one in Primary demand.
decision_rule: >-
  Contrasts are per-technology CRPS differences (arm minus comparison), averaged over three seeds,
  with whole months resampled at 99.58% (a 5% level with a Bonferroni correction across the 12
  planned contrasts, two-sided). Only C1 (lagged calibrator minus lag-free calibrator) and C5
  (14-day correction minus the mixture) decide the verdict, per technology. The threshold is 0.05
  points of capacity, because each change adds a serving step that reads observed power. A change is
  a gain when its interval lies wholly below zero and its estimate is at most -0.05, a small gain
  when the interval lies wholly below zero and the estimate is above -0.05, refuted when the lower
  bound is above -0.05 and the interval does not lie wholly below zero, and inconclusive otherwise.
  The overall verdict is supported when a gain exists for both technologies on C1 or C5, partly
  supported when it exists for one technology, small gain when none is a gain and at least one is a
  small gain, refuted when every decisive contrast is refuted, and inconclusive otherwise.
benchmark: >-
  `mixture`: experiment 00002's pooled ERA5-trained mixture (the `base` arm of 00020, checked
  against 11.11 for wind and 5.05 for solar to within 0.05 points), for the correction; `cal0`, a
  lag-free calibrator trained here, for lagged power. The month-by-half-hour climatology of
  experiment 00002 is a reference row.
row_set_id: fold_a_tuesdays_v1
metric: >-
  CRPS estimated from 9 quantile levels, points of capacity (power over effective capacity, times
  100); coverage of the 5% to 95% band by lead band
---

# Does recent observed power help solar and wind forecasts?

**Status: planned.** This is proposal P18 of the [research plan](../research-plan.md), with the
14-day correction of [experiment 00017](../00017-training-window-and-retraining-rule-drift-demand/result.md)
and [experiment 00019](../00019-primary-demand-stack-vs-manual-heuristic/result.md) added.

## Size

- **What gets stored:** private prediction sets only. No schema in `packages/contracts/` changes.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes (the form of the calibrator and of the correction), but
  the plan fixes one of each by copying experiments 00005 and 00019.
- **Code whose callers cannot be named without searching:** no.

The size is medium: one Sonnet review of the code and one of the write-up.

## Which earlier advice this plan follows

- **Followed:** compare lagged arms against a calibrated arm without lags (00005); keep the raw
  lags and add summaries beside them (00008); test the 14-day correction on forecasts made from
  forecast-time weather and with a realistic data delay (00017, 00019); pool the members before
  scoring and report the pooled mixture beside every stage 2 arm (00002).
- **Departs from:** none.

## Design

**Rows.** The 6 solar farms and 3 wind farms of `fold_a_tuesdays_v1`: 53 Tuesday runs with valid
times from 2025-07-01 to 2026-06-30, three seeds.

**Weather source at forecast time.** ENS members only, through the ERA5-trained stage 1a quantile
models of experiment 00002. No ERA5 value is read at a lag time, so no arm relies on a late source.
Recent observed power is usable once it is 3 hours old plus the half-hour it spans, as in
experiment 00019.

**Calibrators.** One multi-quantile XGBoost model (9 levels, depth 4, 300 rounds, the settings of
experiment 00005) per technology, seed, and arm, trained on the ENS control member's rows with
out-of-fold stage 1a quantiles from 2024-07 to 2025-06 (the training table of experiment 00002), with
the observed power over effective capacity as the target. It takes the member's 9 stage 1a quantiles
and the lead in hours, plus the recent-power columns of its arm. No calendar column. It is applied to
each of the 51 members, and the members' calibrated quantiles are pooled into a mixture as in
experiment 00020. The calibrator is trained on the control member and applied to all members, as in
experiment 00002.

**Recent-power columns.** Lagged power (`lags`) is the newest usable value, the value at the
valid time's half-hour on the latest earlier usable day, and the same on the latest earlier usable
week (experiment 00019's three lags). Summaries (`sums`) are the minimum, mean, and maximum of the
48 half-hours ending at the newest usable value, and of the latest earlier usable day on the
target's weekday (experiment 00008).

**14-day correction.** For a run at time `t`, the correction is the mean of `observed - median` of
the mixture over earlier validation runs' cases with a valid time in `[t - 14 d, t - 2 d]`, using
for each valid time the forecast of the latest earlier run, over non-trivial cases (95% quantile
above 1% of capacity). The quantiles of non-trivial cases are shifted by it, clipped to 0 to 1, and
sorted. Fewer than 500 such cases means no shift (the first run).

**Arms.**

| Arm | What it is |
|---|---|
| `mixture` | the pooled ERA5-trained mixture (00020's `base`) |
| `cal0` | calibrator with no recent-power column |
| `cal_lags` | calibrator + lagged power |
| `cal_sums` | calibrator + summaries |
| `cal_both` | calibrator + lagged power + summaries |
| `mix_c14` | `mixture` + 14-day correction |
| `cal_both_c14` | `cal_both` + 14-day correction |

## Planned contrasts

Each is the arm named first minus the arm named second, per technology (12 tests across the two
technologies).

| Id | Contrast |
|---|---|
| C1 | `cal_lags` - `cal0` (decides) |
| C2 | `cal_sums` - `cal0` |
| C3 | `cal_both` - `cal_lags` |
| C4 | `cal_both` - `mixture` |
| C5 | `mix_c14` - `mixture` (decides) |
| C6 | `cal_both_c14` - `cal_both` |

The write-up also reports each arm's absolute CRPS, the contrasts by lead band, and coverage.

## Limits known in advance

- **With 6 solar farms and 3 wind farms, a lag effect under about 0.3 points of capacity may not be
  resolved**, so the result is read per technology.
- **The calibrator is trained on 12 months of the ENS control member only.**
- **The 14-day correction has history from validation runs only**, one per week, so it is weaker
  than a daily history would give.
