---
id: "00047"
slug: sw2-4-better-residual-switching-detector
status: planned
size: medium
hypothesis: >-
  Adding one column, the signed days to the nearest bank holiday in England, to experiment 17's
  lagless ERA5-trained XGBoost demand model H0.5 (refitted monthly, with H0.5's absolute-error loss
  unchanged) gives a residual in which the single-series detector D0 at the 24-hour window finds
  more injected load transfers at the same rate of unexplained alarms.
prediction: >-
  C1 (D0, 24-hour window, transfers of 25% to 40% of the receiving load lasting over 1 day, starting
  in even months): the share found rises by 0.03 to 0.10 from about 0.29, and event F1 rises at both
  precision bounds. The gain comes through the threshold, not the noise scale: the median change in
  the odd-month noise scale across the 16 series lies within 3%, and the matched threshold falls by
  0.1 to 0.4. Of H0.5's 160 unexplained 24-hour D0 alarms, 26 lie within 1 day of a bank holiday
  outside Christmas and New Year; with the new column that count falls to 13 or fewer. The placebo
  column (holiday dates moved 28 days later) changes C1 by less than 0.02. The mean absolute error
  of the median changes by less than 0.05 points of capacity over the whole period and falls by at
  least 0.5 points on bank holidays.
decision_rule: >-
  The headline is C1, the new arm minus H0.5 in the share of injected transfers found (paired by
  transfer, averaged over seeds 0 to 2), at matched unexplained-alarm rates, read at 98.75% (Bonferroni
  across the four contrasts C1 to C4) from 20,000 resamples of whole calendar months. The threshold
  is zero: the change is one column computed from the timestamp, and any live detector has to refit
  a residual model monthly anyway, so porting it adds no data source and no job. `supported` when the
  C1 interval lies wholly above zero and the event-F1 difference is positive at both precision
  bounds. `refuted` when the C1 interval lies wholly below +0.03. `inconclusive` otherwise. With a
  threshold of zero, `small gain` cannot occur. C2 to C4 are reported with the same intervals and
  change no verdict. Real labelled events are counted without a verdict.
benchmark: >-
  H0.5 (experiment 17, half-life 0.5 years, monthly refit, absolute error) through experiment 43's
  harness unchanged, seed 0 as stored plus seeds 1 and 2 refitted; a placebo arm with the holiday
  dates moved 28 days later; and a detector that raises no alarm.
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled transfers, as
  experiment 43 does, not forecasts)
metric: >-
  Event F1 as in experiment 43: the share of injected transfers found (start within 2 hours, end
  within the larger of 6 hours and 10% of the duration) combined with the alarms' precision bounds;
  the share found decides, with F1, the precision bounds, and the end error beside it.
pull_request: none yet
code: ml-experiments/00047-sw2-4-better-residual-switching-detector/
---

# Experiment 47: does a weather model that knows bank holidays help the switching detector find transfers?

**Plan, written before any fit.** A switching detector compares the power measured at each of 16
Primary substations (substations that step power down to the local distribution network) with what
a weather-driven XGBoost model expected, and raises an alarm when the average difference over 4 or
24 hours steps up or down. Demand drops for a day on a bank holiday, the model does not expect the
drop, and the drop looks like load moved away to another substation. Before writing this plan we
counted that 26 of the 160 alarms of the best detector setting that no logged switching explains
fall within 1 day of a bank holiday outside Christmas and New Year. We will refit the model with one
extra column, the days to the nearest bank holiday, set each detector to raise the same number of
unexplained alarms as before, and count how many of the same 600 known transfers, added to the real
readings, each detector then finds.

## Which model, and why one column

**The best lagless model available is H0.5 itself, so the change under test is a column, not a
model.** "Lagless" means the model reads no recent power, so a transfer cannot leak into its inputs.
The candidates the SW2-4 proposal named fail as follows:

- **Experiment 17's arms:** H0.5 has the lowest mean absolute error of the lagless arms (6.70
  points of capacity) and is already the baseline.
- **Experiment 33's N1 without its lags:** its input set is the 13 columns of experiment 6, the
  same columns H0.5 uses, so it is H0.5 with a 2-year half-life and no monthly refit, which
  experiment 17 found worse.
- **Experiment 42's lagless fits:** they cover July 2025 to June 2026 on runs every 5 days, and
  the harness needs every half-hour from January 2022 to June 2026.
- **Experiment 33's stack:** it reads lagged power and the 14-day error correction, which would
  follow a long transfer and shrink its step, so it is not lagless.

**H0.5 already fits the median with absolute error (`reg:absoluteerror`), so the robust-loss part of
the proposal is already in the baseline.** Pseudo-Huber and Huber losses are quadratic near zero,
which makes them less resistant to a shifted target than absolute error, so the refit keeps
absolute error. The refit excludes no logged or labelled switching from its training target,
because excluding switching is SW2-3 and SW2-5, and SW2-4 tests one change.

**The bank-holiday column has a measured mechanism.** Days within 1 day of a non-Christmas bank
holiday are about 4% of all days and hold 16% of the 24-hour D0 detector's unexplained alarms
(counted from experiment 43's stored alarm tables). The column is `days_to_nearest_holiday`:
signed days to the nearest bank holiday in England from the `holidays` package (`GB`, `ENG`), on the
local date, clipped to 7 either side, computed as experiment 33's `holiday_gaps` computes it. The
forecasting pipeline computes the same column from the timestamp alone.

**Weather is ERA5 at the valid time, which arrives about 5 days late, so the detector stays
retrospective.** The new model's inputs exist at forecast time only as ENS forecasts, and this
experiment does not test a detector on forecast weather.

## Design

- **Arms.** H0.5 (seeds 0, 1, and 2; seed 0 is experiment 17's stored fit), the new arm H0.5 plus
  the holiday column (seeds 0, 1, and 2), and the placebo: H0.5 plus the same column built from
  holiday dates moved 28 days later, which keeps each date's weekday (seed 0). Everything else is
  experiment 17's `e00017_fit.py` for arm H0.5: the 13 columns, 500 rounds, `eta` 0.05, depth 6,
  subsample 0.8, half-life 0.5 years, a refit for each month from January 2022 to June 2026 on
  rows ending 7 days before the month starts.
- **Harness, unchanged.** Experiment 43's 600 injected transfers (`events.parquet`), its 20
  batches and per-batch seeds, `inject_power`, the detectors D0 and D1 at 4-hour and 24-hour
  windows, the interval builder (the plain variant decides; the fixed variant is reported), the
  matching rule, and the precision bounds. Real events: experiment 43's 14 primary and 35
  secondary labelled transfers.
- **Injection.** The training data of every arm is the real power, without injections. A lagless
  model's prediction does not read observed power, so an injected transfer shifts the residual by
  exactly the injected MW, and `fast_residual` is valid for a model trained on uninjected power,
  for both arms alike. This departs from experiment 43's advice to call `inject_power` whenever
  the model is refitted, and a check bounds the error. The check takes the batch with the most
  headline-band transfers and refits, on injected power, the new arm's seed-0 models for every
  month that contains one of the batch's transfers. The check passes when the found and start-found
  outcomes agree with `fast_residual` for at least 95% of the batch's transfer-and-setting pairs.
  If the check fails, the page reports the disagreement as a limit shared by both arms.
- **Tuning on odd months, scoring on even months.** For each arm and seed, the noise scale of each
  series is the odd-month robust scale of its uninjected residual, as the harness computes it. The
  threshold of each of the four settings is the lowest value, on a grid of 0.05, whose count of
  unexplained alarms in odd months (alarms more than 24 hours from a logged closure on their
  series) does not exceed H0.5 seed 0's count at experiment 43's threshold: 120 (D0, 4 h), 118
  (D1, 4 h), 80 (D0, 24 h), and 36 (D1, 24 h). Frozen: windows, neighbour z of 2.0, balance range,
  interval builder, matching tolerances, transfers. Odd-month transfers tune nothing and are
  reported as a check. Recall and precision are scored on transfers starting in even months and on
  even-month alarms: 89 of the 160 headline-band transfers, 48 of them with a metered receiver and 41
  without one.
- **Precision.** As in experiment 43, from the even-month real alarms: the lower bound counts the
  alarms within 24 hours of a logged closure, and the upper bound adds the unmatched alarms times
  the visible share from the SW2-0b sample for that setting. The sample was drawn from H0.5's alarms,
  so the upper bound is an assumption for the new arm, and the page says so.

## Contrasts

| Contrast | Detector, window | Population | Role |
|---|---|---|---|
| C1 | D0, 24 h | headline band, even months | decides the verdict |
| C2 | D0, 4 h | headline band, even months | reported |
| C3 | D1, 24 h | headline band, even months | reported |
| C4 | D1, 4 h | headline band, even months | reported |

**Every contrast is the new arm minus H0.5, paired by transfer, with each transfer's outcome averaged
over the three seeds and whole calendar months resampled 20,000 times at 98.75%.** The headline band
is transfers of 25% to 40% of the receiving load lasting over 1 day, as in experiment 43. Each
contrast is also split by whether a metered receiver exists. Reported beside them, deciding nothing:
the placebo minus H0.5 and the new arm minus the placebo on seed 0, all nine bands, all months, and
the real-event counts. With 89 transfers the 98.75% interval on C1 is expected to be about 0.2 wide,
so a gain under about 0.10 may well come out `inconclusive`; that is the expected outcome if the
predicted gain is real but small.

**Diagnostics** (plots first, with the raw power before any residual): the measured power and both
residuals around three bank holidays at one Primary; the residual's median absolute deviation per
series and arm; the lag-1 and lag-48 autocorrelation of the residual and the lag-1 autocorrelation
of its daily mean; the share of the residual's squared sum inside logged closures and inside
labelled visible transfers, against their share of half-hours; the odd-month unexplained-alarm
count against threshold for each arm (the operating-point figure); unexplained alarms by distance
to the nearest bank holiday; recall by band; and the four contrasts with intervals. The mean
absolute error of the median is reported over the whole period and on bank holidays and the day
either side, as experiment 6's advice asks.

## Compute and checks

**About 5,200 monthly fits on 4 CPU cores, an estimated 2 to 4 hours.** The fits are 864 (16 series
times 54 months) for each of six arm-seeds that are not stored (H0.5 seeds 1 and 2, the new arm's
three seeds, the placebo), plus about 45 for the injection check. One job is one arm, seed, and
series, run by 4 workers with 1 thread each, no graphics card, each job writing its own file
atomically. The first series is timed; at over 15 seconds a fit, the run waits for a re-plan.

**Stop before detection when a check fails:** H0.5 seed 0 refitted for one series must match
experiment 17's stored predictions to 1e-4; the H0.5 panel through the new code must reproduce
experiment 43's alarm counts (432, 444, 297, and 110) and its 47 of 160 found by the 24-hour D0.
**Stop and look for a bug** when the new arm's mean absolute error is worse than H0.5's by more than
0.10 points of capacity.

## Size and process

- **Stored:** residual panels and alarm tables under `data/ml-experiments/00047-*`, never committed.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes; which better residual to test is a choice, made above.
- **Callers:** none outside the experiment; the scripts copy experiment 17's fit code and copy
  experiment 43's harness modules into this folder unchanged.

Medium, run as part of the SW2 exploratory series: Opus writes the plan, Sonnet implements, a
second Sonnet reviews the code, Sonnet runs and writes up, Sonnet reviews the prose, and Opus reviews
the result and writes the paragraph on what the next experiment should answer. **Series stop
rule:** experiment 43 refuted its own hypothesis, so a refuted or inconclusive result here is the
second negative result in the detector line; SW2-8 is already approved to run next, and if it is
negative too, the series stops under the skill's rule.

## Earlier advice

Followed: experiment 17's half-life of 0.5 years and monthly refit; experiment 37's comparison at
equal unexplained-alarm rates and its 24-hour single-series window; experiment 41's headline band;
experiment 43's split by metered receiver, end delay, and precision bracket; experiment 44's
recomputed share of the receiving load (through experiment 43's real-event code); experiment 11's
placebo calendar; experiment 6's scoring of a calendar column on its special days. Departed from:
experiment 43's `inject_power` rule (reason and check above); experiment 34's screening recipe of
150 rounds, because the baseline must be H0.5 exactly; experiment 6's half-year flag control,
because the column is constant beyond 7 days and the placebo covers the concern.

## Risks

- **The transfers avoid H0.5's real alarms.** Experiment 43 drew transfers at least 2 days from
  any H0.5 alarm, so none sits on a bank-holiday alarm, and the new column can help only through a
  lower threshold.
- **Power.** 89 headline transfers in even months resolve only a large gain.
- **Seeds.** Subsampling makes each fit's residual depend on its seed, so a transfer near the
  threshold can flip; averaging three seeds per arm and the placebo bound this.
- **D1 at 24 hours** has only 36 odd-month unexplained alarms to match, so its threshold is coarse.
