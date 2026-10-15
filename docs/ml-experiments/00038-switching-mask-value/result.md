---
id: "00038"
slug: switching-mask-value
status: planned
size: medium
hypothesis: >-
  For the 16 Primary substation demand series, excluding the half-hours of detected switching
  events from the history of the stack's error correction lowers the stack's threshold-weighted
  error at each series' historical 99th percentile on half-hours outside logged switching, by
  enough to justify adding a detector to the serving path.
prediction: >-
  On the proxy stack (experiment 00017's `H0.5` plus a correction), at lead 3.5 to 48 hours, on
  half-hours outside logged events, the 7-day correction with D1-detected half-hours masked
  (`d1_mask7`) lowers the threshold-weighted absolute error at P99 by 2% to 8% against the plain
  7-day correction (`plain7`), captures 40% to 80% of the gain of masking confirmed transfers from
  the logs (`oracle_conf14` against `plain14`), and a random mask of equal volume (`random14`) stays
  within 1% of `plain14`. `d1_mask7` improves 10 to 14 of the 17 confirmed events of 24 hours or
  more.
decision_rule: >-
  The primary contrast is `d1_mask7` minus `plain7` in threshold-weighted absolute error at P99,
  proxy stack, lead 3.5 to 48 hours, half-hours outside logged events, with a 95% interval from
  resampling whole calendar months (10,000 resamples). `supported`: a change of -5% or better with
  the interval below 0, `d1_mask7` better on at least 12 of the 17 long events, the mean absolute
  error on quiet rows within +/-0.01 points, `random14` within +/-1% of `plain14`, and the real
  stack (tier B) of the same sign. `partly supported`: the point estimate and the event count hold
  but the interval includes 0, or the gain falls under -2% once the largest month is removed.
  `refuted`: a change above -2%, or `d1_mask7` captures under 25% of the oracle gain, or `random14`
  gains as much as `d1_mask7`. `inconclusive` otherwise. Fewer than 12 long events with a usable
  window removes the per-event condition. The bar of -5% is set by the cost of porting: masking
  adds a detector and neighbour data to the serving path, so the gain must be about half the oracle
  ceiling and large enough for a reviewer to notice.
benchmark: >-
  `plain7`, the detector-free 7-day correction, for the primary contrast; `plain14` (the stack's
  current rule) and the oracle masks for the ceiling.
row_set_id: >-
  none (post-processing of stored residuals: daily origins at 09:00 UTC from 2022-01-15 to
  2026-03-25 for tier A; experiment 00019's stored runs for tier B)
metric: >-
  Threshold-weighted mean absolute error at each series' historical P99 (points of capacity), with
  the mean absolute error beside it; tier B uses the quantile score and its threshold-weighted form
  as in experiment 00037.
verdict: null
best_arm: null
lessons: null
advice: []
production_status: not-for-production
adopted_in: none
code: ml-experiments/00038-switching-mask-value/
---

# Experiment 38: is detecting a switching event worth anything to the forecast?

**Plan (status `planned`, written before any code runs).** Experiment 00037 found that the
neighbour-balance detector D1 recovers about a third of the MW of the visible transfers. This
experiment asks what that detection is worth: if the half-hours D1 flags are left out of the history
of the stack's error correction, does the forecast improve on half-hours outside logged switching?
The logs are used only to score (the oracle ceiling and the rows to score), never to build the
detector's intervals.

## Sizing (the five `plan-issue` triggers)

- **Stored:** scratch tables under `data/ml-experiments/00038-switching-mask-value/` only.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes (the interval rule and the correction window); both are
  fixed here before any result.
- **Callers nameable without searching:** yes, this experiment's folder only.

The size is medium. Compute is CPU only, 2 cores, under an hour. Sonnet implements, one Opus review
reads the code (this switching line has found real oracle bugs), Sonnet runs and writes up, and
Opus reviews the result and writes the proposed next experiment.

## Purposes the results speak to

1. **Lags in the forecasting model:** measured, as the effect on the 14-day correction of knowing
   which history half-hours lie inside an event.
2. **A normal-running-arrangement forecast in v1:** the out-of-event scoring is that view. Masking
   the training targets needs refits and is not tested.
3. **Telling NGED about their switching events:** the detected-interval table is written as a
   by-product.

**Two views of the target are reported, and the maintainer has to settle which one NGED wants.**
The normal running arrangement (NRA) view scores half-hours outside logged events, so a forecast
that ignores the switched load is rewarded. The metered-load view scores every half-hour, so the
forecast must follow the switched load while the event lasts.

## Forecast sets

- **Tier A, proxy stack (decides the verdict).** Experiment 00017's `H0.5` (ERA5 weather, each
  prediction out of sample, retrained monthly) plus a correction, at daily 09:00 UTC origins from
  2022-01-15 to 2026-03-25. The newest usable observation is 3.5 hours before the origin. Lead bands
  are 3.5 to 48 hours and day 6 to 7. **ERA5 arrives about 5 days after the event, so the proxy's
  weather inputs are not available at forecast time; the numbers are a ceiling for the correction
  rule, not a production forecast.**
- **Tier B, the real stack (reported, three to four events).** Experiment 00019's `n1h` arms (three
  seeds), runs every 5 days from July 2025 to 2026-03-25, with 00019's correction and 1.15 spread,
  scored as in experiment 00037. It must agree in sign with tier A, or the write-up says so first.

## Arms (post-processing; no refit)

Each arm is a rule for the correction at origin `o`; the history ends 8 half-hours before `o`.

- `plain14` (00019's rule), `plain7` (the detector-free bar).
- `d1_mask14`, `d1_mask7`: the plain rule over half-hours outside D1 intervals known at that time;
  with under 3 days left, the series' last correction that had 3 days is carried forward.
- `d0w48_mask14`: the same with D0 (24-hour window).
- `oracle_conf14`, `oracle_logged14`: masks from the logs (confirmed transfers; all 612 logged
  closures of 2 hours to 60 days), the ceiling.
- `random14`: each series' D1 intervals moved by a random whole number of days in +/-30 to 90, the
  placebo for volume.
- `d1_persist14`: `d1_mask14` plus, while a D1 interval is open, the mean residual of the interval's
  last 24 hours minus the masked correction (the metered view; scored on all rows).

## Interval builder

D1 (window 8, h = 4) and D0 (window 48, h = 4) alarms use experiment 00037's thresholds, with scales
from odd months. For each alarm, the builder tracks the level in relative residual (residual over
expected power, so a step that follows the moved load's own shape stays constant): the pre-level is
the median over the 7 days before onset, and the event continues in 24-hour blocks while the block's
mean relative residual minus the pre-level keeps the onset's sign and at least half its size. The end
is the end of the last qualifying block, with a cap of 60 days. The mask covers the alarm series and
every partner whose own first block qualifies. At origin `o`, an alarm counts only when its onset
plus window lies before the newest observation, and an open interval is masked up to that
observation. A leakage test checks that the intervals used at `o` do not change when residuals after
the newest observation are replaced by NaN.

## Scoring rows and intervals

- **Out-of-event rows:** valid times outside every logged interval of 60 days or less on the series,
  plus 2 hours either side.
- **Per-event windows:** for each confirmed event of 24 hours or more, valid times from onset to the
  end plus 16 days, out-of-event rows only.
- **Quiet rows:** out-of-event rows more than 16 days from any logged interval or D1 alarm: the cost
  of false alarms.
- **Intervals:** whole calendar months are resampled; per-event results are a sign count with a
  binomial test. Odd and even months are reported apart, and the aggregate again with the largest
  month removed.

## Injection check (secondary, decides nothing)

600 synthetic transfers are added to quiet periods: the moved load is a fraction (0.05, 0.15, or 0.3)
of the donor's own power, split by Dirichlet fractions across 1 to 3 neighbours, with the fractions
varying through the day in half the events, and durations drawn from the 28 confirmed closures of 6
hours or more. It reports hit recall, interval overlap, end error, and split-fraction error by size
and by number of receivers.

## Pilot context

The pilots behind this plan (in-sample, exploratory) put the oracle gain at 21% to 28% of the
out-of-event tail score of the real stack, 94% of it in one month (the 2025-12-30 closure of Primary
substations 17 and 18), and found a 7-day window better than a masked 14-day window on absolute
error. These are the reasons for the primary contrast being against `plain7`.

## Figures planned

A headline of per-event change by arm with the aggregate and its interval; raw series 17 and 18
around the 2025-12-30 closure with the logged span and the detected interval; six further raw
events and two quiet periods; the correction over time for series 18 under every arm; the monthly
gain; detected against logged interval length; the injection results; and tier B.

## Stop rule

`refuted` ends the switching line's forecasting purposes (1 and 2): the detected-interval table is
offered to NGED as an unvalidated list. `partly supported` also stops new experiments and records
the oracle ceiling. `supported` leads to a proposal to mask detected events from the training
targets as well.
