---
id: "00037"
slug: switching-first-look
status: planned
size: medium
hypothesis: >-
  For the 16 Primary substation demand series, logged network switching (normal open points closed
  for hours or days) moves load that appears in the telemetry as a pair of opposite steps whose
  sum is conserved across the donor and its receiving neighbours, and a detector that tests that
  balance against the static interconnection list finds those transfers at a lower false-alarm
  rate than a detector that looks at each series alone.
prediction: >-
  Between 8% and 20% of logged closures of 2 hours or more show a conserved transfer (the
  date-shifted placebo gives 3% or less). On the even-month validation split at 0.5 unexplained
  alarms per series-month with a 4-hour window, the size-weighted recall of conserved transfers is
  0.20 to 0.40 for the single-series detector (D0) and 0.30 to 0.55 for the neighbour-balance
  detector (D1), the difference is at least +0.10, and the placebo recall is 0.08 or less.
decision_rule: >-
  The headline is the size-weighted recall (each conserved transfer weighted by its transfer size
  in MW) of the neighbour-balance detector D1 on validation transfers at the false-alarm budget
  fixed on the tuning split. `supported` when D1's size-weighted recall is at least 0.35, D1 minus
  D0 is at least +0.10 with a 95% interval above 0, and D1's placebo recall is 0.08 or less.
  `partly supported` when D1 reaches 0.35 but the D1 minus D0 interval includes 0. `refuted` when
  D1's size-weighted recall is below 0.20 or its size-weighted placebo recall is within 0.05 of the
  real recall. `inconclusive` otherwise (D1 between 0.20 and 0.35 with a placebo gap above 0.05,
  or a D1 minus D0 interval that cannot be read). Fewer than 15
  conserved transfers in the validation split removes the verdict on the detectors (Q3), and Q1 and
  Q2 are reported alone. Recall in the small-event band (under 10% of the receiving Primary's load)
  is reported and decides nothing. The threshold costs nothing in production because no production
  code changes, so the bar is set by what makes a second experiment worth running, not by a cost.
benchmark: >-
  The single-series detector D0 at the same false-alarm budget, and the date-shifted placebo logs.
row_set_id: none (this experiment scores detector alarms against logged switching, not forecasts)
metric: >-
  Size-weighted recall of conserved transfers at 0.5 unexplained alarms per series-month; the
  unweighted recall and recall by size band are reported beside it.
---

# Experiment 37: do logged switching closures show up in the telemetry, and does a neighbour-balance detector beat a plain one?

**Planned, not yet run.** NGED cannot give the switching logs in real time or at the scale of a
second model version, so a detector must infer events from the time series and the static
interconnection list. The logs (`Switching Logs.xlsx`, 31 sheets) are used only to validate the
detector and to set its thresholds on a kept-apart split. No detector reads them.

## Purposes the results speak to

1. **Lagged power in the forecast.** Whether a lag falls inside a switching event.
2. **Cleaner targets in the first model version.** Detecting events and removing them from the
   targets, so that the model trains on the normal running arrangement only.
3. **A report back to NGED.** The NGED forecasting team has no access to the switching logs, so a
   list of inferred events is information for them.

The write-up says which of the three each result bears on.

## Size

- **Stored data:** scratch tables and figures under `data/ml-experiments/00037-*`, which no
  pipeline reads.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Defensible designs:** two detectors (D0 and D1) are fixed in advance, with a fixed grid. The
  joint solve over all edges is left to a follow-up.
- **Callers:** all code is in `ml-experiments/00037-switching-first-look/`, and nothing imports it.

Size is medium. CPU only, at most 2 cores, under an hour of compute.

## Method

**Residual.** Experiment 00017's arm `H0.5` (XGBoost, ERA5 weather and calendar, no power lags,
retrained monthly with a 7-day purge). Residual MW is `(target - H0.5)` times the series'
effective capacity. ERA5 arrives about 5 days late, so the detector is retrospective. The weather
used is ERA5 and it is not available at forecast time, so nothing here is a live feature.

**Logs.** `Manual Time` is Europe/London local time and the telemetry stamp marks the end of each
half-hour, so an action at local time `t` affects the stamps after `t` in UTC. Closures of normal
open points are reconstructed per (sheet, switch) as intervals, and closures on different sheets
that share a Primary and start within 2 hours are one operation. Intervals over 60 days are
reported apart and excluded from recall.

**Oracle step (the reference set).** For each logged interval and side, the step is the mean
residual inside the interval minus the mean at the same clock hours a day either side (events up
to 24 hours) or the 3 days either side (longer), divided by a placebo scale. A **conserved
transfer** has both sides present, opposite signs, and both `|z| >= 3`. A one-sided sheet counts
when `|z| >= 4` and is reported apart. The placebo shifts every interval by 7 to 28 days (five
seeds).

**Step size follows the moved load.** Switching moves feeders, not a constant power, so the
transferred power follows the moved load's own daily and weather-dependent shape. The oracle also
reports the relative step (the donor's step divided by its own expected power over the event),
and compares the relative step in the first and second half of each long event to say whether the
step varies within an event. No detector assumes a constant step.

**Several receivers.** Load that leaves one substation may reappear at several neighbours. The
balance test sums the steps of all interconnected neighbours: the donor's drop must equal the sum
of the neighbours' rises (ratio 0.4 to 2.5), and the split fractions are an output. The write-up
lists which logged jobs involve more than one receiver, and reports the multi-receiver check as
its own analysis.

**Detectors.** D0 is a two-window mean difference on the residual per series, kept where a peak
exceeds a robust z threshold `h`. D1 keeps a D0 peak only when interconnected neighbours step the
other way at the same half-hour (plus or minus one) and the balance holds. Windows of 4 hours and
24 hours. Alarms whose window straddles a monthly refit are suppressed for the 24-hour window.
Thresholds and scales are set on **odd calendar months** (tuning) and scored on **even months**
(validation). The budget is the smallest `h` with at most 0.5 unexplained alarms per
series-month on tuning. An alarm hits an event when it falls within plus or minus 4 half-hours of
the event's first affected stamp, on any member series.

**Q2 (does it matter).** The share of the 00019 stack's quantile score and threshold-weighted
score at the 99th-percentile threshold that falls in conserved-transfer half-hours, in
logged-but-unconfirmed half-hours, and elsewhere (validation rows to 2026-03-25).

**Statistics.** The size of a transfer is `|T| = |step_b - step_a| / 2` in MW (`|step|` for a
one-sided sheet). Size-weighted recall is the sum of `|T|` over detected transfers divided by the
sum over all validation transfers. The share-weighted version uses `|T|` divided by the receiving
side's mean absolute power over the 7 days before the start (floor 0.5 MW) and is reported beside
it. Size bands are under 10%, 10% to 25%, and 25% or more of that load, and also by MW. The report
gives the top event's share of the weight, and recall with the largest event left out. Intervals
(95%) resample validation events for recall and series-months for the alarm rate, 10,000 times,
and resample D1 minus D0 pairwise. The headline window is 8 half-hours (4 hours), with 48 half-hours
as a second window. The grids are `h` in 4 to 8 for D0 and 2.5 to 6 for D1, neighbour z of 2, and a
balance ratio of 0.4 to 2.5. The oracle's placebo scales and the detectors' robust scales come from
odd months only. The oracle labels and the placebo recall used for the verdict are scored on even
months. Months from 2026-04 to 2026-06 are reported apart. Intervals of over 60 days are excluded
from recall.

**Tests.** Action classification on every distinct action string, the interval state machine on
hand-built sequences, local-to-UTC across BST, GMT and the October fall-back hour, the stamp
boundaries, the level shift on a synthetic step, D1 on a synthetic split across two neighbours
(accepted) and on a common-mode step (rejected), and a test that the detector module never imports
or opens a logs file.

**Q2 prediction.** Conserved-transfer half-hours hold at least twice the share of the
threshold-weighted score per half-hour that logged-unconfirmed half-hours hold. If conserved
transfers hold under 10% of that score, the write-up says so first, and the next experiment is at
most the cheap baseline check (F4).

## Questions

- **Q1.** How many logged closures since 2022-01 are conserved transfers, by duration and size
  band, against the placebo?
- **Q2.** What share of the stack's large errors lies in conserved-transfer half-hours?
- **Q3.** What size-weighted recall do D0 and D1 reach on validation at the fixed false-alarm
  budget, and against placebo logs?

## Figures planned

Raw data first, before any detector: the donor and its neighbours on a shared time axis with the
logged intervals shaded, zoomed to a few days around each event (best, worst, an invisible event,
and quiet periods). Then the step sizes, the conservation fingerprint (donor drop against the
neighbour sum), the placebo, detector alarms against logs, recall by size band, tail shares, and
alarm share by day of month.

## Risks

- The oracle label uses the same residual as the detectors, so recall on it is an upper bound.
- Unexplained alarms are an upper bound on false alarms, because real unlogged steps exist.
- MVA series (3, 9, 11, 14, 15, 24, 25, 26) break the sign test under reverse flow. Results are
  split by MW pairs and MVA pairs.
- The interconnection list lacks the pair 14-25, so those events are unattributable by design.

## Next experiment

Chosen after the results and written here then. The rule: F1 (event intervals and the value of
masking them from training targets) if the verdict is `supported` and conserved transfers hold
10% or more of the threshold-weighted score; F3 (a joint solve over all edges) if `partly
supported` or more than a third of the missed transfers are multi-sheet operations; F4 (a better
baseline residual) if `refuted` with transfers of 25% or more of the load missed over half the
time; F2 (the detector as a forecast feature) only after F1 shows a gain.
