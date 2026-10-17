---
id: "00075"
slug: sw2-11-nra-forecast-with-labelled-switching-masks
status: planned
size: complex
hypothesis: >-
  On the 292 daily ENS forecast runs of experiment 48, a weekly-refitted Primary substation demand
  model whose training targets leave out the labelled switching half-hours forecasts the normal
  running arrangement (NRA) no worse than the plain weekly model, and giving the model the signed
  MW that switching moved at each lagged power input, or replacing each lag by the lag minus that
  MW, lowers the quantile score on half-hours outside switching.
prediction: >-
  Masking the training targets changes the NRA-row score by -0.01 to +0.01 points of capacity, since
  experiment 44 bounded a perfect detector's gain at 0.34% of 5.44 (0.02 points). Adding the signed
  MW as three columns changes the score by -0.03 to +0.01 points, and replacing the lags by the
  lags minus the MW changes it by -0.03 to +0.01. Neither magnitude interval is expected to exclude
  zero, because only 22 labelled events fall in the scored year.
decision_rule: >-
  Three planned contrasts on the NRA rows (the scored half-hours outside NGED's logged closures
  and outside the labelled events), quantile score over all leads, read at 98.33% (Bonferroni
  across three), whole validation months resampled. C1, the masked arm minus the plain arm: the
  mask is adopted (non-inferior) when the interval's upper end is at most +0.02 points, 0.34% of
  the plain arm's score, which is the most that experiment 44 says a perfect detector could gain,
  and so the size of harm worth worrying about; only the upper end of the two-sided interval is
  read, which is conservative; C1 is a gain when the interval lies wholly below zero. C2, the arm
  with signed MW columns minus the masked arm, and C3, the arm with lags minus MW minus the masked
  arm: each is supported when its interval lies wholly below zero, because either arm needs a
  detector on the serving path. The v1 NRA candidate is the supported magnitude arm; when both are
  supported, the one with the lower score on the NRA rows, and the page reports `plain` minus that
  arm descriptively. With no magnitude arm supported, the masked arm is the candidate when C1 is
  non-inferior and the plain arm otherwise. Every other comparison is descriptive.
benchmark: >-
  The plain weekly arm on the same rows, experiment 48's weekly row (5.44 on all rows), the 14-day
  corrected stack (5.64), `manual_heuristic`, and `climatology`, each from experiment 48's stored
  arms.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels), in points of capacity,
  on the NRA rows; the metered view (all half-hours) and the lag-in-event subset are reported and
  decide nothing
code: ml-experiments/00075-sw2-11-nra-forecast-with-labelled-switching-masks/
---

# Experiment 75: can a forecast learn the normal running arrangement from labelled switching?

**The question.** An electricity network operator sometimes closes a normally open switch, so that
one Primary substation (a substation that steps down to the local distribution network) carries
load that belongs to a neighbour. The metered power then differs from the power the substation
carries in its normal running arrangement (NRA). We ask whether a demand model trained without
those half-hours forecasts the NRA as well as one trained on everything, and whether telling the
model how much load switching moved lowers the error further. We use the 82 switching events that
[experiment 41](../00041-sw2-0a-labeller-gate/result.md) found in the power readings at the times
NGED logged. The result does not need a detector, so it is the ceiling for a model that must find
switching without NGED's log (SW2-12).

## Design

- **Architecture.** Weekly refits of the one-XGBoost-model-per-substation N1 recipe, no error
  correction, trained on the ENS control member, the winner of
  [experiment 48](../00048-weekly-retraining-confirmation-and-extension-demand-
  generators/result.md).
  The refits, the 292 runs, the seed 0, the 16 Primary series, and the scoring code are copied from
  that experiment, and the plain arm is rebuilt in this experiment so that every arm shares one
  code path. Each series trains on its own history.
- **Arms (all weekly, seed 0).**
    - `plain`: nothing removed.
    - `nra`: training rows whose valid time lies inside a labelled event of the series, as donor or
      as metered receiver, are dropped.
    - `nra_mw`: `nra` plus three columns, the signed MW that switching moved at each of the three
      lagged power inputs (newest usable observation, same time on the latest day, same time on the
      latest week), as a fraction of capacity. The sign is positive for a receiver and negative for
      a donor. A receiver's MW is the event's median moved MW times the receiver's share.
    - `nra_lag`: `nra` with each lagged power input replaced by the lag minus its signed MW.
- **Labels.** The visible events (conserved or one-sided) of experiment 41's labelled set, with
  the onset, end, donor, receivers, and median moved MW it stored. Events before the labelled set's
  start (January 2022) are not masked, and the training history runs from September 2019.
- **What the labels are.** The labels, onsets, ends, and moved MW are hindsight values that NGED
  would not have at the forecast origin: each event has one median moved MW for its whole length,
  estimated from telemetry that may follow the origin. This experiment is therefore an oracle
  ceiling for the masks and the MW columns, and SW2-12 measures what a detector loses. The mask
  drops only training rows already inside the refit's cutoff (valid time at least 3.5 hours before
  the refit), and the MW at each lag is looked up at the lag's own time, not the valid time. The
  labelled events of the other visibility classes (not visible, unassessable) stay in training,
  and logged closures without a labelled event are not masked.
- **Event sign and size.** The MW is divided by the series' effective capacity. A donor gets minus
  the event's median moved MW, a metered receiver gets plus that MW times its share, and shares may
  sum to less than 1. A receiver outside the 16 Primary series is ignored. A one-sided event with
  no metered receiver still marks and signs its donor. The events are the 82 visible ones (conserved
  or one-sided, kinds real and alarm), not the synthetic sentinel cards.
- **NRA rows.** The row set is fixed before any arm is scored. A scored case is kept when its valid
  time is outside every labelled event of its
  series and outside the logged closure interval of any card that names the series as a side.
  Every arm is scored on identical rows.
- **Not run.** The `base_margin` forecast-solar arm of
  [experiment 64](../00064-pvdis-5-net-demand-as-demand-minus-forecast-solar/result.md) is left out.
- **Calendar columns.** Every arm uses experiment 48's N1 columns, which include hour and weekday,
  so a model that memorises the calendar does so in all four arms alike. The contrasts are paired
  across arms with identical columns, and no separate control is run.
- **Reproduction check.** The `plain` arm on all 292 runs must match experiment 48's stored
  weekly score (5.44) to 0.01, and its per-case scores to 0.001 where the graphics-card fit is
  repeatable.
- **Descriptive extras.** `plain` minus each other arm on the NRA rows; Experiment 48's
  `manual_heuristic`, `climatology`, and stack rows on the
  same cases; experiment 49's P-roll 5.150 only where its rows match; each series on its own; the
  lag-in-event subset (NRA rows whose lag falls in an event), where the magnitude arms can act.

## Sizing

- **What gets stored:** four arms' predictions in the private data folder; nothing in `src/`.
- **Serving path:** untouched.
- **Degradation rule:** untouched. The magnitude arms would need a detector in production, which
  SW2-12 prices.
- **More than one defensible design:** yes, which is why there are three non-plain arms.
- **Callers not nameable without searching:** no; the code is a copy of experiment 48's.
- **Size:** complex, because the experiment stores a new prediction set and admits more than one
  design. Reviews: Sonnet plan, Sonnet code, Sonnet write-up, and Sonnet prose.

## Earlier advice followed

- **Followed:** retrain weekly with no correction (experiment 48, C1 and C4 to C6), and score a
  daily row set (experiment 48).
- **Departure:** none.
- **Limit carried forward:** experiment 48's weekly gain rests on one validation year, so the
  plain arm's absolute score is reported beside every contrast.

## Compute

Slot A, 10 cores and at most 24 GB of GPU memory: 5 worker processes of 2 threads, one refit per
job, one file per refit, series and arm, written atomically. Utilisation is checked in the first
minutes of the fit.
