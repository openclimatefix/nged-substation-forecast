---
id: '00295'
slug: regime-segmentation-part-b-persistent-regimes-without-labels
title: >-
  Can persistent regimes be segmented in a Primary's history without labels?
theme: lsc-detectors
also_relevant_to:
  - retraining-and-training-history
status: done
size: >-
  medium. Stored: private tables under
  data/ml-experiments/00295-regime-segmentation-part-b-persistent-regimes-without-labels/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one is defensible
  for the residual, the penalty, and the classifier, so each is fixed below before any score is
  read. Callers: this folder's scripts, which call the reviewed `studies.dp_gam` fit and predict
  functions unchanged. Asset graph: not touched.
hypothesis: >-
  A segmentation of a Primary's whole history with hindsight, using only changes in the
  distribution of the residual of a lagless convex additive model (GAM), finds regime changes
  without labels. A regime is the whole period, without gaps, during which one configuration
  holds. For D8's gas engine, the regime is the whole period in which the engine is installed
  and included in the meter, including the spells when the engine is off. Whether the engine
  is running, and whether a load shift is in force, are states inside a regime. A boundary is
  a regime change when the distribution does not return to its earlier form and no neighbouring
  Primary shows the opposite change, and a state change otherwise.
prediction: >-
  Written before any fit. The segmentation places D8's break on 2023-08-15 within 14 days. At the
  14 Primaries with no injection, it finds under 1 regime change per Primary-year (a boundary not
  labelled as a state change). On injected steps in held-out Primaries and even-numbered months,
  it labels at least 80% of persistent steps as regime changes and at least 80% of temporary
  shifts as states. Inside D8's regime after 2023-08-15, at most 20% of the engine's on and off
  spell edges have a boundary labelled as a regime change within 3 days. It is expected to fail
  in two ways. The PV capacity steps of experiment 207 may produce false regime changes in
  summer, and a permanent transfer that moves load between two Primaries is expected to be
  labelled a state, because the neighbour rule reads any mirrored change as a load shift. The
  fade of D8's dips from March 2026 is expected to appear as a boundary, and the experiment
  cannot say from the residual alone whether it is a regime change or the engine not running.
decision_rule: >-
  Verdict `supported` when all four parts of the prediction hold: D8 dated within 14 days; under
  1 regime change per Primary-year (point estimate, with the upper end of the 98.75% bootstrap
  interval over Primaries under 1.5); both the persistent recall and the temporary recall at least
  80% on the held-out injections, with each 98.75% interval lower bound above 70%; and the share
  of engine spell edges with a boundary labelled as a regime change within 3 days at most 20%. A
  classifier that labels everything persistent fails on temporary recall, and a segmentation that
  splits D8's regime at the engine's spells fails the last part. `partly supported` when three
  parts hold. `not supported` otherwise. Steps within 150 days of the end of the data are not
  injected, because they cannot show a return. Every interval is 98.75% (the Bonferroni level for
  four contrasts). The engine's on and off spells come from the night-only detector of experiment
  199 (`v1`), which did not pass its own gate, so the spell-edge share is reported with that
  caveat and beside the share expected for boundaries at random dates. The forecast gain does not
  set the verdict. The offset is a candidate for production at any gain whose paired interval
  excludes zero, since it costs one extra number per series (threshold zero), and only if it also
  beats the trailing 60-day mean residual that needs no segmentation. Tuning uses odd calendar
  months and tuning Primaries; reporting uses even calendar months and held-out Primaries.
benchmark: >-
  No forecast benchmark. For the forecast gain: the same monthly-refitted GAM_ONLY without an
  offset, and with a trailing 60-day mean residual as the offset (no segmentation).
row_set_id: >-
  Own rows (not comparable with the leaderboard). Segmentation: the daily residual features of the
  15 Primaries of experiment 106's cache that have training rows (D15 has none), 2019-09-17 to
  2026-06-30. Forecast gain: the six monthly forecast periods after each injected step, and after
  D8's break.
metric: >-
  Break dating error in days; regime changes per Primary-year at the Primaries without injection;
  classification accuracy of injected persistent and temporary steps; mean absolute error change
  from the latest-regime offset in points of effective capacity.
code: ml-experiments/00295-regime-segmentation-part-b-persistent-regimes-without-labels/
date: 2026-10-23
verdict: not supported
best_arm: >-
  Not applicable (one segmentation method at one penalty). At the chosen penalty of 64: D8's break
  dated 15 days late, 0.87 regime changes per Primary-year (98.75% interval 0.46 to 1.39), 70%
  of injected persistent steps labelled persistent (0.58 to 0.85), 42% of injected temporary
  shifts labelled as states (0.21 to 0.64).
headline: >-
  A segmentation of the residual's daily distribution finds and dates a step of 25% of load (89%
  to 95% labelled correctly), finds only about half of 10% steps, and labels 42% of 14 to 60 day
  load shifts as states. D8's 2023-08-15 break is dated exactly at penalties of 16 or less and 15
  days late at the tuned penalty of 64, which misses the 14-day target by one day.
lessons:
  - >-
    Two of the four pre-registered parts held (false regime changes, engine spell edges), so the
    verdict is not supported. The edge part holds only because the chosen penalty leaves 3
    boundaries after D8's break.
  - >-
    The penalty trades false regime changes against recall. Recall of temporary shifts is 0.92 at
    penalty 4 and 0.42 at penalty 64, while regime changes per Primary-year fall from 6.5 to 0.9.
  - >-
    The first design, which segmented the daily mean level, gave the same recall as the final design
    on the 5 Primaries both ran, so reading the distribution did not change what the method can
    find.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  Looking only at the residual of a forecast that uses no recent power, a segmentation finds a
  lasting change of 25% of a Primary's load and dates it within a day or two, but it misses about
  half of 10% changes and cannot reliably tell a 14 to 60 day load shift from a lasting change.
  At D8 it finds the 2023-08-15 break exactly, but only at a penalty that also finds 2 to 3 false
  changes per Primary-year. An offset for the latest regime lowers the forecast error after a step
  by 2.2 points of capacity, but a plain 60-day mean residual does slightly better, so the
  segmentation is not worth adding for that purpose.
advice:
  - >-
    Do not expect a residual-only segmentation to find a step under about 15% of a Primary's load:
    it labelled 45% to 50% of 10% steps correctly and 89% to 95% of 25% steps (Figure 1).
  - >-
    Try a trailing 60-day mean residual as the offset after a step before building a segmentation,
    because the latest-regime offset was 0.15 points worse (0.03 to 0.27) than the trailing offset
    (Figure 3).
  - >-
    Do not take a boundary in D8's residual as a change of regime without a second source, because
    at penalties of 16 or less the segmentation puts 19 or more regime-change boundaries inside the
    regime that follows 2023-08-15, at about the rate expected from random dates (Figure 2).
  - >-
    Check a segmentation's penalty against the known break as well as against injections, because
    the penalty tuned on injections (64) dated D8's break 15 days late while penalties of 16 or
    less dated it exactly (Figure 2).
  - >-
    Test load-shift recall at several durations, because 14-day shifts were labelled as states 27%
    of the time and 60-day shifts 55% (Temporary shifts).
---

**Reading only the residual of a lagless forecast, a segmentation finds a step of 25% of a Primary's
load and dates it to within a day or two, but finds about half of the 10% steps and cannot reliably
tell a short load shift from a lasting change.** A Primary substation steps 33 kV down to 11 kV for
homes and businesses. A regime is the whole period, without gaps, in which one configuration holds.
Whether a gas engine runs, or a load shift is in force, is a state inside a regime. We fitted a
convex additive model (GAM) with no recent power as an input to each Primary's whole history, took
its residual (reading minus fit), and summarised each day's residual by four numbers: the share of
half-hours in a dip, the 5% and 95% quantiles, and the night-minus-day mean. A penalised
dynamic-programming search ([Killick et al. (2012)](https://doi.org/10.1080/01621459.2012.737745))
then placed boundaries. A boundary is a state change when the features return to their earlier
values within 120 days, or a neighbouring Primary shows the opposite change within 3 days, and a
regime change otherwise. We tested it on D8's break on 2023-08-15, on steps and load shifts that
we injected into the other Primaries, and on the unmodified series.

**Verdict `not supported`: two of the four parts of the prediction held.** The penalty (64) and
the minimum segment length (7 days) were chosen on injections in odd months at 7 tuning Primaries.
Every number below is from the 7 held-out Primaries and even-month injections, except D8.

- **D8's break (failed, narrowly).** The nearest boundary is 2023-08-30, 15 days late. Penalties
  of 16 or less place it on 2023-08-15 exactly (Figure 2).
- **False regime changes (held).** 0.87 per Primary-year (98.75% interval 0.46 to 1.39). Of 33
  regime changes, 8 look like solar capacity steps (a day shift at least twice the night shift)
  and 15 fall in April to August.
- **Persistent steps (failed).** 70% labelled persistent (0.58 to 0.85): 25% steps 89% to 95%, 10%
  steps 45% to 50% (Figure 1). Found steps are dated to a median of 1 day.
- **Temporary shifts (failed).** 42% labelled as states (0.21 to 0.64): 27% at 14 days, 45% at 30,
  55% at 60. A permanent transfer between two Primaries is labelled a state 73% of the time, as
  predicted.
- **Engine spell edges (held, trivially).** At penalty 64, 2% of the 121 on or off spell edges
  after 2023-08-15 have a boundary within 3 days, and 2% at random dates, because only 3
  boundaries remain. At penalty 16, 15% against 12% at random dates, with 19 regime-change
  boundaries inside the regime; at penalty 4, 39% against 29%.
- **The 2026 fade.** Penalties of 32 and above put a boundary on 2026-03-10, labelled unconfirmed,
  where the dip rate falls by 0.15 to 0.23. The engine detector's flagged share falls from 0.53 (October
  2025 to February 2026) to 0.43 (from 15 March 2026), so the residual cannot say whether a regime
  ended or the engine ran less.

## Figures

![Share labelled correctly by penalty, and regime changes per Primary-year](figure_1.svg)

*Figure 1. Recall of injected steps and shifts against the penalty, and the rate of regime changes
at the unmodified held-out Primaries. A larger penalty finds fewer false changes and fewer real
ones.*

![D8's daily features and the boundaries found](figure_2.svg)

*Figure 2. D8's dip rate, 5% quantile, and engine detector flagged share, with boundaries at
penalties 16 and 64 and the known break.*

![Forecast error after an injected step by arm](figure_3.svg)

*Figure 3. Forecast error after an injected step with no offset, with a trailing 60-day offset, and
with the latest-regime offset.*

## Forecast gain from an offset for the latest regime

- **Setup.** At each monthly cut (24-month window), the GAM's in-sample residual is segmented using
  days before the cut only, and the offset is the mean residual since the latest boundary. Scored
  over the month after the cut, for 6 months after each injected persistent step (14 Primaries, 8
  even-month onsets, 4 step types).
- **Injected steps.** Mean absolute error 9.79 points with no offset, 7.60 with the latest-regime
  offset (difference -2.20, 98.75% interval -2.57 to -1.83), and 7.45 with the trailing 60-day
  offset. The latest-regime offset is 0.15 worse than the trailing offset (0.03 to 0.27).
- **D8's real break.** No offset 16.94, latest-regime 16.93, trailing 16.09. The differences have
  intervals that include zero (6 cuts).

## Checks

- **Mean minus median forecast.** Zero by construction: GAM_ONLY gives one point forecast.
- **Quantile score beside the error of the mean.** No quantile score is contrasted here.
- **Audit triggers.** Trigger 1 fired: D8's known-answer test missed its 14-day bound by one day at
  the chosen penalty. The explanation, that the penalty was set on injections and not on D8, is
  tested by the penalty grid in Figure 2 and above. A fresh Opus audit is still required before
  merge. No other trigger fired, with trigger 7 unchecked because the dependants of this result
  are not known to this session.
- **Comparison with the first design.** The first design segmented the daily mean level. On the 5
  Primaries both designs ran (D10, D11, D12, D14, D16, 14-day minimum segment), recall of 25%
  steps was 0.66 to 0.73 for the level design and 0.65 to 0.71 for the distribution features, at
  similar regime changes per Primary-year (`printed_level_only.txt`).

## Caveats

- **The injected steps are easier and harder than real ones.** They are added to the recorded
  power at a fixed date with no change to the weather, and the GAM is fitted on the injected
  series, so month and weather terms absorb part of the step (a median of 2% for persistent steps
  and 6% for temporary shifts).
- **The engine spells come from a detector that did not pass its gate** (experiment 199,
  night-only `v1`), and the spell-edge share is not a measure of the engine's true spells.
- **D15 has no training rows**, so 14 Primaries other than D8 were used, not the 15 planned. D9 and
  D18 have under two years of readings, so their onsets are fewer.
- **The PV check approximates.** Experiment 207's step path was not refitted. A PV-like change is
  one whose daytime shift is at least twice its night shift.
- **Real regimes in the controls count as false changes**, so 0.87 per Primary-year is an upper
  bound.
- **Embedded Capacity Register context.** The register lists one connected entry of 8.9 MW at D8,
  dated 2019 or later, which is probably the engine. The session did not use the register.

## What the next experiment should answer

- Whether a second signal, such as the engine detector's flagged share together with the dip rate,
  separates a regime change from an engine that is off, because the residual alone cannot.
- Whether a lower penalty with a rule that merges boundaries within 14 days, which keeps the D8
  break exact, brings the false regime changes under 1 per Primary-year.
