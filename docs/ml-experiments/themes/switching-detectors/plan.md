# Switching detectors: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [SW2](#sw2-switching-event-detection) | Which change to the switching detector raises the share of moved MW found at an acceptable alarm rate? | Not set | Nothing | Iterations SW2-5 to SW2-9 are done (experiments 68, 61, 60, 55, and 62); further iterations are proposals |
| 203 | Do the extra real transfers that experiment 195's learned score finds lower the masked forecast's error after real transfers, without harm on ordinary days? | High (maintainer-approved 2026-10-22) | Nothing | Planned |
| 186 | Are D8's multi-day drops an embedded gas peaker running (spark spread, triads, flat blocks)? | High (NGED answer relayed 2026-10-22) | Nothing | Planned |

**Experiment 00116 (differentiable physics, NRA) also bears on this theme.** It fits flows on the
interconnection edges between Primary substations and scores them on injected transfers; see the
[Network research plan](../differentiable-physics-network/plan.md).

## Planned experiments

<!-- planned-table -->

## Proposals in detail

### SW2: switching-event detection

**The SW2 line improves the switching-event detector first, scored by event-level F1 against a
labelled set of visible events, and leaves the use of the detector until the detector works.** The
detector serves two purposes later. The first is a lower forecast error, if switching periods act as
unexplained noise for the XGBoost model and corrupt its lagged power inputs. The second, and the
more important to NGED, is a forecast under the normal running arrangement (NRA), into which NGED
can add its own switching. The [switching events](../../../roadmap/switching-events.md) roadmap page
proposes building that forecast by excluding switching periods from the training target. An event
table for NGED is a third purpose, at lower priority. The production detector has to run on power
alone, because the switching logs exist only for the trial area, so the logs are used here for
evaluation only.

**The line reopens a question that experiment 00038 closed, because experiment 00038 tested a
different use of the detector.** Experiment 00038 masked detected switching from the history of the
error correction that runs after the XGBoost model, and the mask made the forecast worse. No
experiment has yet masked switching from an XGBoost training target or from a lagged input.
Experiment 00037 confirmed 37 of 612 logged closures of 2 hours to 60 days as visible in its own
residual, and scored its detectors on the 18 of them in the validation months. That reference set is
small, about 6 of the 37 are expected by chance, and the label was built from the residual the
detectors use, so a detector that shares the residual's blind spots is flattered by the label. The
line therefore starts by building a reference set independent of the detectors.

The full labeller design is in the private file `data/ml-experiments/briefs/sw2_labeller_design.md`.

#### SW2-4 to SW2-9: detector iterations

**Each iteration tests one change to the detector against the [experiment
43](../../00043-sw2-1-detector-scoreboard/result.md) scoreboard, tuned on injections and odd months,
scored on even months, with recall weighted by the MW moved as the headline from SW2-7 onwards.**
The order below is a proposal, by expected gain per hour of work, for the plan of each iteration to
confirm:

- **SW2-5 (done, [experiment 68](../../00068-sw2-5-train-detect-retrain-loop/result.md)):** three
  rounds of refitting the lagless XGBoost model without the detected intervals changed the share of
  the moved MW found by +0.011 and the forecast error by -0.175%, and a random mask did as well.
- **SW2-6 (done, [experiment 61](../../00061-sw2-6-paired-step-intervals/result.md)):** pairing the
  onset and return steps into one alarm score finds less than D0M at the same alarm rate, but taking
  the end of D0M's interval from the strongest opposite step in the next 14 days finds 0.19 more of
  the moved MW.
- **SW2-7 (done, [experiment 60](../../00060-sw2-7-profile-weighted-step-detector/result.md)):** a
  step statistic that weights each half-hour by the expected power found the same transfers as the
  constant step on five injection shapes, so the step's shape does not limit the 24-hour detector.
  About a third of the transfers with a donor step of 6 noise units or more have the right onset and
  the wrong end, which SW2-6 (paired-step intervals) addresses.
- **SW2-8 (done, [experiment
  55](../../00055-sw2-8-neighbour-veto-single-series-detector/result.md)):** a single-series
  detector first, with neighbour balance as a sense check, not a gate. Every single-series alarm
  stands. A metered neighbour's opposite step raises the alarm's confidence and names the receiver,
  a same-sign step on the neighbours vetoes the alarm as a common-mode weather or calendar miss, and
  an alarm with no metered neighbour keeps its single-series confidence. Experiment 00043 found the
  neighbour-balance detector D1, which uses the neighbour test as a gate, finding almost none of the
  injected transfers without a metered receiver. The sum of the neighbourhood's residuals and a
  subset-sum over neighbours for a split feed the confidence score.
- **SW2-9 (done, [experiment 62](../../00062-sw2-9-whitened-residual-step-detector/result.md)):**
  whitening, experiment 00037's proposed F4, an autoregressive model of order 1 per series and hour
  of day fitted to the residual. The filtered detector found less of the moved MW (-0.097 at 24
  hours), because the filter shrinks a day-long step faster than the noise of the 24-hour step.
