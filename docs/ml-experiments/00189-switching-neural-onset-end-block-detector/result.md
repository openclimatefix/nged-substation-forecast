---
id: "00189"
slug: switching-neural-onset-end-block-detector
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00189-switching-neural-onset-end-block-detector/
  only (trained weights, injected-event tables, alarm tables). Serving path: not touched. Degradation
  rule: not touched. Designs: more than one is defensible (a window-level alarm head against a
  per-half-hour heat map; how the injector draws its shapes), so the plan fixes both before any
  real event is scored. Callers: this folder's scripts only, which copy experiment 181's loaders for
  the frozen convex-model residual panel and experiment 130's matching rule. One Opus audit is
  allowed, because injector leakage is the main risk.
hypothesis: >-
  A small neural network (under 50,000 weights) that reads a 7-day window of the convex-model residual
  of a Primary substation and its connected neighbours, and is trained only on injected transfers
  and on real stretches without a logged event, finds more of the 35 real transfers lasting over 1
  day than the pair alarm of experiment 156 does, and finds as many as RO-NC (the refit-offset
  neighbour-count detector), at the same number of unexplained alarms, and places the onset to
  within 3 hours. It can do so only if the injected transfers resemble the real ones, so the
  injector is fitted to the real events and checked for realism before training.
prediction: >-
  Written before any real event is scored. At the budget of 80 unexplained odd-month alarms, on
  real events held out by Primary pair, the network finds at least 15 of the 35 real transfers (pair
  alarm: 5; RO-NC on the convex-model residual: 18 with a 48-hour lag) with a median onset-location
  error under 3 hours. It fails if it finds fewer than RO-NC's 18 by more than 3, or if it beats
  the rule-based detectors only on injected worlds.
decision_rule: >-
  `supported` when, on the 35 real transfers pooled over the held-out folds, the network finds at
  least 15, its median onset error among found transfers is under 3 hours, it finds at least as
  many as RO-NC (18), and both realism checks pass (the injected-against-real classifier has a
  cross-validated area under the curve of at most 0.65, and the rule-based stand-in detector's
  onset recall on injected worlds is within 10 percentage points of its recall on the real events).
  `partly supported` when it finds at least 15 and the onset error is under 3 hours but either it
  finds fewer than RO-NC or a realism check fails. `refuted` when it finds fewer than 15, or when it
  finds at least 15 on injected worlds but fewer than 15 on real events. No porting threshold is
  set: the experiment decides only whether learning the shape from injections helps, and porting
  would add a model and a pipeline step.
benchmark: >-
  RO-NC on the convex-model residual (18 of 35 found, experiment 181), the pair alarm of
  experiment 156 (5 of 35), and the rectangular block detector of experiment 181 (0 of the 17
  unlabelled block days at 80 alarms).
row_set_id: >-
  none (scores alarms against the 35 real transfers of experiment 41, the 17 block days of
  experiment 175, and freshly injected transfers; no forecast)
metric: >-
  Real transfers found out of 35 at 80 unexplained odd-month alarms, with the matching rule of
  experiment 130 (an alarm within 2 hours of the labelled start and an end within the larger of 6
  hours and 10% of the duration), the onset-only count, the median onset error in hours, and
  block days found out of 17.
code: ml-experiments/00189-switching-neural-onset-end-block-detector/
---

# Experiment 189: can a small neural network, trained on realistic injected transfers, find real switching?

Planned. The write-up replaces this page when the experiment finishes.

## Plan

- **Inputs.** For each Primary, a window of 336 half-hours (7 days) of its convex-model residual
  (arm B of experiment 130, ERA5 weather, frozen: a lab shortcut) divided by the series' noise unit,
  the same for up to 4 connected neighbours (zero-padded, with a present-or-not channel for each),
  the sine and cosine of the hour of day, and a bank-holiday flag.
- **Network.** A stack of dilated one-dimensional convolutions, under 50,000 weights, with a
  window-level alarm output and a per-half-hour onset map and end map. The onset and end offsets are
  the maps' peaks.
- **Training data.** Windows from even-numbered months only: half are real stretches with no logged
  or labelled event (a fifth of them near bank holidays, a fifth on D8, a fifth on D9), half carry
  an injected transfer. The network never trains on a real event.
- **Injector.** Distributions are fitted to the real events outside the held-out fold: depth in MW and
  in noise units, duration, onset hour and weekday, shape (constant, donor-demand-shaped,
  PV-shaped, ramp), the receivers' opposite share, and the number of neighbours sharing it. Durations
  also draw on D8's 29 drops and the block days (6 hours to 19 days).
- **Folds.** The 66 labelled transfers are grouped by Primary pair into three folds. A model is
  trained for each fold on an injector fitted without the fold's pairs, and scores the fold's real
  transfers.
- **Scoring.** The alarm threshold is set so that unexplained odd-month alarms number 80, as for
  RO-NC in experiment 181. Real transfers are matched by experiment 130's rule.
- **Realism checks (before the real events are scored).** A classifier on the features of injected
  and real events, and the recall of a rule-based stand-in (a 24-hour step above RO-NC's threshold)
  on injected and real events.
- **Labelled lab shortcuts.** ERA5-based residual; hindsight inside the 7-day window; the injector
  adds MW to the residual, not to the power.

Advice followed: the unlabelled block days are scored as in experiment 181 (the nearest local day),
and the opposite share follows experiment 41's `conserved` and `one_sided` labels.
