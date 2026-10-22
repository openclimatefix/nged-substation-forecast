---
id: "00196"
slug: ps2-quantised-engine-detector-on-d8-with-known-answer-gate
title: >-
  Can a hidden Markov model of 0 to 3 engine units find D8's price-following engine in its residual,
  and does it pass a known-answer test built from the engine's own recorded output?
theme: switching-detectors
also_relevant_to:
  - differentiable-physics-embedded-generation
status: planned
size: >-
  medium. Stored: private tables under
  data/ml-experiments/00196-ps2-quantised-engine-detector-on-d8-with-known-answer-gate/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  (observation noise model, how the known answer is injected, how the retrain round works), all
  fixed below before any score is read. Callers: this folder's scripts only; they read the stored
  residual panel and G2's readings, as experiment 194 does.
hypothesis: >-
  The residual of D8 (metered minus expected power) contains an engine of about three units of
  about 2 MW each that starts and stops in spells of a few hours. A hidden Markov model whose state
  is the number of units running (0 to 3), read through the magnitude meter with no price input,
  finds the running half-hours at a low false-alarm rate. Following Hart (1992) and Kolter and
  Jaakkola (2012), it treats the reading as the expected load minus a whole number of units. The
  model does this where the step detectors built for load transfers (RO-NC, the block detector)
  find fewer than one running half-hour in five.
prediction: >-
  Written before any result. On K1 (G2's recorded output subtracted from D8's reading in
  2023-07-03 to 2023-08-15) and on K2 (the same record shifted by whole weeks into 9 windows of D8's
  history from 2022-01 to 2023-05): recall of half-hours with 2 or 3 units running at least 0.80,
  recall at 1 unit at least 0.50, precision at least 0.80, and the reconstructed energy within 15%
  of the injected energy. False alarms: at most 3% of clean half-hours flagged, and at most one
  false active day per month of clean history, on D8's clean history and on each of D3, D5, D10,
  D11, D14, and D16. The unit size fitted on D8's history from 2023-09 to 2025-12 lies between 1.7
  and 2.3 MW. RO-NC and experiment 181's block detector each find under 20% of the injected running
  half-hours. The prediction fails on any of these.
decision_rule: >-
  Fixed here, before any score is read, from section 1 and section 5 of the design brief. A half-hour
  is flagged running when the smoothed posterior P(k at least 1) exceeds 0.5. `supported`: K1 and K2
  both meet every threshold above, and the unit size lies in 1.7 to 2.3 MW. `partly supported`: only
  the 1-unit recall fails; the page then says PS3 should treat 1-unit spells as noise.
  `not supported`: the recall at 2 and 3 units or the false-alarm rate fails. `inconclusive`: the
  unit size leaves 1.2 to 3.0 MW between rounds (the model has fitted noise), or the known answer
  cannot be built. K2's synthetic injections at doubled and halved running rates are reported with
  the same thresholds but do not decide the verdict. RO-NC and the block detector are scored on the
  same injections; the 20% bound is a prediction, not a gate.
benchmark: >-
  Design brief price_sensitive_generation_design.md, section 1 (thresholds) and section 5 (PS2);
  experiment 194 (G2's record, the false-alarm panel, the left-tail shares); RO-NC (experiment 82)
  and the block detector (experiment 181) as comparison detectors.
row_set_id: none (a detector is scored on injected half-hours and on clean history, not on forecasts)
metric: >-
  Recall by units running, precision, reconstructed-energy error, share of clean half-hours flagged,
  false active days per month, fitted unit size per round.
code: ml-experiments/00196-ps2-quantised-engine-detector-on-d8-with-known-answer-gate/
---

# Experiment 196: can a model of 0 to 3 engine units find D8's engine?

*Planned; the write-up replaces this page when the scripts have run.*

Prediction, decision rule and stop rule are in the front matter. The rest of this page records the
design choices fixed before any score is read.

- **Size, one line per trigger.** Stored: private tables only. Serving path: untouched. Degradation
  rule: untouched. More than one defensible design: yes, fixed below. Callers: this folder's scripts.
- **Observation.** Reading `y_t = sqrt((L_t - k g)^2 + q^2) + b + e_t`, where `L_t` is the expected
  real power recovered from the panel's expected reading, `k` is the number of units running,
  `g` is the unit size, `q` is a reactive-power floor (0.14 MVA; sensitivity at 0, 0.5, and 1.0), and
  `e_t` is a first-order autoregressive error with Student-t innovations. The autoregression
  matters: D8's clean residual has lag-1 correlation 0.94, so an unmodelled slow error would be read
  as units. The noise parameters (`b`, `rho`, innovation scale and its slope in `L`, degrees of
  freedom) are fitted on odd months of D8's clean history (2022-01-01 to 2023-06-12 and July 2023)
  and held fixed, and false alarms are reported on the even months.
- **State.** `k` in 0 to 3 and a slow binary active-period state. Transition tables depend on the
  active state, six 4-hour blocks of the day, and weekday or weekend, with Dirichlet smoothing. They
  are fitted by expectation-maximisation on D8's history from 2023-09-01 to 2025-12-31 (the regime
  before the 2026 fall in the left-tail share), starting from uniform rates and never from G2's
  record. The unit size `g` is fitted by the same procedure with a Gaussian prior centred on 1.97 MW
  (standard deviation 0.3 MW, the design's prior from G2's record); a second fit with a prior
  centred on 2.5 MW and standard deviation 1.0 MW shows whether the data find 2 MW without being
  told.
- **Known answers.** K1 subtracts G2's recorded output from D8's reading through the meter in
  2023-07-03 to 2023-08-15, leaving out the half-hours where G2 has no reading. K2 shifts the same
  record by whole weeks (and by one hour for the clock change, so the evening peak stays at the same
  local time) to 9 windows 7 weeks apart from 2022-01-03. The detector parameters are fitted on
  2023-09 to 2025-12, so no known-answer half-hour is seen in fitting. G2's record is used for
  the injection and for the unit-size prior, and nowhere else.
- **Round rule.** Round 1 detects with the panel's expected reading. Each retrain step re-centres the
  expected reading by a monthly offset and an hour-of-day by weekday-type profile estimated on the
  half-hours with P(k=0) above 0.9, then refits `g` and the transitions. The convex demand model is
  not refitted (a simplification stated on the page). Stop after two retrain rounds, or earlier if
  `g` leaves 1.2 to 3.0 MW.
- **Regimes.** The detector is fitted on the regime before the fall and then applied, frozen, to
  2026-04 to 2026-06. Its output is reported separately for 2023-09 to 2026-01, 2026-02 to 2026-03
  (the transition), and 2026-04 to 2026-06.
- **Comparison detectors.** RO-NC's 24-hour step statistic with the neighbour-count check, and
  experiment 181's block statistic, both with the thresholds those experiments fixed, run on the
  injected residual. A spell counts as found when an alarm lies within 4 half-hours (2 hours) of its
  onset; the second figure counts every half-hour of a found spell.
- **Positive spells.** The report also counts flagged spells that raise the reading, because a
  battery near D8 could make them.
- **Leakage.** The smoothed posterior is for labelling history. Nothing here is a forecast-time
  input, and the detector reads no price.
- **Audit.** One fresh Opus audit attacks the known-answer gate and the handling of the break
  before the prose review, because the labels feed PS3.

Sources: [Hart (1992)](https://doi.org/10.1109/5.192069) for non-intrusive load monitoring and
[Kolter and Jaakkola (2012)](https://proceedings.mlr.press/v22/zico12.html) for the additive
factorial form.
