---
id: "00199"
slug: ps2b-engine-detector-v2-ramp-night-price
title: >-
  Does reading D8's engine by its edge shape, by night-only evidence, and by price cut the false
  alarms of the quantised engine detector, and are the "false alarms" at D5, D10, and D11 real
  engines?
theme: switching-detectors
also_relevant_to:
  - differentiable-physics-embedded-generation
status: planned
size: >-
  medium. Stored: private tables under data/ml-experiments/00199-ps2b-engine-detector-v2-ramp-night-price/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  (how each feature enters the detector, how night is defined, how price enters), all fixed below
  before any score is read. Callers: this folder's scripts only; the hidden Markov model code is
  copied from experiment 196.
hypothesis: >-
  Experiment 196's engine detector over-flags D8 after 2025-03 and raises false alarms at D5, D10,
  and D11 because it reads steps of any origin as engine units. Three facts about a gas engine are
  not used by that detector. (1) Its edges are single-half-hour jumps of a whole unit, unlike the
  gradual multi-hour edges of a switching transfer. (2) It can run at night, when solar panels
  cannot, so the evidence at night is free of solar noise. (3) It runs more often when the system
  price is high. Adding these three, one at a time and together, keeps the known-answer recall and
  removes the false alarms, except where the Embedded Capacity Register (ECR) lists a real engine.
prediction: >-
  Written before any result. Look-first (done before this plan was committed, on G2's recorded
  output): 71% of G2's start and stop edges are a single-half-hour jump and 29% have one partial
  half-hour; the 35 real transfers' edges (usable 12 onsets and 12 ends) have a partial half-hour in
  75% to 92%. The night-only arm keeps recall at 1 unit within 0.05 of 196's 0.83 and 0.85, and cuts
  the night flagged share at D8 from 2025-03 from about 40% to under 10%. The edge-shape arm cuts
  false active days at D10 and D11 by at least a third. The price arm cuts false alarms by under a
  tenth, because price moves the chance of running only from 9% to 27%. All three together meet the
  196 false-alarm bar at D3, D8 clean history, D10, D11, D14, and D16. The flagged days at D5 follow
  the spark spread more than matched unflagged days do; those at D10 and D11 do not.
decision_rule: >-
  Fixed before any score is read. Detector v2 is 196's frozen hidden Markov model with the three
  features. `supported`: v2's recall at 1 unit is at least 0.83 on K1 and on K2 (night rows, the
  v2 headline), its precision is at least 0.80 on both, and the 196 night-only false-alarm bar
  (at most 3% of clean night half-hours flagged and at most one false active day a month, an active
  day being 4 or more flagged night half-hours) is met at every Primary and period on the list
  below except the exception list. `partly supported`: recall and precision pass and the bar is
  missed at only one Primary or period. `not supported`: the recall at 1 unit falls under 0.83, or
  the bar is missed at two or more. Bar list: D3, D8 clean history (2022-01 to 2023-06 and July
  2023), D10, D11, D14, D16, and D8 from 2025-03 (judged in two periods, 2025-03 to 2025-12 and
  2026-01 to 2026-06, where the bar is relaxed to 3% plus the share of night half-hours below -2.5
  MW, because an engine may be present). Exception list, fixed here from the ECR workbook
  (printed_ecr.txt) before any flagged period was read: a Primary with a connected gas or oil engine
  (CHP or not) or battery of at least 0.5 MW registered within 5 km, connected before 2025-01-01.
  That list is D5 only (a 1.0 MW oil engine 3.25 km away, connected 2021-03-11). D11's only battery
  is 0.2 MW and connected 2026-04-13, and its other nearby plant is a 0.499 MW biomass steam turbine,
  which is not price-following; D10 has none. PS3 and later stages run only on `supported`, and on
  `partly supported` only for the periods and Primaries that pass. Each feature's separate value
  and the three together are reported with the same scores; no feature alone decides the verdict.
benchmark: >-
  Experiment 196's detector (frozen fits_fixed.pkl, wide prior, unit size 1.75 MW), scored on all
  rows and on night rows.
row_set_id: >-
  none (a detector is scored on injected half-hours and on clean history, not on forecasts)
metric: >-
  Recall by units running, precision, reconstructed-energy error, share of clean night half-hours
  flagged, false active days a month; at D5, D10, and D11, the night share of flagged half-hours
  and the spark-spread and system-price following of flagged days against matched unflagged days.
code: ml-experiments/00199-ps2b-engine-detector-v2-ramp-night-price/
---

Planned. The write-up replaces this page.

**Pre-registered design (fixed before any score is read).**

- **Detector.** Experiment 196's quantised hidden Markov model (frozen fit, 1.75 MW unit, noise fitted
  on D8's odd clean months), run with 196's detect-and-re-centre loop.
- **Night.** A half-hour is night when the sun's elevation at the Primary is below -3 degrees.
  Feature N1 scores the labels of the existing detector on night rows only. Feature N2 sets the
  reading of every non-night half-hour to missing before the model runs, so that only night
  evidence moves the state; labels are scored on night rows.
- **Edge shape (S).** A flagged spell (maximal run of flagged half-hours, joined across gaps of one
  half-hour) is kept only if both its edges are sharp: the largest one-half-hour change within one
  half-hour of the edge is at least 0.6 of the step between the means of the 3 half-hours before and
  after the edge, and that step is at least 0.5 of one unit. A spell cut by missing data keeps its
  other edge's test.
- **Price (P).** A flagged spell is dropped if its mean spark spread (40% efficiency, as experiment
  186) lies in the lowest fifth of all half-hours from 2023-06-01 to 2026-06-30. Before 2023-06-01
  there are no prices and the gate is not applied. In the injections, the price of an injected
  half-hour is the price at the time G2 recorded that output.
- **Scoring.** Known answers K1 and K2 as in experiment 196, on all rows and on night rows;
  recall, precision, energy error with day-resampled 95% intervals. False alarms as in 196
  (`false_alarms`), on night rows, with all rows as secondary.
- **Look first.** `e00199_shape.py` compares edge shapes of G2's recorded output (alone and injected
  into D8's noise), the 35 labelled real transfers, and large unlabelled steps in D8's clean history.
- **Step 5.** For flagged periods at D5, D10, and D11, report the night share of flagged
  half-hours, and spark-spread and system-price following against matched unflagged days (the
  method of experiment 186), and the ECR evidence. Supporting evidence, not proof; NGED has not been
  asked.
- **Lab shortcuts.** ERA5-driven expected power from the stored panel; smoothed posteriors.
- **Not used.** No forecast-improvement stage runs unless the gate passes.
