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
  half-hour (74% for the recorded output alone, 71% after injection into D8's noise); the real
  transfers' edges (24 of the 70 edges of the 35 transfers are large enough to measure, 12 onsets
  and 12 ends) have a partial half-hour in 75% to 92%; but 65% of large unlabelled steps in D8's
  clean history are also single-half-hour jumps, so edge shape may not separate an engine from the
  steps that cause false alarms. The night-only arm keeps recall at 1 unit within 0.05 of 196's
  0.83 and 0.85, and cuts the night flagged share at D8 from 2025-03 from about 40% to under 10%,
  the cap of the relaxed bar. The edge-shape arm (whose thresholds came from this look at G2's
  edges, so K1 and K2 cannot test it independently) is expected to help little at D10 and D11; the
  false alarms there are the only independent test. The price arm cuts false alarms by under a
  tenth, because price moves the chance of running only from 9% to 27%. All three together meet
  the 196 false-alarm bar at D3, D8 clean history, D10, D11, D14, and D16. The flagged days at D5
  follow the spark spread more than matched unflagged days do; those at D10 and D11 do not.
decision_rule: >-
  Fixed before any score is read, and revised once after a Sonnet plan review (no score had been
  read). The verdict arm is v2 = night evidence (N2) plus edge test (S) plus price gate (P), scored
  on night rows. Known-answer thresholds, pooled over K1 and K2: recall at 1 unit at least 0.83,
  recall at 2 or 3 units at least 0.80, precision at least 0.80, energy error within 15%. K1 and K2
  are also reported separately, with day-resampled 95% intervals. False-alarm bar (the night-row
  version of experiment 196's bar, which that experiment scored on all rows): at most 3% of clean
  night half-hours flagged and at most one false active day a month, an active day being a UTC
  date with 4 or more flagged night half-hours. Bar list, each entry a separate test: D3, D8 clean
  history (even months, which the noise fit did not use), D10, D11, D14, D16, D8 2025-03 to
  2025-12, and D8 2026-01 to 2026-06. For the two D8 periods the share bar is relaxed to 3% plus
  the share of night half-hours below -2.5 MW, capped at 10%, because an engine may be present.
  `supported`: every known-answer threshold passes and every entry of the bar list passes.
  `partly supported`: recall at 1 unit and at 2 or 3 units pass, and either precision or the energy
  error fails, or exactly one entry of the bar list fails. `not supported`: recall at 1 unit falls
  under 0.83, or two or more entries of the bar list fail. D5 is on the bar list as a secondary
  line and does not enter the verdict, because the ECR lists a real engine there. That exception
  list was written after experiment 196 had shown D5 failing, from the ECR workbook alone
  (printed_ecr.txt) and before any flagged period of experiment 199 was read: a Primary with a
  connected gas or oil engine (CHP or not) or battery of at least 0.5 MW registered within 5 km,
  connected before 2025-01-01. Only D5 qualifies (a 1.0 MW oil engine 3.25 km away, connected
  2021-03-11). D11's only battery is 0.2 MW and connected 2026-04-13, and its other nearby plant is
  a 0.499 MW biomass steam turbine, which is not price-following; D10 has none. PS3 and later stages
  run only on `supported`, and on `partly supported` only for the Primaries and periods that pass.
  Each feature's separate value (v1 plus S, v1 plus P, N alone) is reported with the same scores
  and does not decide the verdict. K3 (G2's record injected into night rows of D8 from 2025-03,
  where an engine may already be present) is descriptive: recall only.
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

**Terms.** G2 is the metered generator whose recorded output (3 July to 15 August 2023) is the known
answer; its unit size is 1.75 MW as fitted in experiment 196. D3 to D16 are anonymised labels of
Primary substations. K1 replays G2's record into D8's reading at its true times; K2 replays it at
nine shifted places in D8's engine-free history. A partial half-hour is a half-hour whose reading
lies between the levels before and after an edge, because the engine started or stopped part-way
through it. Spark spread is the system price minus the gas fuel cost at 40% efficiency (experiment
186).

**Pre-registered design (fixed before any score is read).**

1. **Detector.** Experiment 196's quantised hidden Markov model (frozen fit, 1.75 MW unit, noise
   fitted on D8's odd clean months; other Primaries refit only the noise), run with 196's
   detect-and-re-centre loop.
2. **Night.** A half-hour is night when the sun's elevation at the Primary is below -3 degrees, at the
   half-hour's midpoint. Feature N2 sets the reading of every non-night half-hour to missing before the
   model runs, so that only night evidence moves the state; the edge test still reads the real
   reading. Labels are scored on night rows. The v1 arm is also scored on night rows, so a night-only
   score alone is not credited to N2.
3. **Edge shape (S).** A flagged spell (maximal run of flagged half-hours, joined across gaps of one
   half-hour) is kept only if both edges are sharp: the largest one-half-hour change within one
   half-hour of the edge is at least 0.6 of the step between the mean of the 3 half-hours before the
   edge and the mean of the 3 after it (skipping the half-hour next to the edge), and that step is at
   least 0.5 unit. The 0.6 and 0.5 came from the look at G2's edges, so K1 and K2 cannot test S
   independently; levels 0.5 and 0.7 are reported as a sweep. An edge beside missing data passes.
4. **Price (P).** A flagged spell is dropped if its mean spark spread lies in the lowest fifth of all
   half-hours from 2023-06-01 to 2026-06-30. Before 2023-06-01 there are no prices and the gate does
   not run, so P is scored on K1 and on the Primaries' histories from 2023-06 only. In the K2
   injections the price of an injected half-hour is the price at the time G2 recorded that output,
   which is favourable to the gate (a lab shortcut; the shifted-time price would remove the
   association the gate relies on).
5. **Scoring.** Known answers as in experiment 196, on all rows and on night rows, with day-resampled
   95% intervals. False alarms as in 196 (`false_alarms`), on night rows with all rows secondary.
   The synthetic-rate and reactive-floor arms of experiment 196 are not repeated.
6. **Look first.** `e00199_shape.py` compares edge shapes of G2's recorded output (alone and injected
   into D8's noise), the real switching transfers, and large unlabelled steps in D8's clean history.
7. **Flagged periods at D5, D10, and D11.** Report the night share of flagged half-hours, the
   spark-spread and system-price following of flagged days against matched unflagged days (the method
   of experiment 186), and the ECR evidence. Supporting evidence, not proof; NGED has not been asked.
8. **Lab shortcuts.** ERA5-driven expected power from the stored panel; smoothed posteriors.
9. **Not run.** No forecast-improvement stage runs unless the gate passes.
