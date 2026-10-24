---
id: '00334'
slug: primary-demand-other-neural-architectures
title: >-
  Does a neural network that replaces the whole convex stack (a mixture-density network, a
  recurrent network, or a temporal fusion transformer) forecast Primary demand better than Primary
  demand research default 3?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00334-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible, but the four arms are fixed
  by the question. Callers: this folder's scripts, which import the folders of experiments 278,
  311, 317, 320, and 322 and call reviewed `studies.*` modules without changing them. No change
  under `packages/`.
hypothesis: >-
  The maintainer and National Grid Electricity Distribution (NGED) value the explainability of the
  generalised additive model (GAM) with a Kalman filter that makes up Primary demand research
  default 3. A different architecture that gains decisively is the only reason to leave it. Four
  neural networks that replace the whole stack are tested against the default: a small
  mixture-density network (MDN) on the served inputs (M1), the same network given only an explicit
  unmetered-solar signal (M2), a gated-recurrent-unit encoder-decoder network with an MDN head
  (R), and a temporal fusion transformer (T).
prediction: >-
  Written before any score is read; each can fail. (P1) M1 trails the default by 0.10 points or
  more pooled at 0.5 to 240 hours, because the GAM supplies the physics (heating, lighting,
  solar) that a small network has to learn from six Primaries. (P2) M2 beats M1 by at least 0.03
  points, because the explicit solar term carries the sunny-daylight half-hours. (P3) R and T are
  no better than M1 by 0.03 points; they differ from M1 only in how they read the past. (P4) Seed
  noise is 0.01 to 0.02 points pooled (experiment 325), so every contrast under 0.03 points is read
  on at least 6 seed sets averaged. Stop rule: the reproduction gate fails (the default equals
  experiment 322's default to within 0.01 points at 0.5 to 240 hours), or after the first full run
  an arm trails the default by more than 0.30 points at every lead band, in which case that arm is
  reported and dropped; the temporal fusion transformer is dropped first if the budget runs short.
decision_rule: >-
  Adoption bar (maintainer, 2026-10-24): the GAM stays unless an arm shows a decisive improvement,
  and the usual minimum gains (0.02 and 0.03 points) do not apply. No arm is called a candidate for
  production. Each arm's gain over Primary demand research default 3 is reported with its 95%
  month-block interval, with and without D10, and the maintainer decides what counts as decisive.
  The gap of the best neural arm to the physics-plus-adapter-plus-MDN variant (experiment 330's
  STD, 0.5 to 168 hours only) is reported in absolute and relative terms.
benchmark: >-
  Primary demand research default 3 (experiment 251's arm 1 below 168 hours and experiment 322's
  `d1` above), experiment 330's STD (0.5 to 168 hours), the manual heuristic, and climatology.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels), in points of effective capacity;
  twCRPS at the 95th percentile; pinball loss and coverage at q01, q05, q95, q99.
code: ml-experiments/00334-primary-demand-other-neural-architectures/
date: 2026-10-24
---

Plan only. The write-up replaces this paragraph when the scores exist.

## Plan

- **Lead range:** 0.5 to 240 hours on every headline. The headline figure is score against lead.
- **Rows:** `fold_a_daily_gap_v1`, six Primaries (D5, D9, D10, D11, D14, D16), scored year
  2025-07-02 to 2026-06-30, replicate year 2024-25 (even months), origin 09:00 UTC, rows outside
  the hand-filtered events. The default's rows are experiment 251's below 168 hours and experiment
  322's above.
- **Refits and training pool:** monthly refits on a 24-month pool (experiment 330's schedule, the
  P-roll cadence of the research-default rule), with every fourth pool month held out for early
  stopping. Weekly refits are not used, so absolute scores are quoted from monthly refits for
  every arm, the default included.
- **Inputs shared by all arms (served at forecast time):** the 51-member ensemble-mean temperature
  and shortwave irradiance from the ECMWF ensemble (the driver weather of experiment 278), the
  driver temperature 24 hours earlier, the temperature hinges at 15.5 °C, the solar term
  (shortwave times the sine of the sun's elevation), darkness, the sun's elevation, half-hour and
  weekday and day-of-year harmonics, the day type, bank-holiday, Christmas, and Economy 7 flags,
  the lead, a site indicator, and recent power (the power lagged by a whole number of days beyond
  the lead, the mean of the last 24 hours before the origin, and the last reading before the
  origin). Past rows before April 2024 use the analysis proxy as the driver, as in experiment 278.
- **Departures from the brief, stated before any score:** (1) experiment 220's recent-error
  features are errors of the stack, which the arms do not have, so recent power stands in for
  them; (2) the store holds ensemble means of temperature and shortwave only, so member spread and
  wind are not inputs; (3) the explicit solar signal of M2 omits the D5 export clip, which the
  network can learn from the signal and the temperature.
- **Arms:**
    - **M1:** two hidden layers of 64 units and a three-component Gaussian mixture head (the
      network of experiment 311), on all shared inputs, trained on the negative log-likelihood of
      the site-standardised power. Calibration is the as-of calibration of experiment 164, applied
      to the stored quantiles.
    - **M2:** M1 with the weather inputs reduced to the explicit solar signal (the pinned capacity
      of experiment 45's real-time vintages times the unit PV shape of the ensemble-mean
      shortwave, derated 0.4% per kelvin), the temperature and its hinges, and the calendar,
      darkness, lead, site, and recent-power inputs.
    - **R:** a gated recurrent unit encoder over the last 7 days of half-hourly power ending 3
      hours before the origin, and a gated recurrent unit decoder over the 480 lead steps that takes
      the future weather and calendar inputs at each step, with the same mixture head.
    - **T:** a temporal fusion transformer with the smallest published configuration (hidden size
      16, one attention head, one LSTM layer), static (site), known-future (weather, calendar,
      lead), and observed-past (power) channels, and its own 11-level quantile output. Tuning
      budget, fixed now: three configurations (hidden 16 with learning rate 1e-3, hidden 16 with
      3e-4, hidden 32 with 1e-3), chosen once on the held-out pool months of the 2025-07 refit by
      pinball loss, then used for every refit.
- **Seeds:** 6 per refit for the scored year; 3 for the replicate year.
- **Reference rows:** the default, STD (to 168 hours), the manual heuristic, and climatology.
- **Reproduction gate:** the default rebuilt from experiments 251 and 322 equals 2.727 points at
  0.5 to 240 hours to within 0.01 points.
- **Report:** each arm alone first, then gaps to the default with and without D10, the protected
  groups (cold winter nights, sunny daylight, tails) with the research-default tolerances,
  tail calibration at q01, q05, q95, q99, and the checks of the `ml-experiment` skill. One sentence
  per arm says how a reader could explain a single forecast from it compared with the GAM.
- **Five triggers:** stores private forecasts and fits only; not on the serving path; no
  degradation rule; the four arms are fixed by the question; callers are this folder's scripts.
