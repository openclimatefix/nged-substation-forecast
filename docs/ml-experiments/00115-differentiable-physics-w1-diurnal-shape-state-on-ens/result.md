---
id: "00115"
slug: differentiable-physics-w1-diurnal-shape-state-on-ens
title: "Differentiable physics, W1: a diurnal-shape state in the recent-error filter on ENS"
part: "W1"
theme: differentiable-physics-forecast-engine
also_relevant_to: [lagged-power-and-error-correction, differentiable-physics-long-history-and-drift,
production-reference-and-best-guess]
status: planned
size: >-
  medium. Stored: private files under the experiment's folder in data/ml-experiments/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is defensible (the shape's
  constraint, a joint filter or a second stage, the lead-band tuning), and the plan pre-registers
  one of each. Callers: this experiment's scripts only; the shape filter lives in the experiment
  folder and moves to studies.dp_assimilation, with unit tests, only if it is adopted.
hypothesis: >-
  GAM-K's remaining day-ahead error on ENS is mostly in the shape of the day inside each hour band,
  and that shape drifts faster than the convex model's monthly knots follow. A 48-half-hour shape
  state with zero mean in each hour band, added to the Kalman filter and updated from every reading
  the telemetry delay allows, lowers GAM-K's day-ahead quantile score at most decision substations,
  in months that played no part in choosing the idea's form and in the scored year.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  GAM-K as stored by experiment 111 (2.800 over leads 0.5 to 168 hours at the five decision
  substations, scored year), reproduced; the 14-day profile recipe of the debate's round 2 with its
  shrink re-chosen on the tuning months; GAM-0, X-nl+K, Xlag-d, and BG2 as stored by experiment 111;
  climatology; manual_heuristic
row_set_id: >-
  experiment 111's rows: tuning on the odd months and replicate on the even months of 2024-07 to
  2025-06 (origin and valid time both in the period's months), scored on the 292 runs of
  fold_a_daily_gap_v1 (2025-07-02 to 2026-06-29), at D5, D10, D11, D14, and D16, with D9 beside them
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored site, at day
  D+1 (leads 15.5 to 39 hours from the 09:00 UTC power_fcst_init_time, a UTC day); every lead band
  beside it
production_status: not-for-production
code: ml-experiments/00115-differentiable-physics-w1-diurnal-shape-state-on-ens/
---

**Status: planned. No fit has run.** This page is the plan, committed before any result exists.

**W1 asks whether the part of GAM-K's day-ahead error that lies in the shape of the day inside each
hour band can be removed by a filter that learns that shape from recent readings.** GAM-K is the
convex demand model driven by ECMWF ensemble (ENS) weather forecasts, with a Kalman filter on its
recent residuals (experiment 111). W1 adds a 48-half-hour shape state to that filter and scores
the change on the same rows, with every setting chosen on months that the scored year does not
include.

## What W1 asks, and the evidence it starts from

**GAM-K's day-ahead error on ENS sits mostly in the shape of the day.** On experiment 111's
scored-year predictions at six MW Primaries, its D+1 MAE of the median is 4.3 to 4.5 points, against
2.77 for the daily mean's error. Subtracting κ times the 14-day mean D+1 error profile by local
half-hour, day mean removed, lowered the MAE from 4.313 to 4.170 (the reviewer's reproduction):
0.04 to 0.22 per site at 6 of 6, 11 of 12 months, largest in winter, a loss at midday; κ = 0.5 ties
0.75. As a shift of K0's quantiles, the D+1 quantile-score gain at the 5 decision sites is −0.074
(98.75% about −0.112 to −0.037, 5 of 5); by lead band −0.072, −0.071, −0.056, −0.048; at midday
+0.035 to +0.053. **These numbers are a prior, not evidence:** the form and κ were chosen on the
scored year, so W1 re-chooses every setting on the tuning months, treats the even months of 2024-25
(never opened for W1) as the clean replicate, and requires the scored year to agree.

**GAM-K already carries a coarse shape:** its level and transient are per hour band (night 00:00 to
05:59, morning 06:00 to 10:59, midday 11:00 to 15:59, evening 16:00 to 23:59 local). The decision arm
adds only the shape inside each band, which nests GAM-K exactly; a daily zero-mean shape would also
speed up the band levels and mix the two effects. Also in scope: noise settings per lead band, and
the newest intraday readings. **NGED's live telemetry delay has never been measured**
(`data-processing.md`); W1 assumes 3.5 hours and sweeps it.

**Earlier advice followed, none departed from:** experiment 111's "feed the filter ensemble-driven
residuals" and "check a filter gain per substation before pooling" (per-site and leave-one-out
rows); experiment 113's weakly identified heating slopes (no per-site heating panel on the cards).

## The shape state

**Decision arm KW, a joint filter.** Per site, on experiment 111's analysis-proxy residual stream
r(t) (observation minus the DP driven by the latest ENS control run at lead 9 to 33 hours), with
h(t) the local half-hour (0 to 47, Europe/London, from the half-hour's local start time) and b(t)
its hour band:

- r(t) = ℓ_b(t) + z_b(t) + s_h(t) + ε(t), with ℓ_b and z_b GAM-K's per-band level and transient;
- s, 48 values, a random walk per day with step covariance q_s·P_W, where P_W projects onto shapes
  with zero mean inside each band; the initial covariance P_s0 is a multiple of P_W too;
- the forecast at any lead is GAM-K's forecast + s_h at the target's local half-hour, held flat.
- **Clock change:** on a 46-half-hour day the missing local hour gives no reading; on a 50-half-hour
  day the repeated local hour updates its two h values twice.

**Hyperparameters.** GAM-K's q, φ, and σ_z stay as experiment 111 fitted them; log q_s and log σ_ε
are fitted jointly per site by marginal likelihood on the tuning months' stream, under a shared
normal prior (empirical Bayes), q_s started at a 14-day-window gain.

**No switching masks in the filter**, as the series brief rules; a half-hour that ended less than
the delay before τ (09:00 UTC) is invisible; the state runs warm on experiment 111's stitched
out-of-sample stream. **The code** is `e<id>_shape.py` in the experiment folder, a vectorised numpy
update, copying `e00111_arms.py` and `e00111_ens.py`; if KW is adopted, a later PR moves it into
`studies.dp_assimilation` with unit tests.

## Arms

**Screening tier: no new GAM fit.** Every arm reuses experiment 111's monthly GAM-K fits, pinned
capacity, calibration, solar shrink, and ENS store, and changes only the correction. One seed.

| Arm | Correction | Role |
|---|---|---|
| K0 | GAM-K's per-band filter, as experiment 111 | benchmark, reproduced |
| KW | K0 + within-band zero-mean shape state, delay 3.5 h | the decision arm |
| KW-lb | KW with noise settings chosen per lead band | per-lead-band tuning |
| K0-lb | K0 with noise settings chosen per lead band | separates tuning from shape |
| KW-day | KW seeing only readings up to local midnight before τ | the newest intraday readings |
| R | K0 minus κ × the 14-day D+1 error profile, day mean removed | the round-2 recipe, κ re-chosen |
| R75 | R with κ = 0.75 | the screen's setting, for reproduction |
| KS | shape with daily zero mean only (exploratory) | KS − KW is the across-band part |
| KW-nomid | KW with s fixed at 0 from 11:00 to 15:59 local (exploratory) | the expected midday loss |
| KS1 | one daily level + one transient + daily zero-mean s, no bands (exploratory) | do bands matter? |
| KW-pl52 | KW's shape from 52 weeks earlier, scored year only (diagnostic) | drift or a seasonal shape? |
| KW-dX, K0-dX | KW and K0 at delays of 0, 24, and 33 hours (exploratory) | delay sweep |

- **Per-lead-band noise settings (KW-lb, K0-lb).** For each lead band (0–24, 24–48, 48–144,
  144–168 hours) one multiplier m on every state step variance (q, σ_z², and q_s) against σ_ε², from
  {0.25, 0.5, 1, 2, 4}, chosen on the tuning months by quantile score; each band takes its m's state.
- **KS1's hyperparameters** (q, φ, σ_z, σ_ε, q_s) are fitted jointly by marginal likelihood on the
  tuning months, from GAM-K's prior.
- **The recipe (R).** At τ, for each of the last 14 local days fully observed by τ minus the delay,
  take GAM-K's q50 error from the run that forecast that day at D+1 (raw residuals, switching rows
  included, as the series' no-mask rule says), remove the day's mean, average by local half-hour,
  and subtract κ times the profile at every lead. κ from {0, 0.25, 0.5, 0.75, 1} on the tuning
  months, with the spread rebuilt for each κ.
- **KW-pl52** uses the 2024-25 stream's shape on the same local date a year before each scored
  origin. A gain near KW's says the shape is seasonal and the GAM lacks it; a gain near zero says
  it is persistent drift.
- **Spread.** Each arm's quantiles are rebuilt by experiment 111's routine (`spread_with_factors`:
  post-filter residual quantiles per hour band, quarter, and lead band from the tuning months), with
  a spread factor per lead band from {0.8, 0.85, 0.9, 1.0, 1.15, 1.3} for every arm (experiment 111
  chose 0.9, its lower edge, for GAM-K in three bands); any factor at the new edge is reported. The
  51 members are pooled by `mixture_quantiles`. A shift-only check (K0's quantiles moved by KW's
  correction) is reported beside KW, tails included, so a gain in the spread cannot pass for one in
  the mean.
- **References as stored by experiment 111, re-scored on W1's shared rows:** GAM-0, X-nl+K, Xlag-d,
  BG2 (`p_s6_bg2`, "production recipe, not like for like"), climatology, manual_heuristic.
- **Past-weather check (exploratory).** KW's filter, q_s and σ_ε refitted on 2023-24, on experiment
  106's A2 residual streams for 2024-25 and 2025-26, against A2. A2 is the convex model without
  drift on ERA5 and CAMS, which reach a live service about 5 days late, so this is a lab condition,
  and A2's base differs from GAM-K's. A gain on both weathers puts the shape in demand; a gain on
  ENS alone puts part of it in ENS error; a gain on past weather alone means ENS noise swamps a
  demand shape.

## Periods and rows

| Period | Role |
|---|---|
| Odd months of 2024-07 to 2025-06 | tuning: q_s, σ_ε, the multipliers, κ, the spread factors |
| Even months of 2024-07 to 2025-06 | replicate: never seen by the screen; the clean test |
| 2025-07-02 to 2026-06-29, the 292 runs | scored: must agree; it chose the form, not the settings |

- **Period rows.** A tuning or replicate row needs both its origin and its valid time in the
  period's months, so long leads from late-month origins never score the other period's targets.
  The 14-day profile reaching into a neighbouring month is causal input, not leakage.
- **Sites.** Decision sites D5, D10, D11, D14, and D16 (4 of 5 needed); D9 beside them. No
  onboarding or short-history arms. Two series that carry embedded generation are out (see "Privacy").
- **Rows:** present rows outside the hand-filtered events, shared across a contrast's arms, all-rows
  score beside. **Lead bands:** D+1 (headline), 0–24, 24–48, 48–144, 144–168, and 12–36 hours.
- **Robustness rows for every planned contrast:** November to March against April to October;
  leaving one decision site out; each hour band; working days against weekends and bank holidays.
- **Predictions written:** the keys, `q01` to `q99` (11 levels), `actual`, `lead_hours`, `arm`,
  `seed`, `switching_mask`, plus `level`, `anomaly`, and `shape` (the s_h added) for filter arms.

## Known-answer gate (blocking, on the tuning months, before any replicate or scored run)

**A failed blocking test stops scoring until it is fixed; if it cannot be fixed inside the
experiment, the verdict is `inconclusive`.** Tests 1 and 2 run on real streams at the five decision
sites, outside the hand-filtered events.

1. **An injected persistent profile change is recovered.** Add to the stream for 60 days, from
   2024-11-01 and separately from 2025-01-01 (tuning months, after the October clock change):
   (a) +3 points from 17:00 to 20:00 local, inside the evening band; (b) a morning peak moved
   earlier, +2 points 06:30 to 07:30 and −2 points 08:00 to 09:00, zero mean in the morning band.
   Recovery is (KW's D+1 correction with the injection minus without) over the injected amount at
   the injected half-hours, averaged over days 8 to 14. From the tuned q_s and σ_ε, the plan's code
   computes the expected recovery analytically before the test runs. **Blocking pass:** the median
   over sites and starts is within ±0.15 of the expected recovery and at least 0.4, no site exceeds
   1.25, and the correction at every un-injected half-hour stays under 25% of the amplitude. **The
   brief's target (60 to 100% within 14 days)** is reported against the same reading; if the tuned
   q_s predicts a recovery under 0.60, the tuned filter is slower than the brief asked, which is a
   finding about the data, not a code fault. K0's and R's recoveries are reported beside (K0 about
   0.375 on shape a and 0 on shape b).
2. **No false shape on clean data.** Clean streams: (a) real streams with each day's half-hours
   randomly permuted within their band; (b) synthetic AR(1) streams at each site's fitted σ and φ
   with a random-walk level and no diurnal term. **Blocking pass, on each:** q_s and σ_ε refitted on
   the clean stream fall so that q_s is under a tenth of the tuned value; at that refit the RMS of
   s over the 48 half-hours is under 0.3 points and KW's D+1 score is within +0.02 of K0's. At the
   tuned q_s, the RMS of s is at most 1.2 × the square root of the mean diagonal of P_s and at most
   half its RMS on the real stream.
3. **Reproduction:** with q_s = 0, P_s0 = 0, and K0's σ_ε, KW equals K0's stored
   `work/dp_preds/gamk_tuning_<site>.parquet`: `level` and `anomaly` in float64 to 1e-9, the
   quantiles to 1e-4 (they are float32 after the 51-member mixture).
4. **Causality:** changing any reading of a half-hour that ended after τ minus the delay, or any ENS
   run issued after τ, leaves every arm's forecast at τ unchanged exactly; KW-day ignores readings
   after local midnight; the recipe uses only fully observed days.
5. **Zero mean:** s sums to zero within each band to 1e-9 after every update (daily for KS, KS1).

## Planned contrasts and margins

**Four planned contrasts, 98.75% (Bonferroni) month-resampled paired intervals from
`studies.bootstrap`, 3 seeds,** with site counts, absolute scores, the MAE of the median, and the
robustness rows, in the replicate months and the scored year.

- **C1, the shape state:** KW minus K0 at D+1. The decision contrast.
- **C2, per-lead-band tuning:** KW-lb minus KW, over 0.5 to 168 hours and per lead band.
- **C3, the newest intraday readings:** KW minus KW-day at D+1 and at 0–24 hours.
- **C4, state against recipe:** KW minus R at D+1.

**The smallest gain worth porting is 0.03 points at D+1.** The shape state adds no data and no
model, but it adds 48 states and a tuned setting to the serving filter. The threshold sits below
the D+1 noise (the screen's half-width is about 0.04), so it is a judgement about what is worth
porting, not a detection limit.

**Exploratory at 95%:** K0-lb minus K0; KS minus KW; KW-nomid and KS1 minus KW; R75 minus K0; the
delay sweep, read as (KW − K0) at each delay; KW-pl52 against K0 and KW; the shift-only check; the
past-weather check; the best W1 arm minus X-nl+K, minus Xlag-d, and minus BG2; D9.

## Predictions

Written before any fit; each line can fail, and none overrides the decision rule.

- **C1:** KW minus K0 is −0.03 to −0.10 at D+1 in both periods, 4 or 5 of 5 sites better in each;
  largest in November to February and in the evening band.
- **Midday:** KW is worse than K0 at midday by +0.01 to +0.05, and KW-nomid beats KW over all rows.
- **C1 by lead:** at least half of the D+1 gain remains at 48–144 hours.
- **C2:** KW-lb minus KW is 0 to −0.04 over all leads, m at 144–168 hours below m at 0–24 hours.
- **C3:** KW minus KW-day is −0.01 to −0.05 at 0–24 hours and under 0.02 at D+1.
- **C4:** KW minus R lies within ±0.03; κ is 0.5 or 0.75 (near certain, after the reproduction).
- **KS minus KW** within ±0.02; **KW-pl52 minus K0** above KW minus K0 (persistent drift); **R75
  minus K0** reproduces the screen's −0.14 MAE within 0.02; KW − K0 shrinks towards zero at delays
  of 24 and 33 hours; KW minus A2 below zero in both past-weather years.
- **Tails:** KW's q05 and q95 coverage within 3 points of nominal and q01 and q99 within 1.5 points
  in every lead band, as GAM-K's were. **Run time:** under 45 minutes of wall clock on 6 cores.

## Decision rule

**The verdict is C1's label.** C2 to C4 are labelled in the bullets and the take-home only. **This
rule departs from the series' Phase A rule on purpose:** W1 is not a numbered part, its contrast
compares two filters on one base with a half-width near 0.04, and the scored year chose its form, so
an interval and a second period are required, as in Phase B, with W1's own porting threshold of 0.03
in place of Phase B's 0.1.

- **`supported`:** C1 is below zero in both periods with at least 4 of 5 decision sites better in
  each, the scored-year interval lies wholly below zero, the scored-year estimate is at least 0.03
  points, and the gate passed.
- **`small gain`:** the scored-year interval lies wholly below zero with an estimate under 0.03,
  and the replicate months' estimate is below zero (an interval there is not required).
- **`partly supported`:** C1's estimate is below zero in both periods, but the scored-year interval
  includes zero or fewer than 4 sites improve in one period; or the scored year is `supported` and
  the replicate months reverse the sign.
- **`refuted`:** C1's interval lies wholly above zero in either period.
- **`inconclusive`:** anything else, or a gate test that cannot be fixed.

**Hand-ons, each needing the same sign in both periods:** KW-lb replaces KW only if C2 is below zero
in both at 4 of 5 sites; if C4 is not below zero, the simpler recipe R is the candidate; a
`supported` or `small gain` C1 makes the shape state a candidate for GAM-K.

## Checks: the seven audit-before-merge triggers

"Checks" gives median minus mean forecast per site and arm, the MAE of the median per contrast, and:

1. **A control fails:** a blocking gate test; K0's reproduction; R75's reproduction of the screen.
2. **A result contradicts a prediction or an earlier finding:** C1 against the screen's −0.074;
   the midday band; the past-weather check against ENS.
3. **Quantile score and MAE of the median disagree in sign** on any of C1 to C4.
4. **Leaving one decision site out flips a sign or a label** of C1 to C4.
5. **A nested arm beats its container:** K0 beating KW (which nests it), or KW-day beating KW.
   KW-pl52 is a diagnostic, not a null control, and is not read here.
6. **An interval more than twice or under half as wide as its sister's:** C1 against R75 − K0 at
   D+1 on the same rows and at the same level, and against GAM-K − GAM-0 at D+1 from experiment
   111's stored predictions.
7. **A later design depends on an unreplicated result:** fires if C1 is adopted while the replicate
   months disagree.

## What the write-up must contain

The take-home says how much day-ahead error the shape removes, where and when, whether the newest
readings and per-lead tuning add to it, and what that narrows the next step to.

- **Figures, raw data first:** a winter and a summer week per site (actual, K0, KW at D+1); GAM-K's
  residual by local half-hour per month; the C1 to C4 headline chart with intervals, site counts,
  and both periods; score by lead band per arm with BG2, X-nl+K, and GAM-0; gain by hour band and
  month; the gate (expected and measured recovery for KW, K0, and R); the tails.
- **Tails table:** coverage and pinball at q01, q05, q95, and q99, every arm and the shift-only
  check, per lead band, with threshold-weighted CRPS (`studies.tail_calibration`). No curtailment
  cost: experiment 87 found no event at the MW Primaries that could rank forecasts. **Absolute
  scores** of every arm and reference at D+1 and over 0.5 to 168 hours, on shared rows.
- **Substation cards** ("Provisional: fitted to past weather, forecast on ENS, screening run"; KW in
  the scored year): per decision site, the shape state as a half-hour-by-week heat map, the band
  level plus shape at four dates, the weekday GAM profile with and without the final shape, and the
  site's q_s, σ_ε, and lead-band multipliers. One cross-substation figure compares the shape's
  amplitude by month and its gain by hour band.
- **Caveats:** the scored year chose the idea's form; tuning and replicate months interleave; the
  3.5-hour delay is assumed; hindsight switching masks in scoring; the analysis proxy is the control
  member of the latest run; five substations. Inherited from experiment 111: the ENS calibration is
  fitted per hour band on the tuning months and applied in-sample there, which shrinks the diurnal
  error in the tuning stream and biases q_s and κ against W1; the pinned capacity reads hindsight
  weeks; the GAM's last training rows use ERA5 a live refit would not have; the spread is fitted
  on the tuning months.

## Privacy

**Substations appear by D-label, with dates, for D5, D9, D10, D11, D14, and D16.** Two further
series carry embedded generation and are not used. Any number taken from them would be labelled
G-c, normalised by its own capacity, with day counts and no dates, and never placed beside a
D-label. No generator's name appears with its series.

## Run order and compute

**Order:** code; gate tests 3 to 5; tuning; gate tests 1 and 2; replicate; scored year; exploratory
arms; report. **Compute: 30 to 45 minutes on 6 CPU cores, no GPU.** The 56-state filter, vectorised
in numpy, takes seconds per site and run (about 50 runs for q_s and σ_ε); experiment 111's
per-member run took about 35 s per arm and site, and W1 needs about 25 per site (15 to 20 minutes
over 6 workers). **Cut first:** KS1, KS, the delay sweep, the past-weather check.
