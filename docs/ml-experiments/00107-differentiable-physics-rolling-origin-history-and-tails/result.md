---
id: "00107"
slug: differentiable-physics-rolling-origin-history-and-tails
title: >-
  Differentiable physics: does more history help the convex model at 12 rolling origins, and can
  its tails be calibrated?
part: n/a
theme: differentiable-physics-long-history-and-drift
also_relevant_to: [differentiable-physics-demand-calendar-and-weather, calibration-and-uncertainty,
retraining-and-training-history]
status: done
size: >-
  medium. Stored: private files only. Serving path and degradation rules: not touched. Designs:
  more than one for the spread pool and tail model, one pre-registered. Callers: own scripts only.
hypothesis: >-
  Refitted quarterly from 2023-07 to 2026-04 and scored on the next 3 months, the convex model's two
  history contrasts (all years against 12 months, drift against none) keep one sign at almost every
  origin. A second claim, labelled separately as C5 and not part of the verdict: a level-dependent
  spread brings its 95% and 99% quantiles within 1 point of nominal coverage without worsening the
  quantile score.
prediction: see "Pre-registered predictions"
decision_rule: see "Pre-registered decision rule"
benchmark: >-
  Lagless XGBoost (12 months, same inputs, same origins): default start, no day-of-year column, and
  experiment 64's solar offset with a quantile start; climatology; manual_heuristic
row_set_id: >-
  dp_mw_v1 (experiment 103); MW sites; 12 quarterly origins 2023-07-01 to 2026-04-01, each scored on
  the following 3 months
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored site; tail
  coverage and pinball loss at q01, q05, q95, and q99
verdict: partly supported
best_arm: >-
  G-all+D1 (all years, drift D1, pinned capacity), quantile score 2.776 points; the exploratory drift
  step 10,000 scored 2.753
headline: >-
  Drift minus no drift on all years (C2) is -0.451 points, 99% interval -0.681 to -0.232, with 10 of
  12 origins below zero; all years minus 12 months without drift (C1) is unstable.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  Giving the convex demand model a drift term lowered its score at 10 of 12 quarterly refits, by
  0.45 points over all years (interval 0.23 to 0.68), so drift is worth keeping. More years without
  drift raised the score at 8 of 12 origins, almost entirely because of one substation (D10), and
  more years with drift against 12 months stays unresolved (winter-quarter origins favour more
  years, summer ones are level). The level-dependent and conformal spreads made the 95% and 99%
  quantiles worse, not better, so the tails still need another idea.
date: 2026-10-19
pr: 0
advice:
  - >-
    Give a convex demand model a drift term when it trains on all years and refits quarterly: G-all+D1
    minus G-all was -0.451 points (99% interval -0.681 to -0.232), with 10 of 12 origins below zero
    and all 3 origin years negative (C2).
  - >-
    Never judge a history window on one site or one year: D10 alone made all years without drift
    score worse than 12 months (+1.139 at D10, +0.031 without it, C1), and three of the five
    substations flip the sign of C2's stability when left out.
  - >-
    Leave the day-of-year column out of lagless XGBoost trained on 12 months: it raised the score by
    0.118 points over 12 origins (K3) and by 0.75 points over the four burn-in windows, and the bar
    without it is the fairer benchmark for the convex model (C4).
  - >-
    Keep the two day-of-year harmonics in the all-years convex model with drift: removing them raised
    the score by 0.389 points (K2).
  - >-
    Do not adopt the level-tercile spread (S1) or the conformal shift (S2): S1 raised the score by
    0.050 points and S2 by 0.489 (C5q, T1), and both moved the 95% and 99% coverage further from
    nominal (S1 89.7% and 95.4%, S2 84.0% and 91.5%, against 91.8% and 97.3% for the rolling-pool
    spread S0). Tail calibration needs a different idea, such as a spread that grows with the age of
    the fit.
  - >-
    Score every history-window contrast at many refit origins, split into winter-quarter and
    summer-quarter origins: C3 was -0.461 at winter-quarter origins and +0.058 at summer-quarter
    ones, and the pooled value hides both.
code: >-
  ml-experiments/00107-differentiable-physics-rolling-origin-history-and-tails/
---

**Giving the convex model a drift term is the one history decision that held up across 12 quarterly
refits.** Drift lowered the quantile score at 10 of 12 origins, by 0.45 points over all years. More
years without drift did not help: it raised the score at 8 of 12 origins, and one substation (D10)
caused almost all of that. More years with drift against the last 12 months is unresolved, because
winter-quarter origins favour more years and summer-quarter origins are level. The new spreads did
not fix the tails, so the 95% and 99% quantiles still cover 91.8% and 97.3% of outcomes. Verdict
`partly supported`.

We asked whether the two history contrasts of [experiment
104](../00104-differentiable-physics-2-simple-structural-model/result.md), which had reversed sign
between its tuning year and its scored year, keep one sign when a convex model of demand minus solar
output is refitted every quarter. We refitted it at 12 quarterly cuts, from 1 July 2023 to 1 April
2026, and scored each fit on the 3 months after its cut, at five Primary substations (D5, D10, D11,
D14, and D16). Four earlier cuts (2022-07 to 2023-04) only supply the first spread pools. We also
asked whether a spread that depends on the forecast level brings the upper quantiles to their
nominal coverage. In the figures, "score" means the quantile score: twice the mean pinball loss
over 11 quantiles, in percentage points of each substation's effective capacity, where lower is
better.

![Per-origin contrasts C1 to C4](figure_01_headline.svg)

**Figure 1 shows C2 below zero at 10 of 12 origins and C1 above zero at 8 of 12.** Each panel is one
planned contrast at the 12 origins, with a 95% interval from week blocks, the tie band, and the
pooled estimate at the right. C3 and C4 move with the season of the scored months.

![Every arm's score per origin and substation](figure_02_absolute.svg)

**Figure 2 shows that no arm leads at every substation, and that climatology and the manual heuristic
trail the rest.** Arm scores run from 2.75 to 3.23; climatology scores 3.95 and the manual heuristic
3.92.

![Leave-one-site-out sign counts](figure_03_leave_one_out.svg)

**Figure 3 shows that leaving one substation out changes the sign counts of C1, C2, and C4.** Without
D10, C1 has 3 negative, 5 positive, and 4 tied origins.

![Tail coverage by origin](figure_04_tails.svg)

**Figure 4 shows that none of the three spreads reaches the nominal 95% and 99% coverage at most
origins.** The conformal spread (S2) is furthest away and swings most between origins.

![Summer and winter weeks](figure_05_weeks.svg)

**Figure 5 shows the forecasts in one summer week and one winter week.** The three forecasts differ
by less than the gap to the measured power at D16 in the winter week.

![Controls, replication, and tail-model contrasts](figure_06_exploratory.svg)

**Figure 6 shows that day-of-year terms help the convex model with drift and hurt XGBoost.**

![D16's mean forecast error per origin](figure_07_d16_bias.svg)

**Figure 7 shows that the forecasts without drift miss low at D16 by 3 to 4 points on average.**

![Pinned and fitted PV capacity](figure_08_capacity.svg)

**Figure 8 shows pinned PV capacity above the fitted capacity at four of five substations.**

## Results

**Arms and rows.**

- **Convex arms** (nine, all at Part 2's tuned penalties: ridge 0.1, drift step 3,000): G1 (12 months),
  G-all (all years), G-all+D1 (all years, drift D1), each with PV capacity pinned to experiment 45's
  real-time vintage, and the same three with fitted capacity (`-fc`, Part 2's configuration); G-all+D1
  with drift step 10,000; and G1 and G-all+D1 without the two day-of-year harmonics (`-noharm`).
- **XGBoost arms:** X-12 (12 months, default start), X-12-nodoy (without the day-of-year column),
  and X-12amq (quantile start plus experiment 64's solar offset, from the 2023-10-01 cut on). The
  XGBoost bar is X-12-nodoy, chosen on the four burn-in windows before any scored window was read
  (burn-in score 3.82 against 4.58 for X-12).
- **Spreads:** S0 is Part 2's centred spread, fitted on a rolling pool of the arm's own out-of-sample
  residuals over the 12 months before each origin. S1 adds a tercile of the predicted mean. S2 is S0
  plus a conformal shift calibrated on the newest 3 months of the pool.
- **Rows:** experiment 103's `dp_mw_v1` rows outside the hand-filtered switching events, 254,756
  half-hours at the five decision substations. Fits are frozen from their cut to the next origin
  and run on the CPU.
- **Past weather:** ERA5 temperature and CAMS irradiance, which exist only after the event, so every
  score is a ceiling for a forecast made on a weather forecast.
- **Not run:** D18 (its data start on 2025-03-01, so a 12-month pool of out-of-sample residuals cannot
  exist), and the bank-holiday, Christmas-run, and days-below-0°C rows.

**Absolute scores** (pooled over 12 origins and five substations; mean forecast error is measured
minus forecast, in points):

| Arm | Score | Mean forecast error |
|---|---|---|
| G-all+D1, drift step 10,000 | 2.753 | +0.137 |
| G-all+D1, fitted capacity | 2.768 | +0.014 |
| G-all+D1 | 2.776 | -0.018 |
| G1, fitted capacity | 2.907 | +1.046 |
| G1 | 2.972 | +1.495 |
| X-12-nodoy (the bar) | 3.024 | +0.911 |
| G-all, fitted capacity | 3.067 | +0.094 |
| X-12 | 3.142 | +0.994 |
| G-all | 3.227 | -0.118 |
| Manual heuristic | 3.917 | +0.919 |
| Climatology | 3.954 | -1.075 |

**Planned contrasts** (treatment minus reference; negative favours the treatment; 99% intervals
resample whole months; labels follow the pre-registered rule):

| Contrast | Pooled 12 origins | Origins below / above / tied | Label |
|---|---|---|---|
| C1: G-all minus G1 | +0.255 [-0.003, +0.463] | 3 / 8 / 1 | unstable |
| C2: G-all+D1 minus G-all | -0.451 [-0.681, -0.232] | 10 / 0 / 2 | stable |
| C3: G-all+D1 minus G1 | -0.196 [-0.440, +0.037] | 6 / 3 / 3 | unstable |
| C4: G1 minus bar | -0.052 [-0.267, +0.148] | 6 / 6 / 0 | seasonally stable |
| C4b: G-all+D1 minus bar | -0.249 [-0.579, +0.048] | 5 / 4 / 3 | unstable |
| C5 (score): S1 minus S0 on G-all+D1 | +0.050 [+0.031, +0.069] | 0 / 11 / 1 | worse |

- **C1 rests on D10.** Per substation, G-all minus G1 is +1.139 at D10, +0.275 at D16, +0.087 at D11,
  -0.048 at D5, and -0.187 at D14. Without D10 the pooled value is +0.031 (99% interval -0.114 to
  +0.174); without D16 it is +0.250. Experiment 104's audit blamed D16, and this experiment does not
  reproduce that.
- **C2 holds in every origin year.** Pre-tuning (2023-24) -0.813, in-tuning (2024-25) -0.243, and
  post-tuning (2025-26) -0.295 (each 95%, all excluding zero); on the 8 out-of-tuning origins -0.556
  [-0.792, -0.326]. Winter rows (November to March) -0.560 and other rows -0.377. Leaving out D5, D10,
  or D16 turns the label from stable to unstable; leaving out D11 or D14 keeps it stable.
- **C3 depends on the season.** Winter-quarter origins -0.461 [-0.736, -0.184], summer-quarter
  origins +0.058 [-0.129, +0.239]; rows from November to March -0.528 and from April to October
  +0.031. The label is unstable and the pooled interval spans zero.
- **C4's lead over the bar mostly vanishes.** G1 beat X-12-nodoy by 0.052 points (a tie), against
  0.170 [0.011, 0.334] over X-12 with the day-of-year column (E7, exploratory; experiment 104 found
  0.25 on one year). Winter-quarter origins favour G1 (-0.279) and summer-quarter origins the bar
  (+0.166). G-all+D1 minus the bar is -0.249, with an interval that spans zero.
- **Seasonal labels:** C1 and C3 are not seasonally stable under the rule (C3 has 5 of 6
  winter-quarter origins below zero but its summer-quarter origins are mixed); C4 is, with G1 ahead
  in winter and the bar ahead in summer.
- **The 2025-26 origins reproduce Part 2:** C1 +0.279 [+0.096, +0.432] (Part 2 +0.25), C2 -0.295
  [-0.520, -0.085] (-0.29), C3 -0.016 [-0.345, +0.278] (-0.04).
- **D10 with its two jump windows masked** (training and scoring): C2 -0.512 [-0.728, -0.306] with
  11 of 12 origins below zero, C1 +0.274 [+0.059, +0.474], C3 -0.238 [-0.480, +0.000], and C4 -0.034.
  No label changes.

**Tail model (C5) fails.** S1's coverage is 89.7% at q95 and 95.4% at q99, further from nominal than
S0's 91.8% and 97.3%, and its score is worse in 11 of 12 origins. Four of the pass rule's four
conditions fail (pooled coverage within 1 point, 4 of 5 substations within tolerance, lower tails no
worse, and the score's upper bound at most +0.03). S2 is worse still (T1: +0.489 [+0.335, +0.660]).
From 2024-07-01 on, S1's gaps are -6.3 points at q95 and -4.3 at q99.

| Arm and spread | q01 | q05 | q95 | q99 | Pinball at q01, q05, q95, q99 |
|---|---|---|---|---|---|
| G-all+D1, S0 | 3.9 | 10.0 | 91.8 | 97.3 | 0.232, 0.729, 0.732, 0.220 |
| G-all+D1, S1 | 5.4 | 11.6 | 89.7 | 95.4 | 0.258, 0.756, 0.786, 0.280 |
| G-all+D1, S2 | 9.3 | 15.8 | 84.0 | 91.5 | 0.403, 0.926, 1.058, 0.438 |
| G1, S0 | 2.2 | 6.1 | 85.8 | 94.2 | 0.189, 0.668, 0.920, 0.327 |
| G-all, S0 | 6.7 | 12.9 | 87.4 | 94.7 | 0.345, 0.917, 0.873, 0.288 |
| X-12-nodoy | 1.3 | 11.1 | 82.5 | 97.8 | 0.243, 0.732, 1.013, 0.339 |
| Climatology | 2.0 | 6.3 | 94.6 | 98.0 | 0.226, 0.864, 0.977, 0.265 |
| Manual heuristic | 9.0 | 11.0 | 80.8 | 83.6 | 0.417, 0.893, 1.223, 0.754 |

Cells are the percentage of outcomes at or below the quantile (nominal 1, 5, 95, and 99). The
curtailment-cost saving is not computed: these are demand substations, and the metric scores
generators.

**Controls and replication.**

- **Calendar controls do not change nothing.** G1 without harmonics +0.027 [-0.090, +0.153]
  (tied at 2 of 12 origins); G-all+D1 without harmonics +0.389 [+0.258, +0.521] (tied at 1);
  X-12 minus X-12-nodoy +0.118 [+0.035, +0.213] (tied at 4).
- **Replication column:** fitted capacity shares C1's sign at 12 of 12 origins, C2's at 8 of 12, and
  C3's at 10 of 12. Pinning capacity did not lower G1's or G-all's score (2.972 against 2.907, and
  3.227 against 3.067 with fitted capacity), and left G-all+D1's unchanged (2.776 against 2.768).
- **Drift step 10,000 against 3,000** is -0.023 [-0.048, +0.003]. XGBoost's seed floor is 0.006.
- **X-12amq minus X-12** is -0.008 [-0.019, +0.001]: the solar offset adds nothing.

## Checks

- **Median forecast minus mean forecast, per site:** exactly 0.000 for every arm with S0 or S1,
  because the centred spread has a zero median, and for XGBoost, whose mean forecast is its median.
  S2 moves the median by -0.034 points. The check therefore cannot expose a shifted forecast here;
  the mean forecast error in the absolute-score table does: G1 misses low by +1.495 points and the
  bar by +0.911, while G-all+D1 is unbiased (-0.018).
- **Error of the mean forecast beside each score contrast** (pooled MAE, same level): C1 +0.351
  [+0.007, +0.644]; C2 -0.644 [-0.959, -0.334]; C3 -0.293 [-0.675, +0.088]; **C4 +0.093 [-0.209,
  +0.385], which has the opposite sign to C4's score (-0.052)**; C4b -0.201 [-0.687, +0.242]; C5 0.
  No origin has opposite sides for the score and the error of the mean.
- **Leaving one site out:** changes C1's counts (without D14: 2 below, 10 above; without D10: 3, 5,
  4 tied) and C2's label (unstable without D5, D10, or D16); see Figure 3 and `report.txt`.
- **Blocking checks, all passed** (`checks_report.txt`): Part 2 reproduces (0.0 difference over
  22,080 rows for each `-fc` arm), no lookahead (every row in its window, no D1 knot after its cut,
  no pool row after its origin), the pool rule (60 of 60 fitted tables unchanged when data after
  the origin change, and S0 moves when the pool changes), and the pinned capacity source per cut
  (22 growth-prior, 6 early-vintage, 59 vintage).
- **Audit triggers fired: 1, 2, 3, 4, and 5; not 6 or 7.**
    1. Controls fail: the three calendar controls and replication of C2.
    2. Contradictions: C1's driver is D10, not D16 (Part 2's audit); C2's pooled -0.451 is larger
       than predicted (-0.05 to -0.20) and than Part 2's -0.29; experiment 104's tuning-year
       replicate found drift +0.06, but here the 2024-25 origins give -0.243 (quarterly refits keep
       the drift level current, which one frozen year does not; not separately tested).
    3. C4's score and mean-error contrasts differ in sign.
    4. Leaving one site out flips C2's label and C1's counts.
    5. G1 beats G-all (C1), a nested arm, and the calendar controls move the score.
  An Opus audit is requested before the prose review; its findings will be triaged here.

## Pre-registered predictions and what happened

- **C1** not stable, 3 or more ties, interval spanning zero, D16 out moves it by 0.10: unstable and
  spanning zero (barely), but 1 tie, and D16 out moves it 0.005. **Failed in part.**
- **C2** negative at 7 to 11 origins, most in winter, pooled -0.05 to -0.20: 10 origins, winter larger
  (-0.592 against -0.315), pooled -0.451. **Held except the size.**
- **C3** negative at 4 or more of 6 winter-quarter origins, pooled -0.05 to -0.20: 5 of 6 negative,
  pooled -0.196. **Held.**
- **C4** negative at 8 or more of 12, pooled -0.10 to -0.30, mean error within 0.10: 6 of 12, -0.052,
  +0.093. **Failed in part.**
- **C5** S0 coverage 89 to 93% at q95 and 95 to 98% at q99, S1 passes: S0 91.8% and 97.3%, S1 fails.
- **Controls** change nothing, replication shares sign at 10 or more, step 10,000 ties, seed floor
  under 0.03: only the seed floor held.
- **Run time:** the convex grid took 90 seconds and all fits about 15 minutes.

## Pre-registered decision rule

- **Verdict:** `supported` needed both C1 and C2 stable; `partly supported` applies when exactly one
  is stable (here C2). Labels follow the rule on the plan page: stable means 10 of 12 origins on one
  side outside the tie band, 3 of 4 in each origin year, and a 99% interval excluding zero. The
  Cochran Q test marks every contrast "sign varies" because the per-origin standard errors count
  only within-window noise, so it did not decide any label here.
- **Tie bands** are one week-block standard error per origin, written to `tie_bands.parquet` before
  any sign table (hash in `decisions.json`).

## Caveats

- **Past weather** (ERA5 and CAMS) at forecast time is a ceiling.
- **Penalties:** Part 2's penalties were tuned on 2024-07 to 2025-06, so the 2023-24 origins have a
  lookahead and the 2024-25 origins are in-sample; only 4 origins are strictly causal.
- **Burn-in and the first vintage:** the first four cuts use a hindsight capacity level, and the
  2023-07-01 cut uses a vintage dated 37 days later.
- **Spread pools** describe the site 9 to 12 months before the origin, so a drifting level can bias
  them.
- **Hindsight switching masks** apply to training and scoring. Refits are quarterly, not weekly.
- **The harmonic controls** zero the harmonic design columns inside the experiment's script rather
  than through the framework module.
- **Reference rows:** the manual heuristic counts an analogue only if it ended before the cut.

## What to try next

- **Quote C2 as stable only with its leave-one-out caveat,** and give Part 4 the finding that
  quarterly refits restore the drift gain that one frozen year lost.
- **Replace the 12-month G1 against all-years contrast** with a per-season one, since the sign
  follows the winter and summer split.
- **For tails,** try a spread that widens with the age of the fit or with the weather forecast's
  error, because conditioning on the predicted level (S1) and a conformal shift (S2) both lowered
  coverage.

The code is in `ml-experiments/00107-differentiable-physics-rolling-origin-history-and-tails/`; the
printed report is `report.txt`, the D10-masked variant is `report_d10_masked.txt`, and the bar
choice and tie-band hash are `decisions.json`. To reproduce: run `e00107_gam.py`, `e00107_xgb.py`
(both with `--variant base`, then `d10_masked`), `e00107_decisions.py`, `e00107_assemble.py`
(`--variant base` then `d10_masked`), `e00107_references.py`, `e00107_checks.py`, `e00107_score.py`
(both variants), and `e00107_charts.py`.
