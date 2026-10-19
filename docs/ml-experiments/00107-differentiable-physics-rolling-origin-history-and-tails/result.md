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
status: planned
size: >-
  medium. Stored: private files only. Serving path and degradation rules: not touched. Designs:
  more than one for the spread pool and tail model, one pre-registered. Callers: own scripts only.
hypothesis: >-
  Refitted quarterly from 2023-07 to 2026-04 and scored on the next 3 months, the convex model's two
  history contrasts (all years against 12 months, drift against none) keep one sign at almost every
  origin. A second claim, labelled separately as C5 and not part of the verdict: a level-dependent
  spread brings its 95% and 99% quantiles within 1 point of nominal coverage without worsening the
  quantile score.
prediction: see "Predictions"
decision_rule: see "Decision rule"
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
verdict: planned
production_status: not-for-production
take_home_message: >-
  Planned: written after the results.
date: 2026-10-19
pr: 0
advice: []
code: >-
  ml-experiments/00107-differentiable-physics-rolling-origin-history-and-tails/
---

**Differentiable physics: rolling-origin replication of the history contrasts, and tail calibration
of the convex model.**

**This page is the plan, committed before any code runs.** It replaces experiment 104's single scored
year with 12 quarterly origins, and calibrates the convex model's tails on the same fits. No
substation cards: D-labels only, and no result exists yet.

## What N1+N2 asks, and why

**Experiment 104 cannot say whether more history helps, because two of its three history contrasts
reverse sign between its scored year and its tuning year** (second audit, error of the mean
forecast: G-all minus G1 −0.25 against +0.31; drift +0.06 against −0.38). Twelve origins in three
years can separate a stable effect from an origin-dependent one. The origins are not independent
(adjacent fits share most of their training; D16's rising demand persists), so no sign-test p-value
is quoted; the rule asks for agreement within each year and a pooled interval instead.

**The origin-by-origin sign count is the primary read; the pooled 36-month score is secondary**,
because winter and summer pull opposite ways (Part 2's C1 −0.40 in November to March, +0.19 in
April to October), so a pooled estimate near zero can hide a gain at every winter origin.

**N2 asks a separate question on the same fits.** Part 2's centred spread left G-all+D1 covering
89.5% at q95 and 96.2% at q99, with opposite errors between substations hidden in the pooled
figure. Nothing Part 4 tunes is tuned here; no steps arm, no Kalman filter arm.

## Origins, sites, and rows

**Twelve quarterly origins, each with its own cut and its own fit, scored on the 3 months after
the cut:**

| Origins (the cut) | Scored window | Role |
|---|---|---|
| 2022-07-01, 2022-10-01, 2023-01-01, 2023-04-01 | 2022-07 to 2023-06 | burn-in: the first scored origins' spread pool, the XGBoost bar choice, and the XGBoost seed floor; never scored |
| 2023-07-01, 2023-10-01, …, 2026-04-01 (12) | the 3 months after each cut, 2023-07-01 to 2026-06-30 | scored |

- **Two season splits, named apart in every figure:** an origin's *quarter type* (six
  winter-quarter origins score October to March, six summer-quarter origins April to September),
  and the *row season* (November to March against April to October, as in Parts 2 and 4).
- **Origin years:** 2023-24 (origins 2023-07-01 to 2024-04-01), 2024-25, and 2025-26, four each.
- **Causality of Part 2's penalties, by origin year.** Part 2 tuned its penalties on 2024-07 to
  2025-06. So the 2023-24 origins use penalties chosen on later data (*pre-tuning*, a hyperparameter
  lookahead), the 2024-25 origins are in-sample (*in-tuning*), and only the 2025-26 origins are
  strictly causal (*post-tuning*). **The headline table gives the 4 post-tuning origins and the 8
  out-of-tuning origins (pre-tuning plus post-tuning), each labelled with its causality;** the
  verdict's sign count uses all 12 and is repeated over the 8 out-of-tuning origins.
- **Decision sites:** D5, D10, D11, D14, and D16. **Descriptive:** D9 from the 2025-10-01 origin
  (12 months of training from 2023-10-01, then 12 months of pool), and D18 with its `growth_prior`
  capacity flagged at every cut and the December 2024 to early February 2025 rows cut
  applied in training at every later cut.
- **Rows** as Part 2: experiment 103's cut dates, `default_rows == True`, the lockdown window
  2020-03-23 to 2021-07-19 out, the 66 hand-filtered transfers masked (24-hour padding) in training
  and scoring, and the all-rows score beside the masked headline. Nothing scores a valid time on or
  after 2026-07-01. The arms have no power lags and use past weather, so every scored half-hour
  counts once; `lead_hours` is the time since the cut.
- **Predictions written:** `time_series_id`, `valid_time`, `power_fcst_init_time` (the cut),
  `origin`, all 11 levels (`q01`, `q05` to `q95`, `q99`), `mean`, `actual`, `lead_hours`, `arm`,
  `seed`, and `spread`.

## Arms

**Every fit is frozen from its cut to the next origin, one seed, on the CPU,** with Part 2's
penalties (`work/tuned.json`: ridge 0.1, drift step 3,000), the `free` final knot, and D1 knots
counted back from each cut. **Capacity is pinned in every primary convex arm** by
`studies.dp_pinned_capacity` at each cut. The 2023-07-01 cut uses a vintage 37 days later
(`early_vintage`, disclosed). The burn-in cuts use the growth prior, whose level is hindsight;
each burn-in site's growth-prior capacity is printed beside the 2023-08 vintage.

| Arm | Model | Window | Notes |
|---|---|---|---|
| G1 | convex, pinned capacity | 12 months | |
| G-all | convex, pinned capacity | all years | |
| G-all+D1 | convex, pinned capacity, D1 drift | all years | the arm N2 calibrates |
| G1-fc, G-all-fc, G-all+D1-fc | convex, fitted capacity | as above | Part 2's configuration; replication column |
| G-all+D1-s10k | as G-all+D1, drift step 10,000 | all years | exploratory: Part 2's step sat at its grid edge |
| G1-noharm, G-all+D1-noharm | without the two day-of-year harmonics | as base | calendar controls: on 12 months the harmonics can carry a within-year trend into the next quarter |
| X-12 | lagless XGBoost, default start | 12 months | experiment 104's `x_12` recipe |
| X-12-nodoy | X-12 without the day-of-year column | 12 months | calendar-memorisation control and bar candidate |
| X-12amq | quantile start plus the solar offset | 12 months | experiment 104's `x_12amq`; from the 2023-10-01 origin only (below) |
| climatology, `manual_heuristic` | references | | refitted per origin where they fit anything |

**The XGBoost recipe is `e00104_xgb.py`'s,** including its `nthread: 1`: experiment 87's Primary
hyperparameters without the power lags, 500 rounds, 11 quantiles, one thread per fit.

**The XGBoost bar for C4 is whichever of X-12 and X-12-nodoy scores better over the four burn-in
origins,** recorded in `decisions.json` before any scored window is read. If X-12-nodoy wins, the
X-12 minus X-12-nodoy control is reported unchanged and the write-up says C4's bar has no
day-of-year column.

**X-12amq cannot be built before the 2023-10-01 cut:** its offset weight `w` comes from a
median-only replica trained on the 12 months ending 12 months before the cut, and before 2023-10-01
that window overlaps the lockdown window. It runs at the 11 origins from 2023-10-01, enters no bar
choice and no planned contrast, and reports `w` per site and origin.

**Robustness variants, every planned contrast:** leave one decision site out; the three causality
groups; the 8 origins from 2024-07-01 (pools free of burn-in residuals); all rows; and:

- **Without D16**, whose rising demand drove most of G-all's loss in Part 2 (G-all minus G1 +0.03
  without D16). D16's mean forecast error per origin is printed for G1 and G-all.
- **D10 with its two unlabelled jumps masked:** 2024-11-08 to 2024-12-01 and 2025-11-17 to
  2025-12-14 dropped from training (every arm refitted at D10 at the cuts after each window) and
  from scoring. Both windows fall inside scored windows (origins 2024-10-01 and 2025-10-01).

## The spread (N1's default, and N2's baseline)

**The spread pool for origin c is the arm's own out-of-sample residuals over the 12 months before
c**, the scored windows of the four origins before it. No residual comes from a fit that saw its
target. Each forecast in the pool was made 0 to 3 months after its own cut, but the calendar-quarter
group that serves scored quarter Q is that quarter one year earlier, so it describes the site's level
9 to 12 months before c. A drifting level (D16) can therefore still bias the spread; S0's coverage is
reported by origin year so that shift is visible.

- **S0, the baseline:** Part 2's centred spread, `ResidualSpread.fit(centre=True)` per hour band
  and calendar quarter, fitted on the pool. Every N1 contrast uses S0.
- **S1, the pre-registered alternative tail model:** residual quantiles per hour band × calendar
  quarter × tercile of the arm's predicted mean, centred on each group's median. **The tercile cut
  points come from the pool's predicted means only;** each scored row is assigned by those cut
  points. A group under `MIN_SPREAD_GROUP_ROWS` (200) falls back to its band × quarter group. Each
  group's size, its q99 exceedance count in the pool, and the fallback count are printed: groups
  hold roughly 240 to 365 rows per site, so q99 rests on 2 to 4 exceedances.
- **S2, exploratory:** S0 plus a split-conformal shift per level (q50 excluded) and hour band that
  makes the pool's coverage nominal, keeping S0's quarter shape.

**S1 and S2 are fitted for G-all+D1 and G1;** XGBoost's quantiles are its own and get no spread.

## Planned contrasts

**Five planned contrasts, read in this order of weight:** the sign at each origin, by quarter type
and origin year; the pooled 36-month estimate with a 99% interval (Bonferroni over five,
`studies.bootstrap`, whole months, paired); the row-season split. Each carries site counts,
absolute scores, the error of the mean (median for XGBoost) forecast, and the robustness rows.

- **C1, old years without drift:** G-all minus G1.
- **C2, drift on all years:** G-all+D1 minus G-all.
- **C3, does more history help a model with drift (Part 2's C1):** G-all+D1 minus G1.
- **C4, the convex model's lead over lagless XGBoost at the same window:** G1 minus the XGBoost
  bar. G-all+D1 minus the bar is reported beside it.
- **C5, the tail model (N2):** S1 minus S0 on G-all+D1, on coverage at all four tail levels and on
  the quantile score.

**The tie band is each origin's own sampling noise.** A week-block bootstrap inside each scored
window (13 one-week blocks, 1,000 draws, paired, five decision sites) gives each contrast's
per-origin standard error and 95% interval; an estimate within one standard error of zero is a
**tie**. The script writes `tie_bands.parquet` and its hash to `decisions.json` before computing any
sign table. For C4 the band is at least twice the XGBoost seed floor (X-12 refitted with a second
seed at the burn-in origins). Part 2's interval implies a quarterly standard error near 0.18
points. **A calendar control "changes nothing"** if it is a tie at 10 or more of 12 origins and its
pooled 95% interval spans zero.

**Exploratory at 95%,** grouped:

- **Replication and controls:** the `-fc` arms' C1 to C3; G-all+D1-s10k; the 2025-26 origins
  against Part 2; the three calendar controls; X-12amq minus X-12 over its 11 origins.
- **Tails:** S2 minus S0; S1 and S2 on G1; coverage by origin year; per-site coverage and pinball
  at the four tail levels for every arm and spread; q95 coverage by calendar month.
- **Staleness and descriptive rows:** score by month since the cut; D9; D18; bank holidays, the
  Christmas run, days below 0 °C; climatology and `manual_heuristic`.

## Known-answer checks (blocking, before any scored window is read)

1. **Part 2 reproduces.** At the 2025-07-01 cut, the three `-fc` arms with Part 2's replica spread
   match experiment 104's stored `*_centred` predictions for July to September 2025 to 1e-6 points.
2. **No lookahead, per cut:** `assert_no_lookahead` on every fit; no D1 knot after its cut; no
   residual in a spread pool from a valid time at or after the origin it serves.
3. **The pool rule,** a unit test: changing any actual or any predicted mean at or after origin c
   leaves every quantile issued at c unchanged, for S0, S1 (its tercile cut points included), and
   S2.
4. **Pinned capacity source per cut and site** printed (`vintage`, `early_vintage`,
   `growth_prior`), with the held value beside the `-fc` fit's capacity.

## Predictions

Written before any fit; each can fail, and none overrides the decision rule.

- **C1:** not stable, ties at 3 or more origins, pooled interval spanning zero; dropping D16 moves
  the pooled estimate towards zero by at least 0.10.
- **C2:** negative at 7 to 11 of 12 origins, most at winter-quarter origins; pooled −0.05 to −0.20,
  smaller than Part 2's −0.29, since a quarterly refit leaves drift less to correct.
- **C3:** negative at 4 or more of 6 winter-quarter origins, mixed in summer; pooled −0.05 to −0.20.
- **C4:** negative at 8 or more of 12 origins, pooled −0.10 to −0.30; within ±0.10 on the error of
  the mean forecast, since Part 2 found the convex model's lead mostly in the spread.
- **C5:** S0's pooled coverage on G-all+D1 is 89 to 93% at q95 and 95 to 98% at q99; S1 meets the
  pass rule below.
- **Controls and replication:** each calendar control changes nothing (rule above); the `-fc` arms
  share the pinned arms' sign at 10 or more of 12 origins for C1 to C3; G-all+D1-s10k is a tie with
  G-all+D1 at 10 or more origins; the XGBoost seed floor is under 0.03 points.
- **Run time:** the convex grid under 45 minutes of wall clock; all compute under 5 hours.

## Decision rule

**The verdict is on whether C1 and C2 keep one sign across origins.** C3 (the headline answer to
"does more history help"), C4, and C5 are labelled in the take-home, not in `verdict`.

**Labels for one contrast:**

- **Stable:** at least 10 of 12 origins on one side outside the tie band, at least 3 of 4 on that
  side in each origin year, and the pooled 99% interval excluding zero on that side.
- **Seasonally stable:** not stable; at least 5 of 6 winter-quarter origins on one side and at
  least 5 of 6 summer-quarter origins on the other; and each quarter type's pooled 95% interval
  excluding zero in its own direction.
- **Too small to sign:** 5 or more ties.
- **Unstable:** none of the above.
- **Sign varies (evidence):** at least 2 origins on each side whose per-origin 95% interval
  excludes zero, or a between-origin heterogeneity test (Cochran's Q on the 12 per-origin estimates
  with their bootstrap variances) at p < 0.01.

**Verdict:**

- **`supported`:** C1 and C2 are both stable, and each keeps at least 9 of 12 origins on its side
  whichever decision site is left out.
- **`partly supported`,** in three cases: (a) exactly one of C1 and C2 is stable; (b) neither is
  stable and at least one is seasonally stable; (c) both are stable but leaving one site out drops
  one of them below 9 of 12.
- **`refuted`:** C1 and C2 are both unstable and both show "sign varies". The finding is then that
  the sign of a history contrast depends on the origin, so no single-year history contrast in this
  series is evidence either way.
- **`not supported`:** C1 and C2 are each unstable or too small to sign, neither shows "sign
  varies", and both pooled 99% intervals lie inside ±0.10: the history contrasts are near zero.
- **`inconclusive`:** anything else, or an unfixed blocking check.

**The 8 out-of-tuning origins.** The write-up says whether each label survives on them (stable
needs at least 7 of 8 on its side there); a label that does not survive is reported as resting on
the in-tuning origins.

**C3's label, for the take-home:** "more history helps" if C3 is stable and negative; "helps in
winter only" if it is seasonally stable with negative winter-quarter origins; "level" if its pooled
interval lies inside ±0.10; "no stable answer" otherwise. If C1 or C3 is seasonally stable, the
write-up advises Part 4 to read its own C1 per season first; this plan does not change Part 4's
decision rule.

**C4's label:** "the lead holds" if C4 is stable and negative; "the lead is in the spread" if, in
addition, the same contrast on the error of the mean forecast spans zero.

**C5 passes only if all of these hold:**

- S1's pooled coverage is within 1 point of nominal at q95 and at q99, and the month-block 95%
  interval of each coverage lies within ±1.5 points of nominal;
- at least 4 of 5 decision sites are within 2 points at q95 and within 1.5 points at q99;
- q01 and q05 coverage are each no further from nominal than under S0;
- the upper end of the 99% interval of S1 minus S0 on the quantile score is at most +0.03 points.

C5 is also reported from 2024-07-01 on. A pass sends S1 to Part 6 ("Decisions"); S2 meeting the
rule is an exploratory pass only.

## Audit triggers (the `ml-experiment` skill's seven, checked on these results)

1. **A control fails:** the known-answer checks, the calendar controls, the replication column.
2. **A contradiction:** with experiment 104 (the 2025-26 origins), its second audit (the 2024-25
   origins), or Part 2's tails.
3. **Quantile score and error of the mean disagree in sign:** every planned contrast.
4. **Leaving one site out flips a planned contrast's sign count or label.**
5. **A nested arm wins** (G1 beating G-all pooled), or a calendar control does not change nothing.
6. **Interval width:** the 2025-26 origins' pooled interval against experiment 104's.
7. **A later experiment depends on it:** fires if C5 passes; the audit runs before Part 6 uses S1.

If any trigger fires, a fresh Opus audit runs before the prose review.

## Run order, compute, and time

**Check `uptime` first.** At most 8 cores, no GPU; the 2 reserved cores stay free.

1. **Scripts and checks** (about 3 hours of agent time): copy the `e00104_*.py` patterns, add the
   origin loop, the pool builder, S1, S2, the week-block tie bands, and the pool-rule test; run
   checks 1 to 4.
2. **Convex grid:** 9 arms × 16 cuts × up to 7 sites plus the D10 refits, about 2.5 CPU-hours at
   3 s per site-year; 8 workers with one torch thread each, about 20 to 30 minutes of wall clock.
3. **XGBoost** at `nthread: 1`, 8 workers, after the convex grid: X-12 and X-12-nodoy at 16 cuts
   and 5 sites plus D9 (about 166 fits), X-12amq at 11 origins (55, plus 55 median-only replicas at
   about a tenth of a fit each), the seed check (20), and the D10 refits (12): about 255 full fits.
   At about 2 minutes per single-thread fit (experiment 104's timestamps) that is about 64 minutes,
   or 1.5 hours allowing for memory contention.
4. **Bar choice, tie bands, spreads, and scoring:** under 30 minutes.
5. **Write-up:** one Opus code review (leakage first: each cut's window, the pool rule, S1's cut
   points), a numbers review, any audit, and the Sonnet prose review, about 5 hours of agent time.

**The compute budget is 5 hours** (about 2.5 expected). **Cut in this order if needed:** D18; the
D10-masked XGBoost refits; X-12amq; G-all+D1-s10k; the seed check (C4's band then uses the
bootstrap alone, disclosed). Decision arms, calendar controls, and spreads are never cut. About a
day from approval to a merge-ready PR.

## What the write-up must contain (beyond the standing rules of the differentiable-physics series)

The take-home says plainly whether more years help at most origins, in winter only, or at none;
whether the lead over lagless XGBoost holds; and whether S1 fixes the tails. No substation cards;
D-labels only.

- **Figures, headline first:** a dot per origin for C1 to C4 with its interval (quarter types in two
  colours, tie bands shaded, causality groups marked, pooled estimate at the right); every arm's
  absolute score per origin and site; leave-one-site-out sign counts; tail coverage per origin and
  site for S0 to S2; a winter and a summer week at three sites (actual, G1, G-all+D1, bar).
- **Bullets:** arms and rows; absolute scores; C1 to C5 with intervals, sign counts, and labels by
  causality group; the tail table (coverage and pinball at q01, q05, q95, q99); the replication
  column; the calendar controls; caveats; what to try next.
- **Checks:** median minus mean forecast per site and arm; the error of the mean forecast beside
  each quantile-score contrast; leaving one site out; which triggers fired.
- **Caveats:** past weather; the penalties' causality groups; the burn-in's hindsight capacity
  (first four pools, the bar choice) and the 37-day vintage leak at 2023-07-01; the pool's quarter
  groups lag the level by 9 to 12 months; hindsight switching masks; quarterly, not weekly, refits.
