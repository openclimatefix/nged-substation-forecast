---
id: "00034"
slug: retraining-cadence-and-cheap-proxies-demand
status: planned
size: complex
hypothesis: >-
  For the 16 Primary substation demand series, retraining the single-stage ENS-trained quantile
  model N1 monthly (half-life 0.5 years, no 14-day error correction) lowers the quantile score below
  the best-guess stack (frozen N1 with the 14-day correction and spread factor 1.15) by more than
  0.10 points of capacity; the gain from retraining more often flattens between monthly and weekly
  retraining; and the 14-day correction stops helping once the model is retrained. A second,
  separately reported claim is that at least one cheap proxy for monthly retraining ranks arms the
  way full monthly retraining ranks them.
prediction: >-
  Monthly retraining without the correction scores 0.20 points below the stack (plausible -0.10 to
  -0.30; interim figures from experiment 33 are 5.12 against 5.32, one seed). Adding the correction
  to the monthly model raises the score by 0.05 to 0.15 (the interim figure is +0.12). Weekly
  retraining scores 0.03 to 0.10 below monthly, and daily retraining scores within 0.03 of weekly.
  Moving from three refits a year to monthly lowers the score by 0.05 to 0.15. Under monthly
  retraining a half-life of 0.5 years scores 0.03 to 0.15 below a half-life of 2 years. For BSP and
  GSP demand, monthly retraining scores 0.2 to 0.8 below the frozen model. Of the cheap proxies,
  the three-anchor refit and the rolling 12-month window pass the rank-agreement test, and
  warm-start boosting fails it for the contrast that involves the correction.
decision_rule: >-
  Eight planned contrasts on the 16 Primary substations, quantile score over all leads, read at
  99.375% (Bonferroni across eight), whole validation months and the seeds resampled. Arms without
  the correction are scored with the spread factor (`_s`) and arms with it with both (`_cs`), so
  that every contrast differs in one factor. C1 (primary) is monthly (`_s`) minus the best-guess
  stack (`_cs`, three seeds). C2 is monthly `_cs` minus monthly `_s`. C3 is weekly minus monthly
  (`_s`, seed 0). C4 is monthly minus frozen (`_s`, the 2025-07-01 refit used all year, seed 0).
  C5 is monthly minus three refits a year (`_s`, seed 0). C6 is monthly with a half-life of 0.5
  years minus monthly with a half-life of 2 years (`_s`, seed 0). C7 is weekly `_cs` minus weekly
  `_s` (seed 0). C8 is quarterly minus monthly (`_s`, seed 0). Daily retraining (three months) is
  descriptive and has no interval-based claim. The cadence claim is supported when C1 lies wholly
  below zero. The threshold is zero because monthly retraining adds a scheduled job and lets the
  service drop the correction step, so the pipeline has the same number of steps; it is refuted
  when C1 lies wholly above zero, and otherwise inconclusive. The 99.375% interval on C1 is
  expected to be about plus or minus 0.3 points wide (experiment 19 C2 was plus or minus 0.87 at
  99.44% for a larger structural change), so an inconclusive C1 with a negative point estimate is
  a real possibility; the page then reports the 95% interval too. C1 is not an independent test,
  because experiment 33's seed 0 on these rows suggested it; seeds 1 and 2 add seed noise only.
  The correction stops helping when C2 lies wholly above zero and is still helping when C2 lies
  wholly below zero (experiment 17's arm K, on ERA5 weather, found it helping). The gain flattens
  at the slowest cadence whose point estimate is within 0.05 of weekly's (exploratory). A proxy
  passes the rank-agreement test when, for each of R1 to R3, its difference has the same sign as
  full monthly retraining's and lies within a tolerance of it; the sign is required only where the
  full-monthly difference exceeds 0.02 in size. The tolerance is 0.05 unless the seed-to-seed
  spread of the full-monthly difference on R1 (seed 1 against seed 0) is larger, in which case it
  is that spread. A proxy that fails any contrast is dropped. No proxy that uses data after the
  forecast origin (P-interp) can become the protocol, whatever it scores.
benchmark: >-
  The best-guess stack (N1 with the 14-day correction and spread factor 1.15, experiment 19's
  `stack_minus_h`, rebuilt in experiment 33 as `n1_cs`), frozen N1 with a half-life of 0.5 years
  (experiment 22's `ens`), `manual_heuristic`, and `climatology` (stored, 13).
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels), in points of capacity;
  MAE of the median, the exceedance rate of the 0.95 quantile, and the twCRPS at `historical_p99`
  are reported and decide nothing
advice: []
production_status: not-for-production
adopted_in: none
code: ml-experiments/00034-retraining-cadence-and-cheap-proxies-demand/
---

# Experiment 34: how often should the demand model be retrained, and which cheap proxy stands in for it?

**This page is the plan, committed before any code runs.** The results replace the sections below
when the experiment finishes.

Experiment 33's interim result is that N1 retrained monthly beats the 14-day error correction on
Primary demand (5.12 without the correction, 5.24 with it, 5.32 for the stack; one seed). This
experiment measures how the score changes along the retraining cadence, whether the correction
still helps, where more frequent retraining stops paying off in score, and what each cadence costs
in GPU hours. It then tests four cheap proxies for monthly retraining, so that later experiments
can use a monthly-retrained reference without a twelve-fold rise in fitting time. Units are points
of capacity (1 point is 1% of a series' effective capacity). The validation rows are the 73 runs of
`fold_a_every5d_v1`, one every 5 days from July 2025 to June 2026. No valid time on or after
2026-07-01 is scored.

**Weather at forecast time.** Every arm trains and predicts on ENS forecast weather (the control
member for training, the 51 members at prediction), so no arm reads weather that is unavailable at
forecast time. The numbers are deployable, unlike the ERA5-based ceilings of experiments 5 and 22.
Lagged power uses the observation at least 3.5 hours old.

## Arms

**The recipe is experiment 13's N1 with the settings of experiment 33's monthly arm.** One XGBoost
model per series, the 9-level quantile loss, 500 rounds, the experiment 6 columns plus
`lead_hours` and three origin lags, trained on the ENS control member of every daily run, and the
51 members pooled into nine quantiles as in experiment 19. A refit at time `t` uses rows whose
valid time is at least 3.5 hours before `t` and whose run began at or before `t`. Rows are
weighted by a half-life of 0.5 years anchored at `t`. A validation run is forecast by the latest
refit on or before its origin. The fitting code copies experiment 33's `e00033_fit_demand.py`
(an experiment may not import another).

| Cadence arm | Refit times (UTC) | Fits per series | Source |
|---|---|---|---|
| F: frozen (also "yearly": a refit on 2025-07-01 is the frozen model to within 3.5 hours) | trained to 2025-06-30 | 1 | the 2025-07-01 monthly refit used all year (free, paired with M); experiment 22 `ens` is a cross-check |
| H: half-yearly | 2025-07-01, 2026-01-01 | 2 | shares the monthly fits |
| T3: three a year (the proxy "P-anchor") | 2025-07-01, 2025-11-01, 2026-03-01 | 3 | shares the monthly fits |
| Q: quarterly | 2025-07-01, 10-01, 2026-01-01, 04-01 | 4 | shares the monthly fits |
| M: monthly | the first of each month, 12 refits | 12 | new fits, seeds 0 to 2; experiment 33's seed 0 checks them |
| W: weekly | every 7 days from 2025-07-01 | 52 | new fits, seed 0 |
| D: daily (descriptive) | each validation run's own origin, in October 2025, January 2026, and April 2026 only; its correction history outside those months comes from W | about 18 | new fits, seed 0 |
| M2: monthly, half-life 2 years | as M | 12 | new fits, seed 0 |

Each arm is scored without the correction, with the 14-day correction (`_c`), and with the
correction and the spread factor 1.15 (`_cs`), using experiment 33's code, which is experiment 19's.
The stack is the `_cs` form of frozen N1 with a half-life of 2 years, read from experiment 33's
`n1_cs` set.

**Yearly retraining cannot be told from frozen on this validation year.** A model refitted on
2025-07-01 and kept for a year is the frozen model. The experiment therefore reports the next
cadences down (half-yearly, three a year, quarterly) as the cheap end of the axis. A truly yearly
comparison needs several validation years, which only experiment 17's ERA5 setting has.

**Daily retraining is a refit at every validation origin.** The service runs one NWP cycle a day,
so a refit at the origin is the daily regime. Validation origins are 5 days apart, which makes the
cost 73 fits per series for the whole year; the plan runs three months (October 2025, January 2026,
April 2026, about 18 origins) to bound the cost at about 1 hour.

## Cheap proxies for full retraining

**The four proxies replace most fits with cheaper updates.** Each is run on N1 for Primary demand,
scored against full monthly, and then put through the rank-agreement test below.

- **P-anchor** is arm T3: full refits at three anchors a year, each used for four months. Cost: 3
  fits against 12.
- **P-interp** blends the predictions of the two anchors that bracket a run, linearly by date.
  **The later anchor was trained on data after the run's origin, so P-interp scores are not a
  forecast skill.** It is a screening proxy only, and no result quotes its score as one. The last
  four months use the final anchor alone. Cost: the same fits as P-anchor.
- **P-roll** refits monthly on the latest 12 months of rows, with 150 rounds and `eta` 0.15
  (three times the usual 0.05). Cost: about a fifth of monthly.
- **P-warm** starts from the frozen booster and adds `k` rounds (k = 20 or 50) fitted on the rows
  that arrived since the previous update, at each monthly or weekly update, chaining each update
  on the previous booster. Over one month of rows the recency weights span only 0.89 to 1.0, so
  the updates are effectively unweighted, and nothing makes the booster forget older rows. It is
  tested on R1 only, because R2 and R3 would need a frozen A7 booster and a frozen pooled
  booster. Each update costs a few seconds.
- **P-roll** changes two settings together, the 12-month window and the rounds and `eta`, so a
  failure cannot be traced to either. Rows older than a year carry at most a quarter of the
  weight, so most of the saving comes from the rounds.

**More proxies are admitted only if they are justified before they run.** Candidates are a
rolling 6-month window and a one-step transfer of the last monthly booster to a weekly update; the
plan leaves them out unless the first four all fail.

## Rank-agreement test

**A proxy is useful only if it ranks arms as full monthly retraining does.** The test re-runs four
earlier contrasts under full monthly retraining and under each proxy. Each contrast is the
difference in quantile score over all leads, on 16 Primary series, seed 0, with the correction and
the spread factor unless noted.

| Id | Earlier contrast | Arms that need monthly fits |
|---|---|---|
| R1 | the 14-day correction on versus off (experiment 19 C4, and this experiment's C2) | none: the correction is computed from the arms above |
| R2 | experiment 30: N1 plus the weekly-analogue mean and median (A7) minus N1, both with the correction | A7: a monthly arm with the 00030 lag columns |
| R3 | experiment 24 C1: one model per substation type with the series identifier minus one model per series | the pooled arm: 24 monthly fits across two type pools of 8 series, each fit on 8 times the rows, so the cost equals 192 per-series fits |
| R4 | experiment 32 C3: pooled model with recent-window statistics minus the identifier alone | conditional on the budget; its frozen contrast is about +0.02, below the sign floor, so it could rarely fail a proxy |

For each contrast the page reports the difference under the frozen model (the earlier
experiment's value, recomputed on the same rows), under full monthly, and under each proxy. The
frozen value shows what the proxy adds over doing nothing.

**R2's frozen contrast (-0.05) is as large as the tolerance and R4's is below the sign floor,
so in practice R1 and R3 are the contrasts that can fail a proxy.** The page says so. The
tolerance is calibrated on R1 by comparing full-monthly seed 1 with seed 0, a stand-in for a
proxy that is exactly right.

**Stop rule.** A proxy that fails any contrast of the test is dropped. If every proxy fails, full
monthly retraining becomes the reference only for decisive (release) experiments, and screening
experiments are told to say they ran on a frozen model. The weekly and daily arms are not run if
the point estimate of C1 on seed 0 is not below zero. (Whether weekly beats monthly does not
depend on monthly beating the stack, but the cost of weekly is justified only if retraining
helps at all.)

## Secondary parts

- **BSP and GSP demand (2 and 2 series).** Frozen (the 2025-07-01 refit used all year),
  half-yearly, quarterly, monthly,
  and weekly (seed 0 only), with the correction on and off. Intervals
  are wide on two series each and are reported as exploratory.
- **Solar and wind.** Planned only if all of the above finishes: monthly retraining of the
  ENS-trained N1 for generators against the frozen one. The ERA5-trained generator models of the
  best guess have no retraining arm here. No contrast is pre-registered for this part.

## Compute and the cost of cadence

N1 takes about 12 s per series per fit on the GPU (experiment 33's log: 140 s per series for 12
monthly refits). Counts below are for 16 Primary series.

| Item | Fits | GPU hours |
|---|---|---|
| M, seed 0 (serves F, H, T3, Q, P-interp too) | 192 | 0.65 |
| W, seed 0 | 832 | 2.7 |
| M2 (half-life 2 years), seed 0 | 192 | 0.65 |
| P-roll, P-warm (monthly and weekly) | 192 + updates | 0.6 |
| R2 and R3 arms, seed 0 | 192 + 192 | 1.3 |
| BSP and GSP: M, seed 0 | 48 | 0.15 |
| M seeds 1 and 2 | 384 | 1.3 |
| D, seed 0, three months (descriptive) | 288 | 0.9 |
| **Total** | | **about 8.3** |

**The total exceeds the 6-hour budget, so the plan fixes an order and a cut line.** Stage A (M seed
0, W, M2, P-roll, P-warm, the R2 and R3 arms; about 6 hours) answers the cadence questions and
the rank test. Stage B (M seeds 1 and 2, BSP and GSP) comes next, and stage C (D, then R4) last.
Runs stop between fits when the cumulative GPU wall time reaches 8 hours, and the page records
what ran. The manager is told of the overrun before it starts. One fit runs at a time on the GPU,
with the 12-core quota of slot A.
Scheduling cost for the live service: a scheduled monthly retrain job in Dagster, listed in
`docs/design-philosophy/design-principles.md` as an uncommitted cadence.

## Planned figures

- **Headline:** quantile score of the stack, frozen, and every cadence arm with intervals, at the
  top of the page.
- **Cadence axis:** score and GPU hours against the number of refits a year.
- **Correction on and off at each cadence,** and by lead band.
- **Proxy rank agreement:** each proxy's contrast against full monthly's, one panel per R-contrast.

## Advice followed and departed from

- **Followed:** use a half-life of 0.5 years under retraining and test it against 2 years (00019
  advice); retrain monthly, never freeze (00017); test the correction with monthly retraining in
  place (00019).
- **Departed from:** 00019 advised testing the correction on forecasts made daily, not every 5
  days. The row set is the shared `fold_a_every5d_v1`, so the correction is read from forecasts 5
  days apart. The page says so.

## Plan, written before any code ran

- **Leakage checks.** A refit at `t` sees only rows with valid time at least 3.5 hours before `t`
  and runs begun at or before `t`; power lags and the correction use observations at least 3.5 hours
  before the origin. P-interp's later anchor sees data after the origin, which is why it is never
  scored as a forecast. A reviewer checks the refit cutoffs in the code.
- **Reproduction checks.** The monthly refits at seed 0 must reproduce experiment 33's stored
  `monthly_seed0` set (the maximum difference over all quantiles is printed); a refit at the frozen
  anchor must match experiment 22's `ens` set within seed noise; the rebuilt stack must match
  experiment 19's `stack_minus_h` set.
- **Correction history spans model changes.** At each refit boundary the 14-day correction's
  history comes from the previous model. This lowers the correction's value at weekly cadence
  most, and the page says so.
- **Seeds.** The stack uses three seeds and M three if stage B runs. Every other arm uses seed 0
  and is contrasted with the seed-0 rows of its reference, so those intervals
  resample months only.

### Sizing

- **Stored data:** prediction sets under `data/ml-experiments/00034-.../predictions/`; no table.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Defensible designs:** more than one could test the cadence and the proxies (refit anchors,
  the warm-start length, the definition of "daily"), so the plan is reviewed.
- **Callers:** the scripts copy fitting, correction, and scoring helpers from experiments 13, 19,
  24, 30, and 33; the span of code is wide, so the size is complex.
