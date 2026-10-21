---
id: '00161'
slug: dp-bsp-top-down-against-bottom-up
part: BU
theme: differentiable-physics-network
also_relevant_to:
- hierarchy-and-pooling
- production-reference-and-best-guess
- calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00161-dp-bsp-top-down-against-bottom-up/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (how the weights are estimated, whether the sum covers the children that report
  apparent power, whether the bottom-up spread is refitted), and this plan pre-registers one of each
  and runs the alternatives as labelled variants. Callers: this folder's scripts, which copy the
  chain of experiments 111, 136, and 155 and call the reviewed `studies.*` modules without changing
  them. No new package code.
hypothesis: >-
  A bulk supply point (BSP) is forecast better bottom-up than top-down. The top-down forecast is the
  best-guess GAM-K chain (experiment 155's arm D) applied to the BSP's reading
  as a stand-alone series. The bottom-up forecast adds the arm D forecasts of the BSP's child Primary
  substations that report real power, each times a weight learned from earlier readings, to arm D's
  forecast of the remainder (the BSP's reading minus those children's weighted readings). The
  mechanism is that the child Primaries' own forecasts are better than the BSP's chain can infer
  from the BSP's sum, because each child has its own heating, solar, and calendar response and its
  own recent errors, and the sum is not exact (losses, unmetered load, children that report apparent
  power), so the remainder and the weights absorb the difference.
prediction: >-
  Written before any score is read. The bottom-up arm scores 2% to 8% below the top-down arm over
  leads 0.5 to 168 hours at BSP D13, and 2% to 8% below at BSP D17 (C1). The gain is largest at
  leads under 24 hours, where the children's recent errors help, and near zero beyond 72 hours (C4).
  The fixed-sum variant (weights of 1) scores 0% to 5% worse than the learned weights (C3), because
  experiment 124's closure audit fitted multipliers of 1.01 to 1.16 on the Primaries. The oracle
  (observed children in place of their forecasts) scores 15% to 40% below top-down, in line with
  experiment 36, where the oracle gained 3.4 to 5.6 points and the forecast children gained
  nothing. The explicit sum covers only the children that report real power, 42% of D13's effective
  capacity and 56% of D17's, so a gain much larger than the covered share predicts would be a
  warning sign.
decision_rule: >-
  The headline contrasts are C1 at D13 and C1 at D17: bottom-up minus top-down, in points of the
  BSP's effective capacity, over leads 0.5 to 168 hours in the scored year, on the shared rows. Each
  is read at the 97.5% level of the month-resampled paired bootstrap (12 calendar months, 20,000
  resamples; the level is Bonferroni-corrected for the two BSPs because `partly supported` needs
  only one). The threshold is a gain of 2% of the top-down score, computed from the point estimate
  of the top-down score at that BSP, because bottom-up adds a remainder chain per BSP and a weight
  table, and needs the children's forecasts at serving time. The hypothesis is `supported` when both
  BSPs' intervals lie wholly below minus the threshold, `partly supported` when exactly one does and
  the other interval does not lie wholly above zero, `mixed` when one does and the other lies wholly
  above zero, `small gain` when both lie wholly below zero and the `supported` condition fails,
  `refuted` when both lie wholly above zero, and `inconclusive` otherwise (for example one interval
  wholly below zero but not below the threshold, and the other straddling zero). The verdict is
  capped at `inconclusive` if the reproduction control or the oracle control fails. Contrasts C2 to
  C9 carry no verdict. The prediction range overlaps the threshold, so a gain of about 2% is close
  to a coin flip on the verdict.
benchmark: >-
  Top-down (arm D's chain on the BSP alone) as the reference arm; the fixed-sum bottom-up; the
  oracle; ClimatologyForecaster and manual_heuristic for absolute skill
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  the BSP's effective capacity, on rows where the BSP and its three covered children are outside the
  hand-filtered switching events, over leads 0.5 to 168 hours, then under 24 hours, 24 to 72 hours,
  and 72 to 168 hours; coverage and pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles; the
  mean absolute error of the mean forecast
code: ml-experiments/00161-dp-bsp-top-down-against-bottom-up/
---

# Experiment 161: is a bulk supply point forecast better from its children than on its own?

Plan only; the write-up follows the runs.

## Plan

**Question.** Experiment 36 summed the Primaries' forecasts and fed the sum to a BSP's XGBoost
model, and the sum added nothing. Experiment 124 showed that a BSP reading does not close exactly
over the Primaries beneath it. The maintainer's hierarchy aims treat the topology as a soft hint and
not a hard sum. This experiment asks the question again with the best-guess GAM-K chain
([experiment 155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md)'s arm
D) on both sides: is the BSP forecast better when it is built from its children's forecasts plus a
learned remainder than when the BSP is forecast as a stand-alone series?

**Scope: two BSPs (seven and nine children), and the children that report real power.** NGED's
hierarchy file places 16 Primaries under two BSPs, D13 (seven children) and D17 (nine children).
Four of D13's children (D2, D4, D19, D21) and four of D17's (D3, D8, D12, D22) report apparent power
in megavolt-amperes, which hides the sign of the flow in daylight (see [the data-processing
page](../data-processing.md), "Use a Primary metered in MVA for demand only on dark rows"). No
observed net-power remainder can be formed from such a child, and arm D has no forecast for it. The
MVA children stay inside the remainder. The covered children are the Primaries that report real
power and have a long history and an arm D forecast: D9, D10, and D11 under D13 (23.3 of 55.3 MW of
effective capacity), and D5, D14, and D16 under D17 (26.5 of 47.6 MW). D15 and D18 under D17 (the
two transformers of one Primary substation) report real power but have under 5 months of readings
before the scored year, so they stay in D17's remainder too. The scope is therefore a partial
bottom-up, and the write-up says so in its first paragraph. A bottom-up arm that also forecasts the
MVA children (demand terms only, on dark rows, as experiment 124's joint fit does) is the obvious
follow-up and is not tested here.

**Arms.** All arms use arm D's recipe: the ensemble-mean weather, T-all, the shared filter noise
(POOL), monthly refits, the winter night offset, recipe R, and arm A's spread table.

| Arm | Forecast of the BSP |
|---|---|
| TD (top-down) | Arm D's chain on the BSP's reading as a stand-alone series |
| BU (bottom-up) | The sum over covered children of weight times the child's arm D mean, plus arm D's mean for the remainder series; quantiles from a spread table fitted on the sum's own tuning-month residuals |
| BU1 (fixed-sum) | The same, with every weight 1 (reference) |
| BU-oracle (diagnostic) | The same as BU, with the children's observed readings in place of their forecasts; no verdict |

**The remainder series.** For each BSP the remainder is `r(t) = BSP(t) - sum_i w_i * y_i(t)`, where
`y_i` is the covered child's observed reading in MW. The chain forecasts `r` as it forecasts any
series, with the BSP's location for the weather and calibration, a PV capacity fitted and not pinned
(no capacity vintage exists for a BSP, and the remainder holds the unmetered and the MVA children's
embedded solar), the same filter, and arm D's two corrections applied to the remainder's own errors.
The BU forecast of the BSP is `sum_i w_i * mean_i + mean_r`, because the decomposition is exact in
the observed readings. BU1 runs its own remainder chain on `BSP - sum_i y_i`.

**A confound the design names.** The covered children's arm D forecasts use the pinned PV capacity
of [experiment 139](../00139-differentiable-physics-pv-as-of-capacity-in-gam-k/result.md). No
capacity vintage exists for a BSP, so TD and the remainder fit the PV capacity. Part of any BU gain
may therefore come from the children's pinned capacity and not from the bottom-up structure. A
pinned capacity is information that bottom-up legitimately has and top-down lacks, so C1 stays the
headline. Variant BU-fc replaces the children's forecasts with arm D chains that fit the capacity
(C8, BU-fc minus BU), which splits the gain into the two parts. Arm D also corrects the children and
the remainder separately, so a shared error is corrected twice in BU; this is part of the design.

**Feasibility steps.** (1) Build a feature cache that adds D13 and D17 to experiment 103's 16
Primaries, from experiment 103's `panel.parquet`, with experiment 103's rows rule (present and
unflagged, outside the fixed lockdown window and the hand-filtered switching events), and add each
remainder series as a further site that copies the BSP's weather and replaces the reading. (2) Build
the ensemble weather arrays and the weather calibration for the two BSP locations; the remainders
reuse them. (3) Copy experiment 136's T-all fit and prediction with a fitted-capacity option,
because `predict_tall` asserts a pinned capacity. This is code inside this folder only. (4)
Experiment 155's stored forecasts hold the runs that experiment 155 scored, which are not the
every-fifth-day stride runs; the TD and remainder chains are built on the same runs so that recipe R
sees the same error stream, and every score uses the runs the arms share.

**The weights.** Non-negative least squares of the BSP's reading on the covered children's readings,
with no intercept, on the half-hours in 2023-10-01 to 2024-03-31 where the BSP and the children are
present, unflagged, and outside the hand-filtered switching events. The window ends on the day
before the first run (2024-04-01) and starts at the first usable day of the youngest child (D9,
2023-10-01 at D13). The weights are fixed for the whole experiment, so no weight uses a reading
after any cut. The weights are 15 to 27 months old in the scored year, and the remainder series
before 2024-04 is in-sample for the weights; no scored row is affected. Variant BU-w2 refits the
weights on data to 2025-06-30 and is scored on the scored year only. The window is winter only at
D13, so the weights say little about summer; the write-up shows the fitted weights beside experiment
124's multipliers.

**Spread.** Each arm's quantiles are its mean plus a residual table (hour band by calendar quarter
by lead band) fitted on that arm's own tuning-month forecasts (the odd months of 2024-07 to
2025-06), every factor 1, as experiments 136 and 155 do. TD uses arm D's own rule (arm A's table).
BU uses a table fitted on the sum's residuals. A variant gives BU TD's table (BU-spreadTD), which
isolates the mean.

**Rows and periods.** `fold_a_daily_gap_v1`, 09:00 UTC `power_fcst_init_time`, the 292 runs of the
scored year (2025-07-02 to 2026-06-29), leads 0.5 to 168 hours. A row is scored when the BSP and its
three covered children have a reading and sit outside the hand-filtered switching events, so every
arm is scored on the same rows. Tuning months and replicate months are experiment 136's. Series are
D-labels. No generator series enters.

**Weather and what exists at forecast time.** The weather is the calibrated ensemble mean of
experiment 136 for the BSP's grid cell, and the filter's analysis proxy is built from earlier runs.
The children's arm D forecasts are the stored forecasts of experiment 155, made at the same origin.
Every input exists at forecast time.

**Scoring.** Beside the headline: lead bands, the tail table (coverage and pinball loss at 0.01,
0.05, 0.95, and 0.99), the mean absolute error of the mean forecast beside every quantile-score
contrast, absolute scores for climatology and the manual heuristic, and the node curtailment metric
of `studies.node_curtailment` at the scaled limit (f = 0.9, with 0.75 and 0.5), if the BSP's
generators-beneath inputs it needs exist for these two BSPs. Where it cannot be scored, the write-up
says so.

**Planned contrasts.** C1 BU minus TD (headline, at D13 and at D17), with TD also reported on all of
its own rows beside the shared rows. C2 BU1 minus TD. C3 BU minus BU1 (do learned weights help). C4
C1 by lead band. C5 BU-spreadTD minus TD (the mean alone). C6 BU minus TD at the tail quantiles. C7
BU-oracle minus TD (diagnostic). C8 BU-fc minus BU. C9 BU-w2 minus BU. Each is read through
`studies.bootstrap`.

**Controls, run before any contrast is read.**

- **Reproduction.** Arm D rescored from experiment 155's stored forecasts at the covered children on
  the common rows matches experiment 155's stored per-site scores within 0.001. - **Oracle (positive
  control).** BU-oracle scores below TD at both BSPs by at least half the share of effective
  capacity that the covered children make up (42% at D13, 56% at D17) times TD's score. The
  decomposition is exact in observed readings, so a smaller gain means a plumbing fault in the
  remainder, the weights, or the join. Experiment 36's oracle gained 3.4 to 5.6 points. -
  **Leakage.** The weight window ends before the first run, and an assertion checks that no
  remainder fit trains on a reading after its cut (the chain's `assert_no_lookahead`). - **Closure
  check.** Print the in-sample residual of the weight fit in MW and its share of the BSP's variance,
  to compare with experiment 124's audit (1.2 to 1.8 MW at the BSP that closes, 11 points at the
  other).

**Five triggers (size).** Stored: private files only. Serving path: not touched. Degradation rule:
not touched. More than one design defensible: yes, three, each pre-registered above. Callers not
nameable without searching: no, the scripts copy experiments 111, 136, and 155.

**Audit triggers to watch.** Trigger 1 (a control fails: the oracle), trigger 2 (the result
contradicts experiment 36's null, or this plan's prediction), and trigger 6 (an interval more than
twice as wide as experiment 155's per-site intervals) are the likely ones. The write-up lists which
fired and tells the manager if any did.

**Resources.** CPU only, at most 6 cores with one thread per worker, each job under `systemd-run
--user --scope -p MemoryMax=16G -p OOMPolicy=continue`. No GPU. New work: the ensemble frames for
the two BSP locations for the 456 early runs (the scored year's frames already hold both), the
weather calibration, and six chains (TD, the BU remainder, and the fixed-sum remainder at each BSP).

**Outputs.** Predictions for every arm in the `dp_preds` layout under
`data/ml-experiments/00161-dp-bsp-top-down-against-bottom-up/work/dp_preds/`.

**Advice followed and departures.** From experiment 36: do not expect the children's median
forecasts to help an XGBoost BSP model, and do not read the oracle's gain as headroom; this
experiment uses a different model and reports the oracle only as a control. From experiment 124: the
BSP reading is not an exact sum, so the weights and the remainder are estimated. From experiment
155: use arm D (the instruction for this experiment), apply recipe R at every lead, and read the
tails.
