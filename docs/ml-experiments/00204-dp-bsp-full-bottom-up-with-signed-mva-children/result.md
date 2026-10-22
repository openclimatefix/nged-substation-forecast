---
id: '00204'
slug: dp-bsp-full-bottom-up-with-signed-mva-children
theme: hierarchy-and-pooling
also_relevant_to:
- differentiable-physics-network
- differentiable-physics-embedded-generation
- production-reference-and-best-guess
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00204-dp-bsp-full-bottom-up-with-signed-mva-children/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is defensible (how the signed
  history of a child that reports apparent power is built, how the weights are fitted, which spread
  table the sum uses), and this plan pre-registers one of each. Callers: this folder's scripts, which
  copy experiment 161's, and read experiment 152's and 155's stored forecasts. No new package code.
hypothesis: >-
  A bulk supply point (BSP) is forecast better bottom-up than top-down when the sum covers every
  child Primary. Experiment 161 summed only the three children per BSP that report real power (MW)
  and have a long history, which reached 42% of D13's effective capacity and 56% of D17's, and the
  bottom-up forecast scored 0.5% to 0.6% worse than the top-down forecast. This experiment adds the
  children that report apparent power (MVA). A child's apparent-power reading is the magnitude of
  its flow, so a child with reverse flow at midday needs a signed flow before it can enter a sum.
  The fold-aware model of experiment 152 supplies that signed flow as a latent variable.
prediction: >-
  Written before any score is read. (d), the full bottom-up forecast using the latent signed flow of
  each apparent-power child, scores 3% to 10% below top-down at D13 and 0% to 5% below at D17. The
  gain is smaller at D17 because two of its nine children (D15 and D18) have too little history to
  enter (d), and stay in the remainder. (d) minus (c), the same sum with each apparent-power
  child's reading taken as signed, is within +-0.03 points at D13, where no child has reverse flow
  (experiment 152's Step 0), and -0.05 to -0.30 points at D17, where D12 (Stickney) has reverse flow
  on every sunny midday. The partial bottom-up (b) reproduces experiment 161's 4.387 and 7.706
  points. The oracle (observed children in place of forecasts) gains 20% to 40% over top-down at
  full coverage, more than the 7% to 12% of experiment 161, because the remainder shrinks.
decision_rule: >-
  The headline contrasts are C1 at D13 and C1 at D17: (d) minus (a) in points of the BSP's effective
  capacity, over leads 0.5 to 168 hours in the scored year, on the shared rows, each read at the
  97.5% level of the month-resampled paired bootstrap (12 calendar months, 20,000 resamples; the
  level is Bonferroni-corrected for the two BSPs). The threshold is a gain of 5% of the top-down
  score at that BSP, as the brief sets it, because bottom-up adds a remainder chain, a weight table,
  and the forecasts of up to nine children to the serving path. C2 is (d) minus (c) at each BSP.
  The hypothesis is `supported` when (i) at least one BSP has a C1 interval wholly below minus the
  threshold and (ii) at that BSP the upper end of C2's interval is at most +0.03 points, so (d) is no worse
  than (c) (at D13 the two should match; at D17 C2's point estimate and its interval against zero
  are also reported, because the prediction is that (d) beats (c) there). It is `partly supported` when at least one C1
  interval lies wholly below zero but condition (i) or (ii) fails. It is `refuted` when both C1
  intervals lie wholly above zero. It is `inconclusive` otherwise, and is capped there if the
  reproduction control, the oracle control, or the known-answer check fails. If every full arm
  still scores worse than top-down, the page says that coverage was not the reason and shows where
  the BSP's error lives (experiment 161's figure 3, redrawn at full coverage). Two intervals that both
  straddle zero mean there was no detectable gain from coverage, as in experiment 161; a C1
  interval that touches the threshold counts as not wholly below it. C3 to C8 carry no verdict.
benchmark: >-
  (a) Top-down, experiment 155's arm D chain on the BSP alone; (b) experiment 161's partial
  bottom-up; the oracle; ClimatologyForecaster and manual_heuristic for absolute skill
row_set_id: >-
  fold_a_daily_gap_v1 restricted to rows where the BSP and every child that enters any arm are
  present and outside the hand-filtered switching events (new shared set, printed as
  `bsp_full_v1`); top-down is also scored on experiment 161's own rows for the reproduction
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  the BSP's effective capacity over leads 0.5 to 168 hours, then under 24 hours, 24 to 72 hours, and
  72 to 168 hours; coverage and pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles; the mean
  absolute error of the mean forecast; the share of the BSP's effective capacity each arm covers
code: ml-experiments/00204-dp-bsp-full-bottom-up-with-signed-mva-children/
---

# Experiment 204: does a bulk supply point forecast improve when every child Primary is in the sum?

This page is the plan. Results are added after the run.

## Question

Experiment [161](../00161-dp-bsp-top-down-against-bottom-up/result.md) found that forecasting a BSP
bottom-up scored 0.5% to 0.6% worse than forecasting it as a stand-alone series. Its sum reached only
42% (D13) and 56% (D17) of the BSP's effective capacity, because the children that report apparent
power (MVA) stayed in the remainder with the unmetered load. NGED say we hold data for every Primary
under both BSPs. This experiment puts every child with a usable history into the sum and asks
whether coverage was the reason for the null result.

## The children, from NGED's hierarchy file of 2026-10-15

Both BSPs have every child NGED lists in our data. No child is missing. Years of readings count
present, unflagged half-hours (17,520 to the year). Primaries may be named. The BSP D13 is Skegness
33 kV and D17 is Boston 33 kV.

| BSP | Child | Unit | First usable reading | Years of readings | Enters |
|---|---|---|---|---|---|
| D13 | D9 Trusthorpe | MW | 2023-08-23 | 3.05 | (b), (c), (d) |
| D13 | D10 Spilsby | MW | 2019-09-17 | 6.95 | (b), (c), (d) |
| D13 | D11 Horncastle | MW | 2019-09-17 | 6.99 | (b), (c), (d) |
| D13 | D2 Warth Lane Skegness | MVA | 2019-09-17 | 6.99 | (c), (d) |
| D13 | D4 Ingoldmells | MVA | 2019-09-17 | 6.98 | (c), (d) |
| D13 | D19 Chapel St Leonards | MVA | 2019-09-17 | 6.99 | (c), (d) |
| D13 | D21 Alford | MVA | 2019-09-17 | 6.94 | (c), (d) |
| D17 | D5 Kirton | MW | 2019-09-17 | 6.97 | (b), (c), (d) |
| D17 | D14 Mount Bridge Boston | MW | 2019-10-21 | 6.89 | (b), (c), (d) |
| D17 | D16 Donington | MW | 2020-12-22 | 5.68 | (b), (c), (d) |
| D17 | D3 Sleaford Road Boston | MVA | 2019-09-17 | 6.97 | (c), (d) |
| D17 | D8 Marsh Lane | MVA | 2019-09-17 | 6.98 | (c), (d) from 2025-04-01 |
| D17 | D12 Stickney | MVA | 2019-09-17 | 6.97 | (c), (d) |
| D17 | D22 Langrick | MVA | 2019-09-17 | 6.97 | (c), (d) |
| D17 | D15 Wrangle T1 | MW | 2025-02-06 | 1.56 | (e) only |
| D17 | D18 Wrangle T2 | MW | 2024-12-11 | 1.64 | (e) only |

The two Wrangle transformers have under 8 months of readings at the first scored run, so no arm D
forecast exists for them. NGED's hierarchy also lists metered generators under each BSP (wind, solar,
biofuel, and battery sites). They are not Primaries and stay in the remainder, as in experiment 161.
Experiment 152's Step 0 found reverse flow at D12 only (every sunny midday); D22 has 3.7% and the
other six apparent-power children have none, so arms (c) and (d) differ materially at D17 only.

## Arms

All arms are scored on the same runs as experiment 161 (the 291 daily runs of 2025-07-02 to
2026-06-29), leads 0.5 to 168 hours. Every arm turns a mean forecast into quantiles with
**experiment 155's top-down spread table**, row by row, so the comparison isolates the mean.
Experiment 161's bottom-up arm refitted its table on the sum's corrected errors, and that narrowed
its tails and gave C1 its positive sign (experiment 161's C5). Experiment 152 stored forecasts for
the apparent-power children only in the scored and replicate years, with no tuning months, so a
table fitted on the sum's own residuals could not be built for the full arms in any case.

| Arm | Forecast of the BSP |
|---|---|
| (a) top-down | Arm D's chain on the BSP's reading alone (reference; stored by experiment 161) |
| (b) partial bottom-up | Experiment 161's sum over the three MW children plus arm D's forecast of the remainder; scored with its own table (reproduction) and with the top-down table |
| (c) full, naive | Sum over all children with a long history, each apparent-power child's reading and forecast taken as signed (experiment 152's arm N mean), plus arm D's forecast of the remainder `BSP - sum(w_i * y_i)` |
| (d) full, fold-aware | The same sum with each apparent-power child's signed flow forecast by experiment 152's fold-aware model (the latent `flow` column of arm F, not the folded quantiles), and the remainder built from the child's estimated signed history |
| (e) full plus short-history children | (d) plus D15 and D18, forecast with the pooled shapes of [experiment 158](../00158-dp-partly-pooled-gam-shapes-heating-and-calendar-curves/result.md) at 5 to 7 months of history; D17 only, reported separately |
| oracle | (d) with the children's observed readings in place of their forecasts (control, no verdict) |

**Signed history of an apparent-power child.** The remainder chain trains on `BSP - sum(w_i * y_i)`,
which needs each child's observed signed flow. The observed signed flow does not exist for a child
that reports only magnitude. Arm (c) uses the magnitude as it stands. Arm (d) uses the magnitude
times 0.97 (the power factor of experiment 124, which turns apparent power into active power, the
quantity the latent `flow` forecast measures), with the sign set negative on every half-hour where
the reverse-flow probability exceeds 0.9 (experiment 152's "sure" rule). The probability comes from
the monthly meter fit of experiment 152 that was trained before that half-hour's cut
(`work/fits/real/*_meter.pkl`), not from experiment 110's single fit, so no sign uses a later
reading. A sensitivity arm (d-p50) sets the sign at probability 0.5 instead. The history changes at
D12 on sunny middays and at essentially no other child. The sign of the converted flow is the main
risk, so Step 1 prints it for D12 before anything is fitted. The sign error enters the remainder
and the child forecast through the same meter fit, so the two are coupled, not independent.

**Weights.** Non-negative least squares of the BSP's reading on all children's signed readings with no
intercept, on the winter window 2023-10-01 to 2024-03-31 of experiment 161, bounded to 0.8 to 1.2.
The unbounded fit is printed as a diagnostic only. Winter has no reverse flow, so the weights do not
depend on the sign convention. D8's series changes meaning in April 2025 (experiment 152), so its
weight is fixed at 1 and it enters only from 2025-04-01; D17's full arms are therefore scored in
the scored year only, and D13 also has a replicate year.

## Steps

1. **Feasibility, before any chain runs.** Print the child table above from the stored files, the
   covered share of each BSP's effective capacity for every arm, and the per-half-hour sign estimate
   at D12 (share of sunny-midday half-hours with reverse flow, by month). Stop and report if the
   signed history of D12 cannot be rebuilt from stored experiment 110 outputs or a re-run of
   `studies.dp_meter` on D12 alone.
2. **Controls.** Reproduce experiment 161's top-down and partial bottom-up scores to three decimals.
   **Known-answer check.** For the five MW Primaries of experiment 152's injection test (D5, D10, D11,
   D14, D16), take its folded series at injection levels 0, 1.0, and 2.0 times the midday median, sum
   the five children, and compare the sum's mean forecast built as in (c) (arm N's `mean`) and as in
   (d) (arm F's `flow`) with the signed sum from the `signed` files. The pass criterion uses the
   power-factor-0.97 folding (`pf97`, which matches real meters); the idealised `q0` folding is
   reported beside it. This check tests experiment 152's latent-flow forecast only. It does not test
   the signed-history rule or the remainder chain. Pass: at level 2.0 the mean absolute error of (d)'s summed signed flow is at least 20%
   below (c)'s, and at level 0 (d) is no more than 5% worse than (c). Experiment 152 stored these
   forecasts, so the check costs minutes.
3. **Chains.** Remainder chains for (c) and (d) at both BSPs, and for (e) at D17: experiment 161's
   copy of experiment 136's A2 and experiment 155's arm D. The oracle reuses the (d) remainder.
4. **Score.** The contrasts below, the tail table, absolute skill for climatology and
   `manual_heuristic`, and the covered share beside every arm.

## Planned contrasts

C1 (d) minus (a), headline, at D13 and D17. C2 (d) minus (c). C3 (c) minus (b), the effect of adding
the apparent-power children without a sign model. C4 C1 by lead band (under 24 hours, 24 to 72,
72 to 168). C5 (e) minus (d) at D17. C6 (d) minus (a) at the tail quantiles. C7 oracle minus (a).
C8 (d) minus (a) leaving D12 out of the sum, which tests whether one child with reverse flow decides
D17. Each is read through `studies.bootstrap`. The reported checks include the mean forecast's error
beside every quantile-score contrast and the median minus mean forecast, per BSP.

## Triggers and size

- **Stored:** private files only.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes, three, each pre-registered above.
- **Callers not nameable without searching:** no, the scripts copy experiment 161's.

Audit triggers watched: a failed control (the known-answer check, the oracle), a result that
contradicts experiment 161 (it should not, since (b) reproduces it), and the mean's error pointing
the other way from the quantile score. One Opus audit is allowed and, if a trigger fires, it checks
the sign convention of the converted flow.

## Resources

CPU only, at most 4 cores with every thread cap at 1, each job under `systemd-run --user --scope -p
MemoryMax=16G -p OOMPolicy=continue`. No GPU. The children's forecasts of experiments 152 and 155 and
the spread tables of experiment 161 (`d_tuning_D13`, `d_tuning_D17`, and its `bu_r_tdspread` files)
are reused. Arm (e) needs new pooled-shape fits and chains for D15 and D18, so it runs last and is
dropped if the earlier arms use up the budget.

## Advice followed and departures

- **Followed:** experiment 161's advice to fix the weights' bounds and forecast the apparent-power
  children before expecting a gain; experiment 152's advice to keep the meter fit and fold only where
  reverse flow exists (here, D12); experiment 124's rescaling is not used, since the latent flow is
  already active power.
- **Departure from the brief's wording:** the brief asks for a gain of 5% with an interval wholly
  below zero. This plan requires the interval to lie wholly below minus 5%, which is stricter. For
  (d) beats (c), the plan requires only that (d) is no worse (upper end at most +0.03), because no
  apparent-power child at D13 has reverse flow. At D17 the predicted gain is 0% to 5%, so
  `supported` is expected to be reachable at D13 only.
- **Departure:** the spread table. All bottom-up arms use the top-down table, which experiment 161
  showed isolates the mean, instead of a table refitted on the sum.
- **Departure from the brief's wording:** the brief says the 5% threshold is experiment 161's.
  Experiment 161 used 2%. This plan uses the brief's 5%, and also reports the 2% reading.
- **Chain mismatch:** the apparent-power children's forecasts come from experiment 152's GAM-K chain
  (meter fit, additive filter), which has no winter night offset or error-correction recipe, so the
  MW children (arm D) and the apparent-power children are corrected differently. The page says so.
