---
id: "00152"
slug: dp-mva-fold-aware-gam-k-at-mva-only-primaries
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - differentiable-physics-forecast-engine
  - solar-disaggregation
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00152-dp-mva-fold-aware-gam-k-at-mva-only-primaries/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  design exists for the latent filter (a gated extended Kalman filter, a two-branch update, a
  particle filter), so the plan fixes the cheapest first and names the others as fall-backs.
  Callers: this folder's scripts, which copy experiments 110 and 136 and call the reviewed
  `studies.dp_*` modules without changing them.
hypothesis: >-
  At the MVA-only Primaries, the GAM-K forecast driven by the ensemble mean (experiment 136's arm
  A2) forecasts the MVA reading better when it models the fold: it fits the demand model through
  the meter (experiment 110's L0), corrects the forecast with a Kalman filter whose state is the
  signed flow, and builds its quantiles by folding a signed predictive distribution. The gain is
  concentrated at Stickney (D12), the one trial Primary where reverse flow is the usual state on
  sunny middays, and the change is harmless at the Primaries without reverse flow.
prediction: >-
  Written before any fit. Arm F minus arm N at D12 is -0.5 to -1.0 points of effective capacity
  over leads 0.5 to 168 hours (experiment 110's day-ahead -1.06 shrinks at long leads, where the
  ensemble mean's cloud error blurs the sign). Arm F minus arm A at D12 on sunny-midday rows is
  -0.05 to -0.25 points. Arm F minus arm N pooled over the other seven is within +-0.02 points.
  Coverage below q05 on D12's sunny-midday rows is 9 to 11% for N, 6 to 8% for A, and within 3
  points of 5% for F. F minus BG2 at D12 is -0.1 to +0.2 points.
decision_rule: >-
  Four planned contrasts at the 98.75% level (Bonferroni over four) of the month-block bootstrap on
  paired differences, 10,000 draws, as experiments 56 and 111, each read in the scored year
  (2025-07-01 to 2026-06-30) and required to keep its sign in the replicate (even months of
  2024-07 to 2025-06). R1, the fold-aware engine: F minus N on |S| at D12, all rows; passes at
  -0.30 points or lower with the interval wholly below zero. R2, the latent filter and folded
  quantiles: F minus A at D12 on sunny-midday rows; passes at -0.10 or lower with the interval
  wholly below zero. R3, harmless elsewhere: F minus N pooled over D2, D3, D4, D19, D21, D22, and D8
  after its cut; passes at +0.03 or lower. R4, against production: F minus BG2 at D12, labelled and
  never in the verdict. `supported` needs the known-answer test and R1, R2, R3. `partly supported`
  needs the known-answer test, R1 and R3, with R2 failing. `refuted` when R1's interval lies wholly
  above zero or R3's point estimate exceeds +0.10. `inconclusive` otherwise, and always when the
  known-answer test fails. The thresholds are set by the cost of porting: R1 needs 0.30 points
  because the fold-aware engine adds a meter fit and a new filter to the serving path; R2 needs
  0.10 points because the latent filter is a second port on top of R1.
benchmark: >-
  Arm N (GAM-K on |S| read as signed power), arm A (experiment 110's L0 with the additive filter),
  BG2 as stored by experiment 87, climatology, and manual_heuristic, on the same rows
row_set_id: dp_mva_ens_v1 (new)
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) on the MVA
  reading |S| in points of effective capacity, rows outside the hand-filtered switching events,
  leads 0.5 to 168 hours from the 09:00 UTC power_fcst_init_time; sliced by sunny midday, near-zero
  readings, season, lead band, and site; tail calibration at q01, q05, q95, and q99
code: ml-experiments/00152-dp-mva-fold-aware-gam-k-at-mva-only-primaries/
production_status: not-for-production
adopted_in: none
---

# Experiment 152: does modelling the fold help the forecast at the Primaries metered in MVA?

Plan only; the write-up follows the runs. The full design (observation model, known-answer test,
arms, risks) is in the private brief `data/ml-experiments/briefs/mva_design.md`; this page records
what the experiment commits to and the changes the maintainer made to it on 2026-10-21.

**Eight of the 16 trial Primaries report only the magnitude of apparent power (MVA), and the
reading bounces off zero when embedded solar pushes the net flow below zero.** This experiment asks
whether a forecast that models that bounce explicitly beats one that treats the reading as signed
power. [Experiment 110](../00110-differentiable-physics-5-embedded-generation-and-capacity/result.md)
already fitted the bounce for day-ahead forecasts on past weather and found it helps at Stickney
(D12) alone. This experiment tests the same idea on the ensemble-driven forecast, to seven days
ahead, with a filter and quantiles that also know about the fold.

## Changes to the Opus design (maintainer, 2026-10-21)

- **Go now.** Experiment 139 is not awaited: no as-of capacity can be built at an MVA Primary, so
  the capacity is fitted through the meter in every monthly refit.
- **Step 0 comes first.** It measures how much reverse flow each MVA-only Primary has, and its table
  heads the finished page.
- **Falsifier F3 is dropped.** Stickney has no MW meter, so its signed MW cannot exist. Only F1 and F2
  remain, and the page says the sign at D12 can be checked only indirectly.
- **Marsh Lane (D8) is scored only from 2025-04-01,** the month its series changed meaning.

## Step 0 (stored data only, run while writing this plan)

Experiment 110's stored L0 predictions, its fitted PV capacity, the CAMS irradiance in the feature
cache, and NGED's monitoring file (`Monitor Direction.csv`, 2026-10-15) give, for sunny-midday
half-hours (April to September, half-hours starting 10:00 to 13:30 UTC, CAMS irradiance at least
70% of the Haurwitz clear-sky value, outside switching masks), these first readings. The finished
page repeats them with a figure and the replicate year.

| Site | NGED reverse flow? | PV capacity (MW) | Sunny-midday half-hours with reverse flow | Largest reverse flow (MW) |
|---|---|---|---|---|
| D12 | Maybe | 5.9 | 100% | 5.3 to 5.5 |
| D22 | Maybe | 1.6 to 1.7 | 0.7 to 3.7% | 0.04 to 0.14 |
| D8 | Yes | 3.4 to 4.6 | 0% | 0 |
| D2, D3, D4, D19, D21 | No | 1.3 to 6.9 | 0% | 0 |

**Reading Step 0: D12 decides the test and the other seven carry the harmlessness check.** L0
predicts reverse flow at every sunny-midday half-hour at D12, a trace at D22, and none at D8, where
NGED's file says "Yes" but the series changed meaning in April 2025. All eight series are in the
metadata table as MVA primary demand, and no other MVA series exists there (the trial area's two
other MVA series are wind farms, outside the table). The power of the test follows: one site decides
R1 and R2, so a gain under about 0.5 points is hard to resolve.

## Plan

**Stage 0 (CPU minutes).** F1: the bulk-supply-point (BSP) closure of experiment 124 for the BSP
above D12, run on summer-midday rows with D12's |S| in the sum and again with D12's fitted signed
flow, each with a free term for the BSP's unmetered solar. A clear gain in residual variance from the
signed version supports the sign, no gain fails it. The free solar term is partly collinear with
D12's reverse flow, so a pass is weak evidence and a clear fail is strong. F2: replay the filter on
experiment 110's stored D12 predictions, additive on |S| against the latent filter with gating,
plus folded quantiles, day-ahead. Neither falsifier needs new fits. No reactive-power series exists
for these Primaries (experiment 79: NGED cannot supply P and Q), so there is no reactive check.
Time one multi-start meter refit, and confirm D8's cut date from its monthly profile.

**Stop rules.** S0: if F2 moves D12's sunny-midday score by less than 0.05 points in either
direction, drop the latent filter and run only N, A, and folded quantiles (A plus folded quantiles
becomes arm F). If F1 contradicts the sign at D12, stop and report that the sign cannot be
established from the data we hold. S1: if arm F's balanced accuracy in the known-answer test is
under 0.80 at level 2.0 with Q = 0 at 3 or more of the 5 sites, stop. S2: if F minus N at D12 has an
interval wholly above zero in the replicate months, stop and diagnose. S3: if D12's fitted
reverse-flow share jumps by more than 20 percentage points between consecutive monthly refits, flag
it, and stop if it happens in 3 or more refits. S4: if one multi-start meter refit takes longer than
5 minutes on one core, cut stage 2 to D12, D22, and D8.

**Stage 1, the known-answer test.** Fold the five MW decision Primaries (D5, D10, D11, D14, D16),
where the signed truth is known, by injecting farm output (levels none, 1.0, 2.0 as in experiment
79) and taking |P| with Q = 0 or a power factor of 0.97; run arm F, arm N, and a signed ceiling
arm S through the same ENS GAM-K configuration. Pass thresholds, fixed now: K1, balanced accuracy of
the forecast flow's sign at least 0.80 at 4 of 5 sites at levels 1.0 and 2.0, on rows with
|P_true| of at least 5% of effective capacity; K2, F minus N on the quantile score of the folded
series at most -0.10 at level 2.0 with the interval wholly below zero, and at most +0.03 at the
natural level; K4, the fitted capacity rises by the injected farm's capacity within a ratio of 0.75
to 1.25 at 4 of 5 sites at level 1.0; K5, coverage below q05 and above q95 within 3 points of
nominal on sunny-midday rows at level 2.0. K1, K2, K4, and K5 block; K3 (signed error against the
signed ceiling) blocks only a claim about the signed flow.

**Stage 2, the real test.** Arms N, A, F on the 8 MVA-only Primaries, driven by the calibrated
ensemble mean over experiment 136's 292 daily runs, monthly refits on a 24-month window, all 11
quantiles. A and F share the meter fits and differ in the filter and the quantiles. F's filter has
the signed-flow state, an observation row `H = (P_hat + l + z) / y_hat`, and skips the update where
the flow's sign is ambiguous. F's quantiles fold 200 samples of the signed flow. A two-branch
update replaces the filter only if it fails K1 near crossings. A learned-Q arm at D12 is
exploratory. D8's fits start at 2025-04-01 and are flagged.

**Compute.** CPU only, at most 4 cores, every thread cap at 1, run under `systemd-run --user
--scope -p MemoryMax=16G -p OOMPolicy=continue`: about 10 to 20 core-hours.

## Size triggers

- **What gets stored:** private files under the experiment's data folder only.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes, for the latent filter; the cheapest is planned and the
  others are named fall-backs.
- **Callers whose names need a search:** none; the scripts copy experiments 110 and 136.

## Earlier advice followed

Experiment 39's advice to leave MVA Primaries out of signed capacity fits is followed by fitting
capacity only through the meter. Experiment 56's advice to test the near-zero half-hours on more
series is followed by the near-zero slice. Experiment 79's standing advice to ask NGED for signed
MW at Stickney is dropped, because Stickney has no MW meter.
