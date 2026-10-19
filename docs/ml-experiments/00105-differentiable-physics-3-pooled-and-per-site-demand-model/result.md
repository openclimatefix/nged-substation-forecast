---
id: "00105"
slug: differentiable-physics-3-pooled-and-per-site-demand-model
title: "Differentiable physics, part 3: one pooled demand model across substations"
part: 3
theme: differentiable-physics-pooling-across-substations
also_relevant_to: [differentiable-physics-demand-calendar-and-weather, hierarchy-and-pooling]
status: planned
size: >-
  complex for the module changes (variational posterior, batched fits over row weights and targets,
  onboarding of a new site into a frozen pooled model, all in packages/studies and reused by Parts 4
  to 6); medium for the experiment scripts. Stored: private files under
  data/ml-experiments/00105-differentiable-physics-3-pooled-and-per-site-demand-model/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one is defensible
  (which parameters are site-level, freezing against refitting at onboarding, the onboarding spread
  model), and this plan pre-registers one of each. Callers: the module's callers are Parts 2 to 6.
hypothesis: >-
  One differentiable-physics model fitted across all the Primary substations forecasts a substation
  as well as the same model fitted to that substation alone when both have two years of history,
  and forecasts clearly better when the substation is new (4 weeks of history) or short of history
  (3 months).
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  Per-site DP (the same model fitted one substation at a time); lagless XGBoost on the same ERA5
  and CAMS inputs and window, plain and with experiment 64's solar offset, each with its matching
  start arm; Part 2's convex models G-all+D1, G-all, and G1 (stored predictions); climatology;
  manual_heuristic; lagged XGBoost as a descriptive reference at leads of 24 to 48 hours
row_set_id: dp_mw_v1 and dp_mva_dark_v1 (experiment 103); MW sites scored 2025-07-01 to 2026-06-30
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored site
verdict: <supported | partly supported | refuted | inconclusive>
production_status: not-for-production
take_home_message: <written after the results>
date: <date>
pr: <pr>
advice: <written after the results>
code: ml-experiments/00105-differentiable-physics-3-pooled-and-per-site-demand-model/
---

## What Part 3 asks, and the bar it starts from

**Part 3 asks whether pooling across substations is worth having on its own.** The series brief
counts pooling as a success if onboarding (C2) or short history (C3) is `supported` and the
all-rows contrast (C1) is not `refuted`. Pooling does not have to beat a per-site model on a
substation with plenty of history; it has to match it there and beat it where history is scarce.
Part 3 answers pooling on a 2-year window (series brief, "Merging single-site and pooled DP");
Part 4 answers years.

**Part 2 sets the bar the pooled DP must clear.** Part 2's corrected result (the spread centred
on the mean forecast) found that its convex model trained on all usable years with no drift (G-all)
scored 0.25 points worse than the same model on 12 months (G1, 2.82), that adding a drift term to
G-all helped by 0.29 points (G-all+D1, 2.78), and that G-all was no better than lagless XGBoost
(difference +0.00). Lagged XGBoost at leads of 24.5 to 48 hours still beats the convex models. The
bars for the pooled DP on all years are therefore G-all+D1 (2.78) and G1 (2.82), and G-all is no
longer a bar to beat. Part 3 scores the same rows in the same frozen year, so Part 2's stored G-all,
G-all+D1, and G1 predictions pair row for row with Part 3's arms (contrast C5). The numbers come
from the corrected Part 2 result; the plan quotes them from the design session's note until that
result merges.

**Four Part 2 findings shape this plan:**

- **No drift in Part 3.** Part 2's corrected result found that drift helps the convex model on all
  years (G-all+D1 minus G-all −0.29), but pooling and years stay separate, so every Part 3 arm runs
  without drift. The pooled drift switch stays off (it raises `NotImplementedError` in
  `studies.dp_pooled`, and Part 4 builds it). P-all is therefore compared with G-all+D1 as well as
  with G-all.
- **The solar offset adds nothing when CAMS irradiance is an input** (X-12am minus X-12z +0.006;
  fitted weights −0.02 to −0.34, not −1). The offset arm is therefore effectively plain lagless
  XGBoost on these inputs. Both arms stay, but the offset arm is not treated as the stronger bar:
  C4 is scored against whichever of the two scores better on the tuning year.
- **A zero start is a handicap, not a neutral benchmark.** X-12z scored 0.30 points worse than
  X-12, with q95 and q99 coverage of 66% and 73%. Each start arm is therefore matched to the start
  arm that differs from it only by the offset.
- **Positive-control predictions must not decide whether the verdict contrast is read.** In Part 2
  a failed prediction made C1 unreadable by the plan's own rule. Part 3 separates a blocking gate
  of known-answer tests (run before any scored fit) from non-blocking predictions (marked held or
  failed in the write-up).

## Rulings that fix the plan

The design session ruled on five gaps the planner found in the series brief (2026-10-19); this plan
follows them:

1. **Window.** The decision contrasts stay on the 2-year window, so pooling is measured apart from
   years. P-all against S-all is a secondary pooling contrast beside C1, and C5 is descriptive.
2. **Contrasts.** C1 to C5 are defined as in "Planned contrasts"; C4 is not part of the verdict.
3. **XGBoost bar.** The tuning-year winner of plain (either start) and solar-offset lagless XGBoost.
4. **Variational fit.** The variational fit is the model, so it runs at the screening tier; its
   intervals still come from month-resampling the stored predictions. Seeds, refits, and posterior
   intervals wait for the confirmation tier.
5. **Controls.** The placebo twin and shuffled-capacity tests are defined in the gate.

## Caveats fixed in advance

- The hyperparameters (K, penalties, step count, P-map scales) are tuned on a year that includes the
  held-out site's history, which a genuinely new site would not bring.
- P-all and S-all train on rows before 2023-08, which use the earliest capacity vintage scaled back
  by the growth prior, and before 2022-03, which carry no switching masks.
- Forecasts are issued at 09:00 UTC on the previous day, so on 30 June the last 15 hours before the
  cut are in training; the effect on lagless forecasts is negligible.

## Size triggers

- **Stored data:** only private files under `data/ml-experiments/<id>-<slug>/` and the feature
  cache. No contract or production table changes.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes, for the variational posterior (which parameters are
  site-level, the population priors), for onboarding (freeze the shared modules, or refit them),
  and for the onboarding spread model. This plan pre-registers one of each.
- **Callers:** the module's callers are Parts 2 to 6's scripts only. The new code imports torch,
  which is allowed only inside `packages/studies`.

**The module changes need their own reviewed PR.** They are new code in `packages/studies`, so the
experiment PR cannot merge autonomously. The module PR goes off `main`, gets a full Opus code
review (leakage, the training cut, and the leave-one-site-out weighting first), and the maintainer
merges it. The experiment scripts may run on the module branch once the review passes.

Pipeline: Opus plan review; Opus writes the module changes and the scripts; Opus code review of the
module PR; light review of the experiment scripts (configuration only); Sonnet mechanical fixes;
Opus numbers review against the printed report; Sonnet prose sweep.

## Module changes (`studies.dp_pooled` and `studies.dp_onboarding`, reviewed in PR 1356)

- **Variational posterior (the `variational` switch).** A mean-field Gaussian posterior over six
  families of site-level parameters: the style loadings, the per-site deviation, the heating slope
  and threshold, the level, and the capacity increments. The holiday loadings stay shared point
  estimates, because the model holds them per holiday period and hour band, not per substation. Each
  family has a population mean (zero for the loadings and the deviation) and a population scale,
  learned through the evidence lower bound; shared network weights stay point estimates with weight
  decay. The first trained season-year's capacity increment is a substation's whole capacity, so it
  is its own family (`capacity_first`), and the later growth increments are another. Only
  substations with solar rows enter the capacity prior. The forecast uses the posterior mean. The
  heating threshold's population mean carries a normal prior of 15.5 ± 2 °C. Levels start from the
  training rows only.
- **MAP ablation.** The same model with the population scales fixed by tuning on 2024-07 to
  2025-06 (the brief's ablation arm).
- **Many fits from one batch.** `fit_many` fits one independent model per copy, where a copy is a
  row subset (a leave-one-site-out fit drops one site's rows; a short-history fit drops one site's
  early rows) and optionally replaced targets (an injected-farm fit). The copies run one after
  another, because one fit already fills the GPU; stacking copies would add memory and no speed. The
  peak GPU memory of one copy is recorded.
- **Onboarding.** `onboard_site` takes a fitted pooled model in which the site had no rows, freezes
  the shared modules and the population distributions, and fits only the new site's site-level
  parameters on its rows, starting from the population mean. The site's capacity has an increment
  in every season-year its rows touch and in the forecast season-year.
- **Spread.** The replica spread of `studies.dp_gam` (the arm refitted 12 months before the cut
  with its own window rule, residual quantiles per hour band and quarter), applied to the pooled
  model. The residuals are centred per group, so the spread carries no bias from the replica year;
  Part 2's audit found that adding raw residual quantiles to a mean forecast shifts it by the
  replica year's bias. For an onboarding arm, the spread for site s comes from the replica of the
  same arm with s held out, scored on the other sites, so s's history never reaches s's spread; the
  residuals carry the weeks of history of the arm.
- **Tests.** (1) A pooled fit on one site equals the per-site fit on the same rows. (2) Changing a
  held-out site's targets or starting values leaves the other sites' fit unchanged. (3) A copy of
  `fit_many` equals the same fit alone. (4) MVA rows train demand only; their solar gradient is
  exactly zero. (5) Leakage asserts: an onboarded site's rows all end by its history end, the
  onboarded site is one that was held out, and onboarding leaves every other site's predictions,
  parameters, and centring means bit-identical. (6) The Kullback-Leibler term and the loss match their
  closed forms; the site prior ignores a site with no rows; onboarding fits the capacity of the
  season-year its rows fall in; the spread is centred.

## Sites and rows

- **Pooled training set:** the MW Primaries D5, D9, D10, D11, D14, D16, and D18, plus the 8 MVA
  Primaries (D2, D3, D4, D8, D12, D19, D21, and D22) on dark rows only through the observation
  operator (Q = 0, solar gate off; MVA rows never train the PV term). All 8 MVA sites hold 0.88 to
  0.95 dark site-years in 2023-07 to 2025-06, above the brief's 0.5 threshold. D15 is out (the
  generator case); the grid and bulk supply points and the two "battery or biofuel" series (G-a,
  G-b) are out.
- **Window:** 2023-07-01 to 2025-06-30 (2 years) for every main arm, `default_rows == True` (after
  the early-calibration cuts, outside the lockdown window and the ±24-hour switching masks). D9
  holds 21 months of that window and D18 4 months. The lockdown window 2020-03-23 to 2021-07-19 is
  left out by default; it matters only for P-all and the cold block.
- **Switching masks:** the 66 hand-filtered transfers of experiment 41's `labelled_set.parquet`,
  filtered as experiment 101 does, padded 24 hours each side at the donor and every receiver,
  already applied in Part 1's `rows.parquet`. The 2-year window lies wholly after 2022-03, so every
  training year has masks. The masks are hindsight labels, a mild leak shared by every arm.
- **Scored sites:** the MW Primaries. D10 has two unmasked jumps of 15 to 20 points (from about
  2024-11-08 and 2025-11-01, probably unlabelled switching), so every D10 number is reported with
  and without those windows. The decision count uses D5, D10, D11, D14, and D16 (history
  counted at the tuning year's first day, 2024-07-01, as in Part 2), so Phase A needs 4 of 5. D9
  (9.0 months at 2024-07-01) is scored and reported beside the count. D18 (4 months at the cut) is
  the one real short-history site and is reported descriptively beside C3.
- **Scoring rows:** every present row in 2025-07-01 to 2026-06-30 outside the switching masks at
  the scored site (headline), and every present row (reported beside it). Coastal D5 and D9 are
  also reported separately.
- **Predictions written:** `time_series_id`, `valid_time`, `q01`, `q05` to `q95`, `q99`,
  `actual`, `lead_hours`, `arm`, and `seed`, against a day-ahead initialisation at 09:00 UTC on the
  previous day, as in Part 2 (lead-band reads do not apply to lagless forecasts).

## Arms

**Every arm is screening tier:** one frozen fit on data before 2025-07-01, scored through the year
with no refits, one seed, on the GPU (Part 2's benchmark: the pooled proxy ran 6.6 times faster
there, peak memory 0.87 GB). Batched fits count as one arm per copy against the 10-minute rule.

**DP arms on all rows (2 years):**

| Arm | Model | Notes |
|---|---|---|
| P | pooled DP, variational posterior, interaction network on | the decision arm |
| P-map | pooled DP, MAP with fixed population scales | ablation |
| P-nonet | P with the interaction network off | exploratory |
| S | the same model as P, fitted one MW site at a time | per-site comparator |
| G-2y | Part 2's convex model (`studies.dp_gam`) on the 2-year window, no drift | convex per-site anchor |
| S-all | S on every usable year, lockdown out, no drift | per-site comparator for P-all |
| P-all | P on every usable year, lockdown out, no drift | secondary pooling contrast (P-all minus S-all); descriptive bar against G-all |

**Onboarding arms (C2), leave one site out:** for each scored MW site s (D5, D9, D10, D11, D14,
D16), one batched fit of P on every other site, then `onboard_site` on s's last 4 weeks before the
cut (2025-06-03 to 2025-06-30).

| Arm | Site s's history | Model |
|---|---|---|
| P-onb4w | 4 weeks | pooled shared modules frozen, site parameters fitted |
| S-4w | 4 weeks | per-site DP on 4 weeks |
| X-4w | 4 weeks | lagless XGBoost on 4 weeks (descriptive) |
| P-onb0w, P-onb2w, P-onb8w | 0, 2, 8 weeks | exploratory; 0 weeks is the population mean |
| X-pool-onb4w | 4 weeks | exploratory: one site-agnostic lagless XGBoost on every other site plus s's 4 weeks, in points |

**Short-history arms (C3), one site short at a time:** for each scored MW site s, one batched fit
of P on the full 2-year window in which s keeps only its last 3 months (2025-04-01 to 2025-06-30);
the shared modules are refitted with s inside.

| Arm | Site s's history | Model |
|---|---|---|
| P-short3m | 3 months | pooled, joint fit |
| S-3m | 3 months | per-site DP on 3 months |
| X-3m | 3 months | lagless XGBoost on 3 months (descriptive) |
| P-short6m, S-6m | 6 months | exploratory, run only if the part is inside its 2-hour budget |

**XGBoost benchmarks (2 years, lagless):** experiment 87's Primary hyperparameters with the three
power lags removed, 11 quantiles, the same ERA5 and CAMS inputs as Part 2 (temperature and its
four smoothings, CAMS GHI, solar elevation, half-hour, day type, holiday flags, day of year), on
CPU with `nthread` within the slot.

| Arm | Start | Matching arm |
|---|---|---|
| X-2y | XGBoost's estimated base score | — |
| X-2yq | each quantile from its own training quantile | — |
| X-2yamq | X-2yq plus experiment 64's solar offset | X-2yq |
| X-2yz | zero | — |
| X-2yam | zero plus the solar offset | X-2yz |

- **The solar offset** is built exactly as in Part 2 (`e00104_xgb.py`): capacity from experiment
  45's `frozen` vintage at the latest `origin_monday` on or before the cut, the CAMS unit PV shape
  minus its month-by-half-hour mean over the training window, and a weight w fitted per site on the
  matching arm's cross-fitted residuals in April to September. Report each site's w.
- **Lagged XGBoost (descriptive):** the same recipe with the three power lags, on the 2-year
  window, scored at leads of 24.5 to 48 hours and 144.5 to 168 hours, as in Part 2.
- **References:** Part 2's stored G-all+D1, G-all, and G1 predictions; `baseline_forecasters.climatology` and
  `manual_heuristic` on the same rows.

**Per-site comparators are tuned fairly.** S-4w, S-3m, X-4w, and X-3m are not run with penalties
tuned on full history: their deviation penalty (and XGBoost's rounds and depth) are tuned on the
tuning year with the same truncation applied at the replica cut. All four get the same
cross-fitted spread, stratified by weeks of history, as P-onb4w and P-short3m. P-onb4w minus
X-pool-onb4w is reported beside C2 at the same level, outside the verdict.

**Tuning (fitted on data before 2024-07-01, scored 2024-07-01 to 2025-06-30, by quantile score,
not mean absolute error):** K in {2, 4, 6} shared daily shapes by 3 values of the deviation penalty
(9 batched fits); the optimiser's step count and learning rate by early stopping on the same year;
P-map's population scales on a 3-point grid; the per-site arms' deviation penalty on a 3-point grid.
The winner of X-2y and X-2yamq on the tuning year becomes C4's benchmark, recorded before any
scoring-year number exists. If P does not converge within 10 minutes or fails the gate, P-map
becomes the decision arm, also decided on the tuning year before scoring. The scoring year is never
used to tune. D9 and D18 are left out of the tuning score.

**Cold block (descriptive, no interval):** P, S, G-2y, and X-2y refitted on the 2 years before
2022-11-01 (lockdown rows out) and scored November 2022 to February 2023 at D5, D10, D11, D14, and
D16, by the central forecast's mean absolute error, as in Part 2. No `am` arm, because no capacity
vintage exists before 2023-08. Each MVA site enters only if it holds 0.5 dark site-years in that
window.

## Known-answer gate (blocking, each test 5 minutes or less, before any scored fit)

**A failed gate test stops scoring until it is fixed.** If it cannot be fixed inside the part, the
verdict is `inconclusive`.

1. **Synthetic pooled recovery.** Eight synthetic sites built from real calendar and weather
   tensors, sharing K = 4 daily shapes and a heating threshold of 14 °C (away from the prior's
   centre), with their own loadings, heating slopes, levels, and capacity paths. P recovers the
   threshold within 1 °C, every heating slope within 15%, every capacity within 15%, and loadings
   whose correlation with the truth is at least 0.9. A ninth synthetic site onboarded from 4 June
   weeks recovers its heating slope within 25%, while S-4w on the same weeks misses it by more than
   50% (the test must be able to tell the two apart).
2. **Injected-farm recovery** (experiment 39's harness, `e00039_inject.py`, real solar-farm output
   added to real demand). Inject at one MW site at a time (D10, D11, D14, D16; batched), fit P,
   and pass if the median ratio of recovered to injected capacity increment lies in [0.75, 1.25]
   (experiment 39's band), the other sites' fitted capacities move by less than 5%, and the shared
   modules' midday output moves by less than 0.1 points. The last two conditions are the pooling
   check: a farm at one site must not leak into the others or into the shared terms.
3. **Placebo twin (ruling 5).** Replace each site's CAMS irradiance with the same site's irradiance
   from the same calendar day one year earlier (one year later where the earlier year is missing),
   keeping the solar geometry. Refit P. Pass if the summer-midday score (April to September, 10:00
   to 14:00 local) is worse with the placebo, with a month-resampled interval that excludes zero,
   and the fitted capacity falls at 4 of 6 MW sites, for the 2024-25 season-year. The placebo is one
   draw.
4. **Shuffled capacity (ruling 5).** Permute the capacity priors across the MW sites (a derangement,
   in MW, converted to each site's points) and refit. Pass if, at two-thirds or more of the MW sites
   (4 of 6), the fitted capacity lies closer to the site's own experiment 45 `frozen` vintage than
   to the shuffled prior, for the 2024-25 season-year and with P's own prior scale (a weakened prior
   would pass trivially). For a pair whose own and shuffled priors differ by less than the prior's
   standard deviation, closeness is undefined: the write-up reports the separation and excludes the
   pair from the count.
5. **Module tests 1 to 5** (above) pass.

## Planned contrasts

The verdict contrasts C1 to C3 form the Bonferroni family: 98.33% intervals from
`studies.bootstrap`, resampling whole calendar months of the stored predictions, paired across arms,
as in Part 2. C4, C5, and the secondary contrast are outside the family and use 95% intervals. Each
is reported with the count of sites moving each way and every arm's absolute score. Rulings of the
design session (2026-10-19) fix the definitions:

- **C1, pooling on all rows (non-inferiority, margin +0.10 points):** P minus S on the five
  decision sites' scored rows, with D9 reported beside them. At the screening tier, C1 passes if
  its point estimate is at most +0.10 points and its interval does not lie wholly above zero. At
  the confirmation tier, C1 passes only if the upper end of its interval is at most +0.10 points.
  The margin is the design session's ruling. The take-home says which of the two C1 meets.
- **C2, onboarding:** P-onb4w minus S-4w on each held-out site.
- **C3, short history:** P-short3m minus S-3m.
- **C4, the DP against lagless XGBoost:** P minus the tuning-year winner of X-2y, X-2yq, and X-2yamq.
  Labelled and reported, not part of the verdict.
- **C5, the DP against Part 2's bar:** P-all minus G-all+D1, and P-all minus G1 (Part 2's stored
  predictions, same rows). P-all minus G-all is reported beside them.
  Descriptive: the bar the pooled DP must clear before Part 4.
- **Secondary pooling contrast beside C1:** P-all minus S-all (both on all years, no drift).

Exploratory at 95%: P-map minus P; P-nonet minus P; S minus G-2y (the DP's expressive terms within
one site); P-onb4w minus P (how
much of full-history skill onboarding recovers); P-onb4w minus X-pool-onb4w (does pooled XGBoost
onboard as well); the onboarding curve over 0, 2, 4, and 8 weeks; X-2yamq minus X-2yq and X-2yam
minus X-2yz (the solar offset); X-2yq minus X-2y; the coastal sites; D18 under P, S, and X-2y; the
scarce-data rows (bank holidays, the Christmas run, days below 0 °C, no interval); the cold block
(no interval); and the score by month since the cut.

## Predictions

Written before any fit; each line can fail, and none of them overrides the decision rule.

- **C1:** P minus S lies within ±0.15 points and the interval spans zero (pooling neither helps nor
  hurts with 2 years of history).
- **C2:** P-onb4w minus S-4w is −0.5 to −2.0 points, interval wholly below zero, 5 of 5 sites
  better. S-4w sees no winter, so it cannot fit heating; the pooled heating curve can.
- **C3:** P-short3m minus S-3m is −0.2 to −1.0 points, interval below zero, at least 4 of 5 sites
  better.
- **C4:** P is 0.0 to 0.4 points better than the stronger lagless XGBoost arm.
- **C5:** P-all minus G-all+D1 is −0.3 to +0.3 points, interval spanning zero: the pooled DP
  without drift roughly matches the convex model with drift, because the DP has more expressive
  terms and the convex model has the drift. P-all minus G1 is −0.3 to 0.0 points.
- **Onboarding recovery:** P-onb4w is within 0.5 points of P.
- **P-all minus S-all:** within ±0.15 points, as for C1.
- **Ablations:** P-map within ±0.1 points of P; P-nonet within ±0.15 points of P.
- **The solar offset:** X-2yamq minus X-2yq within ±0.05 points, as in Part 2.
- **Tails:** P's q01 and q99 coverage within 1 percentage point of nominal, and q05 and q95 within
  2 points; the same bands for P-onb4w and P-short3m, whose spread model is the least tested part.
  Part 2's arms missed by 1 and 2 points, so these bands may fail, and the write-up says so
  plainly. The tails stay outside the verdict. The q01 and q99 pinball loss of C1, C2, and C3 is
  reported beside each headline estimate.
- **Capacity:** P's 2024-25 capacities lie within ±20% of experiment 45's `frozen` vintage at D10,
  D11, and D16. The hindsight season-year constant in `capacity_weekly.parquet` is an
  after-the-fact check only, never an input; Part 2 found the two files disagree (ratio 0.30 to
  1.48 in early weeks, about 1.10 at D11 near the cut).
- **Sunny-hour residual against temperature (diagnostic):** the slope is under 0.67 points per
  kelvin (experiment 51's XGBoost figure), because the DP derates PV by temperature.
- **Run time:** every arm under 10 minutes; the whole screening, the gate and tuning included,
  under 2 hours of GPU wall clock.

## Decision rule

**Every contrast is also reported per site and with each decision site dropped in turn** (C1 to C3,
C5, and the P-all minus S-all contrast). Part 2's C1 rested on D14. A label that flips when one
site is dropped is reported as such in the take-home. C2 and C3 are also reported by month since the
cut. Both histories lie in April to June, so both measure summer onboarding; a winter-onboarding arm
is a follow-up, not part of Part 3.

**History windows.** "4 weeks" means the 28 calendar days ending at the cut (2025-06-03 to
2025-06-30); "3 months" is 2025-04-01 to 2025-06-30; 0, 2, and 8 weeks follow the same rule.
Masked or missing rows inside a window are not replaced by reaching further back, and the write-up
reports each site's usable-row count.

**The verdict applies the series brief's pooling success criterion, under its Phase A labels
(history counted at 2024-07-01, so 4 of 5 decision sites).** C1 to C3 are labelled
`supported`, `partly supported`, `refuted`, or `inconclusive` by the Phase A rule; `not supported`
and `small gain` are Phase B labels and are not reachable here.

- **`supported`:** C2 or C3 is `supported` and C1 passes, with the gate passed. Part 3
  then goes to the confirmation tier (8-week refits, three seeds, the coverage check of the
  variational intervals on at least 20 synthetic sites, and the prior-sensitivity check) before
  Part 4 relies on it.
- **`partly supported`:** C1 passes, and C2 or C3 has an estimate below zero with fewer
  than 4 of 5 sites improving, while neither is `supported`.
- **`refuted`:** C1 is `refuted` (pooling costs accuracy on substations with full history), or C2
  and C3 are both `refuted`.
- **`inconclusive`:** anything else, or a gate test that cannot be fixed inside the part.

**C4, C5, and the P-all contrast get a label in the bullets and play no part in the `verdict`
field.** A DP gain over XGBoost is claimed only from C4. C5 is labelled for the hand-on decision
only. A pooled DP that loses C5 (the P-all minus G-all+D1 estimate above zero) is reported in the
take-home whatever the verdict, because Part 4 would then start from the convex all-years model
rather than the pooled DP.

## Run order, budget, and stop rule

The manager allows up to 10 GB of GPU for this part (SIMP1 holds about 5 GB of the 48 GB) and 8 CPU
cores. The module PR measures peak memory at B = 1, 2, 4, and 6 copies; the run uses the largest B
whose peak stays under 8 GB, so the pooled fits batch about 6 copies at a time.

1. Module PR: variational posterior, MAP ablation, batched fits, onboarding, replica spread for
   the pooled model, tests 1 to 5.
2. Gate tests 1 to 4.
3. Tuning.
4. Main arms P, P-map, P-nonet, S, S-all, G-2y, P-all, and their replicas.
5. C2 and C3 arms (leave-one-site-out, onboarding lengths, short-history fits, and their replicas;
   the 6-month arms last).
6. XGBoost arms, lagged reference, and references (CPU, alongside the GPU fits).
7. Cold block.

**Budget (planning estimates, replaced by the module PR's measurements before step 2).** A "copy"
is one fitted model; P-all and S-all copies hold about three times the rows of a 2-year copy.

| Step | Copies | Estimated wall clock |
|---|---|---|
| Gate (synthetic, 4 injected farms, placebo, shuffled) | about 20 | 30 minutes |
| Tuning (9 pooled, 3 P-map, 3 penalties for 6 per-site sites) | about 30 | 45 minutes |
| Main arms with replicas (P, P-map, P-nonet, P-all; S, S-all, G-2y per site) | about 50 | 75 minutes |
| C2 and C3 (6 + 6 batched fits, onboarding, replicas, per-site comparators) | about 30 | 45 minutes |
| XGBoost and references (CPU, in parallel with the GPU) | 0 | 0 minutes extra |
| Cold block | 4 | 15 minutes |

The estimate is about 3.5 hours, so the part's budget is 4 hours of GPU wall clock. The 10-minute
limit per arm applies to a batched fit as a whole, not to each copy. **Stop rule:** at the end of
step 5, if more than 4 hours have passed, cut in this order: the 6-month arms, P-onb2w and P-onb8w,
P-nonet, the cold block. If the gate and tuning alone pass 2 hours, message the manager before
continuing.

## Data paths

- **Part 1 outputs:** `data/ml-experiments/00103-differentiable-physics-1-data-look/`:
  `rows.parquet`, `sites.parquet` (`months_before_2024-07`, `months_before_2025-07`,
  `dark_site_years_two_years`), `panel.parquet`, and `decisions.json`.
- **Part 2 outputs:** `data/ml-experiments/00104-differentiable-physics-2-simple-structural-model/`:
  the feature cache (`cache/`), and stored predictions under `predictions/g_all/`, `predictions/g1/`,
  `predictions/x_12/`, and `predictions/x_all/` for C5 and the references. Code to copy from:
  `ml-experiments/00104-*/e00104_xgb.py` (benchmarks and offset), `e00104_references.py` (lagged
  reference, climatology, `manual_heuristic`), and `e00104_report.py` (scoring, bootstrap, tails),
  on PR 1349's branch until it merges.
- **Capacity vintages:**
  `data/ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/capacity_weekly_vintages.parquet`,
  variant `frozen`. Never read D14 for 2023-09-04 to 2025-08-25; its last observed
  vintage week is 2023-04-17 and that value is held. D14 counts in the shuffled-capacity 4 of 6, and
  the write-up reports it separately because a two-year-stale vintage is a weak "own" target.
- **Injected farms:** experiment 39's harness (`ml-experiments/00039-pv-disaggregation-cams-injection/e00039_inject.py`)
  with the farm output in the experiment 45 folder (`farm_half_hourly.parquet`). The farms are
  metered generators: no farm series, name, or identifier appears in any figure or in the
  write-up; the gate reports ratios only.
- **Framework:** `studies.dp_pooled`, `studies.dp_gam`, `studies.dp_operator`,
  `studies.dp_features`, `studies.dp_calendar`, `studies.bootstrap`, `studies.tail_calibration`.

## What the write-up must contain

Title "Differentiable physics, part 3: one pooled demand model across substations", front matter as above,
at `docs/ml-experiments/<id>-<slug>/result.md`.

- **The take-home paragraph first:** whether one model across substations matches a model per
  substation, how much it gains for a new or short-history substation, and where it stands against
  G-all and lagless XGBoost. Say "better" and "worse", never "lower" and "higher", and gloss
  "pooled", "onboarding", and "frozen fit" in plain words.
- **One or two plain intro paragraphs:** the series in one sentence, the question, the answer.
- **Figures** (raw data first): one scoring-year winter week per scored site, actual against P, S,
  P-onb4w, and S-4w; the headline chart of C1 to C5 with intervals and site counts; every arm's
  absolute score per site; the onboarding curve (score against weeks of history for P-onb and
  S, with P and G-all as horizontal marks); the shared daily shapes and each site's loadings; the
  shared heating curve with each site's threshold and slope; capacity per season-year against the
  vintages; the gate results (injected recovery ratios, placebo and shuffled-capacity bars); the
  tails (coverage and pinball at q01, q05, q95, and q99, every arm); the sunny-hour residual
  against temperature; the score by month since the cut. No generator data appear, and G-a and G-b
  are not plotted.
- **Bullets only after the figures:** arms and absolute scores; C1 to C5 with intervals, counts,
  and labels; the tails table; each prediction marked held or failed, in two lists; the gate;
  the tuned K and penalties; the solar weights w; the cold block; D18; the scarce-data rows; the
  coastal sites; run time and peak GPU memory; caveats (ERA5 and CAMS are estimates of past
  weather and flatter the solar term; the masks are hindsight labels; a frozen fit is staler than
  a production model, which is refitted weekly; five decision sites and one year give wide
  intervals); what the next experiment should answer.
- **Data lessons** go on `docs/ml-experiments/data-processing.md` in the same PR, each with a
  "Confidence:" sentence and a link.

## Decisions handed on

- **The pooled configuration for Parts 4 to 6:** K, the deviation penalty, variational or MAP,
  interaction network on or off.
- **Onboarding:** whether a new site enters through `onboard_site` or a joint refit, and how many
  weeks of history it needs to come within 0.5 points of full history.
- **The XGBoost benchmark set** (which start arm; whether the solar offset arm is kept).
- **Whether Part 4 starts from P-all or from G-all**, by C5.

## Hooks for Part 4

**Part 4 should test a tamer final drift knot alongside the Kalman assimilation.** Part 2's numbers
review found that the drift arm lost most in months 0 to 4 after the cut (G-all+D1 minus G-all
+0.18 to +0.94), when the held drift is freshest, not late in the year, and that D10's last knot rose
4 to 6 points in every band over the last 12 months of training. The final knot appears to chase the
last months of training. Candidate arms beside A0 to A4: D1 with the final knot shrunk towards the
mean of the last 12 months' knots, D1 with the final knot pinned to the one before it, and D1 with
a stronger penalty on the first differences of the last 3 knots. Two more lessons carry over: tune
the drift step by quantile score, not mean absolute error, and extend the grid below 3,000, where
Part 2's chosen value sat at the edge. Part 2's `tuned.json` records `d1_steps` as Part 4's default
drift (steps beat D1 by 0.215 on the tuning year), but steps scored 0.06 worse in the scored year,
so this plan recommends plain D1 as Part 4's default.
