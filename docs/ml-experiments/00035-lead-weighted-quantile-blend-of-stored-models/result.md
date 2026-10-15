---
id: "00035"
slug: lead-weighted-quantile-blend-of-stored-models
status: planned
size: medium
hypothesis: >-
  For Primary substation demand, solar farms, and wind farms, averaging the quantiles of two or
  three forecast models that experiments 19, 20, 30, and 33 already scored, with one weight per
  lead band chosen only on earlier months, lowers the quantile score below the best single member
  by more than the threshold set for each technology (0.05 points of capacity for Primary demand
  and solar, 0.10 for wind).
prediction: >-
  Wind: the blend of N1 and the best guess lowers the all-lead quantile score by 0.10 to 0.30
  against the best guess, mostly under 2 days, where N1 is 0.35 lower at 0 to 24 h. Solar: 0.03 to
  0.10 lower than the best guess. Primary demand: 0.03 to 0.10 lower than the best member of the
  monthly-retrained N1, `a7_cs`, and the stack. BSP and GSP are descriptive. The expected verdict
  is `supported` for wind and `small gain` for solar and Primary demand.
decision_rule: >-
  Three decisive contrasts, the rolling-origin lead-band blend minus the best single member of its
  technology, on the quantile score over all leads, read at 98.33% (Bonferroni across three) with
  whole months resampled (and the three seeds for solar and wind). The best single member is
  whichever member has the lowest all-lead score on the shared rows, so it is chosen with hindsight,
  which makes the contrast stricter, never looser. Per technology the outcome is "no combination"
  when the stop rule fires (below), "port" when the interval is wholly below zero and the estimate
  is at or below minus the threshold, "small gain" when the interval is wholly below zero and the
  estimate is above minus the threshold, "harm" when the interval is wholly above zero, and "null"
  otherwise. The stop rule: a cell is one scored month and one lead band, and it is a corner cell
  when the chosen weights put all the weight on one member; the stop rule fires when at least
  10/12 (83.3%) of a technology's 60 cells are corner cells. The verdict is `supported` when at
  least one technology is "port", `small gain` when none is but one is "small gain", `refuted` when
  every technology is "no combination" or "harm", and `inconclusive` otherwise. Thresholds are
  the effort of serving a second model per technology. BSP and GSP, the equal-weights and
  single-global-weight arms, and the correction-placement contrast decide nothing.
benchmark: >-
  The best single member of each technology on the same rows; also the equal-weights blend and the
  single-global-weight blend.
row_set_id:
  - fold_a_every5d_v1
  - fold_a_tuesdays_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS), in points of capacity; MAE of the median; coverage of the 5% to 95% band
advice: []
production_status: not-for-production
adopted_in: none
code: ml-experiments/00035-lead-weighted-quantile-blend-of-stored-models/
---

# Experiment 35: does a lead-weighted blend of quantiles beat the best single model?

**This page is the plan, committed before any code runs.** The results replace the sections below
when the experiment finishes. The experiment is CPU post-processing of stored predictions. It
starts no fit and needs no GPU.

## Size

**Medium.** The five `plan-issue` triggers, in order:

1. **What gets stored:** one private prediction set per blend arm under `data/ml-experiments/`; no
   Delta table and no production artefact.
2. **Production serving path:** not touched by the experiment. A port would add a blend stage that
   serves two or three models per technology (see "Porting cost").
3. **Degradation rule:** not touched. A port would need a rule for a missing member, which the
   plan leaves to a release plan.
4. **More than one defensible design:** yes. The weight grid, the selection window, the objective,
   and the order of correction and blend can each be set differently; the plan fixes one of each
   and runs the simpler alternatives as arms.
5. **Spans code whose callers could not be named without searching:** no. The scripts are new, read
   stored tables, and call `studies.bootstrap` and `studies.experiment_store`.

Reviews: one Sonnet plan review, one Sonnet code review for leakage (a month never chooses its
own weight), and one Sonnet write-up review.

## Members and rows

**Members are stored predictions from experiments 19, 20, 24, 30, and 33; no model is refitted.**
All tables are read from experiment 33's work folder, where they sit beside the observed power on
one set of keys.

| Technology | Rows | Members |
|---|---|---|
| Primary demand (16 series) | `fold_a_every5d_v1`, 73 runs, seed 0 | `monthly` (N1 retrained on the first of each month, no correction), `a7_cs` (weekly-analogue columns, correction, spread 1.15), `n1_cs` (the 00019 stack) |
| Solar (6 farms) | `fold_a_tuesdays_v1`, 53 runs, 3 seeds | N1 for generators; the ERA5-trained mixture with spread factor (00020 `base-spread`) |
| Wind (3 farms) | `fold_a_tuesdays_v1`, 53 runs, 3 seeds | N1 for generators; the ERA5-trained mixture with power-curve margin and spread factor (00020 `full`) |
| BSP and GSP (4 series) | `fold_a_every5d_v1`, 3 seeds | `n1` and `a7`, both without correction |

Primary demand uses seed 0 only because `monthly` has one seed, so its intervals resample months
only. The rows scored are the cases that every member of a technology holds with finite quantiles
and a finite observation, and the page states the count dropped. No valid time on or after
2026-07-01 is scored.

## Method

**Each blend averages each quantile level across members (Vincentisation), then sorts the nine
quantiles.** Averaging quantiles per level keeps the blend's spread between the members' spreads,
where averaging probabilities would widen it.

- **Lead bands:** 0 to 24 h, 1 to 2 d, 2 to 6 d, 6 to 10 d, and 10 d or more, as in experiment 33.
- **Weights:** one weight per member and lead band, on the grid 0, 0.25, 0.5, 0.75, 1 with weights
  summing to 1 (15 vectors for three members, 5 for two). The vector with the lowest mean quantile
  score on the selection cases is chosen; a tie goes to the vector nearest equal weights.
- **Rolling origin:** the weights for runs initialised in month m use only cases whose run was
  initialised before month m and whose valid time is before the first day of month m, so every
  observation used is known when month m begins. The scoring month is the run's initialisation
  month, as in `experiment_store.losses_for_bootstrap`. July 2025 has no history, so it uses equal
  weights, and it counts as no corner cell.
- **Selection pools** all series of a technology; the page reports the chosen weights per month
  and band.
- **Simpler arms:** equal weights always; one global weight vector per technology chosen by the same
  rolling rule on all leads pooled. An in-sample arm (weights chosen on all 12 months, per band)
  is reported as an optimistic bound and decides nothing.
- **Correction placement (Primary demand only):** arm D3 blends `monthly_cs`, `a7_cs`, and `n1_cs`
  (the correction applied inside each member, "before"), with weights chosen as above on D3's own
  scores. Arm D2 applies D3's weight table to `monthly`, `a7`, and `n1` without correction,
  then applies the 14-day correction, computed from the blend's own median history, and the 1.15
  spread factor to the blend ("after"). The correction code is copied from experiment 19. The
  headline arm D1 blends `monthly`, `a7_cs`, and `n1_cs`, as the proposal states.

## Planned contrasts

**Only C1 to C3 decide the verdict.** All contrasts use the quantile score over all leads.

1. **Decisive (3, 98.33%):** C1, D1 minus the best single Primary member; C2, the solar blend minus
   the best single solar member; C3, the wind blend minus the best single wind member.
2. **Exploratory, each family read at 5% divided by its size:** the equal-weights arm and the
   global-weight arm minus the best single member (technology by technology); the rolling blend
   minus the equal-weights arm and minus the global-weight arm; D2 minus D3 (correction placement);
   and per lead band, the rolling blend minus the best single member of that band.
3. **BSP and GSP:** the same contrasts, descriptive, pooled over 4 series.

**Intervals** resample whole months (and the three seeds where present) with
`studies.bootstrap.bootstrap_difference_at_level`. Intervals resample months, not series, so with
three wind farms they understate between-farm uncertainty.

## Porting cost

**The page states the cost of serving each blend.** The count of models served per run and per
technology, whether a member needs monthly retraining (the retrained N1 does), and the chosen
weight table's size (5 bands by 3 weights) are reported. The result does not decide a port; a
release plan does.

## Anonymisation

**Per-farm results use the anonymised labels** of `studies.anonymise` (solar A to F, wind W1 to
W3); no page names a generator or gives its identifier.

## Advice followed and departed from

The plan follows 00020 (the generator best guesses) and 00019 and 00021 (the 14-day correction does
not help BSP and GSP, so those members carry none). It departs from none. Experiment 33's interim
result, that the correction hurts monthly-retrained N1, motivates the placement contrast.
