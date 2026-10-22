---
id: '00247'
slug: dp-learned-export-cap-for-the-solar-clip
title: >-
  Can the export cap of the solar clip be learned from the metered series instead of set to 60%?
theme: differentiable-physics-embedded-generation
also_relevant_to: []
status: finished
size: >-
  medium. Stored: private files under data/ml-experiments/00247-dp-learned-export-cap-for-the-solar-clip/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (an upper envelope of clear-sky export, an in-sample profile, or a softer saturating
  clip), so we test three. Callers: scripts copy the fit and forecast code of experiments 180 and
  214. Asset graph: not touched.
hypothesis: >-
  A cap on the solar term learned at each monthly cut from the metered series before that cut
  lowers the sunny-daylight error at D5 almost as much as experiment 180's hand-set 60% clip,
  because the clip only needs the level at which clear-sky midday export stops growing.
prediction: >-
  Written in the maintainer's brief before any run. At D5, at least one learned arm closes at least
  80% of the sunny-daylight gain of the hand-set clip, and no learned arm worsens the sunny-daylight
  mean absolute error at D10 or D16 by more than 0.02 MW.
decision_rule: >-
  Closure = (error of arm A minus error of the learned arm) divided by (error of arm A minus error
  of the 60% clip), on D5's sunniest third of daylight half-hours, months resampled as whole blocks,
  95% interval. `supported` when the closure is at least 80% and its interval does not fall wholly
  below 80%; `partly supported` when the point estimate is at least 80% and the interval includes
  less; `not supported` otherwise. The placebos must not worsen by more than 0.02 MW. The threshold
  is 80% because the learned cap adds a fitting step at every cut, and the brief set it.
benchmark: >-
  Arm A (no clip), the hand-set 60% clip of experiment 180, and experiment 214's as-of ceiling
  selection, on experiment 178's sunny-daylight cut
row_set_id: fold_a_daily_gap_v1
metric: >-
  Mean absolute error of the median in MW on the sunniest third of daylight half-hours (experiment
  178's cut), at D5, D9, D10, and D16
code: ml-experiments/00247-dp-learned-export-cap-for-the-solar-clip/
verdict: partly supported
production_status: not-for-production
adopted_in: none
date: 2026-10-22
best_arm: >-
  The profile fit, which learns the ceiling at each cut: at D5 it closes 96% of the 60% clip's
  sunny-daylight gain (95% interval 62% to 104%)
headline: >-
  At D5 the profile-fit cap lowers the sunny-daylight mean absolute error by 0.128 MW against 0.133
  MW for the hand-set 60% clip, which is 96% of the gain (62% to 104%). The clear-sky envelope cap
  closes 60% (22% to 79%). At D9 the shares are 93% (70% to 189%) and 23% (5% to 32%). No arm moves
  the placebos D10 and D16 by more than 0.009 MW. A hand-set 60% ceiling with a softer edge closes
  100% at D5 (99% to 100%) and 104% at D9.
lessons:
  - >-
    A cap learned from data before each cut can match the hand-set 60% clip in the scored year, but
    the interval is too wide to show it reaches 80%.
  - >-
    The clear-sky envelope picks ceilings of 67% to 86% of held capacity at D5, above the 55% to
    70% that the profile fit picks there, and recovers 60% of the gain (22% to 79%).
  - >-
    Making the clip's edge softer changes the result by less than 0.005 MW, so the edge is not
    where the gain comes from.
advice:
  - >-
    Keep the hand-set 60% ceiling as the clip form for the final assembly at D5 and D9 (the
    closure table).
  - >-
    Where no ceiling can be hand-set, learn it with the profile fit over the 12 months before each
    cut, not with a clear-sky envelope quantile (closure figures).
take_home_message: >-
  The demand forecast can learn how much solar a site can export from its own recent metered data
  and gets almost the same improvement on sunny days as a ceiling set by hand at 60%. One year of
  data is too little to prove it, so the final forecast keeps the hand-set ceiling.
---

**The profile fit learns a ceiling that closes most of the hand-set clip's gain, but one year of
data cannot show it reaches 80%.** The profile fit refits the demand model for ceilings of 40% to
100% of held capacity on the 12 months before each cut and keeps the ceiling with the lowest
training objective. At D5 the learned ceiling stays between 55% and 70%, close to the hand-set 60%
(figure 2).

![Change in sunny-daylight error by arm](sunny_error_change.svg)

*Figure 1. The arms against the unclipped fit at D5 and D9.*

**What already exists.** Experiment 214 picks the clip ceiling as-of from a grid of shares by out-
of-sample error, and experiment 110's `profile_clip` fits the ceiling in sample. Neither experiment
learns the cap from the series alone without scoring a grid of candidate shares on forecast error.
The as-of choice of experiment 214 scores about the same as the profile fit at D5 (0.129 MW).

**Closure at the decision site.** Closure is an arm's gain over the unclipped fit divided by the
hand-set clip's gain, on D5's sunniest third of daylight half-hours, months resampled as whole
blocks in pairs.

| Arm | D5 closure (95% interval) | D9 closure (95% interval) |
|---|---|---|
| Profile fit (learned) | 96% (62% to 104%) | 93% (70% to 189%) |
| Clear-sky envelope (learned) | 60% (22% to 79%) | 23% (5% to 32%) |
| 60% clip, softer edge | 100% (99% to 100%) | 104% (98% to 134%) |

The profile fit passes the pre-registered 80% as a point estimate at both sites. The profile fit's interval
reaches below 80%, so the verdict is `partly supported`. The envelope arm fails: the quantile of
implied clear-sky export sits at 67% to 86% of held capacity at D5 and 84% to 100% at D9
(figure 2), above the ceilings that help.

![Ceiling picked at each cut](learned_shares.svg)

*Figure 2. The ceiling each learned arm picked at each monthly cut.*

**The placebos do not move.** The arms change D10's and D16's sunny-daylight error by
between -0.009 and +0.003 MW (figure 3), inside the 0.02 MW limit.

![Placebo sites](placebos.svg)

*Figure 3. The arms against the unclipped fit at D10 and D16.*

**The clip's edge is not the active ingredient.** A softer edge (smoothing 0.5 per point of
capacity instead of 2.0) gives the same gain as the hand-set clip: 0.132 MW at D5 against 0.133 MW.

## Checks

- Each cut's ceiling uses only rows before that cut. The envelope arm's window ends at the cut, and
  the profile fit's training rows end there.
- The arms read the same rows as experiment 178's cut, and the row counts match the control's.
- The scoring script raises when an arm's row count differs from the control's after the join.
- Held capacity can come from an early vintage newer than the cut when no vintage predates it;
  all arms share that choice.
- Closure ratios are unstable when the hand-set clip's own gain is small, as at D9 (the interval
  reaches 189%).

## Which clip form the final assembly should use

**Use the hand-set 60% ceiling.** Its gain is close to the profile fit's, and it needs no fitting
step at each cut. The softer edge changes nothing, so the experiment-180 form stays. Where a
site has no hand-set value, the profile fit is the better of the two learned choices.

## What the next experiment should answer

**The next experiment takes the clip and the profile-fit cap as inputs.** The small monotone neural
network of the brief's experiment 4 receives either ceiling as one of its inputs.
