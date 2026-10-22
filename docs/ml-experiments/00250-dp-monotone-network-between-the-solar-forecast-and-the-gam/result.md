---
id: '00250'
slug: dp-monotone-network-between-the-solar-forecast-and-the-gam
title: >-
  Does a small monotone neural network between the solar forecast and the demand model beat the
  hard clip?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - calibration-and-uncertainty
status: finished
size: >-
  medium. Stored: private files under
  data/ml-experiments/00250-dp-monotone-network-between-the-solar-forecast-and-the-gam/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (where the network sits, how it is kept monotone, which inputs it reads), so the plan
  fixes one. Callers:
  scripts copy the fit, forecast, and member-pass code of experiments 180, 214, 237, and 247. Asset
  graph: not touched.
hypothesis: >-
  A small neural network that is monotone in the explicit solar forecast, saturates, and reads the
  clear-sky index, the hour, and the clip cap maps the solar forecast to the solar output that the
  metered series shows, and so closes more of D5's sunny-daylight gap to BG2 than the hard clip,
  because it can learn how the saturation depends on the weather.
prediction: >-
  Written in the maintainer's brief before any run. At D5, the network closes at least 50% of the
  sunny-daylight gap to BG2 that the 60% clip leaves, does not worsen D10 or D16 by more than 0.01
  MW against the clip, and its 95% quantile has coverage within 2 points of 95%.
decision_rule: >-
  Closure beyond the clip = (clip's gap to BG2 minus the network's gap to BG2) divided by the clip's
  gap to BG2, on D5's sunniest third of daylight half-hours, months resampled as whole blocks in the
  same draw for both arms, 95% interval. `supported` when the closure is at least 50% with an
  interval not wholly below 50%, both placebo changes are at most 0.01 MW, and the q95 coverage is
  93% to 97%; `partly supported` when the closure point estimate passes and a placebo or coverage
  check fails or the interval includes less; `not supported` otherwise. The threshold is 50% because
  the network adds a trained component to the serving path; the brief set it.
benchmark: >-
  The unclipped fit, the hand-set 60% clip of experiment 180, and the profile-fit cap of experiment
  247, on experiment 178's sunny-daylight cut, against BG2
row_set_id: fold_a_daily_gap_v1
metric: >-
  Mean absolute error of the median in MW on the sunniest third of daylight half-hours (experiment
  178's cut), as a gap to BG2, at D5 and D9, with D10 and D16 as placebos; q95 coverage
code: ml-experiments/00250-dp-monotone-network-between-the-solar-forecast-and-the-gam/
verdict: not supported
production_status: not-for-production
adopted_in: none
date: 2026-10-22
best_arm: >-
  The network on the 60%-clipped solar term: at D5 it closes 41% of the clip's remaining
  sunny-daylight gap to BG2 (95% interval 28% to 81%)
headline: >-
  On the clipped solar term, the network lowers D5's sunny-daylight mean absolute error by 0.028 MW
  against the 60% clip alone (95% interval -0.037 to -0.021), which closes 41% of the clip's
  remaining gap to BG2 (28% to 81%), under the 50% threshold. On the unclipped solar term the
  network is 0.088 MW worse than the clip (0.019 to 0.142). The 95% quantile covers 99.5% of sunny
  daylight half-hours and 92.3% of all half-hours at D5, as the clip alone does.
lessons:
  - >-
    The network corrects the clip's remaining error a little at D5 and D9 (-0.028 and -0.016 MW)
    and moves the placebos D10 and D16 by at most 0.007 MW.
  - >-
    At D5 the network cannot replace the clip: without the clip it is worse than the clip.
  - >-
    The mean and standard deviation of the 51 members' solar responses lower the unclipped network's
    D5 error by 0.008 MW, which is small against its 0.088 MW excess over the clip.
advice:
  - >-
    Keep the hand-set 60% clip as the solar saturation in the final assembly and do not add a
    monotone network on top of it unless a gain of 0.02 to 0.03 MW at D5 and D9 is worth a trained
    component (the sunny-daylight figure).
  - >-
    Calibrate the upper tail separately: the clip leaves the 95% quantile covering 99.5% of sunny
    daylight half-hours at D5, and the network does not change that (the coverage table).
take_home_message: >-
  A small neural network that sits between the solar forecast and the demand model improves the
  sunny-day forecast at D5 by about 0.03 MW on top of the hard clip, which is
  less than half of what is left to close. It does not replace the clip, so the final forecast keeps
  the clip alone.
---

**On top of the hard clip, the network lowers sunny-daylight error a little, but not by the 50% of
the remaining gap that the plan required.** The network reads the solar term after the 60% clip, the
clear-sky index, the hour, and the cap. At D5 the network lowers the sunny-daylight error by 0.028
MW against
the clip alone (figure 1).

![Change in sunny-daylight error against the 60% clip](sunny_error_change.svg)

*Figure 1. Each arm against the 60% clip at D5 and D9. Smaller is better.*

**At D5, the network without the clip is worse than the clip.** The network on the unclipped solar
term
raises D5's sunny-daylight error by 0.088 MW (0.019 to 0.142) against the clip, and adding the mean
and standard deviation of the 51 members' solar responses lowers
that excess by 0.008 MW.

| Arm | D5 gap to BG2 (MW) | D5 closure beyond the clip | D9 closure beyond the clip |
|---|---|---|---|
| 60% clip | 0.069 | 0% | 0% |
| Network and 60% clip | 0.041 (0.008 to 0.062) | 41% (28% to 81%) | 38% (-59% to 221%) |
| Network, unclipped | 0.157 | -127% | -56% |
| Network, unclipped, two numbers | 0.149 | -115% | -53% |

The gap is the mean absolute error of the median minus BG2's, on the sunniest third of daylight
half-hours. The closure at D9 is unstable because the clip's own gap there is 0.042 MW with an
interval that includes zero.

**The placebos change by less than 0.01 MW.** The network and clip arm changes the all-half-hour
error at D10 and
D16 by -0.002 and -0.003 MW against the clip (figure 2), inside the 0.01 MW limit.

![Placebo sites](placebos.svg)

*Figure 2. The network arms against the 60% clip at D10 and D16, all half-hours.*

**The tail check fails for the clip and the network alike.** At D5 the 95% quantile covers 99.5% of
sunny-daylight half-hours (the clip alone: 99.6%) and 92.3% of all half-hours (the clip alone:
92.3%). Both coverages are more than 2 points from 95%.

## Checks

- The learned-cap arm of experiment 247 scores 0.005 MW worse than the clip on D5's sunny-daylight
  rows (-0.007 to 0.019), so it does not close the gap either.
- The band-width factors are fitted on the same rows as the network, of which 85% are in the
  network's training split. That may narrow the bands and contribute to the 92.3% coverage.
- Each network fit uses only forecasts whose valid time falls in the 12 months before the monthly
  cut. The early-stopping rows are the latest 15% of those rows.
- The network's output at a zero solar term is subtracted, so the network removes nothing at night.
- The network replaces the solar term in the stored forecast of the unclipped arm (experiment 180)
  or the 60% arm (experiment 214). The Kalman filter in each stored forecast adapted to its own
  solar term, not to the network's output.
- Each network is the mean of 3 seeds.

## Which clip form the final assembly should use

**Use the hand-set 60% clip alone.** The network removes 0.028 MW at D5 and 0.016 MW at D9 of sunny-
daylight error, for a
trained component on the serving path.

## What the next experiment should answer

A network trained end to end inside the demand model's loss, so the filter adapts to its output,
would test whether the stored-forecast replacement in this experiment held the gain down.

---
