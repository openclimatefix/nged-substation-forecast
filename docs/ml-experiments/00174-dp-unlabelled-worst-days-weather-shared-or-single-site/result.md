---
id: '00174'
slug: dp-unlabelled-worst-days-weather-shared-or-single-site
part: BG-EB2
title: >-
  Is the loss on the forecast's unlabelled worst days weather error, a day shared with other
  substations, or an error at one substation only?
theme: differentiable-physics-forecast-engine
also_relevant_to:
- production-reference-and-best-guess
- switching-labels-in-the-forecast
- weather-products-and-blending
status: done
size: >-
  medium. Stored: private tables under data/ml-experiments/00174-dp-unlabelled-worst-days-weather-shared-or-single-site/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (which past-weather estimate, whether the spread is refitted, which forecast ranks the
  other Primaries' days), and this plan pre-registers one of each and reports the alternatives as
  labelled sensitivities. Callers: this folder's scripts, which read experiment 155's and
  experiment 136's stored forecasts and fits, experiment 172's stored day tables, and experiment
  87's stored BG2 forecasts, and call the reviewed `studies.*` modules without changing them. No
  model is fitted.
hypothesis: >-
  Most of the excess loss on the unlabelled worst days of experiment 172 is weather error, and the
  unlabelled worst days are mostly shared with other Primaries. Replacing the ensemble-mean weather
  with the estimate of past weather (ERA5 temperature and CAMS irradiance) in arm D's forecast
  removes at least 50% of the unlabelled days' excess loss, and at least a third of the unlabelled
  days are also in the worst 5% of days of at least one other Primary.
prediction: >-
  Written before any score is read. Past weather removes 40% to 70% of the excess loss on the
  unlabelled worst days (the labelled switching days keep most of theirs, because switching is a
  demand-side event). 35% to 60% of the unlabelled days coincide with another Primary's worst-5%
  day. The chance rate of that coincidence is high with 15 other Primaries (about 54% if days were
  independent), so the test that carries information is the excess over a circular-shift control.
  A residual third group, the excess that stays with perfect weather on a day no other Primary
  shares, is 10% to 30% of the unlabelled days' excess.
decision_rule: >-
  The 72 worst substation-days of experiment 172 (five decision substations, arm D, D+1 band, the
  scored year) are split into its 37 unlabelled days and 35 labelled days. Weather share W is the
  sum over the unlabelled days of min(max(E_ens - E_era5, 0), E_ens) divided by the sum of E_ens,
  where E_ens is the day's excess loss in arm D and E_era5 is the same day's excess in arm D with
  ERA5 temperature and CAMS irradiance in place of the ensemble mean (the excess of a day is its
  mean quantile score minus the substation's median day, each arm against its own median, with the
  spread refitted on each arm's tuning months). Shared share S is the fraction of the unlabelled
  days on which at least one of the other 15 Primaries of experiment 87 is in its own worst 5% of
  days by BG2's score. The hypothesis is `supported` when W is at least 0.5 and S is at least 1/3,
  `partly supported` when exactly one holds, and `refuted` when neither holds. The 95% interval of
  each is a bootstrap over the 12 calendar months of the scored year (20,000 resamples). The split
  of the unlabelled days' excess is: weather (E_ens minus E_era5, clipped to the day's excess),
  shared (the remaining excess on days with S true), and neither (the remaining excess on days with
  S false). The neighbour question is the same count restricted to the Primaries that the
  interconnection table joins to the substation. The verdict is capped at `inconclusive` if the
  injected controls do not recover at least 90% of their injected days in the right group.
benchmark: >-
  The labelled worst days (35) as the comparison set, and a circular-shift control for S (the other
  Primaries' day series shifted by random whole numbers of days, 1,000 draws)
row_set_id: fold_a_daily_gap_v1
metric: >-
  Daily quantile score above the site's median day, in points of effective capacity, over the D+1
  rows (lead 15 to 39 hours) outside the hand-filtered switching masks, 2025-07-03 to 2026-06-30,
  as in experiment 172
code: ml-experiments/00174-dp-unlabelled-worst-days-weather-shared-or-single-site/
verdict: partly supported
best_arm: >-
  Not applicable (descriptive). Past weather removes 31.5% of the unlabelled worst days' excess loss
  (95% interval 21.9% to 39.8%), against the 50% needed. 48.6% of those days are also worst days at
  another Primary; shifting the other Primaries' days at random gives 46.6% (35.5% when they are
  shifted together).
headline: >-
  Replacing the ensemble-mean weather with past weather (ERA5 and CAMS) removes 31.5% of the excess
  loss on the 37 unlabelled worst days, under the 50% threshold, so most of that loss is not weather
  error. Another Primary is also in its worst 5% on 48.6% of the days, above the one-third
  threshold, and only weakly more than random shifting of the days gives.
lessons:
  - >-
    The loss on unlabelled worst days is mostly not weather error: past weather removes about 31% of
    it, against about 6% on the labelled days.
  - >-
    Coincidence with other Primaries is at the level of independent chance (46.6%) and above the
    level of a shared shift (35.5%, 7.8% of draws reach 48.6%), so the data show at most a weak
    shared cause, and the remaining loss could come from events at one substation.
  - >-
    The headline weather share and its four sensitivities all fall between 29% and 37%.
take_home_message: >-
  Most of the forecast's error on its worst unlabelled days is not weather error: with past weather
  in place of the forecast weather, about 69% of the excess loss stays. Other Primaries share these
  days about as often as random shifting gives, so the data show at most a weak shared cause. The
  remaining loss could therefore come from events at one substation, such as undetected switching,
  a change in behaviour, or a metering fault. The next place to look is the substation's own recent
  readings and the switching records on those days.
advice:
  - >-
    Look for single-substation events (undetected switching, a change in behaviour, a metering
    fault) on the unlabelled worst days before improving the weather input; past weather removes
    only 31.5% of their excess loss (figure 1).
  - >-
    Test a coincidence of worst days across Primaries against shifted copies of the other
    Primaries' days, shifted both separately and together, because the answer differs: 46.6% by
    chance when shifted separately and 35.5% when shifted together (figure 3).
  - >-
    In experiment 155's arm D forecast, past weather removed about a third of the excess loss on
    these 37 days (29% to 37% across the sensitivities), so a better weather input alone will not
    fix them (figure 1 and the weather-share sensitivities).
production_status: not-for-production
adopted_in: none
date: 2026-10-22
pr: 1479
---

**Replacing the weather forecast with estimates of past weather removes only about a third of the
loss on the worst days that experiment 172 could not explain, so most of that loss is not weather
error.**
[Experiment 172](../00172-dp-error-budget-worst-days/result.md) ranked each day of five Primary
substations (D5, D10, D11, D14, and D16) by its excess loss, the day's quantile score above the
substation's median day. It labelled the worst 5% of days that overlap a switching event, a
holiday, a cold snap, a flagged reading, or an Economy 7 step. Of its 72 worst days, 37 carry no
label. We re-ran the forecast that experiment 172 scored (experiment 155's arm D) for those days
with past weather in place of the forecast weather. The past weather is ERA5 (the ECMWF
reanalysis) temperature and Copernicus Atmosphere Monitoring Service (CAMS) solar irradiance, and
the forecast weather is the mean of the ECMWF ensemble forecast. We also counted how many of the
other Primaries of [experiment 87](../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)
were in their own worst 5% of days on the same day. Past weather removes 31.5% of the unlabelled
days' excess loss. The other 68.5% splits into 38.9% on days that another Primary also has and
29.5% on days that no other Primary has, so the three shares add to 100% of the excess loss. Other
Primaries share these days about as often as random shifting of their days gives, so the remaining
loss could come from events at one substation.

**The pre-registered verdict is `partly supported`.** The weather condition failed: 31.5% against
the required 50% (95% interval 21.9% to 39.8%). The shared condition passed its stated threshold:
48.6% of the unlabelled days coincide with another Primary's worst-5% day, against one third
required. The shifting control shows that this condition carries little information. Shifting
each other Primary's days separately gives 46.6% on average. Shifting all of them together, which
keeps their own worst days on the same days, gives 35.5%, and 7.8% of those draws reach 48.6%. ERA5
arrives about 5 days late, so the past-weather forecast cannot be issued in practice. ERA5 and
CAMS are themselves estimates, so 31.5% is a lower bound for the weather share.

![Share of the unlabelled days' excess loss that is weather, shared, or neither](fig1_split.svg)

![Each unlabelled day's excess loss with ensemble weather and with past weather](fig2_days.svg)

![The share of unlabelled days that another Primary also has, against random shifting](fig3_chance.svg)

## Results

- **Weather share W (headline):** 0.315 of the 37 unlabelled days' excess loss (total 108.1 points of
  day score), 95% interval 0.219 to 0.398 from resampling the 11 calendar months that hold an
  unlabelled day. By substation: D5 0.60, D14 0.45, D10 0.28, D11 0.20, D16 0.16. Leaving one
  substation out gives 0.28 to 0.35.
- **Labelled comparison set:** the 35 labelled worst days (total excess 188.0) lose 0.063 of their
  excess to past weather.
- **Weather share sensitivities:** 0.319 with the ensemble arm's quantile-band widths kept (the
  quantiles shift only), 0.340 with the Kalman filter also run on the ERA5-driven residuals, 0.367
  with the ensemble arm's median day as the reference, and 0.292 on the mean forecast's absolute
  error. All sit under 0.5.
- **Shared share S:** 0.486 (95% interval 0.279 to 0.704) of the unlabelled days have at least one
  other Primary in its own worst 5% (mean 0.84 others). The labelled days give 0.657 (mean 1.26).
  Leaving one substation out gives 0.42 to 0.56. The 16 series of experiment 87 sit on 15 Primaries
  (two series share one Primary), and a series on the target's own Primary does not count.
- **Chance control:** each other Primary's worst-day series is moved by a random whole number of
  days, 1,000 times. Moved separately, the share averages 0.466 (95% interval 0.297 to 0.649) and
  47.5% of draws reach 0.486. Moved together by one amount per draw, it averages 0.355 (0.189 to
  0.568) and 7.8% of draws reach 0.486. Interconnected neighbours (25 interconnected pairs among
  the 15 Primaries) share 16.2% of the unlabelled days, against 15.6% when moved separately.
- **Split of the unlabelled days' excess:** weather 31.5%, shared 38.9% (interval 20.6% to 56.9%),
  neither 29.5% (15.9% to 48.4%). The shared part is the excess that stays with past weather on a
  day another Primary also has. Chance alone would give a share of days about this large.
- **Known-answer controls:** 10 points added to 10 random days per substation. The pre-registered
  bar was 90% of the 50 days recovered. Added to the observed demand of both arms, 100% of the days
  keep their excess (weather share under 0.5; mean share 0.03). Added to the ensemble arm's
  forecast only, 92% of the days lose it (share at least 0.5; mean 0.89). Both pass, so the controls
  do not cap the verdict.
- **Unlabelled days by month:** every month but February 2026 holds at least one; November 2025
  and May 2026 hold 6 each. The four January 2026 days (all at D16) lose 3.0 of their 18.4 points
  of excess to weather (16%), and all four coincide with another Primary.

## Checks

- **Median forecast minus mean forecast, per site:** 0.0 points at all six substations in the past-
  weather arm, as in experiment 172.
- **Mean forecast's error beside the quantile score:** the share computed on the mean forecast's
  absolute error is 0.292, against 0.315 on the quantile score, so both point the same way.
- **Audit triggers:** the plan listed seven audit triggers (`ml-experiment` skill, "Audit before
  merge"). The reproduction control matches: the ensemble arm's day table equals experiment 172's,
  with a maximum score difference of 0.0. The trigger for a result outside its predicted range
  fired, because W came in below the predicted 40% to 70%. The explanations tested are in the
  sensitivities above: the filter's state carried over from the ensemble-driven runs (0.340 when
  rerun on ERA5), the spread (0.319), and the reference median (0.367). None reaches 0.5, so no
  audit was requested. The other six triggers did not fire.

## Method and caveats

- **Row set:** `fold_a_daily_gap_v1`, the scored year of experiment 155, arm D, D+1 band (lead 15 to
  39 hours), hand-filtered switching rows excluded, days of at least 24 rows, as in experiment 172.
- **Arm D:** experiment 136's monthly T-all fits (a convex demand model) driven by the calibrated
  ensemble mean, plus a Kalman filter on recent errors, experiment 155's winter night offset, and
  its 14-day mean-error profile.
- **Past-weather arm:** the fits and the stored Kalman level and anomaly stay. The temperature,
  irradiance, and PV shape come from the feature cache's ERA5 and CAMS columns at each valid time.
  Experiment 155's offset and profile and the spread are rebuilt from the new forecast's own errors.
  The stored Kalman state came from ensemble-driven residuals and may partly absorb ensemble-weather
  error, which would understate W; the 0.340 sensitivity reruns the filter on ERA5 residuals.
- **Other Primaries:** experiment 87's BG2 forecasts (the production-style XGBoost blend) cover 16
  series of 15 Primaries, including the five decision substations and D9. BG2 has no switching
  mask, so the hand-filtered switching rows stay in its scores. Topology is the interconnection
  table NGED supplied; only D-labels appear here.
- **Controls:** the injected controls act on the day table and do not rebuild experiment 155's
  correction. The weather-side control adds error to the ensemble arm only, so it tests the
  day-level split and not the ERA5 pipeline.
- **One year:** the scored year has one winter, and shares are approximate. The bootstrap resamples
  the 11 months that hold an unlabelled day.

## What to try next

- Read the 13 unlabelled days that keep more than 1.5 points of excess with past weather and have no
  other Primary in its worst 5% against the substation's own readings and NGED's switching logs, to
  find a further label.
- Test whether weather matters more at D5 and D14 (W 0.60 and 0.45) than at D11 and D16 (0.20 and
  0.16), which would point to site-specific weather sensitivity.
