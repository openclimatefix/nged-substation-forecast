# Production reference and best guess

**The forecast that is deployed today is far less accurate than a recipe we can already build, and
the best recipe so far (BG2) scores 5.26 points of capacity at 16 Primary substations, against 6.18
for NGED's `manual_heuristic` and 10.54 for a production-style model.** The score is the quantile
score: twice the mean pinball loss over 11 quantile levels, in points of capacity, where 1 point is
1% of a series' capacity and smaller is better. A Primary substation steps 33 kV or 66 kV down to 11
kV. The gain comes mostly from two cheap changes: a 14-day correction from the model's own recent
errors, and refitting every week. Only one validation year (July 2025 to June 2026) has been
scored, and several settings were chosen on that year, so the gains are probably optimistic.
Nothing from this theme has been adopted in production yet. The [Experiments](experiments.md) page
lists every experiment in the theme with its verdict.

## Does it work?

**The deployed forecast loses to a simple rule.** N1 is an XGBoost model per series, trained on the
control member of the European Centre for Medium-Range Weather Forecasts (ECMWF) ensemble forecast
(ENS), with power measured at the forecast origin as inputs. In [experiment
13](../../00013-primary-demand-against-stored-baselines/result.md), on the error of the median, the
production recipe was 1.69 points worse than `manual_heuristic`, which copies power from the same
time of day in earlier weeks (99% interval +0.98 to +2.59; 16 Primary substations). N1 did not beat
`manual_heuristic` with an interval that excludes zero (+0.22, -0.54 to +0.97), so that verdict was
inconclusive. Every arm forecast the validation year 1.3 to 3.1 points too low.

**A 14-day error correction supplies nearly all of the early gain.** The correction adds the mean
error of the previous 14 days to every quantile. In [experiment
19](../../00019-primary-demand-stack-vs-manual-heuristic/result.md) it lowered N1's quantile score
by 0.99 points, and the full stack (correction, 0.5-year recency half-life, spread factor 1.15)
beat `manual_heuristic` by 0.56 (99.44% interval -0.93 to -0.26). The half-life added nothing to a
frozen model, and the spread factor added 0.03 to 0.05 points. The correction was tested on runs
5 days apart, not daily.

**The best guess beats production by a wide margin.** In [experiment
33](../../00033-best-guess-architecture-vs-production/result.md) the best guess lowered the
quantile score below production's by 4.89 points for Primary demand (98.33% interval -5.66 to
-4.06), 3.81 for solar (-5.07 to -2.64), and 6.27 for wind (-7.65 to -5.28), each wholly below its
threshold. The correction accounts for 1.38 of the 4.89 Primary points. No arm widened production's
band, so the rest is not split between calibration and other causes. Monthly retraining without the
correction scored 5.12 on one seed against 5.32 for the best guess, which is a lead and not a
ranking.

![Quantile score of every headline arm, per family](../../00033-best-guess-architecture-vs-production/figure_1_headline.svg)

**Production's forecast bands are far too narrow, and the best guess fixes most of that.**
Production's 0.95 quantile was exceeded in 49% of Primary cases, 39% of solar cases, and 33% of
wind cases; the aim is 5%. The best guess exceeded it in 7%, 4%, and 7% (experiment 33).

![Share of cases above the 0.95 quantile](../../00033-best-guess-architecture-vs-production/figure_3_tail.svg)

**Weekly refitting and a five-model blend lower BG2's error again.** In [experiment
87](../../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md) BG2 beat the previous
best forecast by 0.283 points at the 16 Primary substations (98.75% interval -0.425 to -0.146) and
by 0.634 at the 4 bulk and grid supply points (-1.111 to -0.279). Refitting weekly instead of
monthly lowered the Primary score by 0.123 (95% interval 0.051 to 0.202) and the blend by 0.086
(0.038 to 0.135). The solar offset lowered it by only 0.010, far below the 0.10 to 0.15 predicted.

![The score after each added idea](../../00087-bg2-best-guess-assembled-with-drop-one-ablations/figure_3_staircase.svg)

**The solar and wind farm models cannot yet be ranked.** Training on the ensemble's own history
scored 0.050 points worse at 6 solar farms (98.75% interval -0.125 to +0.168) and 0.228 worse at 3
wind farms (-0.178 to +0.653) than experiment 33's mixture trained on ERA5, an estimate of past
weather. Neither interval rules out a loss above the margin we would accept (0.10 for solar, 0.30
for wind), so the verdict of experiment 87 is partly supported.

![Mean absolute error against forecast day](../../00087-bg2-best-guess-assembled-with-drop-one-ablations/figure_1_headline.svg)

**Better weather would add little for Primary demand.** In [experiment
22](../../00022-era5-weather-ceiling-primary-demand/result.md) ERA5 weather in place of ENS weather
lowered the score by 0.10 points (99.38% interval -0.16 to -0.04), nearly all from shortwave
radiation, and did not help within 2 days. ERA5 arrives about 5 days late, so this is a ceiling and
not a usable product. Experiment 33 found a larger ceiling for wind (5.46 points).

**MVA-only metering may not hurt the forecast, but 8 series against 8 cannot say.** In [experiment
56](../../00056-mva-only-primaries-skill-against-mw-primaries/result.md) the best guess had 5.6
percentage points less skill against `manual_heuristic` at sunny middays at the 8 Primaries metered
only in MVA than at the 8 metered in MW (95% interval -5.0 to 14.6). The verdict was inconclusive.

## Best design so far

**BG2 is the best design found, and it beats every earlier reference on the same runs.**

- **Primary substations:** a quantile blend of five models (weekly-refitted XGBoost, monthly-refitted
  XGBoost, an analogue model, a pooled-by-type model, and the frozen stack), with an explicit solar
  offset at the 5 substations that have a clean solar capacity. Score 5.260 against 5.543 for the
  stack ([experiment
  87](../../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)). The stack is N1 with
  the 14-day correction, the half-life, and the spread factor ([experiment
  19](../../00019-primary-demand-stack-vs-manual-heuristic/result.md)).
- **Bulk and grid supply points:** the weekly-refitted XGBoost model with no correction. Score
  12.925 against 13.559 for the stack (experiment 87).
- **Solar and wind farms:** no winner. The ERA5-trained mixture of experiment 33 and the
  ensemble-trained model of experiment 87 are within each other's intervals.

## What to implement in production

**Experiments 19, 33, and 87 are marked `candidate`; none is adopted.** Each would change what the
production forecast does.

- **Primary demand:** add the 14-day error correction and the spread factor of 1.15 to the
  single-stage ENS-trained quantile model. The evidence is experiments 19 and 33.
- **Primary and supply-point demand:** refit weekly instead of monthly, and serve the five-model
  blend at Primaries. The evidence is experiment 87.
- **Supply points:** serve the model without the correction (experiment 87).
- **Generators:** nothing is ready. The choice between the ERA5-trained mixture and the
  ensemble-trained model is unresolved, and the solar bands need recalibration because BG2's 0.05
  and 0.95 quantiles cover 0.028 and 0.973 of the actuals (experiment 87).

Experiments 13, 22, and 56 are `not-for-production`.

## Gotchas and implementation details

Each item names the experiment that taught it, what goes wrong without it, and what a later session
should check before relying on it.

- **Score every arm on the same cases.** [Experiment
  33](../../00033-best-guess-architecture-vs-production/result.md) scored each family on the cases
  that every arm holds, and each contrast on the cases of its own two arms. Without this, an arm
  with missing forecasts looks better or worse because it was scored on different rows. Check it
  again whenever a new arm has a different set of runs or forecast members.
- **Compare against `manual_heuristic` as well as production.** Production scored 9.52 against 5.89
  for `manual_heuristic` in [experiment
  13](../../00013-primary-demand-against-stored-baselines/result.md) and experiment 33, so a gain
  over production alone overstates how good a new model is. Check it again if production's recipe
  changes.
- **Do not apply the 14-day correction to supply points.** Adding it raised BG2's supply-point
  score by 0.305 (95% interval 0.060 to 0.594) in [experiment
  87](../../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md), and experiment 33
  found no gain there (+0.38, -0.12 to +0.92). Experiment 87 did not test the correction at farms.
  Check it again with weekly refits replaced by another schedule, or with a different supply-point
  set (only 4 series were scored).
- **Build the correction history from daily forecasts.** [Experiment
  19](../../00019-primary-demand-stack-vs-manual-heuristic/result.md) used runs 5 days apart, so
  past errors had leads of 0 to 5 days and were probably larger than a daily service would see.
  The size of the correction's gain may therefore differ in production. Check it again on daily
  runs.
- **Treat settings tuned on the scored year as optimistic.** The spread factor 1.15, the blend
  scheme, the offset gate, and the half-life were chosen or scored on the validation year in
  experiment 87, so its gains are probably too large. Check it again on a later validation year.
- **Read a staircase and a drop-one ablation separately.** Removing the blend gives 0.086 on the
  staircase and 0.091 in the ablation, because each removes it from a different forecast
  (experiment 87). Mixing the two misattributes the gain. Check it again whenever a new idea is
  added to the blend.
- **Report tail coverage and pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles.** The
  quantile score alone hid that BG2's solar bands are too wide (experiment 87), and production's
  band covers only 10% to 20% of cases (experiment 33). Check it again for any new weather source
  or technology class.
- **Confirm where the production-style row came from.** Experiment 87 read it from the local
  `data/power_forecasts` table and did not confirm which Dagster assets wrote it, so its score of
  10.54 may not match the deployed forecast. Check it again against the stored forecasts of the
  current production model.

## What we have not settled

- Whether the correction still helps when forecasts are made daily and the model is retrained
  monthly.
- Which training source to use for solar and wind farms, which needs more farms or a second year.
- Whether the solar offset helps on daylight rows from April to September.
- Whether the gains hold on a second validation year.

The [Research plan](plan.md) lists what is planned next.
