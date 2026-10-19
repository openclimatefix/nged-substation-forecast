# Solar disaggregation

**At the 5 Primary substations with a clean solar-capacity estimate, giving the XGBoost demand
forecast a forecast of the output of the solar panels behind the substation lowered its daylight
April-to-September error by 0.15 points of capacity in one validation year, and 0.08 points of that
survive when the reference XGBoost forecast also starts from zero.**

A [Primary substation's](../../glossary.md#primary-substation) meter reads demand minus the output
of the solar panels connected below it, and NGED does not meter those panels. A least-squares fit of
calendar-and-temperature demand minus a capacity times a satellite-based sunshine shape recovers
most of the capacity of solar-farm output that we added to real series. The forecast gain depends
on the sunshine forecast: a term built from the true sunshine lowers the error by about a fifth,
and a term built from the ECMWF ensemble forecast gains little beyond 2 days ahead.

![Quantile score of each arm and the planned contrasts](../../00064-pvdis-5-net-demand-as-demand-minus-forecast-solar/figure_1_headline.svg)

**Arm `am` (the forecast solar output, given to XGBoost as a [base
margin](../../glossary.md#base-margin)) scores 4.467 against 4.620 for `d0`, the [best
guess](../../glossary.md#best-guess) of experiment 33, and every arm scores better than its own
placebo twin.** Source: [experiment
64](../../00064-pvdis-5-net-demand-as-demand-minus-forecast-solar/result.md). Scores are [quantile
scores](../../glossary.md#quantile-score-and-pinball-loss) in points of
[capacity](../../glossary.md#capacity-and-points-of-capacity); lower is better. The rows are
daylight half-hours of April to September at the 5 Primaries
([D9, D10, D11, D14, and D16](../../glossary.md#d-codes)), from July to September 2025 and April to
June 2026.

## Does it work?

**A simple fit finds most of the solar capacity that we inject, and the capacity is close to
unbiased on pure solar output.** On the 8 Primaries metered in both directions, the daily fit
recovers 0.854 (95% interval 0.815 to 0.866) of the capacity of real solar-farm output added to the
demand, and the half-hourly fit 0.773 (0.740 to 0.818). Six of the 8 Primaries pass four checks on
their real series ([experiment 39](../../00039-pv-disaggregation-cams-injection/result.md)). A
misfit test gives a median capacity of 0.992 of the truth on pure solar output (0.951 to 1.032),
but detects a load that follows the day's mean cloud in only 0.35 of trials ([experiment
40](../../00040-pv-capacity-error-bars/result.md)).

**An explicit solar term lowers the error of the XGBoost forecast, and the sunshine forecast limits
the gain.** Built from the true sunshine, the term lowers the score of `d1l` (`d0` without its
radiation inputs) by 0.966 (97.5% interval -1.431 to -0.595). Built from ENS sunshine with a
coefficient fixed at 1, it lowers it by 0.056 (-0.191 to +0.064), short of the 0.35 threshold
([experiment 46](../../00046-embedded-solar-share-of-primary-forecast-error/result.md), refuted).
With a weight fitted per series, lead band, and season, and the term given as a base margin, the
score of `d0` falls by 0.152 (98.75% interval -0.199 to -0.111), and by 0.272 (-0.420 to -0.145)
under 48 hours. With monthly refits the explicit term gains 0.149 (99.17% interval -0.201 to
-0.104) and three same-time sunshine analogues gain 0.095 (-0.185 to -0.007); the two are not
separated, and the verdict is inconclusive ([experiment
74](../../00074-an1-same-time-analogues-vs-explicit-solar-forecast-demand/result.md)).

**Part of the gain beyond 10 days is XGBoost starting from zero, and a solar term still helps.** At 10
days and beyond the fitted weight is zero, yet `am` scores 0.082 below `d0`. A refit with a base
margin of zero and no solar term reproduces the gap ([experiment
76](../../00076-pvdis-5-leak-check-beyond-10-days/result.md)). Against a `d0` that also starts from
zero, `am` is 0.082 lower over all leads (-0.113 to -0.052; post hoc, 95% interval; [experiment
78](../../00078-pvdis-5-current-capacity-vintages/result.md)).

![Contrasts and starting-point reads](../../00078-pvdis-5-current-capacity-vintages/figure_1_contrasts.svg)

**A zero start did not lower the score of the production-style forecasts.** For Primary
substations refitted weekly it changed the score by -0.017 (98.75% interval -0.072 to +0.054), and
every other planned difference had an interval that included zero ([experiment
80](../../00080-bm0-zero-base-margin-xgboost-quantile-models/result.md)).

**Neither the panel shape nor a fresher capacity is where the next gain lies.** A fitted blend of
east-, south-, and west-facing shapes changed the score of `am` by -0.001 (98.33% interval -0.018 to
+0.014) ([experiment 77](../../00077-pvdis-6b-fitted-solar-shape-in-arm-am/result.md), refuted). A
capacity re-estimated before every forecast run lowered it by 0.018 (-0.036 to -0.004), under the
0.05 threshold (experiment 78, small gain).

**A weekly capacity series, the panel mix, and a loss monitor were not shown to work.** A weekly
capacity fit that penalises falls more than rises recovered none of a 30% fall and gave 0.66 false
steps per Primary-year, against a limit of 0.5 ([experiment
45](../../00045-pvdis-4-weekly-capacity-asymmetric-l1/result.md), refuted). The east-south-west
shares were not identified: the median share error with noise was 0.200 against a limit of 0.15,
although the blend predicted held-out blocks 1.3% better than all-south panels ([experiment
73](../../00073-pvdis-6-orientation-mix-and-dc-ac-ratio/result.md), partly supported). A change-point
monitor of a solar farm's daily output found 0.84 of 25% losses lasting 90 days and 0.45 of those
lasting 30 days ([experiment 72](../../00072-p35-step-loss-monitor-for-solar-farms/result.md),
refuted).

**At the 8 Primaries metered only in [MVA](../../glossary.md#mw-and-mva), a fit recovers the solar
capacity but not reliably the direction of flow.** With the sign of the MW series hidden, the
capacity error was 0.09 to 0.11 of the signed fit's, and the balanced accuracy of the direction
0.78 to 0.81, below the 0.80 the rule needed. The real MVA Primaries showed almost no reverse flow
([experiment 79](../../00079-pvdis-7-mw-at-mva-only-primaries/result.md), partly supported).

**The data could not tell us** whether the gain holds in a second year, at Primaries without a clean
capacity estimate, or on rows outside April to September. The weights of experiment 64 rest on one
July-to-September period.

## Best design so far

**The best of the recipes tested in this theme gives XGBoost a solar offset built from a capacity
per season-year and a calibrated ENS sunshine anomaly.** The steps are:

1. Fit one capacity per Primary per [season-year](../../glossary.md#season-year), by least squares,
   and refresh it once per season-year ([experiment
   39](../../00039-pv-disaggregation-cams-injection/result.md), [experiment
   45](../../00045-pvdis-4-weekly-capacity-asymmetric-l1/result.md), [experiment
   78](../../00078-pvdis-5-current-capacity-vintages/result.md)).
2. Multiply the capacity by the calibrated ENS sunshine anomaly, with a weight fitted per series,
   lead band, and season on the training window and set to zero from 10 days ([experiment
   64](../../00064-pvdis-5-net-demand-as-demand-minus-forecast-solar/result.md)).
3. Give the product to XGBoost as a base margin, not as an input column or a shift after the fit:
   `am` beats the input-column arm by 0.093 (0.057 to 0.129; experiment 64).
4. Use an all-south panel shape ([experiment
   77](../../00077-pvdis-6b-fitted-solar-shape-in-arm-am/result.md)).

## Candidates for production

**Experiments 64 and 78 are `candidate`, and the live service cannot yet supply their inputs.**

- **The solar offset** ([experiment
  64](../../00064-pvdis-5-net-demand-as-demand-minus-forecast-solar/result.md)). Production would
  need a capacity table, a table of weights, the ENS calibration slopes at predict time, and a base
  margin set on each prediction matrix. The evidence is -0.152 against `d0` at 5 Primaries in one
  validation year.
- **Refresh the capacity table once per season-year** ([experiment
  78](../../00078-pvdis-5-current-capacity-vintages/result.md)). A weekly vintage differs from the
  season-year vintage by -0.005 (-0.013 to +0.003).
- **Not a candidate: a zero start.** Experiment 78 advised testing it, and experiment 80 found no
  gain.
- **The offset's gain inside the full forecast is small and not scoped to daylight.** In BG2 (a
  blend of XGBoost forecasts, with the offset at 5 of 16 Primaries) the offset lowers the score by
  0.010 (0.000 to 0.019) over all rows of the 16 Primaries. Only the 5 offset Primaries in daylight
  from April to September can change, and the read on those rows was not run ([experiment
  87](../../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)).

## Gotchas and implementation details

**Score a zero-start twin before crediting an input beyond about 10 days.** XGBoost's estimated
base score raised the score of `d0` by 0.071 over all leads (-0.099 to -0.047 for the zero start),
and the gain of `am` where its weight is zero is reproduced by a refit with a zero margin and no
solar term ([experiment 76](../../00076-pvdis-5-leak-check-beyond-10-days/result.md), [experiment
78](../../00078-pvdis-5-current-capacity-vintages/result.md)). Experiment 80 found the effect
absent for production-style forecasts. Re-check: the recipe, the correction, and the number of
rounds in use ([experiment 80](../../00080-bm0-zero-base-margin-xgboost-quantile-models/result.md)).

**Give every solar-term arm a placebo twin: the same arm given the sunshine of another date.** The
twin of `am` scores 0.300 worse (0.198 to 0.407), so the gain comes from the timing of the
sunshine. Shift the twin cyclically and do not permute it at random: a random permutation paired
runs that forecast the same valid time, so the permuted anomaly stayed correlated with the real one
([experiment 64](../../00064-pvdis-5-net-demand-as-demand-minus-forecast-solar/result.md),
[experiment 77](../../00077-pvdis-6b-fitted-solar-shape-in-arm-am/result.md)). Re-check: a different
sunshine source. These experiments ran no shuffled-capacity control.

**Report each contrast with and without the 14-day correction.** The [correction](../../glossary.md#14-day-correction)
carried the base margin's effect from shorter leads into leads beyond 10 days: `am` minus `am0` was
-0.035 corrected and +0.004 uncorrected ([experiment
76](../../00076-pvdis-5-leak-check-beyond-10-days/result.md)).

**Score a solar term on daylight April-to-September rows that have both a CAMS shape and an observed
power, and name the row set.** Experiments 46, 64, 74, 77, and 78 share 68,903 rows per seed.
Experiment 87 scored all rows of all 16 Primaries, so its 0.010 cannot be compared with 0.15
([experiment 46](../../00046-embedded-solar-share-of-primary-forecast-error/result.md),
[experiment 87](../../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)). Re-check:
a study that scores winter or night rows.

**Check a contrast's sign in each valid month before reading the pooled interval.** The ENS solar
term lowered the score of `d1l` in April, May, and September (-0.204, -0.232, -0.262) and raised
it in June, July, and August (+0.117, +0.183, +0.024) under a pooled interval of -0.191 to +0.064
([experiment 46](../../00046-embedded-solar-share-of-primary-forecast-error/result.md)). Re-check: a
validation period longer than 6 valid months.

**Use one capacity per season-year, which runs from September to August in these experiments.** The
differentiable-physics code runs it from July to June. The latest week of a real-time weekly
estimate had 3.63 times the path error of the same week in the final fit, and no penalty pair met
the limits ([experiment 45](../../00045-pvdis-4-weekly-capacity-asymmetric-l1/result.md)). Re-check:
a penalty set per Primary, which experiment 45 did not test.

**Do not treat a fitted capacity as a measurement.** The fit can absorb cloud-driven demand, and the
misfit test misses a load that follows the day's mean cloud. Run the placebo-sky, season, cloud-term,
and day-type checks on the real series, and inject more than one farm: the ratios of Solar farm E
left the capacity at 0.825 of the truth ([experiment
39](../../00039-pv-disaggregation-cams-injection/result.md), [experiment
40](../../00040-pv-capacity-error-bars/result.md)). Re-check: an independent register of installed
solar.

**Do not publish the fitted orientation shares.** They changed by a median 0.24 between consecutive
season-years, and the 1.3% fit gain did not reach the forecast (-0.001; [experiment
73](../../00073-pvdis-6-orientation-mix-and-dc-ac-ratio/result.md), [experiment
77](../../00077-pvdis-6b-fitted-solar-shape-in-arm-am/result.md)).

**Do not fit a signed model to an MVA-only series.** The magnitude folds reverse flow into positive
power, and the naive fit's capacity error was 0.29 to 1.00 ([experiment
79](../../00079-pvdis-7-mw-at-mva-only-primaries/result.md)). Re-check: a few weeks of signed power
at the Primary in question.

**Read coverage as well as the score.** The ENS solar term moved the share of sunny-row actuals below
the 0.05 quantile from 0.151 to 0.045, though the score barely changed ([experiment
46](../../00046-embedded-solar-share-of-primary-forecast-error/result.md)).

## What we have not settled

- Whether the gain holds in a second validation year and beyond the 5 Primaries with a clean capacity
  estimate.
- Whether a better sunshine forecast closes the gap to the true-sunshine oracle. The NOAA GEFS
  ensemble was refuted as a replacement for ENS, and the Met Office UKV blended with ENS gave a small
  gain in squared error ([experiment 50](../../00050-second-weather-forecast-for-solar-anomaly/result.md),
  [experiment 53](../../00053-ukv-vs-ens-solar-anomaly/result.md)).
- Why the true-sunshine term loses at sunny summer middays: the coefficient of 1 is too large, and
  the cause is not identified ([experiment 51](../../00051-pvdis-3b-midday-loss-of-the-solar-term/result.md)).
- Whether a solar term helps a joint demand fit ([experiment
  63](../../00063-sw2-10-linear-joint-fit-v2-in-miniature/result.md), [experiment
  104](../../00104-differentiable-physics-2-simple-structural-model/result.md)).

The [Research plan](plan.md) lists the proposed next experiments.
