---
id: "00039"
slug: pv-disaggregation-cams-injection
theme: solar-disaggregation
also_relevant_to: []
status: done
size: medium
hypothesis: >-
  On the 8 directional (MW) Primary substations, a calendar-plus-temperature demand term minus one
  PV shape driven by Copernicus Atmosphere Monitoring Service (CAMS) irradiance recovers the P99
  capacity of real solar-farm output injected into the real demand series to within 25%, and the
  same fits on the un-injected series pass four checks that a weather-correlated demand confound
  would fail.
prediction: >-
  Median capacity ratio R (recovered over injected P99) for M1 with CAMS at the substation is 0.77
  (95% interval 0.70 to 0.83). The daily fit M1d raises it by 0.08, CAMS at the farm raises it by
  0.08, and the richer demand term M1L lowers it by 0.04. Half-hourly nMAE is 0.12. On the real
  series K1 passes at 8 of 8 Primaries in 2025/26, K2 at 7 of 8, K3 fails at Kirton and probably
  at Wrangle T1, and K4 is unknown. The expected verdict is `separates`, narrowly. The scratch
  numbers behind this prediction are exploratory and were not blind.
decision_rule: >-
  Four planned contrasts on medians over (Primary, farm, season-year) units, with 95% two-way
  cluster-bootstrap intervals (Primaries and farms resampled independently). C1 is R of M1 with
  CAMS at the substation (the headline). C2 is R of M1d minus C1. C3 is R of M1 with CAMS at the
  farm minus C1. C4 is R of M1L minus C1. M1d replaces M1 when C2 is at least +0.05 with an
  interval above zero, because M1d costs nothing extra. On the better of M1 and M1d, recovery
  passes when the median R is in [0.75, 1.25], its 95% interval lies inside [0.65, 1.35], and the
  median nMAE is at most 0.15. The real checks pass when at least 6 of the 8 Primaries pass K1 to
  K4 in their latest season-year and K1 passes at all 8. Verdicts: `separates` (both pass);
  `recovers-but-confounded` (recovery passes, real checks fail); `biased-correctable` (median R in
  [0.5, 0.75) or (1.25, 1.5], 10th to 90th percentile spread under 0.20); `fails` (median R under
  0.5 or nMAE over 0.25 at both irradiance points). A verdict never rests on a breakdown. The
  experiment stops without a verdict if the known-answer self-tests or the timestamp scan fail.
  There is no cost-of-porting threshold, because the aim is explanation and capacity estimation, and
  not a production forecast column.
benchmark: >-
  None. The scored quantity is recovery of a known injected capacity. The ±25% band is the
  maintainer's ranking prediction HD1, fixed before any of these fits.
row_set_id: none
metric: >-
  Capacity ratio R (recovered over injected P99), half-hourly nMAE of the recovered PV series as a
  share of the injected P99, and the share of Primary-years passing each real check
verdict: partly supported
separation_verdict: recovers-but-confounded
best_arm: >-
  M1d (daily fit) with CAMS at the substation: median recovery ratio 0.854 (95% interval 0.815 to
  0.866), against 0.773 for the half-hourly fit M1
headline: >-
  C1, the half-hourly fit M1 with CAMS at the substation, recovers a median 0.773 of injected
  capacity (95% 0.740 to 0.818) over 218 units. The daily fit M1d recovers 0.854 (0.815 to 0.866).
  On the real series 6 of 8 Primaries pass checks K1 to K4 in 2025/26, and K1 fails at Wrangle T1.
lessons: >-
  Injected solar-farm output is recovered to within 25% of the true capacity, with a shortfall of
  15% to 23% that the planned contrasts attribute partly to noise in the CAMS irradiance and the
  distance to the farm (C2, C3), and not to the demand. The real series do not all behave like true
  PV: Kirton fails the season check (K2) in 5 of 7 years and the demand-richness check (K3) in 4 of
  7, which means part of its fitted capacity responds to cloud as demand does.
advice:
  - >-
    Divide capacities fitted with M1 by 0.77, and those fitted with M1d by 0.85, before quoting
    megawatts of embedded solar, because injection recovers that share of the true capacity (C1,
    C2).
  - >-
    Fit daily sums (M1d) rather than half-hours when only a capacity is wanted, because M1d
    recovers 0.08 more (+0.04 to +0.10) at no extra cost (C2).
  - >-
    Do not read a capacity at Kirton as PV alone until the demand model is richer, because adding
    a cloud-demand term cuts its capacity by 20% to 60% and its winter fit exceeds its summer fit
    by 29% to 54% in 5 of 7 years (figures 15 and 16).
  - >-
    Treat injection as a test of the PV shape and the irradiance only, because a linear fit makes
    the recovered increment identical to the fit of the injection alone, so it cannot reveal a
    demand confound (self-test c).
  - >-
    Use CAMS irradiance at the substation for estimation, because moving it to the farm adds only
    0.08 (+0.04 to +0.10) and the farm's location is not known for unmetered solar (C3).
  - >-
    Leave out the 8 MVA Primaries from a capacity estimate, because their meter folds PV
    into positive demand and Stickney fits a negative capacity in every year (figure 18).
production_status: not-for-production
adopted_in: none
code: ml-experiments/00039-pv-disaggregation-cams-injection/
---

# Experiment 39: can a CAMS-driven fit separate embedded solar from Primary substation demand?

We asked whether the output of solar farms embedded in the distribution network can be separated
from the demand at a Primary substation (a substation that steps power down to the local
distribution network) by fitting a model to the demand and a satellite estimate of sunlight. We
added the recorded output of real solar farms to the demand of the 8 Primary substations that meter
power in one direction, and checked how much of the added capacity the fit found again. The fit
found a median of 77% of the added capacity, and its daily version found 85%, over 218 units (every
Primary, solar farm, and season-year). We then ran the same fit on the real series with four further
checks. Our rule needed 6 of the 8 substations to pass all four checks and the first check, which
fits a placebo sky, to pass at all 8. Six of the 8 passed all four, but the placebo-sky check failed
at one substation (Wrangle T1).

**Verdict `recovers-but-confounded`: injected solar-farm output is recovered to within 25%, but the
real series pass the four checks at 6 of 8 Primaries, and the pre-registered rule needs K1 to pass
at all 8.** A fit of a calendar-plus-temperature demand term minus one CAMS-driven PV shape recovers
a median 0.773 of the capacity of real solar-farm output injected into the demand of the 8
directional Primary substations (95% interval 0.740 to 0.818). The daily version of the fit recovers
0.854 (0.815 to 0.866) and passes the recovery rule. The aim is explanation and capacity estimation,
not forecast skill. Neither CAMS irradiance nor ERA5 temperature arrives in time to drive a live
forecast, so the fit describes past data only. CAMS is the Copernicus Atmosphere Monitoring
Service's satellite estimate of sunlight.

![Figure 1: Median recovery ratio by model and irradiance point, with 95% intervals](figure_01.svg)

| Contrast (medians over 218 units, 95% two-way bootstrap interval) | Prediction | Result |
|---|---|---|
| C1: M1, CAMS at the substation | 0.77 (0.70 to 0.83) | 0.773 (0.740 to 0.818) |
| C2: M1d minus M1, substation | +0.08 | +0.081 (+0.040 to +0.100) |
| C3: M1 at the farm minus M1 at the substation | +0.08 | +0.078 (+0.041 to +0.102) |
| C4: M1L (cloud term) minus M1, substation | -0.04 | -0.042 (-0.078 to -0.021) |
| Half-hourly nMAE, M1, substation | 0.12 | 0.119 (0.104 to 0.131) |
| nMAE, M1d, substation | | 0.105 (0.092 to 0.113) |

**The decision rule picks M1d.** C2 is above the +0.05 threshold and its interval is above zero, so
M1d is the model the rule scores. Its median ratio, 0.854, lies in [0.75, 1.25], its interval lies in
[0.65, 1.35], and its nMAE, 0.105, is below 0.15, so recovery passes. The 10th to 90th percentile
spread of the units is 0.763 to 0.902. The ratio does not depend on the injected size, because the
fit is linear. The known-answer self-tests passed, and the timestamp scan peaked at a shift of zero
(median ratio 0.850 at zero, 0.81 at plus or minus one half-hour, 0.73 at plus or minus two).

## Real series: 6 of 8 pass, K1 fails at Wrangle T1

**K1 to K4 are the checks that a demand confound would fail.** Injection cannot test them, because a
linear fit gives the same increment whatever demand it sits on. The expectations come from the
injected units: summer and winter fits differ by +0.011 of the capacity (K2), and adding the cloud
term multiplies the capacity by 0.937 (K3). The table shows each Primary in 2025/26.

| Primary | Capacity (MW) | K1 placebo sky | K2 season | K3 cloud term | K4 day type | All pass |
|---|---|---|---|---|---|---|
| Kirton | 2.81 | pass | fail (+0.54) | fail (0.41) | pass | no |
| Trusthorpe | 2.04 | pass | pass | pass | pass | yes |
| Spilsby | 2.22 | pass | pass | pass | pass | yes |
| Horncastle | 2.98 | pass | pass | pass | pass | yes |
| Mount Bridge Boston | 3.24 | pass | pass | pass | pass | yes |
| Wrangle T1 | 0.89 | fail | pass | fail (0.48) | fail (-0.58) | no |
| Donington | 3.16 | pass | pass | pass | pass | yes |
| Wrangle T2 | 1.87 | pass | pass | pass | pass | yes |

![Figure 14: K1: real capacity against 20 placebo fits](figure_14.svg)

**In 2025/26 two Primaries account for the failures.** Over all 41 Primary-years, K1 passes in 38, K2
in 30, K3 in 34, K4 in 40, and all four in 27; Mount Bridge Boston, Spilsby, and Horncastle also
fail K2 or K3 in single earlier years. Kirton fails K2 in 5 of 7 years (winter capacity 29% to 54%
above summer) and K3 in 4 of 7 (the cloud term cuts its capacity by 20% to 60%). Wrangle T1 has 2
season-years and a fitted 0.89 MW. It passes the first half of K1 (its capacity is above the 95th
percentile of the placebo fits, 0.163 MW) and fails the second by a hair: the median absolute
placebo capacity, 0.099 MW, is just above one tenth of 0.886 MW. Its weekend capacity differs from
its weekday capacity by 58%. **This page does not decide which of these failures is a demand confound
and which is noise from two years of data.**

![Figure 13: Fitted capacity per Primary and season-year](figure_13.svg)

**The fitted capacity is 0.9 to 3.2 MW per Primary in 2025/26 and rose at 4 of the 5 Primaries with
4 or more years.** Dividing by the M1 recovery ratio, 0.77, gives about 1.2 to 4.2 MW of solar.
Kirton is flat. Only 2 of 33 year-to-year changes are falls larger than 2 combined standard errors
(Donington 2021, Kirton 2022). The detection limit is about 2.5% of capacity per season-year: the
median day-clustered standard error is 1.2% of the Primary's effective capacity (figure 11).

## What else the injection showed

- **Summer and winter recover alike.** Injected PV gives 0.766 in April to September and 0.770 in
  October to March at the substation (figure 12), so K2 expects no seasonal gap.
- **Farm E is the worst case.** The median ratio by farm runs from 0.70 (Solar farm E) to 0.84
  (Solar farm D) at the substation (figure 10). Farms lie 16 to 24 km from the Primaries on average.
- **Detail is lost at the peaks.** The recovered half-hourly output has a correlation of 0.88 with the
  injected output at the substation point and 0.92 at the farm point (figure 9), and the energy
  recovered is 0.80 and 0.87 of the injected energy.
- **The 8 non-directional (MVA) Primaries fold.** Stickney fits a negative capacity every year, and
  the MVA Primaries' winter and summer fits differ widely (figures 15 and 18). They are left out of
  the verdict.
- **The bulk supply points are descriptive.** The fitted capacity at Boston and Skegness lies above
  the sum of the directional Primaries and metered farms below it, as it should (figure 19).
- **The Embedded Capacity Register comparison was not run.** The register code lives in experiment
  00031's folder, which a script may not import.

## What the next experiment should answer

**The design's rule for `recovers-but-confounded` is PVDIS-2B: does a richer demand model bring K3 to
pass without lowering the recovery below 0.75?** Add demand shapes (a flat cloud term, a dawn and
dusk term, temperature by hour) and the orientation menu in `studies.pv_separation.separate`, inject
a known dull-day demand term into the real series to measure its bias, and report K1 to K4 at
Kirton and Wrangle T1 first. The failures sit at two Primaries, so a result of the 2B kind could move
the count of Primaries that pass from 6 to 8, which the arc's stop rule counts as a move of 2.

## Figures

The first figure is above. Figures 2 to 5 show the raw data, 6 to 8 the method, 9 to 12 the recovery
of injected PV, and 13 to 19 the real series.

![Figure 2: Donington's midday power against CAMS irradiance, 2025/26](figure_02.svg)

![Figure 3: Midday mean power by calendar month, one line per season-year](figure_03.svg)

![Figure 4: Clear-sky minus overcast midday power](figure_04.svg)

![Figure 5: CAMS data dropped and power coverage](figure_05.svg)

![Figure 6: The unit PV shape through a summer and a winter week](figure_06.svg)

![Figure 7: Building an injection](figure_07.svg)

![Figure 8: Recovery in the same week](figure_08.svg)

![Figure 9: Recovered against injected half-hourly PV](figure_09.svg)

![Figure 10: Recovery by farm and distance](figure_10.svg)

![Figure 11: Detection limit](figure_11.svg)

![Figure 12: Summer and winter recovery](figure_12.svg)

![Figure 15: K2: winter against summer capacity](figure_15.svg)

![Figure 16: K3: capacity with and without the cloud term](figure_16.svg)

![Figure 17: K4: weekday against weekend capacity](figure_17.svg)

![Figure 18: A non-directional Primary folds PV](figure_18.svg)

![Figure 19: Bulk supply point against the parts below it](figure_19.svg)

## Method and files

- **Data.** Power is the cleaned half-hourly series of each Primary and of the 6 solar farms. CAMS
  irradiance at hours with reliability 0.9 or more is spread to half-hours by the cosine of the
  zenith. Temperature is ERA5 at each Primary. Season-years run September to August, 2019/20 to
  2025/26, and a window needs 3,000 daylight half-hours.
- **Injection.** The 218 units are every Primary, farm, and season-year with enough shared rows.
  The injected output is minus the farm's output scaled so that its 99th percentile over daylight
  half-hours is 20% (also 5%, 10%, and 40%) of the Primary's effective capacity.
- **Code.** `ml-experiments/00039-pv-disaggregation-cams-injection/`, run in the order `e00039_inputs`,
  `e00039_inject`, `e00039_real`, `e00039_report`, `e00039_charts`. `e00039_report` prints every number
  on this page.
- **Differences from the plan.** The farm's 99th percentile is taken over its daylight half-hours,
  the convention that matches the unit PV shape's, and not over all half-hours.
