# Embedded solar behind six Lincolnshire Primary substations is 2.5 to 4.5 MW, and grew 7% to 16% a year

**Bottom line: in the season-year September 2025 to August 2026, the embedded solar behind six of
the eight directional Primary substations in our trial area had a capacity of 2.5 to 4.5 MW, each
with a sampling interval of about plus or minus 9% to 17%.** A Primary substation is a substation that
steps power down to the local distribution network. At the four Primary substations with 6 or more
season-years of data, the estimate grew by 7% to 16% a year. Two further error terms are larger than
the sampling interval at some Primary substations, and the page keeps all three separate. The
estimate says nothing about a demand that rises with sunshine, because nothing in these data sizes
that load.

![Figure 1: capacity of the embedded solar behind each Primary substation in 2025/26](assets/embedded_pv_capacity_headline.svg)

**Can we say how much solar sits behind each Primary substation?** Yes for six, no for two.

- Six Primary substations have a capacity, with a 95% interval of about plus or minus 9% to 17% in
  2025/26. The widest interval is Trusthorpe's.
- Kirton does not: the test flagged a cloud-driven demand in 6 of its 7 season-years, so we give
  only the range of capacities the assumed demand shapes allow (0.8 to 2.9 MW in 2025/26).
- Wrangle T1 is left out, because its power is too small and noisy for the test.

**Is the growth real?** Probably, at the four long records.

- The fitted growth is 7.3% a year at Spilsby, 7.8% at Horncastle, 7.4% at Mount Bridge Boston, and
  16.1% at Donington (Figure 9).
- The growth rate holds only if any hidden demand stays the same size in megawatts from year to
  year. Nothing here tests that.

**Can the estimates be used for forecasting?** We do not recommend it.

- Experiment 00031 found that embedded-PV capacity estimates did not lower the all-lead score of
  Primary-demand forecasts, and the irradiance they rest on arrives too late for a live forecast.
- They are useful for explaining past demand and for ranking Primary substations by embedded solar.

**What do these data and methods support?** A table with three stated error terms, built from one
trial area, six solar farms, and one irradiance product. They do not support a figure for any
Primary substation outside the trial area, nor an estimate that is independent of the farm-based
ratios (see Limitations).

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Sonnet 5.5). Several independent Claude reviewers have checked the
> method, the evidence, and the prose adversarially.

## Key findings

- **The six-arm capacity C and the daily-fit capacity disagree in a way that separates Kirton from
  the rest.** At Kirton the daily fit is above C in all 7 season-years. At the 32 clean and
  low-power Primary-years it is below C in 30 ([Figure 7](#the-two-capacity-estimates-agree-except-at-kirton)).
- **A load on dull days, undetected, would bias the daily fit about 2.6 times as much as C.**
  Therefore C, not the daily fit, is the capacity ([Figure 6](#an-undetected-load-biases-the-estimates)).
- **In summer 2025/26 the raw data show midday power drawn 1.9 to 3.0 MW lower on clear days than
  on overcast days, but that gap is not a pure solar signal**
  ([Figure 4](#the-clear-day-dip-is-visible-without-a-model)).
- **Kirton's overcast-day excess at midday is larger in November and February than in summer, which
  solar alone would not produce**
  ([Figure 5](#kirtons-overcast-day-excess-is-larger-in-winter)).

## Introduction

**National Grid Electricity Distribution (NGED) meters its Primary substations but not the solar
panels behind them.** The embedded solar lowers the power a Primary substation draws from the
network on sunny days. The question is how many megawatts of solar sit behind each Primary
substation, how that number changed from 2019/20 to 2025/26, and how far each number can be
trusted. The answer is useful for explaining past demand and for ranking Primary substations. It
is not a forecast input (see Discussion).

This page builds a table from two earlier experiments, with no new fitting. [Experiment
00039](../ml-experiments/00039-pv-disaggregation-cams-injection/result.md) fitted embedded-solar
capacity to the demand of the Primary substations, using CAMS irradiance (the Copernicus Atmosphere
Monitoring Service's satellite estimate of sunlight), and checked the fit on solar-farm output added
to demand. [Experiment
00040](../ml-experiments/00040-pv-capacity-error-bars/result.md) built a test that compares six
versions of the fit, and attached an error bar to the capacity where the test finds no cloud-driven
demand.

## Data and methods

**Primary substations.** The eight Primary substations metered in signed megawatts, so that solar
above demand does not fold into positive demand. Eight further Primary substations metered in
megavolt-amperes are left out, because their folded meter turns solar into positive demand
([experiment 00039](../ml-experiments/00039-pv-disaggregation-cams-injection/result.md)).

**Six detector fits and a daily fit.** Each fit regresses a Primary substation's half-hourly power on
temperature, calendar terms, and CAMS-based solar output, and the six fits differ in how much
cloud-driven demand each allows. For solar output of capacity C, fit *a* returns about R*a* times C,
where R*a* is that fit's recovery ratio, measured by adding the output of six solar farms to the
demand. A cloud-driven demand breaks that proportionality, and the misfit T measures by how much.
The six-arm capacity C is the capacity that best matches the six fits to their ratios, and the daily
fit is a seventh fit on whole days. The farms are labelled A to F in the experiments, and none is
drawn here.

**The test and its flag.** A season-year is flagged when T exceeds the 95th percentile of T on
solar-farm output added to the same Primary substation's demand. A Primary substation is a flagged
Primary substation when more than half its season-years are flagged. Every season-year of a flagged
Primary substation is flagged, because an unflagged year there is more likely a missed detection
than a clean year.

**Intervals and the bound.** The 95% interval of C is the 2.5th and 97.5th percentiles of C over 200
resamples of the year's days in blocks of seven consecutive days. It holds the recovery ratios fixed
and ignores undetected loads, so it covers sampling noise in the days only. The bound B is the
median error of C when a day-long cloud-driven demand, undetected, is added at the largest strength
the test detects in fewer than 80% of draws. The strength grid is 0.1, 0.25, 0.5, and 1 times the
true capacity (`bound_amplitude` in the table). Status `clean` means the test flags a load of half
the capacity in at least 80% of resamples, and misses a load of a quarter of it (power 0.25 to 0.62
at a quarter), where the median error of C is 0.17 to 0.23. Status `low-power` means the test misses
a load of half the capacity as well, with the bound read at 0.5 or 1. Both rules were fixed after
the table of experiment 00040 was seen, so they are post hoc choices.

**Planned checks and exploratory numbers.** Four checks were written in the plan before any output
existed. P1: this page's C, flags, and bound reproduce experiment 00040's stored scores (they do,
to a difference of 0). P2: C is the headline capacity and the daily fit is a cross-check. P3: growth
rates are fitted per Primary substation with at least 6 season-years. P4: Kirton gets no capacity.
Every other number is exploratory. The page does no significance test and does not correct for
multiple comparisons.

## Results

### The capacity table

**For 2025/26, the table gives each Primary substation's capacity C, its sampling interval, the
bound on an undetected day-long load, and the daily fit as a cross-check.** The full table, one row per
Primary substation and season-year, is in [a CSV file](assets/embedded_pv_capacity_table.csv).
Bound B is read at the strength in brackets, as a multiple of the true capacity.

<!-- capacity-table-start -->

| Primary substation | Status | Capacity C (MW) | 95% interval (MW) | Bound B (strength) | Daily fit over C |
|---|---|---|---|---|---|
| Donington | clean | 4.50 | 3.94 to 4.92 | 0.18 (0.25) | 0.82 |
| Horncastle | clean | 4.21 | 3.86 to 4.59 | 0.21 (0.25) | 0.81 |
| Mount Bridge Boston | clean | 4.14 | 3.67 to 4.50 | 0.19 (0.25) | 0.93 |
| Trusthorpe | low-power | 3.11 | 2.55 to 3.63 | 0.47 (0.5) | 0.80 |
| Spilsby | clean | 3.06 | 2.73 to 3.32 | 0.21 (0.25) | 0.84 |
| Wrangle T2 | clean | 2.48 | 2.26 to 2.70 | 0.23 (0.25) | 0.92 |
| Kirton | flagged | none | range 0.8 to 2.9 | not given | 1.53 |

<!-- capacity-table-end -->

### The Primary substations lie within 55 km of each other

**Eight Primary substations, one trial area.** The largest distance between two of them is
54.7 km. Kirton and Donington are neighbours.

![Figure 2: map of the eight Primary substations](assets/embedded_pv_capacity_map.svg)

### The clear-day dip is visible without a model

**At the two Primary substations with the clearest record, the clear-day midday power drawn fell:
from 1.58 MW in 2019/20 to 0.58 MW in 2025/26 at Spilsby, and from 4.65 MW in 2020/21 to 3.29 MW
in 2025/26 at Donington.** The other four Primary substations have shorter records, or a smaller
change, so the dip is not deeper at every one.

![Figure 3: summer power drawn by time of day on clear and overcast days](assets/embedded_pv_capacity_profiles.svg)

**In summer 2025/26 the midday power drawn is 1.9 to 3.0 MW lower on clear days than on overcast
days.** The gap includes the solar output and the extra heating and lighting on overcast days, so
it overstates the solar. The gap rose from 1.09 to 3.02 MW at Donington and from 1.03 to 2.08 MW at
Spilsby. At Horncastle and Mount Bridge Boston it was 2.5 and 2.9 MW in 2019/20 and 2.6 and 3.0 MW
in 2025/26.

![Figure 4: summer midday power drawn on clear and overcast days, per season-year](assets/embedded_pv_capacity_gaps.svg)

### Kirton's overcast-day excess is larger in winter

**At Kirton the overcast-day excess at midday peaks at 2.67 MW in winter and 2.09 MW in summer. At
Donington it is 1.93 MW and 1.92 MW.** Solar output is far lower in winter, so a larger winter gap
points to a cloud-driven demand. This is why Kirton gets no capacity. The winter sample is thin:
at 53 degrees north the sun stays below 15 degrees in December and January, so only November and
February days can be classed, and the pooled sample is 30 clear and 33 overcast days at Kirton
(26 and 31 at Donington). Cold overcast days also raise demand at both Primary substations, so the
comparison shows a difference and not a proof.

![Figure 5: overcast minus clear power drawn, Kirton and Donington](assets/embedded_pv_capacity_kirton.svg)

### An undetected load biases the estimates

**An undetected load that is the same on every daylight half-hour of a dull day, at 25% of the true
capacity, makes C 18% too high and the daily fit 48% too high.** A load that rises with sunshine
does the opposite, making C 34% too low and the daily fit 29% too low. The test misses both kinds,
and these error sizes come from noise-free simulations over 218 injected cases.

![Figure 6: bias of the capacity estimates against the size of an undetected load](assets/embedded_pv_capacity_bias.svg)

### The two capacity estimates agree except at Kirton

**The daily fit divided by C has a median of 0.92 over the 32 clean and low-power Primary-years
(10th to 90th percentile 0.81 to 0.98), and exceeds 1 in 2.** At Kirton it has a median of 1.30
and exceeds 1 in all 7 years. A day-long cloud-driven demand would push the ratio above 1, so a
median of 0.92 is evidence against a large such demand at the clean Primary substations. The
alternative reading is that the farms' daily-fit ratio overstates a rooftop fleet's by 8% to 20%.
The data cannot tell the two readings apart. This check was looked at after the table of
experiment 00040, so it is post hoc.

![Figure 7: daily-fit capacity against six-arm capacity](assets/embedded_pv_capacity_crosscheck.svg)

### Adding the fitted solar back shows the demand it hid

**In Donington's clearest summer week, metered power plus the fitted solar output averages 7.8 MW
from 11:00 to 14:00 and 6.5 MW from 19:00 to 22:00.** If the fitted capacity of 4.5 MW is right,
demand at Donington is higher at midday than in the evening. If the capacity is too high, the sum
shows an artificial midday bump. The figure cannot decide between the two.

![Figure 8: one clear week in summer and one in winter, with fitted solar added back](assets/embedded_pv_capacity_weeks.svg)

### The capacity grew 7% to 16% a year

**The log-linear fit of C against season-year gives 7.3% a year at Spilsby (95% interval 3.1% to
14.1%), 7.8% at Horncastle (3.2% to 16.1%), 7.4% at Mount Bridge Boston (2.8% to 10.7%), and 16.1%
at Donington (8.5% to 23.7%).** The interval resamples season-years. An interval from the
bootstrap draws alone, which holds the sampling noise only, is narrower: 4.1% to 11.0% at
Spilsby, and 13.7% to 19.4% at Donington. The daily fit gives 8.6%, 6.0%, 8.2%, and 16.5%, so the
choice of estimator moves a rate by at most 2 points. Donington's rate rests on a rise from 3.2 to
4.5 MW in the last season-year; without that year the rate is 12.0%. Six points per Primary
substation make each interval wide, and the fitted line is a summary and not a forecast. Low-power
season-years are in every fit, and the status of a year flips between clean and low-power as the
power estimate changes (Spilsby's 2021 is low-power, its 2022 clean), so hollow dots alternate
along one trend.

![Figure 9: capacity per season-year and fitted growth](assets/embedded_pv_capacity_trend.svg)

## Discussion: what to use

**Use the table to explain past demand and to rank Primary substations by embedded solar and its
growth.** The 2025/26 capacities are 2.5 to 4.5 MW, with the six-arm interval of about 10%. If a
Primary substation's capacity matters to a decision, quote its three error terms and not one
combined interval.

**Do not use the table as a forecast input.** Experiment 00031 found no all-lead gain from
embedded-PV capacity estimates, and CAMS irradiance arrives too late for a live forecast.

**The error terms stay separate.**

1. **Sampling.** The 95% interval of C is about plus or minus 10% (the 10th to 90th percentile of
   the half-width over C is 8% to 21%).
2. **Transfer from solar farms to rooftop fleets.** The recovery ratios come from six metered solar
   farms. C over truth on injected farm output has a median of 0.99, with a 10th to 90th percentile
   of 0.84 to 1.10. One farm's median is 0.83 and the others' are 0.96 to 1.05, and the daily fit
   over C of 0.92 at the clean Primary substations says a rooftop fleet may differ from the farms by
   about 10%.
3. **Undetected demand.** The bound B on C is 0.18 to 0.23 at the five clean Primary substations in
   2025/26 and 0.47 at Trusthorpe. A load that rises with sunshine would make C too low, by 14% at
   a strength of 0.1 and 34% at 0.25. Nothing in these data sizes either kind of load.

**The one check that would test the table against truth is a comparison with outside statistics.**
The Embedded Capacity Register and the national small-scale solar statistics list capacity by
area. The comparison needs each Primary substation's service area from NGED.

## Proposed follow-up study

**Compare the capacities with the Embedded Capacity Register by service area.** This study found a
table whose three error terms are all stated. Only an outside source can test the transfer from
farms to rooftops, and only a prior from it could size Kirton. The cost is about half a day of agent
time and no download, but it needs NGED's service areas for each Primary substation, which the
maintainer would request. The case against doing it: the register lists installed capacity (DC),
and our capacity is a 99th percentile of AC output, so the comparison needs a conversion that adds
its own error.

## Limitations

- **One trial area, six solar farms, one irradiance product.** The recovery ratios come from six
  farms in Lincolnshire, and CAMS irradiance drives every fit.
- **The capacity is the 99th percentile of half-hourly AC output**, not the installed DC capacity,
  which is larger.
- **Detection power is low for a day-long load, even at a clean Primary substation.** In experiment
  00040 the test flagged such a load at 25% of the true capacity in 0.35 of simulated draws (median
  over the eight Primary substations). In this table the power at that strength is 0.37 to 0.49 at
  the five clean Primary substations in 2025/26, and 0.21 at Trusthorpe.
- **Kirton, Wrangle T1, and the MVA Primary substations are left out**, for the reasons above.
- **Status rules are post hoc.** Their thresholds were fixed after the table of 00040 was seen.
- **The capacity table rests on stored fits of experiment 00040.** This page refits nothing, so
  every flaw of those fits carries over. The tables were built on 2026-10-16 from the stored arm
  vectors.
- **The overcast-day figures use few days.** There are 4 to 15 overcast days a summer.
- **A fleet of rooftop panels with the same orientation as the farms' is assumed.** The estimate
  moves with the fleet's orientation mix, which none of these data measure.

## Scope

The page covers the eight directional Primary substations in the trial area, from 2019/20 to
2025/26. It says nothing about other Primary substations, battery storage, or the weather products
other than CAMS irradiance.

## Data and code availability

The stored arm vectors and the CAMS-based frame come from experiments 00039 and 00040, whose inputs
are NGED's private telemetry and the public CAMS irradiance. The data are private. The study code
is in [`studies/embedded_pv_capacity/`](https://github.com/openclimatefix/nged-substation-forecast/tree/main/studies/embedded_pv_capacity),
and the shared test is in `packages/studies/src/studies/pv_capacity_misfit.py`. No solar farm is
named or drawn. The code is at the commit of the pull request that merged this page.

## Reproducing the figures

```bash
export POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
uv run python studies/embedded_pv_capacity/build_capacity_table.py
uv run python studies/embedded_pv_capacity/build_raw_gaps.py
uv run python studies/embedded_pv_capacity/capacity_charts.py
```

The scripts print every number on this page into `report.md` and `raw_gaps_report.md` in
`data/studies/per_study/embedded_pv_capacity/`.
