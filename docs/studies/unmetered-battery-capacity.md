# A primary's half-hourly flow reveals a simulated unmetered battery of 10% of the flow, and no real battery we tested

**This study asks how well the half-hourly flow of a primary substation can reveal the power (MW)
and the energy capacity (MWh) of a battery that nobody meters.** The estimator knows the public
signals that a battery follows: a day-ahead price, a half-hourly retail price, and the fixed windows
of home-battery tariffs. It fits the battery's power, usable duration, and round-trip efficiency to
the primary's flow, and reports a probability distribution for the power and the energy. The page
tests the estimator first on simulated batteries with a known answer, and then on real batteries
that are not drawn from the estimator's own model.

**A simulated merchant battery that follows the day-ahead price in the estimator's own way is found
from about 10% of the primary's 99th-percentile flow. Real public batteries added to NGED's flows
are not found at any size we tested, and the intervals for a battery that follows another rule hold
the truth far less often than their stated 90%.** The false-alarm rate is 3 of 36 blocks with no
added battery (8.3%, 95% interval 1.8% to 22.5%), all three in June to August. Estimates of power
and energy are therefore trustworthy only where the real battery's dispatch matches the estimator's
assumptions, and no real battery tested has met that condition.

![Figure 1: Simulated batteries are found above a size; real public batteries are not
found](assets/unmetered_battery_capacity_headline.svg)

- **To find a merchant battery inside a primary, the battery needs about 10% of the primary's
  99th-percentile flow, and its dispatch must follow the day-ahead price the way the estimator
  assumes.** In-family sums are flagged in 49% at a 5% share and 93% at 10%. A rank-rule battery is
  flagged in 99% at a 40% share, yet its 90% power interval holds the truth in 25% ([Figure
  13](#batteries-from-outside-the-estimators-family-are-found-but-sized-wrongly)).
- **Do not read the MW or MWh interval of a real battery as calibrated.** On real public batteries
  at a 40% share the 90% power interval holds the truth in 6% of sums ([Figure
  15](#real-public-batteries-are-not-recovered)).
- **Do not use the eight-primary screen to say any primary holds a battery.** The real templates
  rank first of 13 for none of the 8 primaries, and the screen has no power to find a battery
  ([Figure 17](#the-primary-screen-cannot-find-a-battery)).
- **A fleet of home batteries on fixed tariff windows is not identifiable from a primary's flow.**
  The monthly baseline absorbs any schedule that repeats every day, and only a fleet on the Agile
  price is found, from a 20% share ([Figure
  14](#a-simulated-domestic-fleet-is-found-only-through-its-agile-homes)).

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Sonnet 5.5). One independent Claude reviewer has checked the
> design and the results adversarially; the other reviews of the code, the page, and the prose are
> still to come, as the [Limitations](#limitations) say.

## Key findings

- **The positive control passes its committed rule, but its intervals are narrower than 90% on
  calendar replicas** ([Figure
  10](#the-positive-control-passes-its-rule-with-intervals-narrower-than-90-on-replicas)).
- **False alarms are 3 of 36 blocks, all in June to August** ([Figure
  12](#with-no-battery-added-3-of-36-blocks-are-flagged)).
- **Intervals for a simulated merchant battery in the estimator's family hold the truth from a 5%
  share** ([Figure 11](#in-the-family-the-90-interval-holds-the-truth-from-a-5-share)).
- **A power and energy posterior beats a model-free step statistic, in the family only** ([Figure
  11](#in-the-family-the-90-interval-holds-the-truth-from-a-5-share)).
- **Outside the family, large batteries are flagged but sized wrongly** ([Figure
  13](#batteries-from-outside-the-estimators-family-are-found-but-sized-wrongly)).
- **Real public batteries and NGED battery A are not detected outside June to August** ([Figure
  15](#real-public-batteries-are-not-recovered), [Figure
  16](#nged-battery-a-is-not-detected-inside-its-bulk-supply-point)).
- **The primary screen has no power, and the grid estimator is overconfident** ([Figure
  17](#the-primary-screen-cannot-find-a-battery), [Figure
  18](#the-grid-estimator-is-overconfident-where-the-differentiable-one-is-not)).

## Introduction

**NGED needs to know how much storage sits behind its primaries, and no meter reports it for homes
and small businesses.** NGED's Embedded Capacity Register lists storage of 50 kW and above with no
energy capacity in MWh, and is silent below 50 kW. The [battery and solar
study](battery-pv-separation.md) showed why the half-hourly flow is hard to use: a battery schedule
that is free to take any shape absorbs demand noise as readily as a real battery's swings, so its
residual falls by the same amount for real output and for calendar replicas with no battery. This
study therefore gives the battery no free shape. Each battery's schedule is simulated from a public
signal, and only a few physical numbers are fitted.

**Frequency was reviewed and barely moves half-hourly means.** In the Elexon data for September 2025
to September 2026 the half-hourly minimum frequency fell below 49.7 Hz in 6 half-hours on 4 days, so
grid frequency is not used as a signal.

## Data and methods

**The study year is September 2025 to August 2026, split into four 3-month blocks.** The 9
demand-like series are seven NGED primaries metered in MW (labelled S1 to S3 and S5 to S8), one bulk
supply point (BSP2), and one grid supply point (GSP1). A primary metered in MVA is out of scope,
because MVA has no sign. NGED labels its primary series "Disaggregated Demand", and its definition
of that label is unverified. A series with embedded generation added back would change what the
solar columns mean.

**A planned contrast was written into the plan before any result existed, and every other number is
exploratory.** The five planned contrasts are C1 (false alarms), C2 (calibration), C3 (the posterior
against a step statistic), C4 (energy separately from power), and C5 (fleets of real batteries).
Intervals on errors, coverages, and rates resample whole demand series and whole batteries in a
two-level cluster bootstrap of 2,000 resamples, and rates also carry Clopper-Pearson intervals.
These intervals do not cover the month-to-month weather, so a detection rate is read per block and
per season. An exploratory row with no real effect has a nominal 5% chance of being statistically
significant at the 5% level, the number of such rows is unknown, and the page does not correct for
multiple comparisons.

### How the estimator works

**The estimator describes the primary's flow as a baseline, solar, and a few batteries whose
schedules follow public signals.** Four signals schedule the batteries:

- **A merchant battery** follows the N2EX day-ahead price, through a day-by-day linear programme (an
  optimisation of charge and discharge against the price, with a state-of-charge limit and a cap of
  1 or 2 cycles a day).
- **A home-battery fleet** follows fixed tariff windows (Intelligent Octopus Go 23:30 to 05:30,
  Octopus Go 00:30 to 05:30, and Octopus Flux with cheap import 02:00 to 05:00 and high export 16:00
  to 19:00) or the East Midlands Octopus Agile half-hourly price.
- **A commercial and industrial battery** discharges across the weekday red band (16:00 to 19:00),
  an assumption not checked against NGED's charging statement.
- **A baseline** of one daily profile per month and four solar fleet curves absorbs everything else.

**The estimator reports, for each class, a probability distribution for the power, the usable
energy, and the round-trip efficiency.** Figure 2 follows one simulated sum from its inputs to the
posterior.

![Figure 2: One simulated sum, step by step: the posterior interval holds the true power and
energy](assets/unmetered_battery_capacity_worked_example.svg)

**The battery is differentiable, so a graphics card fits it in minutes.** The estimator reads each
price-taking battery's schedule between precomputed linear-programme schedules (3,800 of them for
the merchant battery) and passes the schedule through a smooth state-of-charge recurrence. It then
fits the power, duration, efficiency, and cycle-cap weight by gradient descent (the PyTorch optional
`gpu` dependency group of `packages/studies`, with a fused Triton kernel for the recurrence),
polishes the fit, and approximates the posterior around the optimum (a Laplace approximation). The
likelihood is raised to the power `1 / tau`, where `tau` is the integrated autocorrelation time of
the residual, which widens every interval. The differentiable estimator fitted rung 1's 792 sums in
908 seconds and rung 3's 3,312 sums in 4,326 seconds on one RTX A6000.

![Figure 3: The differentiable battery follows the price through precomputed dispatches and fits
only a few numbers](assets/unmetered_battery_capacity_dispatch.svg)

**A detection is a log Bayes factor above a threshold.** The statistic compares the Laplace evidence
of the model with batteries against the model with none. The threshold for a series is the 95th
percentile of the other 8 series' statistics on blocks with no added battery, so no series sets its
own threshold. The statistic is calibrated only empirically, against 36 null blocks from one year.

**A coverage plot says whether an interval means what it says.** An interval is calibrated when its
share of truths held equals its nominal level, and Figure 4 shows how to read it.

![Figure 4: A coverage plot shows whether an interval means what it
says](assets/unmetered_battery_capacity_coverage.svg)

### The ladder

**Each rung makes the truth harder to match with the estimator's assumptions.**

1. **Rung 1:** a simulated merchant battery that follows the day-ahead price in the estimator's own
   way (the in-family best case), added to each demand-like series at seven shares from 0.5% to 40%
   of its 99th-percentile flow. Rung 1b uses two simulated truths from outside that family.
2. **Rung 2:** a simulated fleet of home batteries on the four tariffs.
3. **Rung 3:** 23 real public batteries and fleets of 2, 4, and 8 of them, added to the series.
4. **Rung 4:** NGED battery A inside its bulk supply point's flow, with 1 to 11 further copies of
   its output subtracted.
5. **Rung 5:** the eight primaries as they are.

**Battery sizes are shares of a series' 99th-percentile flow, and the noise unit makes them
comparable.** The noise unit is `sigma_step`, the robust standard deviation of the half-hour changes
of a series' residual with no battery. A battery's power divided by `sigma_step` says how large it
is against the noise of the series it sits in.

## What the data looks like

**The primaries differ in level, noise, and daily shape.** Figure 5 shows two weeks of each primary,
and Figure 6 shows the mean daily profile and the mean half-hour change. Primaries are substations,
so they are shown in MW under their study labels.

![Figure 5: The primaries differ in level, noise, and daily
shape](assets/unmetered_battery_capacity_primaries.svg)

![Figure 6: Daily shapes dwarf a battery; some primaries step near 05:30, the end of the cheap
windows](assets/unmetered_battery_capacity_profiles.svg)

**The two prices share a daily shape, and Agile's cheap slots move from day to day.**

![Figure 7: The two prices share a daily shape, and Agile's cheap slots move from day to
day](assets/unmetered_battery_capacity_prices.svg)

**A real battery of a few percent of a bulk supply point's flow is hard to see by eye.** The
half-hour changes of NGED battery A correlate at -0.29 with one bulk supply point's flow over the
study year, and at 0.07 or less with every primary and the other supply points. The correlation is
evidence of connection, not a network-topology record. Figure 8 normalises both series by their own
99th percentile and numbers the days.

![Figure 8: A real battery of a few percent of a bulk supply point's flow is hard to see by
eye](assets/unmetered_battery_capacity_battery_a_in_bsp.svg)

**A simulated battery is visible by eye at a 40% share and invisible at 2%.**

![Figure 9: A simulated battery is visible by eye at a 40% share and invisible at
2%](assets/unmetered_battery_capacity_visibility.svg)

## Results

### The positive control passes its rule, with intervals narrower than 90% on replicas

**The positive control passes the rule committed before its run, but its intervals fall short of
their nominal level on calendar replicas.** The 90% interval holds both the merchant power and the
merchant energy in 202 of 280 replica fits (72.1%), against the 187 the rule required. The truths
are 10 fresh draws, off the estimator's grid, on 7 series no earlier diagnostic touched, in all 4
blocks, at a 40% share. On the replicas the 90% power interval holds the truth in 78.2% of fits and
the energy interval in 73.9%. The median absolute power error is 0.36%. This control replaces an
earlier pass that followed tuning on a scored block, which the [Limitations](#limitations) describe.

![Figure 10: The positive control passes its rule, but its intervals are narrower than 90% on
replicas](assets/unmetered_battery_capacity_positive_control.svg)

### With no battery added, 3 of 36 blocks are flagged

**The false-alarm rate is 8.3% against a nominal 5%, and the three false alarms fall in June to
August.** Planned contrast C1 holds: a one-sided exact binomial test against 5% gives p = 0.268. The
false alarms are BSP2, S1, and S7 in June to August, and a detection rate is therefore read
separately for June to August and for September to May. The cause of the June-to-August cluster is
not isolated.

![Figure 12: With no battery added, 3 of 36 blocks are flagged, all in June to
August](assets/unmetered_battery_capacity_false_alarms.svg)

### In the family, the 90% interval holds the truth from a 5% share

**For a simulated merchant battery that follows the estimator's own model, the 90% power interval
holds the truth in 100% of sums at shares of 5% to 20%, and the posterior beats a step statistic.**
Planned contrast C2 fails when pooled over rungs 1 and 2: the 90% interval holds the power in 68.8%
of sums (at least 80% was required). The pooled figure hides the spread by rung and share. Below a
2% share the posterior equals the prior. Planned contrasts C3 and C4 hold in the family: the mean
difference in relative power error (posterior minus step statistic) is -0.39 (95% interval -0.45 to
-0.32), and in relative energy error (posterior minus a rule of 2 hours times the power) -0.61
(-0.77 to -0.46). Both are statistically significant at the 5% level, for in-family batteries only.
The aggregate teaches the estimator the duration: the 90% width of the duration is 0.29 to 0.35 of
its prior width at a 10% share, and 0.08 to 0.09 at 40%.

![Figure 11: The 90% interval for a simulated merchant battery holds the truth from a 5% share; a
domestic fleet's does not](assets/unmetered_battery_capacity_calibration.svg)

### Batteries from outside the estimator's family are found but sized wrongly

**A battery that follows another rule is flagged about as often, but its interval misses the
truth.** The rank rule charges in each day's cheapest and discharges in its dearest half-hours. The
noisy price adds independent noise of standard deviation 0.2 to the price before the dispatch. The
rank rule is flagged in 75%, 96%, and 99% of sums at shares of 10%, 20%, and 40%, but its 90% power
interval holds the truth in 90%, 61%, and 25%, and its median absolute power error stays near 30%.
The noisy price is flagged in 53% at a 40% share, with 6% coverage and a median absolute power error
of 76%. At least half of the in-family sums are flagged from a battery of 2 noise units (4 outside
June to August), the rank-rule sums from 4, and the noisy-price sums from 32. No bin of real public
batteries up to 32 noise units has half of its sums flagged (21% at 32).

![Figure 13: Outside the estimator's family, large batteries are found but their size is
wrong](assets/unmetered_battery_capacity_outside_family.svg)

### A simulated domestic fleet is found only through its Agile homes

**The fixed tariff windows are not identifiable, because the monthly baseline absorbs a schedule
that repeats every day.** Detections of a simulated fleet come from the Agile unit, whose schedule
follows a price that changes daily. The fleet is flagged in 69% of sums at a 20% share and 94% at
40%, and in at most 17% at shares of 10% and below. Moving the windows one hour early leaves the
detections, and moving the prices as well brings the flags to 1 of 36 at every share. The 90% power
interval of a whole fleet holds the truth in 0% to 33% of sums (planned contrast C2 for fleets
fails).

![Figure 14: A simulated domestic fleet is found only through its Agile homes, and not below a 20%
share](assets/unmetered_battery_capacity_fleets.svg)

### Real public batteries are not recovered

**Adding a real public battery to a primary barely moves the estimate.** The posterior median of the
merchant power, as a share of the series' 99th-percentile flow, is 2.7% with a real battery of 5%
and 4.1% with one of 40%, against 2.6% with none. Outside June to August, 51 of 2,484 sums (2.1%)
are flagged, below the nominal 5%. The estimator also fails on calendar replicas, where almost no
demand noise remains: the posterior median is still about 6% of the registered power. The real
dispatch therefore lies outside the estimator's family, and demand noise is not what hides the
batteries. Planned contrast C5 is degenerate: the posterior median lies below the coincident peak in
82.1% of the 2,160 fleet sums and below the registered sum in 92.9%, so its pass is no evidence that
a fleet is identified as its coincident peak.

![Figure 15: Real public batteries added to NGED demand are not
recovered](assets/unmetered_battery_capacity_real_batteries.svg)

### NGED battery A is not detected inside its bulk supply point

**From September to May, NGED battery A is not detected at any multiple of its metered output up to
11.** The log Bayes factor stays between -19.5 and -16.8 and 0 of 15 blocks are flagged. In June to
August every block is flagged, including the matched null with the battery's metered output added
back. The posterior median power lies between 0.74 and 1.84 times the battery's metered
99th-percentile output, against a truth of 0 to 11 times. The result is for one site.

![Figure 16: NGED battery A is not detected inside its bulk supply point, even at 11 times its
output](assets/unmetered_battery_capacity_battery_a.svg)

### The primary screen cannot find a battery

**The screen asks whether the real template set ranks first among 13 sets, and it does for none of
the 8 primaries.** The 12 placebo sets move the tariff windows by 1 to 3 hours in either direction
and take the prices from 6 other weeks. The placebos are not exchangeable with the real set, so 1 in
13 is not the chance rate. Run on 54 simulated lanes, the real set ranks first in 5 (9.3%), no more
often than 1 in 13 (7.7%), so the 13-set screen cannot find a battery. A screen of the real set
against its 6 price-shifted rivals ranks the real set first in 54 of 54 simulated merchant lanes, in
0 of 9 lanes with no battery, and in 16 of 36 lanes holding a real public battery at 40%. By that
screen no primary of the 8 ranks first. The table gives each primary's result, and whether NGED's
register lists connected storage of 50 kW or more at the primary.

| Primary | Rank of the real set among 13 | Rank among the 7 price-differing sets | Register |
|---|---|---|---|
| S1 | 5 | 4 | connected storage |
| S2 | 10 | 7 | connected storage |
| S3 | 10 | 7 | no storage |
| S4 | 7 | 4 | no storage |
| S5 | 6 | 3 | no storage |
| S6 | 10 | 7 | no storage |
| S7 | 6 | 2 | no storage |
| S8 | 8 | 5 | accepted storage |

![Figure 17: The 13-set primary screen cannot find a battery, and by the price-only screen no
primary stands out](assets/unmetered_battery_capacity_screen.svg)

### The grid estimator is overconfident where the differentiable one is not

**The grid estimator places the duration and the efficiency on a fixed grid, and its intervals miss
the truth when the truth lies between grid points.** On the same 756 rung 1 sums, its 90% power
interval holds the truth in 11.1% of sums at a 40% share, against 99.1% for the differentiable
estimator. The grid estimator's own positive control failed. Its medians are no more accurate than
the differentiable estimator's: the mean difference in absolute relative power error (differentiable
minus grid) is +0.04 (95% interval -0.03 to +0.14).

![Figure 18: At a 40% share the grid estimator's 90% interval holds the truth in 11% of sums; the
differentiable estimator's in 99%](assets/unmetered_battery_capacity_grid.svg)

**The second setting gives the same verdicts.** It doubles the duration priors' spread and widens
the efficiency prior. C1 is 3 of 36 again, and C2 pooled is 67.4% (power, 90% interval). C3 is -0.39
(-0.45 to -0.33), and C4 is -0.70 (-0.85 to -0.54). The three starts of the gradient descent agree
in all but 2 of 5,868 sums with an interval.

## Discussion: what to use

**A detection of a battery in a primary's flow should be read as "something follows the day-ahead
price", not as a size.** The estimator flags a battery above about 10% of the primary's flow only if
the battery's dispatch resembles the estimator's. For a real battery, planning should rely on
connection records and metering, which the register's 50 kW floor and missing MWh limit, and treat
unregistered batteries as uncertainty near the evening peak and at the tariff window edges. A longer
record, a battery-free reference series, or a known battery inside a primary would change this.

**The coverage numbers say which intervals to distrust.** A 90% power interval from this estimator
holds the truth at most 25% of the time for a rank-rule battery at a 40% share, and 6% for a real
public battery. No MW or MWh interval for a real battery should be published from this estimator.

## Limitations

**The scripts have had no diff review, and the page has had one science review.** The study has not
yet had the diff review, the mutation pass over the changes to `packages/studies/`, the second
science review, the prose review, or the persona reviews that the `study` skill requires before a
study merges.

**The tuning history is optimistic.** A first learned dispatch failed the positive control. The fine
linear-programme stack was refined until the control passed, guided by diagnostics on a scored
block, so the first pass was optimistic. The clean control replaces it with a rule committed before
the run, on 7 series no diagnostic had touched. Tuning used S6 in September to November and S2 in
all blocks only.

**The branch changes a dependency.** The optional `gpu` group of `packages/studies` adds PyTorch and
Triton, and a spreadsheet reader converts the register once. A change to a dependency needs a human
review before the branch merges.

**The scored-block history matters.** The first positive control passed in 2 of 3 scored blocks on
one truth near a node of the stack, after tuning on a scored block, and it is superseded. The
Jun-Aug cluster of false alarms is unexplained, and the threshold rests on 36 null blocks from one
year.

**Other limits.** The half-normal prior for a class's power scales with the 99th percentile of the
aggregate including the battery, so at large shares the battery widens its own prior. The energy
reference for a real battery is a lower bound on its capacity. The simulated batteries follow a
perfect-foresight daily linear programme, and real batteries also earn from frequency response, the
Balancing Mechanism, and intraday trading, which the estimator does not model. The domestic
durations and the red band are unverified, and the definition of "Disaggregated Demand" is
unverified.

**What is not tested.** A real domestic fleet (none is known), electric-vehicle chargers on the same
tariff edges, a battery behind the meter, a primary metered in MVA, a primary outside NGED's East
Midlands licence area, a year other than September 2025 to August 2026, and whether MWh is
identified separately from MW for any real battery.

## Scope

**The page covers 9 demand-like series from one year and says nothing about other networks.** It
answers how well the half-hourly flow reveals a battery under the estimator described, and does not
estimate how much storage NGED's primaries hold.

## Data and code availability

**The prices and the public batteries' outputs are public.** Contains Balancing Mechanism Reporting
Service (BMRS) data © Elexon Limited, copyright and database right 2026. The N2EX day-ahead prices
are National Energy System Operator (NESO) data under the NESO Open Data Licence, and the East
Midlands Agile rates come from Octopus Energy's public API. NGED's primary and bulk supply point
telemetry and the Embedded Capacity Register extract are private, and no generator's name,
identifier, or coordinates appear on this page. The scripts are in
`studies/unmetered_battery_capacity/`, the shared machinery is in `packages/studies/src/studies/`,
and the results are under `data/studies/per_study/unmetered_battery_capacity/`, which are not
committed. The estimator runs on PyTorch with a Triton kernel on one RTX A6000.

## Reproducing the figures

Set `OMP_NUM_THREADS=2` for each command, and run the stack builds with `OMP_NUM_THREADS=1`.

```bash
uv run python studies/market_downloads/fetch_agile.py
uv run python studies/unmetered_battery_capacity/convert_register.py
uv run python studies/unmetered_battery_capacity/capacity_inputs.py
uv run python studies/unmetered_battery_capacity/capacity_stacks.py
uv run python studies/unmetered_battery_capacity/capacity_stacks.py coarse
uv run python studies/unmetered_battery_capacity/capacity_templates.py
uv run python studies/unmetered_battery_capacity/capacity_positive_control_clean.py
uv run python studies/unmetered_battery_capacity/capacity_rung1.py
uv run python studies/unmetered_battery_capacity/capacity_rung1b.py
uv run python studies/unmetered_battery_capacity/capacity_rung2.py
uv run python studies/unmetered_battery_capacity/capacity_rung3.py
uv run python studies/unmetered_battery_capacity/capacity_rung3_replica.py
uv run python studies/unmetered_battery_capacity/capacity_rung4.py
uv run python studies/unmetered_battery_capacity/capacity_rung5.py
uv run python studies/unmetered_battery_capacity/capacity_screen_power.py
uv run python studies/unmetered_battery_capacity/capacity_sensitivity.py
uv run python studies/unmetered_battery_capacity/capacity_grid_comparison.py
uv run python studies/unmetered_battery_capacity/capacity_report.py
uv run python studies/unmetered_battery_capacity/capacity_charts_data.py
uv run python studies/unmetered_battery_capacity/capacity_charts_explainers.py
uv run python studies/unmetered_battery_capacity/capacity_charts_results.py
```

The numbers that the figure scripts derive are written to `report_figures_explainers.md` and
`report_figures_results.md` beside the report. The helper modules `capacity_inputs.py`,
`capacity_runs.py`, and `capacity_report_tools.py` hold shared code and are not run on their own.
