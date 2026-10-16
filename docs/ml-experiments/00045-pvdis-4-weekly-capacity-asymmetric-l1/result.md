---
id: "00045"
slug: pvdis-4-weekly-capacity-asymmetric-l1
status: planned
size: complex
hypothesis: >-
  On the 8 Primary substations metered in MW, a least-squares fit of the daylight power to a
  calendar-and-temperature demand term minus a weekly capacity times one CAMS-driven PV shape, with
  weekly capacity increments under an asymmetric L1 penalty (light on rises, heavy on falls),
  recovers a known injected capacity path (ramps, steps, and temporary falls, injected into a
  semi-synthetic baseline) with a median error of at most 0.15 of the path's mean capacity, beats a
  capacity that is constant within each season-year by at least 20% on ramps and steps, and
  produces at most 0.5 false steps of 10% of capacity per Primary-year on a placebo sky.
prediction: >-
  On held-out injections, the asymmetric fit reaches a median path error of 0.10 (0.08 to 0.14) of
  the mean capacity on ramps and steps, against 0.25 (0.18 to 0.35) for a free weekly capacity and
  0.17 (0.13 to 0.22) for the season-year constant. A free weekly capacity has a standard error of
  8% to 11% of capacity in June and July and over 30% from November to February. The smallest step
  detected in at least 80% of injections is 10% of capacity for a step placed from April to
  September at 6 of the 8 Primaries, and no step of 30% or less is detected from October to March
  at 5 or more Primaries. The placebo sky gives 0.3 false steps per Primary-year (0.1 to 0.6).
  Because rises are cheap and falls are dear, the path drifts upward on noise alone, by 2% to 6% of
  capacity over 5 years, so the ratchet-drift limit in the decision rule is the likeliest rule to
  fail. A temporary fall of 30% lasting 8 weeks is recovered at under 0.3 of its depth, and a fall
  lasting 26 weeks at 0.5 to 0.8. A bootstrap interval of 90% covers 0.82 to 0.92 of non-step weeks
  and 0.55 to 0.75 of step weeks. The latest week of a real-time vintage has a path error 1.3 to 2
  times the error of the same week in the final fit. The Department for Energy Security and Net
  Zero (DESNZ) small-scale solar statistics agree with the series within the stated tolerance at 4
  to 6 of the 6 comparisons. These numbers are priors from the argument that a season-year standard
  error of 1.2% implies about 8.7% per week, and from experiments 00039 and 00040. No weekly fit has
  been run.
decision_rule: >-
  Contrasts C1 to C5 are scored on the held-out injections, as medians over (Primary, farm, path
  seed) units, with two-way cluster-bootstrap intervals (Primaries and farms resampled
  independently) at the 99% level, which is a Bonferroni correction across the five contrasts. C1 is
  the path error of the asymmetric fit minus that of the free weekly fit, on ramps, steps of 10% or
  more, and unchanged capacity. C2 is the asymmetric fit minus the season-year constant, on ramps
  and steps of 10% or more. C3 is the asymmetric fit minus the symmetric L1 fit with the same rise
  penalty, in false steps per Primary-year on the placebo sky and in the share of a 26-week fall
  that is recovered. C4 is the asymmetric fit minus the strictly non-decreasing fit, in path error
  on temporary falls and on transfers out. C5 is the path error of the latest week of a real-time
  vintage minus that week's error in the final fit. Verdict `supported` needs all six conditions.
  (i) The median path error of the asymmetric fit on ramps and steps is at most 0.15 (the ceiling
  that experiment 00039 set for the half-hourly error). (ii) C2 has an upper limit below -0.20 of
  the season-year constant's error, because a weekly series that does not beat a yearly one by a
  fifth does not justify the penalised fit. (iii) The placebo sky and the unchanged-capacity
  injection both give at most 0.5 false steps of 10% of capacity per Primary-year at the median
  Primary and at most 1.0 at the worst Primary, because one false alarm in two years is the most a
  reader can discount. (iv) The ratchet drift, which is the median rise of the path from its first
  to its last year on unchanged-capacity injections, is at most 0.05 of capacity per 5 years,
  because the real growth that NGED wants to see is 5% to 15% a year. (v) A 30% fall lasting 26
  weeks is recovered at 0.5 or more of its depth and an 8-week fall at under 0.5, which shows that
  falls are held reluctantly and not forbidden. (vi) The coverage of the 90% interval at non-step
  weeks lies in [0.85, 0.95] and at step weeks is at least 0.60. Verdict `partly supported` when
  (i), (ii), and (iii) hold and one or more of (iv), (v), and (vi) fail; the write-up then states
  which penalty or interval change would be needed. Verdict `refuted` when (ii) fails, or when the
  false-step rate in (iii) exceeds 1.5 per Primary-year at the median Primary. Verdict
  `inconclusive` when a self-test fails, or when the smallest detectable step exceeds 25% of
  capacity at 5 or more Primaries in every season, because the weekly grid is then finer than the
  data support. The outside checks and the vintage limit (C5 within 2.0) are reported and never
  alter the verdict. The penalty pair is chosen on tuning units only, frozen in a committed file,
  and scored once on units that share no farm and no path seed with the tuning units. There is no
  cost-of-porting threshold, because the series is an offline table and the production forecast
  does not read it.
benchmark: >-
  The capacity of experiment 00040 held constant within each season-year (the six-arm capacity C,
  and the half-hourly fit M1 divided by its recovery ratio for like-for-like paths), and the free
  weekly fit with no penalty. The known injected path is the truth, so there is no forecast
  benchmark.
row_set_id: none
metric: >-
  Path error: the mean absolute difference between the recovered and the true weekly capacity path,
  as a share of the true path's mean. Also the false steps of 10% of capacity per Primary-year,
  the ratchet drift, the share of a temporary fall recovered, the smallest detectable step by
  season, the 90% interval coverage at step and non-step weeks, and the latest-week error of a
  real-time vintage.
production_status: not-for-production
adopted_in: none
code: ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/
---

# Experiment 45 (PVDIS-4): a weekly capacity series that rises easily and falls reluctantly

**Plan status.** This page is the plan, written by the Opus planner and committed with status
`planned` before any weekly fit has run. The sections after "Plan" are placeholders for the
write-up. The opening plain-language paragraph is written after the run.

## Plan

### The question and why it is worth asking

**The question is whether a weekly series of embedded solar capacity behind each Primary substation
can be estimated well enough to show real steps and no invented ones.** Experiments 00039 and 00040
gave one capacity per season-year, with a detection limit near 2.5% of capacity. NGED holds almost no
information on unmetered solar, and a capacity that changes week by week would show when a large
rooftop fleet or a farm below 1 MW connects. A free capacity per week would be too noisy. The
standard error of a season-year, 1.2% of capacity, implies about 8.7% per week on average (1.2%
times the square root of 52), and far more in winter, when the sun is low. A penalty on the weekly
increments lets neighbouring weeks share evidence, so that the path steps only where the data support
a step.

**The maintainer decided that capacity may fall but reluctantly.** The fit estimates the solar that
the Primary's meter sees, which can fall when home batteries charge from midday solar, when a feeder
is permanently reconfigured, or when a solar site is switched off for repairs. A strict
non-decreasing constraint would lower past capacity to fit a fall. The penalty on a fall is
therefore heavy and finite.

**The aim is explanation and a capacity series, not forecast skill.** Nothing here touches the
production forecast. Experiments 00007, 00009, 00022, and 00031 found small or no forecast gains from
solar inputs, and PVDIS-5 tests the forecast question using the vintages this experiment produces.

### Size, answered against the five triggers

The experiment is **complex**, because two of the five triggers fire.

- **What gets stored.** A private weekly capacity table and a table of real-time vintages under
  `data/ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/`, which SW2-10 reads. Nothing
  enters a production Delta table.
- **The production serving path.** Not touched.
- **A degradation rule.** Not touched.
- **More than one defensible design.** Yes. The estimator (half-hourly M1, daily sums, or the
  six-arm combination), the form of the penalty, the placement of the demand cells, and the
  bootstrap scheme each admit more than one defensible choice. The plan fixes each choice and
  gives the reason.
- **Code whose callers cannot be named without searching.** Yes, for one work item. The register
  loader moves from experiment 00031's folder into `packages/studies`, and that change is a
  separate reviewed pull request (work item W0).

The process is therefore the full one: this Opus plan, a Sonnet code review before the run, Sonnet
runs and write-up, a Sonnet check of the write-up against the printed report, and a Sonnet prose
review. Opus reviews the result when it decides the next part of the series.

### Earlier advice, followed and departed from

**The plan follows nine items of earlier advice and departs from none.**

- **Followed:** "Divide capacities fitted with M1 by 0.77" (00039): the series is divided by a
  recovery ratio measured on injected paths. "Do not read a capacity at Kirton as PV alone" (00039):
  Kirton keeps its status flag in every output. "Leave out the 8 MVA Primaries" (00039): the
  series covers the 8 Primaries metered in MW. "Treat injection as a test of the PV shape and the
  irradiance only" (00039): a penalised, constrained fit is nonlinear, so the plan scores the
  path difference and a semi-synthetic baseline, and adds an additivity check. "Leave Wrangle T1
  out of any claim that a Primary is free of a cloud load" (00040): the status column says so. "Do
  not quote an embedded-solar capacity without a stated bound" (00040): three uncertainty terms are
  stated and never summed. "Report C as the capacity and keep M1d divided by R as a cross-check"
  (the experiment 00040 review): the weekly series is checked against C. "Calibrate ENS irradiance
  to CAMS" (00031): the weekly series uses CAMS only, and PVDIS-5 owns the calibration. "Treat the
  register as a weak prior at Primaries" (00031): the register check is a soft check.

### Hypothesis and the four design choices

**The estimator is the half-hourly fit M1, with one capacity per week and one set of demand cells
per season-year.** The fit is

power = cell(season-year, month, local half-hour, day type) + b times heating degree-days - c(w)
times u + error,

where `u` is experiment 00039's unit PV shape from CAMS at the Primary, `c(w)` is the capacity of
week `w` (weeks start on Monday, Europe/London), and the cells are those of experiments 00039 and
00040. Keeping the cells per season-year means a slow rise in base demand cannot leak into the
capacity, and it keeps the recovery ratio of the weekly fit comparable with the season-year ratio.
The cost is that a step at a month boundary is partly absorbed by the cell mean, so the injections
include steps placed on month boundaries (risk 2).

**M1 is chosen over the daily fit and the six-arm combination for three reasons.** The daily fit
leaks a day-long cloud load at 3.4 MW per MW, against 2.2 for M1, so a weekly daily-sum fit has seven
points per week and the highest exposure. The six-arm combination needs a ratio vector and a
noise bootstrap for every arm at every week. The half-hourly fit recovers 0.77 of injected capacity,
which the series corrects by dividing by a measured ratio. The season-year mean of the weekly
series is checked against the six-arm C of experiment 00040 (check A0), so the series stays tied to
the estimate that carries a bound.

**The penalty is asymmetric L1 on weekly increments, with the capacity held non-negative.** The
objective is

0.5 times the squared residual, over the residual variance, plus `lambda_up` times the sum of the
rises plus `lambda_down` times the sum of the falls.

The problem is convex. The penalty is `lambda_up = a` times a Primary scale `s_P` and `lambda_down =
r` times `lambda_up`. The scale `s_P` is the median over weeks of the free weekly fit's standard
error, so that `a` is in units of one weekly standard error. The grid is `a` in {0.5, 1, 2, 4} and
`r` in {1, 3, 10, 30}, and the same `(a, r)` applies to every Primary, to stop one Primary's noise
from setting its own penalty. The comparison fits are `r = 1` (symmetric L1), `r = infinity` (strictly
non-decreasing), `a = 0` (free weekly), and the season-year constant. No convex-optimisation package
is installed, so the solver is a short ADMM routine in numpy, checked against `scipy.optimize.linprog`
in the self-tests (risk 13). The demand cells are projected out once per Primary, so the fit needs
only the weekly normal matrix (about 360 by 360) and one vector per series.

**The tuning rule picks one `(a, r)` pair on tuning units.** The pair minimises the median path
error over a mixture of ramps, steps, and unchanged capacity, subject to two limits: at most 0.5
false steps per Primary-year on the unchanged-capacity injections, and a 30% fall lasting 26 weeks
recovered at 0.5 or more of its depth. Ties go to the larger `a`. The pair is written to
`frozen_config.json` and committed before the scoring script runs, so the commit order proves the
scoring units played no part in the choice.

**The capacity unit is the 99th percentile of half-hourly AC output of the fleet, as in experiments
00039 and 00040.** Installed DC capacity is larger. The conversion to DC is part of the outside
check against DESNZ.

### Data, weeks, and masks

- **Series.** The 8 Primaries metered in MW (time series 7 Trusthorpe, 8 Horncastle, 10 Spilsby, 12
  Wrangle T1, 13 Wrangle T2, 16 Mount Bridge Boston, 17 Kirton, and 18 Donington), the two bulk
  supply points (descriptive), and the 6 solar farms as injection sources. The farms appear only
  as "Solar farm A" to "Solar farm F" (`studies.pv_dataset.pv_sites()` labels), and their output is
  always normalised by the farm's own 99th percentile, never published in megawatts.
- **Inputs.** The `frame.parquet` of experiment 00039 (cleaned power, CAMS unit shape and
  irradiance at reliability 0.9 or more, ERA5 temperature, zenith under 75 degrees), read from
  `data/ml-experiments/00039-pv-disaggregation-cams-injection/work/`. Reading another
  experiment's private output is allowed. Importing its code is not, so the model code is copied
  into this folder for the third time (risk 7).
- **Weeks.** 361 weeks from Monday 2019-09-02 to Monday 2026-07-27 (the last has 5 days). Each
  Primary starts at its first week with at least 3 usable days. A week with fewer than 3 usable
  days is `unobserved`: the fit has no data term there, the penalty interpolates, and the output
  marks the row and widens its interval.
- **Switching mask.** The logged closures, padded by 24 hours, are removed from every fit. The
  intervals come from the private table `data/ml-experiments/00038-switching-mask-value/
  intervals_d0.parquet` if its columns hold a start, an end, and a series, and otherwise from a
  copy of experiment 00037's log reader. SW2-0b's labels replace the logs once they exist. Unlogged
  transfers stay in the data and are injected as a case.
- **Polygons.** The check already run in `e00045_polygon_check.py` found 16 valid Primary polygons,
  overlaps under 0.3% of the smaller polygon, Mount Bridge Boston's point 95 m outside its
  polygon, and Wrangle T1 and T2 sharing one polygon. Wrangle T1 and T2 keep separate fits and are
  summed for any comparison against the shared polygon. The union against the two bulk supply
  points' areas is compared only if the metadata holds their polygons.
- **Validation before fitting.** Row counts per Primary-week, duplicate keys, the hour-of-day
  profile of the correlation of power with `u` (peak near 12:00 UTC), the share of CAMS hours
  dropped per week, the timestamp scan (the injected recovery must peak at shift zero), and the
  count of unobserved weeks. The `data-validation` skill lists the checks.

### Known-answer injection design

**Every scored result comes from injected paths whose truth is known, scored on units that played no
part in tuning.** The design has five parts.

- **The baseline is semi-synthetic.** For each Primary, the baseline series is the real power minus
  the full-record fitted PV component (so it holds demand, noise, and any confound), plus a known
  baseline PV path equal to the Primary's 2025/26 capacity from experiment 00040 times `u`. The true
  capacity of every week is then known, level included. A secondary scoring uses the real series
  with the injected path added, and compares the fitted path with and without the injection, because
  the production series is fitted to real data that already holds unknown solar. The gap between
  the two scorings measures the departure from additivity.
- **Injected change.** The injected output is minus the normalised output of one solar farm scaled
  by the true path, so that its 99th percentile equals the path's value in MW. Farm output is
  normalised, never published in megawatts.
- **Cases.** Each path is drawn with a fixed seed.
    - Ramp: a rise of 25% or 50% of the baseline over 52 weeks.
    - Step: a rise of 5%, 10%, 20%, or 30% at a random week, with half the steps placed in April to
      September, half from October to March, and one in four on a month boundary.
    - Temporary fall: 30% for 8 weeks and 30% for 26 weeks, and a 15% fall for 26 weeks.
    - Transfer: a share of 0.3 of another Primary's net power (including its solar) added for 5 days
      and for 30 days, once inside the 24-hour padded mask and once outside it, so that the cost of
      an unlogged transfer is measured. The truth is the share times the donor's baseline capacity.
    - Seasonal load: a daytime load (08:00 to 20:00 local) of 10% of the baseline capacity with a
      summer-weighted yearly cycle, once independent of the weather and once scaled by the clear-sky
      index (a load that rises with sunshine). The truth is the unchanged baseline, so the score
      is the seasonal swing that creeps into the capacity path.
    - Unchanged capacity: no injection, five seeds. Its false steps and ratchet drift are scored.
- **Hold-out.** The tuning units use Solar farms A, B, and C and path seeds 0 to 99. The scoring
  units use Solar farms D, E, and F and path seeds 1000 to 1099, at all 8 Primaries. Solar farm E,
  the worst case at 0.70 recovery, falls in the scoring set on purpose, and results are also
  reported without it. The recovery ratio of each Primary is measured on tuning units with a
  constant injected path and applied unchanged to the scoring units.
- **Intervals.** Two-way cluster bootstrap, resampling Primaries and farms independently, 2,000
  draws, at the 99% level for C1 to C5 and 95% elsewhere. With three scoring farms the farm margin
  is coarse, so the write-up also reports the seed-level spread.

**Real-time vintages are scored on the same semi-synthetic units.** The fit is repeated at origins
every 4 weeks over the last 3 years of the record, each truncated at its origin, and the latest
complete week of each vintage is scored against the truth. The write-up reports the path error of
that week against the same week in the final fit, and the revision of the first estimate by the
age of the week (1, 2, 4, 8, 13, and 26 weeks).

**The bootstrap is a residual block bootstrap that refits the whole path.** Each draw resamples
whole 7-day blocks of the penalised fit's residuals, taken from the same calendar month of any
year of the Primary's record (so that winter noise stays winter-sized), adds them to the fitted
values, and refits the path with the frozen penalties. Missing half-hours take a residual of zero.
Each Primary gets 500 draws for the real series and 200 for each injection unit. The 5th and 95th
percentiles give the 90% interval, and the 25th and 75th and the 10th and 90th give the calibration
curve. The coverage of the truth is scored on 96 scoring units (the case families with a step,
a ramp, a fall, and unchanged capacity), separately at weeks with no true step and at weeks within
2 weeks of a true step. A bootstrap of a penalised fit at a boundary tends to under-cover, and
the decision rule counts that as a finding. If coverage at non-step weeks is under 0.85, the
write-up reports an inflation factor chosen on tuning units, labelled post hoc.

**False steps are counted on two skies.** One is experiment 00039's placebo sky, which permutes
days within each month (200 permutations per Primary). The other is the unchanged-capacity
injection. A false step is a rise of at least 10% of the Primary's mean real capacity over any 4
consecutive weeks (5% is reported as well). The placebo count also reports the ratchet drift.

**The detection limit** of each Primary is the smallest step on the grid (5%, 10%, 20%, 30%) that
the path shows as a rise of at least half the step within 3 weeks of the true date in at least 80%
of the scoring injections. It is reported for April to September and for October to March
separately.

### Real series, three uncertainty terms, and the gap diagnostic

**The real weekly series is fitted on all of a Primary's data, then divided by that Primary's
recovery ratio.** The production-style output is the real-time vintage at weekly origins, so a
forecast date reads only what was known on that date.

**The uncertainty statement keeps three terms apart and never sums them.**

1. Sampling: the 90% residual-block-bootstrap interval of the weekly capacity.
2. Transfer from farms to rooftops: a scenario of plus or minus 10%, from experiment 00040's
   clean-Primary ratio of the daily to the six-arm capacity (0.92) and its per-farm spread.
3. Undetected demand: two named scenarios from experiment 00040's table, an undetected day-long
   cloud load (capacity too high by the Primary's bound B at amplitude 0.25) and a load that rises
   with sunshine (capacity too low by 33% at amplitude 0.25).

**The gap diagnostic compares the asymmetric path with the symmetric one on the real series.** A
Primary is flagged when the mean gap between the two paths is 5% of its capacity or more, or when
the symmetric path falls by at least 10% for 8 weeks or more and the asymmetric path does not. The
5% threshold is the weekly detection floor predicted for a summer step. The flagged falls are listed
for SW2-0b to check against switching labels. The write-up never presents a held fall as a real
fall.

**Status per Primary is carried from earlier results.** Kirton is `flagged` and its series appears
with a grey band and the library range of experiment 00040 (1.2 to 4.4 MW in 2025/26). Trusthorpe is
`low-power`. Wrangle T1 is `left-out` of any claim that a Primary is free of a confound. The others
are `clean`. Most coastal Primaries (Skegness, Ingoldmells, and Chapel St Leonards) are metered in
MVA and outside this experiment, so Trusthorpe is the one coastal Primary, and it is reported apart
from the inland Primaries. School-holiday weeks are compared with term-time weeks.

### Outside checks and tolerances, stated before any comparison

**Three outside checks and two internal ones are run, in order of strength. None alters the verdict.**

1. **DESNZ sub-regional statistics of small-scale solar by lower-layer super output area (LSOA).**
   Capacity and installation counts by LSOA are summed over each Primary's polygon, using the
   share of each LSOA's area inside the polygon (a population-weighted centroid assignment is the
   sensitivity check). The implementer confirms the exact DESNZ table, its snapshot date or dates,
   and whether it carries size bands, at download time. Installations of 1 MW or more are metered
   by NGED and already removed from the Disaggregated Demand series, so the sum uses the size
   bands below 1 MW where the table has them, and otherwise subtracts the register's connected
   solar of 1 MW or more in the same Primary. The expected peak AC output is the DC capacity
   times 0.70, with a range of 0.60 to 0.80 (experiment 00029 fitted farm DC-to-AC ratios of 1.12 to
   1.42, an AC-to-DC ratio of 0.70 to 0.89, and a fleet of mixed orientations peaks lower than one
   farm). **Tolerance:** the ratio of the series' mean over the 12 months to the snapshot date to
   the expected peak lies in [0.67, 1.5] at 5 or more of the 6 comparisons (Spilsby, Horncastle,
   Mount Bridge Boston, Donington, Trusthorpe, and Wrangle T1 plus T2 against the shared polygon).
   Kirton is compared and reported without a pass or fail. The factor of 1.5 is the root of the
   summed squares of the stated uncertainties: sampling (15%), farm to rooftop (10%), undetected
   load (25%), and conversion (14%), about 33%. The check can catch only a gross failure. DESNZ
   also under-counts installations after the feed-in tariff closed in 2019, so a ratio above 1 is
   less alarming than a ratio below 1. Where the table gives two snapshot dates, the growth between
   them is compared with the series' growth over the same dates: the signs must agree and the
   ratio must lie within a factor of 2.
2. **The Embedded Capacity Register (ECR) as a soft prior, not a floor.** The check uses only
   connected solar sites of 50 kW to under 1 MW, dated before each week, and first prints which date
   columns the register holds and what each records (the reader lists a `Last Updated` column,
   which is not a connection date). **Tolerance:** the register's total for a Primary does not
   exceed the series' upper 90% bound at 5 or more of the 6 comparisons. The register's median was
   0.08 of effective capacity against 0.33 from the data in experiment 00031, so the check expects
   the register to sit well below the series. The step-date check uses only sites larger than the
   Primary's detectable step and compares the series' rise in the 4 weeks around each date with
   the rise at 1,000 random placebo dates. It is run only if the register holds a connection date
   and at least 5 sites qualify across the six comparisons, and otherwise is reported as not
   testable.
3. **Solar mapped in OpenStreetMap** (`generator:source=solar` inside each polygon, ODbL licence,
   attribution required) is descriptive only, because rooftop coverage is uneven. It has no
   tolerance.
4. **Comparison with each bulk supply point (internal).** The weekly series is fitted at Boston and
   Skegness, and the bulk supply point's path should not fall below the sum of its directional
   Primaries' paths and its metered solar farms' known capacity by more than the intervals added
   in quadrature, in at least 90% of observed weeks.
5. **The season-year anchor (internal, A0).** The mean of the weekly series over each season-year
   lies inside experiment 00040's 95% interval of C at the clean Primaries in at least 80% of
   Primary-years.

**The DESNZ files need maintainer attention only on licence and size.**

- **LSOA boundaries:** Office for National Statistics (ONS) LSOA (2021) generalised-clipped
  boundaries and population-weighted centroids for England and Wales, from the ONS Open Geography
  Portal. The whole boundary file is expected to be tens of megabytes and the centroids a few
  megabytes (to be confirmed at download). The plan requests only the LSOAs that intersect the
  union of the 16 polygons (a few hundred features, under 5 MB) through the portal's feature
  service, and falls back to the full file.
- **DESNZ table:** a spreadsheet of a few megabytes.
- **Licence:** both are published under the Open Government Licence v3.0, which needs an
  acknowledgement and no account or signature. The ONS boundaries add "Contains OS data (c) Crown
  copyright and database right". The cost is nil, well under the £30 cap. The maintainer decides
  whether using an OGL file counts as accepting a licence in someone's name before the download
  runs.

### What SW2-10 reads

**Two private parquet files, with the columns below.** The folder is
`data/ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/`.

- `capacity_weekly.parquet`, the final fit on all data, one row per `time_series_id` and
  `week_start`: `time_series_id` (Int32), `primary_name` (String), `week_start` (Date, Monday,
  Europe/London), `capacity_mw` (Float64, the 99th percentile of the half-hourly AC output of the
  embedded solar, divided by the recovery ratio), `capacity_low_mw` and `capacity_high_mw` (the 5th
  and 95th percentiles of the bootstrap), `capacity_raw_mw` (before the division),
  `recovery_ratio`, `n_usable_days` (Int32), `observed` (Boolean), `step_flag` (Boolean, a rise of
  10% or more within 4 weeks centred on the week), `status` (`clean`, `low-power`, `flagged`, or
  `left-out`), `bound_dayL_relative` (experiment 00040's B for the Primary's season-year), and
  `scenario_dayL_mw` and `scenario_sunload_mw` (the capacity under each named systematic
  scenario, never merged into the interval).
- `capacity_weekly_vintages.parquet`, the same columns plus `origin_monday` (Date): the fit at each
  weekly origin using only data before that Monday. **A forecast date `d` reads the rows whose
  `origin_monday` is the last Monday on or before `d` and whose `week_start` is the latest week**
  (the Monday seven days before the origin), and holds that value flat for the days after it.

SW2-10 uses rows with status `clean` or `low-power`, uses the vintage file for any held-out
scoring, and treats `flagged` rows as unavailable. The path of the finished file goes to the
"Design: Switching event detection experiments" session when it exists.

### Cores, time, and parallel work

**The whole experiment fits in 4 cores and about 2.5 hours of wall time.** Each worker runs one
thread (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `POLARS_MAX_THREADS=1`), four workers at a
time, one output file per worker chunk written atomically, fixed seeds per job. Check `uptime`
first, because the slot's 4 cores are shared with PVDIS-3.

| Stage | Work | Workers | Estimate |
|---|---|---|---|
| Inputs and validation | read the frame, weeks, masks, figures 1 to 4 | 1 | 5 min |
| Self-tests | solver, linearity, known answers | 1 | 2 min |
| Tuning | 16 penalty pairs on about 1,400 tuning units (the normal matrix is reused, so about 0.1 to 0.3 s per fit) | 4 | 20 to 25 min |
| Freeze | write and commit `frozen_config.json` | 1 | 1 min |
| Scoring and contrasts | the chosen pair and the four comparison fits on about 1,400 scoring units | 4 | 15 to 20 min |
| Coverage bootstrap | 96 units times 200 draws | 4 | 15 min |
| Vintages on injections | about 38 origins times 8 Primaries times 12 units | 4 | 10 min |
| Real series | fits, 500-draw bootstrap per Primary, weekly-origin vintages, placebo sky (1,600 fits that rebuild the normal matrix) | 4 | 30 min |
| Outside checks | DESNZ, register, OpenStreetMap, bulk supply points | 1 | 10 min |
| Report and charts | printed report, about 20 figures | 2 | 10 min |

**What runs side by side.** Work items W0 (register move), U1 (inputs and raw figures), and U5a
(the download script) share no file and have no dependency on the solver, so three Sonnet agents
start together. The solver and self-tests (U2) come first for everything that fits. After U2, U3
(injection, tuning, scoring) and U4 (real series, vintages, placebo) run in parallel, because they
read the same solver and write different files. Within a run, the tuning units, the scoring units,
the Primaries, and the bootstrap draws are independent jobs.

### Work list, split into units a Sonnet agent can implement

Each unit is implemented by one Sonnet agent working from the files, and reviewed by a second Sonnet
agent that reads the code for leakage (tuning units reaching the scoring set), for the fit
stopping at the vintage origin, and for farm output reaching a chart in megawatts. No agent starts a
sub-agent.

| Unit | What it builds | Files | Needs |
|---|---|---|---|
| W0 | A separate reviewed pull request that moves the Embedded Capacity Register loader into `packages/studies`: the workbook reader without a spreadsheet library, the column renames, the Primary-name normalisation, and the connected-solar filter. The register path stays a parameter. Tests build a small workbook with `zipfile` and check the columns, the status filter, and the name normalisation. Experiment 00031's folder is switched to the import only if its saved register table is byte-identical afterwards. After an adversarial review, it may merge autonomously. | `packages/studies/src/studies/ecr.py`, `packages/studies/tests/test_ecr.py` | nothing |
| U1 | Inputs, weeks, masks, validation, and the raw-data figures 1 to 4 | `ml-experiments/00045-.../e00045_common.py`, `e00045_inputs.py`, `e00045_raw_charts.py` | nothing |
| U2 | The weekly design, the demand-cell projection, the ADMM solver for asymmetric L1 with non-negativity, the five fits (free, symmetric, asymmetric, strictly non-decreasing, season-year constant), and the self-tests | `e00045_model.py`, `e00045_selftest.py` | U1's common module |
| U3 | The injection builder (cases, semi-synthetic baseline, hold-out split), tuning, the frozen file, scoring, the contrasts, the coverage bootstrap, and vintages on injections | `e00045_inject.py`, `e00045_tune.py`, `e00045_score.py`, `e00045_bootstrap.py` | U2 |
| U4 | Real-series fits, recovery-ratio correction, the 500-draw bootstrap, weekly-origin vintages, the placebo sky, the gap diagnostic, and the two output files | `e00045_real.py`, `e00045_placebo.py`, `e00045_outputs.py` | U2, U3's recovery ratios |
| U5a | The download script for the LSOA files and the DESNZ table, following the `data-download` skill, with a generated README | `e00045_fetch_lsoa.py` | the maintainer's licence decision |
| U5b | The outside checks (DESNZ, register through `studies.ecr`, OpenStreetMap, bulk supply points, the season-year anchor) | `e00045_outside.py` | U4, U5a, W0 |
| U6 | The printed report (every number the page quotes) and figures 5 to 20 | `e00045_report.py`, `e00045_charts.py` | U3, U4, U5b |
| U7 | The write-up with its plain-language opening, the research-plan row and research-log entry, and the `mkdocs.yml` line | `docs/ml-experiments/00045-.../result.md`, `docs/ml-experiments/research-plan.md`, `docs/ml-experiments/research-log.md`, `mkdocs.yml` | U6 |

The order is W0, U1, and U5a together; then U2; then U3 and U4 together; then U5b and U6; then U7.
Each unit is committed separately, and the plan commit with this page and the research-plan row
comes first.

### Figures, raw data first

All charts follow the `dataviz` skill (OCF-brand palette, sized to the text column, SVG optimised
with `svgo`). Farm output is normalised and labelled Solar farm A to F. Substations may be named.

1. **Raw: the model-free trend.** For each Primary, the rolling 8-week mean of the midday gap
   between clear days (daily clear-sky index of 0.8 or more) and overcast days (0.3 or less), in
   April to September, from 2019 to 2026. Where the gap rises, the capacity rises.
2. **Raw: what a week of evidence looks like.** One clear and one overcast week in summer and in
   winter at Donington, with power and CAMS irradiance on twin axes.
3. **Raw: coverage.** A heat map of usable daylight half-hours per Primary-week, with masked closure
   days, dropped CAMS hours, and unobserved weeks marked.
4. **Raw: how much information a week holds.** The free weekly capacity with its standard error at
   Donington, and the standard error by calendar month for all Primaries, with the implied smallest
   detectable step.
5. **Method: one injected path through five fits.** A ramp, a step, and a fall at Donington with
   Solar farm D, drawn with the free, symmetric, asymmetric, strictly non-decreasing, and
   season-year fits.
6. **Method: the penalty grid.** The tuning-set error and false-step rate over `(a, r)`, with the
   chosen pair marked.
7. **Headline: path error by case family and fit,** with 99% intervals, for the scoring units.
8. **The detection curve:** the share of steps detected against step size, for April to September
   and October to March, one panel per Primary.
9. **Falls:** the share of a temporary fall recovered by depth and duration, for the symmetric,
   asymmetric, and strictly non-decreasing fits.
10. **Transfers and seasonal load:** the false step from an unlogged transfer against a masked one,
    and the seasonal swing that creeps into the path, for the weather-independent load and the
    sunshine-correlated load.
11. **False steps:** the distribution over the 200 placebo permutations per Primary, the
    unchanged-capacity injections, and the ratchet drift.
12. **Real-time vintages:** the first estimate against the final fit, and the revision by the age
    of the week, on injections and on the real series.
13. **Interval calibration:** the coverage of the truth against the nominal level (50%, 80%, and 90%)
    at step and non-step weeks.
14. **Headline on real data:** the weekly capacity of each Primary with its 90% band, experiment
    00040's season-year C as points, Kirton in grey with its library range, and the two
    systematic scenarios as separate lines.
15. **The gap between asymmetric and symmetric paths** on the real series, with held falls marked.
16. **Outside checks:** the series against the DESNZ expected peak with the tolerance band, the
    register total, and the OpenStreetMap area as a descriptive mark.
17. **Bulk supply points:** the weekly path at each bulk supply point against the sum of its parts.
18. **Map:** the 8 Primary polygons, shaded by capacity per square kilometre, with the substation
    points and the 95 m offset at Mount Bridge Boston noted.
19. **Coastal Primaries and school holidays:** Trusthorpe against the inland Primaries, and
    school-holiday weeks against term time.
20. **Additivity:** the fitted increment on the real series against the semi-synthetic score for
    the same injection.

### Risks

1. **Winter weeks carry almost no information.** Daylight is short and `u` is small, so the path is
   set by the penalty from about November to February. The standard error by month (figure 4) and
   the seasonal detection limit state this, and the series marks wide-interval weeks.
2. **Per-season-year cells absorb part of a step at a month boundary.** One in four injected steps
   sits on a boundary. If the recovered size is under 0.75 of the true size there, the write-up
   reports it and the penalty cannot fix it.
3. **The asymmetric penalty ratchets on noise.** Rises cost little and falls cost a lot, so noise
   peaks stick. Decision-rule condition (iv) and figure 11 measure the drift. If condition (iv)
   fails, the next part may need a penalty on the level as well as the increments.
4. **The fit is not additive.** The production series is fitted to real data that already holds
   unknown solar, so the semi-synthetic scoring is the primary score, the real-plus-injection scoring
   is the secondary score, and figure 20 shows the gap.
5. **Unlogged switching produces false steps.** The transfer case measures the size. The real
   series' detected steps must be checked against SW2-0b's labels once they exist, and a step that
   coincides with a labelled transfer is not a capacity change.
6. **A load that rises with sunshine looks like solar.** It biases the level low and, if seasonal,
   adds a swing to the path. The seasonal-load case measures the swing. Nothing in the data sizes
   the real load, so the scenario columns carry it separately.
7. **The model code is copied for the third time.** Experiments 00039, 00040, and now 00045 each
   hold a copy of the cell, window, and least-squares code. A move into `packages/studies` is
   overdue and is a separate reviewed change, outside this experiment.
8. **Only three scoring farms.** The farm margin of the two-way bootstrap is coarse. The write-up
   reports results with and without Solar farm E and the seed-level spread.
9. **The outside checks are weak.** The DESNZ tolerance of a factor of 1.5 catches only gross
   failure, the table undercounts installations after 2019, the LSOA boundaries are not the
   electrical network, and the register's date field may not be a connection date.
10. **The vintage end-of-record problem.** Penalised fits are least constrained at the end. The
    ERA5 temperature that production would use arrives about 5 days late, which the vintage scoring
    ignores; dropping the temperature term moved C by 0.006 of capacity in experiment 00039.
11. **Kirton and Wrangle T1.** Kirton's cloud load enters its weekly path as capacity, and Wrangle
    T1 has two years of data and a low signal. Both appear with their status, and neither supports
    a claim.
12. **CAMS dropouts.** Reliability under 0.9 drops 9.7% of hours, up to 12.6% of daylight hours in
    December, so winter weeks may fall below 3 usable days.
13. **The solver is new.** No convex-optimisation package is installed, and adding one changes
    `pyproject.toml`, which belongs to the maintainer. The ADMM routine is checked against
    `scipy.optimize.linprog` on a 30-week problem, against ordinary least squares at `a = 0`, and
    against a constant path at large `a`. A stopping tolerance that is too loose would show as a
    failed self-test.
14. **CPU contention.** The slot's 4 cores are shared with PVDIS-3. Check `uptime` before each stage.

## Results

To be written after the run.

## What the next experiment should answer

To be written after Opus reads the results.
