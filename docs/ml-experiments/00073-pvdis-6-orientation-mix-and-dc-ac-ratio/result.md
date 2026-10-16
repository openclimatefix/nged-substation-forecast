---
id: "00073"
slug: pvdis-6-orientation-mix-and-dc-ac-ratio
status: finished
size: medium
hypothesis: >-
  On the 8 Primary substations metered in MW, (a) the share of east-, south-, and west-facing solar
  behind each Primary can be recovered from its half-hourly power, and (b) the ratio of DC to AC
  capacity of that solar can be recovered from a clipping level fitted on clear days. Both are
  tested first on known answers: synthetic east-south-west mixes of solar output, and the output of
  the six metered solar farms, added to the real demand.
prediction: >-
  The orientation mix is not identified. On injected mixes the free fit's total-variation error
  (half the sum of absolute differences between the true and estimated shares, 0 to 1) is 0.25 to
  0.40, no better than guessing the fleet mix (0.30 on average over the five mixes), because the
  three orientations differ only in the time of day of their output and the calendar cells absorb
  the mean daily shape. The year-pooled ridge fit on clear days reaches 0.15 to 0.25. The DC:AC
  ratio is recovered from a solar farm's output alone to within 0.10 for 4 of the 6 farms, but not
  from a farm injected into a Primary's demand, and the real Primaries give fitted ratios near 1.0
  to 1.2 with intervals wider than 0.3, because a fleet of many systems clips gradually. Neither
  part improves the held-out fit by 0.5%. Probability of the verdict `refuted`: 0.7.
decision_rule: >-
  Four planned contrasts decide, each read at a 98.75% interval (a Bonferroni level for four
  contrasts) from a cluster bootstrap with 4000 resamples and seed 73. Clusters: C1 and C2 resample
  (Primary, farm location) pairs; C3 and C4 resample Primaries. With fewer than 6 distinct Primaries
  the verdict is `inconclusive`. Orientation (O) and DC:AC (D) are scored separately. C1 is the
  total-variation error (half the sum of absolute differences between true and estimated shares)
  of the best orientation arm minus that of guessing the fleet mix (0.2 east, 0.6 south, 0.2 west),
  pooled over the five injected mixes and also reported per mix, because the guess is exact for one
  mix. The arm is the one with the lowest median error on tuning units (farm locations A to C), and
  scoring uses farm locations D to F. O is `identified` when the median error of that arm on scoring
  units is at most 0.15 (the binding clause) and the upper limit of C1 is below -0.10. D has a gate
  first: for each farm-year and each r on the grid, count the clear-day half-hours with r times
  plane-of-array irradiance over 1000 at least 1; if fewer than 100 reach the knee at r = 1.15, D is
  `not testable` and is neither identified nor refuted. C2 is the median absolute error of the
  fitted ratio minus that of the constant guess of the median ratio of the other five farms, over
  the six farms, scored separately for the farm-alone case and the 10 MW injection. D is
  `identified` when both cases have a median absolute error of at most 0.10, the fit beats the
  guess on at least 5 of 6 farms in both cases, and the upper limit of C2 is below 0 in both. With
  six farms the C2 interval is descriptive and the sign count decides. If D fails its known
  answers, the real series are not fitted for D. C3 and C4 use held-out blocks on the real series:
  fit on alternate 4-week blocks, score the others, and the reverse, keeping only cells with at
  least 20 fitted rows. They measure the relative change in the root mean square error of daylight
  power. C3 is the fitted orientation mix against the south-only shape, and C4 is the fitted ratio
  against a fixed ratio of 1.15. Each counts as a gain when the upper limit is below -0.005, and
  the interval, not the point estimate, decides. The threshold of 0.5% of the error is a
  judgement: the explicit solar forecast of PVDIS-5 would need a per-Primary table of three shapes
  (O) or one ratio (D), and a smaller gain than the sampling noise of a half-year contrast would not
  pay for the table. Verdict, over the four outcomes (O identified, C3 gain, D identified, C4
  gain): `supported` when O is identified with a C3 gain and D is identified with a C4 gain;
  `partly supported` when exactly one part is identified with its held-out gain; `small gain` when
  a part is identified, its held-out interval excludes zero, and the upper limit is above -0.005;
  `refuted` when neither part is identified, or D is not testable and O is not identified;
  `inconclusive` when a self-test fails or a gate is missed for O. Where D is `not testable`, the
  verdict is decided by O alone, with `supported` or `partly supported` read as O's result.
benchmark: >-
  Guessing the fleet mix (O), guessing the median ratio of the other farms (D), the south-only
  shape (C3), and a fixed ratio of 1.15 (C4). There is no forecast benchmark.
row_set_id: none
metric: >-
  Total-variation error of the orientation shares, absolute error of the fitted DC:AC ratio, and
  the relative change in held-out root mean square error of daylight power.
verdict: refuted
verdict_note: >-
  The pre-registered rule has no clause for this outcome: the orientation fit passes its known
  answers (so it is `identified`) but has no held-out gain, and the verdict list needs a gain for
  `partly supported` and an interval that excludes zero for `small gain`. The verdict is read from
  the rule's stated purpose, which is whether the mix or the ratio is worth porting into the solar
  forecast of PVDIS-5, and it is not.
best_arm: >-
  Pooled orientation fit with no prior (one mix per Primary shared by its season-years), chosen on
  tuning units: median share error 0.088 on scoring units, against 0.300 for guessing the fleet mix.
headline: >-
  On injected mixes the orientation fit beats guessing the fleet mix by 0.181 (98.75% interval
  -0.228 to -0.122), but the real series give a mix that changes by a median 0.24 between
  consecutive season-years, and the fitted mix changes held-out error by +0.8% (-0.9% to +2.8%)
  against all-south panels. The DC:AC ratio search has a median absolute error of 0.055 on a farm's
  output alone and 0.195 once the farm is added to a Primary's power, and beats the constant guess
  on 0 of 6 farms in that case, so it is dropped before any real-series fit.
lessons: >-
  A known-answer test that passes is necessary and not sufficient. The orientation mix is
  recoverable when the true shapes are the fitted shapes, and still does not stay put on the real
  series, so the real fleet differs from three clean orientations. The DC:AC ratio is visible in a
  farm's own output, but a Primary's demand and its own solar hide it.
production_status: not-for-production
adopted_in: none
code: ml-experiments/00073-pvdis-6-orientation-mix-and-dc-ac-ratio/
advice:
  - >-
    Keep one south-facing shape and a fixed DC:AC ratio of 1.15 in the explicit solar term of
    PVDIS-5, because neither the fitted orientation mix nor a fitted ratio is worth a per-Primary
    table (C3, figure 7; DC:AC known answers, figure 4).
  - >-
    Do not read an orientation mix from one season-year of a Primary's power, because the shares
    change by a median 0.24 between consecutive season-years (figure 5).
  - >-
    Inject known mixes at tilts other than the fitted one before trusting an orientation fit,
    because the median share error rose from 0.09 at the fitted 25 degrees to 0.20 at 15 degrees
    (robustness arm in the known-answer report).
  - >-
    Test a clipping search with the farm added to a real Primary's power and not only on the farm's
    own output, because the median error went from 0.055 to 0.195 between the two (figure 4).
  - >-
    Check whether the west lean of the pooled mix at 5 of 8 Primaries is evening demand before
    calling it solar, using rows after dark as experiment 40 suggested (figure 6).
---

# Experiment 73: can the mix of solar panel orientations and the DC to AC ratio be recovered?

We asked whether two properties of the solar panels behind a substation can be read from the
substation's power alone. The first is the share of panels facing east, south, and west. The second
is the ratio of DC capacity to AC (inverter) capacity, which sets the level at which output flattens
at midday. We first tested each fit on answers we know: we added made-up east-south-west output,
and the real output of six metered solar farms, to the real power of 8 Primary substations (the
substations that step power down to the local network), and checked what the fit recovered. The
orientation fit passed that test, finding the shares with a median error of 0.09 where guessing the
usual mix gives 0.30 (error 0 means exact, 1 means worst). On the real power it did not stay put:
the fitted shares changed by a median 0.24 from one year to the next, and the fitted mix predicted
held-out weeks no better than assuming every panel faces south. The DC:AC ratio search found the
ratio of a farm's own output, but not once that farm's output sat inside a Primary's power, so we
dropped it before touching the real series.

**Take-home message: the explicit solar forecast of PVDIS-5 should keep one south-facing shape and
a fixed DC:AC ratio of 1.15, because neither extra parameter can be read from a Primary's power well
enough to be worth carrying.** The orientation mix is recoverable only when the real fleet looks
like three clean orientations, and the real series show that it does not. The DC:AC ratio is hidden
by the Primary's own demand and solar. The search therefore narrows to better inputs (the sunshine
forecast itself) and away from richer solar shapes.

**Verdict `refuted`, read from the plan's purpose.** The pre-registered rule has no clause for an
orientation fit that passes its known answers and then shows no held-out gain. Orientation is
`identified` by the plan's definition (median error 0.088, at most 0.15; C1 upper limit -0.122,
below -0.10), and C3 is not a gain. DC:AC is not identified. The verdict follows the question the
experiment was asked, which is whether either parameter is worth porting into the solar forecast of
PVDIS-5, and neither is.

![Figure 1: clear-day output of the six solar farms against irradiance](figure_01.svg)

![Figure 2: east, south, and west shapes through one clear day](figure_02.svg)

## Known answers

**Injected east-south-west mixes are recovered far better than a guess.** The arm chosen on the
tuning units (solar-farm locations A to C) was the pooled fit with no prior. On the scoring units
(locations D to F) its median share error is 0.088, against 0.300 for guessing the fleet mix. C1,
the arm's error minus the guess's, is -0.181 (98.75% interval -0.228 to -0.122), pooled over the
five true mixes. Per true mix, the guess is exact at the fleet mix itself and the fit is not (0.097
against 0), and the fit wins at the other four.

| True mix (east/south/west) | Pooled fit | Guess: fleet mix | Guess: all south |
|---|---|---|---|
| all south (0/1/0) | 0.102 | 0.400 | 0.000 |
| fleet mix (0.2/0.6/0.2) | 0.097 | 0.000 | 0.400 |
| east-heavy (0.5/0.4/0.1) | 0.097 | 0.300 | 0.600 |
| west-heavy (0.1/0.4/0.5) | 0.100 | 0.300 | 0.600 |
| east and west (0.5/0/0.5) | 0.045 | 0.600 | 1.000 |

![Figure 3: share error by true mix, fit against guesses](figure_03.svg)

**The known answer is the easy case.** Both the injected output and the fitted shapes come from the
same physics, and only the sky differs (CAMS at the farm location against CAMS at the Primary). The
robustness arm injected shapes at 15° and 40° tilt while the fit stayed at 25°. The pooled fit's
median error was 0.100 at 40° and 0.200 at 15°, so the 0.15 limit holds at 25° and 40° and not at
15°. At the fitted tilt, 0.192 of the scoring units hit one of the two bounds (a negative increment
set to zero, or a total increment under 0.3 MW).

**The DC:AC ratio is found from a farm's own output and lost once the farm is added to a Primary.**
The knee gate passed: the median number of clear-day half-hours at or above the knee at ratio 1.15
is 165 to 188 for five farms and 110.5 for farm D, against the threshold of 100. The search on each
farm's output alone has a median absolute error of 0.055 over the six farms and beats the constant
guess (the median ratio of the other five farms) on 5 of 6. With 10 MW of the farm's output added to
a Primary's power, the median error is 0.195 and the search beats the guess on 0 of 6, so D is not
identified. The searched ratio sits at 1.00 to 1.10 for every farm in that case.

| Farm | True ratio | Error, farm alone | Error, added to a Primary | Error of the guess |
|---|---|---|---|---|
| A | 1.15 | 0.050 | 0.150 | 0.070 |
| B | 1.17 | 0.020 | 0.170 | 0.050 |
| C | 1.22 | 0.070 | 0.220 | 0.050 |
| D | 1.42 | 0.220 | 0.320 | 0.250 |
| E | 1.12 | 0.030 | 0.120 | 0.100 |
| F | 1.26 | 0.060 | 0.260 | 0.090 |

![Figure 4: searched ratio against true ratio](figure_04.svg)

The ratios of experiment 29 come from the same CAMS irradiance and the same clipping form as the
search, so the farm-alone case is the easy bound. With six farms the intervals are descriptive, and
the sign count decides.

## The real series

**The real series do not support a stable orientation mix.** The free fit to each Primary-year
changes its shares by a median total-variation distance of 0.237 between consecutive season-years
(33 pairs), and the clear-day fit by 0.326. The known-answer error of the same arms is about 0.1.
The pooled fit shares one mix per Primary across its season-years, and its 95% intervals (200
resamples of 7-day blocks of clear days) are on average 0.49 wide on a share that runs from 0 to 1.

![Figure 5: shares of the free fit per Primary and season-year](figure_05.svg)

| Primary | Capacity (MW) | East | South | West |
|---|---|---|---|---|
| Kirton (flagged in experiment 40) | 1.58 | 0.00 (0.00 to 0.10) | 0.95 (0.68 to 1.00) | 0.05 (0.00 to 0.29) |
| Trusthorpe | 7.58 | 0.33 (0.01 to 0.49) | 0.08 (0.00 to 0.40) | 0.58 (0.41 to 0.65) |
| Spilsby | 2.57 | 0.27 (0.07 to 0.44) | 0.24 (0.00 to 0.66) | 0.49 (0.21 to 0.65) |
| Horncastle | 3.03 | 0.31 (0.00 to 0.49) | 0.19 (0.00 to 0.66) | 0.50 (0.30 to 0.59) |
| Mount Bridge Boston | 3.65 | 0.06 (0.00 to 0.30) | 0.51 (0.19 to 0.75) | 0.43 (0.23 to 0.57) |
| Wrangle T1 (little power) | 3.71 | 0.27 (0.00 to 0.64) | 0.23 (0.00 to 1.00) | 0.50 (0.00 to 0.92) |
| Donington | 3.27 | 0.30 (0.00 to 0.48) | 0.23 (0.00 to 0.58) | 0.46 (0.32 to 0.59) |
| Wrangle T2 | 2.69 | 0.49 (0.18 to 0.89) | 0.29 (0.00 to 0.71) | 0.22 (0.00 to 0.53) |

The capacity column is the mean of the years' south-only capacities times the sum of the shares, and
it differs from experiment 40's capacities (for example 7.58 MW against 2.48 MW at Trusthorpe). The
orientation fit trades capacity against shape, so these totals are not capacity estimates.

![Figure 6: pooled shares with 95% intervals](figure_06.svg)

**The fitted mix does not predict held-out weeks better than all-south panels.** C3, the relative
change in held-out root mean square error of daylight power from the pooled mix against the
south-only shape, is +0.8% (98.75% interval -0.9% to +2.8%, 80 held-out year-folds). The threshold
for a gain is an upper limit below -0.5%, so C3 is not a gain, and it is not distinguishable from
zero. C4 was not computed, because the DC:AC fit failed its known answers.

![Figure 7: held-out change in error by Primary](figure_07.svg)

**Five of the eight Primaries lean west.** Trusthorpe, Spilsby, Horncastle, Wrangle T1, and
Donington have a pooled west share of 0.46 to 0.58, against 0.2 in the fleet mix. This experiment
cannot say whether the lean is west-facing panels or demand that rises in the evening, which looks
like west-facing output to the fit.

## Changes from the plan

- **The held-out cell minimum is 5 rows, not 20.** With alternate 28-day blocks, no cell reached 20
  fitted rows, so every held-out row was dropped in the first run. The change was made before any
  held-out result existed.
- **The null arm became a bootstrap.** The plan injected nothing and reported the shares, but the
  increment of no injection is exactly zero. The pooled shares' 95% intervals from resampled blocks
  report the noise instead.
- **The pooled arm with ridge 0 won the tuning, so the prior never entered.** Its name in the plan,
  "pooled with a prior", stands for the family whose best member has no prior.
- **The 5 MW injection was dropped,** as the plan review advised, and the 10 MW injection decides.
- **DC:AC has no fitted parameter beyond its grid,** so all six farms are scored, and no tuning
  split applies. The Primaries' own series were searched without injection as a reference only
  (41 Primary-years, quartiles 1.00, 1.00, and 1.30, range 1.00 to 1.60); the plan stops real-series
  fits for D, and this reference is not a claim about the Primaries.

## Caveats

- **The known answers use one physics.** Passing them is necessary and not sufficient, as the 15°
  robustness arm shows.
- **Only 6 farms score DC:AC,** so the sign count is a coarse test.
- **No Opus plan review was run;** the plan had one Sonnet review.
- **The pooled capacity depends on a south-only scale and a floor of 0.1 MW,** which is why its
  total differs from experiment 40's capacity.

## What the next experiment should answer

**The next experiment should ask whether the west lean of the pooled mix is evening demand.** The
evidence is that five Primaries lean west by the same amount and the shares swap from year to year.
A test could fit the same three shapes to rows after sunset, where no solar exists, and see whether
the "west" capacity appears there as well. If it does, the lean is demand, and the plan for PVDIS-5
stands as it is. If it does not, a west-facing share is real at those Primaries and a per-Primary
shape table is worth reconsidering with a larger sample of years.

## Method

### Size: medium

- **What gets stored:** only private work tables under `data/ml-experiments/`; no Delta table.
- **Serving path:** untouched.
- **Degradation rule:** untouched.
- **More than one defensible design:** yes, so the plan fixes three orientation fits and one
  clipping search before any result.
- **Callers:** the machinery is experiment 39's and 40's code, copied into this folder; no caller
  outside `ml-experiments/`.

The process was the full one, with Sonnet throughout: a plan, one plan review, an implementation, a
code review, and a write-up with a prose review.

### Earlier advice followed

- **From experiment 40:** do not quote a capacity without a bound, and leave Wrangle T1 out of any
  claim that a Primary is free of a load. Every real-series table here flags Kirton and
  Wrangle T1.
- **From experiment 45:** the capacity is held constant within each season-year, not weekly.
- **Departure:** none.

### Data and fits

**Inputs are experiment 39's frame, read and never rebuilt:** half-hourly power of the 8
directional Primaries and 6 solar farms, CAMS irradiance, and ERA5 temperature, on daylight
half-hours (zenith under 75°). The fit window is April to September of each season-year. The demand
term is the calendar-cell term of experiment 40's arm M2h (fixed effect per month, half-hour, and
day type, heating degree-days, and cloud terms in three hour bands). The solar shapes come from
`studies.pv_separation.solar_basis` at 25° tilt, which gives the AC output of 1 MW of east-, south-,
and west-facing panels, clipped at `min(r × irradiance / 1000, 1)`.

**Orientation arms.** Each fits non-negative east, south, and west capacities to a Primary's
season-year.

- **O1, free:** three free capacities.
- **O2, free on clear days:** O1 on days whose mean cloud response is below 0.15. The cloud
  response is `1 - min(ghi_u / ghi_u_clear, 1)` at the Primary's CAMS point, averaged over the day's
  daylight half-hours (experiment 39's frame columns). A Primary-year needs at least 20 clear days
  or it is dropped from O2 and O3 and the count is reported.
- **O3, pooled with a prior:** one mix per Primary shared by all its season-years, clear days only,
  with a ridge penalty pulling the shares towards the fleet mix. The penalty is chosen on tuning
  units from the grid 0, 0.1, 1, 10 (relative to the mean squared column norm).
- **References:** G guesses the fleet mix; S assumes all south.

**DC:AC arm.** For each ratio r on the grid 1.00, 1.05, ..., 1.60, fit one capacity to the clear-day
rows with the shape `min(r × irradiance / 1000, 1)` (south, 25° tilt) and keep the r with the lowest
residual sum of squares. A ratio of 1.00 never clips in GB. The interval is the 2.5th to 97.5th
percentile of the best r over 200 resamples of 7-day blocks.

### Known answers, run before any real fit

- **Orientation:** inject 3 MW of synthetic output, `3 × (wE uE + wS uS + wW uW)` with the shapes
  built from CAMS at each solar farm's location, so the injected sky differs from the sky the fit
  uses at the Primary. A robustness arm, reported and not decisive, injects shapes at 15° and 40°
  tilt while the fit stays at 25°, and passing the known answers is necessary but not sufficient.
  The five true mixes (wE, wS, wW) are (0, 1, 0), (0.2, 0.6, 0.2), (0.5, 0.4,
  0.1), (0.1, 0.4, 0.5), and (0.5, 0, 0.5). Units are Primary × season-year × farm location × mix.
  The estimated shares come from the increment of the fitted capacities, with negative increments
  set to zero. A total increment under 0.3 MW counts as an error of 0.5.
  Each arm reports the share of units that hit either rule.
- **DC:AC:** the DC:AC ratios of the six farms are experiment 29's PyTorch fits: 1.15, 1.17, 1.22,
  1.42, 1.12, and 1.26. Run the search on each farm's output alone against its own location's shape,
  and on each farm's output scaled to 10 MW and added to a Primary's real series. The
  farms are tuned and scored in the same split as orientation (A to C, D to F).
- **Self-tests that stop the run:** a made-up all-south mix reproduces the south-only fit; noise-free
  injection into zero demand returns the true mix to 1e-6; noise-free clipped output returns the
  true ratio; linearity of the increment.

### Real series

The real-series fits use the same designs. The free and clear-day fits are per season-year. The
pooled fit has a 200-draw block bootstrap. Contrast C3 fits on alternate 28-day blocks, scores the
others, and reverses, with the capacities fixed from the training blocks and the calendar cells and
nuisance terms refitted on them.
