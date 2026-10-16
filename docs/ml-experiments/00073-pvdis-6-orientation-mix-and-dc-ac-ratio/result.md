---
id: "00073"
slug: pvdis-6-orientation-mix-and-dc-ac-ratio
status: planned
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
verdict: null
best_arm: null
headline: null
lessons: null
production_status: null
adopted_in: null
code: ml-experiments/00073-pvdis-6-orientation-mix-and-dc-ac-ratio/
advice: []
---

# Experiment 73: can the mix of solar panel orientations and the DC to AC ratio be recovered?

*Status: planned. The opening paragraph and the results are written after the run.*

**The plan, in plain words.** Solar panels facing east, south, and west produce power at different
times of day, and panels with more DC capacity than inverter capacity clip their output at midday.
We ask whether either can be read from a substation's power alone. We test each on answers we know
first: we add made-up east-south-west mixes, and the real output of six metered solar farms, to a
substation's real demand, and check what the fitting method recovers. Only if the method passes do
we fit the real series.

## Size: medium

- **What gets stored:** only private work tables under `data/ml-experiments/`; no Delta table.
- **Serving path:** untouched.
- **Degradation rule:** untouched.
- **More than one defensible design:** yes, so the plan fixes three orientation fits and one
  clipping search before any result.
- **Callers:** the machinery is experiment 39's and 40's code, copied into this folder; no caller
  outside `ml-experiments/`.

Because the designs are several, the process is the full one: a Sonnet plan, one Sonnet plan review,
a Sonnet implementation, a Sonnet code review, and a Sonnet write-up and prose review. An Opus plan
review is added only if the Sonnet review finds a flaw it cannot settle.

## Earlier advice followed

- **From experiment 40:** do not quote a capacity without a bound, and leave Wrangle T1 out of any
  claim that a Primary is free of a load. Every real-series table here keeps Kirton as a range and
  marks Wrangle T1.
- **From experiment 45:** the capacity is held constant within each season-year, not weekly.
- **Departure:** none.

## Data and fits

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

## Known answers, run before any real fit

- **Orientation:** inject 3 MW of synthetic output, `3 × (wE uE + wS uS + wW uW)` with the shapes
  built from CAMS at each solar farm's location, so the injected sky differs from the sky the fit
  uses at the Primary. A robustness arm, reported and not decisive, injects shapes at 15° and 40°
  tilt while the fit stays at 25°, and passing the known answers is necessary but not sufficient.
  A null arm injects nothing and reports the estimated shares, so the size of the noise is known.
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

## Real series (only for a part that passes its known answers)

Report the fitted shares or ratio for every Primary-year with the interval, the year-to-year change
in shares (a mix that swaps between years is not identified), and C3 and C4. Keep Kirton as a range
and mark Wrangle T1.

## Figures planned

Raw data first: clear-day midday power against irradiance for the farms and a Primary, to show
whether a clipping knee is visible; then the three orientation shapes through two days; recovery of
the shares per true mix and arm; the DC:AC error by farm; the profile of residual sum of squares
against r; the real shares and ratios per Primary-year; held-out contrasts C3 and C4. Farms carry
anonymous labels A to F.

## Compute

Slot B: 6 cores. Fits are least squares and non-negative least squares on small matrices; the run
uses 6 worker processes, one Primary per job, each with fixed seeds and its own output file.
