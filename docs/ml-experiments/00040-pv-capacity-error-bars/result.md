---
id: "00040"
slug: pv-capacity-error-bars
status: planned
size: medium
hypothesis: >-
  On the 8 directional (MW) Primary substations, comparing the PV capacity that six least-squares
  arms fit to the same series (they differ in how much cloud-driven demand each allows) detects a
  cloud-driven demand load that pushes the half-hourly fit's capacity at least 25% off, with at
  least 80% power and a false-positive rate of at most 10% on pure injected PV, so that a bounded
  error bar can be attached to the capacity at every Primary where no such load is detected.
prediction: >-
  Median capacity ratio C over truth on pure injected PV is 1.00 (0.97 to 1.03) and the
  false-positive rate is 0.05 to 0.10. Power at a confound amplitude of 0.25 is at least 0.8 for all
  five out-of-arm shapes at 7 of the 8 Primaries, with Wrangle T1 the likeliest to fall short.
  Kirton is flagged in 2022/23 to 2025/26. The other 7 Primaries are clean in 2025/26, with at most 3
  flagged Primary-years among them over all years, and the bound B is at most 0.10 at each clean
  Primary. The headline capacities in 2025/26 run from about 1.4 MW (Wrangle T1) to about 3.9 MW
  (Mount Bridge Boston). The scratch numbers behind this prediction are exploratory and were not
  blind. The main risk is that the arm noise of the real series is more strongly correlated than the
  synthetic 0.8, which raises power but may also flag rooftop fleets whose ratio vector differs from
  the solar farms'.
decision_rule: >-
  Calibration passes when the median of C over truth on pure injected PV lies in [0.9, 1.1] and the
  false-positive rate is at most 0.10. Power passes when, for every out-of-arm shape, the median over
  the 8 Primaries' latest season-years of the share of (unit, noise draw) pairs flagged at amplitude
  0.25 is at least 0.8. Verdicts: `bounded` (calibration and power pass, at least 6 of 8 Primaries
  clean in 2025/26, and B at most 0.25 at every clean Primary); `bounded-few-clean` (calibration and
  power pass, but fewer than 6 clean Primaries or B above 0.25 at a clean Primary); `underpowered`
  (calibration passes, power fails); `miscalibrated` (calibration fails). A verdict never rests on a
  breakdown. The experiment stops without a verdict if a self-test fails or the farm-point timestamp
  scan does not peak at a shift of zero. There is no cost-of-porting threshold, because the aim is
  explanation and capacity estimation, and not a production forecast column.
benchmark: >-
  None. The scored quantities are the recovery of a known injected capacity and the detection of a
  known injected demand load. The ±25% band is the maintainer's ranking prediction HD1.
row_set_id: none
metric: >-
  Capacity ratio C over truth, the false-positive rate of the misfit test, its power by confound
  shape and amplitude, and the bound B on the absolute relative error of C at an undetected
  Primary-year
code: ml-experiments/00040-pv-capacity-error-bars/
---

# Experiment 40: how far can the capacity fitted to embedded solar be trusted?

**Status `planned`: this page is the plan, committed before any code runs.** Experiment 39 recovered
a median 0.77 of the capacity of solar-farm output injected into the 8 directional Primary
substations (0.85 for the daily fit), but it could not say how far a demand load that rises with
cloud biases the fitted capacity of the real series. The scratch probes behind this plan found that
such a load leaks into the half-hourly capacity at about 3 MW per MW of load, that richer demand
terms lower the recovery below 0.75, and that no library of confound shapes recovers the PV capacity
when the real confound has a shape outside the library. They also found that the same confound is
easy to detect, because the six fits disagree about the capacity by an amount that a pure PV signal
cannot produce. This experiment tests that detector with known answers and then applies it to the
real series.

## Size

- **What gets stored:** nothing outside `data/ml-experiments/00040-pv-capacity-error-bars/work/`,
  which is private and never committed. No Delta table, no model file, no prediction set.
- **Production serving path:** not touched. The fits describe past data and use CAMS irradiance,
  which arrives too late to drive a live forecast.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes, and the design was chosen by an Opus review of
  experiment 39 over a richer demand model (PVDIS-2B) because the richer model fails a held-out
  shape test.
- **Code whose callers are hard to name:** no. The fitting code is copied from experiment 39's
  folder and lives in this experiment's folder, and `packages/` is untouched.

The size is medium. The Opus design is the plan review. One Sonnet reviewer reads the code (the
injection arithmetic, the false-positive and bias definitions, and leakage), and one Sonnet reviewer
checks the write-up against the printed report.

## Question and design

**Question.** On the 8 directional Primaries, does a cross-arm misfit test detect a cloud-driven
demand load large enough to push M1's capacity outside the ±25% band with at least 80% power, at a
false-positive rate of at most 10% on pure injected PV, and at how many Primaries is no such load
detected in 2025/26?

**Arms.** Each arm fits, within one season-year of daylight half-hours, `power = cell + b * HDD +
(cloud terms) - c * u`, where `u` is the unit PV shape driven by CAMS irradiance at the substation,
`HDD` is heating degree-days, `L` is the flat cloud response `1 - min(GHI / clear-sky GHI, 1)`, and
`cell` is a fixed effect for (month, local half-hour, day type). The six detector arms differ in the
cloud terms:

- **M1:** none.
- **M1L:** `L`.
- **M2:** `L * summer` and `L * winter`.
- **M2h:** `L` by season half and by three local half-hour bands (before 20, 20 to 27, 28 and later).
- **M1_s:** M1 on April to September rows only.
- **M2h_s:** M2h on April to September rows only, with three band terms.

The daily fit M1d is also fitted. It gives the headline capacity and is not one of the six detector
arms. The east, south, and west orientation menu is fitted once per Primary-year as a descriptive
figure and decides nothing.

**Injected PV and the ratios.** Each unit is a (Primary, solar farm, season-year) triple, paired as in
experiment 39. A farm's output is normalised by its 99th percentile (dropouts removed), so the
unit's true capacity is 1 MW. The fits are linear, so the arm vector of a unit is the six capacities
fitted to the normalised output alone. For arm `a` at a Primary, `R_a` is the median over that
Primary's units of the arm's capacity, computed leave-one-farm-out: a unit of farm `f` uses the units
of the other farms.

**Capacity and misfit.** For an arm vector `c`, the capacity under no confound is `C = (R . c) / (R .
R)` and the relative misfit is `T = sqrt(mean((c - R * C)^2)) / C`. The headline capacity of a
Primary-year is `c_M1d / R_M1d`, which is experiment 39's chosen arm, corrected.

**Noise.** For each real Primary-year, 200 bootstrap draws resample whole 7-day blocks of local dates
with replacement, up to the season-year's length, with a fixed seed. Each draw gives an arm vector.
The noise draws are each vector minus the mean of the 200.

**Null and threshold.** For Primary-year `(p, y)` the pure-PV arm vector of a unit is its arm vector
scaled so the truth equals the real `C` of `(p, y)`, plus one noise draw. The threshold `tau(p, y)`
is the 95th percentile of `T` over the units of the other farms at `p` (any year) times the 200
draws, with `R` leaving out both the unit's farm and the scored unit's farm. A Primary-year is
flagged when its real `T` exceeds `tau` built from all six farms.

**Confound shapes.** Each shape is divided by its mean over daylight rows: `flat = L`; `lowsun = L *
(1 - u_clear)`; `winter = L * 1[October to March]`; `dayL` (the day's mean `L` on every daylight row);
`evening = L * 1[local half-hour >= 30]`; `threshold = 1[L > 0.6]`; `mixture` (half flat and half
lowsun, each mean-scaled); and `sunload = -u`. A shape's arm vector is the six capacities fitted to
the shape alone, so adding the confound is a sum of vectors. The amplitude `a`, on the grid {0.1,
0.25, 0.5, 1.0}, is set so that the shape's M1 leakage equals `a` times the truth, using that unit's
own rows. The out-of-arm shapes, which no arm contains, are lowsun, dayL, evening, threshold, and
mixture. Flat and winter are contained in some arms and are reported but not scored. `sunload` is the
known blind spot, a load that rises with sunshine, and is reported but not scored.

**Scores.**

- **V1, calibration:** over all pure-PV units, the median of `C` over truth (leave-one-farm-out `R`)
  and the false-positive rate, the share of (unit, noise draw) pairs whose `T` exceeds the threshold
  built without that unit's farm.
- **V2, power:** for each out-of-arm shape and amplitude, the share of (unit, draw) pairs flagged,
  per Primary-year and as the median over Primaries' latest season-years.
- **Bound B:** for each Primary-year not flagged, and each out-of-arm shape, the median `|C / truth -
  1|` over undetected pairs at the largest amplitude on the grid whose power is under 0.8 (zero when
  power is at least 0.8 at 0.1). B is the largest over the shapes.
- **Real:** `T`, the flag, `C`, the headline capacity, and B per directional Primary-year, and the
  number of Primaries not flagged ("clean") in 2025/26.
- **Continuity:** checks K1 to K4 as experiment 39 defined them (read from its stored report, so a
  redefined check cannot count as progress), and K1 with its second clause replaced by `c / SE > 4`.

**Self-tests, run first, failing the script on a breach.** (a) Linearity: for one Primary-year and
arm M2h, `c(y + g) - c(y) - c(g)` is under 1e-8 for a confound shape `g`. (b) Injecting exactly `R *
C` with no noise gives `T` under 1e-10 and `C` equal to the truth within 1e-10. (c) The flat shape
fitted by arm M1L leaks under 1e-8, because the arm contains it. (d) The shape `-u` fitted by M1 has
leakage equal to minus the M1 capacity of `u` itself within 1e-6, so the blind spot is wired the
right way round. (e) With a pure-PV unit's own 200 draws as its null, the false-positive rate is 0.05
within 0.02.

**Leakage traps to check in code review.** The threshold for a unit must not use that unit's farm. `R`
must be leave-one-farm-out wherever it scores a unit. The confound amplitude comes from the M1
leakage of that unit's own rows, not a Primary-wide median. Summer-only arms must drop rows before
building cells. Bootstrap blocks must be whole 7-day spans of local dates.

## Weather source at the lag times

**CAMS irradiance is the only weather input of the fit, and it is not available at forecast time.**
The fits describe past data, so the result is for explanation and capacity estimation, and not for
a live forecast. Temperature (for HDD) is from ERA5.

## Planned figures

Raw, model-free charts first (overcast-minus-clear power by half-hour, evening power against cloud,
midday power against CAMS irradiance), then the method (shapes, leakage heat map, daily collinearity),
the calibration and power results, and the real results (headline capacity with bound, `T` against
`tau`, the capacity trend, the arm capacities against `R * C`, the orientation menu, and the blind
spot). Solar farms are labelled A to F, and no chart pairs a farm label with a name or identifier.

## Where the code lives

The fitting code is copied from experiment 39 into `ml-experiments/00040-pv-capacity-error-bars/`,
and `packages/` is untouched. The write-up says whether the maintainer should later move it into
`packages/studies`.
