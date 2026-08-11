# NWP variable conventions

This page says how to read a value out of the `nwp` Delta table, and what the feature pipeline does
to it on the way to the model. Every ECMWF ENS variable we store is listed with the convention that
governs it: some values describe an instant, some describe the six hours before that instant, and
one is an angle that wraps.

This page and [Known ECMWF ENS data-quality issues](ecmwf-ens-known-issues.md) split the subject
between them:

- **That page is about *upstream*.** It covers the ways Dynamical.org's data arrives damaged —
  corrupt source accumulations, null grid points, incomplete runs — and the ingest policy that
  decides which of those we reject and which we tolerate. Its subject is data we did not create and
  cannot fix, so a defect described there is a reason to talk to Dynamical.

- **This page is about *us*.** It covers how the data that arrives intact is meant to be
  interpreted, and what our own code does with it. Everything here is a choice this project made, or
  failed to make deliberately, so a defect described here is a reason to change our code.

## The forecast step grid

ECMWF ENS is **not evenly spaced over the horizon**. Measured on the 2026-08-10 00Z run, and
identical in every run in the archive:

| step width | lead times | number of steps |
|---|---|---|
| — | 0 h (analysis) | 1 |
| 3 h | 3 h → 144 h | 48 |
| 6 h | 150 h → 360 h | 36 |

The change at **144 h — six days** matters most, because the primary user band is 3–10 days
([XGBoost improvements](../roadmap/xgboost-improvements.md)), so days 6 to 10 of that band sit on
the coarse half of the grid.

## The three axes

A variable takes one convention from each of the three axes below. Most are instantaneous, linear
and numeric.

### Instantaneous versus period-ending

An **instantaneous** variable describes conditions at `valid_time` itself.

A **period-ending** variable is the *average rate over the interval that ends at `valid_time`*.
Dynamical.org de-accumulates these from ECMWF's cumulative source fields before we receive them.
Three variables are period-ending: `downward_short_wave_radiation_flux_surface`,
`downward_long_wave_radiation_flux_surface` and `precipitation_surface`. They are legitimately null
at lead 0, because there is no preceding interval.

Beyond 144 h the interval is **six hours long**, so a 12:00 shortwave value is the mean over
06:00–12:00. That window covers much of the morning ramp, so the value is far below the
instantaneous irradiance at noon. What changes at day 6 is the averaging window, not the weather.

`PowerTimeSeries` uses the same convention — its `value` is the "average power over the preceding
30-minute period" — so the two sides of the NWP-to-power join agree, and features derived from
period-ending NWP should stay period-ending to match.

### Linear versus circular

`wind_direction_10m` and `wind_direction_100m` are **circular**: 359° and 1° are two degrees apart
in reality, and 358 degrees apart arithmetically. No arithmetic that assumes a number line —
interpolation, mean, gradient, quantile, standardised anomaly — is valid on them without special
handling. Every other numeric variable is linear.

### Numeric versus categorical

`categorical_precipitation_type_surface` is a category code, not a quantity, so it is aggregated by
area-weighted mode rather than averaged, and forward-filled rather than interpolated.

It appears to be instantaneous. The three period-ending variables are 100 % null at lead 0, because
there is no preceding interval for them to average; `categorical_precipitation_type_surface` is
fully populated there. That is good evidence but not proof, since a period field could in principle
be published at lead 0 anyway, so confirm against
[ECMWF's parameter database entry](https://codes.ecmwf.int/grib/param-db/260015) before relying on
its timing. If it did describe the preceding interval, forward-filling it would attribute one
interval's precipitation type to the following one.

## Every variable, and how to read it

The final column measures **what the 3 h → 6 h step change costs**: the error, at the held-out
intermediate steps, from reconstructing the 3-hourly part of the horizon out of a 6-hourly version
of itself the way the feature pipeline reconstructs half-hourly rows. `MAE/SD` normalises that error
by the variable's own spread, so the column is comparable across variables. Measured on the
2026-08-10 00Z run, leads 3–144 h, 4,090,608 rows. For the period-ending variables the 6-hourly
product is *exactly* reproducible by averaging consecutive pairs of 3-hourly means, so their figures
are exact rather than an emulation. The two directions are scored by mean circular distance —
`|((reconstructed − truth + 180) mod 360) − 180|` — since a plain subtraction of two angles is not a
distance.

| variable | convention | unit | MAE at 6-hourly | MAE/SD |
|---|---|---|---|---|
| `downward_short_wave_radiation_flux_surface` | period-ending | W m⁻² | 111 | **0.44** |
| `wind_direction_10m` | instantaneous, **circular** | ° | 20.9 | — |
| `wind_direction_100m` | instantaneous, **circular** | ° | 19.4 | — |
| `downward_long_wave_radiation_flux_surface` | period-ending | W m⁻² | 6.5 | 0.27 |
| `wind_speed_100m` | instantaneous | m s⁻¹ | 0.75 | 0.30 |
| `wind_speed_10m` | instantaneous | m s⁻¹ | 0.57 | 0.28 |
| `precipitation_surface` | period-ending | kg m⁻² s⁻¹ | 9.6 × 10⁻⁶ | 0.19 |
| `dew_point_temperature_2m` | instantaneous | °C | 0.49 | 0.17 |
| `temperature_2m` | instantaneous | °C | 0.66 | 0.15 |
| `pressure_reduced_to_mean_sea_level` | instantaneous | Pa | 29 | 0.09 |
| `geopotential_height_500hpa` | instantaneous | m | 2.4 | 0.03 |
| `pressure_surface` | instantaneous | Pa | 29 | 0.02 |
| `categorical_precipitation_type_surface` | categorical | code | — | — |

The synoptic variables are unharmed by the coarser grid (MAE/SD 0.02–0.09). Shortwave radiation is
the worst affected numeric variable, at half again the next-worst (0.44 against `wind_speed_100m`'s
0.30). The two directions carry no MAE/SD because a standard deviation of a circular quantity is not
meaningful; their error is the mean circular distance, which is a much larger fraction of the range
a direction can take.

## What the half-hourly resample does today

`_upsample_nwp_to_half_hourly` builds a 30-minute grid per `(nwp_init_time, ensemble_member,
time_series_id)` group, left-joins the native steps onto it, then **linearly interpolates every
column in `Nwp.continuous_var_names()`** and forward-fills the categorical one.

`continuous_var_names()` is defined as "every weather variable that is not categorical". That single
definition is the root of two of the three defects below: it silently classifies a period-ending
rate and a wrapped angle as things that may be linearly interpolated between their `valid_time`
stamps.

Repairing all three is scheduled as the first work in v0.5 — see
[Fix the NWP resample to honour the variable conventions](../roadmap/xgboost-improvements.md#fix-the-nwp-resample-to-honour-the-variable-conventions).
Until that lands, the behaviour described below is what the pipeline does.

### Wind direction is interpolated across the 0°/360° wrap

Interpolating 350° → 10° yields 180°: due south for a northerly wind. Measured on the 2026-08-10
00Z run, over every consecutive native-step pair:

| step width | pairs straddling north (`_10m`) | pairs straddling north (`_100m`) |
|---|---|---|
| 3 h | 3.73 % | 3.50 % |
| 6 h | 8.30 % | 8.03 % |

Inside a straddling interval the error grows linearly with distance from the first step: at fraction
$t$ of the way across, the interpolated value is $360t$ degrees from the truth. A straddling
3-hourly interval therefore holds five interpolated rows wrong by 60°, 120°, 180°, 120° and 60°.
Only the midpoint is a full reversal; the mean error across the interval is **108°** at 3-hourly
spacing and **98°** at 6-hourly.

Across the whole horizon that affects **6.57 %** of interpolated `wind_direction_10m` rows and
**6.32 %** of `wind_direction_100m` rows — 5.80 % and 5.58 % of all half-hourly rows, since the
native steps themselves are correct. The error is not random: it always rotates the wind *away* from
North, so northerlies are recorded as coming from somewhere between north-east and south.

Both columns are in `conf/model/xgboost.yaml`'s `selected_features`, so this reaches the model.

### Period-ending variables are interpolated as though they were instantaneous

The resample interpolates between `valid_time` stamps, which treats a backward-looking mean as an
instantaneous reading at the *end* of its window. The reconstructed day is therefore both flattened
and shifted late — by half the step width, so **three hours** in the 6-hourly part of the horizon.
Reconstructing a clear-sky day (Haurwitz clear-sky irradiance at 52.5° N, 1° W, 10 August):

| step | reconstruction | peak W m⁻² | peak at |
|---|---|---|---|
| — | truth (instantaneous) | 816 | 12:00 |
| 3 h | today's resample | 756 | 15:00 |
| 6 h | today's resample | 590 | 18:00 |

Read the 6-hourly row carefully: the reconstruction is a near-flat plateau from 12:00 to 18:00, and
the argmax lands at 18:00 only because the two knots differ by about 3 %. The honest summary is that
the modelled solar day is shifted about three hours late and its peak cut by a quarter, not that it
peaks at 18:00. Daily *energy* is preserved (7.18 against 7.20 kWh m⁻²) — it is the shape and the
timing that are wrong, which is what a PV site responds to.

The archive confirms the period-ending reading. Forming the clear-sky index two ways over the
2026-08-10 00Z run (1671 cells × 51 members, 7.16 M rows, restricted to rows where clear-sky
irradiance exceeds 200 W m⁻² so the ratio is well conditioned) settles it, because a clear-sky index
above 1 is physically impossible:

| step | `GHI / CS(valid_time)` p99 / max | `GHI / CS(mean over preceding step)` p99 / max |
|---|---|---|
| 3 h | 2.0 / 2.2 | 0.99 / **1.02** |
| 6 h | 2.3 / 3.0 | 0.96 / **1.03** |

The period-mean denominator caps at 1.0 in both halves of the horizon; the instantaneous one implies
irradiance twice what the sky can deliver. The exact tail figures depend on the clear-sky model and
the daylight cut-off, so treat the leading digits as the result rather than the third decimal place.

### Instantaneous variables lose diurnal amplitude

Linear interpolation between 6-hourly samples misses the daily temperature maximum, which falls at
around 15:00, between the 12:00 and 18:00 samples. Measured on the same run over complete
24-hour blocks, the mean daily temperature range falls from **6.09 °C to 5.37 °C — an 11.9 %
compression**.

Unlike the other two defects this one has no exact fix. The diurnal cycle is not a pure sinusoid:
its asymmetric afternoon peak needs the second and third harmonics, and four samples a day is
already at the Nyquist limit for the second, so some amplitude is genuinely unrecoverable at
6-hourly spacing. A shape-preserving interpolant would recover more of it than a linear one, though
how much has not been measured. The compression feeds the effective-temperature, degree-day and
`windchill` features.

## Wind is stored as speed and direction, and why

The ingest computes wind speed and direction from ECMWF's native `u`/`v` components and discards the
components ([`convert_to_polars.py`](../api/dynamical_data/index.md)). Two properties of that choice
are worth knowing.

**The conversion happens after the spatial aggregation, which is the right order.** H3 cells are
built from `wind_u_*`/`wind_v_*`, and speed and direction are derived from the already-aggregated
components. Averaging *direction* over grid points would have the same wrap defect in space that the resample
has in time. Direction is never averaged over grid points, so that defect does not arise.

**The stored speed is therefore the magnitude of the cell's *vector* mean, not the mean of the grid
points' scalar speeds.** These differ — |E[**v**]| ≤ E[|**v**|] — and the gap grows wherever wind
direction varies across a cell. The second quantity is the one a turbine power curve wants, because
power responds to the speed at each point and the curve is strongly non-linear. The scalar mean
cannot be recovered from the archive: it is a different spatial reduction, so obtaining it requires
re-ingesting. The size of the gap has not been measured.

**Storing the components instead would cost roughly 6 % of the table.** The two forms carry the same
information, so this is purely a storage question. Round-tripping speed and direction to components
and back, with the components rounded to the
[13-bit significand](../api/delta_store/index.md) the table stores, costs at most
6.8 × 10⁻³ ° of direction and 2.4 × 10⁻³ m s⁻¹ of speed over a full run — the same order as the
rounding the table already applies, and far below anything a forecast responds to.

Written through the production path (same significand rounding, sort order and writer properties),
the full column set both ways:

| run | speed + direction | u + v | change |
|---|---|---|---|
| 2026-08-10 00Z | 142.6 MB | 151.6 MB | +6.3 % |
| 2026-08-03 00Z | 144.6 MB | 152.8 MB | +5.7 % |

A wrapped angle turns out to compress *better* than the components, not worse. Averaged over the
archive — 40 Parquet files, 145 M rows — each direction column costs 1.57 bytes per row and the two
together are 19.3 % of all bytes, making direction the most expensive weather column we store; a
random 40-file sample gives 1.53 bytes per row and the same 19.3 % share. Yet the components cost
more still, because the significand rounding collapses values into repeats that Parquet's dictionary
and RLE encoding capture, and `u`/`v`, being products of a speed and a trigonometric function, take
roughly three times as many distinct values after rounding. It is the same mechanism that makes
`BYTE_STREAM_SPLIT` lose on this table.

Whether to accept that 6 % in exchange for making the wrap defect structurally impossible is being
decided in
[Store wind as u/v components](../roadmap/xgboost-improvements.md#store-wind-as-uv-components-rather-than-speed-and-direction).
