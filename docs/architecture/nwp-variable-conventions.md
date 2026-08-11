# NWP variable conventions

How to read a value out of the `nwp` Delta table, and what the feature pipeline does to it on the
way to the model. Every ECMWF ENS variable we store is listed here with the convention that governs
it, because those conventions are not uniform: some values describe an instant, some describe the
six hours before that instant, and one of them is an angle that wraps.

**This page and [Known ECMWF ENS data-quality issues](ecmwf-ens-known-issues.md) divide the subject
between them, and the split is worth stating plainly because the two are easy to confuse:**

- **That page is about *upstream*.** It covers the ways Dynamical.org's data arrives damaged —
  corrupt source accumulations, null grid points, incomplete runs — and the ingest policy that
  decides which of those we reject and which we tolerate. Its subject is data we did not create and
  cannot fix.

- **This page is about *us*.** It covers how the data we successfully ingested is meant to be
  interpreted, and what our own code does with it. Nothing here is upstream's fault; everything here
  is a choice this project made, or failed to make deliberately.

A defect described on that page is a reason to talk to Dynamical. A defect described on this page is
a reason to change our code.

## The forecast step grid

ECMWF ENS is **not evenly spaced over the horizon**. Measured on the 2026-08-10 00Z run, and
identical in every run in the archive:

| step width | lead times | number of steps |
|---|---|---|
| — | 0 h (analysis) | 1 |
| 3 h | 3 h → 144 h | 48 |
| 6 h | 150 h → 360 h | 36 |

The change at **144 h — six days** is the single most consequential fact on this page. The primary
user band is 3–10 days ([XGBoost improvements](../roadmap/xgboost-improvements.md)), so days 6 to 10
of that band sit on the coarse half of the grid.

## The three conventions

A variable is governed by one convention from each of the axes below. Most variables are
"instantaneous, linear"; the exceptions are what make this page necessary.

### Instantaneous versus period-ending

An **instantaneous** variable describes conditions at `valid_time` itself.

A **period-ending** variable is the *average rate over the interval that ends at `valid_time`*.
Dynamical.org de-accumulates these from ECMWF's cumulative source fields before we receive them.
Three variables are period-ending: `downward_short_wave_radiation_flux_surface`,
`downward_long_wave_radiation_flux_surface` and `precipitation_surface`. They are legitimately null
at lead 0, because there is no preceding interval.

The consequence that catches people: beyond 144 h the interval is **six hours long**, so a 12:00
shortwave value is the mean over 06:00–12:00 — which straddles a large part of the morning ramp and
is therefore much lower than the instantaneous irradiance at noon. Nothing about the weather changes
at day 6; the *question the number answers* changes.

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

Whether it is *also* period-ending is **unconfirmed**. The schema documents the convention for the
three de-accumulated variables and is silent on this one, and it cannot be settled from our own
data. If it turns out to describe the preceding interval, forward-filling it attributes one
interval's precipitation type to the following one. Confirm against ECMWF's parameter database or
with Dynamical before relying on its timing.

## Every variable, and how to read it

The final column measures **what the 3 h → 6 h step change costs**: the error, at the held-out
intermediate steps, from reconstructing the 3-hourly part of the horizon out of a 6-hourly version
of itself the way the feature pipeline reconstructs half-hourly rows. `MAE/SD` normalises that error
by the variable's own spread, so the column is comparable across variables. Measured on the
2026-08-10 00Z run, leads 3–144 h, 4,090,608 rows. For the period-ending variables the 6-hourly
product is *exactly* reproducible by averaging consecutive pairs of 3-hourly means, so their figures
are exact rather than an emulation.

| variable | convention | unit | MAE at 6-hourly | MAE/SD |
|---|---|---|---|---|
| `downward_short_wave_radiation_flux_surface` | period-ending | W m⁻² | 111 | **0.44** |
| `wind_direction_10m` | instantaneous, **circular** | ° | 20.3 | — |
| `wind_direction_100m` | instantaneous, **circular** | ° | 18.8 | — |
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

The bottom three rows matter as much as the top ones: **the synoptic variables are genuinely
unharmed by the coarser grid**, so there is nothing to fix there and no reason to spend effort on
it. Shortwave radiation is three times worse than anything else that is not an angle.

## What the half-hourly resample does today

`_upsample_nwp_to_half_hourly` builds a 30-minute grid per `(nwp_init_time, ensemble_member,
time_series_id)` group, left-joins the native steps onto it, then **linearly interpolates every
column in `Nwp.continuous_var_names()`** and forward-fills the categorical one.

`continuous_var_names()` is defined as "every weather variable that is not categorical". That single
definition is the root of two of the three defects below: it silently classifies a period-ending
rate and a wrapped angle as things that may be linearly interpolated between their `valid_time`
stamps.

All three defects are fixed as the first work in v0.5 — see
[Fix the NWP resample to honour the variable conventions](../roadmap/xgboost-improvements.md#fix-the-nwp-resample-to-honour-the-variable-conventions).

### Wind direction is interpolated across the 0°/360° wrap

Interpolating 350° → 10° yields 180°: due south for a northerly wind. Measured on the 2026-08-10
00Z run, over every consecutive native-step pair:

| step width | pairs straddling north (`_10m`) | pairs straddling north (`_100m`) |
|---|---|---|
| 3 h | 3.73 % | 3.50 % |
| 6 h | 8.30 % | 8.03 % |

Whenever a pair straddles, the interpolated midpoint is wrong by **exactly 180°** — the long way
round is always diametrically opposite the short way. Across the whole horizon that leaves
**6.57 %** of half-hourly `wind_direction_10m` rows and **6.32 %** of `wind_direction_100m` rows
pointing the wrong way, and the error is not random: it is systematically northerlies being recorded
as southerlies.

Both columns are in `conf/model/xgboost.yaml`'s `selected_features`, so this reaches the model.

### Period-ending variables are interpolated as though they were instantaneous

The resample interpolates between `valid_time` stamps, which treats a backward-looking mean as an
instantaneous reading at the *end* of its window. The result is both flattened and shifted late.
Reconstructing a clear-sky day at 52.5° N in mid-August:

| step | reconstruction | peak W m⁻² | peak time |
|---|---|---|---|
| — | truth (instantaneous) | 816 | 12:00 |
| 3 h | today's resample | 756 | 15:00 |
| 6 h | today's resample | 590 | **18:00** |

Beyond day 6 the model is told the solar day peaks at 18:00 UTC. Daily *energy* is preserved
(7.18 against 7.20 kWh m⁻²) — it is the shape and the timing that are wrong, which is precisely what
a PV site responds to.

That the period-ending reading is the correct one is not taken on trust from the schema. Forming the
clear-sky index two ways over the 2026-08-10 00Z run (1671 cells × 51 members, 7.16 M rows) settles
it, because a clear-sky index above 1 is physically impossible:

| step | `GHI / CS(valid_time)` p99 / max | `GHI / CS(mean over preceding step)` p99 / max |
|---|---|---|
| 3 h | 2.02 / 2.22 | 0.99 / **1.02** |
| 6 h | 2.27 / 3.04 | 0.96 / **1.03** |

The period-mean reading caps at 1.0 in both halves of the horizon, which also confirms that nothing
*other* than the window width changes at 144 h.

### Instantaneous variables lose diurnal amplitude

Linear interpolation between 6-hourly samples cuts the corner off a diurnal cycle, because the
~15:00 daily temperature maximum falls between the 12:00 and 18:00 samples. Measured on the same
run, the mean daily temperature range falls from **5.01 °C to 4.43 °C — an 11.6 % compression**.

Unlike the other two defects this one has no exact fix: four samples a day of a once-a-day cycle is
at the Nyquist limit, so some amplitude is genuinely unrecoverable. Linear interpolation is
nonetheless the worst reasonable choice, and it feeds the effective-temperature, degree-day and
`windchill` features.

## Wind is stored as speed and direction, and why

The ingest computes wind speed and direction from ECMWF's native `u`/`v` components and discards the
components ([`convert_to_polars.py`](../api/dynamical_data/index.md)). Two properties of that choice
are worth knowing.

**The conversion happens after the spatial aggregation, which is the right order.** H3 cells are
built from `wind_u_*`/`wind_v_*`, and speed and direction are derived from the already-aggregated
components. Averaging *direction* over grid points would have the same wrap defect in space that the
resample has in time; it does not, because that averaging never happens.

**The stored speed is therefore the magnitude of the cell's *vector* mean, not the mean of the grid
points' scalar speeds.** These differ — |E[**v**]| ≤ E[|**v**|] — and the gap grows wherever wind
direction varies across a cell. The second quantity is the one a turbine power curve wants, because
power responds to the speed at each point and the curve is strongly non-linear. The scalar mean
cannot be recovered from the archive: it is a different spatial reduction, so obtaining it requires
re-ingesting. The size of the gap has not been measured.

**Storing the components instead would cost roughly 6 % of the table.** The polar and Cartesian
forms carry identical information — the round trip is exact, with a maximum error of
3.05 × 10⁻⁵ ° in direction and 1.91 × 10⁻⁶ m s⁻¹ in speed over a full run, far inside the
[13-bit significand budget](../api/delta_store/index.md) the table already accepts — so this is
purely a storage question. Written through the production path (same significand rounding, sort
order and writer properties), the full column set both ways:

| run | speed + direction | u + v | change |
|---|---|---|---|
| 2026-08-10 00Z | 142.6 MB | 151.6 MB | +6.3 % |
| 2026-08-03 00Z | 144.6 MB | 152.8 MB | +5.7 % |

This is the opposite of the intuition that a wrapped angle must compress badly. Direction *is* the
most expensive column in the table — 1.57 bytes per row each, the two directions together 19.3 % of
all bytes — but the significand rounding collapses values into repeats that Parquet's dictionary and
RLE encoding capture, and `u`/`v`, being products of a speed and a trigonometric function, spread
across more distinct values. It is the same mechanism that makes `BYTE_STREAM_SPLIT` lose on this
table.

Whether to accept that 6 % in exchange for making the wrap defect structurally impossible is being
decided in
[Store wind as u/v components](../roadmap/xgboost-improvements.md#store-wind-as-uv-components-rather-than-speed-and-direction).
