---
name: data-validation
description: >-
  How to validate a downloaded or built dataset before trusting it: checks covering gaps in the time
  or space axis, duplicate keys, unexplained nulls and NaNs, physically implausible values, an
  hour-of-day profile that exposes a running-mean or label-convention bug, stuck (unchanging) runs,
  spurious zeros, a step change mid-series suggesting a unit or product change, internal consistency
  between related variables, timezone and valid-time convention, whether a gridded field is oriented
  the right way round, an exact expected row count, and whether the file size and query performance
  are sane — plus a runnable Polars or xarray pattern for each one. Load after any `data-download`
  skill fetch finishes, before treating its output as trustworthy, and before building a study on any
  dataset nobody has looked at row by row yet.
---

# Validating a downloaded dataset

**A download script that runs without an exception is not evidence the data it wrote is right.**
Three bugs in this project's own download work slipped past a clean run: ICON-DREAM-EU's solar
values were silently running means rather than hourly means. CERRA's solar chunking dropped one
timestamp at every chunk boundary. NORA3's wind direction was scaled by the wrong factor. None of
the three raised an exception or failed `ruff check`, and the CERRA case did write fewer rows than
expected — the miss was that nothing checked the row count against what was expected. This skill is
the checklist that catches what a passing run does not.

**Run this checklist after every `data-download` fetch completes, on the combined product file it
wrote, not on the raw download.** A study built on a dataset nobody has looked at row by row risks
the same failure this project's inherent-stability design already treats as a first-class production
risk — `docs/design-philosophy/inherent-stability.md` names a stuck meter and a false zero as
"actively misleading" rather than merely missing. Catching the equivalent fault at ingest, on a
one-off script, costs one read-through; catching it after a study has already published a number
built on it costs a re-run and a retraction.

## Before running any check, name the columns

**Declare the time column, the location columns, and the value column once, then write every check
against those names** — the columns differ by product, and a check copied verbatim from one
product's own script silently checks the wrong column on another:

| Product | Time column(s) | Location column(s) | Extra key |
|---|---|---|---|
| ICON-DREAM-EU (`fetch_icon_dream.py`) | `valid_time` | `cell_id` | `model_level` (wind only) |
| CERRA (`fetch_cerra.py`) | `valid_time` | `y_index`, `x_index` | — |
| NORA3 (`fetch_nora3.py`) | `time` | `y_index`, `x_index` | `height_m` |
| Open-Meteo (`fetch_open_meteo_grid.py`) | `time` | `point_id` | — |
| GFS/GEFS (`fetch_dynamical_zarr.py`) | `init_time`, `lead_time` | `lat_index`, `lon_index` | `ensemble_member` (GEFS) |

```python
time_col = "valid_time"
location_cols = ["cell_id"]
value_col = "ws_m_s"
```

## The checks

Run every check below that applies to the product's shape — a gridded field needs the orientation
check, a point series does not; a multi-source product needs the cross-correlation check, a
single-source one only has itself to check against.

**Gaps in the time (or lead-time) axis.** Build the expected regular timestamp range at the
product's own native frequency, matched to the column's own time unit and time zone, and anti-join
per location — not only on the union of times across every location, which can look gap-free while
one location is missing a run no other location is missing:

```python
dtype = frame.schema[time_col]  # match precision/tz, or pl.datetime_range raises on ns vs us data
expected = pl.datetime_range(start, end, interval="3h", time_unit=dtype.time_unit, eager=True)
locations = frame.select(location_cols).unique()
expected_grid = locations.join(expected.to_frame(time_col), how="cross")
missing = expected_grid.join(
    frame.select([*location_cols, time_col]), on=[*location_cols, time_col], how="anti"
)
```

A gap that lines up with a known outage or a product's own publication schedule is expected and
worth recording in the lineage note; an unexplained gap is not.

**Duplicate keys.** Group by the primary key the product claims and check every group has exactly
one row — filtering out any padding `NaN` first, per the next check, since a source that pads a
ragged (time, step) grid to a rectangle (cfgrib does this for GRIB) leaves a real value and a `NaN`
padding row sharing the same key, and a blind `unique()` can keep either one:

```python
duplicate_keys = (
    frame.filter(pl.col(value_col).is_not_nan())
    .group_by([time_col, *location_cols])
    .len()
    .filter(pl.col("len") > 1)
)
```

**Unexplained nulls and NaNs — checked separately, because Polars' own null count misses NaN.**
`frame.null_count()` reports zero on a column whose missing values are stored as float `NaN` rather
than a Polars null, which is exactly how a padded GRIB-derived frame stores them:

```python
frame.select(pl.col(pl.Float32, pl.Float64).is_nan().sum())
frame.group_by(location_cols).agg(pl.col(value_col).is_nan().sum())
frame.null_count()
```

Look at *where* the nulls or NaNs fall, not only how many there are: concentrated in one time range
or one location points at a real gap or a known padding pattern (worth recording in the lineage
note); scattered roughly uniformly across the whole series more often points at an undecoded fill
value, which the next check catches as an implausible number rather than as a null.

**Physically implausible values — bounded by physical limits, with units checked first.** A
product's stored unit is not always the SI unit: CERRA's solar variables are a 3-hour accumulation in
J/m² (up to roughly 1.1e7), Dynamical's `pressure_surface` is in Pa and `temperature_2m` is in °C,
and some humidity fields are a 0–1 fraction rather than a percentage — confirm the unit from the
product's own lineage note before comparing against a table below, or the table fires on every row of
an otherwise-correct file.

| Variable | Plausible range | Notes |
|---|---|---|
| Global horizontal irradiance | 0 to roughly 1.1x `studies.solar.extraterrestrial_horizontal` for the site and time | The top-of-atmosphere flux (about 1361 W/m² normal to the beam) sets the ceiling before atmospheric losses; a clear-sky surface value at 52°N midsummer noon is roughly 850 W/m², not the top-of-atmosphere figure itself |
| Diffuse horizontal irradiance | 0 to roughly 550 W/m² | Diffuse alone rarely approaches the global ceiling above |
| Downward longwave radiation | roughly 150 to 500 W/m² | |
| Wind speed | 0 to roughly 60 m/s; a negative value is always wrong | Applies to speed, not to a signed component |
| Wind `u`/`v` components | roughly -60 to 60 m/s | A negative value is correct here — it is a direction, not a magnitude |
| Wind direction | 0 to 360°; a distribution confined to a narrow band (e.g. 0-36°) rather than spanning close to the full circle is the signature of a scale-factor bug, not a value outside 0-360° itself | NORA3's 10x-under-scaled direction stayed inside 0-360° and was caught by this narrow-band pattern, not by an out-of-range value |
| 2 m temperature | roughly 220 to 330 K, or -50 to 55 °C | Confirm the unit first |
| Mean sea-level pressure | roughly 940 to 1085 hPa | Surface (station-level) pressure is lower over high ground by roughly 12 hPa per 100 m of elevation, so compare surface pressure against a lower floor |
| Relative humidity | roughly 0 to 100%, occasionally slightly over 100% in raw model output | Confirm whether the product stores a fraction (0-1) or a percentage before comparing |

A value just outside the plausible range at the tails (e.g. one reading at 1250 W/m²) can be a real
extreme; a whole product's mean sitting an order of magnitude off is a decoding error every time.

**An hour-of-day profile — the check that actually catches a running-mean or label-convention bug.**
None of the range, gap, or duplicate checks above would have caught ICON-DREAM-EU's solar values
being a running mean rather than an hourly one: every value was individually plausible, present, and
unique. Group by hour-of-day (in local solar time, or UTC if the trial area is small enough that the
offset does not matter) and look at the shape of the daily curve — solar irradiance should rise and
fall smoothly to a single midday peak, not sawtooth every three hours or peak at the wrong hour:

```python
frame.with_columns(pl.col(time_col).dt.hour().alias("hour")).group_by("hour").agg(
    pl.col(value_col).mean()
).sort("hour")
```

`packages/studies/src/studies/served_column_checks.py`'s
`check_hourly_value_is_a_backward_mean` and `check_direct_is_not_a_separation_model` are the tested
version of this pattern for Open-Meteo's own point downloads, referenced by the `study` skill as
something to run at fetch time; write the equivalent hour-of-day check for a product that pattern
does not already cover.

**Stuck (unchanging) runs.** A sensor or a model level occasionally reports the same value for an
implausibly long run. Keep the value itself in the result, not only the run length, so a reviewer can
tell a legitimate run (an exact zero through the night) from a suspicious one, and remember that
13-bit significand rounding (the `data-download` skill's convention) legitimately produces repeated
values for a slowly-varying field — mean sea-level pressure moves by roughly 12 Pa between
consecutive rounded steps, for instance:

```python
frame.sort(time_col).with_columns(
    (pl.col(value_col) != pl.col(value_col).shift(1)).cum_sum().over(location_cols).alias("run_id")
).group_by([*location_cols, "run_id"]).agg(
    pl.len().alias("run_length"), pl.col(value_col).first().alias("value")
).sort("run_length", descending=True)
```

**Spurious zeros.** Distinguish a zero that is physically expected (solar at night) from one that is
not (wind speed exactly zero for an extended daytime run). Cross-reference the zero-valued rows
against solar geometry (night) or against a second source at the same time and location before
treating a run of zeros as real.

**A step change mid-series suggesting a unit or product change.** Compute the mean and standard
deviation per month across the whole series and look for a jump that lines up with no known product
version change:

```python
frame.with_columns(pl.col(time_col).dt.truncate("1mo").alias("month")).group_by("month").agg(
    pl.col(value_col).mean().alias("mean"), pl.col(value_col).std().alias("std")
).sort("month")
```

Filter out any padding NaN first (per the nulls/NaNs check above), or a month containing padding
rows reports a mean of `NaN` rather than a number. A step at a *known* upgrade date — the Met
Office's UKV upgrade on 21 January 2026 is one this project has already hit — is expected and belongs
in the lineage note; a step at any other date should be treated as a decoding bug until shown
otherwise, since an upstream assimilation change can also produce one without being announced
anywhere this project would see.

**Internal consistency between related variables.** Where a product serves more than one physically
related field, check they agree with each other, not only that each one individually looks
plausible: direct irradiance should never exceed global, and global should be close to direct plus
diffuse; a product serving both wind components and a derived speed should have speed close to
`hypot(u, v)`; wind speed at 100 m should usually exceed the same hour's speed at 10 m. A field that
individually passes the plausibility table above but fails this cross-check is still wrong.

**Timezone and valid-time convention.** Confirm explicitly, not by assumption: is the timestamp
column timezone-aware or timezone-naive, and if naive, is it actually UTC? Does the time column mark
the instant itself (an `analysis` product), the start of an accumulation window, or its end (a
`forecast` accumulation — CERRA's solar product marks the *end* of its 3-hour window, which is what
caused the chunk-boundary bug this skill's introduction cites)? The `study` skill's own convention —
"radiation is a mean over the hour before its label; wind is an instantaneous value at its label" —
is the reference to check a new product against, rather than restating it here. Getting this wrong
does not raise an error; it silently shifts every value by the window length when joined against a
second product with a different convention.

**Whether a gridded field is oriented the right way round.** A gridded product can be stored with
latitude ascending or descending, and getting this backwards silently flips the whole field
north-south without any error — the values all still look individually plausible. **Never print or
log the actual coordinate values to check this on a product cropped to the private trial-area box** —
assert the ordering programmatically instead:

```python
assert np.all(np.diff(latitude) < 0) or np.all(np.diff(latitude) > 0)  # monotonic, either direction
```

Confirm which direction matches the source's own documentation (check on the whole-domain grid file
before any private cropping, where printing a coordinate value reveals nothing about the trial area).
A 0-360 versus -180-180 longitude convention mismatch shifts or mis-crops a box rather than flipping
it — `fetch_cerra.py`'s longitude normalisation is the fix for that case, not an orientation fix.
Beyond the axis-order assertion, a coastal-versus-inland or solar-noon heuristic is not reliable at
this project's own grid spacing and box size: the trial area spans only a few degrees of longitude,
so a reversed axis would shift solar noon by minutes, invisible at hourly resolution, and UK coastal
sites are not reliably cloudier than inland ones. Trust the programmatic assertion over an eyeballed
physical pattern.

**An exact expected row count.** Compute the row count a correct fetch should have produced —
`(distinct timestamps) x (distinct locations) x (distinct levels or members, if any)` — and compare
it against the actual row count, not only against a size estimate. This is the check that would have
caught CERRA's one-dropped-timestamp-per-chunk bug directly: the file's overall size barely moved,
but its row count was short by exactly the number of dropped chunk boundaries. For a forecast
product specifically, also check every `init_time` carries the full set of lead times (and the full
ensemble-member count, for GEFS) that the request asked for.

**File size and query performance are sane.** Compare the written file's size against the estimate
made before the bulk fetch (the `data-download` skill's "measure one chunk" step) — a file far
smaller than estimated likely dropped rows silently; a file far larger likely duplicated them or kept
an unwanted dimension. Then check the file answers the query a study will actually run against it in
a reasonable time:

```python
import time

t0 = time.monotonic()
pl.scan_parquet(path).filter(pl.col(location_cols[0]) == some_value).collect()
print(time.monotonic() - t0)
```

A per-location time series query that takes several seconds on a file of a few million rows usually
means the file's row order does not match the query pattern — see `delta_store`'s sort-order
convention for the production tables for the fix.

**Cross-correlation with an independent source.** Where a second product covers the same time and
place (another weather model, a second variable that should physically track the first), check the
two agree in direction and rough magnitude over a sample of overlapping rows — a correlation near
zero, or strongly negative where physics says positive, usually means one of the two has a decoding,
timezone, or orientation bug rather than "the models disagree". The `study` skill already runs this
same comparison between weather products at study time ("Look for steps in the served data before
attributing a gap to a model… Plot each product's series against a sibling product's"); running it
once at download time catches an obvious bug before it ever reaches a study.

## Recording what the checks found

Record the result of each applicable check — not just the failures — somewhere a later reader will
find it before re-running the checklist from scratch. `lineage.write_lineage_note` (or
`write_lineage_note` with a `filename` override for a product that already writes one lineage note
per variable) overwrites the whole note each time it is called, so pass every field again, including
`extra`, rather than assuming a partial call merges into what is already on disk; where the checks
run as a separate step from the fetch itself, write a sibling `validation.json` next to the lineage
note instead of trying to append to it. A known, expected finding (a product-version step change, an
expected nighttime-zero run) belongs there too, worded so a later reader recognises it as already
understood rather than re-investigating it as new.
