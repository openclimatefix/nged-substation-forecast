---
name: data-validation
description: >-
  How to validate a downloaded or built dataset before trusting it: an eleven-point checklist
  covering gaps in the time or space axis, duplicate keys, unexplained nulls, physically implausible
  values, stuck (unchanging) runs, spurious zeros, a step change mid-series suggesting a unit or
  product change, timezone and valid-time convention, whether a gridded field is oriented the right
  way round, whether the file size and query performance are sane for how the data will actually be
  read — plus how to check each one with Polars/xarray. Load after any `data-download` skill fetch
  finishes, before treating its output as trustworthy, and before building a study on any dataset
  nobody has looked at row by row yet.
---

# Validating a downloaded dataset

**A download script that runs without an exception is not evidence the data it wrote is right.**
Three separate bugs slipped past a clean run and a passing adversarial code review in this project's
own history: ICON-DREAM-EU's solar values were silently running means rather than hourly means
(caught only by looking at the actual numbers by hour of day, not by reading the fetch code), CERRA's
solar chunking dropped one timestamp at every chunk boundary (caught only by fetching a real chunk
and counting its distinct timestamps), and NORA3's wind direction was scaled by the wrong factor
(caught only by fetching real data and noticing the directions were implausibly small). None of the
three raised an exception, wrote a wrong row count, or failed `ruff check`. This skill is the
checklist that catches what a passing run does not.

**Run this after every `data-download` fetch completes, on the combined product file it wrote, not
on the raw download.** A study built on a dataset nobody has looked at row by row is exactly the
"corrupted feature nobody noticed" failure mode `docs/design-philosophy/inherent-stability.md` warns
about for production, applied one step earlier — at ingest, on a one-off script, is the cheap place to
catch it, not after a study has already published a number that used it.

## The eleven checks

Each check below names what to look for and how to check it against a Polars `DataFrame`/`LazyFrame`
or an `xarray.Dataset`, whichever the product's own fetch script builds. Run every check that applies
to the product's shape — a gridded field needs the orientation check, a point series does not; a
multi-source product needs the cross-correlation check, a single-source one only has itself to check
against.

**1. Gaps in the time (or lead-time) axis.** Build the expected regular timestamp range at the
product's own native frequency and anti-join against what is actually present:

```python
expected = pl.datetime_range(start, end, interval="1h", eager=True)
missing = expected.filter(~expected.is_in(frame["valid_time"]))
```

A gap that lines up with a known outage or a product's own publication schedule is expected and
worth recording in the lineage note; an unexplained gap is not. Check this per grid cell/point, not
only on the union of times across all of them — a product can look gap-free in aggregate while one
cell is missing a run no other cell is missing.

**2. Duplicate keys.** Group by the primary key the product claims (`(valid_time, cell_id)`,
`(valid_time, cell_id, model_level)`, and so on) and check every group has exactly one row:

```python
duplicate_keys = frame.group_by(key_columns).len().filter(pl.col("len") > 1)
```

A source that pads a ragged (time, step) grid to a rectangle (cfgrib does this for GRIB) can leave a
real value and a `NaN` padding row sharing the same key — filtering `is_not_nan()` before deduplicating
is what keeps the real row; a blind `unique()` can keep either one.

**3. Unexplained nulls/NaNs.** Count nulls per column and per cell/point, and look at *where* they
fall, not only how many there are — a null concentrated in one time range or one cell points at a
real gap; nulls scattered uniformly point at a fill-value that was not decoded:

```python
frame.null_count()
frame.group_by("cell_id").agg(pl.col(value_col).null_count())
```

Cross-check against the source's own documented `_FillValue`/`missing_value` — a raw fill-value
integer (e.g. `-32767`) that was never masked shows up as a wildly wrong physical value, not a null,
which the next check catches.

**4. Physically implausible values.** Check `min`/`max`/`mean` against the variable's known physical
range, not just "not null":

| Variable | Plausible range |
|---|---|
| Global/direct/diffuse horizontal irradiance | 0 to ~1200 W/m² (clear-sky max near the top of the atmosphere) |
| Wind speed at any height | 0 to ~60 m/s (a severe storm), negative is always wrong |
| Wind direction | 0 to 360°, a value outside that range is a decoding bug |
| 2 m temperature | roughly 220 to 330 K in Kelvin, or -50 to 55 °C |
| Surface pressure | roughly 850 to 1085 hPa |
| Relative humidity | 0 to 100% |

A value just outside the plausible range at the tails (e.g. 1250 W/m² once) can be a real extreme; a
whole product's mean sitting an order of magnitude off (NORA3's wind direction was previously wrong
by 10x from a scale-factor bug) is a decoding error every time.

**5. Stuck (unchanging) runs.** A sensor or a model level occasionally reports the same value for an
implausibly long run — find the longest run of an identical value per cell:

```python
frame.sort("valid_time").with_columns(
    (pl.col(value_col) != pl.col(value_col).shift(1)).cum_sum().over("cell_id").alias("run_id")
).group_by(["cell_id", "run_id"]).agg(pl.len().alias("run_length")).sort("run_length", descending=True)
```

A long run of exact zero at night for a solar variable is physically correct and not a finding; the
same run length for wind speed or temperature almost never is.

**6. Spurious zeros.** Distinguish a zero that is physically expected (solar at night) from one that
is not (wind speed exactly zero for an extended daytime run, temperature exactly zero when the
variable is stored in Kelvin). Cross-reference the zero-valued rows against solar geometry (night) or
against a second source at the same time/place before treating a run of zeros as real.

**7. A step change mid-series suggesting a unit or product change.** Plot (or compute per-month) the
mean and the standard deviation of each continuous variable across the whole series, and look for a
jump that lines up with no known product version change:

```python
frame.with_columns(pl.col("valid_time").dt.truncate("1mo").alias("month")).group_by("month").agg(
    pl.col(value_col).mean(), pl.col(value_col).std()
).sort("month")
```

A step at a *known* upgrade date (the Met Office's UKV upgrade in January 2026 is one this project has
already hit) is expected and belongs in the lineage note; a step at any other date is a decoding bug,
most often a mid-series change of units or scale factor upstream.

**8. Timezone and valid-time convention.** Confirm explicitly, not by assumption: is the timestamp
column timezone-aware or timezone-naive, and if naive, is it actually UTC? Does `valid_time` mark the
instant itself (an `analysis` product), the start of an accumulation window, or its end (a `forecast`
accumulation — CERRA's solar product marks the *end* of its 3-hour window, which is what caused the
chunk-boundary bug above)? Getting this wrong does not raise an error; it silently shifts every value
by the window length when joined against a second product with a different convention.

**9. Whether a gridded field is oriented the right way round.** A gridded product can be stored with
latitude ascending or descending, and getting this backwards silently flips the whole field
north-south (or a longitude wrap gets a field flipped east-west) without any error — the values all
still look individually plausible. Check by picking a coordinate with a known physical answer:

- A coastal point compared with an inland point a few cells away should show the coastal point's
  irradiance/temperature damped by cloud cover more often, not the reverse.
- The point closest to local solar noon (roughly UTC 12:00 minus longitude/15) should show the daily
  irradiance peak nearest that hour, not 12 hours off — a longitude axis stored in the wrong direction
  shows up as every point's solar peak landing at the wrong local time.
- Where the product carries its own `latitude`/`longitude` coordinate arrays (as CERRA's NetCDF and
  ICON-DREAM-EU's grid file do), print the first and last few coordinate values and confirm the sign
  and ordering match the documented convention before trusting any index-based crop built on them.

**10. File size and query shape are sane.** Compare the written file's size against the estimate made
before the bulk fetch (the `data-download` skill's "measure one chunk" step) — a file 10x smaller than
estimated likely dropped rows silently; a file 10x larger likely duplicated them or kept an unwanted
dimension. Then check the file answers the queries a study will actually run against it quickly:

```python
import time
t0 = time.monotonic()
pl.scan_parquet(path).filter(pl.col("cell_id") == some_cell).collect()
print(time.monotonic() - t0)
```

A per-cell time series query that takes seconds on a file with a few million rows usually means the
file's row order does not match the query pattern (see `delta_store`'s sort-order convention for the
production tables) — cheap to notice now, expensive to discover after a study is already slow.

**11. Cross-correlation with an independent source.** Where a second product covers the same
time/place (another weather model, a second variable that should physically track the first), check
the two agree in direction and rough magnitude over a sample of overlapping rows — a correlation near
zero, or strongly negative where physics says positive, usually means one of the two has a decoding,
timezone, or orientation bug rather than "the models disagree". This is the same-conclusion check the
`study` skill already runs between weather products at study time; running it once at download time
catches an obvious bug before it reaches a study at all.

## Recording what the checks found

Record the result of each applicable check — not just the failures — in the product's `lineage.json`
`extra` field (`lineage.write_lineage_note`'s `extra` argument), so a later reader does not have to
re-run the checklist to know it was run. A known, expected finding (a product-version step change, an
expected nighttime-zero run) belongs there too, worded so a later reader recognises it as already
understood rather than re-investigating it as new.
