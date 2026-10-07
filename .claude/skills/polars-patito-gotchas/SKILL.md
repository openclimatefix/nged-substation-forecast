---
name: polars-patito-gotchas
description: >-
  Five Patito/Polars/Delta traps that fail silently rather than raising: cross-model LazyFrame
  joins, a `{column: dtype}` cast swallowed on a model-bearing frame, `ge`/`le` ignored on a
  datetime field, `pt.LazyFrame` methods typed as plain `pl.LazyFrame`, and dictionary-encoded
  columns blocking Delta predicate pushdown. Also lists the Polars 2 behaviour changes that alter
  a result without raising: row order after a lazy join, zero-height empty frames, and the
  `cut`/`qcut` replacements. Load before writing Polars/Patito code that joins, casts, filters a
  `pt.LazyFrame`, declares a Patito field, or reads/writes Delta — or when a `validate()` dtype
  error, a `.join()` TypeError, an `invalid-assignment` on a filtered scan, an over-reading Delta
  scan, or rows in a different order than before looks inexplicable.
---

# Patito + Polars + Delta gotchas

**None of the traps below points at itself.** Most produce no error where the mistake is, surfacing
later as a confusing `validate()` failure or a query that quietly reads the whole table; the rest
raise on the spot but name types and operations that send you after the wrong cause. That is why
they are written down.

## Cross-model LazyFrame joins

Patito creates a unique Python subclass for each model (e.g. `PowerTimeSeriesLazyFrame`,
`PowerForecastLazyFrame`). Polars' `assert_same_type` check inside `.join()` rejects joining two
differently-typed Patito LazyFrames with a `TypeError`.

Workaround: strip the Patito subclass from the right-hand operand before joining:

```python
# Strip Patito model annotation so Polars' cross-subclass type check doesn't reject the join
plain_lf = pl.LazyFrame._from_pyldf(patito_lf._ldf)
left_patito_lf.join(plain_lf.select(...), on=..., how="inner")
```

`pl.LazyFrame._from_pyldf` constructs a plain `pl.LazyFrame` from the same underlying Rust object —
zero-copy, no data movement. The check passes because `type(left_lf)` is a subclass of
`pl.LazyFrame`, so `isinstance(left_lf, type(plain_lf))` is `True`.

## `.cast({...})` on a model-bearing frame

Patito **overrides** `.cast`: its signature is `cast(self, strict=False, columns=None)` and, on a
frame that carries a model (set via `.set_model(...)` or a typed `pt.DataFrame[Schema]`), it casts
every column to the *model's* declared dtypes. So `df.cast({"foo": pl.Int8})` on such a frame does
**not** apply your mapping — Polars' `{column: dtype}` dict is swallowed as the `strict` argument
and your `foo` cast is silently ignored while unrelated columns are reverted to model dtypes. The
result usually only surfaces later as a confusing `validate()` dtype error.

The trap fires only when the Patito model is still attached, and which operations detach it is not
guessable. It differs between eager and lazy frames. Measured on patito 0.8.6 with Polars 2.0.0
(Polars 1.44.2 gives identical results):

- **An eager `pt.DataFrame` keeps the model only through `head`, `drop`, and `with_row_index`.**
  Every other operation returns a plain `pl.DataFrame`: `filter`, `select`, `with_columns`, `sort`,
  `unique`, `rename`, `join`, `unpivot`, `group_by(...).agg(...)`, `pl.concat([...])`, and
  `.as_polars()`. A dict-`.cast` after one of those is plain Polars and fine. *Iterating* a
  `group_by` yields frames that keep the model. Since Polars 1.44 this loss is a Patito defect
  ([Patito issue 167](https://github.com/JakobGM/patito/issues/167)), and it also means a Patito
  method such as `.validate()` called on such a result raises `AttributeError`.
- **A lazy `pt.LazyFrame` drops the model only through `group_by(...).agg(...)` and
  `pl.concat([...])`.** Every other operation keeps it, including `.filter()`, `.select()`,
  `.with_columns()`, `.sort()`, `.head()`, `.unique()`, `.drop()`, `.rename()`, `.join()`,
  `.with_row_index()`, and `.unpivot()`. A dict-`.cast` after any of those is swallowed.

`.collect()` is the one to watch, because it reads as the boundary back into plain Polars and is
not. `pt.LazyFrame(...).set_model(S).collect().cast({"a": pl.Int8})` leaves `a` as `Int64` — no
error, no warning — where the same call on a plain frame gives `Int8`. The same swallowing happens
after an eager `head`.

Workaround: strip the Patito model before a `{column: dtype}` cast (mirrors the join gotcha above):

```python
# Strip the Patito model so the dict-cast uses plain Polars semantics (zero-copy)
result = pl.DataFrame._from_pydf(patito_df._df).cast({"foo": pl.Categorical})
```

(No-arg `df.cast()` — casting a model-bearing frame to its declared dtypes — *is* the intended
Patito use and is correct. Expression/Series casts like `pl.col("foo").cast(pl.Int8)` are always
plain Polars and unaffected.)

This is the caveat behind the Polars style rule in `docs/architecture/code-style.md` that prefers
`df.cast({"foo": pl.Int8})` over `df.with_columns(pl.col("foo").cast(pl.Int8))`: the preference
holds only on a plain Polars frame.

## `ge`/`le` are silently ignored on a datetime field

`pt.Field(ge=..., le=...)` enforces nothing on a `datetime` column. Patito builds its bounds checks
by reading the `minimum`/`maximum` keywords out of the Pydantic JSON schema, and JSON Schema defines
those keywords for numbers only — so a datetime field's `Ge`/`Le` metadata never reaches the JSON
schema, Patito finds no keyword to turn into a filter, and `validate()` accepts every year. There is
no warning and no error; the constraint simply does not exist. (`ge`/`le` on a numeric field works
exactly as documented, which is what makes this so easy to miss.)

**How to apply:** bound a datetime column from the model's `validate` override, not from the field.
`contracts.common.check_datetime_bounds` is the shared helper, and `MIN_PLAUSIBLE_DATETIME` /
`MAX_PLAUSIBLE_DATETIME` are the shared bounds; `PowerTimeSeries.validate` and `Nwp.validate` are
the worked examples. A `constraints=` Polars expression on the field also works, but its failure
message is the generic "1 row does not match custom constraints", so prefer the explicit check when
you want the error to say which bound was broken.

## `pt.LazyFrame` methods are *typed* as plain `pl.LazyFrame`

`ty` types `scan.filter(...)` on a `scan: pt.LazyFrame[Schema]` as
`polars.lazyframe.frame.LazyFrame`, so reassigning `scan = scan.filter(...)` fails its assignment
check:

```text
error[invalid-assignment]: Object of type `polars.lazyframe.frame.LazyFrame`
is not assignable to `patito.polars.LazyFrame[PowerForecast]`
```

**On a lazy frame this is a type-annotation gap, not a runtime one.** At runtime a `pt.LazyFrame`
keeps its model through every one of these methods — see the lists under [`.cast({...})` on a
model-bearing frame](#cast-on-a-model-bearing-frame) — so nothing is lost and the re-wrap below
exists only to satisfy the annotation. **On an eager frame the runtime differs from the annotation.**
`ty` types `df.filter(...)`, `df.select(...)`, and `df.with_columns(...)` as `pt.DataFrame`, but
since Polars 1.44 each returns a plain `pl.DataFrame` that has lost its model, so the re-wrap
below restores something real.

The gap is in `patito/polars.py`, and it is narrower than "lazy versus eager".
`patito.polars.DataFrame` carries a block of type-annotation overrides re-declaring exactly three
methods — `filter`, `select` and `with_columns` — as `(self: DF) -> DF`. Those three, on a
`pt.DataFrame`, need no workaround. **Every other method on either frame type does**, because
`patito.polars.LazyFrame` has no such block at all and `DataFrame`'s block stops at three. So
`df.sort(...)`, `df.head(...)`, `df.unique()` and `df.rename(...)` on an *eager* `pt.DataFrame`
raise the same `invalid-assignment` as the lazy case. (`df.drop(...)` happens not to, because Polars
annotates `DataFrame.drop` as returning `Self`.)

**Upgrading `ty` will not fix this, because `ty` is not wrong.** Polars annotates
`LazyFrame.filter`, `.sort`, `.select`, `.with_columns`, `.head`, `.unique`, `.drop` and `.rename`
as returning `LazyFrame`, not `Self`, so every conforming checker must infer the base class — and
`pyright` reports the identical error. The fix has to come from upstream: either Polars switching
those return annotations to `Self`, or Patito giving its `LazyFrame` the same override block its
`DataFrame` already has. Until one of those lands, the re-wrap below is the workaround.

Workaround: rebind to a plain local for the accumulation, then re-wrap before returning.

```python
def apply(self, scan: pt.LazyFrame[MySchema]) -> pt.LazyFrame[MySchema]:
    lf: pl.LazyFrame = scan  # .filter() is typed as plain pl.LazyFrame; accumulate on one
    if self.foo is not None:
        lf = lf.filter(pl.col("foo") == self.foo)
    return pt.LazyFrame.from_existing(lf).set_model(MySchema)  # zero-copy re-wrap
```

There is no `pt.DataFrame.from_existing`, so the eager re-wrap spells the same thing as
`pt.DataFrame._from_pydf(df._df).set_model(MySchema)`.

The runtime and `patito/polars.py` claims in this section and the lists under `.cast` above are
measured against **patito 0.8.6 / polars 2.0.0**; `pyproject.toml` pins `polars>=1.0.0` and does
not pin `patito` at all, so re-check them after an upgrade to either. The `ty` claim needs no such caveat:
it follows from Polars' own annotations, so no checker version changes it.

## Delta Lake dictionary-encoded columns: declare Delta filter/partition columns as `String`

Delta Lake cannot store a dictionary-encoded (`Categorical`, `Enum`) column, so cast such a column
to `String` before writing. Measured on Polars 1.43.2, `write_deltalake` without the cast
succeeded, leaving the Delta log recording `Utf8` while the parquet file held a dictionary-typed
column, and a later `pl.read_delta` raised `SchemaError: data type mismatch`. On Polars 2.0.0
with deltalake 1.6.6, `write_delta` of an `Enum` or `Categorical` column panics at write time
instead (`cannot downcast Utf8View dictionary value to byte array`).
`delta_store.forecast_metrics.write_forecast_metrics` casts its `Enum` columns to `String` before
writing for exactly this reason. Two consequences:

1. **A contract column you filter or partition on in Delta should be `String`, not `Categorical`.**
   If the schema declared it `Categorical`, every read would need a `String → Categorical` cast to
   satisfy the model — and a cast placed between `pl.scan_delta(...)` and a `.filter()` on that
   column **blocks predicate pushdown** (Polars can no longer prune Delta partitions or skip row
   groups, so it reads the *whole* table even when the filter names one partition). Declaring the
   column `String` matches what is on disk, so the scan is typed by `set_model` with no cast, the
   filter pushes straight down, and there is no dtype tension at the write boundary either.
   Polars does not see through the cast. Measured on Polars 2.0.0 and 1.44.2 on a table
   partitioned by one column, a filter placed after `.cast(pl.Enum(...))` leaves a separate
   `FILTER` above a scan of every partition, whether the filter compares with a string, an `Enum`
   literal, or `is_in`. Filtering on the `String` column *before* the cast prunes to the one
   partition again.
   `PowerForecast.experiment_name` / `fold_id` (the `power_forecasts` partition columns) and
   `power_fcst_model_name` are `String` for exactly this reason; `PopulationFilter.apply` therefore
   takes and returns a typed `pt.LazyFrame[PowerForecast]`. Confirm pushdown with `.explain()` — it
   should list only the matching `partition=value` paths.

2. **For a genuinely low-cardinality column you only *read* (never filter on), cast `String →
   Enum`/`Categorical` lazily** — in the `pl.scan_delta(...)` result, before `set_model`, and after
   any filter on that column — so the scan is typed from the start and the cast stays zero-cost
   until `.collect()`:

    ```python
    typed_scan = pt.LazyFrame.from_existing(
        pl.scan_delta(str(path)).with_columns(
            metric_name=pl.col("metric_name").cast(pl.Enum(METRIC_NAMES)),
        )
    ).set_model(MetricsSchema)
    ```

## Polars 2 changes that alter a result without raising

**Polars 2.0.0 changed five behaviours that give a different result where Polars 1.x gave the
same one, and a passing test suite does not always notice.** The full list, including the changes
that raise, is in Polars' [upgrade guide](https://docs.pola.rs/releases/upgrade/2/). These five are
the ones that bit this repository or that the tests cannot see.

- **`LazyFrame.collect()` runs on the streaming engine by default, so row order after a join,
  `group_by`, `unique`, or `unpivot` is no longer guaranteed.** Order inside each group is kept, and
  eager `DataFrame` operations are unaffected. Sort explicitly before any step that depends on row
  order: positional access, `unique(keep="first")`, `head`, alignment with a NumPy array, or a
  file whose row order a reader relies on. For example, `unique(keep="first")` on a lazily joined
  frame keeps an arbitrary row of each duplicate group, because the row order going in is
  arbitrary.
- **`pl.DataFrame()` has a fixed height of 0, so `pl.DataFrame().with_columns(pl.lit(1))` has no
  rows** ([upgrade
  guide](https://docs.pola.rs/releases/upgrade/2/#preserve-height-in-zero-width-dataframelazyframe-operations)).
  Build a frame from literals with `pl.select(...)`. `pl.DataFrame(height=n)` also works but is
  marked unstable.
- **`cut` and `qcut` are deprecated in favour of `bin_intervals` and `bin_quantiles`.** The new
  methods require a `labels` argument and put values that sit exactly on an edge, and tied values,
  in different bins unless you pass `right_closed=True`. With `right_closed=True` they reproduce
  the old bins exactly: `cut(breaks=...)` becomes `bin_intervals(..., right_closed=True)` and
  `qcut(n)` becomes `bin_quantiles(n, right_closed=True)`.
- **Headerless CSV column names start at `column_0`, and `read_csv` now behaves like `scan_csv`**
  ([upgrade
  guide](https://docs.pola.rs/releases/upgrade/2/#start-csv-column-name-counting-at-0)). A
  `new_columns` list shorter than the file's column count now raises.
- **Combining a signed integer column with a `UInt64` column now gives `Int128`, not
  `Float64`.** A relaxed `concat` of `Int64` and `UInt64` columns, such as an `h3_index` column
  held as `Int64` in one table and `UInt64` in another, changes dtype silently.

These are Polars changes, not Patito traps, so they do not count against the budget below.

## The friction budget

Five Patito traps is the budget. If a sixth workaround becomes necessary, revisit the approach
rather than adding it here — the alternatives are in `docs/architecture/code-style.md` under "Patito
friction budget".
