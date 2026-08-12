---
name: polars-patito-gotchas
description: >-
  Five Patito/Polars/Delta traps that fail silently rather than raising: cross-model LazyFrame
  joins, a `{column: dtype}` cast swallowed on a model-bearing frame, `ge`/`le` ignored on a
  datetime field, `pt.LazyFrame` methods typed as plain `pl.LazyFrame`, and dictionary-encoded
  columns blocking Delta predicate pushdown. Load before writing Polars/Patito code that joins,
  casts, filters a `pt.LazyFrame`, declares a Patito field, or reads/writes Delta — or when a
  `validate()` dtype error, a `.join()` TypeError, an `invalid-assignment` on a filtered scan, or
  an over-reading Delta scan looks inexplicable.
---

# Patito + Polars + Delta gotchas

**None of the traps below points at itself.** Most produce no error where the mistake is, surfacing
later as a confusing `validate()` failure or a query that quietly reads the whole table; the rest
raise on the spot but name types and operations that send you after the wrong cause. That is why
they are written down.

## Cross-model LazyFrame joins

Patito creates a unique Python subclass for each model (e.g. `PowerTimeSeriesLazyFrame`,
`PowerForecastLazyFrame`). Polars' `assert_same_type` check inside `.join()` rejects joining
two differently-typed Patito LazyFrames with a `TypeError`.

Workaround: strip the Patito subclass from the right-hand operand before joining:

```python
# Strip Patito model annotation so Polars' cross-subclass type check doesn't reject the join
plain_lf = pl.LazyFrame._from_pyldf(patito_lf._ldf)
left_patito_lf.join(plain_lf.select(...), on=..., how="inner")
```

`pl.LazyFrame._from_pyldf` constructs a plain `pl.LazyFrame` from the same underlying Rust
object — zero-copy, no data movement. The check passes because `type(left_lf)` is a subclass
of `pl.LazyFrame`, so `isinstance(left_lf, type(plain_lf))` is `True`.

## `.cast({...})` on a model-bearing frame

Patito **overrides** `.cast`: its signature is `cast(self, strict=False, columns=None)` and, on a
frame that carries a model (set via `.set_model(...)` or a typed `pt.DataFrame[Schema]`), it casts
every column to the *model's* declared dtypes. So `df.cast({"foo": pl.Int8})` on such a frame does
**not** apply your mapping — Polars' `{column: dtype}` dict is swallowed as the `strict` argument
and your `foo` cast is silently ignored while unrelated columns are reverted to model dtypes. The
result usually only surfaces later as a confusing `validate()` dtype error.

The trap fires only when the model is still attached. Many Polars ops **drop** the model
(`group_by(...).agg(...)`, `.collect()`, `.unpivot()`, `.as_polars()`), so a dict-`.cast` after
them is plain Polars and fine. But **iterating** `group_by` (`for k, g in df.group_by(...)`) yields
sub-frames that **keep** the model, and `pl.concat` keeps it too — so a dict-`.cast` on the
concatenated result hits the trap.

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
by reading the `minimum`/`maximum` keywords out of the Pydantic JSON schema, and JSON Schema
defines those keywords for numbers only — so a datetime field's `Ge`/`Le` metadata never reaches
the JSON schema, Patito finds no keyword to turn into a filter, and `validate()` accepts every
year. There is no warning and no error; the constraint simply does not exist. (`ge`/`le` on a
numeric field works exactly as documented, which is what makes this so easy to miss.)

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

**This is a type-annotation gap, not a runtime one.** At runtime the model survives: `.filter()`,
`.sort()`, `.select()`, `.with_columns()`, `.head()` and `.unique()` on a `pt.LazyFrame` all return
the model-bearing subclass with `.model` still set. Nothing is lost and nothing needs restoring —
the re-wrap below exists only to satisfy the annotation.

The asymmetry is in `patito/polars.py`: `patito.polars.DataFrame` carries a block of type-annotation
overrides re-declaring `filter`/`select`/`with_columns` as `(self: DF) -> DF`, and
`patito.polars.LazyFrame` has no such block, so it inherits Polars' own annotations. **`pt.DataFrame`
is therefore unaffected** — `df.filter(...)` stays `pt.DataFrame[Schema]` to `ty` as well as at
runtime, and needs no workaround.

Workaround, for the lazy case only: rebind to a plain `pl.LazyFrame` local for the filter
accumulation, then re-wrap before returning:

```python
def apply(self, scan: pt.LazyFrame[MySchema]) -> pt.LazyFrame[MySchema]:
    lf: pl.LazyFrame = scan  # .filter() is typed as plain pl.LazyFrame; accumulate on one
    if self.foo is not None:
        lf = lf.filter(pl.col("foo") == self.foo)
    return pt.LazyFrame.from_existing(lf).set_model(MySchema)  # zero-copy re-wrap
```

## Delta Lake dictionary-encoded columns: declare Delta filter/partition columns as `String`

delta-rs stores all Arrow dictionary-encoded columns (`Categorical`, `Enum`) as plain `String` in
Parquet (this is the write-path gotcha documented in `_write_metrics_to_delta`, which casts the
remaining `Enum` columns to `String` before writing). Two consequences:

1. **A contract column you filter or partition on in Delta should be `String`, not `Categorical`.**
   If the schema declared it `Categorical`, every read would need a `String → Categorical` cast to
   satisfy the model — and a cast placed between `pl.scan_delta(...)` and a `.filter()` on that
   column **blocks predicate pushdown** (Polars can no longer prune Delta partitions or skip row
   groups, so it reads the *whole* table even when the filter names one partition). Declaring the
   column `String` matches what is on disk, so the scan is typed by `set_model` with no cast, the
   filter pushes straight down, and there is no dtype tension at the write boundary either.
   `PowerForecast.experiment_name` / `fold_id` (the `power_forecasts` partition columns) and
   `power_fcst_model_name` are `String` for exactly this reason; `PopulationFilter.apply` therefore
   takes and returns a typed `pt.LazyFrame[PowerForecast]`. Confirm pushdown with `.explain()` — it
   should list only the matching `partition=value` paths.

2. **For a genuinely low-cardinality column you only *read* (never filter on), cast `String →
   Enum`/`Categorical` lazily** — in the `pl.scan_delta(...)` result, before `set_model` — so the
   scan is typed from the start and the cast stays zero-cost until `.collect()`:

    ```python
    typed_scan = pt.LazyFrame.from_existing(
        pl.scan_delta(str(path)).with_columns(
            metric_name=pl.col("metric_name").cast(pl.Enum(METRIC_NAMES)),
        )
    ).set_model(MetricsSchema)
    ```

## The friction budget

Five is the budget. If a sixth workaround becomes necessary, revisit the approach rather than
adding it here — the alternatives are in `docs/architecture/code-style.md` under "Patito friction
budget".
