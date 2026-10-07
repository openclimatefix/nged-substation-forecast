---
name: polars-patito-gotchas
description: >-
  Five Patito/Polars/Delta traps that fail silently or raise a misleading error: cross-model
  LazyFrame joins, a `{column: dtype}` cast swallowed on a model-bearing frame, `ge`/`le` ignored on
  a datetime field, `pt.LazyFrame` methods typed as plain `pl.LazyFrame`, and dictionary-encoded
  columns blocking Delta predicate pushdown. Also lists Polars 2 behaviour changes that alter a
  result without raising, such as row order after a lazy join, zero-height empty frames, and the
  `cut`/`qcut` replacements. Load before writing Polars/Patito code that joins, casts, filters a
  `pt.LazyFrame`, declares a Patito field, or reads/writes Delta — or when a `validate()` dtype
  error, a `.join()` TypeError, an `invalid-assignment` on a filtered scan, an over-reading Delta
  scan, or a change in row order looks inexplicable.
---

# Patito + Polars + Delta gotchas

**None of the traps below points at itself.** Most produce no error where the mistake is, surfacing
later as a confusing `validate()` failure or a query that quietly reads the whole table; the rest
raise on the spot but name types and operations that send you after the wrong cause. That is why
they are written down.

## Cross-model LazyFrame joins

Patito creates a unique Python subclass for each model (e.g. `PowerTimeSeriesLazyFrame`,
`PowerForecastLazyFrame`). Polars' `require_same_type` check inside `.join()` rejects joining two
differently-typed Patito frames, lazy or eager, with a `TypeError`.

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
frame that carries a model (set via `.set_model(...)` or built with `Schema.DataFrame(...)`), it
casts every column to the *model's* declared dtypes. So `df.cast({"foo": pl.Int8})` on such a frame
does **not** apply your mapping — Polars' `{column: dtype}` dict is swallowed as the `strict`
argument and your `foo` cast is silently ignored while unrelated columns are reverted to model
dtypes. The result usually only surfaces later as a confusing `validate()` dtype error.

The trap fires only when the Patito model is still attached, and which operations detach it is not
guessable. Which operations detach it differs between eager and lazy frames. Measured on patito
0.8.6 with Polars 2.0.0 (Polars 1.44.2 gives the same results except for `gather_every`):

- **An eager `pt.DataFrame` keeps the Patito model through some operations and drops it through
  others, and the method name does not tell you which.** A method that Polars implements directly
  on the underlying frame keeps the model. A method that Polars runs as a lazy query and collects
  internally drops it, unless Patito overrides the method, as it does `drop` and `cast`. Which
  route a method takes can change between Polars releases. Measured, the eager frame keeps the
  model through `head`, `tail`, `slice`, `limit`, `clone`, `sample`, `rechunk`, `drop`,
  `with_row_index`, `partition_by`, `vstack`, `hstack`, `extend`, `insert_column`,
  `replace_column`, `to_dummies`, `transpose`, `null_count`, `map_rows`, indexing with a slice, a
  list of row positions, or a list of column names, iterating a `group_by`, `.cast()`, and
  `.lazy().collect()`, and on Polars 2.0.0 (not 1.44.2) through `gather_every`. It drops the model,
  returning a plain `pl.DataFrame`, through `filter`, `select`, `with_columns`, `sort`, `unique`,
  `rename`, `join`, `unpivot`, `reverse`, `drop_nulls`, `fill_null(value)`, `fill_nan`, `shift`,
  `top_k`, `bottom_k`, `explode`, `interpolate`, `update`, `group_by(...).agg(...)`,
  `pl.concat([...])`, and `.as_polars()`. Measure any method not named here before relying on
  either behaviour. A dict-`.cast` after a dropping operation applies, as on a plain Polars frame.
  A dict-`.cast` after a keeping operation is swallowed.
- **A lazy `pt.LazyFrame` keeps the Patito model through most operations and drops it through
  `group_by(...).agg(...)`, `rolling(...).agg(...)`, `group_by_dynamic(...).agg(...)`,
  `pl.concat([...])` (vertical or horizontal), and `.sql(...)`.** It keeps the model through
  `.filter()`, `.select()`, `.with_columns()`, `.sort()`, `.unique()`, `.rename()`, `.join()`,
  `.unpivot()`, `.explode()`, `.head()`, `.tail()`, `.slice()`, `.limit()`, `.clone()`,
  `.gather_every()`, `.reverse()`, `.cache()`, `.drop()`, `.with_row_index()`, `.drop_nulls()`,
  `.fill_null(value)`, `.fill_nan()`, `.shift()`, `.top_k()`, `.cast()`, and `.collect()`. Measure
  any method not named here before relying on either behaviour. A dict-`.cast` after a keeping
  operation is swallowed.

Since Polars 1.44, the eager loss is a Patito defect
([Patito issue 167](https://github.com/JakobGM/patito/issues/167)). The lost model also means that a
Patito method such as `.validate()`, called on the result of an eager `filter`, raises
`AttributeError`.

`.collect()` is the one to watch, because it reads as the boundary back into plain Polars and is
not. `pt.LazyFrame(...).set_model(S).collect().cast({"a": pl.Int8})` leaves `a` as `Int64` — no
error, no warning — where the same call on a plain frame gives `Int8`. The same swallowing happens
after any eager operation that keeps the model, such as `head`, `drop`, or `with_row_index`.

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

**On a lazy frame the `invalid-assignment` is a type-annotation gap, not a runtime one.** At runtime
a `pt.LazyFrame` keeps its model through every one of these methods — see the lists under
[`.cast({...})` on a model-bearing frame](#cast-on-a-model-bearing-frame) — so nothing is lost and
the re-wrap below exists only to satisfy the annotation. **On an eager frame the runtime differs
from the annotation.** `ty` types `df.filter(...)`, `df.select(...)`, and `df.with_columns(...)` as
`pt.DataFrame`, but since Polars 1.44 each returns a plain `pl.DataFrame` that has lost its model,
so the re-wrap below restores something real.

The gap is in `patito/polars.py`, and it is narrower than "lazy versus eager".
`patito.polars.DataFrame` re-declares `filter`, `select` and `with_columns` as `(self: DF) -> DF`,
and its own `drop`, `cast` and `fill_null` overrides return `DF` too. Those methods, on a
`pt.DataFrame`, need no workaround. **Every other method does, on either frame type**, because
`patito.polars.LazyFrame` re-declares none of them. So `df.sort(...)`, `df.head(...)`,
`df.unique()`, `df.rename(...)` and `df.with_row_index()` on an *eager* `pt.DataFrame` raise the
same `invalid-assignment` as the lazy case.

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

There is no `pt.DataFrame.from_existing`; re-wrap an eager frame with
`pt.DataFrame(df).set_model(MySchema)`. (`pt.DataFrame._from_pydf(df._df)` also works at runtime,
but `ty` types its result as a plain `pl.DataFrame` and rejects the `.set_model` call.)

The runtime and `patito/polars.py` claims in this section and the lists under `.cast` above are
measured against **patito 0.8.6 / polars 2.0.0**; `pyproject.toml` pins `polars>=1.0.0` and does
not pin `patito` at all, so re-check them after an upgrade to either. The `ty` claim needs no such
caveat: it follows from Polars' own annotations, so no checker version changes it.

## Delta Lake dictionary-encoded columns: declare Delta filter/partition columns as `String`

Cast a dictionary-encoded (`Categorical`, `Enum`) column to `String` before writing it to Delta.
Measured with deltalake 1.6.6 on Polars 2.0.0, 1.44.2, and 1.43.2 alike, `pl.DataFrame.write_delta`
panics on such a column (`cannot downcast Utf8View dictionary value to byte array`).
`write_deltalake(df.to_arrow())` succeeds, but on an `Enum` column it leaves the Delta log
recording `Utf8` while the parquet file holds a dictionary-typed column, and a later
`pl.read_delta` raises `SchemaError: data type mismatch`.
`delta_store.forecast_metrics.write_forecast_metrics` uses `write_deltalake` and casts its `Enum`
columns to `String` first for exactly this reason. Two consequences:

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

2. **For a genuinely low-cardinality column you only *read*, cast `String → Enum`/`Categorical`
   lazily** — in the `pl.scan_delta(...)` result, before `set_model` (and, if you ever filter on
   that column, filter before the cast) — so the scan is typed from the start and the cast stays
   zero-cost until `.collect()`:

    ```python
    typed_scan = pt.LazyFrame.from_existing(
        pl.scan_delta(str(path)).with_columns(
            metric_name=pl.col("metric_name").cast(pl.Enum(METRIC_NAMES)),
        )
    ).set_model(Metrics)
    ```

## Polars 2 changes that alter a result without raising

**Five Polars 2.0.0 changes give a different result from Polars 1.x without raising, and a passing
test suite does not always notice.** The full list, including the other silent changes and the
changes that raise, is in Polars' [upgrade guide](https://docs.pola.rs/releases/upgrade/2/).

- **`LazyFrame.collect()` runs on the streaming engine by default, so a lazy join, `group_by`,
  `unique`, or `unpivot` no longer returns rows in the input order by accident.** Order inside each
  group is kept, and eager `DataFrame` operations are unaffected. Pass `maintain_order="left"` to a
  join that must keep the left frame's order, or sort explicitly before any step that depends on
  row order: positional access, `unique(keep="first")`, `head`, alignment with a NumPy array, or a
  file whose row order a reader relies on. For example, `unique(keep="first")` on a lazily joined
  frame keeps an arbitrary row of each duplicate group, because the row order going in is
  arbitrary.
- **`pl.DataFrame()` has a fixed height of 0, so `pl.DataFrame().with_columns(pl.lit(1))` has no
  rows** ([upgrade
  guide](https://docs.pola.rs/releases/upgrade/2/#preserve-height-in-zero-width-dataframelazyframe-operations)).
  Build a frame from literals with `pl.select(...)`. `pl.DataFrame(height=n)` also works but is
  marked unstable.
- **`cut` and `qcut` are deprecated in favour of `bin_intervals` and `bin_quantiles`.** The new
  methods put values that sit exactly on an edge, and tied values, in different bins unless you
  pass `right_closed=True`. With `right_closed=True` they reproduce the old bins on finite values:
  `cut(breaks=...)` becomes `bin_intervals(..., right_closed=True)` and `qcut(n)` becomes
  `bin_quantiles(n, right_closed=True)`. A NaN now lands in a bin (the top one for
  `bin_intervals`) where `cut` and `qcut` gave null, and in `bin_quantiles` a NaN also moves the
  quantile edges, so drop or null out NaNs first. `labels` is a required keyword, and
  `labels=False` returns the `UInt32` bin index; there is no equivalent of `cut`'s automatic
  `"(0, 2]"` labels.
- **`explode()` now turns an empty list into zero rows, not one null row.** Pass
  `empty_as_null=True` for the old behaviour.
- **Combining a signed integer column with a `UInt64` column now gives `Int128`, not `Float64`.**
  This happens in a relaxed `concat` (`how="vertical_relaxed"` or `"diagonal_relaxed"`), in `+`, and
  in `when/then/otherwise`. A join keeps the left dtype, and a strict `concat` raises. An
  `h3_index` column held as `Int64` in one table and `UInt64` in another is the case to watch.

These are Polars changes, not Patito traps, so they do not count against the budget below.

## The friction budget

Five Patito traps is the budget. If a sixth workaround becomes necessary, revisit the approach
rather than adding it here — the alternatives are in `docs/architecture/code-style.md` under "Patito
friction budget".
