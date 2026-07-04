# NWP migration: Int16 affine quantisation → Float32 + significand rounding

> **For the implementing agent (no memory of prior conversation):** this is the mechanical
> checklist for a single PR. Read this whole file before touching code — the phase ordering in
> "Implementation phases" is load-bearing (see "Why this order" there). This file is deleted
> when the PR merges, per `plans/README.md` convention; before deleting it, do the "Ship-time
> triage" section at the bottom.
>
> **Closes:** [#198](https://github.com/openclimatefix/nged-substation-forecast/issues/198)
> (NWP row-group layout blocks member/cell pruning on reads).
> **Supersedes:** `plans/nwp-significand-rounding-experiment.md` (deleted — its measured
> results are folded into this file so nothing is lost).

## 1. Context (self-contained — you don't need the prior conversation)

The `nwp` Delta table (`data/NWP`, ~88 GB locally, `init_time`-partitioned, ECMWF ENS ensemble
weather data) currently stores every continuous weather variable as `Int16`: each value is
affine-quantised into `[0, 4095]` (12 bits) using per-variable `(min, max)` ranges read from
`metadata/scaling_params_for_ecmwf_ens_0_25_degree.csv`, and rescaled back to `Float32` physical
units in memory on read. This requires two Patito contracts (`NwpOnDisk`, `NwpInMemory`) plus a
`NwpScalingParams` model, and a `to_nwp_in_memory()` / `from_nwp_in_memory()` conversion pair.

This is the same problem `power_forecasts` solved differently: instead of affine quantisation,
`power_forecasts` stores plain `Float32` values rounded to a small number of significand bits
(`delta_store.precision.round_to_significand_bits`, Veltkamp splitting — zeroes the low,
incompressible mantissa bits so ZSTD compresses well, at a bounded *relative* error). See
`packages/delta_store/src/delta_store/power_forecasts.py` for the working example this
migration follows.

An experiment (branch `experiment/nwp-significand-rounding`, results below) measured whether the
same technique works for NWP, and separately, whether reordering the within-file row sort could
fix a *second*, independent problem: today's sort (`init_time → valid_time → ensemble_member →
h3_index`) puts every ensemble member in every Parquet row group, so a single-member read (the
control-member read every training run does) cannot skip row groups via min/max stats and must
decode-then-filter the whole partition. That's issue #198.

**Both experiments came back favourable — adopt.**

### Measured results (carry these numbers forward; do not re-derive)

**Storage** (9 real partitions spread across every season, 7,243,785 rows each):

| Config | avg MB/partition | extrapolated GB/yr |
|---|---:|---:|
| Baseline (today): Int16, ZSTD-14 | 115.3 | 42.1 |
| **Float32 + significand, `keep_bits=13`, plain ZSTD-3** | **110.6** | **40.4** |
| Same, but member-early sort (the one we're adopting) | 112.9 | 41.2 |
| Same + `BYTE_STREAM_SPLIT` (mirrors `power_forecasts`) | 133.8 | 48.8 |

**`BYTE_STREAM_SPLIT` makes this table *worse*, unlike `power_forecasts` — do not use it.**
Per-column inspection showed every continuous variable individually larger under
`BYTE_STREAM_SPLIT` (e.g. `temperature_2m` roughly doubled). Working hypothesis: significand
rounding collapses NWP values into a small set of repeats (many H3 cells / ensemble members land
on the same rounded value), which Parquet's *default* dictionary+RLE encoding captures directly;
`BYTE_STREAM_SPLIT` scatters that repetition across four separate byte planes and loses more
than it gains. `power_forecasts`' target values have no such repetition (near-continuous ML
output), so `BYTE_STREAM_SPLIT` wins there instead. **The lesson, not just the number: writer
properties are data-dependent and must be re-measured per table — don't copy `power_forecasts`'
choice blindly. Do not "improve" this by adding `BYTE_STREAM_SPLIT` back in.**

Max observed error at `keep_bits=13` (bound 2⁻¹³ ≈ 1.22×10⁻⁴ relative — confirmed not exceeded):

| Variable | max abs error | vs. tolerance |
|---|---:|---|
| `temperature_2m` (°C) | 0.0039 °C | ≤ 0.25 K required — 60× margin |
| `pressure_reduced_to_mean_sea_level` / `pressure_surface` (Pa) | 8 Pa (0.08 hPa) | ≤ 1 hPa required — 12× margin |
| all other variables | ≤ 0.5 in native units | (no formal threshold; comfortably below sensor/ECMWF-grid resolution) |

(The old plan worried Kelvin-scale temperature would be "offset-dominated" and lose ~4.6× more
precision per bit than pressure. That doesn't apply: `temperature_2m` is stored in **Celsius**,
not Kelvin, so its magnitude — and hence its significand-rounding error — is an order of
magnitude smaller than that worst case. Don't second-guess this in code review; it's measured.)

**Read path** (control-member, 9-cell — the real V1 `h3_res_5` set — whole-month streaming
collect, real June 2025 data, `keep_bits=13`, run twice with order swapped to rule out
page-cache artefacts):

| Sort order | wall time | peak RSS |
|---|---:|---:|
| Current (`valid_time` before `ensemble_member`) | 0.15 s | 882–1138 MB |
| **Member-early (`ensemble_member` before `valid_time`)** | **0.02–0.04 s** | **203–206 MB** |

**~5× faster, ~5× less peak memory**, for a ~2% storage cost (110.6 → 112.9 MB/partition
above). Mechanism: with `ensemble_member` sorted early, each ~1M-row Parquet row group spans
only a handful of member values instead of all 51, so row-group min/max stats let the reader
skip most groups outright for a single-member predicate.

### Decision

Adopt: `Float32` + `round_to_significand_bits(keep_bits=13)`, written with plain
`WriterProperties(compression="ZSTD", compression_level=3)` (no per-column encoding overrides),
sorted `(init_time, ensemble_member, valid_time, h3_index)` — i.e. **swap `ensemble_member` and
`valid_time`** relative to today's sort.

## 2. Scope

**In scope:**

- Collapse `NwpOnDisk` + `NwpInMemory` + `NwpScalingParams` into one `Nwp` Patito contract.
- New `delta_store.nwp` module (writer properties, sort cols, `keep_bits` constant, `write_nwp()`)
  mirroring `delta_store.power_forecasts`.
- Update every call site: `ecmwf_ens` asset (write path), `_load_engineering_inputs` (read
  path), `ml_core` feature-engineering modules (type hints only), tests, docs.
- One-off rewrite of the local `data/NWP` table (~88 GB, 810+ daily partitions) into the new
  format — building a **fresh table then swapping directories**, not rewriting in place (Delta
  cannot change a column's dtype partition-by-partition; see Phase B for the measured failure
  modes).
- Delete now-dead code: `NwpOnDisk`, `NwpInMemory`, `NwpScalingParams`,
  `to_nwp_in_memory`/`from_nwp_in_memory`, `packages/dynamical_data/scripts/compute_scaling_params.py`,
  `metadata/scaling_params_for_ecmwf_ens_0_25_degree.csv`.

**Out of scope (do not do these in this PR):**

- Any change to `power_forecasts`' storage format (already shipped, PR #268).
- Re-tuning `keep_bits` or writer properties further (e.g. trying plain-ZSTD at level 14, or
  `keep_bits=12`/`10` for extra compression) — `keep_bits=13` was chosen for consistency with
  `POWER_FCST_SIGNIFICAND_BITS`, and the decision-gate thresholds are already cleared with
  large margin at 13. If you want to explore this, propose it separately; don't fold it into
  this migration.
- Migrating any cloud/production copy of the NWP table — there isn't one yet. `Settings.nwp_data_path`
  defaults to the local `data/NWP` directory and that's the only copy that exists (v0.1 AWS
  deployment is separate, unstarted work).

## 3. Implementation phases

**Do these in order.** Do not delete the old contract code before Phase B has been verified —
Phase B's rewrite script needs `NwpOnDisk.scan_delta()` / `NwpOnDisk.to_nwp_in_memory()` (the
*old* code) to read the *old* on-disk format; it writes through the *new* `delta_store.nwp.write_nwp()`.
If you delete the old classes first, there is nothing left that can read the table you're
migrating away from.

### Why this order

1. **Phase A (additive)** adds the new contract + writer alongside the old ones. Nothing is
   deleted, nothing else changes. This gives Phase B a `write_nwp()` to write through, while the
   old classes are still present to read with.
2. **Phase B (data rewrite)** builds a replacement table partition-by-partition, verifies it
   against the untouched old table, then swaps directories. Old code reads, new code writes.
   The old table is never written to — it stays the recovery point until the final delete
   (checklist below).
3. **Phase C (cutover)** repoints every call site at the new contract/writer and deletes the old
   code. By this point the on-disk table is already in the new format, so there's no window
   where code and data disagree.
4. **Phase D (verify + ship)** runs the full test suite, lints, and opens the PR.

### Phase A — additive: new contract + writer, old code untouched

**A1. `packages/contracts/src/contracts/weather_schemas.py`** — add a new `Nwp` model.

Keep `NwpMetaData`, `NwpModelId`, `NWP_MODEL_ID_DTYPE`, `WeatherFeature`, `_NWP_METADATA_CSV_PATH`
untouched. Add `Nwp` as a new top-level class (don't yet touch `_NwpBase` / `NwpInMemory` /
`NwpOnDisk` / `NwpScalingParams` — those are deleted in Phase C once the rewrite is verified).
`Nwp` is exactly `_NwpBase`'s fields plus `NwpInMemory`'s continuous-variable fields and
validation logic, with a `scan_delta` classmethod (no rescale step — the table is already
`Float32`):

```python
class Nwp(pt.Model):
    """Weather data schema for NWP forecasts: gridded ECMWF ENS ensemble weather, one row per
    (nwp_model_id, init_time, valid_time, ensemble_member, h3_index).

    Stored on disk as plain Float32, rounded to a significand-bit budget by
    `delta_store.nwp.write_nwp` — see `docs/architecture/overview.md` for the physical format
    and measured numbers (replaces the earlier Int16 affine-quantisation scheme).
    """

    nwp_model_id: str = pt.Field(
        dtype=NWP_MODEL_ID_DTYPE,
        description="The primary key for joining with NwpMetaData (e.g. 'ECMWF_ENS_0_25_degree').",
    )
    init_time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description="When the NWP model run was initialised.",
    )
    valid_time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description=(
            "The time for which this NWP value is valid. Most variables (temperature, wind, "
            "pressure, geopotential) are instantaneous — they describe conditions at this moment. "
            "Precipitation and radiation (downward_long_wave_radiation_flux_surface, "
            "downward_short_wave_radiation_flux_surface, precipitation_surface) are period-ending "
            "rates: each value represents the average rate over the period that ends at valid_time "
            "(i.e. the preceding forecast step interval). Dynamical.org de-accumulates these from "
            "ECMWF's raw cumulative fields before we receive them."
        ),
    )
    ensemble_member: int = pt.Field(dtype=pl.UInt8, description="Ensemble member index (0-based).")
    h3_index: int = pt.Field(
        dtype=pl.UInt64,
        description="H3 cell index. The H3 resolution for the nwp_model_id is stored in NwpMetaData.",
    )

    # Copy every `temperature_2m` ... `precipitation_surface` field verbatim from the current
    # `NwpInMemory` (same names, dtypes, descriptions, ge/le bounds) — they don't change.
    temperature_2m: float = pt.Field(dtype=pl.Float32, description="...", ge=-100, le=100)
    dew_point_temperature_2m: float = pt.Field(dtype=pl.Float32, description="...", ge=-100, le=100)
    wind_speed_10m: float = pt.Field(dtype=pl.Float32, description="...", ge=0, le=200)
    wind_direction_10m: float = pt.Field(dtype=pl.Float32, description="...", ge=0, le=360)
    wind_speed_100m: float = pt.Field(dtype=pl.Float32, description="...", ge=0, le=200)
    wind_direction_100m: float = pt.Field(dtype=pl.Float32, description="...", ge=0, le=360)
    pressure_surface: float = pt.Field(dtype=pl.Float32, description="...", ge=0, le=200_000)
    pressure_reduced_to_mean_sea_level: float = pt.Field(
        dtype=pl.Float32, description="...", ge=0, le=200_000
    )
    geopotential_height_500hpa: float = pt.Field(dtype=pl.Float32, description="...", ge=0, le=10_000)
    downward_long_wave_radiation_flux_surface: float | None = pt.Field(
        dtype=pl.Float32, description="...", ge=0, le=1500
    )
    downward_short_wave_radiation_flux_surface: float | None = pt.Field(
        dtype=pl.Float32, description="...", ge=0, le=1500
    )
    precipitation_surface: float | None = pt.Field(dtype=pl.Float32, description="...", ge=0, le=0.01)

    categorical_precipitation_type_surface: int | None = pt.Field(
        dtype=pl.UInt8,
        description="... (copy verbatim from NwpInMemory, including the codes-list docstring) ...",
    )

    categorical_var_names: ClassVar[frozenset[str]] = frozenset(
        {"categorical_precipitation_type_surface"}
    )
    _non_var_column_names: ClassVar[frozenset[str]] = frozenset(
        {"nwp_model_id", "init_time", "valid_time", "ensemble_member", "h3_index"}
    )

    @classmethod
    def all_weather_var_names(cls) -> frozenset[str]:
        """All meteorological variable field names (continuous + categorical)."""
        return frozenset(cls.model_fields) - cls._non_var_column_names

    @classmethod
    def continuous_var_names(cls) -> frozenset[str]:
        """Meteorological variable field names suitable for linear interpolation."""
        return cls.all_weather_var_names() - cls.categorical_var_names

    # Copy `validate()` and its three private `_check_*` helpers verbatim from `NwpInMemory` —
    # unchanged logic, just renamed onto `Nwp`.
    @classmethod
    def validate(cls, dataframe, columns=None, allow_missing_columns=False,
                 allow_superfluous_columns=False, drop_superfluous_columns=False):
        ...  # body identical to today's NwpInMemory.validate

    @classmethod
    def scan_delta(cls, path: Path = SETTINGS.nwp_data_path) -> pt.LazyFrame[Self]:
        """Lazily scan the NWP Delta table, typed and cast to this contract's dtypes.

        No rescale step — unlike the old `NwpOnDisk`/`NwpInMemory` pair, the table stores
        physical-unit `Float32` directly (see `delta_store.nwp.write_nwp`).
        """
        return pt.LazyFrame.from_existing(pl.scan_delta(str(path))).set_model(cls).cast()
```

Do **not** delete `_NwpBase`, `NwpInMemory`, `NwpOnDisk`, `NwpScalingParams`,
`_SCALING_PARAMS_FOR_ECMWF_ENS_0_25_DEGREE_CSV_PATH`, or `_NWP_ON_DISK_DTYPE` /
`_NWP_ON_DISK_MAX_INT_VALUE` yet — Phase B needs them.

**A2. New file `packages/delta_store/src/delta_store/nwp.py`:**

```python
"""Storage policy for the ``nwp`` Delta table.

Owns everything about how ``Nwp`` rows are laid out on disk: the parquet writer properties, the
compression-friendly row order, and the significand-precision reduction of the continuous
weather variables. Callers write through :func:`write_nwp` so it is impossible to land rows in
the table without this format applied.

Replaces the earlier ``Int16`` affine-quantisation scheme (``NwpOnDisk`` / ``NwpInMemory`` /
``NwpScalingParams``) with plain ``Float32`` + ``delta_store.precision.round_to_significand_bits``
— the technique used by ``delta_store.power_forecasts``, but with *different* writer properties.
Measured on real NWP data (9 partitions spread across two years of history):
``BYTE_STREAM_SPLIT`` made every continuous column *larger*, not smaller — the opposite of the
``power_forecasts`` result. Working hypothesis: significand rounding collapses NWP values into a
small set of repeats (many H3 cells / ensemble members round to the same value), which Parquet's
*default* dictionary+RLE encoding captures directly; ``BYTE_STREAM_SPLIT`` scatters that
repetition across four separate byte planes and loses more than it gains.
``power_forecasts``'s target values have no such repetition (near-continuous ML output), so
``BYTE_STREAM_SPLIT`` wins there instead — the two tables need different writer properties.
See `docs/architecture/overview.md` for the measured GB/yr and read-latency numbers.
"""

from pathlib import Path
from typing import Final

import patito as pt
import polars as pl
from contracts.weather_schemas import Nwp
from deltalake import WriterProperties, write_deltalake

from delta_store.precision import round_to_significand_bits

NWP_SIGNIFICAND_BITS: Final[int] = 13
"""Significand bits kept for every continuous NWP variable (1 implicit + 12 explicit fraction
bits) — the same budget as ``delta_store.power_forecasts.POWER_FCST_SIGNIFICAND_BITS``. Caps the
relative error at 2⁻¹³ ≈ 1.2×10⁻⁴. Measured max absolute error on real data: ≤0.004°C for
temperature, ≤8 Pa (0.08 hPa) for mean-sea-level pressure — both well inside tolerance
(temperature ≤0.25 K, MSL pressure ≤1 hPa)."""

NWP_SORT_COLS: Final[tuple[str, ...]] = ("init_time", "ensemble_member", "valid_time", "h3_index")
"""Within-file row order for ``nwp`` writes — **member before valid_time** (the opposite
priority from ``power_forecasts``, which sorts member-adjacent for a different reason: there
it's about compressing near-duplicate ensemble values; here it's about row-group pruning).
Sorting ``ensemble_member`` early means each ~1M-row Parquet row group spans only a handful of
member values instead of all ~51, so a single-member predicate (the control-member read every
training run does) can skip most row groups via min/max stats instead of decoding the whole
partition. Measured on a real 29-day/9-cell/control-member read: ~5x faster, ~5x less peak
memory, than the previous ``valid_time``-first sort, for a ~2% storage cost. Closes
https://github.com/openclimatefix/nged-substation-forecast/issues/198."""

NWP_WRITER_PROPERTIES: Final[WriterProperties] = WriterProperties(
    compression="ZSTD", compression_level=3
)
"""Deliberately **no** per-column encoding overrides (no ``BYTE_STREAM_SPLIT``,
``DELTA_BINARY_PACKED``, or disabled dictionary encoding) — see this module's docstring for why
that choice, which won for ``power_forecasts``, measures worse here."""


def write_nwp(nwp: pt.DataFrame[Nwp], table_uri: str | Path) -> None:
    """Append ``Nwp`` rows to the ``nwp`` Delta table in its storage format.

    Rounds every continuous weather variable to ``NWP_SIGNIFICAND_BITS`` significand bits, sorts
    rows by ``NWP_SORT_COLS``, and writes with ``NWP_WRITER_PROPERTIES``. The table is
    partitioned by ``(nwp_model_id, init_time)``, matching ``Nwp.scan_delta``'s
    partition-pruning assumptions; the first write creates the table.

    Append-only: each ``(nwp_model_id, init_time)`` partition is written exactly once — the
    daily ``ecmwf_ens`` asset downloads one brand-new NWP run per Dagster partition. (No
    ``replace_partition`` option like ``write_power_forecasts``: nothing re-materialises an
    existing NWP partition today, and a partition-replace predicate on a ``Timestamp`` partition
    column would need its own careful verification — add it only when a caller actually needs
    it.)

    Args:
        nwp: Validated NWP rows for a single ``(nwp_model_id, init_time)`` partition.
        table_uri: Path or URI of the ``nwp`` Delta table.
    """
    continuous_vars = sorted(Nwp.continuous_var_names())
    rounded = nwp.with_columns(
        **{
            v: round_to_significand_bits(pl.col(v), keep_bits=NWP_SIGNIFICAND_BITS)
            for v in continuous_vars
        }
    ).sort(*NWP_SORT_COLS)

    # Strip the Patito model before the dict-cast: `nwp_model_id` is declared `Enum` for
    # in-memory type safety, but delta-rs can't store `Enum`/`Categorical` (see the "Delta Lake
    # dictionary-encoded columns" gotcha in CLAUDE.md). A dict-cast on a *model-bearing* frame
    # would silently no-op this cast and revert other columns to the model's declared dtypes
    # instead of applying it — strip first so this is a plain-Polars cast.
    prepared = pl.DataFrame._from_pydf(rounded._df).cast({"nwp_model_id": pl.String}).to_arrow()

    write_deltalake(
        table_or_uri=table_uri,
        data=prepared,
        mode="append",
        partition_by=["nwp_model_id", "init_time"],
        writer_properties=NWP_WRITER_PROPERTIES,
    )
```

**A3. `packages/delta_store/src/delta_store/__init__.py`** — add the new export:

```python
from delta_store.nwp import write_nwp
from delta_store.power_forecasts import write_power_forecasts
from delta_store.precision import round_to_significand_bits

__all__ = ["round_to_significand_bits", "write_nwp", "write_power_forecasts"]
```

Update the module docstring's "One module per table (currently `power_forecasts`)" line to
"(currently `power_forecasts` and `nwp`)".

**A4. New tests: `packages/delta_store/tests/test_nwp.py`.** Mirror
`packages/delta_store/tests/test_power_forecasts.py` structure exactly (a `_make_nwp(...)`
builder + `test_on_disk_format` asserting ZSTD compression and **no**
`BYTE_STREAM_SPLIT`/`DELTA_BINARY_PACKED` encodings + sort order via
`pl.struct(NWP_SORT_COLS).is_sorted()`, `test_continuous_vars_rounded_to_significand_bits`
mirroring `test_power_fcst_rounded_to_significand_bits`'s bit-pattern assertion but looping over
`Nwp.continuous_var_names()`, and — since `write_nwp` is append-only, *not* a mirror of the
replace-partition test — `test_successive_appends_create_separate_partitions`: two `write_nwp`
calls with different `init_time`s land as two Hive partitions (`init_time=...` directories) with
both partitions' rows intact). Use small made-up rows well inside each field's `ge`/`le` bounds.

**Checkpoint:** run `uv run pytest packages/contracts packages/delta_store` — should be all
green, nothing else in the repo has changed yet.

### Phase B — rewrite the historical local table (fresh table + directory swap)

This rewrites `data/NWP` (~88 GB, partitioned by `(nwp_model_id, init_time)`, 810+ daily
partitions as of 2026-07-04) from `Int16` to the new `Float32` format, using the *old* code
(still present after Phase A) to read and the *new* `write_nwp()` to write.

**Strategy: build a brand-new Delta table at a sibling path, verify it, swap directories,
delete the old table.** Do **not** attempt to rewrite the existing table in place,
partition-by-partition — a Delta table has a single schema, and this migration *changes column
dtypes*, so in-place is impossible. Both in-place variants were tested empirically
(delta-rs 1.6.1, 2026-07-04) and both fail dangerously:

- `mode="overwrite"` + partition `predicate` with `Float32` data: delta-rs **silently casts the
  incoming data back to the table's `Int16` schema** — no error, row counts match, but
  physical-unit floats get stored as truncated integer "codes" that the old read path would then
  mis-rescale. Silent data corruption that a row-count check cannot catch.
- Adding `schema_mode="overwrite"`: the commit succeeds and flips the table schema to `Float32`,
  but every not-yet-migrated partition still holds `Int16` parquet — **all scans of the table
  then error** (`SchemaError: data type mismatch ... incoming: Int16 != target: Float32`),
  including the migration's own verification reads.

The fresh-table approach also removes, *by construction*, the version-pinning gotcha that bit
the `power_forecasts` rewrite on 2026-07-04 (an unpinned read of a table being overwritten by
the same script saw partition 1's overwrite and returned 0 pre-rewrite rows for partition 2 —
nearly dropping 10.7M rows): here reads only touch the old table and writes only touch the new
one, so the script cannot read its own writes. The old table is never modified — it *is* the
rollback plan until the final `rm`.

Disk headroom: the new table is approximately the same size as the old (~88–91 GB; see the
storage table in §1), so both coexisting needs ~90 GB free — `df -h` showed 178 GB available on
2026-07-04. Re-check before running.

**Run this as a throwaway script — do not commit it to the repo.** (There's no repo precedent
for keeping one-off rewrite scripts around; the equivalent `power_forecasts` rewrite wasn't
committed either.) Save it anywhere outside the git tree (e.g. your scratch/tmp directory), run
it once, then discard it.

```python
"""One-off: rebuild data/NWP as Float32 + significand rounding, at a sibling path.

Run once, after Phase A has landed `Nwp` / `delta_store.nwp.write_nwp` (needed to write) and
before Phase C deletes `NwpOnDisk` / `NwpScalingParams` (needed to read the old format).
Writes to data/NWP_migrating; the swap into place is a manual step afterwards (see checklist).
"""

import patito as pt
import polars as pl
from contracts.settings import Settings
from contracts.weather_schemas import NwpOnDisk, NwpScalingParams
from delta_store.nwp import write_nwp

SETTINGS = Settings()
OLD_TABLE = str(SETTINGS.nwp_data_path)
NEW_TABLE = str(SETTINGS.nwp_data_path.with_name("NWP_migrating"))


def main() -> None:
    scaling_params = NwpScalingParams.load()

    # engine="streaming" for every whole-table query: the table holds ~1.6e9 rows, so even a
    # single projected column OOMs the in-memory engine (measured: exit 137).
    init_times = (
        pl.scan_delta(OLD_TABLE)
        .select("init_time")
        .unique()
        .sort("init_time")
        .collect(engine="streaming")["init_time"]
        .to_list()
    )
    print(f"{len(init_times)} partitions to migrate {OLD_TABLE} -> {NEW_TABLE}")

    for i, init_time in enumerate(init_times):
        old_lf = pl.scan_delta(OLD_TABLE).filter(pl.col("init_time") == init_time)
        old_lf = pt.LazyFrame.from_existing(old_lf).set_model(NwpOnDisk).cast()
        old_df = old_lf.collect(engine="streaming")

        # NwpInMemory-shaped at runtime; write_nwp only needs the (identical) column names/dtypes,
        # so the pt.DataFrame[Nwp] annotation mismatch is harmless in this throwaway script.
        nwp_in_memory = NwpOnDisk.to_nwp_in_memory(old_df, scaling_params)

        write_nwp(nwp_in_memory, NEW_TABLE)  # append; the first call creates the table

        new_rows = (
            pl.scan_delta(NEW_TABLE)
            .filter(pl.col("init_time") == init_time)
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )
        assert new_rows == old_df.height, (
            f"ROW COUNT MISMATCH at {init_time}: old={old_df.height} new={new_rows}"
        )
        print(f"[{i + 1}/{len(init_times)}] {init_time}: {old_df.height:,} rows verified")

    print("All partitions written and row-counted. Old table untouched.")
    print("Next: spot-check relative error, then swap directories (see plan checklist).")


if __name__ == "__main__":
    main()
```

**Checklist — do not skip any of these:**

- [ ] `df -h`: ≥ ~95 GB free before starting (old + new tables coexist until the final delete).
- [ ] The script must **never write to the old table path** — reads from `OLD_TABLE`, writes to
      `NEW_TABLE`, nothing else. (This is what makes the power_forecasts version-pinning gotcha
      structurally impossible here.)
- [ ] Per-partition row counts asserted in the loop (done above), plus a whole-table total:
      `pl.scan_delta(OLD).select(pl.len())` == `pl.scan_delta(NEW).select(pl.len())`.
- [ ] Spot-check relative error for a few partitions spread across the date range: rescale the
      old partition with `NwpOnDisk.to_nwp_in_memory` and compare against the new table —
      `abs(old_float - new_float) <= 2**-13 * abs(old_float)` for every continuous var (where
      `old_float == 0`, require `new_float == 0` exactly; skip nulls). This is what catches
      silent value corruption — row counts alone cannot.
- [ ] Swap: `mv data/NWP data/NWP_old_int16 && mv data/NWP_migrating data/NWP`. (Note the new
      table's `_delta_log` starts from version 0 — the old table's time-travel history is
      intentionally left behind in `NWP_old_int16`.)
- [ ] Run the Phase D3 read smoke-check against the swapped-in table **before** deleting
      anything.
- [ ] Only then, the point of no return: `rm -rf data/NWP_old_int16`.
- [ ] `du -sh data/NWP` sanity check: 810+ partitions × ~113 MB ≈ 88–92 GB expected for ~2.25
      years of history (matches the ~41 GB/yr extrapolation in §1 — a sanity check, not a new
      experiment).

**Checkpoint:** `data/NWP` is now physically `Float32` + significand-rounded, member-early
sorted. The *code* (Phase A only landed so far) still contains the old `NwpOnDisk` /
`NwpInMemory` classes and nothing calls the new `Nwp.scan_delta()` yet — that's fine, nothing
reads the table between here and Phase C.

### Phase C — cutover: repoint call sites, delete old code

**C1. `packages/dynamical_data/src/dynamical_data/ecmwf_ens/convert_to_polars.py`:**

- `from contracts.weather_schemas import NWP_MODEL_ID_DTYPE, NwpInMemory, NwpModelId` →
  `from contracts.weather_schemas import NWP_MODEL_ID_DTYPE, Nwp, NwpModelId`.
- Return type `pt.DataFrame[NwpInMemory]` → `pt.DataFrame[Nwp]` (function signature and
  docstring return description).
- Remove the compression-driven sort and its comment:

  ```python
  # Sort before validation to ensure consistent output order, and to optimise compression.
  df = df.sort(by=["init_time", "valid_time", "ensemble_member", "h3_index"])
  ```

  Physical row order is now `delta_store.nwp`'s job (single source of truth for on-disk
  layout — see the `delta_store` package docstring), not the conversion step's. Just call
  `return Nwp.validate(df)` directly (drop the intermediate `df = df.sort(...)` line; validation
  doesn't require sorted input — `_check_unique` uses `is_duplicated()`, order-independent).
- `NwpInMemory.validate(df)` → `Nwp.validate(df)`; `NwpInMemory.categorical_var_names` →
  `Nwp.categorical_var_names` (used in `_process_chunk_for_1_lead_time_and_1_ens_member`).

**C2. `src/nged_substation_forecast/defs/assets.py`** — the `ecmwf_ens` asset:

```python
from contracts.weather_schemas import Nwp  # was: NwpOnDisk, NwpScalingParams
from delta_store.nwp import write_nwp
# Remove: from deltalake import WriterProperties, write_deltalake
#   (grep confirms these are only used inside `ecmwf_ens` in this file — remove the import
#   entirely, don't leave it unused)

@asset(...)
def ecmwf_ens(context: AssetExecutionContext) -> None:
    """..."""
    settings = Settings()
    partition_date_str = context.partition_key
    nwp_init_time = datetime.strptime(partition_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    h3_grid = pt.DataFrame(pl.read_parquet(settings.h3_grid_weights_path)).set_model(H3GridWeights)

    ds = download_ecmwf_ens_run(nwp_init_time=nwp_init_time, h3_grid=h3_grid)
    nwp = convert_nwp_xarray_dataset_to_polars_dataframe(ds=ds, h3_grid=h3_grid)

    context.log.info(f"Columns: {nwp.columns}")

    settings.nwp_data_path.parent.mkdir(parents=True, exist_ok=True)
    write_nwp(nwp, settings.nwp_data_path)

    context.add_output_metadata(
        {
            "n_rows": len(nwp),
            "path": str(settings.nwp_data_path),
            "init_time": str(nwp_init_time),
        }
    )
```

Removed: `scaling_params = NwpScalingParams.load()`, `nwp_on_disk =
NwpOnDisk.from_nwp_in_memory(nwp_in_memory, scaling_params)`, the
`.cast({"nwp_model_id": pl.String})` line (now inside `write_nwp`), and the manual
`write_deltalake(...)` call with its inline `WriterProperties(compression="ZSTD",
compression_level=14)`.

Also update the asset **docstring** (elided as `"""..."""` above): it currently says
"converts it to a Polars DataFrame, scales it to integer representation, and appends it to the
Delta table" — there is no integer scaling step any more; say it writes through
`delta_store.nwp.write_nwp` (Float32, significand-rounded).

**C3. `src/nged_substation_forecast/defs/cv_assets.py`** — `_load_engineering_inputs`:

```python
from contracts.weather_schemas import Nwp  # was: NwpInMemory, NwpOnDisk
```

Replace:

```python
nwp_scan = NwpOnDisk.scan_delta(settings.nwp_data_path).filter(
    pl.col("init_time") >= init_time_start,
    pl.col("init_time") <= init_time_end,
    pl.col("valid_time") >= window_start,
    pl.col("valid_time") <= window_end,
    pl.col("h3_index").is_in(cells),
)
if ensemble_members is not None:
    nwp_scan = nwp_scan.filter(pl.col("ensemble_member").is_in(ensemble_members))
nwp_on_disk = pt.LazyFrame.from_existing(nwp_scan).set_model(NwpOnDisk)
nwp_lf = NwpOnDisk.to_nwp_in_memory(nwp_on_disk)
```

with:

```python
nwp_scan = Nwp.scan_delta(settings.nwp_data_path).filter(
    pl.col("init_time") >= init_time_start,
    pl.col("init_time") <= init_time_end,
    pl.col("valid_time") >= window_start,
    pl.col("valid_time") <= window_end,
    pl.col("h3_index").is_in(cells),
)
if ensemble_members is not None:
    nwp_scan = nwp_scan.filter(pl.col("ensemble_member").is_in(ensemble_members))
nwp_lf = pt.LazyFrame.from_existing(nwp_scan).set_model(Nwp)
```

Update the function's return type annotation (`pt.LazyFrame[NwpInMemory]` → `pt.LazyFrame[Nwp]`)
and docstring: the "**Memory: prune the NWP scan at the source**" paragraph currently says
"applied to the *raw* `NwpOnDisk.scan_delta` **before** `NwpOnDisk.to_nwp_in_memory` rescales
Int16→Float32" — there's no rescale step any more, so simplify to "applied directly to the
`Nwp.scan_delta` scan, so only the surviving rows are ever decoded." The `ensemble_member` bullet
("applied before the rescale so we never decode the ~50 members we discard") loses its rescale
framing too — reword to "applied at the scan so we never decode the ~50 discarded members; the
member-early sort (`delta_store.nwp.NWP_SORT_COLS`) additionally lets Parquet row-group stats
skip most of the file outright for this predicate — see `docs/architecture/overview.md`."

**C4. `packages/ml_core/src/ml_core/features/_nwp.py`, `tabular_feature_engineer.py`,
`feature_engineer.py`:** mechanical rename, `NwpInMemory` → `Nwp`, in imports and every
`pt.LazyFrame[NwpInMemory]` / `pt.DataFrame[NwpInMemory]` type hint. In
`tabular_feature_engineer.py`'s docstring (~line 105), the phrase "from Delta Lake should
convert with `NwpOnDisk.to_nwp_in_memory()` first" needs rewording — there's no conversion step
any more; reword to "from Delta Lake comes directly from `Nwp.scan_delta()`".

**C5. Delete now-dead code:**

- `packages/contracts/src/contracts/weather_schemas.py`: delete `_NwpBase`, `NwpInMemory`,
  `_NWP_ON_DISK_DTYPE`, `_NWP_ON_DISK_MAX_INT_VALUE`, `NwpOnDisk`, `NwpScalingParams`,
  `_SCALING_PARAMS_FOR_ECMWF_ENS_0_25_DEGREE_CSV_PATH`. Keep `NwpMetaData` and everything else.
- `packages/dynamical_data/scripts/compute_scaling_params.py` — delete (computed the CSV this
  migration removes).
- `metadata/scaling_params_for_ecmwf_ens_0_25_degree.csv` — delete.
- `metadata/README.md` — remove the `scaling_params_for_ecmwf_ens_0_25_degree.csv` bullet.

**C6. Test updates:**

- `packages/contracts/tests/test_weather_schemas.py`: delete `test_nwp_scaling_roundtrip` and
  `test_nwp_on_disk_var_names_match_nwp_in_memory` (both test the deleted
  quantisation/on-disk-vs-in-memory concept). Rename `NwpInMemory` → `Nwp` in
  `test_weather_feature_literal_matches_model_fields` and
  `test_continuous_and_categorical_partition_all_weather_vars`. Remove the now-unused
  `NwpOnDisk`, `NwpScalingParams` imports.
- `packages/contracts/tests/test_weather_schemas_validation.py`: rename `NwpInMemory` → `Nwp`
  throughout (import + all three `.validate()` call sites).
- `packages/ml_core/tests/test_feature_engineer.py`, `test_features.py`,
  `test_cross_mode_equivalence.py`: rename `NwpInMemory` → `Nwp` (imports + `.set_model(...)`
  calls + type hints). No logic changes — these tests build synthetic `Nwp`-shaped frames
  directly, they don't touch the write path.
- `tests/test_trained_cv_model.py`: rewrite `_write_nwp()` — it currently builds a synthetic
  `NwpOnDisk`-shaped frame (`Int16`, values cast from a flat constant `2000`). The new fixture
  must write physically valid `Float32` values directly (no on-disk-schema concept exists to
  target any more). Use per-column safe constants since `2000` violates several fields' bounds
  (e.g. `precipitation_surface`'s `le=0.01`):

  ```python
  _NWP_CONTINUOUS_COL_VALUES = {
      "temperature_2m": 15.0,
      "dew_point_temperature_2m": 10.0,
      "wind_speed_10m": 5.0,
      "wind_direction_10m": 180.0,
      "wind_speed_100m": 8.0,
      "wind_direction_100m": 180.0,
      "pressure_surface": 101_000.0,
      "pressure_reduced_to_mean_sea_level": 101_500.0,
      "geopotential_height_500hpa": 5_500.0,
      "downward_long_wave_radiation_flux_surface": 300.0,
      "downward_short_wave_radiation_flux_surface": 200.0,
      "precipitation_surface": 0.001,
  }
  ```

  Replace `record.update({col: 2000 for col in _NWP_CONTINUOUS_COLS})` with
  `record.update(_NWP_CONTINUOUS_COL_VALUES)`; in the `.cast({...})` mapping keep the key/index
  entries as they are (`init_time`/`valid_time` → `pl.Datetime("us", "UTC")`, `ensemble_member`
  → `pl.UInt8`, `h3_index` → `pl.UInt64`, `categorical_precipitation_type_surface` → `pl.UInt8`)
  and change only the continuous-column entry: `**{col: pl.Int16 for col in
  _NWP_CONTINUOUS_COLS}` → `**{col: pl.Float32 for col in _NWP_CONTINUOUS_COLS}`. Update the
  docstring ("Write a minimal `NwpOnDisk`-shaped Delta..." → "Write a minimal `Nwp`-shaped
  Delta...").
- `tests/test_cv_power_forecasts.py`: **don't miss this one — it never names the old classes,
  so grepping for `NwpOnDisk`/`NwpInMemory` won't find it.** Its `_write_nwp()` fixture is a
  near-copy of `test_trained_cv_model.py`'s (same flat `2000` constant, same `pl.Int16` cast) and
  needs the identical treatment: the `_NWP_CONTINUOUS_COL_VALUES` dict, the `pl.Float32` cast,
  and the docstring update. Left as-is it would "pass" misleadingly — `Nwp.scan_delta`'s cast
  would coerce the stored Int16 codes to `2000.0` *physical* units (e.g. 2000 °C), out of every
  field's bounds, silently skipping the rescale the fixture was designed around.
- `packages/notebooks/plot_nwp_map.py`: `from contracts.weather_schemas import NwpOnDisk` →
  `Nwp`; `NwpOnDisk.scan_delta()` → `Nwp.scan_delta()`.

**C7. `packages/delta_store/src/delta_store/power_forecasts.py`** docstring cross-reference
(~line 29-30): "Mirrors the 12-bit quantisation already applied to NWP data
(`NwpScalingParams`)." is now false — reword to something like "Mirrors the significand-rounding
scheme `delta_store.nwp` applies to NWP data, but with different writer properties — see that
module's docstring for why the choice doesn't transfer between tables."

**C8. `packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py`** — the comment inside
`train()` (~lines 116-119):

```python
# Stream the collect. An NWP parquet row group holds every ensemble member and H3 cell, so
# the in-memory engine decodes the whole (init_time-pruned) scan before the row-level
# member/cell filters apply — 100+ GB for a multi-month window. The streaming engine applies
# the predicates per morsel, holding peak to a few GB. See docs/architecture/overview.md.
```

is now only half true — after the member-early sort, a single-member predicate mostly skips row
groups via stats rather than decode-then-filter (measured ~5×). Reword to note both: the
`init_time` partition prune, the (now much cheaper) row-group skip for `ensemble_member`, and
that `h3_index` still isn't a sort-early column so cell pruning is still decode-then-filter —
streaming is still the right default. Don't remove the `engine="streaming"` call — it's still
correct and still the safety net for whatever isn't pruned by row-group stats.

### Phase D — docs + verification

**D1. `docs/architecture/overview.md`:**

- Storage bullet (~line 16): replace "NWP data is quantised to 12-bit precision and stored as
  `Int16`... with `zstd` level-14 compression — compressing the full ECMWF ENS dataset for Great
  Britain to approximately 40 GB per year" with a description of the new format: `Float32`
  rounded to 13 significand bits (`delta_store.nwp.round_to_significand_bits`-based), plain
  ZSTD-3 (no `BYTE_STREAM_SPLIT` — measured worse here, unlike `power_forecasts`), sorted
  member-early for row-group pruning, ~40-41 GB/yr measured. Link to this PR for the full
  before/after numbers (mirroring how the `power_forecasts` bullet links to PR #268).
- Lazy-pipeline diagram (~line 28-34): replace

  ```text
  NwpOnDisk.scan_delta(path)               # lazy scan
    → NwpOnDisk.to_nwp_in_memory()        # lazy expression; no collect
  ```

  with

  ```text
  Nwp.scan_delta(path)                     # lazy scan, already Float32 physical units
  ```

- "Bounding feature-engineering memory" section (~lines 46-73): remove the
  "before `to_nwp_in_memory` rescales Int16→Float32" framing at line 50 and the "Applied *after*
  the rescale (the old bug)" aside in the predicate table at line 57 — no rescale step exists any
  more. Rewrite the row-group-pruning paragraph at line 62 (currently: "the data is sorted `init
  → valid → member → h3`, so each row group spans all 1671 cells and 51 members... can't skip
  groups for our 1 member / 9 cells") to describe the **new** behaviour: with `init → member →
  valid → h3`, a control-member predicate now *does* skip most row groups via min/max stats
  (measured ~5× faster / ~5× less peak memory on a real 29-day/9-cell read — cite this PR).
  `h3_index` still isn't sort-early, so cell pruning is still decode-then-filter within whatever
  row groups survive the member skip — the streaming-engine numbers in this section should be
  re-measured post-migration if you have time, but re-measuring isn't required to ship this PR
  (the row-group-skip win is independent of and additive with the existing partition-prune +
  streaming-collect design already documented here).

**D2. `docs/roadmap/xgboost-improvements.md`** — remove the `#198` bullet from "Explicitly
deferred (not quick, or not skill)" (~line 304-306): it's no longer deferred, it's done.

**D3. Full verification:**

```bash
uv run ruff check . --fix
uv run ruff format .
uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md metadata/README.md packages/*/README.md
```

All must pass. Additionally, manually exercise the read path once against the real (now
migrated) local table to catch anything the unit tests can't (they use synthetic tiny frames):

```bash
uv run python -c "
from contracts.weather_schemas import Nwp
from contracts.settings import Settings
import polars as pl
lf = Nwp.scan_delta(Settings().nwp_data_path).filter(pl.col('ensemble_member') == 0).head(5)
print(lf.collect())
print(lf.explain())
"
```

Confirm this returns real rows with plausible physical values (e.g. `temperature_2m` in a
sensible Celsius range, not raw Int16 codes).

## 4. PR conventions (per `CLAUDE.md`)

- **Labels**: `enhancement` (matches #198's label).
- **Assignee**: `JackKelly`.
- **Body**: self-contained — this migration isn't purely "link to a `docs/roadmap/` section"
  (no roadmap page exists for it), so write a real summary: what changed, the measured
  before/after numbers (storage + read-path table from §1 above), and `Closes #198`.
- `gh pr create` can't set labels/assignee — follow with `gh pr edit --add-label enhancement
  --add-assignee JackKelly`.

## 5. Ship-time triage (do this once the PR is merged, before deleting this file)

1. Confirm `docs/architecture/overview.md` and `docs/roadmap/xgboost-improvements.md` already
   reflect the shipped state (done in Phase D above — just double check post-merge, in case of
   review-driven changes).
2. Close [#198](https://github.com/openclimatefix/nged-substation-forecast/issues/198) (link the
   merged PR).
3. Delete this file (`plans/nwp-int16-to-float32-migration.md`).
4. Confirm `plans/nwp-significand-rounding-experiment.md` no longer exists (it was deleted when
   this file was created — its results are folded into §1 above). If it somehow still exists,
   delete it now.
