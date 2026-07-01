# NWP quantisation: count and surface clipped values

## Finding

`NwpOnDisk.from_nwp_in_memory` (`packages/contracts/src/contracts/weather_schemas.py:390`)
clips each variable to its buffered range before Int16 encoding:

```python
clipped_col = pl.col(col_name).clip(lower_bound=buffered_min, upper_bound=buffered_max)
```

The buffered ranges carry only a 5% margin over the min/max observed when
`compute_scaling_params` ran. If ECMWF ENS later produces values outside that range (extreme
storm winds, record pressure lows), they are silently flattened to the boundary — corrupting
exactly the extreme-weather inputs that matter most for peak forecasting, with no signal that
it happened.

## Implementation

### 1. Pure counting helper in contracts

New function in `weather_schemas.py` (near `from_nwp_in_memory`):

```python
def count_out_of_range(
    nwp_in_memory: pt.DataFrame[NwpInMemory],
    scaling_params: pt.DataFrame[NwpScalingParams],
) -> dict[str, int]:
    """Per-variable count of values outside the buffered scaling range (i.e. values that
    from_nwp_in_memory would clip)."""
```

One `select` building `((col < buffered_min) | (col > buffered_max)).sum()` per scaling-params
row (mirror the loop structure at `weather_schemas.py:380-398`), returning only non-zero
entries. Keep `from_nwp_in_memory` itself pure/unchanged — counting is the caller's concern,
and the asset wants the numbers for metadata anyway.

### 2. Call it in the `ecmwf_ens` asset

In `src/nged_substation_forecast/defs/assets.py`, where the day's `NwpInMemory` frame is
converted via `from_nwp_in_memory`:

- Call `count_out_of_range` on the same frame.
- If any count > 0: `context.log.warning(...)` naming the variables and counts.
- Always attach the dict to the asset materialisation metadata (e.g.
  `clipped_values_total` plus per-variable entries), so drift is visible in the Dagster UI
  timeline even at zero.

Cost note: one extra aggregation pass over the day's frame (~7M rows) — negligible next to the
download.

### 3. Docs

One paragraph in the NWP section of `docs/roadmap/data-sources.md` (or wherever the
quantisation scheme is described): clipping is monitored, and persistent non-zero counts mean
`compute_scaling_params` should be re-run with a wider range (which requires re-encoding the
Delta table — a deliberate, documented operation, not something to automate).

## Verification

1. Unit test in `packages/contracts/tests/`: a small `NwpInMemory` fixture with values pushed
   beyond `buffered_max` for one variable → `count_out_of_range` reports exactly those counts,
   and zero for in-range variables.
2. Round-trip test: values at exactly `buffered_min`/`buffered_max` count as in-range (clip is
   inclusive).
3. Materialise one `ecmwf_ens` partition locally and confirm the metadata entry appears (expect
   zeros for a normal day).
