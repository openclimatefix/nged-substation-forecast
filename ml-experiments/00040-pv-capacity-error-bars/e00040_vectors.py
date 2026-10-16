"""Fit the arms to every unit's injected PV, every confound shape, and the real series.

For each unit (Primary, farm, season-year) the script writes the arm vector of the normalised farm
output alone (kind `pv`, a true capacity of 1 MW) and of each confound shape alone, fitted on the
unit's shared rows. For each real directional Primary-year it writes the arm vector of the real
series (kind `real`) and of each confound shape on the Primary-year's own rows. The 00039 timestamp
scan is checked first.

    POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        uv run --with holidays python ml-experiments/00040-pv-capacity-error-bars/e00040_vectors.py
"""

import e00040_arms as arms
import e00040_common as common
import e00040_inputs as inputs
import e00040_model as model
import e00040_selftest as selftest
import numpy as np
import polars as pl


def check_timestamp_scan() -> None:
    """Stop unless 00039's farm-point timestamp scan peaks at a shift of zero.

    The frame is the same frame, so the scan of experiment 00039 applies unchanged.
    """
    scan = pl.read_parquet(common.TIMESTAMP_SCAN_PATH)
    by_shift = scan.group_by("shift").agg(pl.col("R").median()).sort("shift")
    peak = by_shift.sort("R", descending=True)["shift"][0]
    assert peak == 0, f"timestamp scan peaks at shift {peak}, not 0: stop and report"
    print("timestamp scan peaks at shift 0:", by_shift.to_dicts())


def record(
    *, site: str, farm: str | None, year: int, kind: str, vector: np.ndarray, rows: int
) -> dict[str, object]:
    """Return one table row of an arm vector."""
    return {
        "site": site,
        "farm": farm,
        "year": year,
        "kind": kind,
        "rows": rows,
        **dict(zip(common.ALL_FITS, vector.tolist(), strict=True)),
    }


def shape_vectors(*, window: model.Window) -> dict[str, np.ndarray]:
    """Return the arm vector of each confound shape alone on a window."""
    return {
        name: arms.arm_capacities(y=shape, window=window)
        for name, shape in arms.confound_shapes(window=window).items()
    }


def main() -> None:
    """Write the unit and real arm vectors."""
    check_timestamp_scan()
    prepared = inputs.load_prepared()
    selftest.run(prepared=prepared)
    primaries = inputs.primary_rows(prepared=prepared)
    farms = inputs.farm_tables(prepared=prepared)

    unit_rows: list[dict[str, object]] = []
    for primary, farm, year, rows in inputs.units(primaries=primaries, farms=farms):
        window = model.window_from(rows=rows)
        truth = rows["farm_norm"].to_numpy().astype(float)
        unit_rows.append(
            record(
                site=primary,
                farm=farm,
                year=year,
                kind="pv",
                vector=arms.arm_capacities(y=-truth, window=window),
                rows=rows.height,
            )
        )
        unit_rows.extend(
            record(site=primary, farm=farm, year=year, kind=name, vector=v, rows=rows.height)
            for name, v in shape_vectors(window=window).items()
        )
    units = pl.DataFrame(unit_rows)
    units.write_parquet(common.UNIT_VECTORS_PATH)
    n_units = units.filter(pl.col("kind") == "pv").height
    print(f"{n_units} units")

    real_rows: list[dict[str, object]] = []
    for primary in common.DIRECTIONAL_PRIMARIES:
        for year, rows in inputs.season_year_rows(rows=primaries.filter(pl.col("site") == primary)):
            window = model.window_from(rows=rows)
            real_rows.append(
                record(
                    site=primary,
                    farm=None,
                    year=year,
                    kind="real",
                    vector=arms.arm_capacities(y=window.power, window=window),
                    rows=rows.height,
                )
            )
            real_rows.extend(
                record(site=primary, farm=None, year=year, kind=name, vector=v, rows=rows.height)
                for name, v in shape_vectors(window=window).items()
            )
    real = pl.DataFrame(real_rows)
    real.write_parquet(common.REAL_VECTORS_PATH)
    print(f"{real.filter(pl.col('kind') == 'real').height} real Primary-years")


if __name__ == "__main__":
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    main()
