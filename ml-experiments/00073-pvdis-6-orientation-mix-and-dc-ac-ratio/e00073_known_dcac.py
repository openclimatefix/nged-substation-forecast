"""Known-answer test for the DC:AC ratio, and the knee gate.

Experiment 00029 fitted the DC:AC ratio of each of the six metered solar farms. This script
searches the ratio grid (the shape `min(r * g, 1)`, south-facing at 25° tilt) on clear days for

- each farm's output alone, against the plane-of-array fraction at the farm's own location; and
- each farm's output, scaled to 10 MW, added to a Primary's real power, against the fraction at the
  Primary (the nuisance and calendar terms are fitted as in the orientation arms).

It also counts the clear-day half-hours where the shape at ratio 1.15 reaches its knee (the gate).

    POLARS_MAX_THREADS=6 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --with holidays python \
        ml-experiments/00073-pvdis-6-orientation-mix-and-dc-ac-ratio/e00073_known_dcac.py
"""

from multiprocessing import Pool

import e00073_common as common
import e00073_fit as fit
import e00073_inputs as inputs
import numpy as np
import polars as pl


def farm_alone(*, farms: dict[str, pl.DataFrame], shapes: pl.DataFrame) -> list[dict[str, object]]:
    """Return the searched ratio of each farm and season-year alone, and its knee count."""
    out: list[dict[str, object]] = []
    for farm, table in farms.items():
        rows = (
            table.filter(
                pl.col("month").is_in(common.SUMMER_MONTHS)
                & (pl.col("day_cloud") < common.CLEAR_CLOUD_MAX)
            )
            .join(shapes.filter(pl.col("site") == farm).select("time", "gS25"), on="time")
            .drop_nulls()
        )
        for year in sorted(rows["season_year"].unique().to_list()):
            block = rows.filter(pl.col("season_year") == year)
            if block["date"].n_unique() < common.MIN_CLEAR_DAYS:
                continue
            g = block["gS25"].to_numpy()
            profile = fit.ratio_profile(
                design=None, y=block["farm_norm"].to_numpy(), g=g, grid=common.RATIO_GRID
            )
            out.append(
                {
                    "case": "alone", "farm": farm, "site": farm, "year": year,
                    "truth": common.FARM_DC_AC[farm],
                    "fitted": fit.best_ratio(profile=profile, grid=common.RATIO_GRID),
                    "knee_rows": int((common.FLEET_RATIO * g >= 1.0).sum()),
                    "clear_rows": block.height,
                }
            )  # fmt: skip
    return out


def injected_primary(
    *, prepared: pl.DataFrame, primary: str, farms: dict[str, pl.DataFrame]
) -> list[dict[str, object]]:
    """Return the searched ratio with a farm injected into a Primary, and without injection."""
    out: list[dict[str, object]] = []
    for w in inputs.year_windows(prepared=prepared, primary=primary):
        if len(np.unique(w.days[w.clear])) < common.MIN_CLEAR_DAYS:
            continue
        g_all = w.g_primary["S25"]
        design_clear = w.design(mask=w.clear)
        base_profile = fit.ratio_profile(
            design=design_clear, y=w.power[w.clear], g=g_all[w.clear], grid=common.RATIO_GRID
        )
        out.append(
            {
                "case": "base", "farm": "-", "site": primary, "year": w.year, "truth": np.nan,
                "fitted": fit.best_ratio(profile=base_profile, grid=common.RATIO_GRID),
                "knee_rows": int((common.FLEET_RATIO * g_all[w.clear] >= 1.0).sum()),
                "clear_rows": int(w.clear.sum()),
            }
        )  # fmt: skip
        for farm, table in farms.items():
            times = table["time"].to_numpy()
            values = table["farm_norm"].to_numpy()
            order = np.argsort(times)
            times, values = times[order], values[order]
            position = np.clip(np.searchsorted(times, w.time), 0, len(times) - 1)
            has = times[position] == w.time
            mask = w.clear & has
            if len(np.unique(w.days[mask])) < common.MIN_CLEAR_DAYS:
                continue
            injection = common.DCAC_INJECTED_MW * values[position][mask]
            design = w.design(mask=mask)
            profile = fit.ratio_profile(
                design=design, y=w.power[mask] - injection, g=g_all[mask], grid=common.RATIO_GRID
            )
            out.append(
                {
                    "case": "injected", "farm": farm, "site": primary, "year": w.year,
                    "truth": common.FARM_DC_AC[farm],
                    "fitted": fit.best_ratio(profile=profile, grid=common.RATIO_GRID),
                    "knee_rows": int((common.FLEET_RATIO * g_all[mask] >= 1.0).sum()),
                    "clear_rows": int(mask.sum()),
                }
            )  # fmt: skip
    return out


def _job(args: tuple[pl.DataFrame, str, dict[str, pl.DataFrame]]) -> list[dict[str, object]]:
    prepared, primary, farms = args
    return injected_primary(prepared=prepared, primary=primary, farms=farms)


def main() -> None:
    """Write the known-answer table for the DC:AC ratio and print the summary."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    prepared = inputs.load_prepared()
    farms = inputs.farm_tables(prepared=prepared)
    shapes = pl.read_parquet(common.SHAPES_PATH)
    rows = farm_alone(farms=farms, shapes=shapes)
    jobs = [(prepared.filter(pl.col("site") == p), p, farms) for p in common.DIRECTIONAL_PRIMARIES]
    with Pool(6) as pool:
        rows += [row for part in pool.map(_job, jobs) for row in part]
    table = pl.DataFrame(rows).with_columns(error=(pl.col("fitted") - pl.col("truth")).abs())
    table.write_parquet(common.KNOWN_DCAC_PATH)
    with pl.Config(tbl_rows=40):
        print(
            table.filter(pl.col("case") != "base")
            .group_by("case", "farm")
            .agg(
                pl.col("truth").first(),
                pl.col("fitted").median().alias("median_fitted"),
                pl.col("error").median().alias("median_error"),
                pl.col("knee_rows").median().alias("median_knee_rows"),
            )
            .sort("case", "farm")
        )
        print(table.filter(pl.col("case") == "base")["fitted"].describe())


if __name__ == "__main__":
    main()
