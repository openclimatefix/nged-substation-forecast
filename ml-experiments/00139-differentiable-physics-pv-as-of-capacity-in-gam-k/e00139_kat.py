"""Known-answer stage of experiment 139: does the as-of capacity plumbing recover a known change?

For each Primary and each injected source (five real farms, an east-west plant, a south-west plant)
the injected output is `c * shape`, with `c` half the as-of capacity held at the cut (points of
effective capacity). The scored year's actual power falls by that output, and no fit sees it: the
fits and the filter use the real power, so the demand terms cannot absorb the injection.

1. With the capacity unchanged the forecast is the as-of forecast, so its error shifts by the
   injection (identity, checked numerically).
2. With the capacity raised by `c` (the path and the held value), the forecast falls by `c` times
   the ensemble-driven unit PV shape. The recovery is the mean fall of the forecast divided by the
   mean injected output over the sunniest third of the scored half-hours.

The band is 0.75 to 1.25, as in experiment 129. A second ratio raises the capacity by `c` times the
Primary's recovery divisor instead, which puts the raise in the model's own units.

    uv run --no-sync python ml-experiments/00139-*/e00139_kat.py [--workers 5]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import e00139_common as common
import e00139_ens as ens
import e00139_fits as fits_module
import e00139_run as run_module
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamFit

BAND: tuple[float, float] = (0.75, 1.25)
"""Recovery counts as a pass inside this band, as in experiment 129."""
INJECTED_SHARE: float = 0.5
"""The injection is this share of the as-of capacity held at the cut."""
SOURCES: tuple[str, ...] = (*common.FARMS, "east_west", "south_west")


def shapes_of(*, cache: FeatureCache, site: str) -> dict[str, np.ndarray]:
    """Return each source's output per unit of capacity on the grid."""
    shapes = {farm: common.farm_norm(cache=cache, farm=farm) for farm in common.FARMS}
    shapes["east_west"] = common.synthetic_plant_shape(
        cache=cache, site=site, plants=common.EAST_WEST_PLANTS
    )
    shapes["south_west"] = common.synthetic_plant_shape(
        cache=cache, site=site, plants=common.SOUTH_WEST_PLANTS
    )
    return shapes


def raised(*, fits: list[GamFit], share: float) -> tuple[list[GamFit], list[float]]:
    """Return fits whose capacity is raised by `share` of the held value, and the raises."""
    out, raises = [], []
    for fit in fits:
        pinned = fit.pinned
        assert pinned is not None
        raise_by = share * pinned.held_points
        out.append(
            replace(
                fit,
                pinned=replace(
                    pinned,
                    points=pinned.points + raise_by,
                    held_points=pinned.held_points + raise_by,
                ),
            )
        )
        raises.append(raise_by)
    return out, raises


def kat_site(site: str) -> pl.DataFrame:
    """Return one row per source with the recovery ratios at one Primary."""
    started = time.monotonic()
    torch.set_num_threads(1)
    context = run_module.load_site(site=site)
    cache = context.cache
    fits = fits_module.load_fits(arm="asof", site=site)
    runs = context.setup.indices(periods={"scored"})
    base = ens.forecast_frame(
        cache=cache,
        setup=context.setup,
        fits=fits,
        spread=ens.ZERO_SPREAD,
        trace=None,
        run_indices=runs,
        arm="asof",
    )
    raised_fits, _ = raised(fits=fits, share=INJECTED_SHARE)
    up = ens.forecast_frame(
        cache=cache,
        setup=context.setup,
        fits=raised_fits,
        spread=ens.ZERO_SPREAD,
        trace=None,
        run_indices=runs,
        arm="asof_raised",
    )
    joined = base.join(
        up.select("run", "valid_time", mean_up="mean"), on=["run", "valid_time"], validate="1:1"
    )
    valid64 = joined["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    rows = np.searchsorted(cache.end_times, valid64)
    fall = (joined["mean"] - joined["mean_up"]).to_numpy().astype(np.float64)
    unit = np.nan_to_num(cache.sites[site].pv_unit_shape[rows], nan=0.0)
    daylight = unit > common.SUN_UP_UNIT_SHAPE
    sunny = daylight & (unit >= np.quantile(unit[daylight], 2 / 3))
    # The raise is a share of each cut's held value, so the injected capacity varies by cut.
    cuts = np.array([np.datetime64(f.cut.replace(tzinfo=None), "us") for f in fits])
    held_by_cut = np.array([f.pinned.held_points for f in fits])  # ty: ignore[unresolved-attribute]
    origin64 = (
        joined["power_fcst_init_time"]
        .dt.replace_time_zone(None)
        .to_numpy()
        .astype("datetime64[us]")
    )
    which = np.clip(np.searchsorted(cuts, origin64, side="right") - 1, 0, len(fits) - 1)
    injected_capacity = INJECTED_SHARE * held_by_cut[which]
    shapes = shapes_of(cache=cache, site=site)
    table = pl.read_parquet(common.WORK_DIR / "asof" / f"asof_{site}_20250701.parquet")
    divisor = float(table.filter(pl.col("season_year") == 2024)["divisor"][0])
    out = []
    for source in SOURCES:
        output = injected_capacity * shapes[source][rows]
        usable = sunny & np.isfinite(output)
        ratio = float(fall[usable].sum() / output[usable].sum())
        out.append(
            {
                "site": site,
                "source": source,
                "plant": "matched" if source in common.FARMS else source,
                "recovery": ratio,
                "recovery_in_model_units": ratio / divisor,
                "divisor_2024": divisor,
                "sunny_half_hours": int(usable.sum()),
                "mean_injected_points": float(output[usable].mean()),
                "mean_fall_points": float(fall[usable].mean()),
            }
        )
    print(f"{site}: {time.monotonic() - started:.0f} s", flush=True)
    return pl.DataFrame(out)


def main() -> None:
    """Run the stage at the five Primaries and write the recovery table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    arguments = parser.parse_args()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        frames = list(pool.map(kat_site, common.DECISION_SITES))
    table = pl.concat(frames)
    common.write_parquet_atomic(frame=table, path=common.WORK_DIR / "kat.parquet")
    summary = (
        table.group_by("site", "plant")
        .agg(
            pl.col("recovery").median().alias("recovery"),
            pl.col("recovery_in_model_units").median().alias("recovery_model_units"),
        )
        .with_columns(
            passes=pl.col("recovery").is_between(*BAND),
            passes_model_units=pl.col("recovery_model_units").is_between(*BAND),
        )
        .sort("plant", "site")
    )
    with pl.Config(tbl_rows=30, float_precision=3):
        print(summary)
        print(
            summary.group_by("plant").agg(
                pl.col("passes").sum().alias("primaries_passing"),
                pl.col("passes_model_units").sum().alias("primaries_passing_model_units"),
            )
        )


if __name__ == "__main__":
    main()
