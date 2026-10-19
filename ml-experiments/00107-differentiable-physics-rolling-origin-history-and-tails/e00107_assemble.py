"""Assemble experiment 107's scored predictions: apply each spread to each fit's rolling pool.

Reads the per-job means of `e00107_gam.py` and the quantiles of `e00107_xgb.py`, and writes one
parquet file per (arm, spread) to `work/scored/` (or `work/scored_d10_masked/`): every scored
half-hour of the 12 scored origins with the 11 quantiles, the mean, and the actual power.

Convex arms get spread S0 (all arms) and S1 and S2 (`g1` and `g_all_d1`). XGBoost arms keep their
own quantiles (spread `own`). `--variant d10_masked` replaces D10's fits by the refits that leave
its two unlabelled-jump windows out of training, where such a refit exists, and drops those rows
from pools and from scoring.

It also writes `work/s1_groups.parquet` (each S1 group's size, q99 exceedances, and fallback) and
`work/capacity_sources.parquet` (the pinned capacity's source per cut and site).

    uv run python ml-experiments/00107-*/e00107_assemble.py [--variant base|d10_masked]
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Final

import e00107_common as common
import numpy as np
import polars as pl
from e00107_spread import ConformalSpread, LevelTercileSpread
from studies.dp_features import FeatureCache
from studies.dp_gam import N_BANDS, QUANTILE_COLUMNS, ResidualSpread

SPREAD_ARMS: Final[tuple[str, ...]] = ("g1", "g_all_d1")
"""The convex arms that also get S1 and S2."""
KEY_COLUMNS: Final[tuple[str, ...]] = (
    "site",
    "time_series_id",
    "origin",
    "valid_time",
    "power_fcst_init_time",
    "lead_hours",
    "actual",
    "switching_mask",
    "jump_window",
)


def read_folder(*, folder: Path) -> pl.DataFrame:
    """Concatenate every job file of a folder (not its `records` subfolder)."""
    files = sorted(folder.glob("*.parquet"))
    if not files:
        msg = f"no job files in {folder}"
        raise FileNotFoundError(msg)
    return pl.concat([pl.read_parquet(file) for file in files])


def read_fits(*, kind: str, variant: str) -> pl.DataFrame:
    """Return every fit's rows for `kind` (`gam` or `xgb`), with D10's masked refits swapped in."""
    base = read_folder(folder=common.WORK_DIR / kind / "base")
    if variant == "base":
        return base
    masked = read_folder(folder=common.WORK_DIR / kind / "d10_masked")
    key = masked.select("site", "origin", "arm").unique()
    kept = base.join(key, on=["site", "origin", "arm"], how="anti")
    return pl.concat([kept, masked.select(base.columns)])


def grid_positions(*, cache: FeatureCache, frame: pl.DataFrame) -> np.ndarray:
    """Return each row's position on the cache's grid."""
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    position = np.searchsorted(cache.end_times, valid)
    if not np.array_equal(cache.end_times[position], valid):
        msg = "a valid time is not on the grid"
        raise ValueError(msg)
    return position


def previous_cuts(*, cut: datetime) -> list[datetime]:
    """Return the four cuts whose scored windows make the pool for origin `cut`, oldest first."""
    return [
        common.add_months(moment=cut, months=-common.SCORED_MONTHS * step)
        for step in range(common.POOL_MONTHS // common.SCORED_MONTHS, 0, -1)
    ]


def pool_rows(*, arm_rows: pl.DataFrame, cut: datetime, drop_jump_window: bool) -> pl.DataFrame:
    """Return the pool of one (arm, site) for origin `cut`: rows of the four previous windows.

    Rows inside the hand-filtered switching masks are out (and the jump windows, if asked).

    Args:
        arm_rows: One arm's rows at one site, every cut.
        cut: The origin the pool serves.
        drop_jump_window: True to drop D10's jump-window rows too.

    Returns:
        The pool's rows, with their `origin` label.

    Raises:
        ValueError: If a pool row's valid time is after the origin.
    """
    labels = [common.cut_label(cut=previous) for previous in previous_cuts(cut=cut)]
    pool = arm_rows.filter(pl.col("origin").is_in(labels), ~pl.col("switching_mask"))
    if drop_jump_window:
        pool = pool.filter(~pl.col("jump_window"))
    if pool.height and pool.select(pl.col("valid_time").max()).item() > cut:
        msg = f"a pool row for origin {cut} has a valid time after the origin"
        raise ValueError(msg)
    return pool


def apply_spreads(
    *,
    pool_residual: np.ndarray,
    pool_mean: np.ndarray,
    pool_band: np.ndarray,
    pool_quarter: np.ndarray,
    pool_newest: np.ndarray,
    mean: np.ndarray,
    band: np.ndarray,
    quarter: np.ndarray,
    with_alternatives: bool,
) -> tuple[dict[str, np.ndarray], LevelTercileSpread | None]:
    """Return the quantiles of the scored rows under each spread, from pool arrays only.

    Args:
        pool_residual: Pool residuals in points.
        pool_mean: The pool's predicted means.
        pool_band: The pool's hour bands.
        pool_quarter: The pool's calendar quarters.
        pool_newest: True for rows of the pool's newest window.
        mean: The scored rows' predicted means.
        band: The scored rows' hour bands.
        quarter: The scored rows' calendar quarters.
        with_alternatives: True to also fit S1 and S2.

    Returns:
        A map from spread name (`s0`, and `s1` and `s2` if asked) to quantiles of shape
        (len(mean), 11), and the fitted S1 (or `None`).
    """
    s0 = ResidualSpread.fit(
        residuals=pool_residual, band=pool_band, quarter=pool_quarter, centre=True
    )
    out = {"s0": s0.quantiles(mean=mean, band=band, quarter=quarter)}
    s1 = None
    if with_alternatives:
        s1 = LevelTercileSpread.fit(
            residuals=pool_residual, predicted=pool_mean, band=pool_band, quarter=pool_quarter
        )
        out["s1"] = s1.quantiles(mean=mean, band=band, quarter=quarter)
        s2 = ConformalSpread.fit(
            residuals=pool_residual, band=pool_band, quarter=pool_quarter, newest=pool_newest
        )
        out["s2"] = s2.quantiles(mean=mean, band=band, quarter=quarter)
    return out, s1


def convex_predictions(
    *, cache: FeatureCache, gam: pl.DataFrame, variant: str
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Apply the spreads to every convex arm, site, and scored origin.

    Args:
        cache: The feature cache, for each row's hour band and calendar quarter.
        gam: Every convex fit's rows.
        variant: `base` or `d10_masked`.

    Returns:
        A map from `<arm>__<spread>` to the scored rows, and the S1 group table.
    """
    positions = grid_positions(cache=cache, frame=gam)
    gam = gam.with_columns(
        band=pl.Series(cache.calendar.band[positions].astype(np.int64)),
        quarter=pl.Series(cache.calendar.quarter[positions].astype(np.int64)),
        residual=pl.col("actual") - pl.col("mean"),
    )
    frames: dict[str, list[pl.DataFrame]] = {}
    groups: list[dict] = []
    for arm in gam["arm"].unique(maintain_order=True):
        for site in common.ALL_SITES:
            arm_rows = gam.filter(pl.col("arm") == arm, pl.col("site") == site)
            drop_jump = variant == "d10_masked" and site == "D10"
            for cut in common.SCORED_ORIGINS:
                if site == "D9" and cut < common.add_months(moment=common.D9_FIRST_CUT, months=12):
                    continue
                scored = arm_rows.filter(pl.col("origin") == common.cut_label(cut=cut))
                if drop_jump:
                    scored = scored.filter(~pl.col("jump_window"))
                if scored.is_empty():
                    continue
                pool = pool_rows(arm_rows=arm_rows, cut=cut, drop_jump_window=drop_jump)
                newest_label = common.cut_label(cut=previous_cuts(cut=cut)[-1])
                quantiles, s1 = apply_spreads(
                    pool_residual=pool["residual"].to_numpy(),
                    pool_mean=pool["mean"].to_numpy(),
                    pool_band=pool["band"].to_numpy(),
                    pool_quarter=pool["quarter"].to_numpy(),
                    pool_newest=(pool["origin"] == newest_label).to_numpy(),
                    mean=scored["mean"].to_numpy(),
                    band=scored["band"].to_numpy(),
                    quarter=scored["quarter"].to_numpy(),
                    with_alternatives=arm in SPREAD_ARMS,
                )
                if s1 is not None:
                    groups += s1_group_rows(s1=s1, arm=arm, site=site, cut=cut)
                for spread, values in quantiles.items():
                    ordered = np.sort(values, axis=1)
                    frame = scored.select(*KEY_COLUMNS, "mean").with_columns(
                        arm=pl.lit(arm),
                        spread=pl.lit(spread),
                        **{
                            column: pl.Series(ordered[:, position])
                            for position, column in enumerate(QUANTILE_COLUMNS)
                        },
                    )
                    frames.setdefault(f"{arm}__{spread}", []).append(frame)
    return {name: pl.concat(parts) for name, parts in frames.items()}, pl.DataFrame(groups)


def s1_group_rows(*, s1: LevelTercileSpread, arm: str, site: str, cut: datetime) -> list[dict]:
    """Return one record per S1 group: its size, q99 exceedances, and whether it fell back."""
    return [
        {
            "arm": arm,
            "site": site,
            "origin": common.cut_label(cut=cut),
            "band": band,
            "quarter": quarter,
            "tercile": tercile,
            "rows": int(s1.group_rows[band, quarter, tercile]),
            "q99_exceedances": int(s1.group_exceedances_q99[band, quarter, tercile]),
            "fallback": bool(s1.fallback[band, quarter, tercile]),
        }
        for band in range(N_BANDS)
        for quarter in range(4)
        for tercile in range(3)
    ]


def xgb_predictions(*, xgb: pl.DataFrame, variant: str) -> dict[str, pl.DataFrame]:
    """Return the XGBoost arms' scored rows (scored origins only) under spread `own`."""
    scored_labels = [common.cut_label(cut=cut) for cut in common.SCORED_ORIGINS]
    frames: dict[str, pl.DataFrame] = {}
    for arm in xgb["arm"].unique(maintain_order=True):
        rows = xgb.filter(pl.col("arm") == arm, pl.col("origin").is_in(scored_labels))
        if rows.is_empty():
            continue
        if variant == "d10_masked":
            rows = rows.filter(~pl.col("jump_window"))
        frames[f"{arm}__own"] = rows.select(*KEY_COLUMNS, "mean", *QUANTILE_COLUMNS).with_columns(
            arm=pl.lit(arm), spread=pl.lit("own")
        )
    return frames


def capacity_sources() -> pl.DataFrame:
    """Return the pinned capacity's source per site and cut.

    The table also holds the vintage origin and held level, beside the fitted capacity of the
    matching `-fc` arm.
    """
    records = pl.concat(
        [
            pl.read_parquet(file)
            for file in (common.WORK_DIR / "gam" / "base" / "records").glob("*.parquet")
        ]
    )
    pinned = records.filter(pl.col("arm") == "g_all_d1").select(
        "site",
        "origin",
        "pinned_source",
        "pinned_vintage_origin",
        "pinned_held_points",
    )
    fitted = records.filter(pl.col("arm") == "g_all_d1_fc").select(
        "site", "origin", "fitted_last_capacity_points"
    )
    return pinned.join(fitted, on=["site", "origin"]).sort("site", "origin")


def main() -> None:
    """Write every (arm, spread) file of a variant."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("base", "d10_masked"), default="base")
    arguments = parser.parse_args()
    cache = common.load_cache()
    folder = common.WORK_DIR / ("scored" if arguments.variant == "base" else "scored_d10_masked")
    convex, groups = convex_predictions(
        cache=cache, gam=read_fits(kind="gam", variant=arguments.variant), variant=arguments.variant
    )
    xgb = xgb_predictions(
        xgb=read_fits(kind="xgb", variant=arguments.variant), variant=arguments.variant
    )
    for name, frame in {**convex, **xgb}.items():
        common.write_atomic(frame=frame, path=folder / f"{name}.parquet")
        print(f"{name}: {frame.height} rows", flush=True)
    if arguments.variant == "base":
        common.write_atomic(frame=groups, path=common.WORK_DIR / "s1_groups.parquet")
        common.write_atomic(
            frame=capacity_sources(), path=common.WORK_DIR / "capacity_sources.parquet"
        )


if __name__ == "__main__":
    main()
