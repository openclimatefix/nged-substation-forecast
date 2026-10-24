# ruff: noqa: E501
"""Assemble experiment 321's arms from stored forecasts and the raw head fits.

Writes, per band and history setting and under `assembled/`, one parquet per arm and farm:
`recal`, `recal_shrunk`, `recal_stale7`, `recal_swapped` (tables on research default 1's blend);
`zi_alone` (the zero-inflated head, seed-averaged) and `zi_blend` (the same blend as research
default 1 with that head in place of the mixture head); `rd1_5seed` (the blend with experiment
318's `default` head, five seeds, the paired comparator of `zi_blend`); `tail` and `tail_shuffled`
(the blend's 0.01, 0.05, 0.95, and 0.99 quantiles replaced by the tail head's); `pure_lead_pooled`
(arm 4); and `combo` (the table applied to `tail`) only with the argument `combo`.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00321-... \
        uv run --no-sync python ml-experiments/00321-*/e00321_assemble.py [combo]
"""

import multiprocessing
import sys
from pathlib import Path
from typing import Final

import e00310_blend as b310  # ty: ignore[unresolved-import]
import e00310_common as c310  # ty: ignore[unresolved-import]
import e00321_common as common
import e00321_recal as recal
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

QCOLS: Final[list[str]] = recal.QCOLS
KEYS: Final[list[str]] = recal.KEYS
TAIL_COLS: Final[list[str]] = [level_column(level=lv) for lv in common.TAIL_STORED_LEVELS]
ASSEMBLED: Final[Path] = common.DATA_DIR / "assembled"


def read_arm_average(*, files: list[Path], columns: list[str]) -> pl.DataFrame:
    """Return the seed-averaged `columns` per (farm, time, run) over the files.

    Args:
        files: The per-seed, per-farm, per-refit parquet files of one arm.
        columns: The quantile columns to average.

    Returns:
        One row per (farm, time, run) with the first lead and actual and the mean of `columns`.

    Raises:
        ValueError: If the (farm, refit) groups do not all hold the same number of seed files.
    """
    groups: dict[str, int] = {}
    for f in files:
        key = f.name.split("_s", 1)[1][2:]  # drop the seed: "<farm>_<refit>.parquet"
        groups[key] = groups.get(key, 0) + 1
    if len(set(groups.values())) != 1:
        msg = f"unequal seed counts across farm and refit files: {groups}"
        raise ValueError(msg)
    frame = pl.concat(
        [pl.read_parquet(f).select(*KEYS, "lead_hours", "actual", *columns) for f in files]
    ).with_columns(pl.col(columns).cast(pl.Float64))
    return frame.group_by(KEYS).agg(
        pl.col("lead_hours").first().cast(pl.Float64),
        pl.col("actual").first().cast(pl.Float64),
        pl.col(columns).mean(),
    )


def raw_files(*, root: Path, band: str, setting: str, arm: str) -> list[Path]:
    """Return a raw arm's per-seed, per-farm, per-refit files."""
    return sorted((root / band / setting).glob(f"{arm}_s[0-9]_?_20??-??.parquet"))


def write_arm(*, frame: pl.DataFrame, band: str, setting: str, arm: str) -> None:
    """Write an assembled arm's rows per farm, atomically."""
    out_dir = ASSEMBLED / band / setting
    out_dir.mkdir(parents=True, exist_ok=True)
    full = frame.select(*KEYS, "lead_hours", "actual", *QCOLS).with_columns(
        arm=pl.lit(arm), seed=pl.lit(0), setting=pl.lit(setting), band=pl.lit(band)
    )
    for farm in c310.FARMS:
        part = full.filter(pl.col("time_series_id") == farm)
        path = out_dir / f"{arm}_s0_{farm}.parquet"
        tmp = path.with_suffix(".tmp")
        part.write_parquet(tmp)
        tmp.replace(path)


def read_assembled(*, band: str, setting: str, arm: str) -> pl.DataFrame:
    """Return an assembled arm."""
    files = sorted((ASSEMBLED / band / setting).glob(f"{arm}_s0_?.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files])


def blend_with(*, band: str, setting: str, head: pl.DataFrame) -> pl.DataFrame:
    """Return research default 1's rolling blend with `head` (rows keyed like the components) as the mixture member."""
    gam = b310.load_component(band=band, setting=setting, arm="gam")
    xgb = b310.load_component(band=band, setting=setting, arm="xgb")
    frames = {"gam": gam, "head": head.select(*KEYS, "lead_hours", "actual", *QCOLS), "xgb": xgb}
    keys = frames["gam"].select(KEYS)
    for f in frames.values():
        keys = keys.join(f.select(KEYS), on=KEYS)
    keys = keys.sort(KEYS)
    base = (
        keys.join(frames["gam"], on=KEYS)
        .sort(KEYS)
        .select(*KEYS, "lead_hours", "actual")
        .with_columns(month=pl.col("init_time").dt.strftime("%Y-%m"))
    )
    stack = np.stack(
        [
            keys.join(frames[a], on=KEYS).sort(KEYS).select(QCOLS).to_numpy()
            for a in ("gam", "head", "xgb")
        ]
    )
    blended, _ = b310.rolling(base=base, stack=stack, components=("gam", "head", "xgb"))
    return base.with_columns(*[pl.Series(c, blended[:, k]) for k, c in enumerate(QCOLS)])


def with_tail(*, band: str, setting: str, arm: str, seed_dir_arm: str) -> pl.DataFrame | None:
    """Return research default 1 with its tail quantiles replaced by the tail head's, or None if absent."""
    files = raw_files(root=common.PRED_DIR, band=band, setting=setting, arm=seed_dir_arm)
    if not files:
        return None
    tail = read_arm_average(files=files, columns=TAIL_COLS).select(*KEYS, *TAIL_COLS)
    rd1 = recal.load_stored(band=band, setting=setting, arm="blend")
    joined = rd1.join(tail, on=KEYS, how="inner", suffix="_tail")
    swapped = joined.with_columns(*[pl.col(f"{c}_tail").alias(c) for c in TAIL_COLS])
    q = np.sort(swapped.select(QCOLS).to_numpy(), axis=1)
    return swapped.select(*KEYS, "lead_hours", "actual").with_columns(
        *[pl.Series(c, q[:, k]) for k, c in enumerate(QCOLS)]
    )


def run_band_setting(task: tuple[str, str, bool]) -> str:
    """Assemble every arm for one band and setting."""
    band, setting, want_combo = task
    rd1 = recal.load_stored(band=band, setting=setting, arm="blend")
    for variant in recal.VARIANTS:
        write_arm(
            frame=recal.recalibrate(frame=rd1, variant=variant),
            band=band,
            setting=setting,
            arm=variant,
        )
    zi_files = raw_files(root=common.PRED_DIR, band=band, setting=setting, arm="zi")
    if zi_files:
        zi = read_arm_average(files=zi_files, columns=QCOLS)
        write_arm(frame=zi, band=band, setting=setting, arm="zi_alone")
        if not (ASSEMBLED / band / setting / "zi_blend_s0_F.parquet").exists():
            write_arm(
                frame=blend_with(band=band, setting=setting, head=zi),
                band=band,
                setting=setting,
                arm="zi_blend",
            )
    default_files = raw_files(root=common.E318_PRED_DIR, band=band, setting=setting, arm="default")
    if default_files:
        default = read_arm_average(files=default_files, columns=QCOLS)
        write_arm(frame=default, band=band, setting=setting, arm="mdn5_318")
        if not (ASSEMBLED / band / setting / "rd1_5seed_s0_F.parquet").exists():
            write_arm(
                frame=blend_with(band=band, setting=setting, head=default),
                band=band,
                setting=setting,
                arm="rd1_5seed",
            )
    for arm in ("tail", "tail_shuffled"):
        frame = with_tail(band=band, setting=setting, arm=arm, seed_dir_arm=arm)
        if frame is not None:
            write_arm(frame=frame, band=band, setting=setting, arm=arm)
            if arm == "tail" and want_combo:
                write_arm(
                    frame=recal.recalibrate(frame=frame, variant="recal"),
                    band=band,
                    setting=setting,
                    arm="combo",
                )
    root = common.PRED_DIR if setting == "zero_history" else common.E318_PRED_DIR
    pure = raw_files(root=root, band=band, setting=setting, arm="pure_lead_pooled")
    if pure:
        write_arm(
            frame=read_arm_average(files=pure, columns=QCOLS),
            band=band,
            setting=setting,
            arm="pure_lead_pooled",
        )
    return f"{band} {setting}"


def main() -> None:
    """Assemble all bands and settings in parallel."""
    want_combo = "combo" in sys.argv[1:]
    tasks = [(b, s, want_combo) for b in reversed(c310.BAND_ORDER) for s in c310.SETTINGS]
    with multiprocessing.get_context("spawn").Pool(8) as pool:
        for done in pool.imap_unordered(run_band_setting, tasks):
            print("assembled", done, flush=True)


if __name__ == "__main__":
    main()
