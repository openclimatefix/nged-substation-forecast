"""Assemble the arms of experiment 00034 from the refit predictions, in four forms each.

An arm is a regime (which refit forecasts which validation run) applied to one configuration's
refit predictions (`work/anchor_preds/<config>/`, from `e00034_fit.py` and `e00034_fit_warm.py`).
Each arm is written as `work/arms/<arm>.parquet` (the quantiles as fitted), `<arm>_c` (plus the
14-day correction), and `<arm>_cs` (plus the correction and the spread factor 1.15), each joined to
the observed power. The correction is experiment 00019's, computed from each arm's own earlier
forecasts. The daily arms forecast only three months, so their correction history outside those
months comes from the weekly arm of the same configuration (the `history` column of `ARMS`).

The reference arms of experiment 00033 (`work/reference/`) are read as they are: `n1_cs` is the
best-guess stack, `w22_ens` is the frozen N1 with a half-life of 0.5 years, and the others are
the stored baselines and the rank-test arms of experiments 24 and 30.

    uv run python ml-experiments/00034-*/e00034_assemble.py [--only ARM ...]
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Final, NamedTuple

import e00034_common as common
import e00034_fit as fitting
import e00034_regimes as regimes
import polars as pl

ARMS_DIR: Final[Path] = common.ARMS_DIR
REFERENCE_DIR: Final[Path] = common.WORK_DIR / "reference"


class Arm(NamedTuple):
    """One arm: the configuration folder, the regime, the seeds, and the history arm.

    Attributes:
        folder: The folder under `anchor_preds` holding the refit predictions.
        regime: A name from `regimes.FIXED_REGIMES`, `daily`, or `interp`.
        seeds: The seeds the arm is built for.
        history: The arm whose forecasts fill the correction history outside the arm's own
            months, or `None` when the arm covers every run.
    """

    folder: str
    regime: str
    seeds: tuple[int, ...]
    history: str | None = None


ARMS: Final[dict[str, Arm]] = {
    "yearly": Arm("n1h", "yearly", (0, 1, 2)),
    "half_yearly": Arm("n1h", "half_yearly", (0,)),
    "three_a_year": Arm("n1h", "three_a_year", (0,)),
    "interp": Arm("n1h", "interp", (0,)),
    "quarterly": Arm("n1h", "quarterly", (0,)),
    "monthly": Arm("n1h", "monthly", (0, 1, 2)),
    "weekly": Arm("n1h_weekly", "weekly", (0,)),
    "daily": Arm("n1h_daily", "daily", (0,), history="weekly"),
    "monthly_2y": Arm("n1_2y", "monthly", (0,)),
    "roll12": Arm("roll12", "monthly", (0,)),
    "a7_monthly": Arm("a7", "monthly", (0,)),
    "a7_three_a_year": Arm("a7", "three_a_year", (0,)),
    "a7_roll12": Arm("a7_roll12", "monthly", (0,)),
    "type_id_monthly": Arm("type_id_n1h", "monthly", (0,)),
    "type_id_three_a_year": Arm("type_id_n1h", "three_a_year", (0,)),
    "type_id_roll12": Arm("type_id_roll12", "monthly", (0,)),
    **{
        f"warm{k}_{regime}": Arm(f"warm{k}_{regime}", regime, (0,))
        for k in (20, 50)
        for regime in ("monthly", "weekly")
    },
    "warm20_daily": Arm("warm20_daily", "daily", (0,), history="warm20_weekly"),
}


def read_refit(*, folder: str, site: str, seed: int, refit: datetime) -> pl.DataFrame:
    """Read one refit's predictions."""
    return pl.read_parquet(
        fitting.job_path(
            root=fitting.ANCHOR_PREDS_DIR, config=folder, site=site, seed=seed, refit=refit
        )
    )


def regime_rows(*, arm: Arm, site: str, seed: int, origins: list[datetime]) -> pl.DataFrame:
    """Return one series' quantiles for an arm and seed, from the regime's chosen refits."""
    pieces = []
    if arm.regime == "interp":
        anchors = regimes.THREE_A_YEAR
        for origin in origins:
            earlier = regimes.chosen_refit(refits=anchors, origin=origin)
            later = [a for a in anchors if a > origin]
            first = read_refit(folder=arm.folder, site=site, seed=seed, refit=earlier).filter(
                pl.col("power_fcst_init_time") == origin
            )
            if not later:
                pieces.append(first)
                continue
            weight = (origin - earlier) / (min(later) - earlier)
            second = read_refit(folder=arm.folder, site=site, seed=seed, refit=min(later)).filter(
                pl.col("power_fcst_init_time") == origin
            )
            both = first.join(
                second, on=list(common.KEYS), suffix="_next", validate="1:1", maintain_order="left"
            )
            pieces.append(
                both.select(
                    *common.KEYS,
                    *[
                        ((1 - weight) * pl.col(c) + weight * pl.col(f"{c}_next")).alias(c)
                        for c in common.LEVEL_COLUMNS
                    ],
                    "seed",
                )
            )
        return pl.concat(pieces)
    served = (
        {o: o for o in regimes.daily_refits(origins=origins)}
        if arm.regime == "daily"
        else {
            origin: regimes.chosen_refit(
                refits=regimes.regime_refits(name=arm.regime, origins=origins), origin=origin
            )
            for origin in origins
        }
    )
    by_refit: dict[datetime, list[datetime]] = {}
    for origin, refit in served.items():
        by_refit.setdefault(refit, []).append(origin)
    for refit, chosen in by_refit.items():
        frame = read_refit(folder=arm.folder, site=site, seed=seed, refit=refit)
        part = frame.filter(pl.col("power_fcst_init_time").is_in(chosen))
        if part["power_fcst_init_time"].n_unique() != len(chosen):
            msg = (
                f"{arm.folder} {site} seed {seed} refit {refit}: the file lacks some assigned runs"
            )
            raise ValueError(msg)
        pieces.append(part)
    return pl.concat(pieces)


def arm_table(*, name: str, series: pl.DataFrame, origins: list[datetime]) -> pl.DataFrame:
    """Return an arm's raw quantiles for every series and seed that has refit predictions.

    Args:
        name: The arm's name in `ARMS`.
        series: `common.series_table()`.
        origins: The validation runs' forecast origins.

    Returns:
        The quantiles keyed by series, run, valid time, and seed.

    Raises:
        FileNotFoundError: If no series has the arm's seed-0 predictions.
    """
    arm = ARMS[name]
    pieces = []
    for row in series.iter_rows(named=True):
        for seed in arm.seeds:
            try:
                frame = regime_rows(arm=arm, site=row["site"], seed=seed, origins=origins)
            except FileNotFoundError:
                continue
            pieces.append(frame.with_columns(seed=pl.lit(seed).cast(pl.Int8)))
    if not pieces:
        msg = f"no refit predictions for {name}"
        raise FileNotFoundError(msg)
    return pl.concat(pieces).select(*common.KEYS, "seed", *common.LEVEL_COLUMNS)


def write_forms(
    *,
    name: str,
    raw: pl.DataFrame,
    history: pl.DataFrame | None,
    series: pl.DataFrame,
    arrays: dict[str, common.SeriesArrays],
    actual: pl.DataFrame,
) -> None:
    """Write an arm's four forms; corrections use `history`, then keep only `raw`'s runs."""
    base = raw if history is None else history
    corrected = common.with_corrections(base=base, series=series, arrays=arrays)
    if history is not None:
        corrected = corrected.join(
            raw.select(*common.KEYS, "seed"), on=[*common.KEYS, "seed"], how="semi"
        )
    forms = {
        name: raw,
        f"{name}_s": common.stack(frame=corrected, correct=False, spread=True),
        f"{name}_c": common.stack(frame=corrected, correct=True, spread=False),
        f"{name}_cs": common.stack(frame=corrected, correct=True, spread=True),
    }
    for form, table in forms.items():
        joined = table.join(actual, on=list(common.KEYS), how="left", validate="m:1")
        target = ARMS_DIR / f"{form}.parquet"
        joined.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)


def history_for(
    *,
    arm: Arm,
    raw: pl.DataFrame,
    built: dict[str, pl.DataFrame],
    series: pl.DataFrame,
    origins: list[datetime],
) -> pl.DataFrame | None:
    """Return the table the correction history is computed on, or `None` for a full-year arm.

    The history is the history arm's forecasts for the runs outside `raw`'s own runs, plus `raw`.
    """
    if arm.history is None:
        return None
    if arm.history not in built:
        built[arm.history] = arm_table(name=arm.history, series=series, origins=origins)
    outside = built[arm.history].join(
        raw.select("power_fcst_init_time").unique(), on="power_fcst_init_time", how="anti"
    )
    return pl.concat([outside, raw.filter(pl.col("seed").is_in(outside["seed"].unique()))])


def main() -> None:
    """Build and write every arm whose refit predictions exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    series_all = common.series_table()
    series = series_all
    arrays = common.load_series_arrays(series=series_all)
    actual = common.actual_table(series=series_all, arrays=arrays)
    origins = regimes.run_origins()
    built: dict[str, pl.DataFrame] = {}
    for name, arm in ARMS.items():
        if args.only and name not in args.only:
            continue
        try:
            raw = arm_table(name=name, series=series, origins=origins)
        except FileNotFoundError:
            print(f"{name}: refit predictions missing, skipped", flush=True)
            continue
        built[name] = raw
        needs = history_for(arm=arm, raw=raw, built=built, series=series, origins=origins)
        write_forms(name=name, raw=raw, history=needs, series=series, arrays=arrays, actual=actual)
        print(f"{name}: wrote 4 forms, {raw.height:,} rows", flush=True)


if __name__ == "__main__":
    main()
