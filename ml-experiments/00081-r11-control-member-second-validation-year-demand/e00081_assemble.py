"""Assemble the arms of experiment 00081 into one prediction table per arm and fold.

An arm is a regime (which refit forecasts which daily run) applied to one configuration's refit
predictions (`work/anchor_preds/<fold>/<config>/`, from `e00081_fit.py`). Every table is keyed by
series, run, and valid time, holds the nine quantiles, the observed power (`actual`, a fraction of
capacity), and the series label (`site`), and covers every run of the fold.

Arms (`ARM_PLAN`; a form is the raw quantiles, `s` with the spread factor 1.15, `c` with the 14-day
correction, `cs` with both):

- `frozen_cs` (the stack), `frozen_s`, `frozen_c`, `frozen` (fold `b`); `frozen_cs` (fold `c`);
  `frozen_s` (fold `d`).
- `monthly` and `weekly` (raw) and `monthly_s`, `weekly_s` (folds `b` and `c`).
- `roll12_s`, `roll12_no_christmas_s`, `roll12_hl2y_s` (folds `b` and `c`).
- `manual_heuristic` and `climatology` as `e00081_fit.py` wrote them (fold `d`: `manual_heuristic`).
- `blend` (fold `b`): the four members `weekly_s`, `monthly_s`, `roll12_s`, and `frozen_cs` combined
  by one weight vector per forecast-start month, chosen on the earlier months' cases only, from a
  grid of quarters. The first month has equal weights. The weights go to
  `work/results/b/blend_weights`.

The correction reads each past half-hour's forecast as issued: the forecast of the latest earlier
daily run, from the refit that served it. A table holds only the fold's runs, so no correction reads
a forecast older than the fold's first run.

    uv run python ml-experiments/00081-*/e00081_assemble.py [--fold b c d]
        [--parts arms predictions] [--smoke]
"""

import argparse
import itertools
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00081_common as common
import e00081_control_nwp as control
import e00081_fit as fitting
import numpy as np
import polars as pl

from studies import experiment_store

ArmPlanType = dict[str, tuple[str, str, tuple[str, ...]]]
ARM_PLAN: Final[dict[common.FoldNameType, ArmPlanType]] = {
    "b": {
        "frozen": ("frozen", "frozen", ("raw", "s", "cs", "c")),
        "monthly": ("n1h", "monthly", ("raw", "s")),
        "weekly": ("n1h", "weekly", ("raw", "s")),
        "roll12": ("roll12", "monthly", ("s",)),
        "roll12_no_christmas": ("roll12_no_christmas", "monthly", ("s",)),
        "roll12_hl2y": ("roll12_hl2y", "monthly", ("s",)),
    },
    "c": {
        "frozen": ("frozen", "frozen", ("cs",)),
        "monthly": ("n1h", "monthly", ("raw", "s")),
        "weekly": ("n1h", "weekly", ("raw", "s")),
        "roll12": ("roll12", "monthly", ("s",)),
        "roll12_no_christmas": ("roll12_no_christmas", "monthly", ("s",)),
        "roll12_hl2y": ("roll12_hl2y", "monthly", ("s",)),
    },
    "d": {"frozen": ("frozen", "frozen", ("s",))},
}
BLEND_MEMBERS: Final[tuple[str, ...]] = ("weekly_s", "monthly_s", "roll12_s", "frozen_cs")
GRID_STEP: Final[int] = 4
"""Blend weights are multiples of 1/4."""
SMOKE_RUN_DAYS: Final[tuple[int, ...]] = (1, 2, 3)
"""The days of July 2023 that `e00081_fit.py --smoke` forecasts."""


def _arms_dir(*, smoke: bool) -> Path:
    return common.ARMS_DIR.with_name("arms_smoke") if smoke else common.ARMS_DIR


def _preds_dir(*, smoke: bool) -> Path:
    return common.PREDS_DIR.with_name("anchor_preds_smoke") if smoke else common.PREDS_DIR


def _results_dir(*, fold: str, smoke: bool) -> Path:
    base = common.RESULTS_DIR.with_name("results_smoke") if smoke else common.RESULTS_DIR
    return base / fold


def fold_origins(*, fold: common.Fold, smoke: bool) -> list[datetime]:
    """Return the `power_fcst_init_time` of a fold's runs (the three smoke runs in a smoke test)."""
    if smoke:
        return [common.origin_of(run=datetime(2023, 7, day, tzinfo=UTC)) for day in SMOKE_RUN_DAYS]
    runs = common.fold_runs(fold=fold, folder=control.BACKFILL_DIR)
    return [common.origin_of(run=run) for run in runs]


def regime_rows(
    *,
    root: Path,
    fold: common.Fold,
    config: str,
    regime: str,
    site: str,
    origins: list[datetime],
) -> pl.DataFrame:
    """Return one series' quantiles for a configuration and regime, from the chosen refits.

    Raises:
        ValueError: If a refit file lacks a run assigned to it.
        FileNotFoundError: If a refit file of the series is missing.
    """
    pieces = []
    for refit, chosen in fitting.served(fold=fold, regime=regime, origins=origins).items():
        path = fitting.job_path(root=root, fold=fold.name, config=config, site=site, refit=refit)
        frame = pl.read_parquet(path).filter(pl.col("power_fcst_init_time").is_in(chosen))
        if frame["power_fcst_init_time"].n_unique() != len(chosen):
            msg = f"{fold.name} {config} {site} refit {refit}: the file lacks some assigned runs"
            raise ValueError(msg)
        pieces.append(frame)
    return pl.concat(pieces)


def attach_actual(
    *, table: pl.DataFrame, labels: dict[int, str], power_of: dict[int, np.ndarray]
) -> pl.DataFrame:
    """Add `site` and `actual` (observed power over capacity) to a quantile table.

    Args:
        table: Quantiles keyed by series, run, valid time, and seed.
        labels: Each series identifier's site label.
        power_of: Each series identifier's observed power over capacity on the dense grid.

    Returns:
        The table with the two columns added, `actual` NaN where the grid has no observation.
    """
    pieces = []
    for (series_id,), part in table.partition_by("time_series_id", as_dict=True).items():
        index = np.asarray(common.grid_index(times=part["valid_time"]))
        pieces.append(
            part.with_columns(
                site=pl.lit(labels[int(series_id)]),
                actual=pl.Series(common.take(values=power_of[int(series_id)], index=index)),
            )
        )
    return pl.concat(pieces).sort(*common.KEYS, "seed")


def write_arm(*, fold: str, name: str, table: pl.DataFrame, smoke: bool) -> None:
    """Write one arm's table."""
    common.atomic_write(frame=table, target=_arms_dir(smoke=smoke) / fold / f"{name}.parquet")


def weight_grid(*, members: int) -> np.ndarray:
    """Return every weight vector on the grid that sums to 1, shape (vectors, members)."""
    rows = [
        v for v in itertools.product(range(GRID_STEP + 1), repeat=members) if sum(v) == GRID_STEP
    ]
    return np.array(rows, dtype=np.float64) / GRID_STEP


def blend(*, members: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Return the weighted mean of each quantile level, sorted; weights are per member or case."""
    stacked = np.stack(members, axis=0)
    mixed = (
        np.tensordot(weights, stacked, axes=1)
        if weights.ndim == 1
        else np.einsum("cm,mcl->cl", weights, stacked)
    )
    return np.sort(mixed, axis=1)


def history_mask(*, cases: pl.DataFrame, month: str) -> np.ndarray:
    """Return the cases a scored month may learn from: earlier runs with earlier valid times."""
    start = np.datetime64(
        datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC).replace(tzinfo=None), "us"
    )
    valid = cases["valid_time"].to_numpy().astype("datetime64[us]")
    return (cases["month"].to_numpy() < month) & (valid < start)


def rolling_blend(
    *, cases: pl.DataFrame, members: dict[str, np.ndarray]
) -> tuple[np.ndarray, pl.DataFrame]:
    """Return the blend with weights chosen on earlier months, and the weight table.

    A month with no history gets equal weights. Ties go to the vector nearest equal weights.

    Args:
        cases: The shared cases, with `valid_time`, `month` (of the forecast start), and `actual`.
        members: Each member's nine quantiles per case, in the cases' order.

    Returns:
        The blended quantiles per case and a table of each month's weights and history size.
    """
    names = list(members)
    grid = weight_grid(members=len(names))
    equal = np.full(len(names), 1.0 / len(names))
    nearest = np.argsort(np.abs(grid - equal).sum(axis=1), kind="stable")
    actual = cases["actual"].to_numpy().astype(np.float64)
    arrays = [members[name] for name in names]
    scores = np.stack(
        [
            common.pinball_quantile_score(quantiles=blend(members=arrays, weights=w), actual=actual)
            for w in grid
        ],
        axis=1,
    )
    month_col = cases["month"].to_numpy()
    case_weights = np.tile(equal, (cases.height, 1))
    rows = []
    for month in sorted(cases["month"].unique().to_list()):
        history = history_mask(cases=cases, month=month)
        count = int(history.sum())
        if count:
            mean = scores[history].mean(axis=0)
            chosen = grid[nearest[np.argsort(mean[nearest], kind="stable")[0]]]
        else:
            chosen = equal
        case_weights[month_col == month] = chosen
        rows.append(
            {
                "month": month,
                "history_cases": count,
                **{f"w_{name}": float(w) for name, w in zip(names, chosen, strict=True)},
            }
        )
    return blend(members=arrays, weights=case_weights), pl.DataFrame(rows)


def assemble_blend(*, smoke: bool) -> None:
    """Write fold `b`'s blend table and its weight table from the four member tables."""
    folder = _arms_dir(smoke=smoke) / "b"
    paths = [folder / f"{name}.parquet" for name in BLEND_MEMBERS]
    if not all(path.exists() for path in paths):
        print("blend: member tables missing, skipped", flush=True)
        return
    finite = pl.all_horizontal(pl.col(*common.LEVEL_COLUMNS, "actual").is_finite())
    tables = [pl.read_parquet(path).filter(finite) for path in paths]
    keys = tables[0].select(*common.KEYS, "seed")
    for table in tables[1:]:
        keys = keys.join(table.select(*common.KEYS, "seed"), on=[*common.KEYS, "seed"], how="inner")
    keys = keys.sort(*common.KEYS, "seed")
    aligned = [
        keys.join(table, on=[*common.KEYS, "seed"], how="left", validate="1:1").sort(
            *common.KEYS, "seed"
        )
        for table in tables
    ]
    cases = (
        aligned[0]
        .select(*common.KEYS, "seed", "site", "actual")
        .with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
    )
    members = {
        name: table.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
        for name, table in zip(BLEND_MEMBERS, aligned, strict=True)
    }
    quantiles, weights = rolling_blend(cases=cases, members=members)
    table = cases.drop("month").with_columns(
        pl.DataFrame(quantiles.astype(np.float32), schema=list(common.LEVEL_COLUMNS))
    )
    write_arm(fold="b", name="blend", table=table, smoke=smoke)
    results = _results_dir(fold="b", smoke=smoke)
    results.mkdir(parents=True, exist_ok=True)
    common.atomic_write(frame=weights, target=results / "blend_weights.parquet")
    print(f"blend: {table.height:,} cases", flush=True)


def assemble_fold(*, fold_name: common.FoldNameType, smoke: bool) -> None:
    """Write every arm table of one fold."""
    fold = common.FOLDS[fold_name]
    series = common.series_table()
    series = series.head(1) if smoke else series
    power_of_site = common.load_power_arrays(series=series)
    labels = {int(row["time_series_id"]): row["site"] for row in series.iter_rows(named=True)}
    power_of = {
        int(row["time_series_id"]): power_of_site[row["site"]]
        for row in series.iter_rows(named=True)
    }
    root = _preds_dir(smoke=smoke)
    origins = fold_origins(fold=fold, smoke=smoke)
    for name, (config, regime, forms) in ARM_PLAN[fold_name].items():
        started = time.monotonic()
        raw = pl.concat(
            [
                regime_rows(
                    root=root,
                    fold=fold,
                    config=config,
                    regime=regime,
                    site=site,
                    origins=origins,
                )
                for site in series["site"].to_list()
            ]
        )
        corrected = (
            common.with_corrections(
                base=raw, series=series, power_of=power_of_site, window_days=common.CORRECTION_DAYS
            )
            if {"cs", "c"} & set(forms)
            else None
        )
        for form in forms:
            if form == "raw":
                table, label = raw, name
            elif form == "s":
                table, label = common.stack(frame=raw, correct=False, spread=True), f"{name}_s"
            else:
                assert corrected is not None
                table = common.stack(frame=corrected, correct=True, spread=form == "cs")
                label = f"{name}_{form}"
            write_arm(
                fold=fold_name,
                name=label,
                table=attach_actual(table=table, labels=labels, power_of=power_of),
                smoke=smoke,
            )
        print(
            f"{fold_name} {name}: forms {forms}, {raw.height:,} rows, "
            f"{time.monotonic() - started:.0f} s",
            flush=True,
        )
    for baseline in fitting.BASELINES_BY_FOLD[fold_name]:
        files = sorted((root / fold_name / baseline).glob("*.parquet"))
        if not files:
            print(f"{fold_name} {baseline}: files missing, skipped", flush=True)
            continue
        table = pl.concat([pl.read_parquet(path) for path in files])
        write_arm(
            fold=fold_name,
            name=baseline,
            table=attach_actual(table=table, labels=labels, power_of=power_of),
            smoke=smoke,
        )
        print(f"{fold_name} {baseline}: {table.height:,} rows", flush=True)
    if fold_name == "b":
        assemble_blend(smoke=smoke)


def write_arm_predictions(*, fold_name: common.FoldNameType, smoke: bool) -> None:
    """Save every arm's rows of one fold as a prediction set named `<arm>_fold_<fold>`."""
    if smoke:
        print("smoke run: no predictions saved", flush=True)
        return
    fold = common.FOLDS[fold_name]
    for path in sorted((_arms_dir(smoke=smoke) / fold_name).glob("*.parquet")):
        frame = pl.read_parquet(path).with_columns(
            lead_hours=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
            )
        )
        try:
            experiment_store.write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=f"{path.stem}_fold_{fold_name}",
                frame=frame,
                kind="quantiles",
                row_set_id=fold.row_set_id,
            )
        except FileExistsError:
            print(f"{path.stem}_fold_{fold_name}: already saved", flush=True)
            continue
        print(f"{path.stem}_fold_{fold_name}: saved {frame.height:,} rows", flush=True)


def main() -> None:
    """Build and write the requested parts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", nargs="+", choices=sorted(common.FOLDS), default=["b", "c", "d"])
    parser.add_argument(
        "--parts", nargs="+", choices=("arms", "predictions"), default=["arms", "predictions"]
    )
    parser.add_argument(
        "--smoke", action="store_true", help="read the smoke refit files, write apart"
    )
    args = parser.parse_args()
    folds = common.fold_names(selected=["b"] if args.smoke else args.fold)
    for fold_name in folds:
        if "arms" in args.parts:
            assemble_fold(fold_name=fold_name, smoke=args.smoke)
        if "predictions" in args.parts:
            write_arm_predictions(fold_name=fold_name, smoke=args.smoke)


if __name__ == "__main__":
    main()
