"""Build experiment 00021's corrected arms from N1, score them, and write the tables.

Each arm adds a recent-error correction to N1's stored quantiles (experiment 00013). For a run with
origin `T`, the window ends at the newest usable half-hour, `T` minus the delay. For each half-hour
in the window the forecast is the median of the run with the latest origin at or before it, and the
error is the observed power over capacity minus that median. Scores are on the cases all arms hold,
in points of capacity; intervals resample whole validation months and the three seeds. The eight
planned contrasts are read at 99.375% over all leads; every other contrast is exploratory (95%).

Output under `work/results/`: `absolute.parquet`, `contrasts.parquet`, `corrections.parquet`, and
the prediction sets under `data/ml-experiments/00021-.../predictions/`.

    uv run python ml-experiments/00021-error-correction-variants-bsp-gsp-demand/e00021_score.py
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00021_common as common
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
STORED_ARMS: Final[tuple[str, ...]] = ("manual_heuristic", "climatology")
MEDIAN: Final[int] = 4
"""Position of the median (q50) among the nine levels."""
MIN_COUNT: Final[int] = 3 * common.STEPS_PER_DAY
"""Fewest paired half-hours (3 days) for a run to get a correction."""
MIN_CELL_COUNT: Final[int] = 3
RATIO_RANGE: Final[tuple[float, float]] = (0.5, 2.0)
MIN_MEAN_MEDIAN: Final[float] = 0.01
DELAY_STEPS: Final[int] = 2 * common.STEPS_PER_DAY + 1
"""Half-hours between the origin and the newest usable observation: 2 days and a half-hour."""
FAST_DELAY_STEPS: Final[int] = 7
"""Experiment 00019's delay of 3.5 hours."""
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 8.0)
N_RESAMPLES: Final[int] = 20_000
DEFAULT_RESAMPLES: Final[int] = 2_000
SCOPES: Final[tuple[str, ...]] = ("all leads", "up to 48 h", "3 d or more")
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "Primary": ("Primary",),
    "BSP": ("BSP",),
    "GSP": ("GSP",),
    "BSP and GSP": ("BSP", "GSP"),
}


@dataclass(frozen=True)
class Variant:
    """One correction rule.

    Attributes:
        kind: `additive`, `multiplicative`, or `half_hour` (additive per Europe/London half-hour).
        window_days: Length of the window of past errors.
        delay_steps: Half-hours between the origin and the window's end.
        half_life_days: Recency weighting half-life, or None for equal weights.
    """

    kind: str
    window_days: int
    delay_steps: int = DELAY_STEPS
    half_life_days: float | None = None


VARIANTS: Final[dict[str, Variant]] = {
    "a14": Variant(kind="additive", window_days=14),
    "a7": Variant(kind="additive", window_days=7),
    "a28": Variant(kind="additive", window_days=28),
    "m14": Variant(kind="multiplicative", window_days=14),
    "h14": Variant(kind="half_hour", window_days=14),
    "r14": Variant(kind="additive", window_days=14, half_life_days=3.5),
    "a14_fast": Variant(kind="additive", window_days=14, delay_steps=FAST_DELAY_STEPS),
}
PRIMARY: Final[tuple[str, ...]] = ("Primary",)
BSP_GSP: Final[tuple[str, ...]] = ("BSP", "GSP")
PLANNED: Final[dict[str, tuple[str, str, tuple[str, ...]]]] = {
    "C1 (primary): A14 minus N1, BSP and GSP": ("a14", "n1", BSP_GSP),
    "C2: A14 minus manual_heuristic, BSP and GSP": ("a14", "manual_heuristic", BSP_GSP),
    "C3: A14 minus N1, Primary": ("a14", "n1", PRIMARY),
    "C4: A7 minus A14, Primary": ("a7", "a14", PRIMARY),
    "C5: A28 minus A14, Primary": ("a28", "a14", PRIMARY),
    "C6: M14 minus A14, Primary": ("m14", "a14", PRIMARY),
    "C7: H14 minus A14, Primary": ("h14", "a14", PRIMARY),
    "C8: R14 minus A14, Primary": ("r14", "a14", PRIMARY),
}
EXPLORATORY: Final[dict[str, tuple[str, str, tuple[str, ...]]]] = {
    "E1: A14 minus manual_heuristic, Primary": ("a14", "manual_heuristic", PRIMARY),
    "E2: A14 minus N1, BSP": ("a14", "n1", ("BSP",)),
    "E3: A14 minus N1, GSP": ("a14", "n1", ("GSP",)),
    "E4: A14-fast minus A14, Primary": ("a14_fast", "a14", PRIMARY),
    "E5: A14-fast minus A14, BSP and GSP": ("a14_fast", "a14", BSP_GSP),
    "E6: A7 minus A14, BSP and GSP": ("a7", "a14", BSP_GSP),
    "E7: A28 minus A14, BSP and GSP": ("a28", "a14", BSP_GSP),
    "E8: H14 minus A14, BSP and GSP": ("h14", "a14", BSP_GSP),
    "E9: R14 minus A14, BSP and GSP": ("r14", "a14", BSP_GSP),
    "E10: N1 minus manual_heuristic, Primary": ("n1", "manual_heuristic", PRIMARY),
    "E11: A14 minus climatology, BSP and GSP": ("a14", "climatology", BSP_GSP),
}


def _actual_table(*, series: pl.DataFrame, power: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return every validation case with the observed power over capacity (null if unobserved)."""
    pieces = []
    for row in series.iter_rows(named=True):
        keys = pl.concat(
            [
                pl.read_parquet(path, columns=list(KEYS))
                .filter(pl.col("time_series_id") == row["time_series_id"])
                .unique()
                for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
            ]
        )
        index = common.grid_index(times=keys["valid_time"])
        pieces.append(
            keys.with_columns(actual=pl.Series(common.take(values=power[row["site"]], index=index)))
        )
    return pl.concat(pieces)


def _local_half_hour_of_day() -> np.ndarray:
    """Return, for each grid half-hour, its Europe/London half-hour of day (0 to 47)."""
    local = common.grid_times().dt.convert_time_zone("Europe/London")
    return (local.dt.hour() * 2 + local.dt.minute() // 30).to_numpy().astype(np.int64)


def _window_statistic(
    *,
    variant: Variant,
    error: np.ndarray,
    median: np.ndarray,
    observed: np.ndarray,
    window: np.ndarray,
    used: np.ndarray,
    hod: np.ndarray,
) -> np.ndarray | float:
    """Return one run's correction: a scalar shift, a ratio, or 48 per-half-hour shifts."""
    if used.sum() < MIN_COUNT:
        return np.zeros(common.STEPS_PER_DAY) if variant.kind == "half_hour" else 0.0
    index = window[used]
    if variant.kind == "multiplicative":
        mean_median = float(median[index].mean())
        if mean_median < MIN_MEAN_MEDIAN:
            return 1.0
        return float(np.clip(observed[index].mean() / mean_median, *RATIO_RANGE))
    if variant.half_life_days is None:
        weights = np.ones(index.size)
    else:
        age_days = (window[-1] - index) / common.STEPS_PER_DAY
        weights = 0.5 ** (age_days / variant.half_life_days)
    overall = float(np.average(error[index], weights=weights))
    if variant.kind == "additive":
        return overall
    cells = np.full(common.STEPS_PER_DAY, overall)
    for cell in range(common.STEPS_PER_DAY):
        in_cell = hod[index] == cell
        if in_cell.sum() >= MIN_CELL_COUNT:
            cells[cell] = float(error[index][in_cell].mean())
    return cells


def corrected_arms(
    *, frame: pl.DataFrame, power: np.ndarray, hod: np.ndarray
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Return every variant's corrected quantiles for one series and seed, and the shifts.

    Args:
        frame: One series' N1 quantiles for one seed, keyed by run and valid time.
        power: The series' observed power over capacity on the dense grid.
        hod: Europe/London half-hour of day of each grid half-hour.

    Returns:
        The corrected frame of each variant, and one row per run and variant with its mean shift
        and its history count.

    Raises:
        ValueError: If a leakage check fails.
    """
    runs = frame["power_fcst_init_time"].unique().sort()
    origins = common.grid_index(times=runs)
    median = np.full(power.shape, np.nan, dtype=np.float64)
    source = np.full(power.shape, -1, dtype=np.int64)
    for position, origin in enumerate(origins):
        part = frame.filter(pl.col("power_fcst_init_time") == runs[position])
        index = common.grid_index(times=part["valid_time"])
        end = origins[position + 1] if position + 1 < len(origins) else np.inf
        keep = (index >= origin) & (index < end)
        median[index[keep]] = part["q50"].to_numpy()[keep]
        source[index[keep]] = origin
    observed = power.astype(np.float64)
    error = observed - median
    row_run = np.searchsorted(origins, common.grid_index(times=frame["power_fcst_init_time"]))
    row_hod = hod[common.grid_index(times=frame["valid_time"])]
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    arms, log = {}, []
    for name, variant in VARIANTS.items():
        span = variant.window_days * common.STEPS_PER_DAY
        statistics, counts = [], []
        for origin in origins:
            newest = origin - variant.delay_steps
            window = np.arange(newest - span + 1, newest + 1)
            used = ~np.isnan(median[window]) & ~np.isnan(power[window])
            if window.min() < 0:
                msg = "the window starts before the grid"
                raise ValueError(msg)
            if (source[window][used] > window[used]).any():
                msg = "a past forecast comes from a run that began after its half-hour"
                raise ValueError(msg)
            statistics.append(
                _window_statistic(
                    variant=variant,
                    error=error,
                    median=median,
                    observed=observed,
                    window=window,
                    used=used,
                    hod=hod,
                )
            )
            counts.append(int(used.sum()))
        if variant.kind == "multiplicative":
            adjusted = quantiles * np.array(statistics)[row_run][:, None]
        elif variant.kind == "additive":
            adjusted = quantiles + np.array(statistics)[row_run][:, None]
        else:
            adjusted = quantiles + np.stack(statistics)[row_run, row_hod][:, None]
        arms[name] = frame.select(*KEYS, "seed").with_columns(
            pl.DataFrame(
                np.sort(adjusted, axis=1).astype(np.float32),
                schema=list(common.LEVEL_COLUMNS),
                orient="row",
            )
        )
        shift = np.array([np.mean(s) for s in statistics])
        log.append(
            pl.DataFrame(
                {
                    "variant": name,
                    "power_fcst_init_time": runs,
                    "shift": shift,
                    "history_count": counts,
                }
            )
        )
    return arms, pl.concat(log)


def load_n1() -> pl.DataFrame:
    """Return N1's three seeds, keyed by series, run, valid time, and seed."""
    paths = sorted(common.ARMS_DIR_00013.glob("n1_seed*.parquet"))
    if [p.name for p in paths] != [f"n1_seed{s}.parquet" for s in common.SEEDS]:
        msg = f"N1 does not hold exactly seeds {common.SEEDS}"
        raise ValueError(msg)
    return pl.concat([pl.read_parquet(p) for p in paths]).sort(*KEYS, "seed")


def prepare(*, arms: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Restrict every arm to the shared cases and add the lead, losses, coverage, and group."""
    shared: pl.DataFrame | None = None
    for frame in arms.values():
        cases = (
            frame.filter(
                pl.all_horizontal(pl.col("actual", *common.LEVEL_COLUMNS).is_finite())
                & (pl.col("valid_time") < HOLDOUT)
            )
            .select(KEYS)
            .unique()
        )
        shared = cases if shared is None else shared.join(cases, on=KEYS, how="inner")
    assert shared is not None
    groups = common.series_table().select(
        "time_series_id", group=pl.col("substation_type").cast(pl.String)
    )
    out = {}
    for name, frame in arms.items():
        kept = frame.join(shared, on=KEYS, how="semi").sort(*KEYS, "seed")
        quantiles = np.sort(kept.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
        actual = kept["actual"].to_numpy()
        out[name] = (
            kept.join(groups, on="time_series_id", how="left", maintain_order="left")
            .with_columns(
                lead_hours=(
                    pl.col("valid_time") - pl.col("power_fcst_init_time")
                ).dt.total_minutes()
                / 60.0,
                quantile_score=pl.Series(
                    common.pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0
                ),
                median_error=pl.Series(np.abs(quantiles[:, MEDIAN] - actual) * 100.0),
                signed_error=pl.Series((quantiles[:, MEDIAN] - actual) * 100.0),
                covered_90=pl.Series(
                    ((actual >= quantiles[:, 0]) & (actual <= quantiles[:, -1])).astype(float)
                ),
                covered_80=pl.Series(
                    ((actual >= quantiles[:, 1]) & (actual <= quantiles[:, -2])).astype(float)
                ),
            )
            .with_columns(pl.col("quantile_score", "median_error", "signed_error").cast(pl.Float64))
            .drop(*common.LEVEL_COLUMNS, "actual")
        )
    return out


def _scope(*, frame: pl.DataFrame, scope: str, groups: tuple[str, ...]) -> pl.DataFrame:
    frame = frame.filter(pl.col("group").is_in(groups))
    if scope == "all leads":
        return frame
    if scope == "up to 48 h":
        return frame.filter(pl.col("lead_hours") < 48.0)
    return frame.filter(pl.col("lead_hours") >= 72.0)


def contrast(
    *,
    arms: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    groups: tuple[str, ...],
    scope: str,
    metric: str,
    level: float,
    decisive: bool,
) -> dict[str, float | int | str]:
    """Return one paired contrast's estimate, interval, and month agreement."""
    losses = experiment_store.losses_for_bootstrap(
        arm_losses={
            name: _scope(frame=arms[name], scope=scope, groups=groups)
            for name in (treatment, reference)
        }
    )
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric=metric,
        level=level,
        n_resamples=N_RESAMPLES if decisive else DEFAULT_RESAMPLES,
    )
    paired = (
        losses.filter(pl.col("arm") == treatment)
        .select("site", "time", "seed", "month", t=pl.col(metric))
        .join(
            losses.filter(pl.col("arm") == reference).select(
                "site", "time", "seed", r=pl.col(metric)
            ),
            on=["site", "time", "seed"],
        )
        .with_columns(difference=pl.col("t") - pl.col("r"))
    )
    by_month = paired.group_by("month").agg(pl.col("difference").mean())
    return {
        "treatment": treatment,
        "reference": reference,
        "scope": scope,
        "metric": metric,
        "level": level,
        "difference": float(paired["difference"].to_numpy().mean()),
        "lower": lower,
        "upper": upper,
        "months_negative": int((by_month["difference"] < 0).sum()),
        "months": by_month.height,
    }


def main() -> None:
    """Build and score every arm and write the tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    power = common.load_power(series=series)
    cases = _actual_table(series=series, power=power)
    hod = _local_half_hour_of_day()
    sites = series.select("time_series_id", "site")
    n1 = load_n1()
    pieces: dict[str, list[pl.DataFrame]] = {name: [] for name in VARIANTS}
    logs = []
    for (series_id, seed), part in n1.group_by("time_series_id", "seed", maintain_order=True):
        site = sites.filter(pl.col("time_series_id") == series_id)["site"][0]
        arms_part, log = corrected_arms(frame=part, power=power[site], hod=hod)
        for name, frame in arms_part.items():
            pieces[name].append(frame)
        logs.append(log.with_columns(time_series_id=pl.lit(series_id), seed=pl.lit(seed)))
    pl.concat(logs).write_parquet(RESULTS_DIR / "corrections.parquet")
    raw: dict[str, pl.DataFrame] = {"n1": n1}
    raw.update({name: pl.concat(frames) for name, frames in pieces.items()})
    for name in STORED_ARMS:
        stored = pl.read_parquet(common.ARMS_DIR_00013 / f"{name}.parquet").select(
            *KEYS, *common.LEVEL_COLUMNS
        )
        raw[name] = pl.concat(
            [stored.with_columns(seed=pl.lit(seed).cast(pl.Int8)) for seed in common.SEEDS]
        )
    full = {
        name: frame.select(*KEYS, "seed", *common.LEVEL_COLUMNS).join(
            cases, on=KEYS, how="inner", validate="m:1"
        )
        for name, frame in raw.items()
    }
    arms = prepare(arms=full)
    rows = {name: frame.height // frame["seed"].n_unique() for name, frame in arms.items()}
    if len(set(rows.values())) != 1:
        msg = f"arms hold different numbers of shared cases per seed: {rows}"
        raise ValueError(msg)
    for name in VARIANTS:
        try:
            experiment_store.write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=name,
                frame=raw[name]
                .select(*KEYS, "seed", *common.LEVEL_COLUMNS)
                .filter(pl.col("valid_time") < HOLDOUT),
                kind="quantiles",
                row_set_id=common.ROW_SET_ID,
            )
        except FileExistsError:
            continue
    absolute = []
    for name, frame in arms.items():
        for scope in SCOPES:
            for group, members in GROUPS.items():
                part = _scope(frame=frame, scope=scope, groups=members)
                losses = experiment_store.losses_for_bootstrap(arm_losses={name: part})
                for metric in ("quantile_score", "median_error"):
                    interval = bootstrap.bootstrap_absolute(losses=losses, arm=name, metric=metric)
                    absolute.append(
                        {"arm": name, "scope": scope, "group": group, "metric": metric, **interval}
                    )
                absolute.append(
                    {
                        "arm": name,
                        "scope": scope,
                        "group": group,
                        "metric": "coverage",
                        "value": float(part["covered_90"].to_numpy().mean()),
                        "coverage_80": float(part["covered_80"].to_numpy().mean()),
                        "bias": float(part["signed_error"].to_numpy().mean()),
                    }
                )
    pl.DataFrame(absolute, strict=False).write_parquet(RESULTS_DIR / "absolute.parquet")
    contrasts = [
        {
            "label": label,
            "planned": label in PLANNED,
            "groups": " and ".join(groups),
            **contrast(
                arms=arms,
                treatment=treatment,
                reference=reference,
                groups=groups,
                scope=scope,
                metric=metric,
                level=PLANNED_LEVEL if label in PLANNED and scope == "all leads" else 95.0,
                decisive=label in PLANNED and scope == "all leads",
            ),
        }
        for label, (treatment, reference, groups) in {**PLANNED, **EXPLORATORY}.items()
        for scope in SCOPES
        for metric in ("quantile_score", "median_error")
    ]
    result = pl.DataFrame(contrasts)
    result.write_parquet(RESULTS_DIR / "contrasts.parquet")
    print("rows per seed:", rows[next(iter(rows))])


if __name__ == "__main__":
    main()
