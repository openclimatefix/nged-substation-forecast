"""Score experiment 00036's arms and write the tables and prediction sets.

Reads `work/arms/<arm>_seed<s>.parquet`, adds experiment 13's stored `n1` (the parity check on B0)
and `climatology` as reference arms, and scores every arm per series on the quantile score and the
MAE of the median, in points of capacity. The six planned contrasts are read at 99.17% over all
leads (Bonferroni across six) with whole months and the three seeds resampled; every other contrast
is exploratory (95%).

Output under `work/results/`: `absolute.parquet`, `contrasts.parquet`, and `cases.parquet` (the
numbers of scored cases), plus prediction sets under `data/ml-experiments/00036-.../predictions/`.

    uv run python ml-experiments/00036-*/e00036_score.py
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00036_common as common
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
KEYS: Final[list[str]] = list(common.KEYS)
FITTED_ARMS: Final[tuple[str, ...]] = ("b0", "b1", "o", "b2", "b3", "b4")
STORED_ARMS: Final[tuple[str, ...]] = ("n1", "climatology")
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 6.0)
N_RESAMPLES: Final[int] = 20_000
DEFAULT_RESAMPLES: Final[int] = 2_000
SCOPES: Final[tuple[str, ...]] = ("all leads", *(name for name, _, _ in common.LEAD_BANDS))
PLANNED: Final[tuple[tuple[str, str, str], ...]] = (
    ("b1", "b0", "D17"),
    ("b1", "b0", "D13"),
    ("b3", "b0", "D6"),
    ("b3", "b0", "D1"),
    ("b4", "b0", "D6"),
    ("b4", "b0", "D1"),
)
"""Treatment, reference, and series of each planned contrast (Boston, Skegness, then the GSPs)."""
EXPLORATORY: Final[tuple[tuple[str, str, str], ...]] = (
    ("o", "b0", "D17"),
    ("o", "b0", "D13"),
    ("o", "b1", "D17"),
    ("o", "b1", "D13"),
    ("b2", "b1", "D17"),
    ("b2", "b1", "D13"),
    ("b2", "b0", "D17"),
    ("b2", "b0", "D13"),
    ("b3", "b0", "D17"),
    ("b3", "b0", "D13"),
    ("b1", "b0", "D6"),
    ("b1", "b0", "D1"),
    ("b0", "n1", "D17"),
    ("b0", "n1", "D13"),
    ("b0", "n1", "D6"),
    ("b0", "n1", "D1"),
)


def load_arms() -> dict[str, pl.DataFrame]:
    """Return every arm's quantiles with a `seed` column, keyed by `KEYS`."""
    arms = {}
    for arm in FITTED_ARMS:
        paths = sorted(ARMS_DIR.glob(f"{arm}_seed*.parquet"))
        if len(paths) != len(common.SEEDS):
            msg = f"arm {arm} has {len(paths)} seed files, expected {len(common.SEEDS)}"
            raise FileNotFoundError(msg)
        arms[arm] = pl.concat([pl.read_parquet(p) for p in paths]).select(
            *KEYS, "seed", *common.LEVEL_COLUMNS
        )
    stored_n1 = experiment_store.read_predictions(
        experiment_id=13, name="n1", row_set_id=common.ROW_SET_ID
    )
    stored_clim = experiment_store.read_predictions(
        experiment_id=13, name="climatology", row_set_id=common.ROW_SET_ID
    )
    arms["n1"] = stored_n1.select(*KEYS, "seed", *common.LEVEL_COLUMNS)
    arms["climatology"] = pl.concat(
        [
            stored_clim.select(*KEYS, *common.LEVEL_COLUMNS).with_columns(
                seed=pl.lit(seed).cast(pl.Int8)
            )
            for seed in common.SEEDS
        ]
    )
    return arms


def actual_table(*, series: pl.DataFrame, power: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return every validation case with the observed power over capacity (null if unobserved)."""
    pieces = []
    for row in series.iter_rows(named=True):
        keys = pl.concat(
            [
                pl.read_parquet(path, columns=KEYS)
                .filter(pl.col("time_series_id") == row["time_series_id"])
                .unique()
                for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
            ]
        )
        index = common.grid_index(times=keys["valid_time"])
        pieces.append(
            keys.with_columns(
                actual=pl.Series(common.take(values=power[row["site"]], index=index)),
                site=pl.lit(row["site"]),
            )
        )
    return pl.concat(pieces).filter(pl.col("actual").is_finite())


def prepare(*, arms: dict[str, pl.DataFrame], cases: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Restrict every arm to the series and cases it shares with all others, and add losses."""
    shared = cases
    out = {}
    for name, frame in arms.items():
        kept = frame.join(shared.select(*KEYS, "site", "actual"), on=KEYS, how="inner").sort(
            *KEYS, "seed"
        )
        quantiles = np.sort(kept.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
        actual = kept["actual"].to_numpy()
        out[name] = kept.with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
            quantile_score=pl.Series(
                common.pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0
            ),
            median_error=pl.Series(np.abs(quantiles[:, common.MEDIAN_INDEX] - actual) * 100.0),
            signed_error=pl.Series((quantiles[:, common.MEDIAN_INDEX] - actual) * 100.0),
            covered_90=pl.Series(
                ((actual >= quantiles[:, 0]) & (actual <= quantiles[:, -1])).astype(float)
            ),
        ).drop(*common.LEVEL_COLUMNS, "actual")
        out[name] = out[name].with_columns(
            pl.col("quantile_score", "median_error", "signed_error").cast(pl.Float64)
        )
    counts = {
        (name, site): frame.filter(pl.col("site") == site).height
        for name, frame in out.items()
        for site in cases["site"].unique()
    }
    return out | {
        "_counts": pl.DataFrame([{"arm": a, "site": s, "rows": n} for (a, s), n in counts.items()])
    }


def in_scope(*, frame: pl.DataFrame, scope: str, site: str) -> pl.DataFrame:
    """Return a series' rows in a lead scope (all leads, or one lead band)."""
    frame = frame.filter(pl.col("site") == site)
    if scope == "all leads":
        return frame
    low, high = next((lo, hi) for name, lo, hi in common.LEAD_BANDS if name == scope)
    return frame.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))


def contrast(
    *,
    arms: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    site: str,
    scope: str,
    metric: str,
    level: float,
    decisive: bool,
) -> dict[str, float | int | str]:
    """Return one paired contrast's estimate, interval, and month agreement."""
    losses = experiment_store.losses_for_bootstrap(
        arm_losses={
            name: in_scope(frame=arms[name], scope=scope, site=site).join(
                in_scope(frame=arms[other], scope=scope, site=site).select(*KEYS, "seed"),
                on=[*KEYS, "seed"],
                how="semi",
            )
            for name, other in ((treatment, reference), (reference, treatment))
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
        "site": site,
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
    """Score every arm and write the tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table().filter(pl.col("substation_type").is_in(["BSP", "GSP"]))
    arrays = common.load_series_arrays(series=series)
    power = {site: arrays[site].power for site in arrays}
    cases = actual_table(series=series, power=power)
    raw = load_arms()
    sizes = {
        name: frame.filter(pl.col("time_series_id").is_in(series["time_series_id"])).height
        for name, frame in raw.items()
    }
    print("rows per arm before filtering:", sizes)
    prepared = prepare(arms=raw, cases=cases)
    counts = prepared.pop("_counts")
    counts.write_parquet(RESULTS_DIR / "cases.parquet")
    arms = dict(prepared)
    for arm in FITTED_ARMS:
        try:
            experiment_store.write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=arm,
                frame=raw[arm].filter(pl.col("valid_time") < HOLDOUT),
                kind="quantiles",
                row_set_id=common.ROW_SET_ID,
            )
        except FileExistsError:
            continue
    absolute = []
    for name, frame in arms.items():
        for site in sorted(frame["site"].unique()):
            for scope in SCOPES:
                part = in_scope(frame=frame, scope=scope, site=site)
                if part.is_empty():
                    continue
                losses = experiment_store.losses_for_bootstrap(arm_losses={name: part})
                for metric in ("quantile_score", "median_error"):
                    interval = bootstrap.bootstrap_absolute(losses=losses, arm=name, metric=metric)
                    absolute.append(
                        {"arm": name, "site": site, "scope": scope, "metric": metric, **interval}
                    )
                absolute.append(
                    {
                        "arm": name,
                        "site": site,
                        "scope": scope,
                        "metric": "coverage",
                        "value": float(part["covered_90"].to_numpy().mean()),
                        "bias": float(part["signed_error"].to_numpy().mean()),
                    }
                )
    pl.DataFrame(absolute, strict=False).write_parquet(RESULTS_DIR / "absolute.parquet")
    planned = set(PLANNED)
    rows = []
    for treatment, reference, site in (*PLANNED, *EXPLORATORY):
        if treatment not in arms or reference not in arms:
            continue
        for scope in SCOPES:
            for metric in ("quantile_score", "median_error"):
                decisive = (treatment, reference, site) in planned and scope == "all leads"
                rows.append(
                    {
                        "planned": (treatment, reference, site) in planned,
                        **contrast(
                            arms=arms,
                            treatment=treatment,
                            reference=reference,
                            site=site,
                            scope=scope,
                            metric=metric,
                            level=PLANNED_LEVEL if decisive else 95.0,
                            decisive=decisive,
                        ),
                    }
                )
    pl.DataFrame(rows).write_parquet(RESULTS_DIR / "contrasts.parquet")


if __name__ == "__main__":
    main()
