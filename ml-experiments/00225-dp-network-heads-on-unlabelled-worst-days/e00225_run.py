"""Experiment 225: do the neural network's corrections help the worst days that no label explains?

Reads experiment 155's stored arm D rows, experiment 172's day tables, and experiment 201's stored
network predictions and feature files. Fits nothing. Writes `work/results.json` and
`work/days.parquet` under experiment 225's private data folder.

    uv run --no-sync python ml-experiments/00225-*/e00225_run.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 225
SLUG: Final[str] = "dp-network-heads-on-unlabelled-worst-days"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
E155_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
    / "work"
    / "dp_preds"
)
E172_WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00172-dp-error-budget-worst-days" / "work"
E201_WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00201-dp-gam-residual-mlp" / "work"
E200_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00200-dp-pooled-correction-head-on-kalman-state" / "work"
)
DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
LEVELS: Final[dict[str, float]] = {
    "q01": 0.01, "q05": 0.05, "q10": 0.10, "q25": 0.25, "q40": 0.40, "q50": 0.50,
    "q60": 0.60, "q75": 0.75, "q90": 0.90, "q95": 0.95, "q99": 0.99,
}  # fmt: skip
NETWORK_ARMS: Final[dict[str, int]] = {"n_arm2": 2, "n_blind": 0, "n_lagged": 4, "n_outputs": 6}
"""The stored networks of experiment 201: arm 2 (headline), arm 0 (no forecast input), arm 4 (plus
lagged power), and arm 6 (arm D's outputs only)."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
KEYS: Final[tuple[str, ...]] = ("site", "power_fcst_init_time", "valid_time")
LOCAL_TZ: Final[str] = "Europe/London"
HALF_HOUR_US: Final[int] = 30 * 60 * 1_000_000
SCORED_FIRST: Final[datetime] = datetime(
    2025, 7, 2, tzinfo=UTC
)  # local days from 2025-07-03 start at 2025-07-02 23:00 UTC in summer time
SCORED_LAST: Final[datetime] = datetime(2026, 6, 30, 23, 59, tzinfo=UTC)
D1_FIRST_LEAD: Final[float] = 15.0
D1_LAST_LEAD: Final[float] = 39.0
MIN_ROWS_PER_DAY: Final[int] = 24
N_DRAWS: Final[int] = 1_000
OUTER_FRACTION: Final[float] = 0.10
SEED: Final[int] = 225
TEMPERATURE_REFERENCE_END: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)


def score_expr(*, prefix: str = "") -> pl.Expr:
    """Return twice the mean pinball loss over the 11 quantiles of columns named `<prefix>q..`."""
    losses = []
    for column, level in LEVELS.items():
        error = pl.col("actual") - pl.col(f"{prefix}{column}")
        losses.append(pl.max_horizontal(level * error, (level - 1.0) * error))
    return 2.0 * pl.mean_horizontal(*losses)


def arm_d_rows() -> pl.DataFrame:
    """Return arm D's D+1 rows of the scored year outside switching masks, with local dates."""
    frames = [
        pl.read_parquet(E155_PREDS / f"d_scored_{site}.parquet").with_columns(site=pl.lit(site))
        for site in DECISION_SITES
    ]
    start = pl.col("valid_time") - pl.duration(microseconds=HALF_HOUR_US)
    return (
        pl.concat(frames)
        .filter(
            pl.col("actual").is_finite(),
            ~pl.col("switching_mask"),
            pl.col("lead_hours") >= D1_FIRST_LEAD,
            pl.col("lead_hours") < D1_LAST_LEAD,
            pl.col("valid_time") >= SCORED_FIRST,
            pl.col("valid_time") <= SCORED_LAST,
        )
        .with_columns(
            score_d=score_expr(),
            local_date=start.dt.convert_time_zone(LOCAL_TZ).dt.date(),
        )
        .select(*KEYS, "lead_hours", "local_date", "actual", "mean", "q50", "score_d")
    )


def network_scores(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return `rows` with one score column per network arm and seed, and the seed mean."""
    keys = rows.select(*KEYS)
    out = rows
    for name, arm in NETWORK_ARMS.items():
        per_seed = []
        for seed in SEEDS:
            frame = pl.read_parquet(E201_WORK / "preds" / f"arm{arm}_seed{seed}.parquet").filter(
                pl.col("site").is_in(DECISION_SITES)
            )
            frame = frame.join(keys, on=list(KEYS), how="inner").with_columns(
                **{f"{name}_s{seed}": score_expr(), f"{name}_q50_s{seed}": pl.col("q50")}
            )
            per_seed.append(frame.select(*KEYS, f"{name}_s{seed}", f"{name}_q50_s{seed}"))
        for frame in per_seed:
            out = out.join(frame, on=list(KEYS), how="left")
        out = out.with_columns(
            **{
                name: pl.mean_horizontal([f"{name}_s{seed}" for seed in SEEDS]),
                f"{name}_q50": pl.mean_horizontal([f"{name}_q50_s{seed}" for seed in SEEDS]),
            }
        )
    return out


def day_table(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return one row per (site, local day): mean scores per arm and seed."""
    columns = ["score_d"]
    for name in NETWORK_ARMS:
        columns += [name, *[f"{name}_s{seed}" for seed in SEEDS]]
    return (
        rows.group_by(["site", "local_date"])
        .agg(n_rows=pl.len(), **{c: pl.col(c).mean() for c in columns})
        .filter(pl.col("n_rows") >= MIN_ROWS_PER_DAY)
    )


def day_temperatures() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the scored days' ensemble-mean temperature and the reference days' temperatures.

    The reference is each site's days before 2025-07-01 (past-weather and ensemble-era rows), which
    stands in for the site's own training temperatures. A day's temperature is the mean of the
    `drv_temp_c` column of experiment 201's feature files over the day's rows.
    """
    local_date = (
        (pl.col("valid_time") - pl.duration(microseconds=HALF_HOUR_US))
        .dt.convert_time_zone(LOCAL_TZ)
        .dt.date()
    )
    scored, reference = [], []
    for site in DECISION_SITES:
        frame = (
            pl.scan_parquet(E201_WORK / "features" / f"{site}.parquet")
            .select("valid_time", "lead_hours", "drv_temp_c", "period")
            .filter(pl.col("drv_temp_c").is_finite())
            .with_columns(local_date=local_date, site=pl.lit(site))
            .collect()
        )
        scored.append(
            frame.filter(pl.col("period") == "scored")
            .filter(pl.col("lead_hours") >= D1_FIRST_LEAD, pl.col("lead_hours") < D1_LAST_LEAD)
            .group_by("site", "local_date")
            .agg(temperature_c=pl.col("drv_temp_c").mean())
        )
        reference.append(
            frame.filter(pl.col("valid_time") < TEMPERATURE_REFERENCE_END)
            .group_by("site", "local_date")
            .agg(temperature_c=pl.col("drv_temp_c").mean())
        )
    return pl.concat(scored), pl.concat(reference)


def percentile_in(*, values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return the share of `reference` at or below each value, between 0 and 1."""
    ordered = np.sort(reference)
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def main() -> None:
    """Build the day table, score the three classes and the controls, and write the results."""
    rows = network_scores(rows=arm_d_rows())
    days = day_table(rows=rows)
    table172 = pl.read_parquet(E172_WORK / "days.parquet").filter(
        pl.col("site").is_in(DECISION_SITES)
    )
    worst = pl.read_parquet(E172_WORK / "worst_days_table.parquet").filter(
        pl.col("site").is_in(DECISION_SITES)
    )
    check = table172.join(days, on=["site", "local_date"], how="inner")
    control = {
        "days_in_172": table172.height,
        "days_here": days.height,
        "days_in_both": check.height,
        "max_abs_score_difference": float((check["score"] - check["score_d"]).abs().max()),
    }
    scored_temperature, reference = day_temperatures()
    reference_by_site = {
        site: reference.filter(pl.col("site") == site)["temperature_c"].to_numpy()
        for site in DECISION_SITES
    }
    scored_temperature = _with_percentile(scored_temperature, reference_by_site)
    days = (
        days.join(scored_temperature, on=["site", "local_date"], how="left")
        .join(
            table172.select(
                "site", "local_date", "excess", "site_median", "labelled", "day_temperature_c"
            ),
            on=["site", "local_date"],
            how="left",
        )
        .with_columns(
            outer=(pl.col("percentile") <= OUTER_FRACTION)
            | (pl.col("percentile") >= 1.0 - OUTER_FRACTION)
        )
    )
    worst_keys = worst.select(
        "site",
        "local_date",
        unlabelled=~pl.any_horizontal("switching", "flagged", "cold", "economy7", "holiday"),
    )
    days = days.join(worst_keys, on=["site", "local_date"], how="left").with_columns(
        worst=pl.col("unlabelled").is_not_null(),
        unlabelled=pl.col("unlabelled").fill_null(False),
    )
    results = {
        "control": control,
        "networks": {
            name: describe(days=days, name=name, seeds=[f"{name}_s{seed}" for seed in SEEDS])
            for name in NETWORK_ARMS
        },
        "temperature": temperature_summary(days=days),
        "random_control": random_control(days=days),
    }
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    days.write_parquet(WORK_DIR / "days.parquet")
    examples = (
        days.filter(pl.col("worst") & pl.col("unlabelled")).sort("excess", descending=True).head(5)
    )
    rows.join(examples.select("site", "local_date", "excess"), on=["site", "local_date"]).select(
        "site", "local_date", "valid_time", "actual", "q50", "n_arm2_q50", "excess"
    ).write_parquet(WORK_DIR / "examples.parquet")
    (WORK_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


def _with_percentile(frame: pl.DataFrame, reference_by_site: dict[str, np.ndarray]) -> pl.DataFrame:
    """Add each day's percentile among its own site's reference days."""
    pieces = []
    for site in DECISION_SITES:
        part = frame.filter(pl.col("site") == site)
        percentile = percentile_in(
            values=part["temperature_c"].to_numpy(), reference=reference_by_site[site]
        )
        pieces.append(part.with_columns(percentile=pl.Series(percentile)))
    return pl.concat(pieces)


def share_removed(*, days: pl.DataFrame, name: str) -> float | None:
    """Return the share of arm D's excess that a network removes over a set of days."""
    if days.height == 0:
        return None
    excess = float(days["score_d"].sum() - days["site_median"].sum())
    return float((days["score_d"] - days[name]).sum() / excess)


def describe(*, days: pl.DataFrame, name: str, seeds: list[str]) -> dict:
    """Return the share of excess removed per class, per seed, and per temperature decile."""
    classes = {
        "unlabelled_worst": days.filter(pl.col("worst") & pl.col("unlabelled")),
        "labelled_worst": days.filter(pl.col("worst") & ~pl.col("unlabelled")),
        "all_worst": days.filter(pl.col("worst")),
        "all_days": days,
    }
    out: dict = {"by_class": {}, "by_seed": {}, "by_decile": {}}
    for label, frame in classes.items():
        out["by_class"][label] = {
            "n_days": frame.height,
            "share_removed": share_removed(days=frame, name=name),
            "mean_score_d": float(frame["score_d"].mean()),
            f"mean_score_{name}": float(frame[name].mean()),
        }
        out["by_seed"][label] = [share_removed(days=frame, name=seed) for seed in seeds]
    deciles = days.with_columns(
        decile=(pl.col("percentile") * 10).floor().clip(0, 9).cast(pl.Int64)
    )
    for decile in range(10):
        frame = deciles.filter(pl.col("decile") == decile)
        out["by_decile"][decile + 1] = {
            "n_days": frame.height,
            "mean_gain": float((frame["score_d"] - frame[name]).mean()) if frame.height else None,
            "mean_gain_worst": float(
                (
                    frame.filter(pl.col("worst"))["score_d"] - frame.filter(pl.col("worst"))[name]
                ).mean()
            )
            if frame.filter(pl.col("worst")).height
            else None,
        }
    return out


def temperature_summary(*, days: pl.DataFrame) -> dict:
    """Return the share of each class in the outer 10% of temperature, ENS-mean and ERA5."""
    out = {}
    era5_outer = pl.col("day_temperature_c")  # experiment 172's ERA5 day mean, for a sensitivity
    for label, frame in {
        "unlabelled_worst": days.filter(pl.col("worst") & pl.col("unlabelled")),
        "labelled_worst": days.filter(pl.col("worst") & ~pl.col("unlabelled")),
        "all_days": days,
    }.items():
        out[label] = {
            "n_days": frame.height,
            "share_outer_10pct": float(frame["outer"].mean()),
            "n_outer": int(frame["outer"].sum()),
            "share_below_10pct": float((frame["percentile"] <= OUTER_FRACTION).mean()),
            "share_above_90pct": float((frame["percentile"] >= 1.0 - OUTER_FRACTION).mean()),
            "mean_era5_day_temperature_c": float(frame.select(era5_outer.mean()).item()),
            "mean_ens_day_temperature_c": float(frame["temperature_c"].mean()),
        }
    return out


def random_control(*, days: pl.DataFrame) -> dict:
    """Draw ordinary days matching each unlabelled worst day's site and month; return the shares.

    The network's gain on ordinary days is the mean fall in the day score in points, because an
    ordinary day's excess over the site's median day is near zero and a share of it is unstable.
    """
    rng = np.random.default_rng(SEED)
    unlabelled = days.filter(pl.col("worst") & pl.col("unlabelled")).with_columns(
        month=pl.col("local_date").dt.month()
    )
    ordinary = days.filter(~pl.col("worst")).with_columns(month=pl.col("local_date").dt.month())
    pools = {
        (row["site"], row["month"]): ordinary.filter(
            (pl.col("site") == row["site"]) & (pl.col("month") == row["month"])
        )
        for row in unlabelled.select("site", "month").unique().iter_rows(named=True)
    }
    shares, removed = [], {name: [] for name in NETWORK_ARMS}
    for _ in range(N_DRAWS):
        picked = []
        for row in unlabelled.iter_rows(named=True):
            pool = pools[(row["site"], row["month"])]
            picked.append(pool[int(rng.integers(pool.height))])
        draw = pl.concat(picked)
        shares.append(float(draw["outer"].mean()))
        for name in NETWORK_ARMS:
            removed[name].append(float((draw["score_d"] - draw[name]).mean()))
    real = {
        "share_outer": float(unlabelled["outer"].mean()),
        **{name: float((unlabelled["score_d"] - unlabelled[name]).mean()) for name in NETWORK_ARMS},
    }
    out = {
        "draws_outer_share": shares,
        "n_draws": N_DRAWS,
        "outer_share_mean": float(np.mean(shares)),
        "outer_share_p05": float(np.percentile(shares, 5)),
        "outer_share_p95": float(np.percentile(shares, 95)),
        "real_percentile_of_outer_share": float(np.mean(np.array(shares) <= real["share_outer"])),
    }
    for name in NETWORK_ARMS:
        draws = np.array(removed[name])
        out[name] = {
            "random_mean_gain_mean": float(draws.mean()),
            "random_mean_gain_p05": float(np.percentile(draws, 5)),
            "random_mean_gain_p95": float(np.percentile(draws, 95)),
            "real_percentile": float(np.mean(draws <= real[name])),
        }
    return out


if __name__ == "__main__":
    main()
