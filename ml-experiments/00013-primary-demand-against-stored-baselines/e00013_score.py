"""Assemble experiment 00013's arms, save their predictions, score them, and write the tables.

Every arm is scored on the cases all arms hold: the validation runs' rows of the 20 series with an
observed power and a valid time before 2026-07-01. The quantile score is twice the mass-weighted
mean pinball loss of the 9 levels and the error of the median is its absolute error, both in points
of capacity. Intervals resample whole validation months and the three seeds. The five planned
contrasts are read at 99%; every other contrast is exploratory and read at 95%. The stored
baselines have no seed and are repeated once per seed.

Output under `work/results/`: `absolute.parquet`, `contrasts.parquet`, `bias.parquet`,
`by_series.parquet`, and `report.md`; the prediction sets under
`data/ml-experiments/00013-.../predictions/`.

    uv run python ml-experiments/00013-primary-demand-against-stored-baselines/e00013_score.py
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00013_common as common
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
STORED_ARMS: Final[tuple[str, ...]] = ("manual_heuristic", "climatology", "production")
N1_ARMS: Final[tuple[str, ...]] = ("n1", "n1_control", "n1_nolag", "n1_nocal")
REFERENCE_ARMS: Final[dict[str, str]] = {"s1": "r0_from2023", "r1": "r1", "r2": "r2"}
"""Arm name to the name of experiment 00005's prediction set."""
SCOPES: Final[tuple[str, ...]] = (
    "all leads",
    "up to 48 h",
    "3 d or more",
    "0-24 h",
    "1-2 d",
    "3-6 d",
    "7-10 d",
    "11-15 d",
)
GROUPS: Final[tuple[str, ...]] = ("Primary", "BSP", "GSP")
HEADLINE_GROUP: Final[str] = "Primary"
PLANNED_LEVEL: Final[float] = 99.0
N_RESAMPLES: Final[int] = 10_000
DECISION_SCOPES: Final[tuple[str, ...]] = ("all leads", "up to 48 h", "3 d or more")
"""Planned contrasts are read at 99% in these scopes; every other scope is exploratory at 95%."""
DEFAULT_RESAMPLES: Final[int] = 2_000
PLANNED: Final[dict[str, tuple[str, str]]] = {
    "C1 (primary): N1 minus manual_heuristic": ("n1", "manual_heuristic"),
    "C2: N1 minus production": ("n1", "production"),
    "C3: production minus manual_heuristic": ("production", "manual_heuristic"),
    "C4: N1 minus N1-nolag": ("n1", "n1_nolag"),
    "C5: N1-nolag minus S1": ("n1_nolag", "s1"),
}
EXPLORATORY: Final[dict[str, tuple[str, str]]] = {
    "E1: N1 minus climatology": ("n1", "climatology"),
    "E2: manual_heuristic minus climatology": ("manual_heuristic", "climatology"),
    "E3: N1 minus N1-control (member pooling)": ("n1", "n1_control"),
    "E4: N1 without calendar minus N1": ("n1_nocal", "n1"),
    "E5: N1 minus R1 (00005's ENS-trained model with lags as inputs)": ("n1", "r1"),
    "E6: S1 minus manual_heuristic": ("s1", "manual_heuristic"),
    "E7: N1-nolag minus production": ("n1_nolag", "production"),
}


def _actual_table(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return every validation case with the observed power over capacity (null if unobserved)."""
    arrays = common.load_series_arrays(series=series)
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
        index = np.asarray(common.grid_index(times=keys["valid_time"]))
        pieces.append(
            keys.with_columns(
                actual=pl.Series(common.take(values=arrays[row["site"]].power, index=index))
            )
        )
    return pl.concat(pieces)


def collect_arms(*, cases: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return every arm's quantiles keyed by series, run, valid time, and seed, with `actual`."""
    arms: dict[str, pl.DataFrame] = {}
    for name in STORED_ARMS:
        stored = pl.read_parquet(ARMS_DIR / f"{name}.parquet").select(KEYS, *common.LEVEL_COLUMNS)
        arms[name] = pl.concat(
            [stored.with_columns(seed=pl.lit(seed).cast(pl.Int8)) for seed in common.SEEDS]
        )
    for name, set_name in REFERENCE_ARMS.items():
        arms[name] = experiment_store.read_predictions(
            experiment_id=common.PREDICTIONS_00005_ID,
            name=set_name,
            row_set_id=common.ROW_SET_ID,
        )
    for name in N1_ARMS:
        paths = sorted(ARMS_DIR.glob(f"{name}_seed*.parquet"))
        if name == "n1":
            paths = [p for p in paths if "control" not in p.name]
        arms[name] = pl.concat([pl.read_parquet(p) for p in paths])
    return {
        name: frame.join(cases, on=KEYS, how="inner", validate="m:1").select(
            *KEYS, "seed", *common.LEVEL_COLUMNS, "actual"
        )
        for name, frame in arms.items()
    }


def prepare(*, arms: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Restrict every arm to the shared cases and add the lead, losses, and group."""
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
                median_error=pl.Series(np.abs(quantiles[:, 4] - actual) * 100.0),
                signed_error=pl.Series((quantiles[:, 4] - actual) * 100.0),
            )
            .drop(*common.LEVEL_COLUMNS, "actual")
        )
    return out


def save_predictions(*, arms: dict[str, pl.DataFrame]) -> None:
    """Write the new arms as immutable prediction sets, skipping sets that exist."""
    for name in (*N1_ARMS, *STORED_ARMS):
        stored = pl.concat(
            [
                pl.read_parquet(p)
                for p in sorted(ARMS_DIR.glob(f"{name}_seed*.parquet"))
                if name != "n1" or "control" not in p.name
            ]
            or [
                pl.read_parquet(ARMS_DIR / f"{name}.parquet")
                .select(KEYS, *common.LEVEL_COLUMNS)
                .join(
                    pl.DataFrame({"seed": list(common.SEEDS)}, schema={"seed": pl.Int8}),
                    how="cross",
                )
            ]
        )
        try:
            experiment_store.write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=name,
                frame=stored.select(*KEYS, "seed", *common.LEVEL_COLUMNS),
                kind="quantiles",
                row_set_id=common.ROW_SET_ID,
            )
        except FileExistsError:
            continue


def _scope(*, frame: pl.DataFrame, scope: str, group: str) -> pl.DataFrame:
    frame = frame.filter(pl.col("group") == group)
    if scope == "all leads":
        return frame
    if scope == "up to 48 h":
        return frame.filter(pl.col("lead_hours") < 48.0)
    if scope == "3 d or more":
        return frame.filter(pl.col("lead_hours") >= 72.0)
    low, high = next((lo, hi) for name, lo, hi in common.LEAD_BANDS if name == scope)
    return frame.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))


def contrast(
    *,
    arms: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    scope: str,
    group: str,
    metric: str,
    level: float,
) -> dict[str, float | int | str]:
    """Return one paired contrast's estimate, interval, month agreement, and series count."""
    seeds = set(arms[treatment]["seed"].unique()) & set(arms[reference]["seed"].unique())
    losses = experiment_store.losses_for_bootstrap(
        arm_losses={
            name: _scope(frame=arms[name], scope=scope, group=group).filter(
                pl.col("seed").is_in(sorted(seeds))
            )
            for name in (treatment, reference)
        }
    )
    decisive = level == PLANNED_LEVEL and group == HEADLINE_GROUP and scope in DECISION_SCOPES
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
    by_series = paired.group_by("site").agg(pl.col("difference").mean())
    return {
        "treatment": treatment,
        "reference": reference,
        "scope": scope,
        "group": group,
        "metric": metric,
        "level": level,
        "difference": float(paired.select(pl.col("difference").mean()).item()),
        "lower": lower,
        "upper": upper,
        "months_negative": int((by_month["difference"] < 0).sum()),
        "months": by_month.height,
        "series_negative": int((by_series["difference"] < 0).sum()),
        "series": by_series.height,
    }


def _fmt(row: dict) -> str:
    return f"{row['difference']:+.2f} ({row['lower']:+.2f} to {row['upper']:+.2f})"


def verdict(*, contrasts: pl.DataFrame) -> str:
    """Return the pre-registered verdict lines from C1, C3, and C5, for the Primary substations."""

    def read(label: str, scope: str, metric: str) -> dict:
        return contrasts.filter(
            pl.col("label").str.starts_with(label)
            & (pl.col("scope") == scope)
            & (pl.col("metric") == metric)
        ).to_dicts()[0]

    c1_mae, c1_qs = (
        read("C1", "all leads", "median_error"),
        read("C1", "all leads", "quantile_score"),
    )
    c1_short = read("C1", "up to 48 h", "median_error")
    c3 = read("C3", "all leads", "median_error")
    c3_long, c5_long = (
        read("C3", "3 d or more", "median_error"),
        read("C5", "3 d or more", "median_error"),
    )
    gain = c1_mae["upper"] < 0
    both = gain and c1_qs["upper"] < 0
    refuted = c1_mae["lower"] > 0 or c3["lower"] <= 0
    short_pass = c1_short["difference"] <= -0.5
    long_pass = c1_mae["difference"] <= -0.2
    ratio = c5_long["difference"] / c3_long["difference"]
    outcome = (
        "refuted"
        if refuted
        else "supported"
        if both
        else "partly supported"
        if gain
        else "inconclusive"
    )
    lines = [
        f"C1 all leads MAE {_fmt(c1_mae)}, quantile score {_fmt(c1_qs)}",
        f"C1 up to 48 h MAE {_fmt(c1_short)}",
        f"C1 thresholds: up to 48 h <= -0.5: {short_pass}; all leads <= -0.2: {long_pass}",
        f"C3 all leads MAE {_fmt(c3)}",
        f"C5 / C3 beyond 3 d: {ratio:+.2f} (needs <= -0.5)",
        f"verdict: {outcome}",
    ]
    return "\n".join(lines)


def main() -> None:
    """Score every arm and write the tables and the report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    cases = _actual_table(series=series)
    raw = collect_arms(cases=cases)
    for name in (*REFERENCE_ARMS, *N1_ARMS, *STORED_ARMS):
        seeds = set(raw[name]["seed"].unique().to_list())
        expected = {0} if name == "n1_nocal" else set(common.SEEDS)
        if seeds != expected:
            msg = f"arm {name} holds seeds {sorted(seeds)}, expected {sorted(expected)}"
            raise ValueError(msg)
    arms = prepare(arms=raw)
    rows = {name: frame.height // frame["seed"].n_unique() for name, frame in arms.items()}
    if len(set(rows.values())) != 1:
        msg = f"arms hold different numbers of shared cases per seed: {rows}"
        raise ValueError(msg)
    save_predictions(arms=arms)
    arm_rows = {name: frame.height // frame["seed"].n_unique() for name, frame in arms.items()}
    absolute = []
    for name, frame in arms.items():
        for group in GROUPS:
            for scope in SCOPES:
                part = experiment_store.losses_for_bootstrap(
                    arm_losses={name: _scope(frame=frame, scope=scope, group=group)}
                )
                for metric in ("quantile_score", "median_error"):
                    interval = bootstrap.bootstrap_absolute(losses=part, arm=name, metric=metric)
                    absolute.append(
                        {"arm": name, "group": group, "scope": scope, "metric": metric, **interval}
                    )
    pl.DataFrame(absolute).write_parquet(RESULTS_DIR / "absolute.parquet")
    contrasts = [
        {
            "label": label,
            "planned": label in PLANNED,
            **contrast(
                arms=arms,
                treatment=treatment,
                reference=reference,
                scope=scope,
                group=group,
                metric=metric,
                level=PLANNED_LEVEL if label in PLANNED and scope in DECISION_SCOPES else 95.0,
            ),
        }
        for label, (treatment, reference) in {**PLANNED, **EXPLORATORY}.items()
        for group in ((HEADLINE_GROUP,) if label in PLANNED else GROUPS)
        for scope in SCOPES
        for metric in ("median_error", "quantile_score")
    ]
    pl.DataFrame(contrasts).write_parquet(RESULTS_DIR / "contrasts.parquet")
    bias = pl.concat(
        [
            frame.filter(pl.col("seed") == 0)
            .with_columns(
                scope=pl.when(pl.col("lead_hours") < 24)
                .then(pl.lit("0-24 h"))
                .when(pl.col("lead_hours") < 48)
                .then(pl.lit("1-2 d"))
                .when(pl.col("lead_hours") < 144)
                .then(pl.lit("3-6 d"))
                .when(pl.col("lead_hours") < 240)
                .then(pl.lit("7-10 d"))
                .otherwise(pl.lit("11-15 d"))
            )
            .group_by("group", "scope")
            .agg(
                bias=pl.col("signed_error").mean(),
                mae=pl.col("median_error").mean(),
                rows=pl.len(),
            )
            .with_columns(arm=pl.lit(name))
            for name, frame in arms.items()
        ]
    )
    bias.write_parquet(RESULTS_DIR / "bias.parquet")
    by_series = pl.concat(
        [
            frame.filter(pl.col("seed") == 0)
            .group_by("time_series_id", "group")
            .agg(mae=pl.col("median_error").mean(), score=pl.col("quantile_score").mean())
            .with_columns(arm=pl.lit(name))
            for name, frame in arms.items()
        ]
    )
    by_series.write_parquet(RESULTS_DIR / "by_series.parquet")
    print("rows per seed:", arm_rows)
    print(verdict(contrasts=pl.DataFrame(contrasts)))
    print(
        pl.DataFrame(absolute)
        .filter((pl.col("group") == HEADLINE_GROUP) & (pl.col("scope") == "all leads"))
        .select("arm", "metric", "value", "lower_95", "upper_95", "n_rows")
    )


if __name__ == "__main__":
    main()
