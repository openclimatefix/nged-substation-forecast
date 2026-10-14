"""Assemble experiment 00018's arms, save their predictions, score them, and write the tables.

Every arm is scored on the cases all arms hold: the validation runs' rows of the 20 series with an
observed power and a valid time before 2026-07-01. The quantile score is twice the mass-weighted
mean pinball loss of the 9 levels and the error of the median is its absolute error, both in points
of capacity. Intervals resample whole validation months and the three seeds. The six planned
contrasts are read at 99.17% (Bonferroni across six); every other contrast is exploratory and read
at 95%. The stored baselines have no seed and are repeated once per seed. The spread factor `k` of
the spread arm is chosen per substation type from `kgrid_seed<s>.parquet`, which holds calibration
rows only.

Output under `work/results/`: `absolute.parquet`, `contrasts.parquet`, `bias.parquet`,
`by_series.parquet`, `k.json`; the prediction sets under
`data/ml-experiments/00018-.../predictions/`.

    uv run python \
        ml-experiments/00018-staged-chain-vs-n1-serving-latency-lag-anomalies-demand/e00018_score.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00018_common as common
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
NEW_ARMS: Final[dict[str, str]] = {
    "ch_none": "none",
    "ch_e": "e",
    "ch_e_nodelay": "e_nodelay",
    "ch_s": "s",
    "ch_s_nocams": "s_nocams",
    "ch_p": "p",
    "ch_s_delay3_5": "s_delay3_5",
}
"""Arm name to the stem of its file in `work/arms/`; `ch_s_spread` is derived from `ch_s`."""
SPREAD_ARM: Final[str] = "ch_s_spread"
STORED_ARMS: Final[tuple[str, ...]] = ("n1", "manual_heuristic", "climatology", "production")
STORED_ID: Final[int] = 13
"""Experiment 00013 wrote these prediction sets."""
REFERENCE_ARMS: Final[dict[str, str]] = {"r0c": "r0c_from2023"}
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
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 6)
N_RESAMPLES: Final[int] = 10_000
DECISION_SCOPES: Final[tuple[str, ...]] = ("all leads", "up to 48 h", "3 d or more")
"""Planned contrasts are read at 99.17% in these scopes; every other scope is exploratory at 95%."""
DEFAULT_RESAMPLES: Final[int] = 2_000
PLANNED: Final[dict[str, tuple[str, str]]] = {
    "C1 (primary): CH-S minus N1": ("ch_s", "n1"),
    "C2: CH-S minus R0c": ("ch_s", "r0c"),
    "C3: CH-S minus manual_heuristic": ("ch_s", "manual_heuristic"),
    "C4: CH-S minus CH-E": ("ch_s", "ch_e"),
    "C5: CH-S minus CH-none": ("ch_s", "ch_none"),
    "C6: CH-S-spread minus CH-S": (SPREAD_ARM, "ch_s"),
}
EXPLORATORY: Final[dict[str, tuple[str, str]]] = {
    "E1: CH-S minus CH-S-noCAMS": ("ch_s", "ch_s_nocams"),
    "E2: CH-S minus CH-P": ("ch_s", "ch_p"),
    "E3: CH-E minus CH-E-nodelay (cost of the telemetry delay)": ("ch_e", "ch_e_nodelay"),
    "E4: CH-S minus the production recipe": ("ch_s", "production"),
    "E5: CH-S-delay3.5 minus N1 (equal telemetry freshness)": ("ch_s_delay3_5", "n1"),
    "E6: CH-S-delay3.5 minus manual_heuristic": ("ch_s_delay3_5", "manual_heuristic"),
    "E7: CH-S minus CH-S-delay3.5 (cost of 3.5 more hours of delay)": ("ch_s", "ch_s_delay3_5"),
    "E8: N1 minus manual_heuristic": ("n1", "manual_heuristic"),
}


def _actual_table(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return every validation case with the observed power over capacity (null if unobserved)."""
    arrays = common.load_series_arrays(series=series, config="from2023")
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


def choose_k(*, series: pl.DataFrame) -> dict[str, float]:
    """Return the spread factor of each substation type from the calibration rows' held-out loss."""
    grid = pl.concat(
        [pl.read_parquet(p) for p in sorted(common.WORK_DIR.glob("arms/kgrid_*.parquet"))]
    )
    groups = series.select("time_series_id", group=pl.col("substation_type").cast(pl.String))
    totals = (
        grid.join(groups, on="time_series_id")
        .group_by("group", "k")
        .agg(loss=pl.col("loss_sum").sum() / pl.col("n").sum())
    )
    return {
        group: float(totals.filter(pl.col("group") == group).sort("loss")["k"][0])
        for group in groups["group"].unique()
    }


def spread(*, frame: pl.DataFrame, factors: dict[str, float], groups: pl.DataFrame) -> pl.DataFrame:
    """Scale each row's quantiles about its median by the factor of its series' substation type."""
    keyed = frame.join(groups, on="time_series_id", how="left", maintain_order="left")
    k = keyed["group"].replace_strict(factors, return_dtype=pl.Float64).to_numpy()[:, None]
    values = keyed.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    scaled = values[:, [4]] + k * (values - values[:, [4]])
    return frame.drop(*common.LEVEL_COLUMNS).with_columns(
        pl.DataFrame(scaled.astype(np.float32), schema=list(common.LEVEL_COLUMNS), orient="row")
    )


def collect_arms(*, cases: pl.DataFrame, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return every arm's quantiles keyed by series, run, valid time, and seed, with `actual`."""
    arms: dict[str, pl.DataFrame] = {}
    for name, stem in NEW_ARMS.items():
        arms[name] = pl.concat(
            [pl.read_parquet(p) for p in sorted(ARMS_DIR.glob(f"{stem}_seed*.parquet"))]
        )
    groups = series.select("time_series_id", group=pl.col("substation_type").cast(pl.String))
    arms[SPREAD_ARM] = spread(frame=arms["ch_s"], factors=choose_k(series=series), groups=groups)
    for name in STORED_ARMS:
        arms[name] = experiment_store.read_predictions(
            experiment_id=STORED_ID, name=name, row_set_id=common.ROW_SET_ID
        )
    for name, set_name in REFERENCE_ARMS.items():
        arms[name] = experiment_store.read_predictions(
            experiment_id=5, name=set_name, row_set_id=common.ROW_SET_ID
        )
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
                covered=pl.Series(
                    ((actual >= quantiles[:, 0]) & (actual <= quantiles[:, 8])).astype(float)
                ),
            )
            .with_columns(
                pl.col("quantile_score", "median_error", "signed_error", "covered").cast(pl.Float64)
            )
            .drop(*common.LEVEL_COLUMNS, "actual")
        )
    return out


def save_predictions(*, arms: dict[str, pl.DataFrame]) -> None:
    """Write the new arms as immutable prediction sets, skipping sets that exist."""
    for name in (*NEW_ARMS, SPREAD_ARM):
        stored = (
            arms[name]
            .filter(pl.col("valid_time") < HOLDOUT)
            .select(*KEYS, "seed", *common.LEVEL_COLUMNS)
        )
        try:
            experiment_store.write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=name,
                frame=stored,
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
    """Return the pre-registered verdict from C1 and C3 on the quantile score over all leads."""

    def read(label: str, metric: str) -> dict:
        return contrasts.filter(
            pl.col("label").str.starts_with(label)
            & (pl.col("scope") == "all leads")
            & (pl.col("metric") == metric)
            & (pl.col("group") == HEADLINE_GROUP)
        ).to_dicts()[0]

    c1, c3 = read("C1", "quantile_score"), read("C3", "quantile_score")
    below = c1["upper"] < 0 and c3["upper"] < 0
    large = c1["difference"] <= -0.3 and c3["difference"] <= -0.3
    refuted = c1["lower"] > 0 or c3["lower"] > 0
    outcome = (
        "refuted"
        if refuted
        else "supported"
        if below and large
        else "small gain"
        if below
        else "inconclusive"
    )
    return "\n".join(
        [
            f"C1 all leads quantile score {_fmt(c1)}",
            f"C3 all leads quantile score {_fmt(c3)}",
            f"verdict: {outcome}",
        ]
    )


def main() -> None:
    """Score every arm and write the tables and the report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    cases = _actual_table(series=series)
    raw = collect_arms(cases=cases, series=series)
    for name, frame in raw.items():
        seeds = set(frame["seed"].unique().to_list())
        if seeds != set(common.SEEDS):
            msg = f"arm {name} holds seeds {sorted(seeds)}, expected {sorted(common.SEEDS)}"
            raise ValueError(msg)
    arms = prepare(arms=raw)
    rows = {name: frame.height // frame["seed"].n_unique() for name, frame in arms.items()}
    if len(set(rows.values())) != 1:
        msg = f"arms hold different numbers of shared cases per seed: {rows}"
        raise ValueError(msg)
    save_predictions(arms=raw)
    (RESULTS_DIR / "k.json").write_text(json.dumps(choose_k(series=series)))
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
                coverage=pl.col("covered").mean(),
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
