"""Score the capacity arms of experiment 139 and print the planned contrasts.

Rows: the arms' forecast frames from `e00139_run.py`, leads 0.5 to 168 hours, outside the
hand-filtered switching events, in the named period. The replicate period is the even months of
2024-07 to 2025-06 and the scored period is 2025-07 to 2026-06 (experiment 111's periods).

Slices: daylight half-hours (CAMS-driven unit PV shape above 0.01) are split per Primary into
thirds by that shape: the sunniest third and the darkest daylight third. Night half-hours are in
no third. Contrasts use the month-block paired bootstrap at the 98.75% level.

    uv run --no-sync python ml-experiments/00139-*/e00139_score.py --period replicate
    uv run --no-sync python ml-experiments/00139-*/e00139_score.py --period scored
"""

import argparse
from typing import Final

import e00139_common as common
import e00139_run as run_module
import e00139_run_common as run_common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.experiment_store import losses_for_bootstrap

LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
LEVEL: Final[float] = 98.75
N_RESAMPLES: Final[int] = 20000
ARMS: Final[tuple[str, ...]] = ("pin", "asof", "hindsight", "nopv", "asof_scaled", "asof_raw")
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("asof", "pin"),
    ("hindsight", "pin"),
    ("hindsight", "asof"),
    ("nopv", "pin"),
    ("asof_scaled", "pin"),
    ("asof_scaled", "asof"),
    ("asof_raw", "pin"),
    ("asof_raw", "asof"),
)
SEASONS: Final[dict[str, tuple[int, ...]]] = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}


def load_arm(*, arm: str, period: str, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return an arm's scorable rows for one period, with the slice labels."""
    cache = common.load_cache()
    pieces = []
    for site in sites:
        path = run_module.out_path(arm=arm, period=period, site=site)
        if not path.exists():
            continue
        frame = pl.read_parquet(path).with_columns(label=pl.lit(site))
        frame = frame.filter(
            ~pl.col("switching_mask")
            & pl.col("actual").is_finite()
            & (pl.col("lead_hours") <= 168.0)
        )
        valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
        rows = np.searchsorted(cache.end_times, valid)
        pieces.append(
            frame.with_columns(unit=pl.Series(np.nan_to_num(cache.sites[site].pv_unit_shape[rows])))
        )
    return pl.concat(pieces)


def add_slices(*, frame: pl.DataFrame, edges: dict[str, tuple[float, float]]) -> pl.DataFrame:
    """Label each row `dark`, `middle`, `sunny`, or `night` from its Primary's tertile edges."""
    low = pl.col("label").replace_strict(
        {site: edge[0] for site, edge in edges.items()}, return_dtype=pl.Float64
    )
    high = pl.col("label").replace_strict(
        {site: edge[1] for site, edge in edges.items()}, return_dtype=pl.Float64
    )
    return frame.with_columns(
        third=pl.when(pl.col("unit") <= common.SUN_UP_UNIT_SHAPE)
        .then(pl.lit("night"))
        .when(pl.col("unit") <= low)
        .then(pl.lit("dark"))
        .when(pl.col("unit") >= high)
        .then(pl.lit("sunny"))
        .otherwise(pl.lit("middle")),
        calendar_month=pl.col("valid_time").dt.month(),
    )


def tertile_edges(*, frame: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Return each Primary's lower and upper tertile of the daylight unit shape."""
    edges = {}
    for site in frame["label"].unique():
        unit = frame.filter(pl.col("label") == site, pl.col("unit") > common.SUN_UP_UNIT_SHAPE)[
            "unit"
        ]
        edges[site] = (
            float(np.quantile(unit.to_numpy(), 1 / 3)),
            float(np.quantile(unit.to_numpy(), 2 / 3)),
        )
    return edges


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the quantile score (twice the mean pinball loss) and the mean forecast's errors."""
    actual = frame["actual"].to_numpy().astype(np.float64)[:, None]
    quantiles = frame.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)
    error = actual - quantiles
    score = 2.0 * np.mean(
        np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error), axis=1
    )
    return frame.with_columns(
        qs=pl.Series(score),
        abs_mean=(pl.col("actual") - pl.col("mean")).abs(),
        signed_mean=pl.col("actual") - pl.col("mean"),
    )


def contrast(*, losses: pl.DataFrame, treatment: str, reference: str, metric: str = "qs") -> dict:
    """Return a paired contrast: estimate (treatment minus reference) and its 98.75% interval."""
    t = losses.filter(pl.col("arm") == treatment)[metric].to_numpy().mean()
    r = losses.filter(pl.col("arm") == reference)[metric].to_numpy().mean()
    low, high = bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric=metric,
        level=LEVEL,
        n_resamples=N_RESAMPLES,
    )
    return {"estimate": float(t - r), "lower": float(low), "upper": float(high)}


def main() -> None:
    """Print the absolute scores, the planned contrasts, and the checks for one period."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", choices=["replicate", "scored"], required=True)
    arguments = parser.parse_args()
    period = arguments.period
    five = common.DECISION_SITES
    frames = {arm: load_arm(arm=arm, period=period, sites=run_common.SCORED_SITES) for arm in ARMS}
    edges = tertile_edges(frame=frames["pin"])
    frames = {arm: add_slices(frame=with_losses(frame=f), edges=edges) for arm, f in frames.items()}
    stacked = losses_for_bootstrap(
        arm_losses={
            arm: f.filter(pl.col("label").is_in(list(five))).drop("arm")
            for arm, f in frames.items()
        }
    )
    print(f"== {period}: rows per arm at the five Primaries ==")
    print(stacked.group_by("arm").len())
    print("== absolute quantile score (points of effective capacity) ==")
    with pl.Config(float_precision=3, tbl_rows=60):
        print(
            stacked.group_by("arm").agg(qs=pl.col("qs").mean(), abs_mean=pl.col("abs_mean").mean())
        )
        print(
            stacked.group_by("arm", "third")
            .agg(qs=pl.col("qs").mean(), n=pl.len())
            .sort("third", "arm")
        )
    scopes: dict[str, pl.Expr] = {
        "all": pl.lit(True),
        "sunniest third": pl.col("third") == "sunny",
        "darkest daylight third": pl.col("third") == "dark",
        "middle third": pl.col("third") == "middle",
        "night": pl.col("third") == "night",
        **{name: pl.col("calendar_month").is_in(months) for name, months in SEASONS.items()},
        **{f"{site} only": pl.col("label") == site for site in five},
        "D11 sunniest third": (pl.col("label") == "D11") & (pl.col("third") == "sunny"),
    }
    rows = []
    for scope, expression in scopes.items():
        part = stacked.filter(expression)
        if part.is_empty():
            continue
        for treatment, reference in CONTRASTS:
            for metric in ("qs", "abs_mean"):
                result = contrast(
                    losses=part, treatment=treatment, reference=reference, metric=metric
                )
                rows.append(
                    {
                        "scope": scope,
                        "contrast": f"{treatment} - {reference}",
                        "metric": metric,
                        **result,
                    }
                )
    table = pl.DataFrame(rows)
    common.write_parquet_atomic(frame=table, path=common.WORK_DIR / f"contrasts_{period}.parquet")
    with pl.Config(float_precision=3, tbl_rows=200, tbl_cols=8, fmt_str_lengths=40):
        print(table.filter(pl.col("metric") == "qs"))
        print(
            table.filter(
                (pl.col("metric") == "abs_mean")
                & pl.col("scope").is_in(["all", "sunniest third", "darkest daylight third"])
            )
        )
    # Leave one Primary out for the planned contrast.
    print("== leave one Primary out, asof - pin, quantile score ==")
    for site in five:
        part = stacked.filter(pl.col("label") != site)
        for scope in ("sunniest third", "darkest daylight third"):
            result = contrast(losses=part.filter(scopes[scope]), treatment="asof", reference="pin")
            print(site, scope, {k: round(v, 4) for k, v in result.items()})
    # One Primary alone, asof - pin.
    print("== one Primary alone, asof - pin, quantile score ==")
    for site in five:
        part = stacked.filter(pl.col("label") == site)
        for scope in ("sunniest third", "darkest daylight third"):
            result = contrast(losses=part.filter(scopes[scope]), treatment="asof", reference="pin")
            print(site, scope, {k: round(v, 4) for k, v in result.items()})
    # D9: pin and no-PV only.
    d9 = {arm: frames[arm].filter(pl.col("label") == "D9").drop("arm") for arm in ("pin", "nopv")}
    both = losses_for_bootstrap(arm_losses=d9)
    print("D9 qs", both.group_by("arm").agg(pl.col("qs").mean()))
    print("D9 nopv - pin", contrast(losses=both, treatment="nopv", reference="pin"))
    # Tail calibration (coverage and pinball loss) per arm at the five Primaries.
    print("== tail calibration, five Primaries ==")
    for arm, f in frames.items():
        part = f.filter(pl.col("label").is_in(list(five)))
        line = {}
        for level, column in ((0.01, "q01"), (0.05, "q05"), (0.95, "q95"), (0.99, "q99")):
            line[f"cover<{column}"] = float(
                (part["actual"].to_numpy() < part[column].to_numpy()).mean()
            )
            err = part["actual"] - part[column]
            line[f"pinball{level}"] = float(
                np.mean(np.maximum(level * err.to_numpy(), (level - 1.0) * err.to_numpy()))
            )
        print(arm, {k: round(v, 4) for k, v in line.items()})
    # Median minus mean forecast per Primary (check).
    print("== median minus mean forecast, per Primary and arm (points) ==")
    print(
        pl.concat([f.with_columns(a=pl.lit(a)) for a, f in frames.items()])
        .group_by("a", "label")
        .agg(median_minus_mean=(pl.col("q50") - pl.col("mean")).mean())
        .sort("a", "label")
    )


if __name__ == "__main__":
    main()
