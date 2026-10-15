"""Score experiment 00028's arms and the solar forecast, save predictions, and write the report.

Every arm is scored on the validation cases that all arms hold, with an observed demand and a
PV_Live value at the target time, and a valid time before 2026-07-01. The quantile score is twice
the mass-weighted mean pinball loss of the 9 levels, in points of capacity. Intervals resample whole
validation months and the three seeds. C1 (B1 minus B0) is the deciding contrast, read at 95%.

Output: `work/results/{report.md, contrasts.parquet, solar.parquet}` and the prediction sets under
`data/ml-experiments/00028-.../predictions/`.

    uv run python ml-experiments/00028-pvlive-forecast-in-stage2-gsp-demand/e00028_score.py
"""

from pathlib import Path
from typing import Final

import e00028_common as common
import e00028_fit as fit
import e00028_pv as pv_forecast
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
SCOPES: Final[dict[str, tuple[float, float]]] = {
    "all leads": (0.0, 1e9),
    "up to 48 h": (0.0, 48.0),
    "0-24 h": (0.0, 24.0),
    "1-2 d": (24.0, 48.0),
    "3-6 d": (48.0, 144.0),
    "7-15 d": (144.0, 1e9),
}
CONTRASTS: Final[dict[str, tuple[str, str, bool]]] = {
    "C1 solar forecast minus none": ("b1", "b0", True),
    "C2 true PV_Live minus none": ("b2", "b0", False),
    "Solar forecast minus true PV_Live": ("b1", "b2", False),
}
"""Treatment, reference, and whether the contrast decides the verdict."""


def load_arms() -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Return every arm's quantiles and losses on the shared cases, and the case counts."""
    series = common.series_table()
    pv = common.load_pv_live(series=series)
    sites = dict(zip(series["time_series_id"], series["site"], strict=True))
    arms = {a: pl.read_parquet(fit.OUT_DIR / f"{a}.parquet") for a in fit.ARMS}
    shared: pl.DataFrame | None = None
    for frame in arms.values():
        cases = (
            frame.filter(
                pl.all_horizontal(pl.col("actual", *common.LEVEL_COLUMNS).is_finite())
                & (pl.col("valid_time") < common.HOLDOUT)
            )
            .select(KEYS)
            .unique()
        )
        shared = cases if shared is None else shared.join(cases, on=KEYS, how="inner")
    assert shared is not None
    has_share = pl.concat(
        [
            shared.filter(pl.col("time_series_id") == series_id).filter(
                pl.Series(
                    np.isfinite(
                        common.take(
                            values=pv[site]["share"],
                            index=np.asarray(
                                common.grid_index(
                                    times=shared.filter(pl.col("time_series_id") == series_id)[
                                        "valid_time"
                                    ]
                                )
                            ),
                        )
                    )
                )
            )
            for series_id, site in sites.items()
        ]
    )
    counts = has_share.group_by("time_series_id").len().sort("time_series_id")
    out = {}
    for name, frame in arms.items():
        kept = frame.join(has_share, on=KEYS, how="semi").sort(*KEYS, "seed")
        quantiles = np.sort(kept.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
        actual = kept["actual"].to_numpy()
        out[name] = kept.select(*KEYS, "seed", *common.LEVEL_COLUMNS, "actual").with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
            quantile_score=pl.Series(
                common.pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0
            ),
            median_error=pl.Series(np.abs(quantiles[:, 4] - actual) * 100.0),
        )
    return out, counts


def scope(*, frame: pl.DataFrame, name: str, series_id: int | None = None) -> pl.DataFrame:
    """Return the rows of one lead scope, optionally of one series."""
    low, high = SCOPES[name]
    out = frame.filter((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
    return out if series_id is None else out.filter(pl.col("time_series_id") == series_id)


def contrast(
    *,
    arms: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    name: str,
    series_id: int | None = None,
) -> tuple[float, float, float]:
    """Return the mean paired quantile-score difference and its 95% interval."""
    losses = experiment_store.losses_for_bootstrap(
        arm_losses={
            arm: scope(frame=arms[arm], name=name, series_id=series_id).select(
                *KEYS, "seed", "quantile_score"
            )
            for arm in (treatment, reference)
        }
    )
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric="quantile_score",
        level=95.0,
        n_resamples=10_000,
    )
    differences, _ = bootstrap.paired_differences(
        losses=losses, treatment=treatment, reference=reference, metric="quantile_score"
    )
    return float(differences.mean()), lower, upper


def verdict(*, estimate: float, lower: float, upper: float) -> str:
    """Apply the pre-registered rule to C1."""
    if upper < 0 and estimate <= -0.30:
        word = "supported"
    elif lower > -0.30 and not upper < 0:
        word = "refuted"
    elif upper < 0:
        word = "small gain"
    else:
        word = "inconclusive"
    return f"{word}: C1 {estimate:+.3f} ({lower:+.3f} to {upper:+.3f} at 95%)"


def solar_quality(*, series: pl.DataFrame) -> tuple[list[str], pl.DataFrame]:
    """Return the solar forecast's own score tables and a frame for the chart."""
    pv = common.load_pv_live(series=series)
    sites = dict(zip(series["time_series_id"], series["site"], strict=True))
    times = common.grid_times()
    train_end = int(common.grid_index(times=common.FINAL_TRAIN_CUTOFF))
    climatology = {}
    for site in sites.values():
        share = pv[site]["share"][:train_end]
        frame = (
            pl.DataFrame(
                {
                    "month": times.dt.month()[:train_end],
                    "slot": times.dt.hour()[:train_end] * 2 + times.dt.minute()[:train_end] // 30,
                    "share": share,
                }
            )
            .with_columns(pl.col("share").fill_nan(None))
            .drop_nulls()
        )
        climatology[site] = frame.group_by("month", "slot").agg(climatology=pl.col("share").mean())

    def with_actual(*, split: str) -> pl.DataFrame:
        stats = pl.concat(
            [pl.read_parquet(p) for p in sorted((pv_forecast.STATS_DIR / split).glob("*.parquet"))]
        ).filter(
            (pl.col("valid_time") < common.HOLDOUT)
            & ((pl.col("valid_time") < common.VALIDATION_START) == (split == "calibration"))
        )
        pieces = []
        for series_id, site in sites.items():
            part = stats.filter(pl.col("time_series_id") == series_id)
            index = np.asarray(common.grid_index(times=part["valid_time"]))
            pieces.append(
                part.with_columns(
                    actual=pl.Series(common.take(values=pv[site]["share"], index=index)),
                    month=part["valid_time"].dt.month(),
                    slot=part["valid_time"].dt.hour() * 2 + part["valid_time"].dt.minute() // 30,
                    site=pl.lit(site),
                ).join(climatology[site], on=["month", "slot"], how="left")
            )
        return (
            pl.concat(pieces)
            .filter(pl.col("actual").is_finite())
            .with_columns(
                lead_hours=(
                    pl.col("valid_time") - pl.col("power_fcst_init_time")
                ).dt.total_minutes()
                / 60.0
            )
        )

    validation = with_actual(split="validation")
    quantiles = np.sort(validation.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    validation = validation.with_columns(
        quantile_score=pl.Series(
            common.pinball_quantile_score(
                quantiles=quantiles, actual=validation["actual"].to_numpy()
            )
            * 100.0
        ),
        error=(pl.col("mean") - pl.col("actual")) * 100.0,
        climatology_error=(pl.col("climatology") - pl.col("actual")) * 100.0,
    ).with_columns(
        band=pl.when(pl.col("lead_hours") < 24)
        .then(pl.lit("0-24 h"))
        .when(pl.col("lead_hours") < 48)
        .then(pl.lit("1-2 d"))
        .when(pl.col("lead_hours") < 144)
        .then(pl.lit("3-6 d"))
        .otherwise(pl.lit("7-15 d"))
    )
    daytime = validation.filter(pl.col("actual") > 0.01)
    rows = (
        daytime.group_by("site", "band")
        .agg(
            n=pl.len(),
            mae=pl.col("error").abs().mean(),
            bias=pl.col("error").mean(),
            climatology_mae=pl.col("climatology_error").abs().mean(),
            quantile_score=pl.col("quantile_score").mean(),
        )
        .sort("site", "band")
    )
    calibration = with_actual(split="calibration").filter(pl.col("actual") > 0.01)
    calibration = calibration.with_columns(
        quarter=(
            pl.col("valid_time").dt.year() * 10 + (pl.col("valid_time").dt.month() - 1) // 3 + 1
        ),
        error=(pl.col("mean") - pl.col("actual")).abs() * 100.0,
    )
    by_quarter = (
        calibration.group_by("site", "quarter")
        .agg(mae=pl.col("error").mean())
        .sort("site", "quarter")
    )
    lines = [
        "| series | lead band | daytime cases | MAE | bias | climatology MAE | quantile score |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['site']} | {r['band']} | {r['n']} | {r['mae']:.2f} | {r['bias']:+.2f} | "
        f"{r['climatology_mae']:.2f} | {r['quantile_score']:.2f} |"
        for r in rows.iter_rows(named=True)
    ]
    lines += ["", "| series | quarter of valid time | calibration MAE |", "|---|---|---|"]
    lines += [
        f"| {r['site']} | {r['quarter']} | {r['mae']:.2f} |"
        for r in by_quarter.iter_rows(named=True)
    ]
    return lines, rows


def main() -> None:
    """Score every arm and write the report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    ids = dict(zip(series["site"], series["time_series_id"], strict=True))
    arms, counts = load_arms()
    for name in fit.ARMS:
        try:
            experiment_store.write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=name,
                frame=arms[name].select(*KEYS, "seed", *common.LEVEL_COLUMNS),
                kind="quantiles",
                row_set_id=common.ROW_SET_ID,
            )
        except FileExistsError:
            continue
    lines = ["# Experiment 00028: scores", "", "## Cases scored per series", ""]
    lines += [f"- series {r['time_series_id']}: {r['len']}" for r in counts.iter_rows(named=True)]
    lines += ["", "## Mean quantile score and median error (points of capacity)", ""]
    lines += ["| arm | series | scope | quantile score | median error |", "|---|---|---|---|---|"]
    for name, frame in arms.items():
        for site, series_id in [("both", None), *ids.items()]:
            for scope_name in ("all leads", "up to 48 h"):
                part = scope(frame=frame, name=scope_name, series_id=series_id)
                lines.append(
                    f"| {name} | {site} | {scope_name} | {part['quantile_score'].mean():.2f} | "
                    f"{part['median_error'].mean():.2f} |"
                )
    lines += ["", "## Contrasts, both GSPs (95%; negative favours the first arm)", ""]
    lines += ["| contrast | scope | difference | interval |", "|---|---|---|---|"]
    records: list[dict[str, object]] = []
    decided = ""
    for label, (treatment, reference, decides) in CONTRASTS.items():
        for scope_name in SCOPES:
            estimate, lower, upper = contrast(
                arms=arms, treatment=treatment, reference=reference, name=scope_name
            )
            records.append(
                {
                    "label": label,
                    "scope": scope_name,
                    "difference": estimate,
                    "lower": lower,
                    "upper": upper,
                }
            )
            lines.append(
                f"| {label} | {scope_name} | {estimate:+.3f} | {lower:+.3f} to {upper:+.3f} |"
            )
            if decides and scope_name == "all leads":
                decided = verdict(estimate=estimate, lower=lower, upper=upper)
    lines += ["", f"Verdict by the pre-registered rule: {decided}", ""]
    lines += [
        "## Each GSP alone (exploratory, 95%)",
        "",
        "| contrast | series | scope | difference | interval |",
    ]
    lines += ["|---|---|---|---|---|"]
    for label, (treatment, reference, _) in list(CONTRASTS.items())[:2]:
        for site, series_id in ids.items():
            for scope_name in ("all leads", "up to 48 h"):
                estimate, lower, upper = contrast(
                    arms=arms,
                    treatment=treatment,
                    reference=reference,
                    name=scope_name,
                    series_id=series_id,
                )
                lines.append(
                    f"| {label} | {site} | {scope_name} | {estimate:+.3f} | "
                    f"{lower:+.3f} to {upper:+.3f} |"
                )
    pl.DataFrame(records).write_parquet(RESULTS_DIR / "contrasts.parquet")
    solar_lines, solar = solar_quality(series=series)
    solar.write_parquet(RESULTS_DIR / "solar.parquet")
    lines += ["", "## The solar forecast's own score (points of PV_Live capacity, daytime)", ""]
    lines += solar_lines
    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
