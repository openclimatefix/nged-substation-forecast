"""Score experiment 00027's arms, save their predictions, and write the report.

Every arm is scored on the validation cases that all arms hold with an observed power before
2026-07-01. The quantile score is twice the mass-weighted mean pinball loss of the 9 levels, and
the median error is its absolute error, both in points of capacity. Intervals resample whole
validation months and the three seeds. The three planned contrasts (A1, A2, A3 against A0) are read
at 98.33%; everything else is exploratory at 95%.

Output: `work/results/report.md` and the prediction sets under
`data/ml-experiments/00027-.../predictions/`.

    uv run python ml-experiments/00027-pvlive-lags-in-stage2-gsp-demand/e00027_score.py
"""

from pathlib import Path
from typing import Final

import e00027_common as common
import e00027_fit as fit
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
PLANNED_LEVEL: Final[float] = 100 - 5 / 3
PLANNED: Final[dict[str, str]] = {
    "C1 PV_Live level": "a1",
    "C2 PV_Live share": "a2",
    "C3 PV_Live-adjusted delta": "a3",
}
SCOPES: Final[dict[str, tuple[float, float]]] = {
    "all leads": (0.0, 1e9),
    "up to 48 h": (0.0, 48.0),
    "0-24 h": (0.0, 24.0),
    "1-2 d": (24.0, 48.0),
    "3-6 d": (48.0, 144.0),
    "7-15 d": (144.0, 1e9),
}
STORED: Final[Path] = common.WORK_DIR_00005 / "arms"


def load_arms() -> dict[str, pl.DataFrame]:
    """Return every arm's quantiles and actual power for the two GSPs, on the shared cases."""
    ids = common.series_table()["time_series_id"].to_list()
    arms = {a: pl.read_parquet(fit.OUT_DIR / f"{a}.parquet") for a in fit.ARMS}
    arms["l3_stored"] = pl.read_parquet(STORED / "l3_from2023.parquet").filter(
        pl.col("time_series_id").is_in(ids)
    )
    arms["climatology"] = pl.read_parquet(STORED / "r2.parquet").filter(
        pl.col("time_series_id").is_in(ids)
    )
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
    out = {}
    for name, frame in arms.items():
        kept = frame.join(shared, on=KEYS, how="semi").sort(*KEYS, "seed")
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
    return out


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
    level: float,
    series_id: int | None = None,
) -> tuple[float, float, float]:
    """Return the mean paired quantile-score difference and its interval bounds."""
    losses = experiment_store.losses_for_bootstrap(
        arm_losses={
            treatment: scope(frame=arms[treatment], name=name, series_id=series_id).select(
                *KEYS, "seed", "quantile_score"
            ),
            reference: scope(frame=arms[reference], name=name, series_id=series_id).select(
                *KEYS, "seed", "quantile_score"
            ),
        }
    )
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric="quantile_score",
        level=level,
        n_resamples=10_000 if level > 98 else 2_000,
    )
    differences, _ = bootstrap.paired_differences(
        losses=losses, treatment=treatment, reference=reference, metric="quantile_score"
    )
    return float(differences.mean()), lower, upper


def verdict(*, rows: list[tuple[str, float, float, float]]) -> str:
    """Apply the pre-registered rule to the planned contrasts at all leads."""
    label, estimate, lower, upper = min(rows, key=lambda r: r[1])
    if upper < 0 and estimate <= -0.30:
        word = "supported"
    elif lower > -0.30 and not upper < 0:
        word = "refuted"
    elif upper < 0:
        word = "small gain"
    else:
        word = "inconclusive"
    return f"{word}: {label} {estimate:+.3f} ({lower:+.3f} to {upper:+.3f} at 98.33%)"


def availability(*, series: pl.DataFrame) -> list[str]:
    """Return the share of validation cases with a usable PV_Live value per position and band."""
    lines = [
        "| series | lead band | " + " | ".join(common.PV_POSITIONS) + " |",
        "|---|---|---|---|---|---|---|---|",
    ]
    stats = fit._stats(split="validation", series_ids=series["time_series_id"].to_list()).filter(
        pl.col("seed") == 0
    )
    for site, series_id in zip(series["site"], series["time_series_id"], strict=True):
        table = stats.filter(pl.col("time_series_id") == series_id)
        target = np.asarray(common.grid_index(times=table["valid_time"]))
        origin = np.asarray(common.grid_index(times=table["power_fcst_init_time"]))
        pv = common.load_pv_live(series=series)[site]["level"]
        positions = common.pv_positions(target=target, origin=origin)
        usable = (positions >= 0) & np.isfinite(pv[np.where(positions >= 0, positions, 0)])
        lead = (target - origin) * 0.5
        for band in ("0-24 h", "1-2 d", "3-6 d", "7-15 d"):
            low, high = SCOPES[band]
            mask = (lead >= low) & (lead < high)
            shares = usable[mask].mean(axis=0)
            lines.append(f"| {site} | {band} | " + " | ".join(f"{s:.2f}" for s in shares) + " |")
    return lines


def sign_check(*, series: pl.DataFrame) -> list[str]:
    """Return, per series, the correlation of PV_Live with the delta and the adjusted delta.

    The delta is the observed power minus the final stage 1a prediction at ERA5 weather on summer
    middays (June to August, 10:00 to 14:00 UTC) of the validation year; seed 0.
    """
    power = common.load_power(series=series)
    pv = common.load_pv_live(series=series)
    era5 = common.read_era5_pred(seed=0, sites=common.all_sites())
    times = common.grid_times()
    mask = (
        (times >= common.VALIDATION_START).to_numpy()
        & (times < common.HOLDOUT).to_numpy()
        & times.dt.month().is_in([6, 7, 8]).to_numpy()
        & times.dt.hour().is_between(10, 14).to_numpy()
    )
    lines = ["| series | corr(delta, PV_Live) | corr(delta + PV_Live, PV_Live) |", "|---|---|---|"]
    for site in series["site"]:
        delta = power[site] - era5[site]
        level = pv[site]["level"]
        ok = mask & np.isfinite(delta) & np.isfinite(level)
        adjusted = delta + level
        lines.append(
            f"| {site} | {np.corrcoef(delta[ok], level[ok])[0, 1]:+.2f} "
            f"| {np.corrcoef(adjusted[ok], level[ok])[0, 1]:+.2f} |"
        )
    return lines


def stage1_summary() -> list[str]:
    """Return the S1 table: stage 1a's point error with and without PV_Live at the valid time."""
    errors = pl.read_parquet(common.WORK_DIR / "s1" / "errors.parquet").with_columns(
        absolute=pl.col("error").abs() * 100.0,
        daytime=pl.col("pv_live") > 0.01,
        month=pl.col("valid_time").dt.strftime("%Y-%m"),
    )
    lines = [
        "| series | rows | without | with PV_Live | difference | 95% interval |",
        "|---|---|---|---|---|---|",
    ]
    summary = []
    for site in sorted(errors["site"].unique()):
        for label, part in (
            ("all", errors.filter(pl.col("site") == site)),
            ("daytime", errors.filter((pl.col("site") == site) & pl.col("daytime"))),
        ):
            wide = part.pivot(
                on="arm", index=["valid_time", "month", "seed"], values="absolute"
            ).drop_nulls()
            wide = wide.group_by("valid_time", "month").agg(
                pl.col("without", "with_pv_live").mean()
            )
            difference = (wide["with_pv_live"] - wide["without"]).to_numpy()
            interval = bootstrap.bootstrap_row_difference(
                values=difference, months=wide["month"].to_numpy()
            )
            summary.append(
                {
                    "series": site,
                    "scope": label,
                    "difference": interval["difference"],
                    "lower": interval["lower_95"],
                    "upper": interval["upper_95"],
                }
            )
            lines.append(
                f"| {site} {label} | {wide.height} | {wide['without'].mean():.2f} | "
                f"{wide['with_pv_live'].mean():.2f} | {interval['difference']:+.3f} | "
                f"{interval['lower_95']:+.3f} to {interval['upper_95']:+.3f} |"
            )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(summary).write_parquet(RESULTS_DIR / "s1.parquet")
    return lines


def main() -> None:  # noqa: PLR0912
    """Score every arm and write the report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    ids = dict(zip(series["site"], series["time_series_id"], strict=True))
    arms = load_arms()
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
    planned = []
    records: list[dict[str, object]] = []
    lines = ["# Experiment 00027: scores", ""]
    lines += ["## Mean quantile score and median error (points of capacity)", ""]
    lines += ["| arm | series | scope | quantile score | median error |", "|---|---|---|---|---|"]
    for name, frame in arms.items():
        for site, series_id in [("both", None), *ids.items()]:
            for scope_name in ("all leads", "up to 48 h"):
                part = scope(frame=frame, name=scope_name, series_id=series_id)
                lines.append(
                    f"| {name} | {site} | {scope_name} | {part['quantile_score'].mean():.2f} | "
                    f"{part['median_error'].mean():.2f} |"
                )
    lines += ["", "## Planned contrasts, both GSPs (98.33%; negative favours the PV_Live arm)", ""]
    lines += ["| contrast | scope | difference | interval |", "|---|---|---|---|"]
    for label, arm in PLANNED.items():
        for scope_name in SCOPES:
            estimate, lower, upper = contrast(
                arms=arms, treatment=arm, reference="a0", name=scope_name, level=PLANNED_LEVEL
            )
            records.append(
                {
                    "label": label,
                    "scope": scope_name,
                    "difference": estimate,
                    "lower": lower,
                    "upper": upper,
                    "planned": True,
                }
            )
            if scope_name == "all leads":
                planned.append((label, estimate, lower, upper))
            lines.append(
                f"| {label} | {scope_name} | {estimate:+.3f} | {lower:+.3f} to {upper:+.3f} |"
            )
    lines += ["", f"Verdict by the pre-registered rule: {verdict(rows=planned)}", ""]
    lines += [
        "## Exploratory contrasts (95%)",
        "",
        "| contrast | scope | difference | interval |",
        "|---|---|---|---|",
    ]
    exploratory = [
        ("a3 minus a1", "a3", "a1", None),
        ("a3m (post hoc) minus a0", "a3m", "a0", None),
        ("a3m (post hoc) minus a3", "a3m", "a3", None),
        ("a0 minus stored l3 (reproduction)", "a0", "l3_stored", None),
        ("a0 minus climatology", "a0", "climatology", None),
        *[
            (f"{label} {site} minus a0", arm, "a0", sid)
            for label, arm in PLANNED.items()
            for site, sid in ids.items()
        ],
    ]
    for label, treatment, reference, series_id in exploratory:
        for scope_name in ("all leads", "up to 48 h"):
            estimate, lower, upper = contrast(
                arms=arms,
                treatment=treatment,
                reference=reference,
                name=scope_name,
                level=95.0,
                series_id=series_id,
            )
            records.append(
                {
                    "label": f"{label}",
                    "scope": scope_name,
                    "difference": estimate,
                    "lower": lower,
                    "upper": upper,
                    "planned": False,
                }
            )
            lines.append(
                f"| {label} | {scope_name} | {estimate:+.3f} | {lower:+.3f} to {upper:+.3f} |"
            )
    pl.DataFrame(records).write_parquet(RESULTS_DIR / "contrasts.parquet")
    lines += ["", "## Hour and season split of a3 minus a0 (mean difference, both GSPs)", ""]
    lines += ["| season | hours (UTC) | difference | rows |", "|---|---|---|---|"]
    joined = (
        arms["a3"]
        .join(arms["a0"].select(*KEYS, "seed", base="quantile_score"), on=[*KEYS, "seed"])
        .with_columns(
            difference=pl.col("quantile_score") - pl.col("base"),
            month=pl.col("valid_time").dt.month(),
            hour=pl.col("valid_time").dt.hour(),
        )
    )
    for season, months in (
        ("Jun-Aug", [6, 7, 8]),
        ("Dec-Feb", [12, 1, 2]),
        ("Mar-May, Sep-Nov", [3, 4, 5, 9, 10, 11]),
    ):
        for label in ("night 20-04", "midday 10-14", "other"):
            hours = pl.col("hour")
            if label == "midday 10-14":
                mask = hours.is_between(10, 14)
            elif label == "night 20-04":
                mask = (hours >= 20) | (hours < 5)
            else:
                mask = ~(hours.is_between(10, 14) | (hours >= 20) | (hours < 5))
            part = joined.filter(pl.col("month").is_in(months) & mask)
            lines.append(
                f"| {season} | {label} | {part['difference'].mean():+.3f} | {part.height} |"
            )
    lines += ["", "## Share of validation cases with a usable PV_Live value", ""]
    lines += availability(series=series)
    lines += ["", "## Sign check on summer middays", "", *sign_check(series=series)]
    lines += ["", "## S1: stage 1a point error (mean absolute error, points of capacity)", ""]
    lines += stage1_summary()
    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
