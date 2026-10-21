"""Score experiment 133's arms and print every number the write-up quotes.

Reads `work/dp_preds/`, puts every arm on common rows (scored runs or replicate runs, a reading,
outside the hand-filtered switching events, lead of at most 168 hours), and writes
`work/report/report.json` and a printed report with: each arm's quantile score at 12 to 36 hours,
at every lead band, and per substation; the planned contrasts C1 to C3b with 99% month-resampled
intervals, the replicate estimate, the tuning-month estimate, the sites better, and the months
negative; the breakouts; the tail table; the fitted values; the mechanism correlation; and the
checks (median minus mean forecast per site, and the error of the mean forecast beside each
contrast).

    uv run python ml-experiments/00133-*/e00133_report.py
"""

import json
from datetime import UTC, datetime
from typing import Final

import e00133_arms as arms
import e00133_common as common
import e00133_run as run
import e00133_select as select
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_calendar import BAND_NAMES
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import losses_for_bootstrap

LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24 h", 0.0, 24.0),
    ("24-48 h", 24.0, 48.0),
    ("48-144 h", 48.0, 144.0),
    ("144-168 h", 144.0, 168.0),
    ("12-36 h", 12.0, 36.0),
    ("all", 0.0, 168.0),
)
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1": ("grid", "fix"),
    "C2": ("pool", "grid"),
    "C3a": ("vk0", "grid"),
    "C3b": ("vk1", "grid"),
}
LEVEL_PERCENT: Final[float] = 99.0
N_RESAMPLES: Final[int] = 20_000
COLD_TEMPERATURE_C: Final[float] = 3.0
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
NIGHT_BANDS: Final[tuple[int, ...]] = (BAND_NAMES.index("night"), 3)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
TAIL_QUANTILES: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)


def load(*, arm: str, period: str) -> pl.DataFrame:
    """Return an arm's scorable rows for a period at the scored substations."""
    cache = common.load_cache()
    last = cache.end_times[-1].astype("datetime64[us]").astype(datetime).replace(tzinfo=UTC)
    pieces = []
    for site in common.SCORED_SITES:
        frame = pl.read_parquet(run.out_path(arm=arm, period=period, site=site)).with_columns(
            site=pl.lit(site)
        )
        frame = frame.filter(
            ~pl.col("switching_mask")
            & pl.col("actual").is_finite()
            & (pl.col("lead_hours") <= 168.0)
            & (pl.col("valid_time") <= last)
        )
        features = cache.sites[site]
        target = np.searchsorted(
            cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        frame = frame.with_columns(
            hour_band=pl.Series(cache.calendar.band[target].astype(np.int8)),
            holiday=pl.Series(cache.calendar.holiday_period[target] >= 0),
            cold=pl.Series(
                np.nan_to_num(features.temperature_c[target], nan=99.0) <= COLD_TEMPERATURE_C
            ),
            valid_month=pl.col("valid_time").dt.month(),
        )
        pieces.append(frame)
    return pl.concat(pieces, how="diagonal")


def common_rows(*, frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Restrict every frame to the rows all frames hold."""
    shared = None
    for frame in frames.values():
        current = frame.select(KEYS).unique()
        shared = current if shared is None else shared.join(current, on=KEYS, how="inner")
    assert shared is not None
    return {
        name: frame.join(shared, on=KEYS, how="inner").sort(KEYS) for name, frame in frames.items()
    }


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the quantile score, the mean and median absolute errors, and the coverage flags."""
    actual = frame["actual"].to_numpy().astype(np.float64)[:, None]
    quantiles = frame.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64)
    error = actual - quantiles
    score = 2.0 * np.mean(
        np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error), axis=1
    )
    extra = {
        "qs": score,
        "abs_mean": np.abs(frame["actual"].to_numpy() - frame["mean"].to_numpy()).astype(
            np.float64
        ),
        "abs_median": np.abs(frame["actual"].to_numpy() - frame["q50"].to_numpy()).astype(
            np.float64
        ),
    }
    for name, level in TAIL_QUANTILES:
        value = frame[name].to_numpy().astype(np.float64)
        loss = np.maximum(
            level * (frame["actual"].to_numpy() - value),
            (level - 1.0) * (frame["actual"].to_numpy() - value),
        )
        extra[f"pin_{name}"] = loss
        extra[f"cov_{name}"] = (frame["actual"].to_numpy() <= value).astype(np.float64)
    return frame.with_columns([pl.Series(k, v) for k, v in extra.items()])


def lead_filter(*, frame: pl.DataFrame, low: float, high: float) -> pl.DataFrame:
    """Keep rows with a lead above `low` and at most `high` hours."""
    return frame.filter((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))


def contrast(*, losses: pl.DataFrame, treatment: str, reference: str, metric: str = "qs") -> dict:
    """Return a paired contrast with a 99% month-resampled interval and its site and month signs."""
    t = losses.filter(pl.col("arm") == treatment)
    r = losses.filter(pl.col("arm") == reference)
    estimate = float(t[metric].to_numpy().mean() - r[metric].to_numpy().mean())
    lower, upper = bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric=metric,
        level=LEVEL_PERCENT,
        n_resamples=N_RESAMPLES,
    )
    by_site = (
        t.group_by("site")
        .agg(t=pl.col(metric).mean())
        .join(r.group_by("site").agg(r=pl.col(metric).mean()), on="site")
    )
    by_month = (
        t.group_by("month")
        .agg(t=pl.col(metric).mean())
        .join(r.group_by("month").agg(r=pl.col(metric).mean()), on="month")
    )
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "sites_better": int((by_site["t"] < by_site["r"]).sum()),
        "n_sites": by_site.height,
        "months_negative": int((by_month["t"] < by_month["r"]).sum()),
        "n_months": by_month.height,
    }


def stack(*, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return the bootstrap loss frame of arms already on common rows, with D-labels as `site`."""
    cache = common.load_cache()
    labels = {cache.sites[s].table.time_series_id: s for s in common.SCORED_SITES}
    stacked = losses_for_bootstrap(
        arm_losses={
            name: with_losses(frame=f).drop(
                "site", "arm", "tail_factor", "tail_statistic", strict=False
            )
            for name, f in frames.items()
        }
    )
    return stacked.with_columns(site=pl.col("site").replace_strict(labels, return_dtype=pl.String))


def subset_filters() -> dict[str, pl.Expr]:
    """Return the breakout subsets, each as a row filter."""
    night = pl.col("hour_band").is_in(list(NIGHT_BANDS))
    winter = pl.col("valid_month").is_in(list(WINTER_MONTHS))
    return {
        "winter": winter,
        "winter nights": winter & night,
        "cold nights": pl.col("cold") & night,
        "holidays and Christmas": pl.col("holiday"),
    }


def arm_table(*, losses: pl.DataFrame, sites: tuple[str, ...]) -> dict:
    """Return each arm's mean losses by lead band, by site at 12 to 36 hours, and by breakout."""
    out: dict = {"by_lead": {}, "by_site": {}, "breakouts": {}}
    scope = losses.filter(pl.col("site").is_in(list(sites)))
    for name, low, high in LEAD_BANDS:
        part = lead_filter(frame=scope, low=low, high=high)
        out["by_lead"][name] = part.group_by("arm").agg(pl.col("qs").mean()).sort("arm").to_dicts()
    day_ahead = lead_filter(frame=scope, low=12.0, high=36.0)
    out["by_site"] = (
        day_ahead.group_by(["arm", "site"])
        .agg(pl.col("qs").mean())
        .sort(["arm", "site"])
        .to_dicts()
    )
    for label, expression in subset_filters().items():
        part = day_ahead.filter(expression)
        out["breakouts"][label] = {
            "rows": part.filter(pl.col("arm") == part["arm"][0]).height,
            "scores": part.group_by("arm").agg(pl.col("qs").mean()).sort("arm").to_dicts(),
        }
    return out


def tail_table(*, losses: pl.DataFrame, sites: tuple[str, ...]) -> dict:
    """Return coverage and pinball loss at the four tail quantiles, and cold-night q95 coverage."""
    scope = losses.filter(pl.col("site").is_in(list(sites)))
    out: dict = {}
    for label, part in (
        ("12-36 h", lead_filter(frame=scope, low=12.0, high=36.0)),
        ("all leads", scope),
        (
            "cold nights 12-36 h",
            lead_filter(frame=scope, low=12.0, high=36.0).filter(subset_filters()["cold nights"]),
        ),
    ):
        aggregations = [pl.col(f"cov_{n}").mean() for n, _ in TAIL_QUANTILES]
        aggregations += [pl.col(f"pin_{n}").mean() for n, _ in TAIL_QUANTILES]
        out[label] = part.group_by("arm").agg(aggregations).sort("arm").to_dicts()
    return out


def checks(*, frames: dict[str, pl.DataFrame]) -> dict:
    """Return the median minus mean forecast per site and arm, at 12 to 36 hours."""
    out = []
    for name, frame in frames.items():
        part = lead_filter(frame=frame, low=12.0, high=36.0)
        grouped = part.group_by("site").agg(
            (pl.col("q50") - pl.col("mean")).mean().alias("median_minus_mean")
        )
        out.extend({"arm": name, **row} for row in grouped.sort("site").to_dicts())
    return {"median_minus_mean": out}


def fitted_values() -> list[dict]:
    """Return each arm's setting per substation in ratio units next to FIX's."""
    rows = []
    for site in common.SCORED_SITES:
        for arm in ("fix", "grid", "pool"):
            settings = arms.settings_of_hyper(hyper=run.arm_hyper(arm=arm, site=site))
            rows.append(
                {
                    "site": site,
                    "arm": arm,
                    "q_over_sigma_eps_squared": settings.q / settings.sigma_eps**2,
                    "sigma_z_over_sigma_eps": settings.sigma_z / settings.sigma_eps,
                    "phi": settings.phi,
                }
            )
    return rows


def mechanism(*, frames: dict[str, pl.DataFrame]) -> list[dict]:
    """Return the correlation of the tail statistic at the cutoff with the later absolute error."""
    out = []
    for arm in ("vk1", "trail"):
        part = lead_filter(frame=frames[arm], low=12.0, high=36.0).filter(
            pl.col("tail_statistic").is_finite()
        )
        for label, sub in (
            ("all", part),
            ("cold nights", part.filter(subset_filters()["cold nights"])),
        ):
            value = np.corrcoef(
                np.log(sub["tail_statistic"].to_numpy().astype(np.float64) + 1e-9),
                np.log(np.abs((sub["actual"] - sub["mean"]).to_numpy().astype(np.float64)) + 1e-3),
            )[0, 1]
            out.append({"arm": arm, "rows": label, "log_correlation": float(value)})
    return out


POST_HOC_CONTRASTS: Final[dict[str, tuple[str, str]]] = {"PH1": ("pool", "fix")}
"""Contrasts added after the scored year was read; they carry no verdict."""


def leave_one_out(*, losses: pl.DataFrame, treatment: str, reference: str) -> dict[str, float]:
    """Return the 12-to-36-hour quantile-score contrast with each decision site left out in turn."""
    scope = lead_filter(
        frame=losses.filter(pl.col("site").is_in(list(common.DECISION_SITES))), low=12.0, high=36.0
    )
    out = {}
    for site in common.DECISION_SITES:
        part = scope.filter(pl.col("site") != site)
        t = part.filter(pl.col("arm") == treatment)["qs"].to_numpy().mean()
        r = part.filter(pl.col("arm") == reference)["qs"].to_numpy().mean()
        out[site] = float(t - r)
    return out


def site_gains(*, losses: pl.DataFrame, treatment: str, reference: str) -> dict[str, float]:
    """Return each decision site's 12-to-36-hour contrast, treatment minus reference."""
    scope = lead_filter(
        frame=losses.filter(pl.col("site").is_in(list(common.DECISION_SITES))), low=12.0, high=36.0
    )
    out = {}
    for site in common.DECISION_SITES:
        part = scope.filter(pl.col("site") == site)
        out[site] = float(
            part.filter(pl.col("arm") == treatment)["qs"].to_numpy().mean()
            - part.filter(pl.col("arm") == reference)["qs"].to_numpy().mean()
        )
    return out


def period_report(
    *, period: str, arm_names: tuple[str, ...]
) -> tuple[dict, dict[str, pl.DataFrame]]:
    """Return the period's tables and contrasts, and the common-row frames."""
    frames = common_rows(frames={arm: load(arm=arm, period=period) for arm in arm_names})
    losses = stack(frames=frames)
    sites = common.DECISION_SITES
    decision = losses.filter(pl.col("site").is_in(list(sites)))
    report: dict = {
        "rows": frames[arm_names[0]].height,
        "tables": arm_table(losses=losses, sites=sites),
    }
    report["tail"] = tail_table(losses=losses, sites=sites)
    report["contrasts"] = {}
    report["leave_one_out"] = {
        label: leave_one_out(losses=losses, treatment=t, reference=r)
        for label, (t, r) in {**CONTRASTS, **POST_HOC_CONTRASTS}.items()
    }
    report["site_gains"] = {
        label: site_gains(losses=losses, treatment=t, reference=r)
        for label, (t, r) in {**CONTRASTS, **POST_HOC_CONTRASTS}.items()
    }
    for label, (treatment, reference) in {**CONTRASTS, **POST_HOC_CONTRASTS}.items():
        report["contrasts"][label] = {}
        for scope_name, low, high in (("12-36 h", 12.0, 36.0), ("all", 0.0, 168.0)):
            part = lead_filter(frame=decision, low=low, high=high)
            entry = {"qs": contrast(losses=part, treatment=treatment, reference=reference)}
            entry["abs_mean"] = contrast(
                losses=part, treatment=treatment, reference=reference, metric="abs_mean"
            )
            report["contrasts"][label][scope_name] = entry
        cold = lead_filter(frame=decision, low=12.0, high=36.0).filter(
            subset_filters()["cold nights"]
        )
        report["contrasts"][label]["cold_night_q95_coverage"] = contrast(
            losses=cold, treatment=treatment, reference=reference, metric="cov_q95"
        )
    return report, frames


def main() -> None:
    """Score the replicate months and the scored year and write the report."""
    names = arms.ARM_NAMES
    report: dict = {"fitted_values": fitted_values()}
    for period in ("scored", "replicate"):
        period_data, frames = period_report(period=period, arm_names=names)
        report[period] = period_data
        if period == "scored":
            report["checks"] = checks(frames=frames)
            report["mechanism"] = mechanism(frames=frames)
    tuning_frames = common_rows(frames={arm: load(arm=arm, period="tuning") for arm in names})
    tuning = stack(frames=tuning_frames)
    report["tuning"] = {
        "contrasts": {
            label: contrast(
                losses=lead_filter(
                    frame=tuning.filter(pl.col("site").is_in(list(common.DECISION_SITES))),
                    low=12.0,
                    high=36.0,
                ),
                treatment=t,
                reference=r,
            )
            for label, (t, r) in {**CONTRASTS, **POST_HOC_CONTRASTS}.items()
        },
        "site_gains": {
            label: site_gains(losses=tuning, treatment=t, reference=r)
            for label, (t, r) in {**CONTRASTS, **POST_HOC_CONTRASTS}.items()
        },
    }
    for name in ("p1", "p2", "track"):
        path = common.WORK_DIR / "gate" / f"{name}.json"
        if path.exists():
            report[name] = json.loads(path.read_text())
    report["select"] = {
        site: json.loads((select.SELECT_DIR / f"step2_{site}.json").read_text())
        for site in common.SCORED_SITES
    }
    report["rho"] = json.loads((run.RUN_DIR / "rho.json").read_text())
    report["factors"] = json.loads((run.RUN_DIR / "factors.json").read_text())["factors"]
    target = common.WORK_DIR / "report" / "report.json"
    common.write_json_atomic(path=target, content=report)
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
