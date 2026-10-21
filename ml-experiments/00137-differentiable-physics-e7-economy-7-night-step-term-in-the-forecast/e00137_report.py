"""Score experiment 137: contrasts, breakouts, tails, and checks. Prints the report and writes JSON.

OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00137-*/e00137_report.py
"""

import e00137_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.experiment_store import losses_for_bootstrap, read_predictions

ARMS = ("base", "t_site", "t_pool", "fixed123", "fixed123b", "t_const", "oracle", "placebo")
LEVEL_PERCENT = 99.0
N_RESAMPLES = 20_000
KEYS = ["time_series_id", "power_fcst_init_time", "valid_time", "seed"]
TAILS = (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99))
CONTRASTS = {
    "C1": ("t_site", "base", "headline"),
    "C2": ("t_pool", "base", "headline"),
    "C3": ("fixed123", "base", "headline"),
    "C3b": ("fixed123b", "base", "headline"),
    "Cconst": ("t_const", "base", "headline"),
    "C4": ("t_site", "base", "all"),
    "C5": ("t_site", "t_pool", "headline"),
    "oracle": ("oracle", "base", "headline"),
    "placebo": ("placebo", "base", "headline"),
}
MIN_MONTHS = 3
MIN_ROWS_PER_SITE = 50


def mean(series: pl.Series) -> float:
    """Return the mean of a polars series as a float."""
    return float(np.mean(series.to_numpy()))


def load() -> dict[str, pl.DataFrame]:
    """Return each arm's day-ahead-and-all-lead rows on the shared keys, with losses added."""
    arms = {}
    for name in ARMS:
        frame = read_predictions(
            experiment_id=common.EXPERIMENT_ID, name=name, row_set_id=common.ROW_SET_ID
        )
        arms[name] = frame
    return arms


def add_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the quantile score, mean and median absolute error, and tail pinball and coverage."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    q = frame.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    error = actual[:, None] - q
    score = 2.0 * np.mean(
        np.maximum(common.LEVELS[None, :] * error, (common.LEVELS[None, :] - 1.0) * error), axis=1
    )
    extra = {
        "qs": score,
        "abs_mean": np.abs(actual - frame["mean"].to_numpy()),
        "abs_median": np.abs(actual - frame["q50"].to_numpy()),
        "err_mean": actual - frame["mean"].to_numpy().astype(np.float64),
    }
    for name, level in TAILS:
        value = frame[name].to_numpy().astype(np.float64)
        extra[f"pin_{name}"] = np.maximum(
            level * (actual - value), (level - 1.0) * (actual - value)
        )
        extra[f"cov_{name}"] = (actual <= value).astype(np.float64)
    return frame.with_columns([pl.Series(k, v) for k, v in extra.items()])


def subsets() -> dict[str, pl.Expr]:
    """Return the row subsets: all rows, and the headline rows (window, cold, winter, 12-36 h)."""
    day_ahead = (pl.col("lead_hours") > 12.0) & (pl.col("lead_hours") <= 36.0)
    winter = pl.col("valid_time").dt.month().is_in(list(common.WINTER_MONTHS))
    night = pl.col("slot") >= 0
    cold = pl.col("temp_c") < common.COLD_DAY_C
    return {
        "all": pl.lit(True),
        "day_ahead": day_ahead,
        "winter_nights": day_ahead & winter & night,
        "window": day_ahead & pl.col("window"),
        "window_winter": day_ahead & pl.col("window") & winter,
        "headline": day_ahead & pl.col("window") & winter & cold,
        "cold_nights": day_ahead & winter & night & cold,
    }


def contrast(*, losses: pl.DataFrame, treatment: str, reference: str, metric: str = "qs") -> dict:
    """Return a paired contrast with a 99% month-resampled interval and its signs by site/month."""
    t = losses.filter(pl.col("arm") == treatment)
    r = losses.filter(pl.col("arm") == reference)
    estimate = mean(t[metric]) - mean(r[metric])
    n_months = int(t["month"].n_unique())
    rows_by_site = t.group_by("site").len().sort("site")
    readable = n_months >= MIN_MONTHS and bool((rows_by_site["len"] >= MIN_ROWS_PER_SITE).all())
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
        .sort("site")
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
        "rows": t.height,
        "months": n_months,
        "readable": readable,
        "rows_by_site": dict(zip(rows_by_site["site"], rows_by_site["len"], strict=True)),
        "site_difference": {
            s: float(a - b)
            for s, a, b in zip(by_site["site"], by_site["t"], by_site["r"], strict=True)
        },
        "sites_better": int((by_site["t"] < by_site["r"]).sum()),
        "months_negative": int((by_month["t"] < by_month["r"]).sum()),
    }


def main() -> None:
    """Print and store the report."""
    raw = load()
    mapping = raw["base"].select(["time_series_id", "site"]).unique()
    labels = dict(zip(mapping["time_series_id"], mapping["site"], strict=True))
    stacked = add_losses(frame=pl.concat([f.with_columns(arm=pl.lit(n)) for n, f in raw.items()]))
    meta = raw["base"].select([*KEYS, "site", "slot", "temp_c", "window", "lead_hours"])
    out: dict = {"contrasts": {}, "absolute": {}, "tails": {}, "checks": {}, "leave_one_out": {}}
    losses_cols = ["qs", "abs_mean", "abs_median", "err_mean"]
    losses_cols += [c for c in stacked.columns if c.startswith(("pin_", "cov_"))]
    for subset_name, expression in subsets().items():
        keep = meta.with_columns(keep=expression).filter(pl.col("keep")).select(KEYS)
        if keep.is_empty():
            continue
        arm_losses = {
            name: stacked.filter(pl.col("arm") == name)
            .join(keep, on=KEYS, how="inner")
            .select([*KEYS, *losses_cols])
            .sort(KEYS)
            for name in ARMS
        }
        loss_frame = losses_for_bootstrap(arm_losses=arm_losses).with_columns(
            site=pl.col("site").replace_strict(labels, return_dtype=pl.String)
        )
        out["absolute"][subset_name] = {
            "rows": arm_losses["base"].height,
            "mean_qs": {n: mean(f["qs"]) for n, f in arm_losses.items()},
            "mean_abs_mean_error": {n: mean(f["abs_mean"]) for n, f in arm_losses.items()},
            "mean_signed_error": {n: mean(f["err_mean"]) for n, f in arm_losses.items()},
        }
        out["tails"][subset_name] = {
            n: {
                **{f"cov_{q}": mean(f[f"cov_{q}"]) for q, _ in TAILS},
                **{f"pin_{q}": mean(f[f"pin_{q}"]) for q, _ in TAILS},
            }
            for n, f in arm_losses.items()
        }
        for cname, (treatment, reference, where) in CONTRASTS.items():
            if where != ("all" if subset_name == "all" else "headline") and not (
                where == "headline" and subset_name in {"window_winter", "cold_nights", "window"}
            ):
                continue
            if where == "all" and subset_name != "all":
                continue
            out["contrasts"][f"{cname}@{subset_name}"] = contrast(
                losses=loss_frame, treatment=treatment, reference=reference
            )
            if cname in {"C1", "C4"}:
                out["contrasts"][f"{cname}@{subset_name}@abs_mean"] = contrast(
                    losses=loss_frame, treatment=treatment, reference=reference, metric="abs_mean"
                )
        if subset_name == "headline":
            for site in common.SITES:
                part = loss_frame.filter(pl.col("site") != site)
                out["leave_one_out"][site] = contrast(
                    losses=part, treatment="t_site", reference="base"
                )["estimate"]
    out["checks"]["median_minus_mean"] = {
        n: {
            s: mean(f.filter(pl.col("site") == s)["q50"] - f.filter(pl.col("site") == s)["mean"])
            for s in common.SITES
        }
        for n, f in raw.items()
        if n in {"base", "t_site"}
    }
    five = ["D5", "D10", "D11", "D14", "D16"]
    d = raw["base"].filter(
        pl.col("site").is_in(five) & (pl.col("lead_hours") > 12) & (pl.col("lead_hours") <= 36)
    )
    out["checks"]["base_five_site_qs_12_36h"] = mean(add_losses(frame=d)["qs"])
    common.write_json_atomic(path=common.WORK_DIR / "report.json", content=out)
    print_report(out=out)


def print_report(*, out: dict) -> None:
    """Print every number the write-up quotes."""
    for key, value in out["contrasts"].items():
        print(
            f"{key}: {value['estimate']:+.3f} ({value['lower']:+.3f} to {value['upper']:+.3f})"
            f" rows {value['rows']} months {value['months']} readable {value['readable']}"
            f" sites_better {value['sites_better']}/6 months_neg {value['months_negative']}"
            f" rows/site {value['rows_by_site']}"
        )
    for subset, value in out["absolute"].items():
        print(subset, value["rows"])
        for metric in ("mean_qs", "mean_abs_mean_error", "mean_signed_error"):
            print("  ", metric, {k: round(v, 3) for k, v in value[metric].items()})
    print(
        "tails",
        {
            s: {n: {k: round(v, 3) for k, v in d.items()} for n, d in t.items()}
            for s, t in out["tails"].items()
            if s in {"all", "headline"}
        },
    )
    print("leave one out (C1)", out["leave_one_out"])
    print("median minus mean", out["checks"]["median_minus_mean"])
    print("base five-site QS 12-36 h (133 pool: 2.462)", out["checks"]["base_five_site_qs_12_36h"])


if __name__ == "__main__":
    main()
