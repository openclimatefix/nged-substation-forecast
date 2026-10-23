"""Print experiment 279's numbers: scores, the planned contrasts, tails, checks, and the curves.

Every number the write-up quotes comes from this script. It reads the stored prediction sets
(`<year>_<arm>_k`, the arm with the Kalman filter), scores each row with the quantile score (twice
the mean pinball loss over the 11 levels, in points of effective capacity) on rows outside the
hand-filtered switching events, and pairs the arms row by row at day D+1.

- **Contrasts C1 to C3** (arm minus T-all-K on cold nights, 99% interval, month-resampled), C4
  (A3 minus A2) and C5 (A2w minus A2). Each has the Nov-Mar and all-rows contrasts, the replicate
  year, the count of decision sites that improve on cold nights, the leave-one-site-out range, and
  the mean-forecast error beside the score.
- **Tails** at q01, q05, q95, and q99, on cold nights and on all rows.
- **Checks**: T-all-K against experiment 126's stored forecast, median minus mean per site, and the
  triggers of the `ml-experiment` skill's audit.
- **Curves**: the slopes of A3's fitted heating curve, from `work/curves_*.json`.

    uv run --no-sync python ml-experiments/00279-*/e00279_report.py > report.txt
"""

import json
from functools import cache
from typing import Any, Final

import e00279_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_features import FeatureCache
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import losses_for_bootstrap

DECISION: Final[tuple[str, ...]] = common.DECISION_SITES
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time", "seed"]
REFERENCE: Final[str] = "t_all_k"
ARMS: Final[tuple[str, ...]] = ("a1", "a2", "a2w", "a3", "a3nw", "a3b", "a3bnw", "a3c")
COLD_DAY_MEAN_C: Final[float] = 3.0
NIGHT_START_HOUR: Final[int] = 22
NIGHT_END_HOUR: Final[int] = 8
THRESHOLD_A1_A2: Final[float] = -0.02
THRESHOLD_A3: Final[float] = -0.04
NON_INFERIORITY: Final[float] = 0.02
MIN_SITES_IMPROVING: Final[int] = 4
TAIL_LEVELS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
STEP_HOURS_UTC: Final[tuple[float, float]] = (0.5, 7.5)
"""The Economy 7 night-step hours, UTC hour of the half-hour's start in [0.5, 7.5)."""
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
LOCAL: Final[str] = "Europe/London"


@cache
def _cache() -> FeatureCache:
    return common.load_cache()


@cache
def _site_of_id() -> dict[int, str]:
    return {_cache().sites[s].table.time_series_id: s for s in common.SCORED_SITES}


@cache
def _daily_mean() -> dict[str, pl.DataFrame]:
    """Return each site's mean temperature by local calendar day (past weather)."""
    out = {}
    for site in common.SCORED_SITES:
        features = _cache().sites[site]
        start = pl.Series(_cache().end_times - np.timedelta64(30, "m")).dt.replace_time_zone("UTC")
        frame = pl.DataFrame(
            {"day": start.dt.convert_time_zone(LOCAL).dt.date(), "t": features.temperature_c}
        )
        out[site] = frame.group_by("day").agg(day_mean=pl.col("t").mean())
    return out


@cache
def load_arm(*, year: str, arm: str) -> pl.DataFrame:
    """Read a stored set with its per-row score, error of the mean, and subset masks."""
    frame = pl.read_parquet(
        common.DATA_DIR / "predictions" / f"{year}_{arm}" / "predictions.parquet"
    ).filter(~pl.col("switching_mask"))
    quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    start = (pl.col("valid_time") - pl.duration(minutes=30)).dt.convert_time_zone(LOCAL)
    frame = frame.with_columns(
        qs=pl.Series(common.quantile_score(actual=actual, quantiles=quantiles)),
        ae=(pl.col("actual") - pl.col("mean")).abs(),
        site_name=pl.col("time_series_id").replace_strict(_site_of_id()),
        local_hour=start.dt.hour(),
        utc_hour=(pl.col("valid_time") - pl.duration(minutes=30)).dt.hour()
        + (pl.col("valid_time") - pl.duration(minutes=30)).dt.minute() / 60.0,
        month=start.dt.month(),
        night_day=(start - pl.duration(hours=NIGHT_END_HOUR)).dt.date(),
    )
    parts = []
    for site, table in _daily_mean().items():
        rows = frame.filter(pl.col("site_name") == site).join(
            table.rename({"day": "night_day"}), on="night_day", how="left"
        )
        rows = rows.with_columns(start_day=start_expr_date()).join(
            table.rename({"day": "start_day", "day_mean": "own_day_mean"}),
            on="start_day",
            how="left",
        )
        parts.append(rows)
    frame = pl.concat(parts)
    night = (pl.col("local_hour") >= NIGHT_START_HOUR) | (pl.col("local_hour") < NIGHT_END_HOUR)
    return frame.with_columns(
        cold_night=night & (pl.col("day_mean") <= COLD_DAY_MEAN_C),
        winter=pl.col("month").is_in(WINTER_MONTHS),
        warm=pl.col("own_day_mean") > common.BASE_THRESHOLD_C,
        step=(pl.col("utc_hour") >= STEP_HOURS_UTC[0])
        & (pl.col("utc_hour") < STEP_HOURS_UTC[1])
        & pl.col("month").is_in(WINTER_MONTHS),
    )


def start_expr_date() -> pl.Expr:
    """Return the local calendar date of a row's half-hour start."""
    return (pl.col("valid_time") - pl.duration(minutes=30)).dt.convert_time_zone(LOCAL).dt.date()


def paired(
    *,
    year: str,
    treatment: str,
    reference: str,
    sites: tuple[str, ...],
    subset: str | None,
    invert: bool = False,
) -> pl.DataFrame:
    """Return both arms' rows on the sites and subset, paired, for `studies.bootstrap`."""
    arms = {}
    for name in (treatment, reference):
        frame = load_arm(year=year, arm=name).filter(pl.col("site_name").is_in(sites))
        if subset is not None:
            frame = frame.filter(~pl.col(subset) if invert else pl.col(subset))
        arms[name] = frame.select(*KEYS, "qs", "ae")
    first, second = list(arms.values())
    shared = first.select(KEYS).join(second.select(KEYS), on=KEYS, how="inner")
    arms = {name: frame.join(shared, on=KEYS, how="inner") for name, frame in arms.items()}
    return losses_for_bootstrap(arm_losses=arms)


def estimate(
    *,
    treatment: str,
    reference: str = REFERENCE,
    year: str = "scored",
    sites: tuple[str, ...] = DECISION,
    subset: str | None = None,
    invert: bool = False,
) -> tuple[float, float, int]:
    """Return the mean paired difference in score, in error of the mean, and the row count."""
    losses = paired(
        year=year,
        treatment=treatment,
        reference=reference,
        sites=sites,
        subset=subset,
        invert=invert,
    )
    means = losses.group_by("arm").agg(
        pl.col("qs").mean(), pl.col("ae").mean(), pl.len().alias("n")
    )
    first = means.filter(pl.col("arm") == treatment).row(0, named=True)
    second = means.filter(pl.col("arm") == reference).row(0, named=True)
    return first["qs"] - second["qs"], first["ae"] - second["ae"], int(first["n"])


def interval(
    *,
    treatment: str,
    reference: str = REFERENCE,
    sites: tuple[str, ...] = DECISION,
    subset: str | None = None,
    invert: bool = False,
    level: float = common.PLANNED_LEVEL,
) -> tuple[float, float]:
    """Return the month-resampled interval of the contrast."""
    losses = paired(
        year="scored",
        treatment=treatment,
        reference=reference,
        sites=sites,
        subset=subset,
        invert=invert,
    )
    return bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric="qs",
        level=level,
        n_resamples=common.N_RESAMPLES,
    )


def contrast_line(*, name: str, treatment: str, reference: str = REFERENCE) -> dict[str, Any]:
    """Print and return one contrast on the planned subsets."""
    record: dict[str, Any] = {"contrast": name, "treatment": treatment, "reference": reference}
    for label, subset, invert in (
        ("cold_night", "cold_night", False),
        ("nov_mar", "winter", False),
        ("all_rows", None, False),
        ("not_cold_night", "cold_night", True),
        ("step_hours", "step", False),
        ("warm_days", "warm", False),
    ):
        scored, ae, n = estimate(
            treatment=treatment, reference=reference, subset=subset, invert=invert
        )
        low, high = interval(treatment=treatment, reference=reference, subset=subset, invert=invert)
        replicate, _, _ = estimate(
            treatment=treatment, reference=reference, year="replicate", subset=subset, invert=invert
        )
        record[label] = {
            "estimate": scored,
            "low": low,
            "high": high,
            "mean_error": ae,
            "replicate": replicate,
            "rows": n,
        }
        print(
            f"{name} {label}: {scored:+.4f} ({low:+.4f} to {high:+.4f}), replicate "
            f"{replicate:+.4f}, mean-forecast error {ae:+.4f}, rows {n}"
        )
    better = {}
    for site in DECISION:
        scored, _, _ = estimate(
            treatment=treatment, reference=reference, sites=(site,), subset="cold_night"
        )
        better[site] = scored
    record["sites_cold_night"] = better
    record["sites_improving"] = sum(v < 0.0 for v in better.values())
    print(f"{name} cold-night by site: { {k: round(v, 4) for k, v in better.items()} }")
    loo = {}
    for site in DECISION:
        rest = tuple(s for s in DECISION if s != site)
        loo[site], _, _ = estimate(
            treatment=treatment, reference=reference, sites=rest, subset="cold_night"
        )
    record["leave_one_out"] = loo
    print(f"{name} cold-night leave-one-site-out: { {k: round(v, 4) for k, v in loo.items()} }")
    return record


def label_of(*, record: dict[str, Any], threshold: float) -> str:
    """Return the decision rule's label for an arm from its contrast record."""
    cold = record["cold_night"]
    other = record["not_cold_night"]
    if cold["low"] > 0.0:
        return "worse"
    if cold["high"] < 0.0:
        if other["high"] > NON_INFERIORITY:
            return "worse on other rows"
        passes = (
            cold["estimate"] <= threshold
            and cold["replicate"] < 0.0
            and record["sites_improving"] >= MIN_SITES_IMPROVING
        )
        if passes:
            return "gain"
        return "small gain" if cold["replicate"] < 0.0 else "not replicated"
    return "tie"


def tails() -> list[dict[str, Any]]:
    """Return coverage and pinball loss at the tail quantiles, on cold nights and all rows."""
    rows = []
    for arm in (REFERENCE, *(f"{a}_k" for a in ARMS)):
        frame = load_arm(year="scored", arm=arm).filter(pl.col("site_name").is_in(DECISION))
        for label, subset in (("cold_night", frame.filter(pl.col("cold_night"))), ("all", frame)):
            actual = subset["actual"].to_numpy().astype(np.float64)
            entry: dict[str, Any] = {"arm": arm, "rows": label, "n": subset.height}
            for column, level in TAIL_LEVELS.items():
                prediction = subset[column].to_numpy().astype(np.float64)
                error = actual - prediction
                entry[f"{column}_coverage"] = float((actual <= prediction).mean())
                entry[f"{column}_pinball"] = float(
                    np.maximum(level * error, (level - 1.0) * error).mean()
                )
            rows.append(entry)
            print(
                f"tails {arm} {label}: "
                + ", ".join(
                    f"{c} cov {entry[f'{c}_coverage']:.3f} pin {entry[f'{c}_pinball']:.4f}"
                    for c in TAIL_LEVELS
                )
            )
    return rows


def absolute_scores() -> dict[str, Any]:
    """Return each arm's mean score on the scored year's subsets."""
    out: dict[str, Any] = {}
    for arm in (REFERENCE, *(f"{a}_k" for a in ARMS)):
        frame = load_arm(year="scored", arm=arm).filter(pl.col("site_name").is_in(DECISION))
        entry = {
            "all": float(frame["qs"].mean()),
            "nov_mar": float(frame.filter(pl.col("winter"))["qs"].mean()),
            "cold_night": float(frame.filter(pl.col("cold_night"))["qs"].mean()),
            "step_hours": float(frame.filter(pl.col("step"))["qs"].mean()),
            "winter_mae_median": float(
                (frame.filter(pl.col("winter"))["actual"] - frame.filter(pl.col("winter"))["q50"])
                .abs()
                .mean()
            ),
            "cold_night_mae_median": float(
                (
                    frame.filter(pl.col("cold_night"))["actual"]
                    - frame.filter(pl.col("cold_night"))["q50"]
                )
                .abs()
                .mean()
            ),
        }
        out[arm] = entry
        print(f"absolute {arm}: " + ", ".join(f"{k} {v:.4f}" for k, v in entry.items()))
    return out


def median_minus_mean() -> dict[str, dict[str, float]]:
    """Return median forecast minus mean forecast per site, mean over the scored year."""
    out: dict[str, dict[str, float]] = {}
    for arm in (REFERENCE, *(f"{a}_k" for a in ARMS)):
        frame = load_arm(year="scored", arm=arm)
        per_site = frame.group_by("site_name").agg(gap=(pl.col("q50") - pl.col("mean")).mean())
        out[arm] = {r["site_name"]: float(r["gap"]) for r in per_site.iter_rows(named=True)}
        print(f"median minus mean {arm}: { {k: round(v, 4) for k, v in sorted(out[arm].items())} }")
    return out


def reproduction() -> dict[str, float]:
    """Compare T-all-K here with experiment 126's stored scored-year forecast."""
    path = (
        common.ML_EXPERIMENTS_DATA_DIR
        / "00126-differentiable-physics-s1-cheap-screens-added-to-gam-k"
        / "predictions"
        / "scored_t_all_k"
        / "predictions.parquet"
    )
    old = pl.read_parquet(path).select(*KEYS, old=pl.col("mean"), old_q50=pl.col("q50"))
    new = pl.read_parquet(
        common.DATA_DIR / "predictions" / "scored_t_all_k" / "predictions.parquet"
    ).select(*KEYS, new=pl.col("mean"), new_q50=pl.col("q50"))
    joined = old.join(new, on=KEYS, how="inner")
    gap = (joined["old"] - joined["new"]).abs().max()
    gap_q = (joined["old_q50"] - joined["new_q50"]).abs().max()
    print(
        f"reproduction: {joined.height} rows of {old.height}, max |mean difference| {gap}, "
        f"max |median difference| {gap_q}"
    )
    return {"rows": joined.height, "max_mean_gap": float(gap), "max_median_gap": float(gap_q)}  # type: ignore[arg-type]


def curve_slopes() -> dict[str, Any]:
    """Return A3's fitted heating curve per site, averaged over groups, and its shape slopes."""
    curves: dict[str, dict[str, Any]] = {}
    for path in sorted(common.WORK_DIR.glob("curves_*.json")):
        for site, arms in json.loads(path.read_text()).items():
            curves.setdefault(site, {}).update(arms)
    if not curves:
        return {}
    grid = np.arange(-4.0, 18.5, 1.0)
    out: dict[str, Any] = {}
    for variant in ("a3", "a3nw", "a3b", "a3bnw", "a3c"):
        for site, arms in curves.items():
            if variant not in arms:
                continue
            groups = arms[variant]["scored"]
            mean_curve = np.mean([groups[g] for g in groups], axis=0)

            def slope(low: float, high: float, curve: np.ndarray = mean_curve) -> float:
                i, j = int(np.argmin(abs(grid - low))), int(np.argmin(abs(grid - high)))
                return float((curve[i] - curve[j]) / (high - low))

            entry = {
                "curve": [float(v) for v in mean_curve],
                "slope_below_3": slope(-4.0, 3.0),
                "slope_6_12": slope(6.0, 12.0),
                "slope_12_18": slope(12.0, 18.0),
            }
            entry["flat_above_12"] = entry["slope_12_18"] < 0.5 * entry["slope_6_12"]
            entry["steep_below_3"] = entry["slope_below_3"] > 1.25 * entry["slope_6_12"]
            out[f"{variant}_{site}"] = entry
            print(
                f"curve {variant} {site}: slope below 3 {entry['slope_below_3']:.3f}, 6-12 "
                f"{entry['slope_6_12']:.3f}, 12-18 {entry['slope_12_18']:.3f}, flat above 12 "
                f"{entry['flat_above_12']}, steep below 3 {entry['steep_below_3']}"
            )
    return out


def main() -> None:
    """Print every number and store them in `work/numbers.json`."""
    numbers: dict[str, Any] = {"reproduction": reproduction()}
    numbers["absolute"] = absolute_scores()
    contrasts = {
        "C1": contrast_line(name="C1 A1-A0", treatment="a1_k"),
        "C2": contrast_line(name="C2 A2-A0", treatment="a2_k"),
        "C3": contrast_line(name="C3 A3-A0", treatment="a3_k"),
        "C4": contrast_line(name="C4 A3-A2", treatment="a3_k", reference="a2_k"),
        "C5": contrast_line(name="C5 A2w-A2", treatment="a2w_k", reference="a2_k"),
        "C6": contrast_line(name="C6 A3nw-A0", treatment="a3nw_k"),
        "C7": contrast_line(name="C7 A3b-A0 (post hoc)", treatment="a3b_k"),
        "C8": contrast_line(name="C8 A3bnw-A0 (post hoc)", treatment="a3bnw_k"),
        "C9": contrast_line(name="C9 A3b-A2 (post hoc)", treatment="a3b_k", reference="a2_k"),
        "C10": contrast_line(
            name="C10 A3b-A3bnw (post hoc)", treatment="a3b_k", reference="a3bnw_k"
        ),
    }
    contrasts["C11"] = contrast_line(name="C11 A3c-A0 (post hoc)", treatment="a3c_k")
    contrasts["C12"] = contrast_line(
        name="C12 A3c-A2 (post hoc)", treatment="a3c_k", reference="a2_k"
    )
    numbers["contrasts"] = contrasts
    numbers["labels"] = {
        "C1": label_of(record=contrasts["C1"], threshold=THRESHOLD_A1_A2),
        "C2": label_of(record=contrasts["C2"], threshold=THRESHOLD_A1_A2),
        "C3": label_of(record=contrasts["C3"], threshold=THRESHOLD_A3),
        "C7": label_of(record=contrasts["C7"], threshold=THRESHOLD_A3),
        "C11": label_of(record=contrasts["C11"], threshold=THRESHOLD_A3),
    }
    gate = json.loads((common.WORK_DIR / "gate.json").read_text())["verdicts"]
    numbers["gate"] = gate
    if not all(v for k, v in gate.items() if k.endswith("_curved")):
        numbers["labels_after_gate"] = {
            k: ("inconclusive" if k in ("C7", "C11") else v) for k, v in numbers["labels"].items()
        }
        print("gate failed on a curved truth; network labels capped:", numbers["labels_after_gate"])
    print("labels:", numbers["labels"])
    numbers["tails"] = tails()
    numbers["median_minus_mean"] = median_minus_mean()
    numbers["curves"] = curve_slopes()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (common.WORK_DIR / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")


if __name__ == "__main__":
    main()
