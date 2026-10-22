"""Score experiment 190 and print every number the write-up quotes.

Reads `work/dp_preds/` (arms `d0` to `d3`, and the stage-1 arms `s0` to `s3`), the gate's
`work/gate/cases.parquet`, the A1 trigger times, experiment 169's arm A, and experiment 175's day
list. Writes `work/report/report.json`.

    uv run --no-sync python ml-experiments/00190-*/e00190_report.py
"""

import json
from typing import Final

import e00190_common as common
import e00190_run as run
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_calendar import BAND_NAMES
from studies.experiment_store import losses_for_bootstrap
from studies.sources import ML_EXPERIMENTS_DATA_DIR

QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
SITES: Final[tuple[str, ...]] = common.DECISION_SITES
FINAL_ARMS: Final[tuple[str, ...]] = ("d0", "d1", "d2", "d3")
STAGE1_ARMS: Final[tuple[str, ...]] = ("s0", "s1", "s2", "s3")
E169_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00169-calibration-combined-fixes-and-ensemble-spread" / "work"
)
E175_DAYS: Final = ML_EXPERIMENTS_DATA_DIR / "00175-dp-unlabelled-day-shapes" / "work"
LEVEL_PERCENT: Final[float] = 95.0
N_RESAMPLES: Final[int] = 20_000
NIGHT_BANDS: Final[tuple[int, ...]] = (BAND_NAMES.index("night"), BAND_NAMES.index("evening"))
WINTER: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
TAILS: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1 gated minus reference": ("d1", "d0"),
    "C2 always-on minus reference": ("d2", "d0"),
    "C3 gated minus always-on": ("d1", "d2"),
    "C4 shared minus gated": ("d3", "d1"),
}
OUT_DIR: Final = common.WORK_DIR / "report"


def load_arm(*, arm: str) -> pl.DataFrame:
    """Return an arm's scored-year rows at the five decision Primaries."""
    pieces = [
        pl.read_parquet(common.WORK_DIR / "dp_preds" / f"{arm}_scored_{site}.parquet").with_columns(
            site=pl.lit(site)
        )
        for site in SITES
    ]
    return pl.concat(pieces)


def scorable(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep rows with a reading, outside switching events, at leads of 0.5 to 168 hours."""
    return frame.filter(
        ~pl.col("switching_mask")
        & pl.col("actual").is_finite()
        & (pl.col("lead_hours") <= 168.0)
        & (pl.col("lead_hours") >= 0.5)
    )


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the quantile score, the mean's absolute error, and the tail coverage and pinball loss."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    quantiles = frame.select(list(QUANTILES)).to_numpy().astype(np.float64)
    error = actual[:, None] - quantiles
    columns = {
        "qs": 2.0 * np.mean(np.maximum(LEVELS * error, (LEVELS - 1.0) * error), axis=1),
        "abs_mean": np.abs(actual - frame["mean"].to_numpy().astype(np.float64)),
        "abs_median": np.abs(actual - frame["q50"].to_numpy().astype(np.float64)),
    }
    for name, level in TAILS:
        diff = actual - frame[name].to_numpy().astype(np.float64)
        columns[f"cov_{name}"] = (diff <= 0.0).astype(np.float64)
        columns[f"pin_{name}"] = np.maximum(level * diff, (level - 1.0) * diff)
    return frame.with_columns(pl.Series(k, v) for k, v in columns.items())


def fixed_spread(*, frame: pl.DataFrame, arm: str) -> pl.DataFrame:
    """Return the frame with experiment 169 arm A's quantile offsets from its own mean.

    Rows are matched on the run and the valid time. The result keeps only the matched rows.
    """
    pieces = [
        pl.read_parquet(E169_DIR / "dp_preds" / f"a_scored_{site}.parquet")
        .select("time_series_id", "run", "valid_time", "mean", *QUANTILES)
        .rename({"mean": "mean_169", **{q: f"{q}_169" for q in QUANTILES}})
        for site in SITES
    ]
    other = pl.concat(pieces)
    joined = frame.join(other, on=["time_series_id", "run", "valid_time"], how="inner")
    return (
        joined.with_columns(
            [(pl.col("mean") + pl.col(f"{q}_169") - pl.col("mean_169")).alias(q) for q in QUANTILES]
        )
        .drop([f"{q}_169" for q in QUANTILES] + ["mean_169"])
        .with_columns(arm=pl.lit(f"{arm}_f169"))
    )


def annotate(*, frame: pl.DataFrame, cache) -> pl.DataFrame:  # noqa: ANN001
    """Add the hour band, the winter flag, the cold-night flag, and the local date."""
    pieces = []
    for site in SITES:
        part = frame.filter(pl.col("site") == site)
        features = cache.sites[site]
        target = np.searchsorted(
            cache.end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        days = cache.end_times.astype("datetime64[D]")
        codes = np.unique(days, return_inverse=True)[1]
        temperature = features.temperature_c.astype(np.float64)
        finite = np.isfinite(temperature)
        total = np.bincount(codes, weights=np.where(finite, temperature, 0.0))
        count = np.bincount(codes, weights=finite.astype(np.float64))
        daily = (total / np.maximum(count, 1))[codes]
        pieces.append(
            part.with_columns(
                cold2=pl.Series(np.nan_to_num(daily[target], nan=99.0) < 2.0),
                local_date=pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.date(),
                valid_month=pl.col("valid_time").dt.month(),
            )
        )
    return pl.concat(pieces)


def triggered(*, frame: pl.DataFrame) -> pl.Series:
    """Return True for rows whose run origin lies within 24 hours after a gate trigger."""
    out = np.zeros(frame.height, dtype=bool)
    for site in SITES:
        triggers = pl.read_parquet(common.WORK_DIR / "triggers" / f"trigger_{site}.parquet")
        times = np.sort(triggers["trigger_end_time"].dt.replace_time_zone(None).to_numpy())
        rows = np.flatnonzero((frame["site"] == site).to_numpy())
        origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()[rows]
        found = np.searchsorted(times, origin, side="right") - 1
        valid = found >= 0
        gap = np.full(len(rows), np.timedelta64(10**9, "h"))
        gap[valid] = origin[valid] - times[found[valid]]
        out[rows] = gap <= np.timedelta64(24, "h")
    return pl.Series("after_trigger", out)


def contrast(*, losses: pl.DataFrame, treatment: str, reference: str, metric: str = "qs") -> dict:
    """Return a paired contrast with a month-resampled interval and site and month signs."""
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
        t.group_by("site").agg(t=pl.col(metric).mean())
        .join(r.group_by("site").agg(r=pl.col(metric).mean()), on="site")
    )  # fmt: skip
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "sites_better": int((by_site["t"] < by_site["r"]).sum()),
        "n_sites": by_site.height,
        "rows": t.height,
    }


def day_bootstrap(*, per_day: np.ndarray, n: int = N_RESAMPLES, seed: int = 190) -> list[float]:
    """Return the 95% interval of the mean of per-day values, resampling days."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(per_day, size=(n, len(per_day)), replace=True).mean(axis=1)
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def day_subsets() -> dict[str, pl.DataFrame]:
    """Return experiment 175's smooth-offset and block days (site, local_date)."""
    shapes = pl.read_parquet(E175_DAYS / "day_shapes.parquet")
    return {
        "smooth-offset days (experiment 175)": shapes.filter(pl.col("shape") == "b_offset").select(
            "site", "local_date"
        ),
        "block days (experiment 175)": shapes.filter(pl.col("shape") == "a_block").select(
            "site", "local_date"
        ),
    }


def excess_reduction(*, losses: pl.DataFrame, days: pl.DataFrame, treatment: str) -> dict:
    """Return the cut in excess loss on the listed days, treatment against `d0`.

    A day's excess loss is its mean quantile score minus arm D's mean over all scored rows of
    the same substations (one common baseline for every arm). The reduction is one minus the treatment's excess over
    the reference's.
    """
    sub = losses.join(days, on=["site", "local_date"], how="inner")
    if sub.is_empty():
        return {}
    base = losses.filter(pl.col("arm") == "d0").group_by("site").agg(base=pl.col("qs").mean())
    daily = (
        sub.group_by(["arm", "site", "local_date"]).agg(qs=pl.col("qs").mean(), n=pl.len())
        .join(base, on="site")
        .with_columns(excess=pl.col("qs") - pl.col("base"))
    )  # fmt: skip
    t = daily.filter(pl.col("arm") == treatment).sort(["site", "local_date"])
    r = daily.filter(pl.col("arm") == "d0").sort(["site", "local_date"])
    excess_t, excess_r = t["excess"].to_numpy(), r["excess"].to_numpy()
    reduction = 1.0 - excess_t.sum() / excess_r.sum()
    rng = np.random.default_rng(190)
    draws = []
    for _ in range(N_RESAMPLES):
        pick = rng.integers(0, len(excess_t), len(excess_t))
        draws.append(1.0 - excess_t[pick].sum() / excess_r[pick].sum())
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return {
        "days": len(excess_t),
        "excess_reference": float(excess_r.mean()),
        "excess_treatment": float(excess_t.mean()),
        "reduction": float(reduction),
        "lower": float(lower),
        "upper": float(upper),
    }


def gate_report() -> dict:
    """Return the known-answer gate: recovery by size, duration, and arm, and harm on quiet days."""
    cases = pl.read_parquet(common.WORK_DIR / "gate" / "cases.parquet")
    injected = cases.filter(pl.col("kind") == "injected")
    arms = [v for v in injected["variant"].unique().to_list() if v != "clean_a0"]
    out: dict = {"recovery": {}, "curve": {}, "alone": {}}
    rng = np.random.default_rng(190)

    def weighted(frame: pl.DataFrame) -> float:
        return float((frame["loss"] * frame["n"]).sum() / frame["n"].to_numpy().sum())

    def recovery(*, frame: pl.DataFrame, arm: str, pick: np.ndarray | None = None) -> float:
        by_case = {}
        for name in ("clean_a0", "a0", arm):
            part = frame.filter(pl.col("variant") == name).sort(["site", "case"])
            by_case[name] = part
        keys = by_case["a0"].select("site", "case")
        lc = by_case["clean_a0"]["loss"].to_numpy() * by_case["clean_a0"]["n"].to_numpy()
        l0 = by_case["a0"]["loss"].to_numpy() * by_case["a0"]["n"].to_numpy()
        la = by_case[arm]["loss"].to_numpy() * by_case[arm]["n"].to_numpy()
        del keys
        if pick is not None:
            lc, l0, la = lc[pick], l0[pick], la[pick]
        return 1.0 - (la.sum() - lc.sum()) / (l0.sum() - lc.sum())

    for size in sorted(injected["size"].unique().to_list()):
        for hours in sorted(injected["hours"].unique().to_list()):
            cell = injected.filter(
                (pl.col("size") == size) & (pl.col("hours") == hours) & (pl.col("bin") == "all")
            )
            n_cases = cell.filter(pl.col("variant") == "a0").height
            for arm in arms:
                if arm == "a0":
                    continue
                point = recovery(frame=cell, arm=arm)
                draws = [
                    recovery(frame=cell, arm=arm, pick=rng.integers(0, n_cases, n_cases))
                    for _ in range(2000)
                ]
                lower, upper = np.percentile(draws, [2.5, 97.5])
                out["recovery"][f"{size}|{hours}|{arm}"] = {
                    "recovery": float(point),
                    "lower": float(lower),
                    "upper": float(upper),
                    "cases": n_cases,
                    "excess_a0": weighted(cell.filter(pl.col("variant") == "a0"))
                    - weighted(cell.filter(pl.col("variant") == "clean_a0")),
                }
    for hours in sorted(injected["hours"].unique().to_list()):
        for arm in ("a1", "a2", "a3"):
            for label in ("0-3.5 h", "3.5-6 h", "6-12 h", "12-24 h", "24-36 h"):
                cell = injected.filter(
                    (pl.col("size") == 0.1) & (pl.col("hours") == hours) & (pl.col("bin") == label)
                )
                if cell.is_empty():
                    continue
                n_cases = cell.filter(pl.col("variant") == "a0").height
                # Bins hold fewer cases than "all" when a duration is shorter than the bin.
                if (
                    cell.filter(pl.col("variant") == "clean_a0").height != n_cases
                    or cell.filter(pl.col("variant") == arm).height != n_cases
                ):
                    continue
                out["curve"][f"{hours}|{arm}|{label}"] = {
                    "recovery": recovery(frame=cell, arm=arm),
                    "cases": n_cases,
                    "excess_a0": weighted(cell.filter(pl.col("variant") == "a0"))
                    - weighted(cell.filter(pl.col("variant") == "clean_a0")),
                }
    alone = cases.filter(pl.col("kind") == "alone")
    reference = alone.filter(pl.col("variant") == "a0").sort(["site", "case"])
    for arm in alone["variant"].unique().to_list():
        if arm == "a0":
            continue
        other = alone.filter(pl.col("variant") == arm).sort(["site", "case"])
        diff = other["loss"].to_numpy() - reference["loss"].to_numpy()
        out["alone"][arm] = {
            "difference": float(diff.mean()),
            "interval": day_bootstrap(per_day=diff),
            "cases": len(diff),
            "reference": float(reference["loss"].to_numpy().mean()),
        }
    return out


def main() -> None:  # noqa: PLR0915
    """Score everything and write the report."""
    cache = common.load_cache()
    labels = {cache.sites[s].table.time_series_id: s for s in SITES}
    raw = {arm: scorable(frame=load_arm(arm=arm)) for arm in (*FINAL_ARMS, *STAGE1_ARMS)}
    shared = None
    for frame in raw.values():
        keys = frame.select(KEYS).unique()
        shared = keys if shared is None else shared.join(keys, on=KEYS, how="inner")
    assert shared is not None
    raw = {k: v.join(shared, on=KEYS, how="inner").sort(KEYS) for k, v in raw.items()}
    for name in FINAL_ARMS:
        raw[f"{name}_f169"] = fixed_spread(frame=raw[name], arm=name)
    shared169 = raw["d0_f169"].select(KEYS).unique()
    for name in FINAL_ARMS:
        raw[f"{name}_f169"] = raw[f"{name}_f169"].join(shared169, on=KEYS, how="inner").sort(KEYS)
    report: dict = {"rows": raw["d0"].height, "rows_f169": raw["d0_f169"].height}

    def stack(names: list[str]) -> pl.DataFrame:
        pieces = {}
        for name in names:
            frame = raw[name]
            frame = annotate(frame=frame, cache=cache)
            frame = frame.with_columns(after_trigger=triggered(frame=frame))
            keep = [
                *KEYS, "seed", "lead_hours", *QUANTILES, "actual", "mean", "hour_band", "cold2",
                "local_date", "valid_month", "after_trigger",
            ]  # fmt: skip
            pieces[name] = with_losses(frame=frame.select(keep))
        stacked = losses_for_bootstrap(arm_losses=pieces)
        return stacked.with_columns(
            site=pl.col("site").replace_strict(labels, return_dtype=pl.String)
        )

    losses = stack([*FINAL_ARMS])
    report["scores"] = {
        arm: float(losses.filter(pl.col("arm") == arm)["qs"].to_numpy().mean())
        for arm in FINAL_ARMS
    }
    stage1 = stack([*STAGE1_ARMS])
    report["stage1_scores"] = {
        arm: float(stage1.filter(pl.col("arm") == arm)["qs"].to_numpy().mean())
        for arm in STAGE1_ARMS
    }
    report["contrasts"] = {
        label: contrast(losses=losses, treatment=t, reference=r)
        for label, (t, r) in CONTRASTS.items()
    }
    report["contrast_abs_mean"] = {
        label: contrast(losses=losses, treatment=t, reference=r, metric="abs_mean")
        for label, (t, r) in CONTRASTS.items()
    }
    fixed = stack([f"{a}_f169" for a in FINAL_ARMS])
    report["fixed_spread_scores"] = {
        arm: float(fixed.filter(pl.col("arm") == f"{arm}_f169")["qs"].to_numpy().mean())
        for arm in FINAL_ARMS
    }
    report["fixed_spread_contrasts"] = {
        label: contrast(losses=fixed, treatment=f"{t}_f169", reference=f"{r}_f169")
        for label, (t, r) in CONTRASTS.items()
    }
    # Subsets.
    night = pl.col("hour_band").is_in(list(NIGHT_BANDS))
    winter = pl.col("valid_month").is_in(list(WINTER))
    subsets = {
        "rows within 24 h after a trigger": pl.col("after_trigger"),
        "cold nights (day mean below 2 C)": winter & pl.col("cold2") & night,
    }
    report["subsets"] = {}
    for label, expression in subsets.items():
        part = losses.filter(expression)
        report["subsets"][label] = {
            "rows": part.filter(pl.col("arm") == "d0").height,
            "scores": {
                arm: float(part.filter(pl.col("arm") == arm)["qs"].to_numpy().mean())
                for arm in FINAL_ARMS
            },
            "c1": day_free_difference(part=part, treatment="d1", reference="d0"),
            "c2": day_free_difference(part=part, treatment="d2", reference="d0"),
        }
    report["days"] = {}
    for label, days in day_subsets().items():
        report["days"][label] = {
            arm: excess_reduction(losses=losses, days=days, treatment=arm)
            for arm in ("d1", "d2", "d3")
        }
    # Tails and checks.
    tails = {}
    for arm in FINAL_ARMS:
        part = losses.filter(pl.col("arm") == arm)
        cold = part.filter(winter & pl.col("cold2") & night)
        tails[arm] = {
            "all": {n: float(part[f"cov_{n}"].to_numpy().mean()) for n, _ in TAILS},
            "cold_nights": {n: float(cold[f"cov_{n}"].to_numpy().mean()) for n, _ in TAILS},
            "pinball_all": {n: float(part[f"pin_{n}"].to_numpy().mean()) for n, _ in TAILS},
        }
    report["tails"] = tails
    checks = {}
    for arm in FINAL_ARMS:
        frame = raw[arm]
        checks[arm] = {}
        for site in SITES:
            part = frame.filter(pl.col("site") == site)
            checks[arm][site] = float((part["q50"] - part["mean"]).to_numpy().mean())
    report["median_minus_mean"] = checks
    report["triggers"] = trigger_summary(cache=cache)
    report["gate"] = gate_report()
    report["frozen"] = json.loads(run.FROZEN_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common.write_json_atomic(path=OUT_DIR / "report.json", content=report)
    print(json.dumps(report, indent=1, default=float))


def day_free_difference(*, part: pl.DataFrame, treatment: str, reference: str) -> dict:
    """Return a subset's paired difference with a month-resampled interval, if it has 3+ months."""
    if part.filter(pl.col("arm") == treatment)["month"].n_unique() < 3:
        t = part.filter(pl.col("arm") == treatment)["qs"].to_numpy().mean()
        r = part.filter(pl.col("arm") == reference)["qs"].to_numpy().mean()
        return {"estimate": float(t - r)}
    return contrast(losses=part, treatment=treatment, reference=reference)


def trigger_summary(*, cache) -> dict:  # noqa: ANN001
    """Return, per substation, the number of triggers and the share of readings it was on."""
    out = {}
    for site in SITES:
        triggers = pl.read_parquet(common.WORK_DIR / "triggers" / f"trigger_{site}.parquet")
        on = pl.read_parquet(common.WORK_DIR / "triggers" / f"on_{site}.parquet")
        readings = int(np.isfinite(cache.sites[site].y).sum())
        out[site] = {"triggers": triggers.height, "share_on": on.height / readings}
    return out


if __name__ == "__main__":
    main()
