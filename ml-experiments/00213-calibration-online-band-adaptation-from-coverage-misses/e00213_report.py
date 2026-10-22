"""Score online band adaptation against arm A; print every number the write-up quotes.

Picks gamma on the tuning months by the pre-registered rule, scores the replicate months and the
scored year, and writes `work/report/report.json`.

    uv run --no-sync python ml-experiments/00213-*/e00213_report.py
"""

import datetime as dt
from typing import Final

import e00213_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_calendar import BAND_NAMES, bank_holiday_names
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import losses_for_bootstrap
from studies.sources import ML_EXPERIMENTS_DATA_DIR

KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
LEVEL_PERCENT: Final[float] = 95.0
N_RESAMPLES: Final[int] = 20_000
PASS_COVERAGE: Final[float] = 0.90
PASS_UPPER_BOUND: Final[float] = 0.01
Q95_RANGE: Final[tuple[float, float]] = (0.94, 0.96)
Q05_EXTRA_DISTANCE: Final[float] = 0.01
COLD_DAY_C: Final[float] = 2.0
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
NIGHT_BANDS: Final[tuple[int, ...]] = (BAND_NAMES.index("night"), BAND_NAMES.index("evening"))
CHRISTMAS_START: Final[tuple[int, int]] = (12, 24)
CHRISTMAS_END: Final[tuple[int, int]] = (1, 1)
ARM_A_SCORE: Final[float] = 2.7519
"""Experiment 169's printed arm A score over leads 0.5 to 168 hours in the scored year."""
TAIL_QUANTILES: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)
STORED_87: Final[dict[str, str]] = {
    "bg2": "p_s6_bg2",
    "climatology": "ref_climatology",
    "manual_heuristic": "ref_manual_heuristic",
}
E87_PREDICTIONS: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
)
SPELL_LABELS: Final[tuple[str, ...]] = ("first night", "days 1-3 after", "later nights")
BASE_COLUMNS: Final[list[str]] = [
    *KEYS, "lead_hours", "seed", *QUANTILE_COLUMNS, "actual", "mean", "site",
]  # fmt: skip


def daily_mean_temperature(*, features, end_times: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Return the ERA5 temperature's mean over each half-hour's UTC date, on the grid."""
    days = end_times.astype("datetime64[D]")
    codes = np.unique(days, return_inverse=True)[1]
    temperature = features.temperature_c.astype(np.float64)
    finite = np.isfinite(temperature)
    total = np.bincount(codes, weights=np.where(finite, temperature, 0.0))
    count = np.bincount(codes, weights=finite.astype(np.float64))
    return (total / np.maximum(count, 1))[codes]


def annotate(*, frame: pl.DataFrame, cache) -> pl.DataFrame:  # noqa: ANN001
    """Keep scorable rows; add the cold-day flag, month, holiday flags, and local date."""
    pieces = []
    for site in common.SCORED_SITES:
        part = frame.filter(pl.col("site") == site)
        features = cache.sites[site]
        target = np.searchsorted(
            cache.end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        daily = daily_mean_temperature(features=features, end_times=cache.end_times)
        pieces.append(
            part.with_columns(
                masked=pl.Series(features.switching_mask[target]),
                hour_band=pl.Series(cache.calendar.band[target].astype(np.int8)),
                cold2=pl.Series(np.nan_to_num(daily[target], nan=99.0) < COLD_DAY_C),
                valid_month=pl.col("valid_time").dt.month(),
            )
        )
    bank = list(bank_holiday_names())
    local_date = pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.date()
    return (
        pl.concat(pieces)
        .with_columns(
            local_date=local_date,
            night_date=pl.when(pl.col("hour_band") == BAND_NAMES.index("night"))
            .then(local_date - dt.timedelta(days=1))
            .otherwise(local_date),
            bank_holiday=local_date.is_in(bank),
            christmas=(
                (
                    (local_date.dt.month() == CHRISTMAS_START[0])
                    & (local_date.dt.day() >= CHRISTMAS_START[1])
                )
                | (
                    (local_date.dt.month() == CHRISTMAS_END[0])
                    & (local_date.dt.day() <= CHRISTMAS_END[1])
                )
            ),
        )
        .filter(~pl.col("masked") & pl.col("actual").is_finite() & (pl.col("lead_hours") <= 168.0))
    )


def load_ours(*, arm: str, period: str, cache) -> pl.DataFrame:  # noqa: ANN001
    """Return an arm's scorable rows for a period."""
    pieces = [
        pl.read_parquet(common.out_path(arm=arm, period=period, site=site)).with_columns(
            site=pl.lit(site)
        )
        for site in common.SCORED_SITES
    ]
    return annotate(frame=pl.concat(pieces).select(BASE_COLUMNS), cache=cache)


def load_stored(*, name: str, origins: list, cache) -> pl.DataFrame:  # noqa: ANN001
    """Return a stored experiment 87 arm's scorable rows for the scored year."""
    ids = {s: cache.sites[s].table.time_series_id for s in common.SCORED_SITES}
    label_of = {series: site for site, series in ids.items()}
    frame = pl.read_parquet(E87_PREDICTIONS / STORED_87[name] / "predictions.parquet").filter(
        pl.col("time_series_id").is_in(list(label_of))
    )
    frame = frame.with_columns(
        site=pl.col("time_series_id").replace_strict(label_of, return_dtype=pl.String),
        actual=100.0 * pl.col("actual"),
        **{q: 100.0 * pl.col(q) for q in QUANTILE_COLUMNS},
    ).with_columns(mean=pl.col("q50"))
    frame = frame.select(BASE_COLUMNS).filter(pl.col("power_fcst_init_time").is_in(origins))
    return annotate(frame=frame, cache=cache)


def common_rows(*, frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Restrict every frame to the rows all frames hold."""
    shared = None
    for frame in frames.values():
        current = frame.select(KEYS).unique()
        shared = current if shared is None else shared.join(current, on=KEYS, how="inner")
    assert shared is not None
    schema = {
        "time_series_id": pl.Int32,
        "lead_hours": pl.Float64,
        "seed": pl.Int8,
        "actual": pl.Float32,
        "mean": pl.Float32,
        **dict.fromkeys(QUANTILE_COLUMNS, pl.Float32),
    }
    return {
        name: frame.join(shared, on=KEYS, how="inner")
        .sort(KEYS)
        .with_columns(pl.col(column).cast(dtype) for column, dtype in schema.items())
        for name, frame in frames.items()
    }


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the quantile score, the mean absolute error, and the tail flags and losses."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    quantiles = frame.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64)
    error = actual[:, None] - quantiles
    score = 2.0 * np.mean(
        np.maximum(common.LEVELS[None, :] * error, (common.LEVELS[None, :] - 1.0) * error), axis=1
    )
    extra = {"qs": score, "abs_mean": np.abs(actual - frame["mean"].to_numpy())}
    for name, level in TAIL_QUANTILES:
        value = frame[name].to_numpy().astype(np.float64)
        diff = actual - value
        extra[f"pin_{name}"] = np.maximum(level * diff, (level - 1.0) * diff)
        extra[f"cov_{name}"] = (actual <= value).astype(np.float64)
    return frame.with_columns([pl.Series(k, v) for k, v in extra.items()])


def spell_labels(*, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return, per site and night date holding a cold night, its place in a cold spell.

    A spell is a run of consecutive night dates at one site, each with a cold night (a winter
    night or evening row whose ERA5 day mean is below 2 C). `spell_label` is the date's offset
    from the spell's first date: day 0, days 1 to 3, or later.
    """
    reference = next(iter(frames.values()))
    cold = reference.filter(
        pl.col("valid_month").is_in(list(WINTER_MONTHS))
        & pl.col("cold2")
        & pl.col("hour_band").is_in(list(NIGHT_BANDS))
    )
    dates = cold.select(["site", "night_date"]).unique().sort(["site", "night_date"])
    breaks = (pl.col("night_date").diff().over("site") != dt.timedelta(days=1)).fill_null(True)
    dates = dates.with_columns(spell=breaks.cum_sum().over("site"))
    dates = dates.with_columns(
        offset=(
            pl.col("night_date") - pl.col("night_date").first().over(["site", "spell"])
        ).dt.total_days()
    )
    return dates.with_columns(
        spell_label=pl.when(pl.col("offset") == 0)
        .then(pl.lit(SPELL_LABELS[0]))
        .when(pl.col("offset") <= 3)
        .then(pl.lit(SPELL_LABELS[1]))
        .otherwise(pl.lit(SPELL_LABELS[2]))
    )


def subsets() -> dict[str, pl.Expr]:
    """Return the breakout subsets, each as a row filter."""
    night = pl.col("hour_band").is_in(list(NIGHT_BANDS))
    winter = pl.col("valid_month").is_in(list(WINTER_MONTHS))
    return {
        "winter nights": winter & night,
        COLD_KEY: winter & pl.col("cold2") & night,
        "bank holidays": pl.col("bank_holiday"),
        "Christmas (24 Dec to 1 Jan)": pl.col("christmas"),
    }


COLD_KEY: Final[str] = "cold nights (day mean below 2 C)"


def stack(*, frames: dict[str, pl.DataFrame], spells: pl.DataFrame) -> pl.DataFrame:
    """Return the bootstrap loss frame of arms already on common rows, with the spell label."""
    keep = [
        *KEYS, "lead_hours", "seed", *QUANTILE_COLUMNS, "actual", "mean", "site", "hour_band",
        "cold2", "valid_month", "local_date", "night_date", "bank_holiday", "christmas",
    ]  # fmt: skip
    labels = {
        series: site
        for site in common.SCORED_SITES
        for series in {
            next(iter(frames.values())).filter(pl.col("site") == site)["time_series_id"][0]
        }
    }
    arm_losses = {
        name: with_losses(frame=frame.select(keep).drop("site")) for name, frame in frames.items()
    }
    stacked = losses_for_bootstrap(arm_losses=arm_losses)
    stacked = stacked.with_columns(
        site=pl.col("site").replace_strict(labels, return_dtype=pl.String)
    )
    return stacked.join(
        spells.select(["site", "night_date", "spell_label"]), on=["site", "night_date"], how="left"
    )


def contrast(*, losses: pl.DataFrame, treatment: str, reference: str, metric: str = "qs") -> dict:
    """Return a paired contrast with a 95% month-resampled interval."""
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
    return {"estimate": estimate, "lower": float(lower), "upper": float(upper)}


def decision(*, losses: pl.DataFrame, treatment: str) -> dict:
    """Return the pre-registered pass flags of a treatment arm against arm A in one period."""
    scope = losses.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
    cold = scope.filter(subsets()[COLD_KEY])
    cover = {
        arm: {
            "cold_q95": cold.filter(pl.col("arm") == arm)["cov_q95"].to_numpy().mean().item(),
            "q95": scope.filter(pl.col("arm") == arm)["cov_q95"].to_numpy().mean().item(),
            "q05": scope.filter(pl.col("arm") == arm)["cov_q05"].to_numpy().mean().item(),
        }
        for arm in ("a", treatment)
    }
    c1 = contrast(losses=scope, treatment=treatment, reference="a")
    t, a = cover[treatment], cover["a"]
    flags = {
        "cold_q95_ok": t["cold_q95"] >= PASS_COVERAGE,
        "q95_ok": Q95_RANGE[0] <= t["q95"] <= Q95_RANGE[1],
        "q05_ok": abs(t["q05"] - 0.05) <= abs(a["q05"] - 0.05) + Q05_EXTRA_DISTANCE,
        "score_ok": c1["upper"] <= PASS_UPPER_BOUND,
    }
    return {
        "coverage": cover,
        "c1": c1,
        "flags": flags,
        "passes": all(flags.values()),
        "cold_gain": t["cold_q95"] - a["cold_q95"],
    }


def spell_table(*, losses: pl.DataFrame) -> list[dict]:
    """Return q95 coverage and rows of cold nights by place in the spell, per arm."""
    scope = losses.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
    cold = scope.filter(subsets()[COLD_KEY])
    return (
        cold.group_by(["arm", "spell_label"])
        .agg(
            cov_q95=pl.col("cov_q95").mean(),
            cov_q99=pl.col("cov_q99").mean(),
            rows=pl.len(),
            dates=pl.col("night_date").n_unique(),
        )
        .sort(["arm", "spell_label"])
        .to_dicts()
    )


def tables(*, losses: pl.DataFrame, treatment: str) -> dict:
    """Return the tail table, subset contrasts, special days, and leave-one-site-out figures."""
    scope = losses.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
    aggregations = [pl.col(f"cov_{n}").mean() for n, _ in TAIL_QUANTILES]
    aggregations += [pl.col(f"pin_{n}").mean() for n, _ in TAIL_QUANTILES]
    aggregations += [pl.col("qs").mean(), pl.col("abs_mean").mean(), pl.len().alias("rows")]
    out: dict = {"tail": {}, "subset_contrasts": {}, "leave_one_out": {}}
    parts = {"all rows": scope, **{label: scope.filter(e) for label, e in subsets().items()}}
    for label, part in parts.items():
        out["tail"][label] = part.group_by("arm").agg(aggregations).sort("arm").to_dicts()
        out["subset_contrasts"][label] = {
            "qs": contrast(losses=part, treatment=treatment, reference="a"),
            "cov_q95": contrast(losses=part, treatment=treatment, reference="a", metric="cov_q95"),
            "cov_q05": contrast(losses=part, treatment=treatment, reference="a", metric="cov_q05"),
        }
    cold = scope.filter(subsets()[COLD_KEY])
    for site in common.DECISION_SITES:
        rest = cold.filter(pl.col("site") != site)
        out["leave_one_out"][site] = {
            arm: rest.filter(pl.col("arm") == arm)["cov_q95"].to_numpy().mean().item()
            for arm in ("a", "cw", treatment)
        }
    out["by_site_qs"] = (
        scope.group_by(["arm", "site"]).agg(pl.col("qs").mean()).sort(["arm", "site"]).to_dicts()
    )
    out["spells"] = spell_table(losses=scope)
    out["median_minus_mean"] = (
        scope.group_by(["arm", "site"])
        .agg((pl.col("q50") - pl.col("mean")).mean().alias("median_minus_mean"))
        .sort(["arm", "site"])
        .to_dicts()
    )
    return out


def score_period(
    *,
    period: str,
    cache,  # noqa: ANN001
    arms: list[str],
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Return the arms' common-row frames and the stacked loss frame for a period."""
    frames = common_rows(
        frames={arm: load_ours(arm=arm, period=period, cache=cache) for arm in arms}
    )
    spells = spell_labels(frames=frames)
    return frames, stack(frames=frames, spells=spells)


def choose_gamma(*, sweep: dict[float, dict]) -> float:
    """Return the gamma with the highest cold-night q95 coverage among those under the bound."""
    allowed = {g: d for g, d in sweep.items() if d["c1"]["upper"] <= PASS_UPPER_BOUND}
    if not allowed:
        return min(sweep)
    best = max(d["coverage"][common.gamma_arm(gamma=g)]["cold_q95"] for g, d in allowed.items())
    return min(
        g
        for g, d in allowed.items()
        if d["coverage"][common.gamma_arm(gamma=g)]["cold_q95"] >= best - 1e-12
    )


def main() -> None:
    """Choose gamma on the tuning months, then score the replicate months and the scored year."""
    cache = common.load_cache()
    all_arms = list(common.ARMS)
    report: dict = {}
    _, tuning_losses = score_period(period="tuning", cache=cache, arms=all_arms)
    sweep = {
        g: decision(losses=tuning_losses, treatment=common.gamma_arm(gamma=g))
        for g in common.GAMMAS
    }
    gamma = choose_gamma(sweep=sweep)
    treatment = common.gamma_arm(gamma=gamma)
    report["gamma"] = gamma
    report["tuning_sweep"] = {str(g): d for g, d in sweep.items()}
    print("chosen gamma", gamma, flush=True)
    for period in ("replicate", "scored"):
        frames, losses = score_period(period=period, cache=cache, arms=all_arms)
        entry: dict = {
            "rows": frames["a"].height,
            "decision": decision(losses=losses, treatment=treatment),
            "gamma_sweep": {
                str(g): decision(losses=losses, treatment=common.gamma_arm(gamma=g))
                for g in common.GAMMAS
            },
            "cw_decision": decision(losses=losses, treatment="cw"),
            "tables": tables(losses=losses, treatment=treatment),
            "aci_minus_cw_cold_q95": contrast(
                losses=losses.filter(
                    pl.col("site").is_in(list(common.DECISION_SITES)) & subsets()[COLD_KEY]
                ),
                treatment=treatment,
                reference="cw",
                metric="cov_q95",
            ),
        }
        if period == "scored":
            origins = frames["a"]["power_fcst_init_time"].unique().to_list()
            loaded = {"a": frames["a"], treatment: frames[treatment], "cw": frames["cw"]}
            for name in STORED_87:
                loaded[name] = load_stored(name=name, origins=origins, cache=cache)
            shared = common_rows(frames=loaded)
            shared_losses = stack(frames=shared, spells=spell_labels(frames=shared))
            scope = shared_losses.filter(pl.col("site").is_in(list(common.DECISION_SITES)))
            entry["absolute"] = (
                scope.group_by("arm").agg(pl.col("qs").mean()).sort("arm").to_dicts()
            )
            entry["absolute_rows"] = shared["a"].height
            a_score = (
                frames["a"]
                .pipe(lambda f: with_losses(frame=f))
                .filter(pl.col("site").is_in(list(common.DECISION_SITES)))["qs"]
                .to_numpy()
                .mean()
                .item()
            )
            entry["reproduction"] = {
                "arm_a_score": a_score,
                "printed": ARM_A_SCORE,
                "gap": a_score - ARM_A_SCORE,
            }
        report[period] = entry
        print(period, "done", flush=True)
    scored, replicate = report["scored"]["decision"], report["replicate"]["decision"]
    if abs(report["scored"]["reproduction"]["gap"]) > 0.001:
        verdict = "inconclusive"
    elif scored["passes"] and replicate["passes"]:
        verdict = "supported"
    elif scored["passes"]:
        verdict = "partly supported"
    elif scored["cold_gain"] < 0.01:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    report["verdict"] = verdict
    common.write_json_atomic(path=common.REPORT_DIR / "report.json", content=report)
    print("verdict", verdict)


if __name__ == "__main__":
    main()
