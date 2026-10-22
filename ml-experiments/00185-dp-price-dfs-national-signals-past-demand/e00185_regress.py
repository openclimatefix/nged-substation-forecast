"""Experiment 185, Parts 1 and 2: do national signals explain arm D's daily loss?

Reads the day tables of `e00185_days.py` and the signal table of `e00185_signals.py`. Writes
`regress_report.json` and the tables the charts read. Part 1 asks for the out-of-fold R-squared
increment of price and the Demand Flexibility Service (DFS) flag over a baseline of site effects,
a spline of the day-mean temperature, and day type. Part 2 compares the other national signals.
"""

import json
from datetime import date
from typing import Final

import e00185_common as common
import numpy as np
import polars as pl

SEED: Final[int] = 185
N_BOOT: Final[int] = 2000
N_DRAWS: Final[int] = 1000
KNOT_QUANTILES: Final[tuple[float, ...]] = (0.05, 0.275, 0.5, 0.725, 0.95)
PRICE_SET: Final[tuple[str, ...]] = ("price_mean", "price_max", "price_eve")
PRICE_PAIR: Final[tuple[str, ...]] = ("price_mean", "price_eve")
SIGNALS: Final[dict[str, tuple[str, ...]]] = {
    "price (mean, evening)": PRICE_PAIR,
    "national demand": ("nd_mean", "nd_eve"),
    "embedded wind": ("wind_emb_mean", "wind_emb_eve"),
    "transmission wind": ("wind_tx_mean", "wind_tx_eve"),
    "embedded solar": ("solar_emb_mean", "solar_emb_eve"),
    "net interconnector flow": ("ic_net_mean", "ic_net_eve"),
}


def rcs_basis(*, x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Return Harrell's restricted cubic spline basis: the linear term and k - 2 curved terms."""
    k = len(knots)
    t = knots
    cols = [x]
    denom = (t[-1] - t[-2]) * (t[-1] - t[0]) ** 2
    for j in range(k - 2):
        term = (
            np.maximum(x - t[j], 0) ** 3
            - np.maximum(x - t[-2], 0) ** 3 * (t[-1] - t[j]) / (t[-1] - t[-2])
            + np.maximum(x - t[-1], 0) ** 3 * (t[-2] - t[j]) / (t[-1] - t[-2])
        )
        cols.append(term / (t[-1] - t[0]) ** 2 if denom else term)
    return np.column_stack(cols)


def design(
    *, frame: pl.DataFrame, signals: tuple[str, ...], interact: bool, dfs: bool
) -> np.ndarray:
    """Return the baseline columns, then the signals, then their temperature products."""
    temp = frame["day_temperature_c"].to_numpy()
    knots = np.quantile(temp, KNOT_QUANTILES)
    columns = [rcs_basis(x=temp, knots=knots)]
    columns += [(frame["site"] == site).to_numpy().astype(float)[:, None] for site in common.SITES]
    columns += [
        (frame["day_type"] == kind).to_numpy().astype(float)[:, None]
        for kind in ("saturday", "sunday", "holiday")
    ]
    extra = [frame[c].to_numpy().astype(float) for c in signals]
    if dfs:
        extra.append(frame["dfs"].to_numpy().astype(float))
    if extra:
        z = np.column_stack(extra)
        z = (z - z.mean(axis=0)) / np.where(z.std(axis=0) > 0, z.std(axis=0), 1.0)
        columns.append(z)
        if interact:
            tz = (temp - temp.mean()) / temp.std()
            columns.append(z * tz[:, None])
    return np.column_stack(columns)


def ols(*, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return least-squares coefficients with a tiny ridge for numerical safety."""
    gram = x.T @ x + 1e-8 * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def out_of_fold(*, x: np.ndarray, y: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Return each row's prediction from a fit that leaves out the row's month block."""
    pred = np.zeros_like(y)
    for b in np.unique(block):
        held = block == b
        beta = ols(x=np.column_stack([np.ones((~held).sum()), x[~held]]), y=y[~held])
        pred[held] = np.column_stack([np.ones(held.sum()), x[held]]) @ beta
    return pred


def r2(*, y: np.ndarray, pred: np.ndarray) -> float:
    """Return one minus the squared error over the squared deviation from the overall mean."""
    return float(1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def block_interval(
    *,
    y: np.ndarray,
    base: np.ndarray,
    full: np.ndarray,
    block: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return the increment of R-squared and its 95% month-block bootstrap interval."""
    blocks = np.unique(block)
    index = {b: np.flatnonzero(block == b) for b in blocks}
    draws = []
    for _ in range(N_BOOT):
        picks = np.concatenate([index[b] for b in rng.choice(blocks, size=len(blocks))])
        draws.append(r2(y=y[picks], pred=full[picks]) - r2(y=y[picks], pred=base[picks]))
    return (
        r2(y=y, pred=full) - r2(y=y, pred=base),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def increment(
    *,
    frame: pl.DataFrame,
    outcome: str,
    signals: tuple[str, ...],
    interact: bool,
    dfs: bool,
    rng: np.random.Generator,
) -> dict:
    """Return the out-of-fold increment of a signal set over the baseline, with its interval."""
    y = frame[outcome].to_numpy().astype(float)
    block = frame["year_month"].to_numpy()
    base = out_of_fold(
        x=design(frame=frame, signals=(), interact=False, dfs=False), y=y, block=block
    )
    full = out_of_fold(
        x=design(frame=frame, signals=signals, interact=interact, dfs=dfs), y=y, block=block
    )
    estimate, low, high = block_interval(y=y, base=base, full=full, block=block, rng=rng)
    fitted = {}
    for key, matrix in (
        ("base", design(frame=frame, signals=(), interact=False, dfs=False)),
        ("full", design(frame=frame, signals=signals, interact=interact, dfs=dfs)),
    ):
        x = np.column_stack([np.ones(len(y)), matrix])
        fitted[key] = r2(y=y, pred=x @ ols(x=x, y=y))
    return {
        "in_sample_increment": fitted["full"] - fitted["base"],
        "n_days": len(y),
        "baseline_oof_r2": r2(y=y, pred=base),
        "full_oof_r2": r2(y=y, pred=full),
        "increment": estimate,
        "ci95": [low, high],
        "_base": base,
        "_full": full,
    }


def strip(*, result: dict) -> dict:
    """Drop the private prediction arrays before writing JSON."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def paired_difference(
    *,
    y: np.ndarray,
    a: dict,
    b: dict,
    base: np.ndarray,
    block: np.ndarray,
    rng: np.random.Generator,
) -> list[float]:
    """Return the increment of `a` minus that of `b`, with a month-block 95% interval."""
    blocks = np.unique(block)
    index = {k: np.flatnonzero(block == k) for k in blocks}
    draws = []
    for _ in range(N_BOOT):
        picks = np.concatenate([index[k] for k in rng.choice(blocks, size=len(blocks))])
        ia = r2(y=y[picks], pred=a["_full"][picks]) - r2(y=y[picks], pred=base[picks])
        ib = r2(y=y[picks], pred=b["_full"][picks]) - r2(y=y[picks], pred=base[picks])
        draws.append(ia - ib)
    est = (r2(y=y, pred=a["_full"]) - r2(y=y, pred=base)) - (
        r2(y=y, pred=b["_full"]) - r2(y=y, pred=base)
    )
    return [est, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def coefficient_intervals(
    *,
    frame: pl.DataFrame,
    outcome: str,
    signals: tuple[str, ...],
    dfs: bool,
    rng: np.random.Generator,
) -> dict:
    """Return the effect of one standard deviation of each signal (no temperature product)."""
    y = frame[outcome].to_numpy().astype(float)
    block = frame["year_month"].to_numpy()
    x = design(frame=frame, signals=signals, interact=False, dfs=dfs)
    n_extra = len(signals) + int(dfs)
    names = [*signals, *(["dfs"] if dfs else [])]
    sd = [frame[c].to_numpy().astype(float).std() for c in names]

    def effects(rows: np.ndarray) -> np.ndarray:
        beta = ols(x=np.column_stack([np.ones(len(rows)), x[rows]]), y=y[rows])
        return beta[-n_extra:]

    point = effects(np.arange(len(y)))
    blocks = np.unique(block)
    index = {k: np.flatnonzero(block == k) for k in blocks}
    draws = np.array(
        [
            effects(np.concatenate([index[k] for k in rng.choice(blocks, size=len(blocks))]))
            for _ in range(N_BOOT)
        ]
    )
    return {
        name: {
            "per_sd": float(point[i]),
            "sd_of_signal": float(sd[i]),
            "ci95": [
                float(np.percentile(draws[:, i], 2.5)),
                float(np.percentile(draws[:, i], 97.5)),
            ],
        }
        for i, name in enumerate(names)
    }


def matched_draws(
    *,
    days: pl.DataFrame,
    targets: pl.DataFrame,
    worst: pl.DataFrame,
    indicators: list[str],
    rng: np.random.Generator,
) -> dict:
    """Compare the target days with 1,000 draws of matched ordinary days."""
    season = (pl.col("local_date").dt.month() % 12) // 3
    keyed = (
        days.with_columns(weekday=pl.col("local_date").dt.weekday(), season=season)
        .join(
            worst.select("site", "local_date").with_columns(is_worst=pl.lit(True)),
            on=["site", "local_date"],
            how="left",
        )
        .with_columns(is_worst=pl.col("is_worst").fill_null(False))
    )
    assert targets.join(worst, on=["site", "local_date"], how="anti").height == 0
    pool = keyed.filter(~pl.col("is_worst"))
    tgt = targets.join(keyed.drop("is_worst"), on=["site", "local_date"], how="left")
    candidates, widened = [], 0
    for row in tgt.iter_rows(named=True):
        base = pool.filter(
            pl.col("site") == row["site"],
            pl.col("weekday") == row["weekday"],
            pl.col("season") == row["season"],
        )
        for width in (2.0, 4.0):
            near = base.filter(
                (pl.col("day_temperature_c") - row["day_temperature_c"]).abs() <= width
            )
            if near.height:
                widened += int(width > 2.0)
                break
        else:
            near = pool.filter(pl.col("site") == row["site"], pl.col("season") == row["season"])
            widened += 1
        candidates.append(near)
    result: dict = {"_targets_widened_or_fallback": widened, "_n_targets": tgt.height}
    for ind in indicators:
        # DFS has no record outside its archive, so only covered targets and covered days count.
        keep = [
            i
            for i, row in enumerate(tgt.iter_rows(named=True))
            if ind != "p1" or row["dfs_covered"]
        ]
        drawn = [
            candidates[i].filter(pl.col("dfs_covered")) if ind == "p1" else candidates[i]
            for i in keep
        ]
        keep = [i for i, c in zip(keep, drawn, strict=True) if c.height]
        drawn = [c for c in drawn if c.height]
        observed = int(tgt[ind][keep].sum())
        counts = np.array(
            [sum(int(c[ind][int(rng.integers(c.height))]) for c in drawn) for _ in range(N_DRAWS)]
        )
        result[ind] = {
            "n_targets": len(keep),
            "observed_of_37": observed,
            "ordinary_mean": float(counts.mean()),
            "ordinary_95pct_range": [
                float(np.percentile(counts, 2.5)),
                float(np.percentile(counts, 97.5)),
            ],
            "excess_days": float(observed - counts.mean()),
            "one_sided_p": float(((counts >= observed).sum() + 1) / (N_DRAWS + 1)),
        }
    return result


def main() -> None:  # noqa: PLR0915
    """Run Parts 1 and 2 and write the report."""
    rng = np.random.default_rng(SEED)
    signals = pl.read_parquet(common.WORK_DIR / "national_signals.parquet")
    days = pl.read_parquet(common.WORK_DIR / "days_scored.parquet").join(
        signals, on="local_date", how="left"
    )
    assert days["price_mean"].null_count() == 0, "a scored day has no price"
    assert signals.drop_nulls(subset=["nd_mean", "nd_eve"]).height == signals.height
    covered = days.filter(pl.col("dfs_covered"))
    report: dict = {"n_scored_days": days.height, "n_dfs_covered_days": covered.height}

    # Part 1: the pre-registered models.
    part1 = {}
    for outcome in ("excess", "signed_error"):
        part1[outcome] = {
            "price_three_summaries_x_temperature_all_days": strip(
                result=increment(
                    frame=days,
                    outcome=outcome,
                    signals=PRICE_SET,
                    interact=True,
                    dfs=False,
                    rng=rng,
                )
            ),
            "price_three_summaries_no_interaction_all_days": strip(
                result=increment(
                    frame=days,
                    outcome=outcome,
                    signals=PRICE_SET,
                    interact=False,
                    dfs=False,
                    rng=rng,
                )
            ),
            "price_plus_dfs_x_temperature_dfs_covered_days": strip(
                result=increment(
                    frame=covered,
                    outcome=outcome,
                    signals=PRICE_SET,
                    interact=True,
                    dfs=True,
                    rng=rng,
                )
            ),
            "price_only_x_temperature_dfs_covered_days": strip(
                result=increment(
                    frame=covered,
                    outcome=outcome,
                    signals=PRICE_SET,
                    interact=True,
                    dfs=False,
                    rng=rng,
                )
            ),
            "dfs_only_dfs_covered_days": strip(
                result=increment(
                    frame=covered, outcome=outcome, signals=(), interact=False, dfs=True, rng=rng
                )
            ),
        }
    report["part1"] = part1

    # Per-site increments of the full price + DFS model on covered days.
    per_site = {}
    for site in common.SITES:
        frame = covered.filter(pl.col("site") == site)
        per_site[site] = strip(
            result=increment(
                frame=frame, outcome="excess", signals=PRICE_SET, interact=False, dfs=True, rng=rng
            )
        )
    report["part1_per_site_excess"] = per_site

    # Effect sizes (no temperature product), month-block intervals.
    report["effects_excess_per_sd"] = coefficient_intervals(
        frame=covered, outcome="excess", signals=PRICE_SET, dfs=True, rng=rng
    )
    report["effects_signed_error_per_sd"] = coefficient_intervals(
        frame=covered, outcome="signed_error", signals=PRICE_SET, dfs=True, rng=rng
    )
    # Sensitivity (post hoc): rank-transformed price, which is heavy-tailed.
    ranked = days.with_columns(*[pl.col(c).rank().over(pl.lit(1)).alias(c) for c in PRICE_SET])
    report["sensitivity_ranked_price_excess"] = strip(
        result=increment(
            frame=ranked, outcome="excess", signals=PRICE_SET, interact=True, dfs=False, rng=rng
        )
    )

    # Matched ordinary days: the 37 unlabelled worst days of experiment 175.
    e177_price = float(
        pl.concat(  # ty: ignore[invalid-argument-type]
            [
                pl.read_parquet(f)
                for f in sorted((common.E177_PUBLIC / "system_prices").glob("*.parquet"))
            ]
        )
        .with_columns(price=pl.max_horizontal("systemSellPrice", "systemBuyPrice"))
        .filter(
            pl.col("settlementDate").str.to_date().is_between(date(2025, 7, 3), date(2026, 6, 30))
        )["price"]
        .quantile(0.95)
    )
    prices = pl.concat(
        [
            pl.read_parquet(f)
            for f in sorted((common.E177_PUBLIC / "system_prices").glob("*.parquet"))
        ]
    ).with_columns(price=pl.max_horizontal("systemSellPrice", "systemBuyPrice"))
    spike = prices.group_by(pl.col("settlementDate").str.to_date().alias("local_date")).agg(
        p2=(pl.col("price") >= e177_price).any()
    )
    flagged_days = days.join(spike, on="local_date", how="left").with_columns(
        p2=pl.col("p2").fill_null(False), p1=pl.col("dfs")
    )
    targets = pl.read_parquet(common.E175_WORK / "day_shapes.parquet").select("site", "local_date")
    worst = pl.read_parquet(common.E172_WORK / "worst_days.parquet")
    report["matched"] = matched_draws(
        days=flagged_days, targets=targets, worst=worst, indicators=["p2", "p1"], rng=rng
    )
    report["matched"]["_p95_price_threshold"] = e177_price
    report["matched"]["_p1_note"] = "Days after 2026-03-27 have no DFS record; counted as no event."

    # Post hoc: how the heavy-tailed outcome is handled, and the worst-5% rate by indicator.
    logged = days.with_columns(log_score=pl.col("score").log())
    ranked_log = logged.with_columns(*[pl.col(c).rank().alias(c) for c in PRICE_SET])
    report["sensitivity_log_score_ranked_price"] = strip(
        result=increment(
            frame=ranked_log,
            outcome="log_score",
            signals=PRICE_SET,
            interact=True,
            dfs=False,
            rng=rng,
        )
    )
    report["sensitivity_log_score_dfs_covered"] = strip(
        result=increment(
            frame=ranked_log.filter(pl.col("dfs_covered")),
            outcome="log_score",
            signals=PRICE_SET,
            interact=True,
            dfs=True,
            rng=rng,
        )
    )
    worst_keys = worst.select("site", "local_date").with_columns(worst5=pl.lit(True))
    tail = flagged_days.join(worst_keys, on=["site", "local_date"], how="left").with_columns(
        worst5=pl.col("worst5").fill_null(False)
    )
    rates = {}
    for ind, frame in (("p2", tail), ("p1", tail.filter(pl.col("dfs_covered")))):
        on, off = frame.filter(pl.col(ind)), frame.filter(~pl.col(ind))
        rates[ind] = {
            "days_with_indicator": on.height,
            "worst5_rate_with": float(on["worst5"].mean()),  # ty: ignore[invalid-argument-type]
            "worst5_rate_without": float(off["worst5"].mean()),  # ty: ignore[invalid-argument-type]
        }
    report["worst5_rate_by_indicator"] = rates

    # Part 2: each national signal alone, on all days (price pair for the paired difference).
    part2 = {}
    results = {}
    for name, columns in SIGNALS.items():
        results[name] = increment(
            frame=days, outcome="excess", signals=columns, interact=True, dfs=False, rng=rng
        )
        part2[name] = strip(result=results[name])
    y = days["excess"].to_numpy().astype(float)
    block = days["year_month"].to_numpy()
    base = results["price (mean, evening)"]["_base"]
    for name in SIGNALS:
        if name != "price (mean, evening)":
            part2[name]["minus_price_increment"] = paired_difference(
                y=y,
                a=results[name],
                b=results["price (mean, evening)"],
                base=base,
                block=block,
                rng=rng,
            )
    report["part2_excess"] = part2

    # Tuning months: choose the Part 3 signal.
    tuning = pl.read_parquet(common.WORK_DIR / "days_tuning.parquet").join(
        signals, on="local_date", how="left"
    )
    chosen = {}
    for name, columns in SIGNALS.items():
        chosen[name] = strip(
            result=increment(
                frame=tuning, outcome="excess", signals=columns, interact=True, dfs=False, rng=rng
            )
        )
    chosen["dfs flag"] = strip(
        result=increment(
            frame=tuning, outcome="excess", signals=(), interact=False, dfs=True, rng=rng
        )
    )
    report["tuning_months_excess"] = chosen
    report["tuning_best_signal"] = max(chosen, key=lambda k: chosen[k]["increment"])

    out = common.WORK_DIR / "regress_report.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
