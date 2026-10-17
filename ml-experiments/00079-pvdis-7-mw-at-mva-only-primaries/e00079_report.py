"""Compute the planned contrasts, the verdict, and the real-series summaries; write `report.txt`.

POLARS_MAX_THREADS=4 uv run python \
        ml-experiments/00079-pvdis-7-mw-at-mva-only-primaries/e00079_report.py
"""

from collections.abc import Callable

import e00079_common as common
import numpy as np
import polars as pl

MIN_PRIMARIES: int = 6


def two_stage_draws(
    *, sites: np.ndarray, statistic: Callable[[np.ndarray, np.ndarray], float]
) -> tuple[float, float, float]:
    """Return a statistic's point value and its interval from a two-stage cluster bootstrap.

    The first stage resamples the Primaries, the second resamples the windows within each drawn
    Primary. `statistic` takes the row indices of the resampled windows and the matching drawn
    Primary label (so that a Primary drawn twice counts as two).

    Args:
        sites: The Primary of each window row.
        statistic: Maps (row indices, draw labels) to a number.

    Returns:
        The point value on all rows, and the lower and upper limits at `common.LEVEL`.
    """
    rng = np.random.default_rng(common.BOOTSTRAP_SEED)
    labels = np.unique(sites)
    rows_of = {s: np.flatnonzero(sites == s) for s in labels}
    point = statistic(np.arange(len(sites)), sites)
    draws = np.empty(common.BOOTSTRAP_DRAWS)
    for d in range(common.BOOTSTRAP_DRAWS):
        picked = rng.choice(labels, size=len(labels), replace=True)
        index, owner = [], []
        for k, s in enumerate(picked):
            r = rng.choice(rows_of[s], size=len(rows_of[s]), replace=True)
            index.append(r)
            owner.append(np.full(len(r), k))
        draws[d] = statistic(np.concatenate(index), np.concatenate(owner))
    tail = (1.0 - common.LEVEL) / 2.0
    low, high = np.nanquantile(draws, [tail, 1.0 - tail])
    return float(point), float(low), float(high)


def as_float(value: object) -> float:
    """Return a Polars scalar (typed as a wide union) as a float."""
    return float(value)  # ty: ignore[invalid-argument-type]


def injected(*, known: pl.DataFrame, level: float, q_share: float = 0.0) -> pl.DataFrame:
    """Return the windows of one injection level with their relative capacity errors."""
    return known.filter(
        (pl.col("level") == level) & (pl.col("q_share") == q_share) & (pl.col("farm") != "-")
    ).with_columns(
        err_unfold=(pl.col("c_unfold") - pl.col("c_signed")).abs() / pl.col("c_signed"),
        err_naive=(pl.col("c_naive") - pl.col("c_signed")).abs() / pl.col("c_signed"),
        err_unfold_injected=(
            (pl.col("c_unfold") - pl.col("c_base_signed")) / pl.col("injected_peak_mw") - 1.0
        ).abs(),
    )


def capacity_contrast(*, frame: pl.DataFrame) -> tuple[float, float, float]:
    """Return C1: median error of the unfolding fit minus that of the naive fit, with interval."""
    eu, en = frame["err_unfold"].to_numpy(), frame["err_naive"].to_numpy()
    return two_stage_draws(
        sites=frame["site"].to_numpy(),
        statistic=lambda i, _o: float(np.median(eu[i]) - np.median(en[i])),
    )


def sign_eligible(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the windows with enough reverse-flow and forward-flow rows for a sign score."""
    return frame.filter(
        (pl.col("truth_negative") >= common.MIN_CLASS_ROWS)
        & (pl.col("truth_positive") >= common.MIN_CLASS_ROWS)
    )


def balanced_accuracy_stat(*, frame: pl.DataFrame, which: str) -> tuple[float, float, float]:
    """Return the Primary-averaged pooled balanced accuracy minus 0.5, with interval."""
    tp, tn = frame[f"tp_{which}"].to_numpy(), frame[f"tn_{which}"].to_numpy()
    neg, pos = frame["truth_negative"].to_numpy(), frame["truth_positive"].to_numpy()

    def stat(index: np.ndarray, owner: np.ndarray) -> float:
        scores = []
        for o in np.unique(owner):
            r = index[owner == o]
            scores.append(0.5 * (tp[r].sum() / neg[r].sum() + tn[r].sum() / pos[r].sum()))
        return float(np.mean(scores)) - 0.5

    return two_stage_draws(sites=frame["site"].to_numpy(), statistic=stat)


def main() -> None:  # noqa: PLR0915
    """Write the report."""
    known = pl.read_parquet(common.KNOWN_PATH)
    lines: list[str] = []
    out = lines.append
    verdict_parts: dict[str, bool] = {}
    pooled_ba: dict[float, tuple[float, float, float]] = {}
    for level in (1.0, 1.5):
        frame = injected(known=known, level=level)
        c1 = capacity_contrast(frame=frame)
        out(f"[C1 level {level}] windows={frame.height} primaries={frame['site'].n_unique()}")
        out(
            f"  median relative capacity error: unfold={frame['err_unfold'].median():.3f} "
            f"naive={frame['err_naive'].median():.3f}; vs injected peak, unfold="
            f"{frame['err_unfold_injected'].median():.3f}"
        )
        out(f"  C1 (unfold - naive) = {c1[0]:+.3f}  [{c1[1]:+.3f}, {c1[2]:+.3f}]")
        eligible = sign_eligible(frame=frame)
        out(
            f"  windows with enough reverse flow for a sign score: {eligible.height} of "
            f"{frame.height}; dropped {frame.height - eligible.height}"
        )
        ba = balanced_accuracy_stat(frame=eligible, which="unfold")
        ba_naive = balanced_accuracy_stat(frame=eligible, which="naive")
        pooled_ba[level] = ba
        out(
            f"  C2 balanced accuracy - 0.5: unfold={ba[0]:+.3f} [{ba[1]:+.3f}, {ba[2]:+.3f}]; "
            f"naive point={ba_naive[0]:+.3f}"
        )
        capacity_ok = (
            as_float(frame["err_unfold"].median()) <= common.CAPACITY_ERROR_LIMIT and c1[2] < 0.0
        )
        direction_ok = (ba[0] + 0.5) >= common.BALANCED_ACCURACY_LIMIT and ba[
            1
        ] >= common.C2_LOWER_LIMIT
        verdict_parts[f"capacity_{level}"] = bool(capacity_ok)
        verdict_parts[f"direction_{level}"] = bool(direction_ok)
    for q in common.REACTIVE_SHARES:
        frame = injected(known=known, level=1.0, q_share=q)
        c3 = capacity_contrast(frame=frame)
        eligible = sign_eligible(frame=frame)
        ba = balanced_accuracy_stat(frame=eligible, which="unfold")
        out(
            f"[reactive q={q}] unfold err={frame['err_unfold'].median():.3f} "
            f"naive err={frame['err_naive'].median():.3f}; C1-style={c3[0]:+.3f} "
            f"[{c3[1]:+.3f}, {c3[2]:+.3f}]; BA-0.5={ba[0]:+.3f} [{ba[1]:+.3f}, {ba[2]:+.3f}]"
        )
    frame = injected(known=known, level=0.5)
    out(
        f"[level 0.5, no-signal] unfold err={frame['err_unfold'].median():.3f} "
        f"naive err={frame['err_naive'].median():.3f}; fitted negative share median "
        f"{frame['fitted_negative_share'].median():.4f}"
    )
    base = known.filter(
        (pl.col("level") == 0.0) & pl.col("site").is_in(common.ALMOST_NEVER_NEGATIVE)
    )
    false_share = as_float(base["fitted_negative_share"].median())
    out(
        f"[null check] almost-never-negative Primaries, no injection: windows={base.height}, "
        f"median fitted negative share={false_share:.4f}"
    )
    natural = sign_eligible(frame=known.filter(pl.col("level") == 0.0))
    out(
        f"[natural reverse flow] eligible windows: {natural.height}: "
        f"{natural.select('site', 'year', 'truth_negative').rows()}"
    )
    distinct = known.filter(pl.col("level") > 0)["n_distinct"].value_counts().sort("n_distinct")
    out(
        f"[starts] distinct capacities reached: {distinct.rows()}; median loss gap "
        f"{known['loss_gap'].median():.4g}"
    )
    for level in (1.0, 1.5):
        rest = sign_eligible(frame=injected(known=known, level=level)).filter(
            pl.col("site") != "D15"
        )
        ba = balanced_accuracy_stat(frame=rest, which="unfold")
        out(
            f"[post hoc, level {level}, without D15] BA={ba[0] + 0.5:.3f} "
            f"[{ba[1] + 0.5:.3f}, {ba[2] + 0.5:.3f}] over {rest['site'].n_unique()} Primaries"
        )
        pooled = injected(known=known, level=level).filter(pl.col("site") != "D15")
        recall = as_float(pooled["tp_unfold"].sum()) / as_float(pooled["truth_negative"].sum())
        specificity = as_float(pooled["tn_unfold"].sum()) / as_float(pooled["truth_positive"].sum())
        out(f"  pooled recall of reverse flow={recall:.3f}, specificity={specificity:.3f}")
    d15 = injected(known=known, level=1.0).filter(pl.col("site") == "D15")
    out(
        f"[post hoc] D15 level 1.0: median signed-fit capacity={d15['c_signed'].median():.2f}, "
        f"unfold={d15['c_unfold'].median():.2f}"
    )
    capacity = all(verdict_parts[f"capacity_{lv}"] for lv in (1.0, 1.5))
    direction = (
        all(verdict_parts[f"direction_{lv}"] for lv in (1.0, 1.5))
        and false_share <= common.FALSE_REVERSE_LIMIT
    )
    primaries = known["site"].n_unique()
    if primaries < MIN_PRIMARIES:
        verdict = "inconclusive"
    elif capacity and direction:
        verdict = "supported"
    elif capacity or direction:
        verdict = "partly supported"
    else:
        verdict = "refuted"
    out(f"[verdict] capacity identified={capacity}, direction identified={direction} -> {verdict}")
    out(str(verdict_parts))
    real_lines = real_summary()
    text = "\n".join([*lines, "", *real_lines]) + "\n"
    common.REPORT_PATH.write_text(text)
    print(text)


def real_summary() -> list[str]:
    """Summarise the real-series fits per Primary, with the week-block bootstrap limits."""
    point = pl.read_parquet(common.REAL_PATH)
    boot = pl.read_parquet(common.REAL_BOOT_PATH)
    lines = ["[real series, per Primary: median over season-years of the point fits]"]
    summary = (
        point.group_by("site", "units")
        .agg(
            pl.len().alias("windows"),
            pl.col("c_naive").median(),
            pl.col("c_unfold").median(),
            pl.col("negative_share").median(),
            pl.col("loss_reduction").median(),
            pl.col("n_distinct").median().alias("starts_distinct"),
        )
        .sort("units", "site")
    )
    limits = boot.group_by("site").agg(
        pl.col("negative_share").quantile(0.05).alias("neg_p05"),
        pl.col("negative_share").quantile(0.95).alias("neg_p95"),
        pl.col("c_unfold").quantile(0.05).alias("c_unfold_p05"),
        pl.col("c_unfold").quantile(0.95).alias("c_unfold_p95"),
    )
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    lines.append(str(summary.join(limits, on="site").sort("units", "site")))
    return lines


if __name__ == "__main__":
    main()
