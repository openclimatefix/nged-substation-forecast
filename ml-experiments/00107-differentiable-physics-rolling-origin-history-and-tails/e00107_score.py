"""Score experiment 107: per-origin contrasts and tie bands, pooled intervals, labels, and tails.

Reads `work/scored/` (from `e00107_assemble.py`) and `work/decisions.json` (from
`e00107_decisions.py`, which must have been written first). Prints the report that the write-up
quotes, and saves its tables to `work/tables/` for the charts.

**Metric.** The quantile score is twice the mean pinball loss over 11 levels, in points of effective
capacity, on rows outside the hand-filtered switching masks at the scored site, over the five
decision sites. Every contrast is treatment minus reference, so a negative value favours the
treatment.

**Tie bands.** For each contrast and origin, a week-block bootstrap inside the scored window (13
one-week blocks, 1,000 draws, paired, resampling the five decision sites together) gives a standard
error and a 95% interval. An estimate within one standard error of zero is a tie. The band is
written to `tie_bands.parquet` and its hash to `decisions.json` before any sign table is computed.
For C4 the band is at least twice the XGBoost seed floor.

    uv run python ml-experiments/00107-*/e00107_score.py [--variant base|d10_masked]
"""

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import e00107_common as common
import numpy as np
import polars as pl
from scipy import stats
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_gam import QUANTILE_COLUMNS

DECISIONS_PATH: Final = common.WORK_DIR / "decisions.json"
TABLE_DIR: Final = common.WORK_DIR / "tables"
N_BLOCK_DRAWS: Final[int] = 1000
N_WEEK_BLOCKS: Final[int] = 13
N_POOLED_RESAMPLES: Final[int] = 4000
BONFERRONI_LEVEL: Final[float] = 99.0
"""A 95% familywise level over the five planned contrasts: 95 / 5 = 1% per contrast."""
TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
LOSO_LABEL_CONTRASTS: Final[frozenset[str]] = frozenset({"C1", "C2", "C3", "C4"})
"""The planned contrasts whose label is recomputed with each decision site left out."""
WINTER_ROW_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
"""The row season of Parts 2 and 4: November to March against April to October."""


@dataclass(frozen=True)
class Contrast:
    """One treatment-minus-reference contrast.

    Attributes:
        name: A short id such as `C1`.
        treatment: The treatment's `<arm>__<spread>` name.
        reference: The reference's name.
        planned: True for a planned contrast, False for an exploratory one.
        description: What the contrast asks.
        floor: The seed floor to double into the tie band, or 0.
    """

    name: str
    treatment: str
    reference: str
    planned: bool
    description: str
    floor: float = 0.0


def planned_contrasts(*, bar: str, seed_floor: float) -> list[Contrast]:
    """Return the five planned contrasts (C5 is scored apart, on coverage)."""
    return [
        Contrast("C1", "g_all__s0", "g1__s0", True, "old years without drift: G-all - G1"),
        Contrast("C2", "g_all_d1__s0", "g_all__s0", True, "drift on all years: G-all+D1 - G-all"),
        Contrast("C3", "g_all_d1__s0", "g1__s0", True, "more history with drift: G-all+D1 - G1"),
        Contrast("C4", "g1__s0", f"{bar}__own", True, "G1 - XGBoost bar", floor=seed_floor),
        Contrast(
            "C4b", "g_all_d1__s0", f"{bar}__own", True, "G-all+D1 - XGBoost bar", floor=seed_floor
        ),
        Contrast(
            "C5q", "g_all_d1__s1", "g_all_d1__s0", True, "S1 - S0 on G-all+D1 (quantile score)"
        ),
    ]


def exploratory_contrasts(*, bar: str) -> list[Contrast]:
    """Return the exploratory contrasts: replication, controls, and tail-model variants."""
    return [
        Contrast("E1", "g_all_fc__s0", "g1_fc__s0", False, "fitted capacity: G-all - G1"),
        Contrast(
            "E2", "g_all_d1_fc__s0", "g_all_fc__s0", False, "fitted capacity: G-all+D1 - G-all"
        ),
        Contrast("E3", "g_all_d1_fc__s0", "g1_fc__s0", False, "fitted capacity: G-all+D1 - G1"),
        Contrast("E4", "g_all_d1_s10k__s0", "g_all_d1__s0", False, "drift step 10,000 - 3,000"),
        Contrast("K1", "g1_noharm__s0", "g1__s0", False, "calendar control: G1 without harmonics"),
        Contrast(
            "K2",
            "g_all_d1_noharm__s0",
            "g_all_d1__s0",
            False,
            "calendar control: G-all+D1 w/o harmonics",
        ),
        Contrast(
            "K3", "x_12__own", "x_12_nodoy__own", False, "calendar control: X-12 - X-12-nodoy"
        ),
        Contrast("E5", "x_12amq__own", "x_12__own", False, "X-12amq - X-12 (11 origins)"),
        Contrast("T1", "g_all_d1__s2", "g_all_d1__s0", False, "S2 - S0 on G-all+D1"),
        Contrast("T2", "g1__s1", "g1__s0", False, "S1 - S0 on G1"),
        Contrast("T3", "g1__s2", "g1__s0", False, "S2 - S0 on G1"),
        Contrast("E7", "g1__s0", "x_12__own", False, "G1 - X-12 (with day of year)"),
    ]


def load_scored(*, variant: str) -> dict[str, pl.DataFrame]:
    """Read every (arm, spread) file of a variant."""
    folder = common.WORK_DIR / ("scored" if variant == "base" else "scored_d10_masked")
    return {file.stem: pl.read_parquet(file) for file in sorted(folder.glob("*.parquet"))}


def build_wide(*, scored: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return one row per scored half-hour with each arm's score and absolute error as columns.

    Args:
        scored: Each (arm, spread)'s rows.

    Returns:
        Keys `site`, `origin`, `valid_time`, `actual`, `switching_mask`, `jump_window`,
        `lead_hours`, then `score__<name>` and `mae__<name>` per name (null where an arm has no
        row).
    """
    first = scored["g1__s0"]
    wide = first.select(
        "site", "origin", "valid_time", "actual", "switching_mask", "jump_window", "lead_hours"
    )
    for name, frame in scored.items():
        per_row = common.quantile_score(
            actual=frame["actual"].to_numpy(), quantiles=frame.select(QUANTILE_COLUMNS).to_numpy()
        )
        columns = frame.select("site", "origin", "valid_time").with_columns(
            pl.Series(f"score__{name}", per_row),
            (pl.col("valid_time").is_not_null()).cast(pl.Float64).alias("_one"),
        )
        columns = columns.with_columns(
            pl.Series(f"mae__{name}", np.abs(frame["actual"].to_numpy() - frame["mean"].to_numpy()))
        ).drop("_one")
        wide = wide.join(columns, on=["site", "origin", "valid_time"], how="left")
    return wide.with_columns(
        month=pl.col("valid_time").dt.strftime("%Y-%m"),
        calendar_month=pl.col("valid_time").dt.month(),
    )


def week_blocks(*, wide: pl.DataFrame) -> np.ndarray:
    """Return each row's week-block number, 0 to 12, counted from its cut."""
    days = (
        (wide["valid_time"] - wide["origin"].str.to_date().cast(pl.Datetime("us", "UTC")))
        .dt.total_days()
        .to_numpy()
    )
    return np.minimum(days // 7, N_WEEK_BLOCKS - 1).astype(np.int64)


def block_bootstrap(
    *, values: np.ndarray, blocks: np.ndarray, seed: int
) -> tuple[float, float, float, float]:
    """Resample whole week blocks and return the estimate, standard error, and 95% interval.

    Args:
        values: Per-row differences.
        blocks: Each row's week block, 0 to 12.
        seed: Seeds the draws.

    Returns:
        The mean of all rows, the standard deviation of the resampled means, and their 2.5th and
        97.5th percentiles.
    """
    sums = np.bincount(blocks, weights=values, minlength=N_WEEK_BLOCKS)
    counts = np.bincount(blocks, minlength=N_WEEK_BLOCKS).astype(np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, N_WEEK_BLOCKS, size=(N_BLOCK_DRAWS, N_WEEK_BLOCKS))
    resampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return (
        float(values.mean()),
        float(resampled.std()),
        float(np.percentile(resampled, 2.5)),
        float(np.percentile(resampled, 97.5)),
    )


def per_origin_table(
    *,
    wide: pl.DataFrame,
    blocks: np.ndarray,
    contrast: Contrast,
    metric: str,
    sites: Sequence[str],
    masked_out: bool = True,
) -> pl.DataFrame:
    """Return a contrast's estimate, standard error, interval, and tie flag at each origin.

    Args:
        wide: The wide table.
        blocks: Each row's week block.
        contrast: The contrast.
        metric: `score` or `mae`.
        sites: The sites pooled (the decision sites, or fewer for a leave-one-out).
        masked_out: True to drop rows inside the switching masks (and D10's jump windows if the
            wide table's `jump_window` marks any, in the masked variant).

    Returns:
        One row per origin with `estimate`, `se`, `lower_95`, `upper_95`, `band` (the larger of the
        standard error and twice the seed floor), `tie`, `side` (-1, 0, or +1), and `n_rows`.
    """
    column_a, column_b = (f"{metric}__{contrast.treatment}", f"{metric}__{contrast.reference}")
    difference = (wide[column_a] - wide[column_b]).to_numpy()
    keep = wide["site"].is_in(list(sites)).to_numpy() & np.isfinite(difference)
    if masked_out:
        keep &= ~wide["switching_mask"].to_numpy()
    origin = wide["origin"].to_numpy()
    rows = []
    for position, label in enumerate(sorted(set(origin[keep]))):
        pick = keep & (origin == label)
        estimate, se, lower, upper = block_bootstrap(
            values=difference[pick], blocks=blocks[pick], seed=position
        )
        band = max(se, 2.0 * contrast.floor)
        tie = abs(estimate) <= band
        rows.append(
            {
                "origin": label,
                "estimate": estimate,
                "se": se,
                "lower_95": lower,
                "upper_95": upper,
                "band": band,
                "tie": tie,
                "side": 0 if tie else int(np.sign(estimate)),
                "n_rows": int(pick.sum()),
            }
        )
    return pl.DataFrame(rows)


def origin_labels() -> list[str]:
    """Return the 12 scored origins' labels."""
    return [common.cut_label(cut=cut) for cut in common.SCORED_ORIGINS]


def pooled_interval(
    *,
    wide: pl.DataFrame,
    contrast: Contrast,
    metric: str,
    sites: Sequence[str],
    origins: Sequence[str],
    level: float,
    extra_filter: pl.Expr | None = None,
) -> dict[str, float]:
    """Return the pooled estimate and its interval, resampling whole calendar months.

    Args:
        wide: The wide table.
        contrast: The contrast.
        metric: `score` or `mae`.
        sites: The sites pooled.
        origins: The origins pooled.
        level: The interval's coverage in percent.
        extra_filter: A further row filter, such as a row season.

    Returns:
        `estimate`, `lower`, `upper`, and `n_rows`.
    """
    rows = wide.filter(
        pl.col("site").is_in(list(sites)),
        pl.col("origin").is_in(list(origins)),
        ~pl.col("switching_mask"),
        pl.col(f"{metric}__{contrast.treatment}").is_not_null(),
        pl.col(f"{metric}__{contrast.reference}").is_not_null(),
    )
    if extra_filter is not None:
        rows = rows.filter(extra_filter)
    losses = pl.concat(
        [
            rows.select(
                arm=pl.lit(name),
                site="site",
                time=pl.col("valid_time").cast(pl.String),
                seed=pl.lit(0),
                month="month",
                loss=pl.col(f"{metric}__{name}"),
            )
            for name in (contrast.treatment, contrast.reference)
        ]
    )
    lower, upper = bootstrap_difference_at_level(
        losses=losses,
        treatment=contrast.treatment,
        reference=contrast.reference,
        metric="loss",
        level=level,
        n_resamples=N_POOLED_RESAMPLES,
    )
    estimate = rows.select(
        (
            pl.col(f"{metric}__{contrast.treatment}") - pl.col(f"{metric}__{contrast.reference}")
        ).mean()
    ).item()
    return {"estimate": estimate, "lower": lower, "upper": upper, "n_rows": rows.height}


def cochran_q(*, table: pl.DataFrame) -> tuple[float, float]:
    """Return Cochran's Q and its p-value over the per-origin estimates (bootstrap variances)."""
    estimate = table["estimate"].to_numpy()
    weight = 1.0 / np.maximum(table["se"].to_numpy() ** 2, 1e-12)
    pooled = (weight * estimate).sum() / weight.sum()
    q = float((weight * (estimate - pooled) ** 2).sum())
    return q, float(stats.chi2.sf(q, df=len(estimate) - 1))


def sides_in(*, table: pl.DataFrame, origins: Sequence[str]) -> tuple[int, int, int]:
    """Return how many of `origins` are negative, positive, and tied (outside the tie band)."""
    rows = table.filter(pl.col("origin").is_in(list(origins)))
    return (
        int((rows["side"] == -1).sum()),
        int((rows["side"] == 1).sum()),
        int((rows["side"] == 0).sum()),
    )


@dataclass(frozen=True)
class Label:
    """A contrast's label under the decision rule.

    Attributes:
        name: `stable`, `seasonally stable`, `too small to sign`, or `unstable`.
        side: -1 or +1 for the side of a stable label, else 0.
        sign_varies: True if the contrast shows "sign varies".
        detail: The counts the label rests on.
    """

    name: str
    side: int
    sign_varies: bool
    detail: str


def label_contrast(
    *, table: pl.DataFrame, pooled_99: dict[str, float], pooled_by_type: dict
) -> Label:
    """Apply the decision rule's labels to one contrast.

    Args:
        table: The per-origin table.
        pooled_99: The pooled 99% interval over all 12 origins.
        pooled_by_type: The pooled 95% interval of `winter` and `summer` quarter-type origins.

    Returns:
        The label.
    """
    origins = origin_labels()
    negative, positive, ties = sides_in(table=table, origins=origins)
    winter = [
        common.cut_label(cut=c)
        for c in common.SCORED_ORIGINS
        if common.quarter_type(cut=c) == "winter"
    ]
    summer = [
        common.cut_label(cut=c)
        for c in common.SCORED_ORIGINS
        if common.quarter_type(cut=c) == "summer"
    ]
    clearly = table.filter((pl.col("lower_95") > 0) | (pl.col("upper_95") < 0))
    below = int((clearly["estimate"] < 0).sum())
    above = int((clearly["estimate"] > 0).sum())
    _, q_p = cochran_q(table=table)
    varies = (below >= 2 and above >= 2) or q_p < 0.01
    detail = f"{negative} negative, {positive} positive, {ties} tied of 12; Cochran Q p={q_p:.4f}"
    for side, count in ((-1, negative), (1, positive)):
        if count < 10:
            continue
        per_year_ok = all(
            sides_in(
                table=table,
                origins=[
                    common.cut_label(cut=c)
                    for c in common.SCORED_ORIGINS
                    if common.origin_year(cut=c) == year
                ],
            )[0 if side == -1 else 1]
            >= 3
            for year in ("2023-24", "2024-25", "2025-26")
        )
        excludes = pooled_99["upper"] < 0 if side == -1 else pooled_99["lower"] > 0
        if per_year_ok and excludes:
            return Label("stable", side, varies, detail)
    w_neg, w_pos, _ = sides_in(table=table, origins=winter)
    s_neg, s_pos, _ = sides_in(table=table, origins=summer)
    for w_side, s_side in ((-1, 1), (1, -1)):
        w_count = w_neg if w_side == -1 else w_pos
        s_count = s_pos if s_side == 1 else s_neg
        w_ok = (
            pooled_by_type["winter"]["upper"] < 0
            if w_side == -1
            else pooled_by_type["winter"]["lower"] > 0
        )
        s_ok = (
            pooled_by_type["summer"]["lower"] > 0
            if s_side == 1
            else pooled_by_type["summer"]["upper"] < 0
        )
        if w_count >= 5 and s_count >= 5 and w_ok and s_ok:
            return Label("seasonally stable", w_side, varies, detail + f"; winter side {w_side}")
    if ties >= 5:
        return Label("too small to sign", 0, varies, detail)
    return Label("unstable", 0, varies, detail)


def month_block_mean(
    *, values: np.ndarray, months: np.ndarray, seed: int = 0
) -> tuple[float, float, float]:
    """Return a mean and its 95% interval from resampling whole calendar months."""
    unique, inverse = np.unique(months, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(unique), size=(2000, len(unique)))
    resampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(resampled, 2.5)),
        float(np.percentile(resampled, 97.5)),
    )


def tail_rows(*, frame: pl.DataFrame, label: str) -> list[dict]:
    """Return coverage and pinball loss at the four tail levels for one arm's scored rows."""
    rows = []
    months = frame["valid_time"].dt.strftime("%Y-%m").to_numpy()
    actual = frame["actual"].to_numpy()
    for level in TAIL_LEVELS:
        quantile = frame[f"q{round(level * 100):02d}"].to_numpy()
        covered = (actual <= quantile).astype(np.float64)
        error = actual - quantile
        pinball = np.maximum(level * error, (level - 1.0) * error)
        coverage, c_low, c_high = month_block_mean(values=covered, months=months)
        loss, l_low, l_high = month_block_mean(values=pinball, months=months)
        rows.append(
            {
                "arm": label,
                "level": level,
                "n_rows": frame.height,
                "coverage": coverage,
                "coverage_lower_95": c_low,
                "coverage_upper_95": c_high,
                "gap_points": 100.0 * (coverage - level),
                "pinball": loss,
                "pinball_lower_95": l_low,
                "pinball_upper_95": l_high,
            }
        )
    return rows


def decision_rows(*, frame: pl.DataFrame, origins: Sequence[str] | None = None) -> pl.DataFrame:
    """Keep the decision sites' rows outside the switching masks, for the chosen origins."""
    kept = frame.filter(pl.col("site").is_in(common.DECISION_SITES), ~pl.col("switching_mask"))
    if origins is not None:
        kept = kept.filter(pl.col("origin").is_in(list(origins)))
    return kept


def c5_verdict(
    *, scored: dict[str, pl.DataFrame], origins: Sequence[str], pooled_score: dict
) -> dict:
    """Apply the C5 pass rule to S1 on G-all+D1 over `origins`.

    Args:
        scored: Each (arm, spread)'s rows.
        origins: The origins pooled.
        pooled_score: The 99% interval of S1 minus S0 on the quantile score over `origins`.

    Returns:
        Each criterion's value and whether C5 passes.
    """
    s1 = decision_rows(frame=scored["g_all_d1__s1"], origins=origins)
    s0 = decision_rows(frame=scored["g_all_d1__s0"], origins=origins)
    table = {
        (name, row["level"]): row
        for name, frame in (("s1", s1), ("s0", s0))
        for row in tail_rows(frame=frame, label=name)
    }
    pooled_ok = all(
        abs(table[("s1", level)]["gap_points"]) <= 1.0
        and table[("s1", level)]["coverage_lower_95"] >= level - 0.015
        and table[("s1", level)]["coverage_upper_95"] <= level + 0.015
        for level in (0.95, 0.99)
    )
    site_hits = {0.95: 0, 0.99: 0}
    for site in common.DECISION_SITES:
        part = s1.filter(pl.col("site") == site)
        for level, tolerance in ((0.95, 2.0), (0.99, 1.5)):
            covered = part["actual"] <= part[f"q{round(level * 100):02d}"]
            coverage = float(np.mean(covered.to_numpy()))
            site_hits[level] += int(abs(100.0 * (coverage - level)) <= tolerance)
    sites_ok = site_hits[0.95] >= 4 and site_hits[0.99] >= 4
    lower_tails_ok = all(
        abs(table[("s1", level)]["gap_points"]) <= abs(table[("s0", level)]["gap_points"])
        for level in (0.01, 0.05)
    )
    score_ok = pooled_score["upper"] <= 0.03
    return {
        "pooled_coverage_ok": pooled_ok,
        "sites_within_tolerance": site_hits,
        "sites_ok": sites_ok,
        "lower_tails_no_worse": lower_tails_ok,
        "score_upper_99": pooled_score["upper"],
        "score_ok": score_ok,
        "passes": pooled_ok and sites_ok and lower_tails_ok and score_ok,
        "s1_gap_points": {level: table[("s1", level)]["gap_points"] for level in TAIL_LEVELS},
        "s0_gap_points": {level: table[("s0", level)]["gap_points"] for level in TAIL_LEVELS},
    }


def verdict_of(*, c1: Label, c2: Label, loso_min: dict[str, int], pooled: dict[str, dict]) -> str:
    """Apply the decision rule's verdict, in its stated order."""
    names = (c1.name, c2.name)
    near_zero = all(
        max(abs(pooled[k]["lower"]), abs(pooled[k]["upper"])) <= 0.10 for k in ("C1", "C2")
    )
    if names == ("stable", "stable") and min(loso_min["C1"], loso_min["C2"]) >= 9:
        verdict = "supported"
    elif "stable" in names or "seasonally stable" in names:
        verdict = "partly supported"
    elif names == ("unstable", "unstable") and c1.sign_varies and c2.sign_varies:
        verdict = "refuted"
    elif not (c1.sign_varies or c2.sign_varies) and near_zero:
        verdict = "not supported"
    else:
        verdict = "inconclusive"
    return verdict


def origin_groups() -> dict[str, list[str]]:
    """Return the origin groups the pooled estimates are read over, each a list of origin labels."""

    def labels(*, keep: Callable[[datetime], bool]) -> list[str]:
        return [common.cut_label(cut=c) for c in common.SCORED_ORIGINS if keep(c)]

    return {
        "all 12": origin_labels(),
        "winter-quarter origins": labels(keep=lambda c: common.quarter_type(cut=c) == "winter"),
        "summer-quarter origins": labels(keep=lambda c: common.quarter_type(cut=c) == "summer"),
        "2023-24 (pre-tuning)": labels(keep=lambda c: common.origin_year(cut=c) == "2023-24"),
        "2024-25 (in-tuning)": labels(keep=lambda c: common.origin_year(cut=c) == "2024-25"),
        "2025-26 (post-tuning)": labels(keep=lambda c: c in common.POST_TUNING_ORIGINS),
        "8 out-of-tuning": labels(keep=lambda c: c in common.OUT_OF_TUNING_ORIGINS),
        "8 from 2024-07-01": labels(keep=lambda c: c >= datetime(2024, 7, 1, tzinfo=UTC)),
    }


def contrast_result(
    *, wide: pl.DataFrame, blocks: np.ndarray, contrast: Contrast, tables: dict
) -> dict:
    """Return every pooled interval, season split, per-site value, and leave-one-out table."""
    sites = common.DECISION_SITES
    score = tables[(contrast.name, "score")]
    available = [o for o in origin_labels() if o in set(score["origin"])]
    level = BONFERRONI_LEVEL if contrast.planned else 95.0
    winter_rows = pl.col("calendar_month").is_in(WINTER_ROW_MONTHS)

    def pooled(
        *, origins: Sequence[str], metric: str, level: float, flt: pl.Expr | None = None
    ) -> dict[str, float]:
        return pooled_interval(
            wide=wide,
            contrast=contrast,
            metric=metric,
            sites=sites,
            origins=origins,
            level=level,
            extra_filter=flt,
        )

    by_group = {
        group: pooled(
            origins=[o for o in members if o in available],
            metric="score",
            level=level if group == "all 12" else 95.0,
        )
        for group, members in origin_groups().items()
        if any(o in available for o in members)
    }
    column_a, column_b = f"score__{contrast.treatment}", f"score__{contrast.reference}"
    per_site = {
        site: float(
            wide.filter(
                pl.col("site") == site,
                ~pl.col("switching_mask"),
                pl.col(column_a).is_not_null(),
                pl.col(column_b).is_not_null(),
            )
            .select((pl.col(column_a) - pl.col(column_b)).mean())
            .item()
        )
        for site in sites
    }
    loso_pooled: dict[str, dict[str, dict[str, float]]] = {}
    if contrast.name in LOSO_LABEL_CONTRASTS:
        groups = origin_groups()
        for left_out in sites:
            kept = [s for s in sites if s != left_out]
            loso_pooled[left_out] = {
                key: pooled_interval(
                    wide=wide,
                    contrast=contrast,
                    metric="score",
                    sites=kept,
                    origins=groups[group],
                    level=BONFERRONI_LEVEL if key == "all" else 95.0,
                )
                for key, group in (
                    ("all", "all 12"),
                    ("winter", "winter-quarter origins"),
                    ("summer", "summer-quarter origins"),
                )
            }
    return {
        "loso_pooled": loso_pooled,
        "contrast": contrast.name,
        "description": contrast.description,
        "planned": contrast.planned,
        "table": score,
        "mae_table": tables[(contrast.name, "mae")],
        "pooled": by_group,
        "pooled_mae": pooled(origins=available, metric="mae", level=level),
        "season": {
            "rows Nov-Mar": pooled(origins=available, metric="score", level=95.0, flt=winter_rows),
            "rows Apr-Oct": pooled(origins=available, metric="score", level=95.0, flt=~winter_rows),
        },
        "all_rows": pooled_interval(
            wide=wide.with_columns(switching_mask=pl.lit(value=False)),
            contrast=contrast,
            metric="score",
            sites=sites,
            origins=available,
            level=95.0,
        ),
        "per_site": per_site,
        "loso": {
            left_out: per_origin_table(
                wide=wide,
                blocks=blocks,
                contrast=contrast,
                metric="score",
                sites=[s for s in sites if s != left_out],
            )
            for left_out in sites
        },
    }


def interval_text(*, value: dict[str, float]) -> str:
    """Format a pooled estimate with its interval."""
    return f"{value['estimate']:+.3f} [{value['lower']:+.3f}, {value['upper']:+.3f}]"


def print_contrast(*, row: dict, label: Label, loso_text: str) -> None:
    """Print one contrast's block of the report."""
    kind = "planned" if row["planned"] else "exploratory"
    level = "99%" if row["planned"] else "95%"
    table = row["table"]
    print(f"### {row['contrast']}: {row['description']} ({kind})")
    print(f"- label: **{label.name}** (sign varies: {label.sign_varies}); {label.detail}")
    print(
        f"- pooled 12 origins ({level}): {interval_text(value=row['pooled']['all 12'])}, "
        f"{row['pooled']['all 12']['n_rows']} rows"
    )
    print(f"- error of the mean forecast (same level): {interval_text(value=row['pooled_mae'])}")
    for group, value in {**row["pooled"], **row["season"]}.items():
        if group != "all 12":
            print(f"- {group} (95%): {interval_text(value=value)}")
    print(f"- all rows including switching (95%): {interval_text(value=row['all_rows'])}")
    print("- per site: " + ", ".join(f"{s} {v:+.3f}" for s, v in row["per_site"].items()))
    opposite = sum(
        1
        for a, b in zip(table["side"].to_list(), row["mae_table"]["side"].to_list(), strict=False)
        if a * b == -1
    )
    print(f"- origins where the score and mean-error contrasts have opposite sides: {opposite}")
    print(f"- leave one site out, origins (negative/positive/tied): {loso_text}")
    print("\n| origin | estimate | SE | 95% interval | band | side |\n|---|---|---|---|---|---|")
    for r in table.iter_rows(named=True):
        print(
            f"| {r['origin']} | {r['estimate']:+.3f} | {r['se']:.3f} | "
            f"[{r['lower_95']:+.3f}, {r['upper_95']:+.3f}] | {r['band']:.3f} | {r['side']:+d} |"
        )
    print()


def report_contrasts(*, results: list[dict]) -> tuple[dict[str, Label], dict[str, int]]:
    """Label every contrast and print its block; return the labels and leave-one-out minima."""
    origins = origin_labels()
    labels: dict[str, Label] = {}
    loso_min: dict[str, int] = {}
    print("## Contrasts (treatment minus reference; negative favours the treatment)\n")
    for row in results:
        pooled = row["pooled"]
        by_type = {
            "winter": pooled.get("winter-quarter origins", pooled["all 12"]),
            "summer": pooled.get("summer-quarter origins", pooled["all 12"]),
        }
        label = label_contrast(
            table=row["table"], pooled_99=pooled["all 12"], pooled_by_type=by_type
        )
        labels[row["contrast"]] = label
        counts = {s: sides_in(table=t, origins=origins) for s, t in row["loso"].items()}
        if label.side:
            loso_min[row["contrast"]] = min(
                count[0 if label.side == -1 else 1] for count in counts.values()
            )
        loso_text = "; ".join(f"-{s}: {'/'.join(map(str, c))}" for s, c in counts.items())
        for left_out, pooled_kept in row["loso_pooled"].items():
            kept_label = label_contrast(
                table=row["loso"][left_out],
                pooled_99=pooled_kept["all"],
                pooled_by_type={"winter": pooled_kept["winter"], "summer": pooled_kept["summer"]},
            )
            loso_text += (
                f"\n  - without {left_out}: {kept_label.name}; pooled "
                f"{interval_text(value=pooled_kept['all'])}"
            )
        print_contrast(row=row, label=label, loso_text=loso_text)
    return labels, loso_min


def report_verdict(
    *, results: list[dict], labels: dict[str, Label], loso_min: dict[str, int]
) -> str:
    """Print the decision rule's verdict, the 8-origin survival check, and the C3 and C4 labels."""
    by_name = {row["contrast"]: row for row in results}
    pooled = {k: by_name[k]["pooled"]["all 12"] for k in ("C1", "C2")}
    verdict = verdict_of(c1=labels["C1"], c2=labels["C2"], loso_min=loso_min, pooled=pooled)
    print(f"## Verdict by the decision rule: **{verdict}**")
    print(f"- C1 {labels['C1'].name}, C2 {labels['C2'].name}; leave-one-out minima: {loso_min}")
    out_of_tuning = origin_groups()["8 out-of-tuning"]
    for name in ("C1", "C2"):
        out = by_name[name]["pooled"]["8 out-of-tuning"]
        sides = sides_in(table=by_name[name]["table"], origins=out_of_tuning)
        text = interval_text(value=out)
        print(f"- {name} on the 8 out-of-tuning origins (95%): {text}; sides {sides}")
    c3, c4 = labels["C3"], labels["C4"]
    c3_text = interval_text(value=by_name["C3"]["pooled"]["all 12"])
    c4_text = interval_text(value=by_name["C4"]["pooled_mae"])
    print(f"- C3 label: {c3.name} (side {c3.side}); 99% {c3_text}")
    print(f"- C4 label: {c4.name} (side {c4.side}); mean-error contrast {c4_text}")
    return verdict


def report_tails(*, scored: dict[str, pl.DataFrame], wide: pl.DataFrame, suffix: str) -> None:
    """Print C5 and the tail-calibration table of every arm and spread, and save the table."""
    c5q = next(c for c in planned_contrasts(bar="x_12", seed_floor=0.0) if c.name == "C5q")
    print("\n## C5: the tail model\n")
    groups = origin_groups()
    for group in ("all 12", "8 from 2024-07-01"):
        pooled_score = pooled_interval(
            wide=wide,
            contrast=c5q,
            metric="score",
            sites=common.DECISION_SITES,
            origins=groups[group],
            level=BONFERRONI_LEVEL,
        )
        outcome = c5_verdict(scored=scored, origins=groups[group], pooled_score=pooled_score)
        print(f"### {group}\n{json.dumps(outcome, indent=2, default=float)}\n")
    print("## Tail calibration at q01, q05, q95, q99 (12 origins, five decision sites)\n")
    rows: list[dict] = []
    for name, frame in scored.items():
        rows += tail_rows(frame=decision_rows(frame=frame), label=name)
    table = pl.DataFrame(rows)
    table.write_parquet(TABLE_DIR / f"tails{suffix}.parquet")
    with pl.Config(tbl_rows=400, tbl_cols=12, float_precision=4):
        print(table)


def report_absolute(*, scored: dict[str, pl.DataFrame], suffix: str) -> None:
    """Print each arm's absolute score, mean-forecast error, and median minus mean forecast."""
    print("\n## Absolute quantile score and mean-forecast checks\n")
    parts = []
    for name, frame in scored.items():
        rows = decision_rows(frame=frame).cast({"q50": pl.Float64, "mean": pl.Float64})
        per_row = common.quantile_score(
            actual=rows["actual"].to_numpy(), quantiles=rows.select(QUANTILE_COLUMNS).to_numpy()
        )
        rows = rows.with_columns(
            pl.Series("score", per_row),
            median_minus_mean=pl.col("q50") - pl.col("mean"),
            bias=pl.col("actual") - pl.col("mean"),
            abs_error=(pl.col("actual") - pl.col("mean")).abs(),
        )
        parts.append(
            rows.group_by("site", "origin")
            .agg(
                score=pl.col("score").mean(),
                mae=pl.col("abs_error").mean(),
                median_minus_mean=pl.col("median_minus_mean").mean(),
                bias=pl.col("bias").mean(),
                n=pl.len(),
            )
            .with_columns(arm=pl.lit(name))
        )
    table = pl.concat(parts)
    table.write_parquet(TABLE_DIR / f"absolute{suffix}.parquet")

    def weighted(column: str) -> pl.Expr:
        return (pl.col(column) * pl.col("n")).sum() / pl.col("n").sum()

    with pl.Config(tbl_rows=100, float_precision=4):
        print(
            table.group_by("arm")
            .agg(
                score=weighted("score"),
                mae=weighted("mae"),
                median_minus_mean=weighted("median_minus_mean"),
                bias=weighted("bias"),
            )
            .sort("score")
        )
        print(
            table.group_by("arm", "site")
            .agg(score=weighted("score"), median_minus_mean=weighted("median_minus_mean"))
            .sort("arm", "site")
        )


def report_controls(*, results: list[dict]) -> None:
    """Print the pre-registered controls: calendar controls, replication, and the seed floor."""
    by_name = {row["contrast"]: row for row in results}
    print("\n## Controls and replication\n")
    for name in ("K1", "K2", "K3", "E4"):
        row = by_name[name]
        pooled = row["pooled"]["all 12"]
        ties = int(row["table"]["tie"].sum())
        spans = pooled["lower"] <= 0.0 <= pooled["upper"]
        verdict = "changes nothing" if ties >= 10 and spans else "does NOT change nothing"
        print(
            f"- {name} ({row['description']}): tied at {ties} of 12 origins; pooled 95% "
            f"{interval_text(value=pooled)}; **{verdict}**"
        )
    for fitted, pinned in (("E1", "C1"), ("E2", "C2"), ("E3", "C3")):
        agree = sum(
            1
            for a, b in zip(
                by_name[fitted]["table"]["estimate"].to_list(),
                by_name[pinned]["table"]["estimate"].to_list(),
                strict=True,
            )
            if a * b > 0
        )
        print(f"- {fitted} shares {pinned}'s sign at {agree} of 12 origins (need 10)")
    decisions = json.loads(DECISIONS_PATH.read_text())
    print(f"- XGBoost seed floor {decisions['seed_floor']:.4f} (need under 0.03)")


def report_descriptive(*, scored: dict[str, pl.DataFrame], suffix: str) -> None:
    """Print the exploratory tables: D16's bias, D9, staleness, and tail coverage by group."""
    arms = ("g1__s0", "g_all__s0", "g_all_d1__s0", "x_12_nodoy__own")
    print("\n## D16's mean forecast error (actual minus mean) per origin, points\n")
    print("| origin | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
    for origin in origin_labels():
        cells = []
        for arm in arms:
            rows = scored[arm].filter(
                pl.col("site") == "D16", pl.col("origin") == origin, ~pl.col("switching_mask")
            )
            cells.append(f"{(rows['actual'] - rows['mean']).mean():+.2f}")
        print(f"| {origin} | " + " | ".join(cells) + " |")
    print("\n## D9 (descriptive; origins from 2025-10-01): mean quantile score per arm\n")
    for name, frame in scored.items():
        rows = frame.filter(pl.col("site") == "D9", ~pl.col("switching_mask"))
        if rows.is_empty():
            continue
        per_row = common.quantile_score(
            actual=rows["actual"].to_numpy(), quantiles=rows.select(QUANTILE_COLUMNS).to_numpy()
        )
        print(f"- {name}: {per_row.mean():.3f} over {rows.height} rows")
    print("\n## Staleness: mean quantile score by month since the cut (decision sites)\n")
    print("| arm | month 1 | month 2 | month 3 |\n|---|---|---|---|")
    for arm in arms:
        rows = decision_rows(frame=scored[arm]).with_columns(
            month_since=pl.min_horizontal(pl.col("lead_hours") // (24 * 30.5), pl.lit(2.0))
        )
        per_row = common.quantile_score(
            actual=rows["actual"].to_numpy(), quantiles=rows.select(QUANTILE_COLUMNS).to_numpy()
        )
        by_month = (
            rows.with_columns(pl.Series("score", per_row))
            .group_by("month_since")
            .agg(pl.col("score").mean())
            .sort("month_since")
        )
        print(f"| {arm} | " + " | ".join(f"{v:.3f}" for v in by_month["score"]) + " |")
    print("\n## q95 coverage by calendar month of the valid time (decision sites, %)\n")
    for arm in ("g_all_d1__s0", "g_all_d1__s1", "g1__s0", "x_12_nodoy__own"):
        rows = decision_rows(frame=scored[arm]).with_columns(
            month=pl.col("valid_time").dt.month(),
            covered=(pl.col("actual") <= pl.col("q95")).cast(pl.Float64),
        )
        by_month = rows.group_by("month").agg(pl.col("covered").mean()).sort("month")
        print(f"- {arm}: " + " ".join(f"{m}:{100 * c:.0f}" for m, c in by_month.iter_rows()))
    print("\n## Tail coverage by origin year and by site (S0, S1, S2 on G-all+D1; bar)\n")
    groups = origin_groups()
    for arm in ("g_all_d1__s0", "g_all_d1__s1", "g_all_d1__s2", "x_12_nodoy__own"):
        for group in ("2023-24 (pre-tuning)", "2024-25 (in-tuning)", "2025-26 (post-tuning)"):
            rows = tail_rows(
                frame=decision_rows(frame=scored[arm], origins=groups[group]), label=arm
            )
            text = ", ".join(
                f"q{round(r['level'] * 100):02d} {100 * r['coverage']:.1f}" for r in rows
            )
            print(f"- {arm}, {group}: {text}")
        for site in common.DECISION_SITES:
            rows_site = decision_rows(frame=scored[arm]).filter(pl.col("site") == site)
            coverage_text = []
            for level in TAIL_LEVELS:
                column = f"q{round(level * 100):02d}"
                covered = np.mean((rows_site["actual"] <= rows_site[column]).to_numpy())
                coverage_text.append(f"{column} {100 * covered:.1f}")
            text = ", ".join(coverage_text)
            print(f"- {arm}, {site}: {text}")


def main() -> None:
    """Compute the tie bands, then every table of the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("base", "d10_masked"), default="base")
    arguments = parser.parse_args()
    if not DECISIONS_PATH.exists():
        msg = "run e00107_decisions.py first: the bar must be chosen before any scored window"
        raise FileNotFoundError(msg)
    decisions = json.loads(DECISIONS_PATH.read_text())
    bar = decisions["bar_arm"]
    scored = load_scored(variant=arguments.variant)
    wide = build_wide(scored=scored)
    blocks = week_blocks(wide=wide)
    contrasts = planned_contrasts(bar=bar, seed_floor=decisions["seed_floor"])
    contrasts += exploratory_contrasts(bar=bar)
    suffix = "" if arguments.variant == "base" else f"_{arguments.variant}"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        (c.name, metric): per_origin_table(
            wide=wide, blocks=blocks, contrast=c, metric=metric, sites=common.DECISION_SITES
        )
        for c in contrasts
        for metric in ("score", "mae")
    }
    bands = pl.concat(
        [
            tables[(c.name, "score")]
            .select("origin", "se", "band")
            .with_columns(contrast=pl.lit(c.name))
            for c in contrasts
        ]
    ).sort("contrast", "origin")
    bands.write_parquet(TABLE_DIR / f"tie_bands{suffix}.parquet")
    if arguments.variant == "base":
        decisions["tie_bands_sha256"] = hashlib.sha256(bands.write_csv().encode()).hexdigest()
        decisions["tie_bands_written_at"] = datetime.now(UTC).isoformat()
        DECISIONS_PATH.write_text(json.dumps(decisions, indent=2) + "\n")
    print(f"Variant {arguments.variant}; bar arm {bar}; tie-band hash written to decisions.json\n")

    results = [
        contrast_result(wide=wide, blocks=blocks, contrast=c, tables=tables) for c in contrasts
    ]
    for row in results:
        for metric, key in (("score", "table"), ("mae", "mae_table")):
            row[key].with_columns(
                contrast=pl.lit(row["contrast"]), metric=pl.lit(metric)
            ).write_parquet(TABLE_DIR / f"origin_{metric}_{row['contrast']}{suffix}.parquet")
    (TABLE_DIR / f"pooled{suffix}.json").write_text(
        json.dumps(
            {
                row["contrast"]: {
                    "pooled": row["pooled"],
                    "pooled_mae": row["pooled_mae"],
                    "season": row["season"],
                    "per_site": row["per_site"],
                    "loso_pooled": row["loso_pooled"],
                    "loso_counts": {
                        s: sides_in(table=t, origins=origin_labels())
                        for s, t in row["loso"].items()
                    },
                }
                for row in results
            },
            indent=2,
        )
    )
    labels, loso_min = report_contrasts(results=results)
    verdict = report_verdict(results=results, labels=labels, loso_min=loso_min)
    report_controls(results=results)
    report_tails(scored=scored, wide=wide, suffix=suffix)
    report_descriptive(scored=scored, suffix=suffix)
    report_absolute(scored=scored, suffix=suffix)
    summary = {
        name: {"label": item.name, "side": item.side, "sign_varies": item.sign_varies}
        for name, item in labels.items()
    }
    (TABLE_DIR / f"labels{suffix}.json").write_text(
        json.dumps({**summary, "verdict": verdict}, indent=2)
    )


if __name__ == "__main__":
    main()
