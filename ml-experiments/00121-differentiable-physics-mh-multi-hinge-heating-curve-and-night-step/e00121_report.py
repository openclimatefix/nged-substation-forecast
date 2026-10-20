"""Print experiment 121's numbers: scores, the planned contrasts, labels, tails, checks, and gates.

Every number the write-up quotes comes from this script. It reads the stored prediction sets
(`<year>_<arm>`), scores each row with the quantile score (twice the mean pinball loss over the 11
levels, in points of effective capacity) on rows outside the hand-filtered switching masks, and
pairs arms row by row at day D+1.

- **C1** (MH-K minus B0-K, 97.5%, all rows) with the replicate year, the site count, the splits
  (November to March, April to October, cold snaps, evening peaks, hour bands), leaving one site
  out, and the same contrast on the mean absolute error of the mean forecast. Beside it: the
  unfiltered contrast and the filter-fit control.
- **N1 to N4** (NS-L-K, NS-U-K, NS-H-K, NS-HW-K minus B0-K, 99.375%, on winter nights) with
  non-inferiority on all other rows, the replicate year, and the site count.
- **The indexing contrast** (NS-U-K minus NS-L-K on British Summer Time nights, 95%).
- **Exploratory contrasts** at 95%, **winter breakdown**, **cold British Summer Time nights**,
  **labels and verdict** from the decision rule, **tails**, **checks**, and **gates**.

    uv run python ml-experiments/00121-*/e00121_report.py > data/.../work/report.txt
"""

import json
from dataclasses import dataclass
from functools import cache
from typing import Any, Final

import e00121_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import losses_for_bootstrap

SITE_OF_ID: Final[dict[int, str]] = {7: "D9", 8: "D11", 10: "D10", 16: "D14", 17: "D5", 18: "D16"}
DECISION: Final[tuple[str, ...]] = common.DECISION_SITES
MH_LEVEL: Final[float] = common.SCORED_LEVEL
NIGHT_LEVEL: Final[float] = common.NIGHT_LEVEL
EXPLORATORY_LEVEL: Final[float] = 95.0
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2)
WINTER_BREAKDOWN_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
COLD_BST_THRESHOLD_C: Final[float] = 8.0
MIN_COLD_BST_NIGHTS: Final[int] = 10
COLD_SNAP_SHARE: Final[float] = 0.10
MIN_COLD_SNAP_DAYS: Final[int] = 5
EVENING_PEAK_HOURS: Final[tuple[int, int]] = (16, 20)
MH_GAIN_POINTS: Final[float] = 0.03
NIGHT_GAIN_POINTS: Final[float] = 0.05
NON_INFERIORITY_POINTS: Final[float] = 0.02
MIN_SITES_MH: Final[int] = 4
MIN_SITES_NIGHT: Final[int] = 3
TAIL_LEVELS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
TAIL_FLAG_POINTS: Final[float] = 1.0
DATA_114: Final = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00114-differentiable-physics-n3-n4-interactions-and-heating-threshold"
)
PART_4_DIR: Final = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
)
BG2_PATH: Final = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
    / "p_s6_bg2"
    / "predictions.parquet"
)
PART_4_SETS: Final[tuple[str, ...]] = ("a2", "wk", "x_best_k", "climatology", "manual_heuristic")
YEARS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
MH_ARMS: Final[tuple[str, ...]] = ("mh", "mh5", "SHRUNK")
NIGHT_ARMS: Final[tuple[str, ...]] = ("ns_l", "ns_u", "ns_h", "ns_hw")


@dataclass(frozen=True)
class Contrast:
    """A treatment arm minus a reference arm.

    Attributes:
        name: The contrast's label in the report.
        treatment: The treatment arm's name.
        reference: The reference arm's name.
    """

    name: str
    treatment: str
    reference: str


def shrunk_name() -> str:
    """Return the name of the MH-sh arm at the chosen multiplier."""
    decisions = json.loads((common.WORK_DIR / "decisions.json").read_text())
    return f"mh_sh{float(decisions['shrink']):g}"


def folder_of(*, year: str, arm: str) -> Any:
    """Return the folder that holds an arm's stored set for a year."""
    if arm.startswith("p4:"):
        return PART_4_DIR / "predictions" / f"{year}_{arm.removeprefix('p4:')}"
    if arm.startswith("e114:"):
        return DATA_114 / "predictions" / f"{year}_{arm.removeprefix('e114:')}"
    return common.DATA_DIR / "predictions" / f"{year}_{arm}"


@cache
def cold_days() -> pl.DataFrame:
    """Return, per decision site and local date, the daily mean temperature and snap flags.

    A cold snap is a day of November to March whose daily mean ERA5 temperature at the site's cell
    lies in the coldest 10% of days of that winter (July to June year), fixed before any score is
    read. Winters with fewer than `MIN_COLD_SNAP_DAYS` such days are reported as too few days.
    """
    cache_ = common.load_cache()
    frames = []
    for site in common.SCORED_SITES:
        features = cache_.sites[site]
        frame = pl.DataFrame(
            {
                "end": cache_.end_times,
                "temperature": features.smoothed_temperature_c[:, 0],
            }
        ).drop_nans()
        frame = frame.with_columns(
            local_date=(pl.col("end") - pl.duration(minutes=30))
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("Europe/London")
            .dt.date()
        )
        daily = frame.group_by("local_date").agg(
            pl.col("temperature").mean().alias("tmean"), pl.len().alias("n")
        )
        daily = daily.filter(pl.col("n") >= 40).with_columns(
            site_name=pl.lit(site),
            winter_year=pl.when(pl.col("local_date").dt.month() >= 7)
            .then(pl.col("local_date").dt.year())
            .otherwise(pl.col("local_date").dt.year() - 1),
            in_winter=pl.col("local_date").dt.month().is_in(WINTER_BREAKDOWN_MONTHS),
        )
        winter = daily.filter(pl.col("in_winter"))
        cutoffs = winter.group_by("winter_year").agg(
            pl.col("tmean").quantile(COLD_SNAP_SHARE).alias("cutoff"),
        )
        daily = daily.join(cutoffs, on="winter_year", how="left").with_columns(
            cold_snap=pl.col("in_winter") & (pl.col("tmean") <= pl.col("cutoff"))
        )
        frames.append(daily.select("site_name", "local_date", "tmean", "cold_snap", "winter_year"))
    return pl.concat(frames)


@cache
def load_arm(*, year: str, arm: str, drop_switching: bool = True) -> pl.DataFrame:
    """Read a stored set with its per-row quantile score, mean absolute error, and row flags.

    Args:
        year: `tuning`, `replicate`, or `scored`.
        arm: The arm's name; `p4:` and `e114:` prefix stored sets of experiments 106 and 114.
        drop_switching: True to leave out the rows inside the switching masks.

    Returns:
        The set with `site_name`, `qs`, `ae`, `band`, `calendar_month`, and the row flags
        `winter_night`, `bst_night`, `cold_snap`, `evening_winter`, and `cold_bst_night` added.
    """
    frame = pl.read_parquet(folder_of(year=year, arm=arm) / "predictions.parquet")
    if drop_switching:
        frame = frame.filter(~pl.col("switching_mask"))
    quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    start = (
        (pl.col("valid_time") - pl.duration(minutes=30))
        .dt.convert_time_zone("Europe/London")
        .alias("local_start")
    )
    frame = frame.with_columns(start)
    local_hour = pl.col("local_start").dt.hour()
    month = pl.col("local_start").dt.month()
    night = (local_hour >= 22) | (local_hour < 8)
    bst = pl.col("local_start").dt.dst_offset() > pl.duration(seconds=0)
    frame = frame.with_columns(
        qs=pl.Series(common.quantile_score(actual=actual, quantiles=quantiles)),
        ae=(pl.col("actual") - pl.col("mean")).abs(),
        site_name=pl.col("time_series_id").replace_strict(SITE_OF_ID),
        band=pl.col("local_start")
        .dt.hour()
        .replace_strict(
            {h: (0 if h < 6 else 1 if h < 11 else 2 if h < 17 else 3) for h in range(24)}
        ),
        calendar_month=month,
        local_date=pl.col("local_start").dt.date(),
        winter_night=night & month.is_in(WINTER_MONTHS),
        bst_night=night & bst,
        evening_winter=(local_hour >= EVENING_PEAK_HOURS[0])
        & (local_hour < EVENING_PEAK_HOURS[1])
        & month.is_in(WINTER_BREAKDOWN_MONTHS),
    )
    days = cold_days().select("site_name", "local_date", "cold_snap", "tmean")
    return frame.join(days, on=["site_name", "local_date"], how="left").with_columns(
        cold_snap=pl.col("cold_snap").fill_null(False),
        cold_bst_night=pl.col("bst_night") & (pl.col("tmean") < COLD_BST_THRESHOLD_C),
    )


KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time", "seed"]


def paired(
    *,
    year: str,
    treatment: str,
    reference: str,
    sites: tuple[str, ...],
    where: pl.Expr | None = None,
    drop_switching: bool = True,
) -> pl.DataFrame:
    """Return both arms' rows on the sites, paired, as the frame `studies.bootstrap` reads.

    Args:
        year: The scored year.
        treatment: The treatment arm.
        reference: The reference arm.
        sites: The sites to keep.
        where: A row filter on the loaded columns, or `None` for all rows.
        drop_switching: True to leave out the rows inside the switching masks.

    Returns:
        The paired loss frame.
    """
    arms = {}
    for name in (treatment, reference):
        frame = load_arm(year=year, arm=name, drop_switching=drop_switching).filter(
            pl.col("site_name").is_in(sites)
        )
        if where is not None:
            frame = frame.filter(where)
        arms[name] = frame.select(*KEYS, "qs", "ae")
    first, second = list(arms.values())
    shared = first.select(KEYS).join(second.select(KEYS), on=KEYS, how="inner")
    arms = {name: frame.join(shared, on=KEYS, how="inner") for name, frame in arms.items()}
    return losses_for_bootstrap(arm_losses=arms)


def estimate(
    *, year: str, contrast: Contrast, sites: tuple[str, ...] = DECISION, **kwargs: Any
) -> tuple[float, float, int]:
    """Return the mean paired difference in quantile score, in the mean absolute error, and rows."""
    losses = paired(
        year=year, treatment=contrast.treatment, reference=contrast.reference, sites=sites, **kwargs
    )
    means = losses.group_by("arm").agg(
        pl.col("qs").mean(), pl.col("ae").mean(), pl.len().alias("n")
    )
    treatment = means.filter(pl.col("arm") == contrast.treatment).row(0, named=True)
    reference = means.filter(pl.col("arm") == contrast.reference).row(0, named=True)
    return treatment["qs"] - reference["qs"], treatment["ae"] - reference["ae"], int(treatment["n"])


def interval(
    *,
    year: str,
    contrast: Contrast,
    level: float,
    sites: tuple[str, ...] = DECISION,
    where: pl.Expr | None = None,
) -> tuple[float, float]:
    """Return the month-resampled interval of the contrast."""
    losses = paired(
        year=year,
        treatment=contrast.treatment,
        reference=contrast.reference,
        sites=sites,
        where=where,
    )
    return bootstrap_difference_at_level(
        losses=losses,
        treatment=contrast.treatment,
        reference=contrast.reference,
        metric="qs",
        level=level,
        n_resamples=common.N_RESAMPLES,
    )


def sites_improving(*, year: str, contrast: Contrast, where: pl.Expr | None = None) -> int:
    """Return how many decision sites have a negative difference (treatment better)."""
    return sum(
        int(estimate(year=year, contrast=contrast, sites=(site,), where=where)[0] < 0.0)
        for site in DECISION
    )


def label_of(
    *,
    low: float,
    high: float,
    scored: float,
    replicate: float,
    improving: int,
    gain_points: float,
    min_sites: int,
) -> str:
    """Return the decision rule's label for one question.

    Args:
        low: The interval's lower end.
        high: The interval's upper end.
        scored: The scored year's estimate.
        replicate: The replicate year's estimate.
        improving: The number of decision sites that improve in the scored year.
        gain_points: The gain threshold (positive number of points).
        min_sites: The sites that must improve for `gain`.

    Returns:
        `gain`, `small gain`, `not replicated`, `worse`, or `tie`.
    """
    if low > 0.0:
        return "worse"
    if high < 0.0:
        if scored > -gain_points:
            return "small gain" if replicate < 0.0 else "not replicated"
        return "gain" if replicate < 0.0 and improving >= min_sites else "not replicated"
    return "tie"


WINTER_NIGHT: Final[pl.Expr] = pl.col("winter_night")
SUBSETS: Final[dict[str, pl.Expr]] = {
    "all rows": pl.lit(True),
    "Nov-Mar": pl.col("calendar_month").is_in(WINTER_BREAKDOWN_MONTHS),
    "Apr-Oct": ~pl.col("calendar_month").is_in(WINTER_BREAKDOWN_MONTHS),
    "winter nights": WINTER_NIGHT,
    "other than winter nights": ~WINTER_NIGHT,
    "cold snaps": pl.col("cold_snap"),
    "winter evening peaks": pl.col("evening_winter"),
    "BST nights": pl.col("bst_night"),
    "cold BST nights": pl.col("cold_bst_night"),
}


def subset_row(*, contrast: Contrast, year: str = "scored") -> dict[str, Any]:
    """Return a contrast's mean difference on every subset (winter breakdown), with row counts."""
    row: dict[str, Any] = {"contrast": contrast.name}
    for name, where in SUBSETS.items():
        losses = paired(
            year=year,
            treatment=contrast.treatment,
            reference=contrast.reference,
            sites=DECISION,
            where=where,
        )
        if losses.height == 0:
            row[name] = None
            row[f"{name} rows"] = 0
            continue
        point, _, rows = estimate(year=year, contrast=contrast, where=where)
        row[name] = point
        row[f"{name} rows"] = rows
    return row


def contrast_row(*, contrast: Contrast, level: float, where: pl.Expr | None) -> dict[str, Any]:
    """Return one contrast's row: scored year with interval, replicate beside, site counts."""
    scored, scored_ae, rows = estimate(year="scored", contrast=contrast, where=where)
    replicate, replicate_ae, _ = estimate(year="replicate", contrast=contrast, where=where)
    low, high = interval(year="scored", contrast=contrast, level=level, where=where)
    rep_low, rep_high = interval(year="replicate", contrast=contrast, level=level, where=where)
    rep_low95, rep_high95 = interval(
        year="replicate", contrast=contrast, level=EXPLORATORY_LEVEL, where=where
    )
    return {
        "contrast": contrast.name,
        "rows": rows,
        "scored QS": scored,
        "low": low,
        "high": high,
        "sites better (of 5)": sites_improving(year="scored", contrast=contrast, where=where),
        "replicate QS": replicate,
        "replicate low": rep_low,
        "replicate high": rep_high,
        "replicate low 95": rep_low95,
        "replicate high 95": rep_high95,
        "replicate sites better": sites_improving(year="replicate", contrast=contrast, where=where),
        "scored MAE of mean": scored_ae,
        "replicate MAE of mean": replicate_ae,
    }


def non_inferiority(*, contrast: Contrast) -> dict[str, Any]:
    """Return the contrast on every row outside winter nights, with its upper bound."""
    where = ~WINTER_NIGHT
    point, _, _ = estimate(year="scored", contrast=contrast, where=where)
    low, high = interval(year="scored", contrast=contrast, level=NIGHT_LEVEL, where=where)
    return {
        "estimate": point,
        "low": low,
        "high": high,
        "holds": bool(high <= NON_INFERIORITY_POINTS),
    }


def leave_one_site_out(*, contrast: Contrast, level: float, where: pl.Expr | None) -> pl.DataFrame:
    """Return the contrast with each decision site left out: estimate and interval."""
    rows = []
    for left_out in (None, *DECISION):
        sites = tuple(s for s in DECISION if s != left_out)
        point = estimate(year="scored", contrast=contrast, sites=sites, where=where)[0]
        low, high = interval(
            year="scored", contrast=contrast, level=level, sites=sites, where=where
        )
        rows.append(
            {
                "left out": left_out or "none",
                "estimate": point,
                "low": low,
                "high": high,
                "replicate": estimate(
                    year="replicate", contrast=contrast, sites=sites, where=where
                )[0],
            }
        )
    return pl.DataFrame(rows)


def own_arms() -> list[str]:
    """Return every stored arm of this experiment, in a fixed order."""
    base = ["b0", "mh", "mh5", shrunk_name(), "ns_l", "ns_u", "ns_h", "ns_hw", "ns_ut"]
    arms = []
    for name in base:
        arms += [name, f"{name}_k"]
        if name not in ("b0", "mh5"):
            arms.append(f"{name}_kb0")
    return arms


def absolute_table(*, where: pl.Expr | None = None) -> pl.DataFrame:
    """Return every arm's mean quantile score per site and pooled, for each year."""
    rows = []
    arms = [
        *own_arms(),
        "e114:t_all_k",
        "e114:h_site_k",
        *(f"p4:{name}" for name in PART_4_SETS),
    ]
    for year in YEARS:
        for arm in arms:
            try:
                frame = load_arm(year=year, arm=arm)
            except FileNotFoundError:
                continue
            if where is not None:
                frame = frame.filter(where)
            decision = frame.filter(pl.col("site_name").is_in(DECISION))
            row: dict[str, Any] = {
                "year": year,
                "arm": arm,
                "pooled (5 sites)": decision["qs"].mean(),
            }
            for site in common.SCORED_SITES:
                part = frame.filter(pl.col("site_name") == site)
                row[site] = part["qs"].mean() if part.height else None
            rows.append(row)
    return pl.DataFrame(rows)


def tail_table() -> pl.DataFrame:
    """Return coverage and pinball loss at q01, q05, q95, and q99 for the scored year."""
    rows = []
    for arm in ("b0", "b0_k", "mh", "mh_k", "ns_u_k", "ns_hw_k", "p4:manual_heuristic"):
        frame = load_arm(year="scored", arm=arm).filter(pl.col("site_name").is_in(DECISION))
        actual = frame["actual"].to_numpy().astype(np.float64)
        row: dict[str, Any] = {"arm": arm}
        for column, level in TAIL_LEVELS.items():
            predicted = frame[column].to_numpy().astype(np.float64)
            error = actual - predicted
            row[f"{column} coverage"] = common.as_float((actual <= predicted).mean())
            row[f"{column} pinball"] = float(
                np.maximum(level * error, (level - 1.0) * error).mean()
            )
        rows.append(row)
    return pl.DataFrame(rows)


def tail_flags(*, table: pl.DataFrame) -> list[str]:
    """Return the arms whose q95 or q99 coverage is more than a point further from nominal."""
    reference = {r["arm"]: r for r in table.iter_rows(named=True)}
    flags = []
    for row in table.iter_rows(named=True):
        if row["arm"] in ("b0_k", "b0", "p4:manual_heuristic"):
            continue
        base = reference["b0_k" if row["arm"].endswith("_k") else "b0"]
        for column in ("q95", "q99"):
            nominal = TAIL_LEVELS[column]
            moved = abs(row[f"{column} coverage"] - nominal) - abs(
                base[f"{column} coverage"] - nominal
            )
            if moved > TAIL_FLAG_POINTS / 100.0:
                flags.append(
                    f"{row['arm']} {column}: {100 * moved:.2f} points further from nominal"
                )
    return flags


def median_minus_mean() -> pl.DataFrame:
    """Return the mean of (q50 minus mean forecast) per site for B0 and the filtered arms."""
    rows = []
    for arm in ("b0", "b0_k", "mh_k", "ns_u_k", "ns_hw_k"):
        frame = load_arm(year="scored", arm=arm)
        row: dict[str, Any] = {"arm": arm}
        for site in common.SCORED_SITES:
            part = frame.filter(pl.col("site_name") == site)
            row[site] = (
                common.as_float((part["q50"] - part["mean"]).mean()) if part.height else None
            )
        rows.append(row)
    return pl.DataFrame(rows)


def cold_bst_counts() -> pl.DataFrame:
    """Return, per decision site and year, the cold nights in British Summer Time (nights)."""
    rows = []
    for year in ("replicate", "scored"):
        frame = load_arm(year=year, arm="b0_k").filter(pl.col("site_name").is_in(DECISION))
        nights = (
            frame.filter(pl.col("cold_bst_night"))
            .select("site_name", "local_date")
            .unique()
            .group_by("site_name")
            .len()
        )
        counts = dict(zip(nights["site_name"].to_list(), nights["len"].to_list(), strict=True))
        for site in DECISION:
            n = int(counts.get(site, 0))
            rows.append(
                {
                    "year": year,
                    "site": site,
                    "cold BST nights": n,
                    "UTC vs local": "ok" if n >= MIN_COLD_BST_NIGHTS else "cannot tell",
                }
            )
    return pl.DataFrame(rows)


def cold_snap_counts() -> pl.DataFrame:
    """Return the cold-snap days per decision site and winter, with a 'too few days' flag."""
    days = cold_days().filter(pl.col("site_name").is_in(DECISION) & pl.col("cold_snap"))
    table = days.group_by("site_name", "winter_year").len().rename({"len": "cold-snap days"})
    return table.with_columns(
        note=pl.when(pl.col("cold-snap days") < MIN_COLD_SNAP_DAYS)
        .then(pl.lit("too few days"))
        .otherwise(pl.lit("ok"))
    ).sort("site_name", "winter_year")


def external_contrast(*, treatment: str, where: pl.Expr | None) -> dict[str, Any]:
    """Return the treatment arm minus BG2 on the rows both sets hold (scored year)."""
    bg2 = pl.read_parquet(BG2_PATH)
    scale = 100.0
    bg2 = bg2.filter(pl.col("site").is_in(DECISION)).with_columns(
        *[(pl.col(c) * scale).alias(c) for c in (*QUANTILE_COLUMNS, "actual")]
    )
    actual = bg2["actual"].to_numpy().astype(np.float64)
    quantiles = bg2.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    bg2 = bg2.with_columns(qs=pl.Series(common.quantile_score(actual=actual, quantiles=quantiles)))
    ours = load_arm(year="scored", arm=treatment)
    if where is not None:
        ours = ours.filter(where)
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    joined = ours.select(*keys, pl.col("qs").alias("ours")).join(
        bg2.select(*keys, pl.col("qs").alias("bg2")), on=keys, how="inner"
    )
    if joined.height == 0:
        return {"rows": 0}
    return {
        "rows": joined.height,
        "treatment minus BG2": common.as_float((joined["ours"] - joined["bg2"]).mean()),
    }


def print_table(*, title: str, frame: pl.DataFrame) -> None:
    """Print a titled table with every row and column and collect it for the charts."""
    COLLECTED[title] = frame.to_dicts()
    print(f"\n== {title} ==")
    with pl.Config(
        tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=80, float_precision=4, tbl_width_chars=250
    ):
        print(frame)


COLLECTED: dict[str, list[dict[str, Any]]] = {}
"""Every printed table by title, written to `work/numbers.json` for the charts."""


def night_label(*, row: dict[str, Any], non_inf: dict[str, Any], powered: bool | None) -> str:
    """Return a night arm's label from its winter-night row and non-inferiority result.

    A night `gain` also needs the replicate winter to confirm it: its estimate below zero with a 95%
    interval wholly below zero, or at most half the scored estimate.
    """
    if powered is None:
        return "null control"
    if not powered:
        return "not powered"
    replicate_ok = row["replicate high 95"] < 0.0 or row["replicate QS"] <= 0.5 * row["scored QS"]
    label = label_of(
        low=row["low"],
        high=row["high"],
        scored=row["scored QS"],
        replicate=row["replicate QS"] if replicate_ok else 0.0,
        improving=row["sites better (of 5)"],
        gain_points=NIGHT_GAIN_POINTS,
        min_sites=MIN_SITES_NIGHT,
    )
    if not non_inf["holds"] and label in ("gain", "small gain"):
        return "worse on other rows"
    return label


RANK: Final[dict[str, int]] = {
    "gain": 4,
    "small gain": 3,
    "tie": 2,
    "not replicated": 1,
    "worse on other rows": 0,
    "null control": -1,
    "not powered": -1,
    "worse": -2,
}


def verdict_of(*, mh_label: str, night_labels: dict[str, str]) -> tuple[str, str]:
    """Return the verdict word and the NS label from the MH label and the four night labels."""
    best = max(
        (label for name, label in night_labels.items() if name != "ns_l"),
        key=lambda label: RANK[label],
    )
    labels = [mh_label, best]
    if all(label == "gain" for label in labels):
        verdict = "supported"
    elif any(label == "gain" for label in labels):
        verdict = "partly supported"
    elif "small gain" in labels:
        verdict = "small gain"
    elif "worse" in labels:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    return verdict, best


def main() -> None:  # noqa: PLR0915
    """Print every number the write-up quotes and write `work/numbers.json`."""
    gate2 = json.loads((common.WORK_DIR / "gate2.json").read_text())
    gate4 = json.loads((common.WORK_DIR / "gate4.json").read_text())
    gate4b = json.loads((common.WORK_DIR / "gate4b.json").read_text())
    shrunk = shrunk_name()
    print(f"MH-sh multiplier: {shrunk}")
    print_table(
        title="Cold-snap days (rule: coldest 10% of days of each winter)", frame=cold_snap_counts()
    )
    print_table(
        title="Cold nights in British Summer Time (daily mean < 8 C)", frame=cold_bst_counts()
    )
    print_table(title="Absolute quantile score (all rows)", frame=absolute_table())
    print_table(
        title="Absolute quantile score, winter nights",
        frame=absolute_table(where=WINTER_NIGHT),
    )
    print_table(
        title="Absolute quantile score, cold snaps",
        frame=absolute_table(where=pl.col("cold_snap")),
    )
    mh_contrasts = [
        Contrast("C1: MH-K - B0-K", "mh_k", "b0_k"),
        Contrast("MH - B0 (unfiltered)", "mh", "b0"),
        Contrast("MH-K with B0's filter - B0-K (control)", "mh_kb0", "b0_k"),
        Contrast("MH-K - MH-K with B0's filter (filter-fit effect)", "mh_k", "mh_kb0"),
        Contrast("MH-5-K - B0-K", "mh5_k", "b0_k"),
        Contrast("MH-sh-K - B0-K", f"{shrunk}_k", "b0_k"),
        Contrast("MH-K - H-site-K (experiment 114)", "mh_k", "e114:h_site_k"),
        Contrast("MH-K - T-all-K (experiment 114)", "mh_k", "e114:t_all_k"),
        Contrast("B0-K - lagless XGBoost with the filter", "b0_k", "p4:x_best_k"),
    ]
    c1 = mh_contrasts[0]
    rows = [contrast_row(contrast=c1, level=MH_LEVEL, where=None)]
    rows += [
        contrast_row(contrast=c, level=EXPLORATORY_LEVEL, where=None) for c in mh_contrasts[1:]
    ]
    print_table(title="MH contrasts (C1 at 97.5%, others 95%)", frame=pl.DataFrame(rows))
    control = mh_contrasts[2]
    c1_row = rows[0]
    control_estimate = estimate(year="scored", contrast=control)[0]
    half_width = 0.5 * (c1_row["high"] - c1_row["low"])
    mh_label = label_of(
        low=c1_row["low"],
        high=c1_row["high"],
        scored=c1_row["scored QS"],
        replicate=c1_row["replicate QS"],
        improving=c1_row["sites better (of 5)"],
        gain_points=MH_GAIN_POINTS,
        min_sites=MIN_SITES_MH,
    )
    control_label = label_of(
        low=interval(year="scored", contrast=control, level=MH_LEVEL)[0],
        high=interval(year="scored", contrast=control, level=MH_LEVEL)[1],
        scored=control_estimate,
        replicate=estimate(year="replicate", contrast=control)[0],
        improving=sites_improving(year="scored", contrast=control),
        gain_points=MH_GAIN_POINTS,
        min_sites=MIN_SITES_MH,
    )
    filter_control_moves = abs(control_estimate - c1_row["scored QS"]) > half_width
    if not gate2["passed"]:
        mh_label = "not powered"
    elif control_label != mh_label or filter_control_moves:
        mh_label = "inconclusive"
    print(
        f"\nMH label: {mh_label} (control label {control_label}, moves C1 by more than its "
        f"half-width: {filter_control_moves})"
    )
    print_table(
        title="C1 leave one site out",
        frame=leave_one_site_out(contrast=c1, level=MH_LEVEL, where=None),
    )
    night_contrasts = [
        Contrast(f"N{i}: {name.upper().replace('_', '-')}-K - B0-K", f"{name}_k", "b0_k")
        for i, name in enumerate(NIGHT_ARMS, start=1)
    ]
    night_rows = [
        contrast_row(contrast=c, level=NIGHT_LEVEL, where=WINTER_NIGHT) for c in night_contrasts
    ]
    print_table(title="Night contrasts on winter nights (99.375%)", frame=pl.DataFrame(night_rows))
    non_inf = {c.name: non_inferiority(contrast=c) for c in night_contrasts}
    print_table(
        title="Non-inferiority on rows other than winter nights (upper bound <= +0.02)",
        frame=pl.DataFrame([{"contrast": k, **v} for k, v in non_inf.items()]),
    )
    controls = [
        contrast_row(
            contrast=Contrast(f"{name}-K with B0's filter - B0-K", f"{name}_kb0", "b0_k"),
            level=NIGHT_LEVEL,
            where=WINTER_NIGHT,
        )
        for name in NIGHT_ARMS
    ]
    print_table(title="Night filter-fit controls on winter nights", frame=pl.DataFrame(controls))
    powered = {
        "ns_l": None,
        "ns_u": bool(gate4b["a_pass"]),
        "ns_h": bool(gate4b["c_pass"]),
        "ns_hw": bool(gate4b["b_pass"]),
    }
    night_labels = {
        name: night_label(row=row, non_inf=non_inf[c.name], powered=powered[name])
        for name, row, c in zip(NIGHT_ARMS, night_rows, night_contrasts, strict=True)
    }
    print(f"\nnight labels: {night_labels}")
    evolving = Contrast("NS-UT-K - NS-U-K (evolving against fixed step)", "ns_ut_k", "ns_u_k")
    evolving_row = contrast_row(contrast=evolving, level=EXPLORATORY_LEVEL, where=WINTER_NIGHT)
    print_table(
        title="Evolving against fixed UTC step, winter nights (95%, exploratory)",
        frame=pl.DataFrame([evolving_row]),
    )
    indexing = Contrast("NS-U-K - NS-L-K on BST nights", "ns_u_k", "ns_l_k")
    index_row = contrast_row(contrast=indexing, level=EXPLORATORY_LEVEL, where=pl.col("bst_night"))
    print_table(title="Indexing contrast (BST nights, 95%)", frame=pl.DataFrame([index_row]))
    index_label = "indexing not identified"
    if gate4b["a_pass"]:
        index_label = (
            "UTC"
            if index_row["high"] < 0.0
            else "local"
            if index_row["low"] > 0.0
            else "unresolved"
        )
    print(f"indexing label: {index_label}")
    verdict, ns_label = verdict_of(mh_label=mh_label, night_labels=night_labels)
    print(f"\nNS label: {ns_label}; verdict: {verdict}")
    best_night = min(
        (c for c in night_contrasts),
        key=lambda c: estimate(year="scored", contrast=c, where=WINTER_NIGHT)[0],
    )
    print(f"best night arm by winter-night estimate: {best_night.name}")
    subset_contrasts = [c1, *night_contrasts, indexing]
    print_table(
        title="Winter breakdown: scored-year mean difference by subset (points)",
        frame=pl.DataFrame([subset_row(contrast=c) for c in subset_contrasts]),
    )
    print_table(
        title="Winter breakdown, replicate year",
        frame=pl.DataFrame([subset_row(contrast=c, year="replicate") for c in subset_contrasts]),
    )
    winter_loses = [
        c.name
        for c in subset_contrasts
        if (v := subset_row(contrast=c)).get("Nov-Mar") is not None
        and v["Nov-Mar"] > 0.0
        and v["all rows"] is not None
        and v["all rows"] < 0.0
    ]
    print(f"\nwinter loses while the annual mean improves: {winter_loses}")
    for c in (c1, *night_contrasts):
        print(
            f"BG2 comparison for {c.treatment}:",
            external_contrast(treatment=c.treatment, where=None),
        )
    print_table(title="Tails (scored year)", frame=tail_table())
    print("tail flags:", tail_flags(table=tail_table()))
    print_table(title="Median minus mean forecast (q50 - mean), by site", frame=median_minus_mean())
    print_table(
        title="Night contrasts leaving one site out (best arm)",
        frame=leave_one_site_out(contrast=best_night, level=NIGHT_LEVEL, where=WINTER_NIGHT),
    )
    print("\n== Gates ==")
    print(
        json.dumps(
            {
                "gate2": gate2,
                "gate4_first_written": {k: v for k, v in gate4.items() if k != "mean_scores"},
                "gate4b": gate4b,
            },
            indent=2,
        )
    )
    (common.WORK_DIR / "numbers.json").write_text(
        json.dumps(
            {
                "tables": COLLECTED,
                "mh_label": mh_label,
                "night_labels": night_labels,
                "indexing_label": index_label,
                "verdict": verdict,
                "ns_label": ns_label,
                "best_night": best_night.name,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
