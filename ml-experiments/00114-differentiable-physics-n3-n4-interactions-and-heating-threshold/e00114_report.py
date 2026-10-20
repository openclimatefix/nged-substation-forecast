"""Print experiment 114's numbers: scores, the planned contrasts, labels, tails, checks, and gates.

Every number the write-up quotes comes from this script. It reads the stored prediction sets
(`<year>_<arm>`), scores each row with the quantile score (twice the mean pinball loss over the 11
levels, in points of effective capacity) on rows outside the hand-filtered switching masks, and
pairs arms row by row at day D+1.

- **Planned contrasts** (Bonferroni across two, so a 97.5% interval resampling whole calendar
  months): C1 = T-all-K minus B0-K and C2 = H-site-K minus B0-K, each with the count of decision
  sites that improve, the replicate year, the splits, leaving one site out, and the same contrast
  on the mean absolute error of the mean forecast. Beside them: the unfiltered contrasts and the
  filter-fit control.
- **Exploratory contrasts** at 95%: T1-K, T2-K, and T3-K each minus B0-K.
- **Labels and verdict** from the decision rule, **tails**, **checks**, **audit triggers**,
  **profiles**, the **capacity check**, and the gates.

    uv run python ml-experiments/00114-*/e00114_report.py > docs/ml-experiments/00114-*/report.txt
"""

import json
import sys
from dataclasses import dataclass
from functools import cache
from typing import Any, Final

import e00114_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_calendar import BAND_FIRST_HOURS, BAND_NAMES
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import losses_for_bootstrap

SITE_OF_ID: Final[dict[int, str]] = {7: "D9", 8: "D11", 10: "D10", 16: "D14", 17: "D5", 18: "D16"}
DECISION: Final[tuple[str, ...]] = common.DECISION_SITES
PLANNED_LEVEL: Final[float] = common.SCORED_LEVEL
EXPLORATORY_LEVEL: Final[float] = 95.0
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
GAIN_THRESHOLD_POINTS: Final[float] = 0.03
MIN_SITES_IMPROVING: Final[int] = 4
TAIL_LEVELS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
TAIL_FLAG_POINTS: Final[float] = 1.0
PART_4_DIR: Final = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
)
PART_4_C4_HALF_WIDTH: Final[float] = 0.023
"""Half the 97.5% interval of Part 4's C4 (A2 minus A3), the sister contrast on these rows."""
PART_4_SETS: Final[tuple[str, ...]] = (
    "a2",
    "wk",
    "a1",
    "x_best",
    "x_best_k",
    "climatology",
    "manual_heuristic",
)
OWN_ARMS: Final[tuple[str, ...]] = (
    "b0",
    "b0_k",
    "t1",
    "t1_k",
    "t2",
    "t2_k",
    "t3",
    "t3_k",
    "t_all",
    "t_all_k",
    "t_all_kb0",
    "h_site",
    "h_site_k",
)
YEARS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")


@dataclass(frozen=True)
class Contrast:
    """A treatment arm minus a reference arm.

    Attributes:
        name: The contrast's label in the report.
        treatment: The treatment arm's name.
        reference: The reference arm's name.
        planned: True for the two pre-registered contrasts.
    """

    name: str
    treatment: str
    reference: str
    planned: bool = False


C1: Final[Contrast] = Contrast("C1 (N3): T-all-K - B0-K", "t_all_k", "b0_k", planned=True)
C2: Final[Contrast] = Contrast("C2 (N4): H-site-K - B0-K", "h_site_k", "b0_k", planned=True)
BESIDE: Final[tuple[Contrast, ...]] = (
    Contrast("C1 beside: T-all - B0 (unfiltered)", "t_all", "b0"),
    Contrast("C1 control: T-all-K with B0's filter - B0-K", "t_all_kb0", "b0_k"),
    Contrast("C1 filter-fit effect: T-all-K - T-all-K with B0's filter", "t_all_k", "t_all_kb0"),
    Contrast("C2 beside: H-site - B0 (unfiltered)", "h_site", "b0"),
)
EXPLORATORY: Final[tuple[Contrast, ...]] = (
    Contrast("T1-K - B0-K (heating by day type)", "t1_k", "b0_k"),
    Contrast("T2-K - B0-K (heating by half-band)", "t2_k", "b0_k"),
    Contrast("T3-K - B0-K (lighting by day type)", "t3_k", "b0_k"),
    Contrast("T1 - B0 (unfiltered)", "t1", "b0"),
    Contrast("T2 - B0 (unfiltered)", "t2", "b0"),
    Contrast("T3 - B0 (unfiltered)", "t3", "b0"),
)
REFERENCE_CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast("B0-K - Part 4's WK", "b0_k", "p4:wk"),
    Contrast("B0-K - Part 4's A2", "b0_k", "p4:a2"),
    Contrast("B0-K - lagless XGBoost with the filter", "b0_k", "p4:x_best_k"),
    Contrast("B0-K - manual heuristic", "b0_k", "p4:manual_heuristic"),
    Contrast("B0-K - climatology", "b0_k", "p4:climatology"),
)


def folder_of(*, year: str, arm: str) -> Any:
    """Return the folder that holds an arm's stored set for a year."""
    if arm.startswith("p4:"):
        return PART_4_DIR / "predictions" / f"{year}_{arm.removeprefix('p4:')}"
    return common.DATA_DIR / "predictions" / f"{year}_{arm}"


@cache
def load_arm(*, year: str, arm: str, drop_switching: bool = True) -> pl.DataFrame:
    """Read a stored set with its per-row quantile score, mean absolute error, and hour band.

    Args:
        year: `tuning`, `replicate`, or `scored`.
        arm: The arm's name; `p4:` prefixes a stored set of experiment 106.
        drop_switching: True to leave out the rows inside the switching masks.

    Returns:
        The set with columns `site_name`, `qs`, `ae`, `band`, and `calendar_month` added.
    """
    frame = pl.read_parquet(folder_of(year=year, arm=arm) / "predictions.parquet")
    if drop_switching:
        frame = frame.filter(~pl.col("switching_mask"))
    quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    local_hour = (
        (pl.col("valid_time") - pl.duration(minutes=30))
        .dt.convert_time_zone("Europe/London")
        .dt.hour()
    )
    band = (
        pl.when(local_hour >= BAND_FIRST_HOURS[3])
        .then(3)
        .when(local_hour >= BAND_FIRST_HOURS[2])
        .then(2)
        .when(local_hour >= BAND_FIRST_HOURS[1])
        .then(1)
        .otherwise(0)
    )
    return frame.with_columns(
        qs=pl.Series(common.quantile_score(actual=actual, quantiles=quantiles)),
        ae=(pl.col("actual") - pl.col("mean")).abs(),
        site_name=pl.col("time_series_id").replace_strict(SITE_OF_ID),
        band=band,
        calendar_month=pl.col("valid_time").dt.month(),
    )


def paired(
    *,
    year: str,
    treatment: str,
    reference: str,
    sites: tuple[str, ...],
    drop_switching: bool = True,
    months: tuple[int, ...] | None = None,
    band: int | None = None,
) -> pl.DataFrame:
    """Return both arms' rows on the sites, paired, as the frame `studies.bootstrap` reads.

    Args:
        year: The scored year.
        treatment: The treatment arm.
        reference: The reference arm.
        sites: The sites to keep.
        drop_switching: True to leave out the rows inside the switching masks.
        months: Calendar months of the valid time to keep, or `None` for all.
        band: The hour band to keep, or `None` for all.

    Returns:
        The paired loss frame.
    """
    arms = {}
    for name in (treatment, reference):
        frame = load_arm(year=year, arm=name, drop_switching=drop_switching).filter(
            pl.col("site_name").is_in(sites)
        )
        if months is not None:
            frame = frame.filter(pl.col("calendar_month").is_in(months))
        if band is not None:
            frame = frame.filter(pl.col("band") == band)
        arms[name] = frame.select(
            "time_series_id", "power_fcst_init_time", "valid_time", "seed", "qs", "ae"
        )
    first, second = list(arms.values())
    keys = ["time_series_id", "power_fcst_init_time", "valid_time", "seed"]
    shared = first.select(keys).join(second.select(keys), on=keys, how="inner")
    arms = {name: frame.join(shared, on=keys, how="inner") for name, frame in arms.items()}
    return losses_for_bootstrap(arm_losses=arms)


def estimate(
    *,
    year: str,
    contrast: Contrast,
    sites: tuple[str, ...] = DECISION,
    **kwargs: Any,
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
    *, year: str, contrast: Contrast, level: float, sites: tuple[str, ...] = DECISION
) -> tuple[float, float]:
    """Return the month-resampled interval of the contrast."""
    losses = paired(
        year=year, treatment=contrast.treatment, reference=contrast.reference, sites=sites
    )
    return bootstrap_difference_at_level(
        losses=losses,
        treatment=contrast.treatment,
        reference=contrast.reference,
        metric="qs",
        level=level,
        n_resamples=common.N_RESAMPLES,
    )


def sites_improving(*, year: str, contrast: Contrast) -> int:
    """Return how many decision sites have a negative difference (treatment better)."""
    return sum(
        int(estimate(year=year, contrast=contrast, sites=(site,))[0] < 0.0) for site in DECISION
    )


def label_of(*, low: float, high: float, scored: float, replicate: float, improving: int) -> str:
    """Return the decision rule's label for one question.

    Args:
        low: The 97.5% interval's lower end.
        high: The upper end.
        scored: The scored year's estimate.
        replicate: The replicate year's estimate.
        improving: The number of decision sites that improve in the scored year.

    Returns:
        `gain`, `small gain`, `not replicated`, `worse`, or `tie`.
    """
    if low > 0.0:
        return "worse"
    if high < 0.0:
        if scored > -GAIN_THRESHOLD_POINTS:
            return "small gain"
        return "gain" if replicate < 0.0 and improving >= MIN_SITES_IMPROVING else "not replicated"
    return "tie"


def contrast_row(*, contrast: Contrast, level: float) -> dict[str, Any]:
    """Return one contrast's row: the scored year with interval and splits, the replicate beside."""
    row: dict[str, Any] = {"contrast": contrast.name}
    scored, scored_ae, _ = estimate(year="scored", contrast=contrast)
    replicate, replicate_ae, _ = estimate(year="replicate", contrast=contrast)
    low, high = interval(year="scored", contrast=contrast, level=level)
    row.update(
        {
            "scored QS": scored,
            "low": low,
            "high": high,
            "sites better (of 5)": sites_improving(year="scored", contrast=contrast),
            "replicate QS": replicate,
            "replicate sites better": sites_improving(year="replicate", contrast=contrast),
            "scored MAE of mean": scored_ae,
            "replicate MAE of mean": replicate_ae,
            "all rows": estimate(year="scored", contrast=contrast, drop_switching=False)[0],
            "Nov-Mar": estimate(year="scored", contrast=contrast, months=WINTER_MONTHS)[0],
            "Apr-Oct": estimate(
                year="scored",
                contrast=contrast,
                months=tuple(m for m in range(1, 13) if m not in WINTER_MONTHS),
            )[0],
        }
    )
    for position, name in enumerate(BAND_NAMES):
        row[f"band {name}"] = estimate(year="scored", contrast=contrast, band=position)[0]
    return row


def leave_one_site_out(*, contrast: Contrast) -> pl.DataFrame:
    """Return the contrast with each decision site left out: estimate and 97.5% interval."""
    rows = []
    for left_out in (None, *DECISION):
        sites = tuple(s for s in DECISION if s != left_out)
        point = estimate(year="scored", contrast=contrast, sites=sites)[0]
        low, high = interval(year="scored", contrast=contrast, level=PLANNED_LEVEL, sites=sites)
        rows.append(
            {
                "left out": left_out or "none",
                "estimate": point,
                "low": low,
                "high": high,
                "replicate": estimate(year="replicate", contrast=contrast, sites=sites)[0],
            }
        )
    return pl.DataFrame(rows)


def absolute_table() -> pl.DataFrame:
    """Return every arm's mean quantile score per site and pooled, for each year."""
    rows = []
    arms = [*OWN_ARMS, *(f"p4:{name}" for name in PART_4_SETS)]
    for year in YEARS:
        for arm in arms:
            try:
                frame = load_arm(year=year, arm=arm)
            except FileNotFoundError:
                continue
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
    for arm in ("b0", "b0_k", "t_all", "t_all_k", "h_site", "h_site_k", "p4:manual_heuristic"):
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
    """Return the arms with q95 or q99 coverage over a point further from nominal than their base.

    A filtered arm is compared with B0-K and an unfiltered arm with B0.
    """
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
    for arm in ("b0", "b0_k", "t_all_k", "h_site_k"):
        frame = load_arm(year="scored", arm=arm)
        row: dict[str, Any] = {"arm": arm}
        for site in common.SCORED_SITES:
            part = frame.filter(pl.col("site_name") == site)
            row[site] = (
                common.as_float((part["q50"] - part["mean"]).mean()) if part.height else None
            )
        rows.append(row)
    return pl.DataFrame(rows)


def verdict_of(*, labels: dict[str, str]) -> str:
    """Return the verdict word from the two labels (`n3`, `n4`)."""
    values = list(labels.values())
    gains = values.count("gain")
    if gains == len(values):
        return "supported"
    if gains >= 1:
        return "partly supported"
    if "small gain" in values:
        return "small gain"
    if "worse" in values:
        return "refuted"
    return "inconclusive"


COLLECTED: dict[str, list[dict[str, Any]]] = {}
"""Every printed table by title, written to `work/numbers.json` for the charts."""


def print_contrasts(*, title: str, frame: pl.DataFrame) -> None:
    """Print a contrast table with one column per contrast, which reads better than 16 columns."""
    COLLECTED[title] = frame.to_dicts()
    shown = frame.with_columns(pl.col("contrast").str.slice(0, 40)).transpose(
        include_header=True, header_name="measure", column_names="contrast"
    )
    print(f"\n== {title} ==")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=60, tbl_width_chars=250):
        print(shown)


def print_table(*, title: str, frame: pl.DataFrame) -> None:
    """Print a titled table with every row and column."""
    COLLECTED[title] = frame.to_dicts()
    print(f"\n== {title} ==")
    with pl.Config(
        tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=80, float_precision=4, tbl_width_chars=250
    ):
        print(frame)


def gate_summary() -> dict[str, Any]:
    """Print the known-answer tests and gates and return their records."""
    test1 = (common.WORK_DIR / "test1.txt").read_text().strip().splitlines()[-1]
    gate2 = json.loads((common.WORK_DIR / "gate2.json").read_text())
    gate3 = json.loads((common.WORK_DIR / "gate3.json").read_text())
    decisions_hash = (common.WORK_DIR / "decisions.sha256").read_text().strip()
    print("\n== known-answer tests and gates ==")
    print(f"test 1: {test1}")
    print(f"decisions.json sha256 (written before the gates): {decisions_hash}")
    print(
        f"gate 2 (unfiltered, tuned weight {gate2['tuned_penalty_weight']:g}, amplitude "
        f"{gate2['amplitude']:.3f}, pilot oracle gain {gate2['pilot_gain']:.4f}): "
        f"passed {gate2['passed']}"
    )
    for weight, entry in gate2["by_weight"].items():
        for kind, text in (("i", "truth I"), ("zero", "truth 0")):
            part = entry[kind]
            print(
                f"  weight {weight:>6} {text}: T-all minus B0 {part['mean_t_all_minus_b0']:+.4f} "
                f"({part['interval_95'][0]:+.4f} to {part['interval_95'][1]:+.4f}), "
                f"T-all better in {100 * part['share_t_all_better']:.0f} of 100 draws, "
                f"by site {part['wins_by_site']}"
            )
    print(f"gate 3: {json.dumps(gate3)}")
    return {"test1": test1, "gate2": gate2, "gate3": gate3}


def profile_summary(*, gate3: dict[str, Any]) -> None:
    """Print each substation's threshold profile at each cut and the printing rule's outcome."""
    records = json.loads((common.WORK_DIR / "profiles.json").read_text())
    rows = [
        {
            "site": r["site"],
            "cut": r["label"],
            "argmin C": r["argmin_c"],
            "interval low": r["interval_low_c"],
            "interval high": r["interval_high_c"],
            "nats at 15.5": r["nats_at_base"],
            "within 2 nats from": r["within_2_nats_low_c"],
            "within 2 nats to": r["within_2_nats_high_c"],
            "printable": r["printable"],
        }
        for r in records
    ]
    print_table(title="threshold profiles (tau = gate 3 median)", frame=pl.DataFrame(rows))
    scored = [r for r in records if r["label"] == "scored" and r["site"] in DECISION]
    flat = [
        r["site"] for r in scored if r["within_2_nats_high_c"] - r["within_2_nats_low_c"] >= 3.0
    ]
    edge = [r["site"] for r in scored if r["argmin_c"] in (common.MIN_HINGE_C, common.MAX_HINGE_C)]
    printable = [r["site"] for r in scored if r["printable"]]
    print(
        f"scored cut: printable sites {printable}; profile within 2 nats over 3 C or more at "
        f"{flat}; arg-min at a grid edge at {edge}"
    )
    print(
        f"gate 3 label: {gate3['label']}; tau median {gate3['tau_median']:.1f} "
        "(N5's 27 for comparison)"
    )


def capacity_summary() -> None:
    """Print the descriptive capacity check."""
    rows = json.loads((common.WORK_DIR / "capacity_check.json").read_text())
    print_table(
        title="capacity check (fitted capacity, scored cut, points)", frame=pl.DataFrame(rows)
    )


def tuning_summary() -> None:
    """Print the tuning year's scores, chosen weights, and the new columns' degrees of freedom."""
    record = json.loads((common.WORK_DIR / "tuning.json").read_text())
    decisions = json.loads((common.WORK_DIR / "decisions.json").read_text())
    rows = [
        {
            "arm": arm,
            "tuning-year QS unfiltered": record["score_unfiltered"][arm],
            "tuning-year QS filtered": record["score_filtered"][arm],
        }
        for arm in record["score_filtered"]
    ]
    print_table(title="tuning year (2023-07 to 2024-06), five sites", frame=pl.DataFrame(rows))
    print(f"chosen penalty weights: {decisions['lambda']}")
    print(
        "new columns' effective degrees of freedom by site: "
        f"{json.dumps(record['new_column_edf_by_site'])}"
    )


def trigger_table(*, rows: dict[str, Any], gates: dict[str, Any]) -> pl.DataFrame:
    """Return the seven audit triggers with whether each fired and the evidence."""
    c1, c2 = rows["c1"], rows["c2"]
    tuning, _, _ = estimate(year="tuning", contrast=C1)
    tuning_low, tuning_high = interval(year="tuning", contrast=C1, level=PLANNED_LEVEL)
    tuning_half = (tuning_high - tuning_low) / 2.0
    control = rows["control"]
    reference = rows["wk_gap"]
    out = []

    def add(number: int, fired: bool, evidence: str) -> None:
        out.append({"trigger": number, "fired": fired, "evidence": evidence})

    controls_failed = (
        "FAILED" in gates["test1"]
        or not gates["gate2"]["passed"]
        or not gates["gate3"]["placebo_pass"]
    )
    add(
        1,
        controls_failed,
        f"test 1 '{gates['test1']}'; gate 2 passed {gates['gate2']['passed']}; "
        f"N4 placebo passed {gates['gate3']['placebo_pass']}",
    )
    contradiction = (
        c1["scored QS"] < -0.10
        or gates["gate3"]["identified"]
        or reference > 0.10
        or rows["printable"]
    )
    add(
        2,
        contradiction,
        f"C1 {c1['scored QS']:+.4f}; gate 3 identified {gates['gate3']['identified']}; "
        f"a site passes the printing rule {rows['printable']}; B0-K minus WK {reference:+.4f}",
    )
    sign_disagrees = [
        c["contrast"] for c in (c1, c2) if (c["scored QS"] < 0.0) != (c["scored MAE of mean"] < 0.0)
    ]
    add(
        3,
        bool(sign_disagrees),
        f"score and mean-error contrasts disagree in sign for {sign_disagrees or 'none'}",
    )
    flips = rows["leave_out_flips"]
    add(4, bool(flips), f"leaving one site out flips sign or interval status: {flips or 'none'}")
    nested = tuning > tuning_half
    moved = abs(control - c1["scored QS"]) > (c1["high"] - c1["low"]) / 2.0
    add(
        5,
        bool(nested or moved),
        f"C1 on the tuning year {tuning:+.4f} (half-width {tuning_half:.4f}); filter-fit control "
        f"moves C1 by {abs(control - c1['scored QS']):.4f} (half-width "
        f"{(c1['high'] - c1['low']) / 2.0:.4f}); the 15.5 hinge reproduces B0 exactly (test 1)",
    )
    widths = {c["contrast"]: (c["high"] - c["low"]) / 2.0 for c in (c1, c2)}
    wide = [
        n
        for n, w in widths.items()
        if w > 2.0 * PART_4_C4_HALF_WIDTH or w < 0.5 * PART_4_C4_HALF_WIDTH
    ]
    add(
        6,
        bool(wide),
        f"half-widths {json.dumps({k: round(v, 4) for k, v in widths.items()})} against "
        f"Part 4's C4 {PART_4_C4_HALF_WIDTH}",
    )
    add(
        7,
        rows["labels"]["n3"] == "gain" or rows["labels"]["n4"] == "gain",
        f"labels {rows['labels']}: a gain would change GAM-K's base; the result is one year plus "
        "a replicate at the screening tier",
    )
    return pl.DataFrame(out)


def main() -> None:
    """Print every number of the write-up."""
    gates = gate_summary()
    tuning_summary()
    print_table(
        title="absolute quantile scores (points), rows outside switching masks",
        frame=absolute_table(),
    )
    contrast_rows = {
        "c1": contrast_row(contrast=C1, level=PLANNED_LEVEL),
        "c2": contrast_row(contrast=C2, level=PLANNED_LEVEL),
    }
    print_contrasts(
        title=f"planned contrasts ({PLANNED_LEVEL}% intervals, whole calendar months)",
        frame=pl.DataFrame(list(contrast_rows.values())),
    )
    print_contrasts(
        title="beside the planned contrasts",
        frame=pl.DataFrame([contrast_row(contrast=c, level=PLANNED_LEVEL) for c in BESIDE]),
    )
    print_contrasts(
        title="exploratory contrasts (95%)",
        frame=pl.DataFrame(
            [contrast_row(contrast=c, level=EXPLORATORY_LEVEL) for c in EXPLORATORY]
        ),
    )
    print_contrasts(
        title="B0-K against the references (95%)",
        frame=pl.DataFrame(
            [contrast_row(contrast=c, level=EXPLORATORY_LEVEL) for c in REFERENCE_CONTRASTS]
        ),
    )
    leave_out = {key: leave_one_site_out(contrast=c) for key, c in (("c1", C1), ("c2", C2))}
    for contrast, key in ((C1, "c1"), (C2, "c2")):
        print_table(title=f"leaving one site out: {contrast.name}", frame=leave_out[key])
    d9_rows = []
    for contrast in (C1, C2):
        point, point_ae, rows_n = estimate(year="scored", contrast=contrast, sites=("D9",))
        d9_rows.append(
            {
                "contrast": contrast.name,
                "scored QS": point,
                "scored MAE of mean": point_ae,
                "rows": rows_n,
            }
        )
    print_table(
        title="D9 beside the decision sites (scored year, descriptive)", frame=pl.DataFrame(d9_rows)
    )
    labels = {
        "n3": label_of(
            low=contrast_rows["c1"]["low"],
            high=contrast_rows["c1"]["high"],
            scored=contrast_rows["c1"]["scored QS"],
            replicate=contrast_rows["c1"]["replicate QS"],
            improving=contrast_rows["c1"]["sites better (of 5)"],
        )
    }
    if gates["gate3"]["identified"]:
        labels["n4"] = label_of(
            low=contrast_rows["c2"]["low"],
            high=contrast_rows["c2"]["high"],
            scored=contrast_rows["c2"]["scored QS"],
            replicate=contrast_rows["c2"]["replicate QS"],
            improving=contrast_rows["c2"]["sites better (of 5)"],
        )
    else:
        labels["n4"] = "not identified"
    decision_labels = {k: v for k, v in labels.items() if v != "not identified"}
    verdict = verdict_of(labels=decision_labels)
    print(f"\n== labels and verdict ==\nlabels {labels}; verdict {verdict}")
    print_table(title="tails, scored year, five decision sites", frame=(tails := tail_table()))
    print(
        "tail flags (q95 or q99 more than 1 point further from nominal than the matching "
        f"B0 arm): {tail_flags(table=tails) or 'none'}"
    )
    print_table(
        title="median forecast minus mean forecast per site (points), scored year",
        frame=median_minus_mean(),
    )
    profile_summary(gate3=gates["gate3"])
    capacity_summary()
    records = json.loads((common.WORK_DIR / "profiles.json").read_text())
    printable = [
        r["site"]
        for r in records
        if r["label"] == "scored" and r["site"] in DECISION and r["printable"]
    ]
    flips = []
    for key in ("c1", "c2"):
        table = leave_out[key]
        base_row = table.row(0, named=True)
        flips.extend(
            f"{key} without {row['left out']}"
            for row in table.iter_rows(named=True)
            if (row["estimate"] < 0.0) != (base_row["estimate"] < 0.0)
            or (row["high"] < 0.0) != (base_row["high"] < 0.0)
            or (row["low"] > 0.0) != (base_row["low"] > 0.0)
        )
    wk_gap = estimate(year="scored", contrast=REFERENCE_CONTRASTS[0])[0]
    control = estimate(year="scored", contrast=BESIDE[1])[0]
    triggers = trigger_table(
        rows={
            "c1": contrast_rows["c1"],
            "c2": contrast_rows["c2"],
            "control": control,
            "wk_gap": wk_gap,
            "printable": printable,
            "leave_out_flips": flips,
            "labels": labels,
        },
        gates=gates,
    )
    print_table(title="audit triggers", frame=triggers)
    COLLECTED["labels and verdict"] = [{"labels": labels, "verdict": verdict}]
    (common.WORK_DIR / "numbers.json").write_text(json.dumps(COLLECTED, default=str) + "\n")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
