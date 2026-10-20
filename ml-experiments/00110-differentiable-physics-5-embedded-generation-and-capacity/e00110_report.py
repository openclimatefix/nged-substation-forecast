"""Print experiment 110's numbers.

The tables are the absolute scores, the planned contrasts, the capacity labels, and the checks.

Every number the write-up quotes comes from this script. It reads the stored prediction sets
(`<cut>_<arm>`, each holding a filtered and an unfiltered forecast), scores each row with the
quantile score (twice the mean pinball loss over the 11 levels, in points of effective capacity)
on rows outside the hand-filtered switching masks, and pairs arms row by row.

- **Planned contrasts** (Bonferroni across the five, so a 99% interval resampling whole calendar
  months): C1 to C5, filtered, for the scored year and, beside it, the replicate year, with the
  count of sites that improve, the unfiltered contrast, the estimate without D16, the
  November-to-March and April-to-October estimates, the summer-midday estimate, and the same
  contrast on the mean absolute error of the mean forecast.
- **Exploratory contrasts** at 95%.
- **The capacity rule**: `identified`, `robust`, and `consistent` labels, from
  `work/capacity.parquet`.
- **Checks**: the median forecast minus the mean forecast per site and arm, and tail calibration
  (coverage and pinball loss at q01, q05, q95, and q99).

    uv run python ml-experiments/00110-*/e00110_report.py > docs/ml-experiments/00110-*/report.txt
"""

import json
from dataclasses import dataclass
from functools import cache
from typing import Final

import e00110_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS
from studies.experiment_store import losses_for_bootstrap

N_PLANNED: Final[int] = 5
PLANNED_LEVEL: Final[float] = 100.0 - 5.0 / N_PLANNED
EXPLORATORY_LEVEL: Final[float] = 95.0
N_RESAMPLES: Final[int] = 20000
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
MIDDAY_HOURS_UTC: Final[tuple[int, int]] = (10, 14)
TAIL_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time", "seed"]


@dataclass(frozen=True)
class Side:
    """One side of a contrast: a stored arm, filtered or not.

    Attributes:
        arm: The arm's name.
        filtered: True to read the filtered forecast, False for the unfiltered one.
        part_4: True to read Part 4's stored set of that name, which has no `filtered` column.
    """

    arm: str
    filtered: bool = True
    part_4: bool = False

    @property
    def label(self) -> str:
        """The name the paired frame uses for this side."""
        return f"{self.arm}{'+K' if self.filtered else ''}{'(p4)' if self.part_4 else ''}"


@dataclass(frozen=True)
class Contrast:
    """A treatment minus a reference, with its scope and label.

    Attributes:
        name: The contrast's label in the report.
        treatment: The treatment side.
        reference: The reference side.
        planned: True for the five pre-registered contrasts.
        sites: The sites the contrast is read on.
        years: The years the contrast exists in.
    """

    name: str
    treatment: Side
    reference: Side
    planned: bool = False
    sites: tuple[str, ...] = common.DECISION_SITES
    years: tuple[str, ...] = ("scored", "replicate")


@cache
def site_of_id() -> dict[int, str]:
    """Return the map from series identifier to D-label."""
    return {site.table.time_series_id: label for label, site in common.load_cache().sites.items()}


@cache
def tuned() -> dict:
    """Return this experiment's tuned settings."""
    return json.loads((common.WORK_DIR / "tuned.json").read_text())


@cache
def part_4_bar() -> str:
    """Return the name of Part 4's C6 bar: `x_best` or `x_bestamq`, chosen on the tuning year."""
    chosen = json.loads((common.PART_4_DIR / "work" / "tuned_xgb.json").read_text())
    return str(chosen["bar"]) if "bar" in chosen else "x_best"


def build_contrasts() -> tuple[Contrast, ...]:
    """Return every contrast: the five planned ones and the exploratory ones."""
    c4_arm = tuned()["c4_arm"]
    bar = part_4_bar()
    mva = common.MVA_SITES
    return (
        Contrast(
            "C1 generation side harms MW forecast: B-gen - B-D1",
            Side("b_gen"),
            Side("b_d1"),
            planned=True,
        ),
        Contrast("C2 the meter: L0 - N on |S|", Side("l0"), Side("n"), planned=True, sites=mva),
        Contrast("C3 learned Q: LQ - L0 on |S|", Side("lq"), Side("l0"), planned=True, sites=mva),
        Contrast(
            f"C4 fitted against pinned capacity: {c4_arm} - B",
            Side(c4_arm),
            Side("b"),
            planned=True,
        ),
        Contrast(
            f"C5 DP against lagless XGBoost: B - {bar} (both unfiltered)",
            Side("b", filtered=False),
            Side(bar, filtered=False, part_4=True),
            planned=True,
        ),
        Contrast("C5 beside: B - X-best with the filter", Side("b"), Side("x_best_k", part_4=True)),
        Contrast("B-node - B-D1", Side("b_node"), Side("b_d1")),
        Contrast("B-clip - B (all MW sites)", Side("b_clip"), Side("b")),
        Contrast("B-clip - B at D5", Side("b_clip"), Side("b"), sites=("D5",)),
        Contrast("B-gen - B-node at D5", Side("b_gen"), Side("b_node"), sites=("D5",)),
        Contrast("K-base - B", Side("k_base"), Side("b")),
        Contrast("K-i - K-base", Side("k_i"), Side("k_base")),
        Contrast("K-ii - K-base", Side("k_ii"), Side("k_base")),
        Contrast("K-iii - K-base", Side("k_iii"), Side("k_base")),
        Contrast("K-ii+iii - K-base", Side("k_iiiii"), Side("k_base")),
        Contrast("K-i - B", Side("k_i"), Side("b")),
        Contrast("K-ii - B", Side("k_ii"), Side("b")),
        Contrast("K-iii - B", Side("k_iii"), Side("b")),
        Contrast(
            "L0 - G-dark (solar term on sunlit MVA rows)", Side("l0"), Side("g_dark"), sites=mva
        ),
        Contrast("N - G-dark", Side("n"), Side("g_dark"), sites=mva),
        Contrast("B - climatology", Side("b"), Side("climatology", filtered=False, part_4=True)),
        Contrast(
            "B - manual heuristic", Side("b"), Side("manual_heuristic", filtered=False, part_4=True)
        ),
        Contrast("B-D1 - B (Part 4's A3 - A2)", Side("b_d1"), Side("b")),
    )


@cache
def load_side(*, label: str, side: Side, drop_switching: bool = True) -> pl.DataFrame:
    """Read a stored set's rows of one side, with each row's score and error.

    Args:
        label: The cut label, `replicate` or `scored`.
        side: The arm and whether its filtered forecast is wanted.
        drop_switching: True to leave out the rows inside the switching masks.

    Returns:
        The rows with columns `site_name`, `qs`, and `ae` added.
    """
    root = common.PART_4_DIR if side.part_4 else common.DATA_DIR
    frame = pl.read_parquet(root / "predictions" / f"{label}_{side.arm}" / "predictions.parquet")
    if not side.part_4:
        frame = frame.filter(pl.col("filtered") == side.filtered)
    if drop_switching:
        frame = frame.filter(~pl.col("switching_mask"))
    quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    return frame.with_columns(
        qs=pl.Series(common.quantile_score(actual=actual, quantiles=quantiles)),
        ae=(pl.col("actual") - pl.col("mean")).abs(),
        site_name=pl.col("time_series_id").replace_strict(site_of_id()),
    )


def paired(
    *,
    label: str,
    treatment: Side,
    reference: Side,
    sites: tuple[str, ...],
    months: tuple[int, ...] | None = None,
    midday: bool = False,
    dark: bool | None = None,
    drop_switching: bool = True,
) -> pl.DataFrame:
    """Return both sides' rows on the sites, paired, as the frame `studies.bootstrap` reads.

    Args:
        label: The cut label.
        treatment: The treatment side.
        reference: The reference side.
        sites: The sites to keep.
        months: Calendar months of the valid time to keep, or `None` for all.
        midday: True to keep only 10:00 to 14:00 UTC valid times.
        dark: True to keep only dark rows, False for only sunlit rows, `None` for all (MVA arms).
        drop_switching: True to leave out the rows inside the switching masks.

    Returns:
        The paired loss frame, with `arm` set to each side's label.
    """
    sides = {}
    for side in (treatment, reference):
        frame = load_side(label=label, side=side, drop_switching=drop_switching).filter(
            pl.col("site_name").is_in(sites)
        )
        if months is not None:
            frame = frame.filter(pl.col("valid_time").dt.month().is_in(months))
        if midday:
            frame = frame.filter(
                (pl.col("valid_time") - pl.duration(minutes=30))
                .dt.hour()
                .is_between(MIDDAY_HOURS_UTC[0], MIDDAY_HOURS_UTC[1] - 1)
            )
        if dark is not None:
            reading = load_side(label=label, side=treatment, drop_switching=drop_switching)
            if "dark" in reading.columns:
                keys = reading.filter((pl.col("dark") > 0.5) == dark).select(KEYS)
                frame = frame.join(keys, on=KEYS, how="inner")
        sides[side.label] = frame.select(*KEYS, "qs", "ae")
    first, second = list(sides.values())
    shared = first.select(KEYS).join(second.select(KEYS), on=KEYS, how="inner")
    sides = {name: frame.join(shared, on=KEYS, how="inner") for name, frame in sides.items()}
    return losses_for_bootstrap(arm_losses=sides)


def estimate(
    *, label: str, contrast: Contrast, sites: tuple[str, ...] | None = None, **scope: object
) -> tuple[float, float, int]:
    """Return the mean paired difference in quantile score, in mean absolute error, and rows.

    Args:
        label: The cut label.
        contrast: The contrast.
        sites: The sites to include, the contrast's own if not given.
        **scope: Further filters passed to `paired` (`months`, `midday`, `dark`, `drop_switching`).

    Returns:
        The treatment-minus-reference means of `qs` and `ae`, and the number of rows.
    """
    losses = paired(
        label=label,
        treatment=contrast.treatment,
        reference=contrast.reference,
        sites=sites or contrast.sites,
        **scope,  # ty: ignore[invalid-argument-type]
    )
    if losses.is_empty():
        return float("nan"), float("nan"), 0
    means = losses.group_by("arm").agg(
        pl.col("qs").mean(), pl.col("ae").mean(), pl.len().alias("n")
    )
    treatment = means.filter(pl.col("arm") == contrast.treatment.label).row(0, named=True)
    reference = means.filter(pl.col("arm") == contrast.reference.label).row(0, named=True)
    return treatment["qs"] - reference["qs"], treatment["ae"] - reference["ae"], int(treatment["n"])


def interval(*, label: str, contrast: Contrast, level: float) -> tuple[float, float]:
    """Return the month-resampled interval of the contrast on its sites."""
    losses = paired(
        label=label,
        treatment=contrast.treatment,
        reference=contrast.reference,
        sites=contrast.sites,
    )
    return bootstrap_difference_at_level(
        losses=losses,
        treatment=contrast.treatment.label,
        reference=contrast.reference.label,
        metric="qs",
        level=level,
        n_resamples=N_RESAMPLES,
    )


def sites_improving(*, label: str, contrast: Contrast) -> int:
    """Return how many of the contrast's sites have a negative difference (treatment better)."""
    return sum(
        int(estimate(label=label, contrast=contrast, sites=(site,))[0] < 0.0)
        for site in contrast.sites
    )


def _unfiltered_side(*, side: Side) -> Side:
    """Return the same arm without its filter; Part 4's X-best+K becomes plain X-best."""
    arm = "x_best" if side.part_4 and side.arm == "x_best_k" else side.arm
    return Side(arm, filtered=False, part_4=side.part_4)


def unfiltered(*, contrast: Contrast) -> Contrast:
    """Return the same contrast on the unfiltered forecasts (Part 4's sets stay as they are)."""
    return Contrast(
        contrast.name + " (unfiltered)",
        _unfiltered_side(side=contrast.treatment),
        _unfiltered_side(side=contrast.reference),
        sites=contrast.sites,
        years=contrast.years,
    )


def _contrast_row(*, contrast: Contrast, level: float) -> dict[str, object]:
    """Return one contrast's row: scored year with intervals and splits, replicate beside it."""
    sites = contrast.sites
    mva = sites == common.MVA_SITES
    row: dict[str, object] = {"contrast": contrast.name}
    for label in contrast.years:
        difference, mean_error, rows = estimate(label=label, contrast=contrast)
        row[f"{label} QS"] = difference
        row[f"{label} MAE of mean"] = mean_error
        row[f"{label} rows"] = rows
        row[f"{label} sites better"] = sites_improving(label=label, contrast=contrast)
        row[f"{label} all rows"] = estimate(label=label, contrast=contrast, drop_switching=False)[0]
        row[f"{label} unfiltered"] = estimate(label=label, contrast=unfiltered(contrast=contrast))[
            0
        ]
    if "scored" in contrast.years:
        low, high = interval(label="scored", contrast=contrast, level=level)
        row["scored low"], row["scored high"] = low, high
        row["Nov-Mar"] = estimate(label="scored", contrast=contrast, months=WINTER_MONTHS)[0]
        row["Apr-Oct"] = estimate(
            label="scored",
            contrast=contrast,
            months=tuple(m for m in range(1, 13) if m not in WINTER_MONTHS),
        )[0]
        row["summer midday"] = estimate(
            label="scored", contrast=contrast, months=SUMMER_MONTHS, midday=True
        )[0]
        if mva:
            row["sunlit rows"] = estimate(label="scored", contrast=contrast, dark=False)[0]
            row["dark rows"] = estimate(label="scored", contrast=contrast, dark=True)[0]
        elif len(sites) > 1 and "D16" in sites:
            row["without D16"] = estimate(
                label="scored", contrast=contrast, sites=tuple(s for s in sites if s != "D16")
            )[0]
    return row


def c4_runs() -> bool:
    """Return True if C4's arm is `identified`; the plan runs C4 only then."""
    try:
        _, labels = capacity_labels()
    except FileNotFoundError:
        return False
    row = labels.filter(pl.col("arm") == tuned()["c4_arm"])
    return bool(row.height and row["identified"][0])


def reproduction_check() -> pl.DataFrame:
    """Compare B with Part 4's stored A2, the plan's base arm, row by row.

    B is refitted here, not read from Part 4's files, so this check shows how far the refit and
    its filter are from the stored sets. Both are filtered, day-ahead, outside the switching masks.
    """
    rows = []
    for label in ("scored", "replicate"):
        mine = load_side(label=label, side=Side("b")).filter(
            pl.col("site_name").is_in(common.DECISION_SITES)
        )
        theirs = load_side(label=label, side=Side("a2", filtered=True, part_4=True)).filter(
            pl.col("site_name").is_in(common.DECISION_SITES)
        )
        joined = mine.select(*KEYS, "mean", "qs").join(
            theirs.select(*KEYS, pl.col("mean").alias("mean_p4"), pl.col("qs").alias("qs_p4")),
            on=KEYS,
            how="inner",
        )
        rows.append(
            {
                "year": label,
                "rows": joined.height,
                "B score": mine["qs"].mean(),
                "Part 4 A2 score": theirs["qs"].mean(),
                "mean |forecast difference|": float(
                    np.abs(joined["mean"].to_numpy() - joined["mean_p4"].to_numpy()).mean()
                ),
            }
        )
    return pl.DataFrame(rows)


def interval_gate_table() -> pl.DataFrame:
    """Return the calibration gate's coverage counts with their binomial ranges."""
    path = common.WORK_DIR / "gate_intervals.json"
    rows = [
        {
            "arm": line["arm"],
            "noise": line["noise"],
            "80% covered": f"{line['covered_80']}/{line['n_substations']}",
            "80% binomial 95%": "{:.2f} to {:.2f}".format(*line["coverage_80_binomial_95"]),
            "80% calibrated": line["calibrated_80"],
            "90% range covered": f"{line['covered_90']}/{line['n_substations']}",
            "90% binomial 95%": "{:.2f} to {:.2f}".format(*line["coverage_90_binomial_95"]),
            "90% calibrated": line["calibrated_90"],
            "mean 90% width": line["mean_width_90_points"],
            "mean bias": line["mean_bias_points"],
        }
        for line in json.loads(path.read_text())["summary"]
    ]
    return pl.DataFrame(rows)


def contrast_table(*, planned: bool) -> pl.DataFrame:
    """Return the contrast table: scored-year interval and splits, with the replicate beside."""
    level = PLANNED_LEVEL if planned else EXPLORATORY_LEVEL
    rows = []
    for contrast in build_contrasts():
        if contrast.planned != planned:
            continue
        if contrast.name.startswith("C4") and not c4_runs():
            rows.append(
                {"contrast": contrast.name, "missing": "not run: its arm is not identified"}
            )
            continue
        try:
            rows.append(_contrast_row(contrast=contrast, level=level))
        except FileNotFoundError as error:
            rows.append({"contrast": contrast.name, "missing": str(error)})
    return pl.DataFrame(rows, infer_schema_length=None)


def leave_one_site_out() -> pl.DataFrame:
    """Return each planned MW contrast with each site left out, scored and replicate years."""
    rows = []
    for contrast in build_contrasts():
        if not contrast.planned or contrast.sites != common.DECISION_SITES:
            continue
        for label in ("scored", "replicate"):
            row: dict[str, object] = {"contrast": contrast.name, "year": label}
            row["all five"] = estimate(label=label, contrast=contrast)[0]
            for left_out in contrast.sites:
                kept = tuple(s for s in contrast.sites if s != left_out)
                row[f"without {left_out}"] = estimate(label=label, contrast=contrast, sites=kept)[0]
            rows.append(row)
    return pl.DataFrame(rows)


def absolute_table() -> pl.DataFrame:
    """Return every arm's quantile score by site for both years, filtered and unfiltered."""
    rows = []
    for label in ("scored", "replicate"):
        for folder in sorted((common.DATA_DIR / "predictions").iterdir()):
            if not folder.name.startswith(f"{label}_"):
                continue
            arm = folder.name.removeprefix(f"{label}_")
            for filtered in (True, False):
                frame = load_side(label=label, side=Side(arm, filtered=filtered))
                by_site = frame.group_by("site_name").agg(pl.col("qs").mean())
                decision = frame.filter(pl.col("site_name").is_in(common.DECISION_SITES))
                row: dict[str, object] = {
                    "year": label,
                    "arm": arm,
                    "filtered": filtered,
                    "decision-site QS": decision["qs"].mean() if decision.height else None,
                    "decision-site MAE of mean": decision["ae"].mean() if decision.height else None,
                    "all-site QS": frame["qs"].mean(),
                }
                rows.append({**row, **dict(by_site.select("site_name", "qs").iter_rows())})
    return pl.DataFrame(rows, infer_schema_length=None).sort("year", "filtered", "arm")


def median_minus_mean() -> pl.DataFrame:
    """Return the mean of (q50 - mean forecast) per site for every arm, in points."""
    rows = []
    for label in ("scored", "replicate"):
        for folder in sorted((common.DATA_DIR / "predictions").iterdir()):
            if not folder.name.startswith(f"{label}_"):
                continue
            arm = folder.name.removeprefix(f"{label}_")
            for filtered in (True, False):
                frame = load_side(label=label, side=Side(arm, filtered=filtered))
                by_site = frame.group_by("site_name").agg((pl.col("q50") - pl.col("mean")).mean())
                rows.append(
                    {"year": label, "arm": arm, "filtered": filtered, **dict(by_site.iter_rows())}
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def tail_table() -> pl.DataFrame:
    """Return coverage and pinball loss at the four tail levels for every MW arm, filtered."""
    rows = []
    for label in ("scored", "replicate"):
        for folder in sorted((common.DATA_DIR / "predictions").iterdir()):
            if not folder.name.startswith(f"{label}_"):
                continue
            arm = folder.name.removeprefix(f"{label}_")
            for filtered in (True, False):
                frame = load_side(label=label, side=Side(arm, filtered=filtered))
                if frame.is_empty():
                    continue
                actual = frame["actual"].to_numpy().astype(np.float64)
                row: dict[str, object] = {"year": label, "arm": arm, "filtered": filtered}
                for column in TAIL_LEVELS:
                    level = QUANTILE_LEVELS[QUANTILE_COLUMNS.index(column)]
                    predicted = frame[column].to_numpy().astype(np.float64)
                    error = actual - predicted
                    row[f"{column} coverage"] = float((actual <= predicted).mean())
                    row[f"{column} pinball"] = float(
                        np.maximum(level * error, (level - 1.0) * error).mean()
                    )
                rows.append(row)
    return pl.DataFrame(rows)


def capacity_labels() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the per-site capacity checks and the arm labels from the pre-registered rule.

    An arm other than K-iii is `identified` at a cut if, at 4 or more of the 5 decision sites,
    the 90% bootstrap range is at most max(3 points, 10% of the capacity) and the capacity lies
    within 20% of the vintage's season-year mean. The label needs both capacity cuts and gate
    tests 1 and 2. `robust` needs two non-base arms `identified`, both agreeing with K-base within
    the floor at 4 or more of 5 sites; handing fitted capacity to Part 6 also needs K-base
    `identified`. K-iii is `consistent` if the bootstrap range of K-iii over K-base, both fitted
    under the same month draws, overlaps [0.65, 0.87] at 4 or more of 5 sites. The bootstrap keeps
    each calendar quarter's months and is widened by sqrt(m / (m - 1)), as experiment 113 advises.

    Returns:
        A per-site table, and a per-arm table of the labels.
    """
    table = pl.read_parquet(common.WORK_DIR / "capacity.parquet")
    gate = json.loads((common.WORK_DIR / "gate.json").read_text())["summary"]
    floor = pl.max_horizontal(
        pl.lit(common.PRECISION_FLOOR_POINTS), common.PRECISION_FLOOR_SHARE * pl.col("capacity")
    )
    sites = table.with_columns(
        width=pl.col("range_high") - pl.col("range_low"),
        precise=(pl.col("range_high") - pl.col("range_low")) <= floor,
        near=(pl.col("capacity") / pl.col("vintage_mean") - 1.0).abs() <= common.NEAR_VINTAGE_SHARE,
        ratio_to_vintage=pl.col("capacity") / pl.col("vintage_mean"),
    )
    labels = []
    for arm in sites["arm"].unique().sort():
        if arm in ("k_iii", "k_iiiii"):
            continue
        per_cut = {}
        for cut in common.CAPACITY_CUTS:
            rows = sites.filter((pl.col("arm") == arm) & (pl.col("cut") == cut))
            good = rows.filter(pl.col("precise") & pl.col("near")).height
            per_cut[cut] = good >= 4
        gate_ok = gate["test_1"].get(arm, False) and gate["test_2"].get(arm, {}).get(
            "passed", False
        )
        labels.append(
            {
                "arm": arm,
                "identified at replicate cut": per_cut["replicate"],
                "identified at scored cut": per_cut["scored"],
                "gate tests 1 and 2": gate_ok,
                "identified": all(per_cut.values()) and gate_ok,
            }
        )
    arm_table = pl.DataFrame(labels)
    identified = set(arm_table.filter(pl.col("identified"))["arm"].to_list())
    non_base = identified - {"k_base"}
    agree = {arm: _agrees_with_base(sites=sites, arm=arm) for arm in non_base}
    robust = len(non_base) >= 2 and all(agree.values())
    _, consistent = _k_iii_ratio(sites=sites)
    arm_table = arm_table.with_columns(
        robust=pl.lit(robust),
        k_iii_consistent=pl.lit(consistent),
        fitted_capacity_may_go_to_part_6=pl.lit(robust and "k_base" in identified),
    )
    return sites, arm_table


def _agrees_with_base(*, sites: pl.DataFrame, arm: str) -> bool:
    """Return True if an arm is within the floor of K-base at 4 of 5 sites at both cuts."""
    for cut in common.CAPACITY_CUTS:
        base = sites.filter((pl.col("arm") == "k_base") & (pl.col("cut") == cut)).sort("site")
        other = sites.filter((pl.col("arm") == arm) & (pl.col("cut") == cut)).sort("site")
        floor = np.maximum(
            common.PRECISION_FLOOR_POINTS,
            common.PRECISION_FLOOR_SHARE * base["capacity"].to_numpy(),
        )
        close = np.abs(other["capacity"].to_numpy() - base["capacity"].to_numpy()) <= floor
        if close.sum() < 4:
            return False
    return True


def _k_iii_ratio(*, sites: pl.DataFrame) -> tuple[list[dict[str, object]], bool]:
    """Return K-iii over K-base per site and cut, with its bootstrap range, and the verdict."""
    rows = []
    consistent = True
    for cut in common.CAPACITY_CUTS:
        base = sites.filter((pl.col("arm") == "k_base") & (pl.col("cut") == cut)).sort("site")
        other = sites.filter((pl.col("arm") == "k_iii") & (pl.col("cut") == cut)).sort("site")
        overlaps = 0
        for b, o in zip(base.iter_rows(named=True), other.iter_rows(named=True), strict=True):
            ratios = np.sort(np.array(o["bootstrap"]) / np.array(o["paired_base_bootstrap"]))
            median = float(np.median(ratios))
            widen = float(np.sqrt(o["n_months"] / max(o["n_months"] - 1, 1)))
            low = median + widen * (float(ratios[1]) - median)
            high = median + widen * (float(ratios[-2]) - median)
            overlap = high >= common.K_III_RATIO_RANGE[0] and low <= common.K_III_RATIO_RANGE[1]
            overlaps += int(overlap)
            rows.append(
                {
                    "cut": cut,
                    "site": o["site"],
                    "K-iii / K-base": o["capacity"] / b["capacity"],
                    "range low": low,
                    "range high": high,
                    "overlaps [0.65, 0.87]": overlap,
                }
            )
        consistent = consistent and overlaps >= 4
    return rows, consistent


def k_iii_table() -> pl.DataFrame:
    """Return the K-iii over K-base table."""
    sites = pl.read_parquet(common.WORK_DIR / "capacity.parquet")
    rows, _ = _k_iii_ratio(sites=sites)
    return pl.DataFrame(rows)


def fit_details() -> dict:
    """Return the fitted quantities `e00110_fit.py` stored for every arm, site, and cut."""
    return json.loads((common.WORK_DIR / "fit_details.json").read_text())


def node_table() -> pl.DataFrame:
    """Return the node's evidence (deepest outage, in points) at each MW site, arm, and cut.

    The real-data clause of the node is that the evidence stays under 1 point at 4 or more of the 5
    decision sites in both years. A generator is evidenced only by an outage, so a fall in demand
    read as a step (large range of the path, no dip) counts as zero.
    """
    rows = []
    for key, detail in fit_details().items():
        label, arm, site = key.split("/")
        if "node" not in detail:
            continue
        rows.append(
            {
                "cut": label,
                "arm": arm,
                "site": site,
                "evidence points": detail["node"]["evidence"],
                "path range points": max(detail["node"]["path"]) - min(detail["node"]["path"]),
                "under 1 point": detail["node"]["evidence"] < 1.0,
            }
        )
    return pl.DataFrame(rows).sort("arm", "cut", "site")


def clip_table() -> pl.DataFrame:
    """Return the profiled clip (multiplier and ceiling share) at each MW site, arm, and cut.

    The clip stays open at a site when the share is at the top of the grid (1.0).
    """
    rows = []
    for key, detail in fit_details().items():
        label, arm, site = key.split("/")
        if "clip" not in detail:
            continue
        rows.append(
            {
                "cut": label,
                "arm": arm,
                "site": site,
                "multiplier": detail["clip"]["multiplier"],
                "ceiling share": detail["clip"]["p_max_share"],
                "open": detail["clip"]["p_max_share"] >= 1.0 - 1e-9,
            }
        )
    return pl.DataFrame(rows).sort("arm", "cut", "site")


def meter_table() -> pl.DataFrame:
    """Return the meter fits' power factor, q0, reactive mode, and reverse-flow share per site."""
    rows = []
    for key, detail in fit_details().items():
        label, arm, site = key.split("/")
        if "meter" not in detail:
            continue
        meter = detail["meter"]
        rows.append(
            {
                "cut": label,
                "arm": arm,
                "site": site,
                "reactive": meter["reactive"],
                "reverse-flow share": meter["reverse_flow_share"],
                "power factor": meter["power_factor"],
                "q0 points": meter["q0"],
                "converged": meter["converged"],
            }
        )
    table = pl.DataFrame(rows).sort("arm", "cut", "site")
    boot = common.WORK_DIR / "meter_boot_summary.parquet"
    if boot.exists():
        table = table.join(
            pl.read_parquet(boot).select(
                "cut",
                "site",
                "power_factor_low",
                "power_factor_high",
                "q0_low",
                "q0_high",
                "recovered",
            ),
            on=["cut", "site"],
            how="left",
        )
    return table


def main() -> None:
    """Print every table of the report as comma-separated values."""
    print("== Absolute quantile score by arm, points of capacity (rows outside the masks) ==")
    print(absolute_table().write_csv(float_precision=3), end="")
    print(f"== Planned contrasts ({PLANNED_LEVEL:.2f}% intervals), negative = treatment better ==")
    print(contrast_table(planned=True).write_csv(float_precision=3), end="")
    print("== Exploratory contrasts (95% intervals) ==")
    print(contrast_table(planned=False).write_csv(float_precision=3), end="")
    print("== Planned MW contrasts with each decision site left out (quantile score) ==")
    print(leave_one_site_out().write_csv(float_precision=3), end="")
    print("== Calibration gate of the capacity intervals (weak evidence: 20 substations) ==")
    print(interval_gate_table().write_csv(float_precision=3), end="")
    print("== Reproduction of Part 4's A2 by B ==")
    print(reproduction_check().write_csv(float_precision=4), end="")
    print("== Capacity by site, arm, and cut ==")
    sites, labels = capacity_labels()
    print(sites.drop("bootstrap").write_csv(float_precision=2), end="")
    print("== Capacity labels ==")
    print(labels.write_csv(float_precision=3), end="")
    print("== K-iii over K-base ==")
    print(k_iii_table().write_csv(float_precision=3), end="")
    print("== Node: deepest outage by site, arm, and cut ==")
    print(node_table().write_csv(float_precision=3), end="")
    print("== Clip: profiled multiplier and ceiling by site, arm, and cut ==")
    print(clip_table().write_csv(float_precision=3), end="")
    print("== Meter: reactive pair by site, arm, and cut ==")
    print(meter_table().write_csv(float_precision=3), end="")
    print("== Median forecast minus mean forecast, points ==")
    print(median_minus_mean().write_csv(float_precision=3), end="")
    print("== Tail calibration ==")
    print(tail_table().write_csv(float_precision=4), end="")


if __name__ == "__main__":
    main()
