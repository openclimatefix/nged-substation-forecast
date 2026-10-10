"""Print every table the post hoc follow-up analyses quote, and save the tables the charts read.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1138>. It
reads the first run's losses and the follow-up fits' losses, writes
`<output-root>/ens_mean/followups/report_followups_ens_mean.md` and `tables_followups_ens_mean/`,
and never writes into the first run's files. **Every number is post hoc**, computed after the first
run's results were known, and the report says so under each heading.

The sections:

1. Positive controls (month-level and plant-specific): each arm's difference to B0, its 99%
   interval, whether the interval lies wholly below zero, and the share of the oracle's gain it
   recovers. The 2% rule is not used. The realised shifted share per plant, from the build report.
2. Long leads (lead-days 7, 10, 14): CL, W7+CL, N2-3 and the 50/50 blends with climatology, against
   B0, and the sensitivity setting at lead-days 10 and 14.
3. PC with a 2-day latency, against B0 and against PC.
4. Fingerprint decomposition, and global against per-plant on the same fleet-wide folds.
5. Every fitted arm's difference to B0 on the 8 months the sweep never screened, split by forecast
   product era. AN was selected on the 10 screening months, which this table does not reuse.
6. Per-quantile hit rates of B0 and L1, overall and by month.
7. The count of exploratory intervals in the first report and here, and how many exclude zero on
   each side.
8. The hindsight scaling bound: B0 after a per-(plant, month) rescaling fitted on the same data.
9. The month-cluster t-interval beside the bootstrap interval for the first run's planned contrasts.

Run it with `uv run python studies/lag_features/report_followups.py`.
"""

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import polars as pl
from build_lag_frame import (
    FULL_SWEEP_LEAD_DAY,
    LAG_FEATURES_DIR,
    SWEEP_ARMS,
    TARGET,
    output_paths,
    run_suffix,
    scored_months,
    write_parquet_atomic,
)
from fit_lag_arms import METRIC, QUANTILE_COLUMNS, QUANTILE_LEVELS
from followup_frames import (
    CONTROL_SHIFTS,
    LONG_LEADS,
    PRODUCT,
    followup_dir,
    frame_path,
)
from report_lag_features import (
    EXPLORATORY_LEVEL,
    P2_FIRST_MONTH,
    PERCENTAGE_POINTS,
    PLANNED_LEVEL,
    paired_interval,
    planned_contrasts,
    read_frame,
    reference_losses,
    subset,
)
from scipy import stats
from studies.bootstrap import arm_values, bootstrap_absolute, paired_differences
from studies.cross_validation import score_prediction
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("report_followups")

ERAS: Final[dict[str, tuple[str, str]]] = {
    "2025-10 to 2025-12": ("2025-10", "2025-12"),
    "2026-02 onwards": ("2026-02", "2026-12"),
}
"""The forecast-product eras of the 8 unselected months, by first and last month."""

UNSELECTED_NOTE: Final[str] = (
    "Post hoc. The 8 months are the ones the sweep never screened (2025-10 onwards, without "
    "2026-01). AN was selected as X on the 10 screening months, so its row here is out of sample."
)
"""The line printed under the unselected-month table."""

MIN_MONTHS_FOR_T: Final[int] = 3
"""The fewest months a t-interval is computed from."""

INTERVAL_CELL: Final[re.Pattern[str]] = re.compile(r"^\[([+−-]?\d[\d.]*), ?([+−-]?\d[\d.]*)\]$")
"""A markdown table cell holding an interval such as `[-0.249, -0.140]`."""

NOT_EXPLORATORY_HEADINGS: Final[tuple[str, ...]] = (
    "Build report",
    "Phase 2",
    "Positive control",
    "Stage-1 anchor",
)
"""Headings of the first report whose intervals are planned or checks, not exploratory."""

CHECK_NAMES: Final[tuple[str, ...]] = ("month", "step")
"""The two kinds of positive control."""


class Ledger(NamedTuple):
    """The exploratory intervals this report computed."""

    records: list[dict[str, object]]


def contrast(
    *,
    ledger: Ledger,
    source: str,
    scoped: pl.DataFrame,
    treatment: str,
    reference: str,
    level: float = EXPLORATORY_LEVEL,
    metric: str = METRIC,
    exploratory: bool = True,
) -> dict[str, object] | None:
    """Compute one paired contrast and record it in the ledger of exploratory intervals.

    Args:
        ledger: Where to record the interval.
        source: Which table the interval belongs to.
        scoped: Losses of both arms.
        treatment: The arm compared.
        reference: The arm it is compared with.
        level: The interval's coverage in percent.
        metric: The loss column.
        exploratory: Whether to count the interval among the exploratory ones.

    Returns:
        The difference, its bounds, the reference's own value, and the rows and months it rests on,
        or `None` if an arm is missing.
    """
    interval = paired_interval(
        scoped=scoped, treatment=treatment, reference=reference, metric=metric, level=level
    )
    if interval is None:
        return None
    row: dict[str, object] = {
        "treatment": treatment,
        "reference": reference,
        "difference": interval["difference"],
        "lower": interval["lower_95"],
        "upper": interval["upper_95"],
        "n_months": interval["n_months"],
        "n_rows": interval["n_rows"],
        "reference_value": float(arm_values(losses=scoped, arm=reference, metric=metric)[0].mean()),
    }
    if exploratory:
        ledger.records.append({"source": source, **row})
    return row


def _share(value: float | None) -> str:
    """Format a share of the oracle's gain, blank where the oracle has no gain."""
    return "" if value is None else f"{value:.2f}"


def _pp(value: float) -> str:
    """Format a fraction of capacity as signed percentage points."""
    return f"{value * PERCENTAGE_POINTS:+.3f}"


def contrast_table(*, rows: list[dict[str, object]], extra: list[str]) -> list[str]:
    """Format contrast rows as a markdown table.

    Args:
        rows: Rows from `contrast`, each with the keys in `extra` too.
        extra: Extra columns to print before the difference, by key.

    Returns:
        The table's lines.
    """
    header = [
        *(name.replace("_", " ") for name in extra),
        "Difference (pp)",
        "Interval (pp)",
        "Months",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += [
        "| "
        + " | ".join(
            [
                *(str(row[name]) for name in extra),
                _pp(float(row["difference"])),  # ty: ignore[invalid-argument-type]
                f"[{_pp(float(row['lower']))}, {_pp(float(row['upper']))}]",  # ty: ignore[invalid-argument-type]
                str(row["n_months"]),
            ]
        )
        + " |"
        for row in rows
    ]
    return [*lines, ""]


# --- 1. Positive controls -----------------------------------------------------------------------


def controls_section(
    *, follow: pl.DataFrame, root: Path, ledger: Ledger, smoke: bool
) -> tuple[list[str], pl.DataFrame]:
    """Tabulate the positive controls: gain over B0 and the share of the oracle's gain recovered.

    "Recovers" is judged by whether the 99% interval lies wholly below zero and by the share of the
    oracle's gain, never by the 2% rule, which an ideal estimator cannot meet below a 10% shift.

    Args:
        follow: The follow-up losses.
        root: The output root, for the realised shifted shares.
        ledger: The ledger of exploratory intervals.
        smoke: Whether to read a smoke run's subsampled frames.

    Returns:
        The report lines and a table with one row per (kind, shift, arm).
    """
    records = []
    for kind in CHECK_NAMES:
        for shift in CONTROL_SHIFTS:
            percent = f"{round(shift * 100):02d}"
            scope = f"fu_control_{kind}_s{percent}"
            scoped = follow.filter((pl.col("scope") == scope) & (pl.col("setting") == "primary"))
            arms = [a for a in scoped["arm"].unique().sort().to_list() if a != "B0"]
            oracle = contrast(
                ledger=ledger,
                source=f"control {kind} {percent}",
                scoped=scoped,
                treatment="O",
                reference="B0",
                level=PLANNED_LEVEL,
                exploratory=False,
            )
            for arm in arms:
                row = contrast(
                    ledger=ledger,
                    source=f"control {kind} {percent}",
                    scoped=scoped,
                    treatment=arm,
                    reference="B0",
                    level=PLANNED_LEVEL,
                    exploratory=False,
                )
                if row is None or oracle is None:
                    continue
                oracle_gain = -float(oracle["difference"])  # ty: ignore[invalid-argument-type]
                oracle_found = float(oracle["upper"]) < 0.0  # ty: ignore[invalid-argument-type]
                records.append(
                    {
                        "kind": kind,
                        "shift": shift,
                        "arm": arm,
                        **row,
                        "wholly_below_zero": float(row["upper"]) < 0.0,  # ty: ignore[invalid-argument-type]
                        "oracle_difference": float(oracle["difference"]),  # ty: ignore[invalid-argument-type]
                        "share_of_oracle_gain": -float(row["difference"]) / oracle_gain  # ty: ignore[invalid-argument-type]
                        if oracle_found
                        else None,
                    }
                )
    table = pl.DataFrame(records)
    lines = [
        (
            "Post hoc. The month-level control shifts 9 of the 18 scored months (and a random "
            "half of the others); the plant-specific control gives each plant persistent steps of"
            " 4 to 12 weeks. O is B0 plus the true shift factor. The 2% rule is not used here."
        ),
        "",
        (
            "| Control | Shift | Arm | Difference to B0 (pp) | 99% interval (pp) | Wholly below "
            "zero | Share of the oracle's gain recovered |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['kind']} | {row['shift']:.0%} | {row['arm']} | {_pp(row['difference'])} | "
        f"[{_pp(row['lower'])}, {_pp(row['upper'])}] | "
        f"{'yes' if row['wholly_below_zero'] else 'no'} | {_share(row['share_of_oracle_gain'])} |"
        for row in table.iter_rows(named=True)
    ]
    return [*lines, "", *realised_share_lines(root=root, smoke=smoke), ""], table


def realised_share_lines(*, root: Path, smoke: bool) -> list[str]:
    """Report each plant's realised shifted share of the scored months in each control frame.

    Args:
        root: The output root.
        smoke: Whether to read a smoke run's subsampled frames.

    Returns:
        A table with one row per (control, plant): the scored months, the months with half or more
        of the rows shifted, and the mean share of rows shifted.
    """
    lines = [
        "Realised shifted share of the scored months, per plant:",
        "",
        (
            "| Control | Plant | Scored months | Months with half or more of the rows shifted "
            "| Mean share of rows shifted |"
        ),
        "|---|---|---|---|---|",
    ]
    for kind in CHECK_NAMES:
        for shift in CONTROL_SHIFTS:
            name = f"control_{kind}_s{round(shift * 100):02d}"
            frame = read_frame(path=frame_path(root=root, name=name), smoke=smoke)
            per_plant = (
                frame.with_columns(shifted=pl.col("oracle_factor") < 1.0)
                .group_by("site", "month")
                .agg(share=pl.col("shifted").mean())
                .group_by("site")
                .agg(
                    months=pl.len(),
                    mostly=(pl.col("share") >= 0.5).sum(),
                    mean_share=pl.col("share").mean(),
                )
                .sort("site")
            )
            lines += [
                f"| {kind} {shift:.0%} | {row['site']} | {row['months']} | {row['mostly']} | "
                f"{row['mean_share']:.3f} |"
                for row in per_plant.iter_rows(named=True)
            ]
    return lines


# --- 2. Long leads ------------------------------------------------------------------------------


def long_leads_section(
    *, first: pl.DataFrame, follow: pl.DataFrame, root: Path, ledger: Ledger, smoke: bool
) -> tuple[list[str], pl.DataFrame, pl.DataFrame]:
    """Tabulate the long-lead arms and blends against B0, and the sensitivity setting.

    Args:
        first: The first run's losses.
        follow: The follow-up losses.
        root: The output root.
        ledger: The ledger of exploratory intervals.
        smoke: Whether to read a smoke run's subsampled frames.

    Returns:
        The report lines, a table of each arm's difference to B0 and a table of the sensitivity
        setting's differences.
    """
    primary: list[dict[str, object]] = []
    sensitivity: list[dict[str, object]] = []
    for lead in LONG_LEADS:
        frame = read_frame(
            path=output_paths(root=root, product=PRODUCT, lead_days=(lead,))[f"day{lead}"],
            smoke=smoke,
        )
        references = reference_losses(frame=frame, lead_day=lead, losses=first)
        first_primary = subset(
            losses=first,
            scope=f"lead{lead}",
            setting="primary",
            arms=("B0", "N2", "W7", "Q30"),
        ).select("site", "time", "month", "seed", "arm", METRIC)
        followed = follow.filter(
            (pl.col("scope") == f"fu_lead{lead}") & (pl.col("setting") == "primary")
        ).select("site", "time", "month", "seed", "arm", METRIC)
        both = pl.concat(
            [first_primary, followed, references.filter(pl.col("arm") == "climatology")]
        )
        for arm in sorted(set(both["arm"].unique().to_list()) - {"B0"}):
            row = contrast(
                ledger=ledger,
                source=f"long lead {lead}",
                scoped=both,
                treatment=arm,
                reference="B0",
            )
            if row is not None:
                primary.append({"lead_day": lead, "arm": arm, **row})
        if lead in (10, 14):
            first_sensitivity = subset(
                losses=first, scope=f"lead{lead}", setting="sensitivity", arms=("B0",)
            ).select("site", "time", "month", "seed", "arm", METRIC)
            followed_sensitivity = follow.filter(
                (pl.col("scope") == f"fu_lead{lead}") & (pl.col("setting") == "sensitivity")
            ).select("site", "time", "month", "seed", "arm", METRIC)
            both_sensitivity = pl.concat([first_sensitivity, followed_sensitivity]).unique(
                subset=["site", "time", "seed", "arm"]
            )
            for arm in sorted(set(both_sensitivity["arm"].unique().to_list()) - {"B0"}):
                row = contrast(
                    ledger=ledger,
                    source=f"long lead {lead} sensitivity",
                    scoped=both_sensitivity,
                    treatment=arm,
                    reference="B0",
                )
                if row is not None:
                    sensitivity.append({"lead_day": lead, "arm": arm, **row})
    lines = [
        (
            "Post hoc. Each row is the arm's mean absolute error minus B0's at the same lead-day,"
            " with a 95% interval from resampling whole months. CL is B0 plus the out-of-fold "
            "climatology as a column; B0xCL and W7xCL are 50/50 blends of the saved predictions "
            "with climatology, with no fit; N2-3 is a null as wide as W7; `climatology` is the "
            "no-fit forecast itself."
        ),
        "",
        "**Primary setting**",
        "",
        *contrast_table(rows=primary, extra=["lead_day", "arm"]),
        "**Sensitivity setting, lead-days 10 and 14**",
        "",
        *contrast_table(rows=sensitivity, extra=["lead_day", "arm"]),
    ]
    return lines, pl.DataFrame(primary), pl.DataFrame(sensitivity)


# --- 3. PC2 -------------------------------------------------------------------------------------


def pc2_section(
    *, first: pl.DataFrame, follow: pl.DataFrame, ledger: Ledger
) -> tuple[list[str], pl.DataFrame]:
    """Tabulate PC2 (a 2-day CAMS latency) against B0 and against PC, on all and the later months.

    Args:
        first: The first run's losses.
        follow: The follow-up losses.
        ledger: The ledger of exploratory intervals.

    Returns:
        The report lines and a table with one row per (contrast, months).
    """
    base = subset(losses=first, scope="lead1", setting="primary", arms=("B0", "PC")).select(
        "site", "time", "month", "seed", "arm", METRIC
    )
    pc2 = follow.filter(
        (pl.col("scope") == "fu_lead1_pc2") & (pl.col("setting") == "primary")
    ).select("site", "time", "month", "seed", "arm", METRIC)
    both = pl.concat([base, pc2])
    records = []
    for months, label in (("all", "all 18 months"), ("later", "the 8 unselected months")):
        scoped = both if months == "all" else both.filter(pl.col("month") >= P2_FIRST_MONTH)
        for treatment, reference in (("PC2", "B0"), ("PC", "B0"), ("PC2", "PC")):
            row = contrast(
                ledger=ledger,
                source="PC2",
                scoped=scoped.filter(pl.col("arm").is_in([treatment, reference])),
                treatment=treatment,
                reference=reference,
            )
            if row is not None:
                records.append(
                    {"months": label, "contrast": f"{treatment} minus {reference}", **row}
                )
    lines = [
        (
            "Post hoc. PC2 reads CAMS irradiance two whole days back (windows days 2 to 8 and 2 "
            "to 31), because 3 of the 8 unselected months fall before CAMS's February 2026 "
            "latency cut."
        ),
        "",
        *contrast_table(rows=records, extra=["months", "contrast"]),
    ]
    return lines, pl.DataFrame(records)


# --- 4. Fingerprint decomposition ---------------------------------------------------------------


def fingerprint_section(
    *, first: pl.DataFrame, follow: pl.DataFrame, ledger: Ledger
) -> tuple[list[str], pl.DataFrame]:
    """Tabulate the fingerprint decomposition and global against per-plant on the same folds.

    Args:
        first: The first run's losses.
        follow: The follow-up losses.
        ledger: The ledger of exploratory intervals.

    Returns:
        The report lines and a table of each arm's absolute error.
    """
    names = {"B0": "G-B0", "L1": "G-L1"}
    pooled = subset(
        losses=first, scope="global", setting="primary", arms=("B0", "L1", "G-ID", "G-FP")
    ).with_columns(arm=pl.col("arm").replace(names))
    left_out = subset(
        losses=first, scope="lopo", setting="primary", arms=("B0", "G-FP")
    ).with_columns(arm="LOPO " + pl.col("arm").replace(names))
    new_pooled = follow.filter(pl.col("scope") == "fu_global")
    new_lopo = follow.filter(pl.col("scope") == "fu_lopo").with_columns(arm="LOPO " + pl.col("arm"))
    per_plant = follow.filter(pl.col("scope") == "fu_fleetfold").with_columns(
        arm=pl.lit("Per-plant B0, fleet folds")
    )
    everything = pl.concat(
        [
            frame.select("site", "time", "month", "seed", "arm", METRIC)
            for frame in (pooled, left_out, new_pooled, new_lopo, per_plant)
        ]
    )
    absolute = []
    for arm in sorted(everything["arm"].unique().to_list()):
        interval = bootstrap_absolute(losses=everything, arm=arm, metric=METRIC)
        absolute.append(
            {
                "arm": arm,
                "error": interval["value"],
                "lower": interval["lower_95"],
                "upper": interval["upper_95"],
            }
        )
    pairs = (
        ("G-ID+TF", "G-B0"),
        ("G-FPnoCK", "G-B0"),
        ("G-FP", "G-FPnoCK"),
        ("G-ID+TF", "G-ID"),
        ("G-ID+TF", "G-FP"),
        ("G-B0", "Per-plant B0, fleet folds"),
        ("G-ID", "Per-plant B0, fleet folds"),
        ("G-L1", "Per-plant B0, fleet folds"),
        ("LOPO G-FPnoCK", "LOPO G-B0"),
        ("LOPO G-FP", "LOPO G-FPnoCK"),
    )
    rows = []
    for treatment, reference in pairs:
        row = contrast(
            ledger=ledger,
            source="fingerprint decomposition",
            scoped=everything.filter(pl.col("arm").is_in([treatment, reference])),
            treatment=treatment,
            reference=reference,
        )
        if row is not None:
            rows.append({"contrast": f"{treatment} minus {reference}", **row})
    lines = [
        (
            "Post hoc. All arms use the first run's fleet-wide folds, so the contrasts are "
            "paired. G-FPnoCK is G-FP without the clipping-ceiling columns; Per-plant B0 is B0 "
            "fitted per plant on the same folds and capacity-normalised target."
        ),
        "",
        "| Arm | Mean absolute error (pp) | 95% interval (pp) |",
        "|---|---|---|",
        *(
            f"| {row['arm']} | {row['error'] * PERCENTAGE_POINTS:.3f} | "
            f"[{row['lower'] * PERCENTAGE_POINTS:.3f}, {row['upper'] * PERCENTAGE_POINTS:.3f}] |"
            for row in absolute
        ),
        "",
        *contrast_table(rows=rows, extra=["contrast"]),
    ]
    return lines, pl.DataFrame(absolute)


# --- 5. Unselected months -----------------------------------------------------------------------


def unselected_section(*, first: pl.DataFrame, ledger: Ledger) -> tuple[list[str], pl.DataFrame]:
    """Tabulate every fitted arm's difference to B0 on the 8 months the sweep never screened.

    Args:
        first: The first run's losses.
        ledger: The ledger of exploratory intervals.

    Returns:
        The report lines and a table with one row per (arm, era).
    """
    scoped = subset(
        losses=first, scope=f"lead{FULL_SWEEP_LEAD_DAY}", setting="primary", arms=SWEEP_ARMS
    ).select("site", "time", "month", "seed", "arm", METRIC)
    later = scoped.filter(pl.col("month") >= P2_FIRST_MONTH)
    records = []
    eras = {"2025-10 onwards (8 months)": ("2025-10", "2026-12"), **ERAS}
    for arm in sorted(set(scoped["arm"].unique().to_list()) - {"B0"}):
        for era, (low, high) in eras.items():
            in_era = later.filter((pl.col("month") >= low) & (pl.col("month") <= high))
            row = contrast(
                ledger=ledger,
                source="unselected months",
                scoped=in_era.filter(pl.col("arm").is_in([arm, "B0"])),
                treatment=arm,
                reference="B0",
            )
            if row is not None:
                records.append({"arm": arm, "era": era, **row})
    records.sort(key=lambda r: (str(r["era"]), float(r["difference"])))  # ty: ignore[invalid-argument-type]
    lines = [
        UNSELECTED_NOTE,
        "",
        (
            "Difference to B0 at lead-day 1, per-plant fits, primary setting; negative beats B0. "
            "The interval is a 95% interval from resampling whole months, and an era of fewer "
            "than 6 months is a weak interval."
        ),
        "",
        *contrast_table(rows=records, extra=["arm", "era"]),
    ]
    return lines, pl.DataFrame(records)


# --- 6. Per-quantile hit rates ------------------------------------------------------------------


def hit_rate_section(*, root: Path, smoke: bool) -> tuple[list[str], pl.DataFrame]:
    """Tabulate the share of rows with the actual at or below each of the nine quantiles.

    A calibrated quantile model hits at its own level. The table is a probability-integral-
    transform-style input for the chart.

    Args:
        root: The output root, whose first run holds `intervals__<arm>.parquet`.
        smoke: Whether to read a smoke run's checkpoints.

    Returns:
        The report lines and a table with one row per (arm, month, level), month `all` for overall.
    """
    records = []
    for arm in ("B0", "L1"):
        rows = pl.read_parquet(
            root / PRODUCT / f"checkpoints{run_suffix(smoke=smoke)}" / f"intervals__{arm}.parquet"
        )
        for level, column in zip(QUANTILE_LEVELS, QUANTILE_COLUMNS, strict=True):
            hit = pl.col("actual") <= pl.col(column)
            records.append(
                {
                    "arm": arm,
                    "month": "all",
                    "level": level,
                    "hit_rate": rows.select(hit.mean()).item(),
                    "n_rows": rows.height,
                }
            )
            by_month = (
                rows.group_by("month").agg(hit_rate=hit.mean(), n_rows=pl.len()).sort("month")
            )
            records += [
                {
                    "arm": arm,
                    "month": r["month"],
                    "level": level,
                    "hit_rate": r["hit_rate"],
                    "n_rows": r["n_rows"],
                }
                for r in by_month.iter_rows(named=True)
            ]
    table = pl.DataFrame(records)
    overall = table.filter(pl.col("month") == "all").pivot(
        on="arm", index="level", values="hit_rate"
    )
    lines = [
        (
            "Post hoc. The share of rows with the measured power at or below each predicted "
            "quantile, one fitting seed, per-plant fits at lead-day 1. A calibrated model's share"
            " equals its level. By-month shares are in `hit_rates.parquet`."
        ),
        "",
        "| Level | B0 | L1 |",
        "|---|---|---|",
        *(
            f"| {row['level']:.1f} | {row['B0']:.3f} | {row['L1']:.3f} |"
            for row in overall.sort("level").iter_rows(named=True)
        ),
        "",
    ]
    return lines, table


# --- 7. Count of exploratory intervals ----------------------------------------------------------


def first_report_intervals(*, path: Path) -> list[dict[str, object]]:
    """Read the exploratory difference intervals out of the first report's markdown tables.

    A table counts if its header has a `Difference` column and an interval column, and it is not
    under a planned or check heading (`NOT_EXPLORATORY_HEADINGS`).

    Args:
        path: The first run's `report_ens_mean.md`.

    Returns:
        One record per interval, with `source` (the heading), `lower` and `upper` in the table's
        own units.
    """
    records: list[dict[str, object]] = []
    heading = ""
    columns: dict[str, int] = {}
    for line in path.read_text().split("\n"):
        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
            columns = {}
            continue
        if not line.startswith("|"):
            columns = {}
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        lowered = [cell.lower() for cell in cells]
        if any("difference" in cell for cell in lowered) and any(
            "interval" in cell for cell in lowered
        ):
            columns = {"interval": next(i for i, cell in enumerate(lowered) if "interval" in cell)}
            continue
        if not columns or set("".join(cells)) <= set("-: "):
            continue
        if heading.startswith(NOT_EXPLORATORY_HEADINGS):
            continue
        match = INTERVAL_CELL.match(cells[columns["interval"]].replace("−", "-"))
        if match:
            records.append(
                {"source": heading, "lower": float(match.group(1)), "upper": float(match.group(2))}
            )
    return records


def count_lines(
    *, first: list[dict[str, object]], ledger: Ledger
) -> tuple[list[str], pl.DataFrame]:
    """Count the exploratory intervals and how many exclude zero on each side.

    A difference is the treatment's error minus the reference's, so an interval wholly below zero
    is a win for the treatment and one wholly above zero is a loss.

    Args:
        first: `first_report_intervals`' result.
        ledger: The follow-up ledger.

    Returns:
        The report lines and a table with one row per report.
    """
    rows = []
    for name, records in (("first report", first), ("follow-ups", ledger.records)):
        lower = np.array([float(r["lower"]) for r in records])  # ty: ignore[invalid-argument-type]
        upper = np.array([float(r["upper"]) for r in records])  # ty: ignore[invalid-argument-type]
        rows.append(
            {
                "report": name,
                "intervals": len(records),
                "win_side": int((upper < 0).sum()),
                "loss_side": int((lower > 0).sum()),
                "include_zero": int(((lower <= 0) & (upper >= 0)).sum()),
            }
        )
    table = pl.DataFrame(rows)
    lines = [
        (
            "Post hoc. Exploratory intervals are the paired-difference intervals outside the "
            "planned contrasts and the controls. A difference is the treatment minus the "
            "reference, so wholly below zero is a win for the treatment. At 95%, about 1 in 20 "
            "intervals excludes zero by chance alone."
        ),
        "",
        "| Report | Exploratory intervals | Wholly below zero | Wholly above zero | Include zero |",
        "|---|---|---|---|---|",
        *(
            f"| {row['report']} | {row['intervals']} | {row['win_side']} | {row['loss_side']} | "
            f"{row['include_zero']} |"
            for row in table.iter_rows(named=True)
        ),
        "",
    ]
    return lines, table


# --- 8. Hindsight scaling bound -----------------------------------------------------------------


def hindsight_losses(*, losses: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """Score B0 after a per-(plant, month, seed) rescaling fitted in hindsight, and B0 itself.

    The scale for a plant, month and seed is the sum of the measured power over the sum of the
    predicted power, so the rescaled forecast matches each plant-month's measured energy exactly.
    No forecast could know that scale in advance, so the result is a bound on what any slow
    calibration of B0 could gain.

    Args:
        losses: B0's per-row losses, with `prediction`, in one scope and setting.
        frame: The frame the losses were fitted on, for the cap and the capacity.

    Returns:
        Per-row losses for `B0` and `B0 rescaled in hindsight`, as `paired_interval` reads them.
    """
    base = losses.filter(pl.col("arm") == "B0").select(
        "site", "time", "month", "seed", "prediction", "actual"
    )
    scale = base.group_by("site", "month", "seed").agg(
        scale=pl.col("actual").sum() / pl.col("prediction").sum()
    )
    scaled = (
        base.join(scale, on=["site", "month", "seed"])
        .with_columns(prediction=pl.col("prediction") * pl.col("scale"))
        .select("site", "time", "seed", "prediction")
    )
    rows = frame.select(
        "site", "time", "month", "fold", "constrained", "effective_capacity_mw", "cap_mw", TARGET
    )
    scored = score_prediction(rows=rows, prediction=scaled, target=TARGET).select(
        "site", "time", "month", "seed", METRIC
    )
    plain = losses.filter(pl.col("arm") == "B0").select("site", "time", "month", "seed", METRIC)
    return pl.concat(
        [
            plain.with_columns(arm=pl.lit("B0")),
            scored.with_columns(arm=pl.lit("B0 rescaled in hindsight")),
        ]
    )


def hindsight_section(
    *, first: pl.DataFrame, follow: pl.DataFrame, root: Path, ledger: Ledger, smoke: bool
) -> tuple[list[str], pl.DataFrame]:
    """Tabulate the hindsight scaling bound on the real data and on each shifted control.

    Args:
        first: The first run's losses.
        follow: The follow-up losses.
        root: The output root.
        ledger: The ledger of exploratory intervals.
        smoke: Whether to read a smoke run's subsampled frames.

    Returns:
        The report lines and a table with one row per dataset.
    """
    lead1 = read_frame(
        path=output_paths(root=root, product=PRODUCT, lead_days=(FULL_SWEEP_LEAD_DAY,))["day1"],
        smoke=smoke,
    )
    datasets = [
        (
            "unshifted data",
            subset(losses=first, scope="lead1", setting="primary", arms=("B0",)),
            lead1,
        )
    ]
    for kind in CHECK_NAMES:
        for shift in CONTROL_SHIFTS:
            name = f"control_{kind}_s{round(shift * 100):02d}"
            datasets.append(
                (
                    f"{kind} control, {shift:.0%}",
                    follow.filter(
                        (pl.col("scope") == f"fu_{name}")
                        & (pl.col("setting") == "primary")
                        & (pl.col("arm") == "B0")
                    ),
                    read_frame(path=frame_path(root=root, name=name), smoke=smoke),
                )
            )
    records = []
    for label, losses, frame in datasets:
        both = hindsight_losses(losses=losses, frame=frame)
        row = contrast(
            ledger=ledger,
            source="hindsight scaling",
            scoped=both,
            treatment="B0 rescaled in hindsight",
            reference="B0",
        )
        if row is not None:
            records.append({"dataset": label, **row})
    lines = [
        (
            "Post hoc. B0's forecast after multiplying each plant-month's predictions by that "
            "month's measured-over-predicted energy, fitted on the same rows. No forecast could "
            "know that scale in advance; the difference bounds what any slow calibration could "
            "gain."
        ),
        "",
        *contrast_table(rows=records, extra=["dataset"]),
    ]
    return lines, pl.DataFrame(records)


# --- 9. Month-cluster t-intervals ---------------------------------------------------------------


def t_interval_section(
    *, first: pl.DataFrame, root: Path, smoke: bool
) -> tuple[list[str], pl.DataFrame]:
    """Tabulate the month-cluster t-interval beside the bootstrap interval for P1 to P5.

    The t-interval treats each calendar month's mean paired difference (over plants, seeds and
    hours) as one observation, so it has one fewer degree of freedom than there are months.

    Args:
        first: The first run's losses.
        root: The output root, whose first run holds the shortlist.
        smoke: Whether to read a smoke run's checkpoints.

    Returns:
        The report lines and a table with one row per (contrast, setting).
    """
    shortlist = root / PRODUCT / f"checkpoints{run_suffix(smoke=smoke)}" / "shortlist.json"
    chosen = json.loads(shortlist.read_text())["x"]
    records = []
    for planned in planned_contrasts(chosen=chosen):
        for setting in ("primary", "sensitivity"):
            scoped = subset(
                losses=first,
                scope=planned.scope,
                setting=setting,
                arms=(planned.treatment, planned.reference),
                months="after" if planned.months == "after" else "all",
            )
            interval = paired_interval(
                scoped=scoped,
                treatment=planned.treatment,
                reference=planned.reference,
                metric=planned.metric,
                level=PLANNED_LEVEL,
            )
            if interval is None:
                continue
            differences, months = paired_differences(
                losses=scoped,
                treatment=planned.treatment,
                reference=planned.reference,
                metric=planned.metric,
            )
            per_month = (
                pl.DataFrame({"month": months, "difference": differences.mean(axis=0)})
                .group_by("month")
                .agg(pl.col("difference").mean())["difference"]
                .to_numpy()
            )
            if len(per_month) < MIN_MONTHS_FOR_T:
                continue
            centre = float(per_month.mean())
            half = float(
                stats.t.ppf(0.5 + PLANNED_LEVEL / 200.0, df=len(per_month) - 1)
                * per_month.std(ddof=1)
                / np.sqrt(len(per_month))
            )
            records.append(
                {
                    "contrast": planned.name,
                    "setting": setting,
                    "months": len(per_month),
                    "difference": centre,
                    "bootstrap_lower": interval["lower_95"],
                    "bootstrap_upper": interval["upper_95"],
                    "t_lower": centre - half,
                    "t_upper": centre + half,
                }
            )
    table = pl.DataFrame(records)
    lines = [
        (
            "Post hoc. The planned contrasts' 99% bootstrap interval beside a 99% t-interval over"
            " the calendar months' mean differences (degrees of freedom: months minus one)."
        ),
        "",
        (
            "| Contrast | Setting | Months | Difference (pp) | Bootstrap interval (pp) "
            "| t-interval (pp) |"
        ),
        "|---|---|---|---|---|---|",
        *(
            f"| {row['contrast']} | {row['setting']} | {row['months']} | "
            f"{_pp(row['difference'])} | "
            f"[{_pp(row['bootstrap_lower'])}, {_pp(row['bootstrap_upper'])}] | "
            f"[{_pp(row['t_lower'])}, {_pp(row['t_upper'])}] |"
            for row in table.iter_rows(named=True)
        ),
        "",
    ]
    return lines, table


def report_text(*, root: Path, tables: Path, smoke: bool) -> str:
    """Build the whole follow-up report and save the tables the charts read.

    Args:
        root: The output root.
        tables: The directory to write the tables into.
        smoke: Whether to report a smoke run's files.

    Returns:
        The report's markdown.
    """
    suffix = run_suffix(smoke=smoke)
    first = pl.read_parquet(root / PRODUCT / f"losses_{PRODUCT}{suffix}.parquet")
    follow = pl.read_parquet(
        followup_dir(root=root) / f"losses_followups_{PRODUCT}{suffix}.parquet"
    )
    ledger = Ledger(records=[])
    sections: list[tuple[str, list[str], dict[str, pl.DataFrame]]] = []

    def add(title: str, lines: list[str], **named: pl.DataFrame) -> None:
        sections.append((title, lines, named))

    lines, controls = controls_section(follow=follow, root=root, ledger=ledger, smoke=smoke)
    add("1. Positive controls (post hoc)", lines, controls=controls)
    lines, long_primary, long_sensitivity = long_leads_section(
        first=first, follow=follow, root=root, ledger=ledger, smoke=smoke
    )
    add(
        "2. Long leads (post hoc)",
        lines,
        long_leads=long_primary,
        long_leads_sensitivity=long_sensitivity,
    )
    lines, pc2 = pc2_section(first=first, follow=follow, ledger=ledger)
    add("3. PC with a 2-day latency (post hoc)", lines, pc2=pc2)
    lines, fingerprint = fingerprint_section(first=first, follow=follow, ledger=ledger)
    add("4. Fingerprint decomposition (post hoc)", lines, fingerprint_decomposition=fingerprint)
    lines, unselected = unselected_section(first=first, ledger=ledger)
    add("5. The 8 months the sweep never screened (post hoc)", lines, unselected=unselected)
    lines, hits = hit_rate_section(root=root, smoke=smoke)
    add("6. Per-quantile hit rates (post hoc)", lines, hit_rates=hits)
    lines, hindsight = hindsight_section(
        first=first, follow=follow, root=root, ledger=ledger, smoke=smoke
    )
    add("8. The hindsight scaling bound (post hoc)", lines, hindsight=hindsight)
    lines, t_table = t_interval_section(first=first, root=root, smoke=smoke)
    add(
        "9. Month-cluster t-intervals for the planned contrasts (post hoc)",
        lines,
        t_intervals=t_table,
    )
    first_records = first_report_intervals(path=root / PRODUCT / f"report_{PRODUCT}{suffix}.md")
    lines, counts = count_lines(first=first_records, ledger=ledger)
    add("7. Count of exploratory intervals (post hoc)", lines, interval_counts=counts)

    out = ["# Follow-ups after the first science review (post hoc)", ""]
    for title, body, named in sorted(sections, key=lambda s: s[0]):
        out += [f"## {title}", "", *body]
        for name, table in named.items():
            if not table.is_empty():
                write_parquet_atomic(frame=table, path=tables / f"{name}.parquet")
    out += [
        "## Scored months",
        "",
        f"The study scores {len(scored_months())} calendar months: {', '.join(scored_months())}.",
        "",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    """Write the follow-up report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=LAG_FEATURES_DIR)
    parser.add_argument("--smoke", action="store_true", help="Report a `--smoke` fit's output.")
    arguments = parser.parse_args()
    root: Path = arguments.output_root
    suffix = run_suffix(smoke=arguments.smoke)
    directory = followup_dir(root=root)
    path = directory / f"report_followups_{PRODUCT}{suffix}.md"
    final_tables = directory / f"tables_followups_{PRODUCT}{suffix}"
    refuse_to_overwrite(paths=[path, final_tables])
    partial_tables = directory / f"tables_followups_{PRODUCT}{suffix}.partial"
    if partial_tables.exists():
        shutil.rmtree(partial_tables)
    partial_tables.mkdir()
    text = report_text(root=root, tables=partial_tables, smoke=arguments.smoke)
    partial_tables.rename(final_tables)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    partial.replace(path)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
