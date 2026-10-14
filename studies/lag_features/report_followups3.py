"""Print every number the round-3 post hoc analyses quote, and save the tables the charts read.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1138>. The
third science review accepted the planned results and asked for two small fits and several no-fit
computations. This script reads the first run's losses, round 1's and round 2's, and (when they
exist) round 3's fit losses, writes
`<output-root>/ens_mean/followups3/report_followups3_ens_mean.md`
and `tables_followups3_ens_mean/`, and never writes into an earlier run's files. **Every number is
post hoc**, computed after the first run's results were known.

**The no-fit sections read only saved files:**

1. Each plant's output relative to the other plants by month, for the audit.
2. Width-matched null contrasts over 18 months without the dominant plant-month.
3. Month-cluster t-intervals beside the bootstrap intervals for PC, W7, TF, Q30 and AN.
4. PC, W7, and TF against B0 refitted without `day_of_year`.
5. The same-rows comparison at lead-day 14, and the blend against W7.
6. Which cleaning rules look at later hours.
7. The equivalence table with t-intervals.

**The fit sections** (round 3's two fit blocks) read `losses_followups3_ens_mean.parquet`, and say
so while it does not exist. A report written before the fits is named
`report_followups3_ens_mean_nofit.md`, with `tables_followups3_ens_mean_nofit/`.

Run it with `uv run python studies/lag_features/report_followups3.py`.
"""

import argparse
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from build_lag_frame import LAG_FEATURES_DIR, lag_source_hourly, run_suffix, write_parquet_atomic
from fit_followups2 import verify_followup2_manifest
from fit_followups3 import verify_followup3_manifest
from fit_lag_arms import METRIC, verify_manifest_inputs
from followup2_frames import followup2_dir
from followup3_plan import followup3_dir, plan_path
from followup_frames import PRODUCT
from report_followups import verify_followup_manifest
from report_followups2 import (
    BLEND_SHARE,
    Aligned,
    _interval,
    _pp,
    _table,
    climatology_section,
    dominant_plant_month,
    mean_difference,
    month_bootstrap,
    read_losses,
    wide,
)
from report_lag_features import P2_FIRST_MONTH, PERCENTAGE_POINTS, paired_interval
from scipy import stats
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("report_followups3")

POST_HOC: Final[str] = "Post hoc: written after the first run's results were known."
"""The line under every section heading."""

T_LEVEL: Final[float] = 95.0
"""The coverage of the month-cluster t-intervals, in percent."""

THRESHOLD_SHARE: Final[float] = 0.02
"""The decision threshold, as a share of the reference arm's own metric."""

REVIEWED_ARMS: Final[tuple[str, ...]] = ("PC", "W7", "TF", "Q30", "AN")
"""The arms the t-interval table covers."""

WIDTH_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("PC", "N2-2"),
    ("TF", "N2-2"),
    ("Q30", "N2-2"),
    ("W7", "N2-3"),
    ("AN", "N2-3"),
)
"""The arms paired with the null of the same number of columns."""

NO_DAY_OF_YEAR: Final[str] = "-noDOY"
"""The suffix of an arm fitted without `day_of_year`."""

DERATED_ARMS: Final[tuple[str, ...]] = ("PC", "W7", "TF", "Q30")
"""The arms compared with B0 in the derated-months refit."""

NO_DOY_ARMS_ROUND3: Final[tuple[str, ...]] = ("L1", "Q30", "AN", "N2-2")
"""The arms round 3 refits without `day_of_year`."""


# --- Helpers -------------------------------------------------------------------------------------


def joined_aligned(*, parts: list[tuple[pl.DataFrame, Mapping[str, str]]]) -> Aligned:
    """Align series from several loss sources on their common rows.

    Args:
        parts: Pairs of a loss frame and the column name to give each arm kept from it.

    Returns:
        The series on the rows (and seeds) every source holds.
    """
    frames = [wide(losses=losses, names=names) for losses, names in parts]
    joined = frames[0]
    for frame in frames[1:]:
        joined = joined.join(frame.drop("month"), on=["site", "time", "seed"], how="inner")
    return Aligned(frame=joined, names=[name for _, names in parts for name in names.values()])


def month_t_interval(
    *, difference: np.ndarray, months: np.ndarray, level: float = T_LEVEL
) -> tuple[float, float, float, int]:
    """Return the month-cluster t-interval of a paired difference.

    Each calendar month's mean difference (over rows and seeds) is one observation, so the
    interval has one fewer degree of freedom than there are months.

    Args:
        difference: Per-seed, per-row differences, shape (seeds, rows).
        months: Each row's month label.
        level: The coverage in percent.

    Returns:
        The mean of the monthly means, the interval's lower and upper ends, and the months.
    """
    labels = np.unique(months)
    means = np.array([difference[:, months == label].mean() for label in labels])
    centre = float(means.mean())
    half = float(
        stats.t.ppf(0.5 + level / 200.0, df=len(means) - 1)
        * means.std(ddof=1)
        / np.sqrt(len(means))
    )
    return centre, centre - half, centre + half, len(means)


def contrast_record(
    *, data: Aligned, treatment: str, reference: str, label: str, **fields: object
) -> dict[str, object]:
    """Compute a paired contrast with a 95% bootstrap interval and a 95% t-interval.

    Args:
        data: The aligned series.
        treatment: The series compared.
        reference: The series it is compared with.
        label: A name for the row.
        **fields: Extra columns for the record, such as the window.

    Returns:
        The record.
    """
    boot = mean_difference(data=data, treatment=treatment, reference=reference)
    centre, t_lower, t_upper, n_months = month_t_interval(
        difference=data.values[treatment] - data.values[reference], months=data.month
    )
    return {
        **fields,
        "contrast": label,
        "difference": boot["value"],
        "boot_lower": boot["lower_95"],
        "boot_upper": boot["upper_95"],
        "t_centre": centre,
        "t_lower": t_lower,
        "t_upper": t_upper,
        "n_months": n_months,
        "n_rows": data.n_rows,
    }


def record_rows(*, records: list[dict[str, object]], extra: list[str]) -> list[list[str]]:
    """Format contrast records as table rows.

    Args:
        records: Rows from `contrast_record`.
        extra: The record keys to print before the contrast's name.

    Returns:
        One list of cells per record.
    """
    return [
        [
            *(str(r[key]) for key in extra),
            str(r["contrast"]),
            _pp(float(r["difference"])),  # ty: ignore[invalid-argument-type]
            _interval(float(r["boot_lower"]), float(r["boot_upper"])),  # ty: ignore[invalid-argument-type]
            _pp(float(r["t_centre"])),  # ty: ignore[invalid-argument-type]
            _interval(float(r["t_lower"]), float(r["t_upper"])),  # ty: ignore[invalid-argument-type]
            str(r["n_months"]),
        ]
        for r in records
    ]


CONTRAST_HEADER: Final[list[str]] = [
    "Contrast",
    "Difference (pp)",
    "95% bootstrap interval (pp)",
    "Centre of the t-interval (pp)",
    "95% t-interval (pp)",
    "Months",
]
"""The columns `record_rows` fills after the extra keys. The t-interval's centre is the unweighted
mean of the monthly means, which can differ from the row-weighted difference beside it."""


def windows_and_exclusions(
    *, data: Aligned, site: str, month: str
) -> list[tuple[str, str, Aligned]]:
    """Return each window with and without the dominant plant-month.

    Args:
        data: The aligned series on all 18 months.
        site: The dominant plant-month's plant.
        month: Its month.

    Returns:
        (window, rows, data) triples for the 8 unselected months and for all 18 months, each with
        all rows and without the plant-month the rule named in the 8 unselected months.
    """
    eight = [m for m in np.unique(data.month) if m >= P2_FIRST_MONTH]
    out = []
    for window, subset in (
        ("8 unselected months", data.in_months(months=eight)),
        ("all 18 months", data),
    ):
        out.append((window, "all rows", subset))
        keep = ~((subset.site == site) & (subset.month == month))
        out.append(
            (
                window,
                "without the largest-error plant-month of the 8 unselected months",
                subset.where(keep=keep),
            )
        )
    return out


# --- 1. Output relative to the other plants ------------------------------------------------------


def output_ratio_section(
    *, root: Path, lead1: pl.DataFrame, fits2: pl.DataFrame
) -> tuple[list[str], pl.DataFrame]:
    """Print the dominant plant's output relative to the others, and B0 without day of year.

    Args:
        root: The output root.
        lead1: The first run's lead-day 1 losses.
        fits2: Round 2's losses, for B0 without `day_of_year`.

    Returns:
        The report lines and a table with one row per month of the plant.
    """
    plan = json.loads(plan_path(root=root).read_text())
    site = str(plan["site"])
    ratios = pl.read_parquet(followup3_dir(root=root) / f"output_ratios_{PRODUCT}.parquet")
    mine = ratios.filter(pl.col("site") == site).select(
        "month", "hours", "ratio", "n_alt", "ratio_alt"
    )
    cells = lead1.filter((pl.col("site") == site) & (pl.col("arm") == "B0")).select(
        "month",
        "seed",
        "time",
        with_year=pl.col(METRIC),
        with_bias=pl.col("signed_error_capped_mw") / pl.col("effective_capacity_mw"),
    )
    without = fits2.filter(
        (pl.col("site") == site) & (pl.col("arm") == f"B0{NO_DAY_OF_YEAR}")
    ).select(
        "month",
        "seed",
        "time",
        without_year=pl.col(METRIC),
        without_bias=pl.col("signed_error_capped_mw") / pl.col("effective_capacity_mw"),
    )
    joined = (
        cells.join(without, on=["month", "seed", "time"])
        .group_by("month")
        .agg(
            b0_error=pl.col("with_year").mean(),
            nodoy_error=pl.col("without_year").mean(),
            b0_bias=pl.col("with_bias").mean(),
            nodoy_bias=pl.col("without_bias").mean(),
        )
    )
    interest = set(plan["months"])
    table = (
        mine.join(joined, on="month", how="full", coalesce=True)
        .sort("month")
        .with_columns(
            difference=pl.col("nodoy_error") - pl.col("b0_error"),
            derated=pl.col("month").is_in(list(interest)),
        )
    )
    lines = [
        (
            "Audit, not for publication (the plant is named by letter). The dominant plant is "
            f"plant {site}: the plant with the largest mean B0 error in a single month of the 8 "
            "unselected months. The ratio is the plant's mean capacity-normalised output as a "
            "share of the mean of the other five plants' output, on the hours at which all six "
            "plants are scored. The derated months are the months below "
            f"{plan['threshold']:.0%}. Errors are B0's mean absolute error and mean signed error "
            "(positive is over-forecast) in percentage points of capacity, with and without "
            "`day_of_year`."
        ),
        "",
        (
            "The selection of derated months is "
            + (
                "robust to the alternative ratio: the same months fall below the threshold."
                if plan["selection_robust"]
                else "not robust to the alternative ratio: the alternative gives "
                f"{', '.join(plan['months_alternative'])}."
            )
        ),
        "",
        *_table(
            header=[
                "Month",
                "Output ratio",
                "Hours at which all six are scored",
                "Alternative ratio (the plant and at least two others)",
                "Hours of the alternative",
                "Derated",
                "B0 error",
                "B0-noDOY error",
                "B0-noDOY minus B0",
                "B0 bias",
                "B0-noDOY bias",
            ],
            rows=[
                [
                    r["month"],
                    "" if r["ratio"] is None else f"{r['ratio']:.3f}",
                    "" if r["hours"] is None else str(r["hours"]),
                    "" if r["ratio_alt"] is None else f"{r['ratio_alt']:.3f}",
                    "" if r["n_alt"] is None else str(r["n_alt"]),
                    "yes" if r["derated"] else "",
                    _pp(r["b0_error"], signed=False),
                    _pp(r["nodoy_error"], signed=False),
                    _pp(r["difference"]),
                    _pp(r["b0_bias"]),
                    _pp(r["nodoy_bias"]),
                ]
                for r in table.iter_rows(named=True)
            ],
        ),
    ]
    return lines, table


# --- 2 and 3. Width-matched nulls and t-intervals -------------------------------------------------


def contrasts_sections(
    *, lead1: pl.DataFrame, fits2: pl.DataFrame
) -> tuple[list[str], dict[str, pl.DataFrame]]:
    """Print the width-matched null contrasts, the t-intervals and the no-day-of-year contrasts.

    Args:
        lead1: The first run's lead-day 1 losses.
        fits2: Round 2's losses.

    Returns:
        The report lines and the tables.
    """
    arms = (*REVIEWED_ARMS, "B0")
    data = joined_aligned(
        parts=[
            (lead1, {arm: arm for arm in arms}),
            (
                fits2,
                {
                    "N2-2": "N2-2",
                    "N2-3": "N2-3",
                    **{
                        f"{a}{NO_DAY_OF_YEAR}": f"{a}{NO_DAY_OF_YEAR}"
                        for a in ("B0", *REVIEWED_ARMS[:3])
                    },
                },
            ),
        ]
    )
    b0 = lead1.filter(pl.col("arm") == "B0")
    eight = [m for m in sorted(set(lead1["month"].to_list())) if m >= P2_FIRST_MONTH]
    site, month, _, _ = dominant_plant_month(b0=b0, months=eight)
    variants = windows_and_exclusions(data=data, site=site, month=month)
    tables: dict[str, pl.DataFrame] = {}
    lines: list[str] = []

    # 2. Width-matched null contrasts over 18 months without the dominant plant-month.
    removed = data.where(keep=~((data.site == site) & (data.month == month)))
    width = [
        contrast_record(
            data=removed,
            treatment=treatment,
            reference=reference,
            label=f"{treatment} minus {reference}",
        )
        for treatment, reference in WIDTH_PAIRS
    ]
    tables["width_null_contrasts"] = pl.DataFrame(width)
    lines += [
        "### 2. Width-matched null contrasts over 18 months, without the dominant plant-month",
        "",
        POST_HOC,
        "",
        *_table(header=CONTRAST_HEADER, rows=record_rows(records=width, extra=[])),
    ]

    # 3. t-intervals beside the bootstrap intervals.
    t_records = [
        contrast_record(
            data=subset,
            treatment=arm,
            reference="B0",
            label=f"{arm} minus B0",
            window=window,
            rows=rows,
        )
        for window, rows, subset in variants
        for arm in REVIEWED_ARMS
    ]
    tables["t_intervals_arms"] = pl.DataFrame(t_records)
    lines += [
        "### 3. Month-cluster t-intervals beside the bootstrap intervals",
        "",
        POST_HOC,
        "",
        (
            "The t-interval treats each calendar month's mean difference as one observation "
            "(degrees of freedom: months minus one). The plant-month removed is the largest-error "
            "plant-month of the 8 unselected months."
        ),
        "",
        *_table(
            header=["Window", "Rows", *CONTRAST_HEADER],
            rows=record_rows(records=t_records, extra=["window", "rows"]),
        ),
    ]

    # 4. PC, W7, TF and B0-noDOY against B0 and each other.
    doy: list[dict[str, object]] = []
    for window, rows, subset in variants:
        doy += [
            contrast_record(
                data=subset,
                treatment=f"{arm}",
                reference=f"B0{NO_DAY_OF_YEAR}",
                label=f"{arm} minus B0-noDOY",
                window=window,
                rows=rows,
            )
            for arm in ("PC", "W7", "TF")
        ]
        doy.append(
            contrast_record(
                data=subset,
                treatment=f"B0{NO_DAY_OF_YEAR}",
                reference="B0",
                label="B0-noDOY minus B0",
                window=window,
                rows=rows,
            )
        )
    tables["against_b0_nodoy"] = pl.DataFrame(doy)
    lines += [
        "### 4. PC, W7, and TF against B0 refitted without day of year",
        "",
        POST_HOC,
        "",
        *_table(
            header=["Window", "Rows", *CONTRAST_HEADER],
            rows=record_rows(records=doy, extra=["window", "rows"]),
        ),
    ]
    return lines, tables


# --- 5. Same rows at lead-day 14 -----------------------------------------------------------------


def day14_section(*, first_path: Path, root: Path) -> tuple[list[str], pl.DataFrame]:
    """Compare W7 with B0 and with B0's blend, on the rows where past-only climatology exists.

    Args:
        first_path: The first run's losses.
        root: The output root.

    Returns:
        The report lines and a table with one row per (lead-day, contrast).
    """
    kept: dict[int, pl.DataFrame] = {}
    climatology_section(first_path=first_path, root=root, hourly=lag_source_hourly(), keep=kept)
    blend = "B0 x past-only climatology"
    records = []
    for lead, scored in sorted(kept.items()):
        for treatment, reference in (("W7", "B0"), (blend, "W7"), (blend, "B0")):
            interval = paired_interval(
                scoped=scored.filter(pl.col("arm").is_in([treatment, reference])),
                treatment=treatment,
                reference=reference,
                metric=METRIC,
                level=95.0,
            )
            if interval is not None:
                records.append(
                    {
                        "lead_day": lead,
                        "contrast": f"{treatment} minus {reference}",
                        "difference": interval["difference"],
                        "lower": interval["lower_95"],
                        "upper": interval["upper_95"],
                        "n_rows": scored.filter(pl.col("arm") == reference).height // 3,
                    }
                )
    table = pl.DataFrame(records)
    lines = [
        (
            "Post hoc. On exactly the rows at which the past-only climatology exists (paired by "
            f"row and seed, 95% bootstrap interval from resampling months and a seed). The blend "
            f"is {1 - BLEND_SHARE:.0%} B0 and {BLEND_SHARE:.0%} past-only climatology."
        ),
        "",
        *_table(
            header=["Lead-day", "Contrast", "Difference (pp)", "95% interval (pp)", "Rows"],
            rows=[
                [
                    str(r["lead_day"]),
                    r["contrast"],
                    _pp(r["difference"]),
                    _interval(r["lower"], r["upper"]),
                    str(r["n_rows"]),
                ]
                for r in table.iter_rows(named=True)
            ],
        ),
    ]
    return lines, table


# --- 6. Cleaning causality -----------------------------------------------------------------------


def _cell(*parts: str) -> str:
    """Join the parts of one table cell."""
    return "".join(parts)


def cleaning_lines() -> list[str]:
    """State which cleaning rules look at hours after the one they flag.

    The text is read from `nged_data.cleaning`, `studies.pv_dataset`, `studies.commissioning`,
    `studies.export_cap` and `studies.power`. A lag in the backtest is read from the cleaned
    record, so a rule that looks ahead makes the backtest's lag cleaner than a live service's.

    Returns:
        The report lines.
    """
    rows = [
        [
            _cell(
                "`substation_zero` (`nged_data.cleaning.flag_nged_power`, read through ",
                "`scan_cleaned_power`)",
            ),
            "one half-hour",
            "none (the reading alone)",
            _cell(
                "a reading of exactly 0 from a Primary, BSP, or GSP series. The study's ",
                "photovoltaic plants are customer series, so the rule flags none of them.",
            ),
        ],
        [
            "spike filter (`studies.pv_dataset.drop_outages_and_spikes`)",
            "one hour",
            "none (the hour alone)",
            _cell(
                "an hour above 1.5 times the plant's effective capacity. The threshold is a ",
                "constant of the whole record (each plant's last recorded capacity) and removes ",
                "no rows today: the largest hourly power is 1.38 times capacity.",
            ),
        ],
        [
            "zero-run rule (`studies.pv_dataset.drop_outages_and_spikes`)",
            "a run of consecutive hourly zeros, flagged when the run reaches 24 hours",
            _cell(
                "**backward and forward**: the run's length is known only at its end, and ",
                "every hour of a long run is removed, including the first 23",
            ),
            _cell(
                "every hour of a run of at least 24 consecutive zero hours (a gap in the ",
                "readings ends a run). An hour inside a run that a live service would still ",
                "see as short, and keep as a zero, is removed in the backtest.",
            ),
        ],
        [
            "commissioning ramp (`studies.commissioning.drop_commissioning_ramp`)",
            "all rows before a fixed date for one plant",
            "the date was fitted on the later record, but the rule is a fixed cut-off",
            "the rows before the plant reached its settled output (before October 2024).",
        ],
        [
            "export cap and `constrained` (`studies.export_cap.with_export_cap`)",
            _cell(
                "the cap readings inside the hour, and a cut-off at the first reading at the ",
                "connection limit",
            ),
            _cell(
                "the flag looks at the hour itself; the cut-off is a fixed date taken from ",
                "the whole cap record",
            ),
            _cell(
                "an hour in which the operator had moved the cap off the connection limit ",
                "(one plant); the lag source drops these hours. The connection limit is the ",
                "largest cap in the plant's whole record.",
            ),
        ],
        [
            "hourly average (`studies.power.hourly_from_half_hourly`)",
            "the two half-hours ending in the hour",
            "none beyond the hour",
            _cell(
                "not a flag: the hour is the mean of its two half-hours, and an hour missing ",
                "either half-hour is dropped.",
            ),
        ],
        [
            "read cut-off",
            "all rows from 2026-07-01",
            "a fixed date",
            "no row from that date is read at all.",
        ],
    ]
    return [
        _cell(
            "Read from the code. A rule is **causal** when its decision about an hour uses only ",
            "that hour and earlier hours. A rule that is not causal can remove a bad hour using ",
            "what happened after it, which a live service, deciding at the issue time, cannot do.",
        ),
        "",
        *_table(header=["Rule (module)", "Window", "Direction", "What it flags"], rows=rows),
        _cell(
            "**Summary.** Only the zero-run rule looks at later hours when it flags an hour. ",
            "A lag in the backtest that lies in a long run of zeros is removed, so the backtest ",
            "can see a cleaner lag than a live service, which would read that hour as zero until ",
            "the run reached 24 hours. The commissioning and export-cap cut-offs are fixed dates ",
            "chosen with the whole record in view, but they decide by date and not by looking ",
            "ahead from an hour. The false-zero filter in `studies.pv_dataset` is not applied ",
            "to the lag source `lag_source_hourly` reads.",
        ),
        "",
    ]


# --- 7. Equivalence with t-intervals -------------------------------------------------------------


def equivalence_t_section(*, root: Path) -> tuple[list[str], pl.DataFrame]:
    """Add the t-interval to the equivalence table of the planned contrasts.

    Args:
        root: The output root.

    Returns:
        The report lines and a table with one row per (contrast, setting).
    """
    planned = pl.read_parquet(root / PRODUCT / f"tables_{PRODUCT}" / "planned.parquet").select(
        "contrast", "setting", "reference_value", "lower", "upper"
    )
    t_table = pl.read_parquet(
        root / PRODUCT / "followups" / f"tables_followups_{PRODUCT}" / "t_intervals.parquet"
    ).select("contrast", "setting", "t_lower", "t_upper")
    table = (
        planned.join(t_table, on=["contrast", "setting"])
        .with_columns(threshold=THRESHOLD_SHARE * pl.col("reference_value"))
        .with_columns(
            bootstrap_excludes=pl.col("lower") > -pl.col("threshold"),
            t_excludes=pl.col("t_lower") > -pl.col("threshold"),
        )
        .sort("contrast", "setting")
    )
    lines = [
        (
            "Post hoc. A 99% interval excludes a gain as large as the 2% threshold when its lower "
            "end is above minus the threshold. The t-interval is the month-cluster 99% "
            "t-interval of the first report's planned contrasts."
        ),
        "",
        *_table(
            header=[
                "Contrast",
                "Setting",
                "Threshold, 2% (pp)",
                "99% bootstrap lower (pp)",
                "99% t-interval lower (pp)",
                "Bootstrap excludes the threshold",
                "t-interval also excludes it",
            ],
            rows=[
                [
                    r["contrast"],
                    r["setting"],
                    _pp(r["threshold"], signed=False),
                    _pp(r["lower"]),
                    _pp(r["t_lower"]),
                    "yes" if r["bootstrap_excludes"] else "no",
                    "yes" if r["t_excludes"] else "no",
                ]
                for r in table.iter_rows(named=True)
            ],
        ),
    ]
    return lines, table


# --- Round 3's fits ------------------------------------------------------------------------------


def gap_windows(
    *, eight: list[str], all_months: list[str], dominant_month: str, derated_months: list[str]
) -> list[tuple[str, list[str]]]:
    """Return the windows the chosen plant's gaps are tabulated on.

    Args:
        eight: The 8 unselected months.
        all_months: All 18 scored months.
        dominant_month: The month of the dominant plant-month.
        derated_months: The chosen plant's derated months.

    Returns:
        (name, months) pairs.
    """
    return [
        ("8 unselected months", eight),
        ("all 18 months", all_months),
        ("the dominant plant-month alone", [dominant_month]),
        (
            "scored without the derated months, all 18 months",
            [m for m in all_months if m not in derated_months],
        ),
        (
            "scored without the derated months, 8 unselected months",
            [m for m in eight if m not in derated_months],
        ),
    ]


def gap_section(
    *, plant: Aligned, windows: list[tuple[str, list[str]]]
) -> tuple[list[str], pl.DataFrame]:
    """Tabulate each arm's gap to B0 at the chosen plant from both fits, and the change in it.

    The gap is the arm minus B0 fitted the same way. The change is the gap with the derated
    months removed from training minus the gap with all training months, paired by row and seed.

    Args:
        plant: The chosen plant's series `<arm>_all` and `<arm>_der`.
        windows: (name, months) pairs.

    Returns:
        The report lines and a table with one row per (window, arm).
    """
    records = []
    for window, months in windows:
        subset = plant.in_months(months=months)
        if subset.n_rows == 0:
            continue
        for arm in DERATED_ARMS:
            gap_all = mean_difference(data=subset, treatment=f"{arm}_all", reference="B0_all")
            gap_der = mean_difference(data=subset, treatment=f"{arm}_der", reference="B0_der")
            change = month_bootstrap(
                data=subset,
                statistic=lambda d, arm=arm: float(
                    ((d[f"{arm}_der"] - d["B0_der"]) - (d[f"{arm}_all"] - d["B0_all"])).mean()
                ),
            )
            records.append(
                {
                    "window": window,
                    "arm": arm,
                    "gap_all": gap_all["value"],
                    "gap_all_lower": gap_all["lower_95"],
                    "gap_all_upper": gap_all["upper_95"],
                    "gap_derated": gap_der["value"],
                    "gap_derated_lower": gap_der["lower_95"],
                    "gap_derated_upper": gap_der["upper_95"],
                    "change": change["value"],
                    "change_lower": change["lower_95"],
                    "change_upper": change["upper_95"],
                    "n_months": len(months),
                    "n_rows": subset.n_rows,
                }
            )
    table = pl.DataFrame(records)
    lines = [
        "### The chosen plant's gap to B0 from both fits, and the change in it",
        "",
        (
            "The gap is each arm minus B0 fitted the same way, with all training months and with "
            "the derated months removed from training. The change is (gap with the derated "
            "months removed) minus (gap with all months), paired by row and seed, with a 95% "
            "interval from resampling months and a seed. A window of one month has only the seed "
            "to resample, so its interval is narrow and does not describe month-to-month "
            "variation."
        ),
        "",
        *_table(
            header=[
                "Rows scored (the chosen plant)",
                "Months",
                "Arm",
                "Gap, all training (pp)",
                "Interval (pp)",
                "Gap, derated months removed (pp)",
                "Interval (pp)",
                "Change in the gap (pp)",
                "Interval (pp)",
            ],
            rows=[
                [
                    r["window"],
                    str(r["n_months"]),
                    r["arm"],
                    _pp(r["gap_all"]),
                    _interval(r["gap_all_lower"], r["gap_all_upper"]),
                    _pp(r["gap_derated"]),
                    _interval(r["gap_derated_lower"], r["gap_derated_upper"]),
                    _pp(r["change"]),
                    _interval(r["change_lower"], r["change_upper"]),
                ]
                for r in table.iter_rows(named=True)
            ],
        ),
    ]
    return lines, table


def plant_control_lines(
    *, first: pl.DataFrame, control: pl.DataFrame
) -> tuple[list[str], pl.DataFrame]:
    """Compare the control refit with the first run's B0 at the chosen plant.

    Args:
        first: The first run's lead-day 1 losses at the chosen plant.
        control: The control's losses: B0 on the chosen plant's rows with nothing removed.

    Returns:
        The report lines and a one-row table of the differences.
    """
    pair = (
        first.filter(pl.col("arm") == "B0")
        .select(
            "site", "time", "seed", loss_first=pl.col(METRIC), prediction_first=pl.col("prediction")
        )
        .join(
            control.filter(pl.col("arm") == "B0").select(
                "site",
                "time",
                "seed",
                loss_control=pl.col(METRIC),
                prediction_control=pl.col("prediction"),
            ),
            on=["site", "time", "seed"],
            how="inner",
        )
        .with_columns(
            loss_difference=(pl.col("loss_control") - pl.col("loss_first")).abs(),
            prediction_difference=(pl.col("prediction_control") - pl.col("prediction_first")).abs(),
        )
    )
    stats_row = pair.select(
        rows=pl.len(),
        rows_differing=(pl.col("prediction_difference") > 0).sum(),
        max_prediction_difference_mw=pl.col("prediction_difference").max(),
        max_loss_difference_pp=pl.col("loss_difference").max() * PERCENTAGE_POINTS,
        mean_loss_difference_pp=pl.col("loss_difference").mean() * PERCENTAGE_POINTS,
    )
    table = stats_row
    row = table.row(0, named=True)
    verdict = (
        "bit for bit identical to the first run's B0 at the plant, so the refits are paired with "
        "the first run's fits without error"
        if row["rows_differing"] == 0
        else (
            f"not bit for bit: {row['rows_differing']} of {row['rows']} predictions differ, by "
            f"at most {row['max_prediction_difference_mw']:.2e} MW, which is a mean absolute "
            f"loss difference of {row['mean_loss_difference_pp']:.2e} pp"
        )
    )
    lines = [
        "### Control: B0 on the chosen plant with nothing removed",
        "",
        (
            f"The control refit is {verdict}. This shows whether a refit on the chosen plant's "
            "rows alone matches the first run's B0 at that plant."
        ),
        "",
    ]
    return lines, table


def did_record(*, data: Aligned, arm: str, window: str) -> dict[str, object]:
    """Compute the difference-in-differences of an arm's gap to B0 with and without day of year.

    The statistic is (arm minus B0) minus (arm-noDOY minus B0-noDOY), paired by row and seed.

    Args:
        data: The aligned series, with each arm, B0, and their `-noDOY` refits.
        arm: The arm.
        window: The window's name, for the record.

    Returns:
        The record, as `contrast_record` returns it.
    """
    without = f"{arm}{NO_DAY_OF_YEAR}"
    boot = month_bootstrap(
        data=data,
        statistic=lambda d: float(
            ((d[arm] - d["B0"]) - (d[without] - d[f"B0{NO_DAY_OF_YEAR}"])).mean()
        ),
    )
    centre, t_lower, t_upper, n_months = month_t_interval(
        difference=(data.values[arm] - data.values["B0"])
        - (data.values[without] - data.values[f"B0{NO_DAY_OF_YEAR}"]),
        months=data.month,
    )
    return {
        "window": window,
        "contrast": f"({arm} minus B0) minus ({arm}-noDOY minus B0-noDOY)",
        "difference": boot["value"],
        "boot_lower": boot["lower_95"],
        "boot_upper": boot["upper_95"],
        "t_centre": centre,
        "t_lower": t_lower,
        "t_upper": t_upper,
        "n_months": n_months,
        "n_rows": data.n_rows,
    }


def fit_sections(  # noqa: PLR0915
    *, root: Path, lead1: pl.DataFrame, fits2: pl.DataFrame, fit_path: Path
) -> tuple[list[str], dict[str, pl.DataFrame]]:
    """Report the two fit blocks, or say that the fits have not run.

    Args:
        root: The output root.
        lead1: The first run's lead-day 1 losses.
        fits2: Round 2's losses.
        fit_path: Round 3's losses.

    Returns:
        The report lines and the tables.
    """
    if not fit_path.exists():
        return [
            (
                "The round-3 fits have not been run, so this section is empty. Re-run this "
                "report after `fit_followups3.py`."
            ),
            "",
        ], {}
    plan = json.loads(plan_path(root=root).read_text())
    site = str(plan["site"])
    fits = read_losses(path=fit_path, scopes=["fu3_derated", "fu3_lead1", "fu3_plant_control"])
    derated = fits.filter(pl.col("scope") == "fu3_derated").drop("scope")
    tables: dict[str, pl.DataFrame] = {}
    lines: list[str] = []
    # 1. Derated months removed from training at the plant.
    eight = [m for m in sorted(set(lead1["month"].to_list())) if m >= P2_FIRST_MONTH]
    b0_site, b0_month, _, _ = dominant_plant_month(
        b0=lead1.filter(pl.col("arm") == "B0"), months=eight
    )
    names_all = {a: f"{a}_all" for a in ("B0", *DERATED_ARMS)}
    names_der = {a: f"{a}_der" for a in ("B0", *DERATED_ARMS)}
    plant_first = lead1.filter(pl.col("site") == site)
    plant = joined_aligned(parts=[(plant_first, names_all), (derated, names_der)])
    # The fleet: every other plant from the first run, the chosen plant from the refit.
    others = lead1.filter(pl.col("site") != site)
    fleet_all = joined_aligned(parts=[(lead1, names_all)])
    fleet_mixed = joined_aligned(
        parts=[
            (
                pl.concat([others, derated]),
                names_der,
            )
        ]
    )
    records = []
    for window, months in (
        ("8 unselected months", eight),
        ("all 18 months", sorted(set(lead1["month"].to_list()))),
    ):
        for scope, aligned_all, aligned_der in (
            ("the chosen plant only", plant, plant),
            ("all six plants", fleet_all, fleet_mixed),
        ):
            for arm in DERATED_ARMS:
                if scope == "the chosen plant only":
                    subset = aligned_all.in_months(months=months)
                    all_train = mean_difference(
                        data=subset, treatment=f"{arm}_all", reference="B0_all"
                    )
                    der_train = mean_difference(
                        data=subset, treatment=f"{arm}_der", reference="B0_der"
                    )
                else:
                    a = aligned_all.in_months(months=months)
                    d = aligned_der.in_months(months=months)
                    all_train = mean_difference(data=a, treatment=f"{arm}_all", reference="B0_all")
                    der_train = mean_difference(data=d, treatment=f"{arm}_der", reference="B0_der")
                records.append(
                    {
                        "window": window,
                        "scope": scope,
                        "arm": arm,
                        "all_training": all_train["value"],
                        "all_lower": all_train["lower_95"],
                        "all_upper": all_train["upper_95"],
                        "derated_removed": der_train["value"],
                        "derated_lower": der_train["lower_95"],
                        "derated_upper": der_train["upper_95"],
                    }
                )
    table = pl.DataFrame(records)
    tables["derated_refit"] = table
    lines += [
        "### 1. Derated months removed from the chosen plant's training",
        "",
        POST_HOC,
        "",
        (
            "Each arm minus B0, with B0 fitted the same way: first with all training months (the "
            "first run), then with the chosen plant's derated months "
            f"({', '.join(plan['months'])}) removed from its training only. 'The chosen plant "
            "only' scores that plant's rows; 'all six plants' takes the other five plants' rows "
            "from the first run. Intervals are 95%, from resampling months and a seed."
        ),
        "",
        *_table(
            header=[
                "Window",
                "Rows scored",
                "Arm",
                "Difference, all training (pp)",
                "Interval (pp)",
                "Difference, derated months removed (pp)",
                "Interval (pp)",
            ],
            rows=[
                [
                    r["window"],
                    r["scope"],
                    r["arm"],
                    _pp(r["all_training"]),
                    _interval(r["all_lower"], r["all_upper"]),
                    _pp(r["derated_removed"]),
                    _interval(r["derated_lower"], r["derated_upper"]),
                ]
                for r in table.iter_rows(named=True)
            ],
        ),
    ]
    # The chosen plant's dominant month: B0 with and without the derated months in training.
    cell = joined_aligned(parts=[(plant_first, names_all), (derated, names_der)])
    month_rows = cell.in_months(months=[b0_month])
    if b0_site == site and month_rows.n_rows:
        shift = mean_difference(data=month_rows, treatment="B0_der", reference="B0_all")
        lines += [
            (
                f"B0's error in the dominant plant-month ({b0_site}, {b0_month}; audit, not for "
                "publication) with the derated months removed from training, minus B0's with all "
                f"training: {_pp(shift['value'])} pp "
                f"{_interval(shift['lower_95'], shift['upper_95'])}."
            ),
            "",
        ]
    gap_lines, gap_table = gap_section(
        plant=plant,
        windows=gap_windows(
            eight=eight,
            all_months=sorted(set(lead1["month"].to_list())),
            dominant_month=b0_month,
            derated_months=list(plan["months"]),
        ),
    )
    tables["derated_gaps"] = gap_table
    lines += gap_lines
    control_lines, control_table = plant_control_lines(
        first=lead1.filter(pl.col("site") == site),
        control=fits.filter(pl.col("scope") == "fu3_plant_control"),
    )
    tables["plant_control"] = control_table
    lines += control_lines
    # 2. No day of year for more arms.
    nodoy = fits.filter(pl.col("scope") == "fu3_lead1").drop("scope")
    arms = NO_DOY_ARMS_ROUND3
    data = joined_aligned(
        parts=[
            (lead1, {a: a for a in ("B0", *arms[:3])}),
            (fits2, {f"B0{NO_DAY_OF_YEAR}": f"B0{NO_DAY_OF_YEAR}", "N2-2": "N2-2"}),
            (nodoy, {f"{a}{NO_DAY_OF_YEAR}": f"{a}{NO_DAY_OF_YEAR}" for a in arms}),
        ]
    )
    eight_months = [m for m in np.unique(data.month) if m >= P2_FIRST_MONTH]
    doy_records: list[dict[str, object]] = []
    for window, subset in (
        ("8 unselected months", data.in_months(months=eight_months)),
        ("all 18 months", data),
    ):
        for arm in arms:
            with_name = "N2-2" if arm == "N2-2" else arm
            without_name = f"{arm}{NO_DAY_OF_YEAR}"
            doy_records.append(
                contrast_record(
                    data=subset,
                    treatment=without_name,
                    reference=with_name,
                    label=f"{arm} without day_of_year minus {arm} with it",
                    window=window,
                )
            )
            doy_records.append(
                contrast_record(
                    data=subset,
                    treatment=without_name,
                    reference=f"B0{NO_DAY_OF_YEAR}",
                    label=f"{arm}-noDOY minus B0-noDOY",
                    window=window,
                )
            )
            doy_records.append(
                contrast_record(
                    data=subset,
                    treatment=with_name,
                    reference="B0",
                    label=f"{arm} minus B0 (with day_of_year)",
                    window=window,
                )
            )
            doy_records.append(did_record(data=subset, arm=arm, window=window))
    tables["no_day_of_year_more"] = pl.DataFrame(doy_records)
    lines += [
        "### 2. No day of year for more arms",
        "",
        POST_HOC,
        "",
        *_table(
            header=["Window", *CONTRAST_HEADER],
            rows=record_rows(records=doy_records, extra=["window"]),
        ),
    ]
    return lines, tables


# --- The report ----------------------------------------------------------------------------------


def report_text(*, root: Path, tables: Path, smoke: bool) -> str:
    """Build the whole round-3 report and save the tables the charts read.

    Args:
        root: The output root.
        tables: The directory to write the tables into.
        smoke: Whether to report a smoke run's files.

    Returns:
        The report's markdown.
    """
    suffix = run_suffix(smoke=smoke)
    first_path = root / PRODUCT / f"losses_{PRODUCT}{suffix}.parquet"
    fits2_path = followup2_dir(root=root) / f"losses_followups2_{PRODUCT}{suffix}.parquet"
    fit_path = followup3_dir(root=root) / f"losses_followups3_{PRODUCT}{suffix}.parquet"
    lead1 = read_losses(path=first_path, scopes=["lead1"]).drop("scope")
    fits2 = read_losses(path=fits2_path, scopes=["fu2_lead1"]).drop("scope")
    out = [
        "# Round-3 follow-ups after the third science review (post hoc)",
        "",
        POST_HOC,
        "",
    ]
    named: dict[str, pl.DataFrame] = {}

    def add(title: str, lines: list[str], **tables_: pl.DataFrame) -> None:
        out.extend([f"## {title}", "", POST_HOC, "", *lines])
        named.update(tables_)

    lines, ratio_table = output_ratio_section(root=root, lead1=lead1, fits2=fits2)
    add(
        "1. Plant output relative to the other plants (audit)",
        lines,
        output_ratio_audit=ratio_table,
    )
    lines, contrast_tables = contrasts_sections(lead1=lead1, fits2=fits2)
    add(
        "2 to 4. Null contrasts, t-intervals, and the no-day-of-year baseline",
        lines,
        **contrast_tables,
    )
    lines, day14 = day14_section(first_path=first_path, root=root)
    add("5. The same rows at lead-day 14", lines, day14_same_rows=day14)
    add("6. Cleaning rules that look at later hours", cleaning_lines())
    lines, equivalence = equivalence_t_section(root=root)
    add("7. Equivalence with t-intervals", lines, equivalence_t=equivalence)
    lines, fit_tables = fit_sections(root=root, lead1=lead1, fits2=fits2, fit_path=fit_path)
    add("8. Round 3's fits (derated months, no day of year for more arms)", lines, **fit_tables)
    for name, table in named.items():
        if not table.is_empty():
            write_parquet_atomic(frame=table, path=tables / f"{name}.parquet")
    return "\n".join(out) + "\n"


def main() -> int:
    """Write the round-3 report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=LAG_FEATURES_DIR)
    parser.add_argument("--smoke", action="store_true", help="Report a `--smoke` fit's output.")
    arguments = parser.parse_args()
    root: Path = arguments.output_root
    suffix = run_suffix(smoke=arguments.smoke)
    directory = followup3_dir(root=root)
    fit_path = directory / f"losses_followups3_{PRODUCT}{suffix}.parquet"
    verify_manifest_inputs(root=root, product=PRODUCT, smoke=arguments.smoke)
    verify_followup_manifest(root=root, smoke=arguments.smoke)
    verify_followup2_manifest(root=root, smoke=arguments.smoke)
    if fit_path.exists():
        verify_followup3_manifest(root=root, smoke=arguments.smoke)
    tag = "" if fit_path.exists() else "_nofit"
    path = directory / f"report_followups3_{PRODUCT}{suffix}{tag}.md"
    final_tables = directory / f"tables_followups3_{PRODUCT}{suffix}{tag}"
    refuse_to_overwrite(paths=[path, final_tables])
    directory.mkdir(parents=True, exist_ok=True)
    partial_tables = final_tables.with_name(final_tables.name + ".partial")
    if partial_tables.exists():
        for old in partial_tables.iterdir():
            old.unlink()
        partial_tables.rmdir()
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
