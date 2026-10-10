"""Build the frames of the post hoc follow-ups to the lagged-power-features study.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1138>. The
first science review of the first run's results asked for these re-runs; every output is post hoc
and sits under `<output-root>/ens_mean/followups/`, with its own report, so no file of the first
run is read for writing or changed. The frames are:

- **Month-level positive controls** (`control_month_s<percent>`): the first run's control, power
  scaled by one minus the shift in 9 of the 18 scored months and a random half of the other months,
  at 5% and 10%, with the columns of B0, O, W7, Q30, TF, AN and PC built from the shifted power.
- **Plant-specific positive controls** (`control_step_s<percent>`): for each plant, persistent
  multiplicative steps of one minus the shift with a seeded random start and a random duration of 4
  to 12 weeks, applied to the hourly power before any lag is built, at 5% and 10%, with the columns
  of B0, O, L1, W7, Q30, TF and AN.
- **Long-lead frames** (`lead<N>` for N in 7, 10, 14): B0, W7, Q30 and N2's 16 random lags, plus
  `climatology_fold<k>`, the out-of-fold median power of the plant's calendar month and hour.
- **PC2** (`lead1_pc2`): PC with a 2-day CAMS latency, windows days 2 to 8 and days 2 to 31.

**O is an oracle.** O is B0 plus the true shift factor of the row (one minus the shift in a shifted
period, otherwise one), the best any lag feature could do about the shift.

**Every new column is anchored at the issue day.** The anchor assertions and a leak probe (rows
rebuilt from inputs cut at the issue time) run as in the first run's build. The probe runs on the
10% control of each kind, on lead-day 14 and on PC2, because the other frames share their code. The
climatology columns are `{fold}` columns: for a scored fold, a training row in another fold reads
the median of the folds outside both, like the stage-1 predictions.

Run it with `uv run python studies/lag_features/followup_frames.py`.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from build_lag_frame import (
    FULL_SWEEP_LEAD_DAY,
    LAG_FEATURES_DIR,
    WeatherProduct,
    arms_columns_table,
    assert_lag_arithmetic,
    build_frame,
    lag_inputs,
    lag_source_hourly,
    leak_probe,
    raw_hourly_power,
    scored_months,
    shared_rows,
    shifted_months,
    weather_table,
    write_parquet_atomic,
)
from studies.baselines import climatology
from studies.cross_validation import N_FOLDS
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("followup_frames")

PRODUCT: Final[WeatherProduct] = "ens_mean"
"""The follow-ups use the ENS mean only."""

CONTROL_SHIFTS: Final[tuple[float, ...]] = (0.05, 0.10)
"""The shifts of the follow-up positive controls, 5% and 10%; the 2% rule fails below 10%."""

MONTH_CONTROL_ARMS: Final[tuple[str, ...]] = ("B0", "O", "L1", "W7", "Q30", "TF", "AN", "PC")
"""The arms of the month-level control: B0, the oracle, L1, and the arms the review named."""

STEP_CONTROL_ARMS: Final[tuple[str, ...]] = ("B0", "O", "L1", "W7", "Q30", "TF", "AN")
"""The arms of the plant-specific control."""

LONG_LEADS: Final[tuple[int, ...]] = (7, 10, 14)
"""The lead-days of the long-lead follow-up."""

LONG_LEAD_FRAME_ARMS: Final[tuple[str, ...]] = ("B0", "W7", "Q30", "N2-wide")
"""The arms whose columns a long-lead frame holds, besides the climatology columns."""

PLANT_STEP_SEED: Final[int] = 1139
"""Seeds each plant's steps; the generator for plant `i` (in sorted label order) is seeded with
this and `i`."""

STEPS_PER_PLANT: Final[int] = 8
"""How many steps each plant gets over its record."""

STEP_WEEKS: Final[tuple[int, int]] = (4, 12)
"""The shortest and longest step, in whole weeks."""

STEP_RANGE_START: Final[datetime] = datetime(2024, 9, 1, tzinfo=UTC)
"""Steps start from this date, three months before the first scored month, so that the scored
period is covered and the windows have history before it."""

LEAK_PROBED: Final[frozenset[str]] = frozenset({"control_month_s10", "control_step_s10", "lead14"})
"""The frames the leak probe runs on, besides PC2's."""

LEAK_PROBE_PC2_CAMS_DAYS: Final[int] = 2
"""PC2's probe cuts CAMS at whole days through the issue day minus this many."""


def followup_dir(*, root: Path) -> Path:
    """Return the folder every follow-up output lives in.

    Args:
        root: The output root.

    Returns:
        `<root>/ens_mean/followups`.
    """
    return root / PRODUCT / "followups"


REQUIRED_ARM: Final[str] = "L1"
"""Every frame also builds L1's lag, so the row set is the first run's: the shared rows minus those
where B0's columns or L1's strict lag are null."""


def frame_names() -> dict[str, tuple[str, ...]]:
    """Return every follow-up frame's name and the arms it serves.

    Returns:
        Frame name to arms, in build order.
    """
    names: dict[str, tuple[str, ...]] = {}
    for shift in CONTROL_SHIFTS:
        names[f"control_month_s{round(shift * 100):02d}"] = MONTH_CONTROL_ARMS
    for shift in CONTROL_SHIFTS:
        names[f"control_step_s{round(shift * 100):02d}"] = STEP_CONTROL_ARMS
    for lead in LONG_LEADS:
        names[f"lead{lead}"] = LONG_LEAD_FRAME_ARMS
    names["lead1_pc2"] = ("B0", "PC2")
    return names


def frame_path(*, root: Path, name: str) -> Path:
    """Return a follow-up frame's file.

    Args:
        root: The output root.
        name: The frame's name from `frame_names`.

    Returns:
        The parquet path, with the product in the file name.
    """
    return followup_dir(root=root) / "frames" / f"{name}_{PRODUCT}.parquet"


def month_factor(*, shift: float) -> pl.Expr:
    """Return the month-level control's shift factor as an expression of `time`.

    Args:
        shift: The fraction of power lost in a shifted month.

    Returns:
        One minus `shift` in a month of `shifted_months()`, otherwise 1. A solar hour is labelled
        by its end, so its month is its midpoint's.
    """
    month = (pl.col("time") - pl.duration(minutes=30)).dt.strftime("%Y-%m")
    return pl.when(month.is_in(sorted(shifted_months()))).then(1.0 - shift).otherwise(1.0)


def plant_steps(*, hourly: pl.DataFrame) -> pl.DataFrame:
    """Draw each plant's persistent steps, seeded.

    A step starts at a uniformly random hour between `STEP_RANGE_START` (or the plant's first
    reading, if later) and its last reading, and lasts a
    uniformly random whole number of weeks from `STEP_WEEKS[0]` to `STEP_WEEKS[1]`, so a plant has
    history before the scored period for its windows to read.

    Args:
        hourly: The lag source, with `site` and `time`.

    Returns:
        One row per step, with `site`, `start` and `end`.
    """
    spans = hourly.group_by("site").agg(first=pl.col("time").min(), last=pl.col("time").max())
    records = []
    for index, row in enumerate(spans.sort("site").iter_rows(named=True)):
        rng = np.random.default_rng([PLANT_STEP_SEED, index])
        row["first"] = max(row["first"], STEP_RANGE_START)
        total_hours = int((row["last"] - row["first"]).total_seconds() // 3600)
        starts = rng.integers(0, total_hours, size=STEPS_PER_PLANT)
        weeks = rng.integers(STEP_WEEKS[0], STEP_WEEKS[1] + 1, size=STEPS_PER_PLANT)
        records += [
            {
                "site": row["site"],
                "start": row["first"] + np.timedelta64(int(start), "h"),
                "end": row["first"] + np.timedelta64(int(start) + 7 * 24 * int(week), "h"),
            }
            for start, week in zip(starts, weeks, strict=True)
        ]
    return pl.DataFrame(records).with_columns(
        pl.col("start").dt.replace_time_zone("UTC"), pl.col("end").dt.replace_time_zone("UTC")
    )


def step_factor(*, steps: pl.DataFrame, shift: float) -> pl.Expr:
    """Return the plant-specific control's shift factor as an expression of `site` and `time`.

    Args:
        steps: `plant_steps`' result.
        shift: The fraction of power lost during a step; steps that overlap do not compound.

    Returns:
        One minus `shift` while any of the plant's steps covers the hour, otherwise 1.
    """
    covered = pl.lit(value=False)
    for row in steps.iter_rows(named=True):
        covered = covered | (
            (pl.col("site") == row["site"])
            & (pl.col("time") >= row["start"])
            & (pl.col("time") < row["end"])
        )
    return pl.when(covered).then(1.0 - shift).otherwise(1.0)


def add_climatology_columns(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `climatology_fold<k>`, the out-of-fold median power of each plant's month and hour.

    For scored fold `k`, a row of fold `k` reads the median over the plant's other folds, which is
    `studies.baselines.climatology`. A row of another fold `j` reads the median over the folds
    outside both `k` and `j`, so the training rows' column holds no information from the scored
    fold.

    Args:
        frame: A frame with `site`, `time`, `fold`, `constrained` and `power_mw`.

    Returns:
        The frame, sorted by site and time, with the five columns.
    """
    ordered = frame.sort("site", "time").with_columns(
        _whole=climatology(frame=frame.sort("site", "time"))
    )
    for fold in range(N_FOLDS):
        rest = ordered.filter(pl.col("fold") != fold)
        part = rest.select("site", "time").with_columns(_part=climatology(frame=rest))
        ordered = ordered.join(part, on=["site", "time"], how="left", maintain_order="left")
        ordered = ordered.with_columns(
            **{
                f"climatology_fold{fold}": pl.when(pl.col("fold") == fold)
                .then(pl.col("_whole"))
                .otherwise(pl.col("_part"))
            }
        ).drop("_part")
    return ordered.drop("_whole")


def shifted_share_lines(*, name: str, frame: pl.DataFrame) -> list[str]:
    """Report each plant's realised share of scored rows and months that a control shifted.

    Args:
        name: The frame's name.
        frame: A control frame with `oracle_factor`, `site` and `month`.

    Returns:
        Report lines: a table with one row per plant.
    """
    per_plant = (
        frame.with_columns(shifted=pl.col("oracle_factor") < 1.0)
        .group_by("site", "month")
        .agg(share=pl.col("shifted").mean())
        .group_by("site")
        .agg(
            months=pl.len(),
            months_mostly_shifted=(pl.col("share") >= 0.5).sum(),
            mean_row_share=pl.col("share").mean(),
        )
        .sort("site")
    )
    lines = [
        f"Realised shifted share of the scored months, `{name}`:",
        "",
        (
            "| Plant | Scored months | Months with half or more of the rows shifted "
            "| Mean share of rows shifted |"
        ),
        "|---|---|---|---|",
    ]
    lines += [
        f"| {row['site']} | {row['months']} | {row['months_mostly_shifted']} | "
        f"{row['mean_row_share']:.3f} |"
        for row in per_plant.iter_rows(named=True)
    ]
    return [*lines, ""]


def _scaled(*, hourly: pl.DataFrame, factor: pl.Expr) -> pl.DataFrame:
    """Return the hourly power multiplied by a shift factor.

    Args:
        hourly: The lag source.
        factor: An expression of `site` and `time`.

    Returns:
        The lag source with `power_mw` scaled.
    """
    return hourly.with_columns(power_mw=pl.col("power_mw") * factor).sort("site", "time")


def _build_control(
    *,
    name: str,
    arms: tuple[str, ...],
    factor: pl.Expr,
    shared: pl.DataFrame,
    hourly: pl.DataFrame,
    raw: pl.DataFrame,
    weather_lead0: pl.DataFrame,
    probe: bool,
) -> tuple[pl.DataFrame, list[str]]:
    """Build one control frame and its report lines.

    Args:
        name: The frame's name.
        arms: The arms it serves.
        factor: The shift factor, an expression of `site` and `time`.
        shared: The shared rows.
        hourly: The unshifted lag source.
        raw: The unshifted raw hourly power, for the lag assertions.
        weather_lead0: The ENS mean's lead-day 0 weather.
        probe: Whether to run the leak probe on this frame.

    Returns:
        The frame and the report lines.
    """
    scaled = _scaled(hourly=hourly, factor=factor)
    inputs = lag_inputs(hourly=scaled)
    frame, lines = build_frame(
        product=PRODUCT,
        lead_day=FULL_SWEEP_LEAD_DAY,
        arms=arms,
        shared=shared,
        inputs=inputs,
        weather_lead0=weather_lead0,
        factor=factor,
    )
    reference_free = frame.drop(
        [c for c in frame.columns if c.startswith(("persistence_day", "diurnal_persistence_day"))]
    )
    lines += assert_lag_arithmetic(
        frame=reference_free, lead_day=FULL_SWEEP_LEAD_DAY, raw=_scaled(hourly=raw, factor=factor)
    )
    if probe:
        lines += leak_probe(
            frame=frame,
            inputs=inputs,
            weather=weather_table(product=PRODUCT, lead_day=FULL_SWEEP_LEAD_DAY),
            weather_lead0=weather_lead0,
            arms=arms,
            lead_day=FULL_SWEEP_LEAD_DAY,
        )
    lines += shifted_share_lines(name=name, frame=frame)
    return frame, lines


def main() -> int:
    """Build every follow-up frame and write the build report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=LAG_FEATURES_DIR)
    arguments = parser.parse_args()
    root: Path = arguments.output_root
    names = frame_names()
    report_path = followup_dir(root=root) / f"build_report_followups_{PRODUCT}.md"
    refuse_to_overwrite(paths=[report_path, *(frame_path(root=root, name=n) for n in names)])

    shared = shared_rows()
    hourly = lag_source_hourly()
    raw = raw_hourly_power()
    weather_lead0 = weather_table(product=PRODUCT, lead_day=0)
    steps = plant_steps(hourly=hourly)
    report = [
        "# Follow-up frames (post hoc)",
        "",
        "All frames are post hoc, built after the first run's results.",
        "",
        (
            f"The month-level control shifts {len(shifted_months() & set(scored_months()))} of the "
            f"{len(scored_months())} scored months. The plant-specific control gives each plant "
            f"{STEPS_PER_PLANT} steps of {STEP_WEEKS[0]} to {STEP_WEEKS[1]} weeks."
        ),
        "",
    ]
    for name, served in names.items():
        _LOG.info("building %s", name)
        arms = tuple(dict.fromkeys((REQUIRED_ARM, *served)))
        if name.startswith("control_month"):
            shift = int(name[-2:]) / 100
            frame, lines = _build_control(
                name=name,
                arms=arms,
                factor=month_factor(shift=shift),
                shared=shared,
                hourly=hourly,
                raw=raw,
                weather_lead0=weather_lead0,
                probe=name in LEAK_PROBED,
            )
        elif name.startswith("control_step"):
            shift = int(name[-2:]) / 100
            frame, lines = _build_control(
                name=name,
                arms=arms,
                factor=step_factor(steps=steps, shift=shift),
                shared=shared,
                hourly=hourly,
                raw=raw,
                weather_lead0=weather_lead0,
                probe=name in LEAK_PROBED,
            )
        else:
            frame, lines = _build_plain(
                name=name,
                arms=arms,
                shared=shared,
                hourly=hourly,
                raw=raw,
                weather_lead0=weather_lead0,
            )
        write_parquet_atomic(frame=frame, path=frame_path(root=root, name=name))
        report += [f"## {name}", "", f"Arms: {', '.join(arms)}.", "", *lines, ""]
        report += [*arms_columns_table(arms=arms), ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n")
    sys.stdout.write("\n".join(report) + "\n")
    return 0


def _build_plain(
    *,
    name: str,
    arms: tuple[str, ...],
    shared: pl.DataFrame,
    hourly: pl.DataFrame,
    raw: pl.DataFrame,
    weather_lead0: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    """Build a long-lead or PC2 frame, with the climatology columns for a long-lead frame.

    Args:
        name: The frame's name, `lead<N>` or `lead1_pc2`.
        arms: The arms it serves.
        shared: The shared rows.
        hourly: The lag source.
        raw: The raw hourly power, for the lag assertions.
        weather_lead0: The ENS mean's lead-day 0 weather.

    Returns:
        The frame and the report lines.
    """
    is_pc2 = name == "lead1_pc2"
    lead_day = FULL_SWEEP_LEAD_DAY if is_pc2 else int(name.removeprefix("lead"))
    inputs = lag_inputs(hourly=hourly)
    frame, lines = build_frame(
        product=PRODUCT,
        lead_day=lead_day,
        arms=arms,
        shared=shared,
        inputs=inputs,
        weather_lead0=weather_lead0,
    )
    lines += assert_lag_arithmetic(frame=frame, lead_day=lead_day, raw=raw)
    if is_pc2 or name in LEAK_PROBED:
        lines += leak_probe(
            frame=frame,
            inputs=inputs,
            weather=weather_table(product=PRODUCT, lead_day=lead_day),
            weather_lead0=weather_lead0,
            arms=arms,
            lead_day=lead_day,
            **({"cams_available_days": LEAK_PROBE_PC2_CAMS_DAYS} if is_pc2 else {}),
        )
    if not is_pc2:
        frame = add_climatology_columns(frame=frame)
        lines.append(
            f"- `climatology_fold0` to `climatology_fold{N_FOLDS - 1}` added: the out-of-fold "
            "median of the plant's calendar month and hour, withholding the scored fold and the "
            "row's own fold."
        )
    return frame, lines


if __name__ == "__main__":
    sys.exit(main())
