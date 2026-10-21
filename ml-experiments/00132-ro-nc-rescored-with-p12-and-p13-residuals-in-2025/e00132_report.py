"""Print every number of the experiment 132 write-up.

Reads the tables that `e00132_restore.py`, `e00132_fit.py`, `e00132_real.py`, and `e00132_run.py`
wrote, and experiment 101's `event_table.parquet`. Real-transfer outcomes are compared between the
panel of experiment 101 (rerun here, which must reproduce its outcomes), the new panel with each of
the three blocked-fit variants, and the injected draws of the old and the new panel.

    uv run --no-sync python ml-experiments/00132-*/e00132_report.py
"""

from __future__ import annotations

import itertools
from datetime import timedelta
from typing import Final

import numpy as np
import polars as pl
from e00132_common import GRID_START, WORK_DIR, fits_folder, load_panel
from e00132_fit import FITS, SERIES, VARIANTS
from e00132_real import SUFFIX
from e00132_run import all_scales
from e00132_variants import load_offset_panel
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EVENT_TABLE_101: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00101-ro-nc-detector-on-real-hand-filtered-switching-events"
    / "event_table.parquet"
)
FEATURES: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00006-stage1a-demand-input-feature-sweep"
    / "inputs"
    / "features.parquet"
)
ARM_TABLES: Final[dict[str, str]] = {
    "old panel": "_old_panel",
    "half_life": "",
    "equal": "_equal",
    "masked": "_masked",
}
"""The suffix of `event_arms*.parquet` for each scored panel and blocked-fit variant."""
ARMS: Final[tuple[str, ...]] = ("D0", "PF14", "X", "LOGIT_SAME", "RO-NC")
STEP_EDGES: Final[tuple[float, ...]] = (0.0, 2.0, 4.0, 6.0, 10.0, float("inf"))
N_RESAMPLES: Final[int] = 20_000
SEED: Final[int] = 132
BLOCKED: Final[frozenset[int]] = frozenset({12, 13})


def wilson(*, k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a binomial share."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def weighted_recall(*, found: np.ndarray, weight: np.ndarray) -> str:
    """MW-weighted recall with a 95% interval from resampling whole events."""
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(found), size=(N_RESAMPLES, len(found)))
    draws = (found[picks] * weight[picks]).sum(axis=1) / weight[picks].sum(axis=1)
    point = (found * weight).sum() / weight.sum()
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return f"{point:.3f} ({low:.3f} to {high:.3f})"


def count_text(*, k: int, n: int) -> str:
    """`k of n (share; Wilson interval)`."""
    low, high = wilson(k=k, n=n)
    return f"{k} of {n} ({k / n:.2f}; {low:.2f} to {high:.2f})"


def real_table() -> pl.DataFrame:
    """The labelled transfers (experiment 101's labels) with each scored panel's outcome."""
    base = pl.read_parquet(EVENT_TABLE_101).select(
        "card_id",
        "event",
        "onset",
        "end",
        "hours",
        "series",
        "share",
        "moved_mw_p50",
        "substations",
    )
    for name, suffix in ARM_TABLES.items():
        arms = pl.read_parquet(WORK_DIR / f"event_arms{suffix}.parquet")
        wide = arms.pivot(on="arm", index="card_id", values="found", separator="|")
        wide = wide.rename({arm: f"{arm}|{name}" for arm in wide.columns if arm != "card_id"})
        base = base.join(wide, on="card_id", validate="1:1")
    peak = pl.read_parquet(WORK_DIR / "event_arms.parquet").filter(pl.col("arm") == "PF14")
    old_peak = pl.read_parquet(WORK_DIR / "event_arms_old_panel.parquet").filter(
        pl.col("arm") == "PF14"
    )
    return (
        base.join(peak.select("card_id", step_z=pl.col("peak_step_z")), on="card_id")
        .join(old_peak.select("card_id", old_step_z=pl.col("peak_step_z")), on="card_id")
        .sort("onset")
    )


def section_reproduction(*, table: pl.DataFrame) -> None:
    """Experiment 101's panel, rerun here, must give experiment 101's outcomes on every event."""
    old = pl.read_parquet(EVENT_TABLE_101)
    for arm in ARMS:
        same = table.join(old.select("card_id", old=f"found|{arm}"), on="card_id")
        mismatches = same.filter(pl.col(f"{arm}|old panel") != pl.col("old")).height
        print(
            f"reproduction, {arm}: {mismatches} of {same.height} events differ from experiment 101"
        )


def section_real(*, table: pl.DataFrame) -> None:
    """Every arm on the 35 transfers over 1 day, split into the 6 at P12 and P13 and the rest."""
    long = table.filter(pl.col("hours") > 24)
    at_pair = long.filter(pl.col("series") == "12,13")
    elsewhere = long.filter(pl.col("series") != "12,13")
    print(f"\n== {long.height} transfers over 1 day: {at_pair.height} at P12 and P13")
    weight = long["moved_mw_p50"].fill_null(0).to_numpy()
    for name in ARM_TABLES:
        print(f"-- {name}")
        for arm in ARMS:
            column = f"{arm}|{name}"
            print(
                f"  {arm:10s} all {count_text(k=int(long[column].sum()), n=long.height)}; "
                f"at P12/P13 {int(at_pair[column].sum())} of {at_pair.height}; "
                f"other {count_text(k=int(elsewhere[column].sum()), n=elsewhere.height)}"
            )
        found = long[f"RO-NC|{name}"].to_numpy().astype(float)
        print(f"  RO-NC MW-weighted recall {weighted_recall(found=found, weight=weight)}")
    print("\nRO-NC outcome per transfer, old panel -> new panel (half_life, equal, masked):")
    for row in long.iter_rows(named=True):
        outcomes = [row[f"RO-NC|{n}"] for n in ARM_TABLES]
        flag = "  CHANGED" if outcomes[0] != outcomes[1] else ""
        step = "" if row["step_z"] is None else f"{row['step_z']:.1f}"
        old_step = "" if row["old_step_z"] is None else f"{row['old_step_z']:.1f}"
        print(
            f"  {row['event']} {row['substations']:10s} step old {old_step:>5s} new {step:>5s} "
            f"{outcomes}{flag}"
        )
    for name in ("half_life", "equal", "masked"):
        lost = long.filter(pl.col("RO-NC|old panel") & ~pl.col(f"RO-NC|{name}")).height
        gained = long.filter(~pl.col("RO-NC|old panel") & pl.col(f"RO-NC|{name}")).height
        print(f"{name}: {gained} transfers newly found, {lost} no longer found")


def section_alarms() -> None:
    """RO-NC and the other arms' alarm counts over the whole history, old panel and new."""
    for name, suffix in (("old panel", "_old_panel"), ("new panel", "")):
        counts = pl.read_parquet(WORK_DIR / f"alarm_counts{suffix}.parquet")
        print(f"\n== alarm counts, {name}\n{counts}")


def section_noise() -> None:
    """Residual noise at P12 and P13 beside the other Primaries, in MW and as a capacity share."""
    panel = load_panel()
    old_panel = load_panel(blocked=False)
    capacity = (
        pl.scan_parquet(FEATURES)
        .group_by("time_series_id")
        .agg(pl.col("effective_capacity_mw").median())
        .collect()
    )
    offset = load_offset_panel(panel=panel)
    scales = all_scales(panel=panel, offset_panel=offset)
    print(
        "\n== residual noise by series (new panel): MAD of the residual, MW and share of capacity"
    )
    rows = []
    for series, residual in sorted(panel["residual"].items()):
        cap = capacity.filter(pl.col("time_series_id") == series)["effective_capacity_mw"].item()
        values = residual[np.isfinite(residual)]
        mad = float(np.median(np.abs(values - np.median(values))))
        rows.append((series, mad, mad / cap, scales["PF14"][series], scales["PF14"][series] / cap))
        print(
            f"  P{series:02d} n {values.size:6d} MAD {mad:.3f} MW = {mad / cap:.4f} of capacity; "
            f"detector scale {scales['PF14'][series]:.3f} MW = {scales['PF14'][series] / cap:.4f}"
        )
    others = [r for r in rows if r[0] not in BLOCKED]
    print(
        f"  median over the {len(others)} other series: "
        f"MAD {np.median([r[1] for r in others]):.3f} MW "
        f"= {np.median([r[2] for r in others]):.4f} of capacity; "
        f"detector scale {np.median([r[3] for r in others]):.3f} MW "
        f"= {np.median([r[4] for r in others]):.4f}"
    )
    print("  P12 and P13 MAD by month (MW), blocked fits and stored panel together:")
    months = pl.Series(
        [GRID_START + timedelta(minutes=30 * i) for i in range(len(panel["residual"][12]))]
    ).dt.strftime("%Y-%m")
    for series in sorted(BLOCKED):
        frame = pl.DataFrame({"month": months, "residual": panel["residual"][series]}).drop_nans(
            "residual"
        )
        by_month = frame.group_by("month").agg(
            n=pl.len(), mad=(pl.col("residual") - pl.col("residual").median()).abs().median()
        )
        print(
            f"  P{series}: "
            + ", ".join(
                f"{m} {v:.2f}" for m, v in by_month.sort("month").select("month", "mad").iter_rows()
            )
        )
    added = {
        s: int(
            np.isfinite(panel["residual"][s]).sum() - np.isfinite(old_panel["residual"][s]).sum()
        )
        for s in BLOCKED
    }
    print(f"  residual half-hours added to the panel: {added}")


def section_leak() -> None:
    """Training rows inside a held-out block or its purge, and the block of each transfer."""
    print("\n== leakage check: training rows inside a held-out block or its purge")
    for variant in VARIANTS:
        for name in FITS:
            for series in SERIES:
                folder = fits_folder(name=name, variant=variant)
                frame = pl.read_parquet(WORK_DIR / folder / f"{series}.parquet")
                print(
                    f"  {variant:9s} {name:6s} P{series}: "
                    f"blocks {frame['block_start'].n_unique()}, "
                    f"leaked rows {frame['train_rows_in_block_or_purge'].sum()}, "
                    f"training rows per block {frame['n_train'].min()} to {frame['n_train'].max()}"
                )
    table = real_table().filter(pl.col("series") == "12,13", pl.col("hours") > 24)
    print("  block of each transfer's onset and end, half_life variant (main; offset):")
    for row in table.iter_rows(named=True):
        onset = GRID_START + timedelta(minutes=30 * row["onset"])
        end = GRID_START + timedelta(minutes=30 * row["end"])
        parts = []
        for name in FITS:
            frame = pl.read_parquet(WORK_DIR / f"fits_{name}" / "12.parquet")
            blocks = (
                frame.filter(pl.col("valid_time").is_in([onset, end]))
                .select("block_start", "block_end")
                .unique()
                .sort("block_start")
            )
            parts.append("; ".join(f"[{a:%Y-%m-%d}, {b:%Y-%m-%d})" for a, b in blocks.iter_rows()))
        print(
            f"  {row['event']}: {onset:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}: "
            f"{parts[0]} | {parts[1]}"
        )


def injected_cells(*, folder: str) -> dict[tuple[bool, int], float]:
    """RO-NC recall on one draw's injected transfers by metered receiver and step band."""
    events = pl.read_parquet(WORK_DIR / folder / "events.parquet")
    results = pl.read_parquet(WORK_DIR / folder / "event_results.parquet").filter(
        pl.col("setting") == "XL", pl.col("shape").is_in(["S2", "S3", "S4"])
    )
    frame = results.join(events, on="event_id")
    out = {}
    for flag in (True, False):
        for i, (lo, hi) in enumerate(itertools.pairwise(STEP_EDGES)):
            cell = frame.filter(
                (pl.col("metered_receivers") > 0) == flag,
                pl.col("donor_step_z").abs().is_between(lo, hi, "left"),
            )
            out[flag, i] = float(cell["found"].to_numpy().mean()) if cell.height else float("nan")
    return out


def section_injected(*, table: pl.DataFrame) -> None:
    """Injected recall at P12 and P13, and the expected count of real transfers found."""
    for folder in ("score_old", "score_new"):
        events = pl.read_parquet(WORK_DIR / folder / "events.parquet")
        results = pl.read_parquet(WORK_DIR / folder / "event_results.parquet").filter(
            pl.col("setting") == "XL", pl.col("shape").is_in(["S2", "S3", "S4"])
        )
        frame = results.join(events, on="event_id")
        touches = (
            pl.col("donor").is_in(list(BLOCKED))
            | pl.col("receivers")
            .str.split(",")
            .list.eval(pl.element().cast(pl.Int64, strict=False).is_in(list(BLOCKED)))
            .list.any()
        )
        print(f"\n== injected transfers, {folder}: {events.height} drawn")
        for name, part in (
            ("all", frame),
            ("touching P12 or P13", frame.filter(touches)),
            ("touching neither", frame.filter(~touches)),
        ):
            n = part["event_id"].n_unique()
            if part.height == 0:
                print(f"  {name}: none")
                continue
            weight = part["moved_mw"].to_numpy()
            found = part["found"].to_numpy().astype(float)
            print(
                f"  {name}: {n} transfers x 3 shapes; found {found.mean():.3f}; "
                f"MW-weighted {weighted_recall(found=found, weight=weight)}"
            )
    cells = injected_cells(folder="score_new")
    print("\ninjected RO-NC recall by (metered receiver, step band), new draw:")
    for (flag, i), value in cells.items():
        print(f"  metered={flag} step [{STEP_EDGES[i]:g}, {STEP_EDGES[i + 1]:g}): {value:.3f}")
    print("an empty step band takes the recall of the nearest lower band that holds transfers")
    long = table.filter(pl.col("hours") > 24)
    total = 0.0
    for part_name, part in (
        ("at P12 and P13", long.filter(pl.col("series") == "12,13")),
        ("the other 29", long.filter(pl.col("series") != "12,13")),
    ):
        expected = 0.0
        for row in part.iter_rows(named=True):
            step = row["step_z"]
            if step is None or not np.isfinite(step):
                continue
            i = max(j for j in range(5) if abs(step) >= STEP_EDGES[j])
            while not np.isfinite(cells[row["share"] is not None, i]):
                i -= 1
            expected += cells[row["share"] is not None, i]
        total += expected
        print(
            f"expected found {part_name}: {expected:.1f}; "
            f"found {int(part['RO-NC|half_life'].sum())} of {part.height}"
        )
        if part.height == 6:
            for row in part.iter_rows(named=True):
                print(
                    f"    {row['event']}: step {row['step_z']:.1f}, "
                    f"metered receiver {row['share'] is not None}, "
                    f"found {row['RO-NC|half_life']}"
                )
    print(f"expected found over the 35: {total:.1f}")


def main() -> None:
    """Print all numbers."""
    print(f"suffix of this run's own tables: {SUFFIX!r}")
    table = real_table()
    section_reproduction(table=table)
    section_real(table=table)
    section_alarms()
    section_noise()
    section_leak()
    section_injected(table=table)


if __name__ == "__main__":
    main()
