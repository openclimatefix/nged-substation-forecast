"""Rescore RO-NC arm B on the 35 labelled real transfers over 1 day, one row per transfer.

Reads experiment 138's stored arm B tables (identical to experiment 130's), adds the first step
at each transfer's start in noise units and the number of metered receivers, and assigns each miss
one reason. Nothing is fitted.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00146-*/e00146_rescore.py
"""

from __future__ import annotations

import json
from typing import Final

import numpy as np
import polars as pl
from e00146_common import LABELLED_PATH, ML_EXPERIMENTS_DATA_DIR, WORK_DIR
from e00146_variants import load_both, scales_for, shifts

RUN_138: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00138-weaker-onset-detector-on-the-ro-nc-convex-residual" / "run"
)
ARM: Final[str] = "B"
ONSET_TOLERANCE: Final[int] = 4
"""Stamps (2 hours) within which an alarm counts as at the labelled start."""
N_RESAMPLES: Final[int] = 20_000
SEED: Final[int] = 146
WIDE_STAMPS: Final[int] = 96
"""Stamps (48 hours) within which the nearest alarm of any size is reported for a miss."""


def wilson(*, k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval for `k` of `n`."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def first_steps(*, results: pl.DataFrame, panel: dict, offset_panel: dict) -> dict[str, float]:
    """Largest |24-hour step| in noise units within 2 hours of each transfer's start (0 if none)."""
    residual, offset = panel["residual"], offset_panel["residual"]
    scales = scales_for(variant="X", residual=residual, offset=offset)
    shift = shifts(variant="X", residual=residual, offset=offset)
    out: dict[str, float] = {}
    for row in results.iter_rows(named=True):
        best = float("nan")
        for series in (int(s) for s in row["series"].split(",")):
            if series not in shift or not np.isfinite(scales.get(series, np.nan)):
                continue
            window = shift[series][
                row["onset"] - ONSET_TOLERANCE : row["onset"] + ONSET_TOLERANCE + 1
            ]
            if np.isfinite(window).any():
                value = float(np.nanmax(np.abs(window)) / scales[series])
                best = value if np.isnan(best) else max(best, value)
        out[row["card_id"]] = best
    return out


def metered_receivers(*, power: dict) -> dict[str, int]:
    """The number of each labelled transfer's receivers that have power readings."""
    labelled = pl.read_parquet(LABELLED_PATH).select("card_id", "receivers")
    return {
        row["card_id"]: len({int(r["series"]) for r in json.loads(row["receivers"])} & set(power))
        for row in labelled.iter_rows(named=True)
    }


def reason(*, found: bool, onset_found: bool, step: float) -> str:
    """The first reason that holds for a transfer, in the order of the plan's decision rule."""
    if found:
        return "found"
    if np.isnan(step):
        return "no residual at the start"
    if onset_found:
        return "alarm at the start, end too far off"
    if step < 2.0:
        return "first step under 2 noise units"
    if step < 4.0:
        return "first step of 2 to 4 noise units"
    return "first step of 4 noise units or more, no alarm"


def main() -> None:
    """Print the table and the summary numbers, and write the per-transfer table."""
    panel, offset_panel = load_both()
    results = pl.read_parquet(RUN_138 / "real_results.parquet").filter(
        pl.col("setting") == ARM, pl.col("hours") > 24
    )
    alarms = pl.read_parquet(RUN_138 / "alarms.parquet").filter(pl.col("setting") == ARM)
    steps = first_steps(results=results, panel=panel, offset_panel=offset_panel)
    metered = metered_receivers(power=panel["power"])
    rows = []
    for row in results.iter_rows(named=True):
        series = {int(s) for s in row["series"].split(",")}
        near = alarms.filter(
            pl.col("series").is_in(list(series)),
            (pl.col("stamp") - row["onset"]).abs() <= ONSET_TOLERANCE,
        )
        timing = (
            float(
                near["stamp"].to_numpy()[np.argmin(np.abs(near["stamp"].to_numpy() - row["onset"]))]
                - row["onset"]
            )
            / 2
            if near.height
            else float("nan")
        )
        wide = alarms.filter(
            pl.col("series").is_in(list(series)),
            (pl.col("stamp") - row["onset"]).abs() <= WIDE_STAMPS,
        )
        offsets = (wide["stamp"].to_numpy() - row["onset"]) / 2
        nearest = float(offsets[np.argmin(np.abs(offsets))]) if wide.height else float("nan")
        step = steps[row["card_id"]]
        rows.append(
            {
                "card_id": row["card_id"],
                "hours": row["hours"],
                "moved_mw": row["moved_mw_p50"],
                "metered_receivers": metered[row["card_id"]],
                "first_step_z": step,
                "found": row["found"],
                "onset_found": row["onset_found"],
                "timing_error_hours": timing,
                "nearest_alarm_hours": nearest,
                "end_error_hours": row["end_error"] / 2,
                "reason": reason(found=row["found"], onset_found=row["onset_found"], step=step),
            }
        )
    table = pl.DataFrame(rows).sort("moved_mw", descending=True, nulls_last=True)
    table.write_parquet(WORK_DIR / "per_transfer.parquet")
    with pl.Config(tbl_rows=40, tbl_cols=12, fmt_str_lengths=60, tbl_width_chars=250):
        print(table.drop("onset_found"))
    n, k = table.height, int(table["found"].sum())
    low, high = wilson(k=k, n=n)
    print(f"\nfound {k} of {n} ({k / n:.2f}; 95% Wilson {low:.2f} to {high:.2f})")
    weighted = table.drop_nulls("moved_mw")
    weight, hit = weighted["moved_mw"].to_numpy(), weighted["found"].to_numpy().astype(float)
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(weight), size=(N_RESAMPLES, len(weight)))
    boot = (weight[picks] * hit[picks]).sum(axis=1) / weight[picks].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(
        f"MW-weighted recall {np.sum(weight * hit) / weight.sum():.3f} ({lo:.3f} to {hi:.3f}); "
        f"{len(weight)} of {n} transfers have a labelled moved MW, total {weight.sum():.1f} MW"
    )
    print("\nReasons (count, moved MW):")
    for r in (
        table.group_by("reason")
        .agg(n=pl.len(), mw=pl.col("moved_mw").sum())
        .sort("n", descending=True)
        .iter_rows()
    ):
        print(f"  {r[0]}: {r[1]}, {r[2]:.1f} MW")
    f = table.filter(pl.col("found"))
    print(
        f"\nTiming error of the alarm against the labelled start, found transfers (hours): "
        f"median {f['timing_error_hours'].median():+.2f}, "
        f"min {f['timing_error_hours'].min():+.2f}, "
        f"max {f['timing_error_hours'].max():+.2f}"
    )
    print(
        f"End error, found transfers (hours): median {f['end_error_hours'].median():.2f}, "
        f"max {f['end_error_hours'].max():.2f}"
    )
    print("\nFound by metering:")
    for flag, label in ((True, "metered receiver"), (False, "no metered receiver")):
        part = table.filter((pl.col("metered_receivers") > 0) == flag)
        print(f"  {label}: {int(part['found'].sum())} of {part.height}")


if __name__ == "__main__":
    main()
