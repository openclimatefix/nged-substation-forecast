"""Print every number the write-up of experiment 66 quotes, and the verdict the decision rule gives.

Reads the tables that `e00066_run.py` wrote: MW-weighted recall of D0M, PE, PL and PF for each
injection shape with month-resampled intervals (C1: PL minus D0M and C2: PF minus D0M, pooled over
S2, S3 and S4), recall by donor step in noise units and metered receiver, the labelled real
transfers (C3), and alarm counts.

    uv run python ml-experiments/00060-*/e00066_report.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00066_common import (
    ML_EXPERIMENTS_DATA_DIR,
    WORK_DIR,
    load_panel,
    stamp_months,
    stamp_year_months,
)
from e00066_detectors import prepare
from e00066_inject import HEADLINE_BAND
from e00066_paired import GRID, flat_prior, labelled_duration_prior, paired_alarms
from e00066_profile import NON_MATCHING, SHAPE_NAMES, SHAPES

EXPERIMENT_58_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00058-sw2-8b-donor-step-recut-and-stronger-veto"
)
N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 1 - 0.05 / 3
"""Two-sided level of the intervals: Bonferroni across the three contrasts C1 to C3 (98.33%)."""
SEED: Final[int] = 6000
SUPPORT_AT_LEAST: Final[float] = 0.02
ALARM_TOLERANCE: Final[float] = 1.10
STEP_EDGES: Final[tuple[float, ...]] = (0.0, 2.0, 4.0, 6.0, 10.0, float("inf"))
DETECTORS: Final[tuple[str, ...]] = ("D0M_48", "PE_48", "PL_48", "PF_48")
ARMS: Final[dict[str, str]] = {"D0M_48": "f0", "PE_48": "fe", "PL_48": "fl", "PF_48": "ff"}
NAMES: Final[dict[str, str]] = {"f0": "D0M", "fe": "PE", "fl": "PL", "ff": "PF"}


def population(*, out: Path) -> pl.DataFrame:
    """The 620 headline-band even-month transfers, a row per transfer, shape, and detector.

    The weight of a transfer is its mean moved MW under S1.
    """
    events = pl.read_parquet(EXPERIMENT_58_DIR / "events.parquet")
    onset = events["onset"].to_numpy()
    events = events.with_columns(
        month=pl.Series(stamp_months()[onset]),
        year_month=pl.Series(stamp_year_months()[onset]),
        headline=(pl.col("size_band") == HEADLINE_BAND[0])
        & (pl.col("duration_band") == HEADLINE_BAND[1]),
    ).filter(pl.col("headline"), pl.col("month") % 2 == 0, pl.col("set") != "holiday")
    results = pl.read_parquet(out / "event_results.parquet")
    weight = (
        results.filter(pl.col("shape") == "S1", pl.col("setting") == "D0M_48")
        .select("event_id", weight=pl.col("moved_mw"))
        .unique("event_id")
    )
    return (
        results.join(events, on="event_id", validate="m:1")
        .join(weight, on="event_id", validate="m:1")
        .sort("shape", "setting", "event_id")
    )


def wide(*, data: pl.DataFrame, shape: str) -> pl.DataFrame:
    """One row per transfer with every arm's `found` for one shape."""
    one = data.filter(pl.col("shape") == shape)
    frame = one.filter(pl.col("setting") == "D0M_48").select(
        "event_id", "weight", "year_month", "metered_receivers", "donor_step_z", f0="found"
    )
    for setting, column in ARMS.items():
        if setting != "D0M_48":
            frame = frame.join(
                one.filter(pl.col("setting") == setting).select("event_id", **{column: "found"}),
                on="event_id",
                validate="1:1",
            )
    return frame.sort("event_id")


def interval(*, resampled: np.ndarray) -> tuple[float, float]:
    """The two-sided interval of resampled statistics at `LEVEL`."""
    tail = 100 * (1 - LEVEL) / 2
    low, high = np.percentile(resampled, [tail, 100 - tail])
    return float(low), float(high)


def month_sums(*, frame: pl.DataFrame, found: str) -> tuple[np.ndarray, np.ndarray]:
    """Per calendar month: the weight-times-found sum and the weight sum."""
    grouped = (
        frame.with_columns(wf=pl.col("weight") * pl.col(found).cast(pl.Float64))
        .group_by("year_month")
        .agg(pl.col("wf").sum(), pl.col("weight").sum())
        .sort("year_month")
    )
    return grouped["wf"].to_numpy(), grouped["weight"].to_numpy()


def diffs(
    *, frames: list[pl.DataFrame], arm: str, base: str, picks: np.ndarray | None
) -> np.ndarray:
    """MW-weighted recall of `arm` minus `base`, averaged over shapes; `picks` resamples months."""
    total = 0.0
    for frame in frames:
        wf_a, w = month_sums(frame=frame, found=arm)
        wf_b, _ = month_sums(frame=frame, found=base)
        if picks is None:
            total = total + (wf_a.sum() - wf_b.sum()) / w.sum()
        else:
            total = total + (wf_a[picks].sum(axis=1) - wf_b[picks].sum(axis=1)) / w[picks].sum(
                axis=1
            )
    return np.atleast_1d(total / len(frames))


def contrast(
    *, frames: list[pl.DataFrame], arm: str = "fl", base: str = "f0"
) -> tuple[float, float, float]:
    """Point value and interval of the shape-averaged MW-weighted recall difference."""
    months = np.unique(frames[0]["year_month"].to_numpy())
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(months), size=(N_RESAMPLES, len(months)))
    point = float(diffs(frames=frames, arm=arm, base=base, picks=None)[0])
    low, high = interval(resampled=diffs(frames=frames, arm=arm, base=base, picks=picks))
    return point, low, high


def recall_rows(*, frame: pl.DataFrame) -> dict[str, float]:
    """Weighted and unweighted recall of every arm for one set of transfers."""
    w = frame["weight"].to_numpy()
    out = {}
    for name in NAMES:
        found = frame[name].to_numpy().astype(float)
        out[f"{name}_mw"] = float((w * found).sum() / w.sum())
        out[f"{name}_n"] = float(found.mean())
    return out


def fmt(point: float, low: float, high: float) -> str:
    """A point value with its interval."""
    return f"{point:+.3f} ({low:+.3f} to {high:+.3f})"


def print_shapes(*, data: pl.DataFrame) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Per-shape table; returns each shape's differences from D0M and from PF, with intervals."""
    found: dict[str, dict[str, tuple[float, float, float]]] = {}
    print("\nMW-weighted recall (share of the moved MW in found transfers), even months")
    for shape in SHAPES:
        frame = wide(data=data, shape=shape)
        rec = recall_rows(frame=frame)
        found[shape] = {
            "PL-D0M": contrast(frames=[frame], arm="fl", base="f0"),
            "PF-D0M": contrast(frames=[frame], arm="ff", base="f0"),
            "PE-D0M": contrast(frames=[frame], arm="fe", base="f0"),
            "PL-PF": contrast(frames=[frame], arm="fl", base="ff"),
        }
        recalls = " ".join(f"{NAMES[n]} {rec[f'{n}_mw']:.3f}" for n in NAMES)
        unweighted = " ".join(f"{NAMES[n]} {rec[f'{n}_n']:.3f}" for n in NAMES)
        print(f"{shape} {SHAPE_NAMES[shape]:22s} n={frame.height} {recalls}")
        print("   missed MW: " + " ".join(f"{NAMES[n]} {1 - rec[f'{n}_mw']:.3f}" for n in NAMES))
        print(f"   unweighted: {unweighted}")
        print("   " + "; ".join(f"{k} {fmt(*v)}" for k, v in found[shape].items()))
        metered = frame.filter(pl.col("metered_receivers") > 0)
        rec = recall_rows(frame=metered)
        print(
            f"   metered receiver only n={metered.height} "
            + " ".join(f"{NAMES[n]} {rec[f'{n}_mw']:.3f}" for n in NAMES)
        )
    return found


def print_by_step(*, data: pl.DataFrame) -> None:
    """Standard table: recall by donor step in noise units, split by metered receiver."""
    print("\nRecall by donor step in noise units (count found / transfers), D0M | PE | PL | PF")
    for shape in SHAPES:
        frame = wide(data=data, shape=shape)
        for flag, label in ((True, "metered"), (False, "unmetered")):
            part = frame.filter((pl.col("metered_receivers") > 0) == flag)
            cells = []
            for low, high in itertools.pairwise(STEP_EDGES):
                bin_ = part.filter(pl.col("donor_step_z").abs().is_between(low, high, "left"))
                counts = "|".join(str(bin_[c].sum()) for c in NAMES)
                cells.append(f"[{low:g},{high:g}) {counts}/{bin_.height}")
            print(f"{shape} {label:9s} " + "; ".join(cells))


def print_big_steps(*, data: pl.DataFrame) -> None:
    """Among transfers with a donor step of 6 noise units or more: found, wrong end, no start."""
    print("\nTransfers whose donor step is 6 noise units or more (post hoc)")
    for shape in SHAPES:
        for setting in DETECTORS:
            part = data.filter(
                pl.col("shape") == shape,
                pl.col("setting") == setting,
                pl.col("donor_step_z").abs() >= 6,
            )
            n, found = part.height, part["found"].sum()
            wrong_end = part.filter(pl.col("onset_found") & ~pl.col("found")).height
            print(
                f"{shape} {setting}: {n} transfers, found {found}, onset found but end wrong "
                f"{wrong_end}, no alarm at the onset {n - found - wrong_end}"
            )


def print_real(*, out: Path) -> None:
    """C3: the labelled real transfers of more than 1 day, found by each arm."""
    real = pl.read_parquet(out / "real_results.parquet").filter(pl.col("hours") > 24)
    n = real.filter(pl.col("setting") == "D0M_48").height
    print(f"\nC3: labelled real transfers of more than 1 day: {n}")
    for setting in DETECTORS:
        part = real.filter(pl.col("setting") == setting)
        w = part["moved_mw_p50"].fill_null(0).to_numpy()
        f = part["found"].to_numpy().astype(float)
        print(
            f"  {setting}: found {int(f.sum())}, onset found {part['onset_found'].sum()}, "
            f"MW-weighted recall {(w * f).sum() / w.sum():.3f}"
        )
        for vis in ("conserved", "one_sided"):
            sub = part.filter(pl.col("visibility") == vis)
            print(f"      {vis}: {sub.height}, found {sub['found'].sum()}")
    base = real.filter(pl.col("setting") == "D0M_48").sort("card_id")
    pl_ = real.filter(pl.col("setting") == "PL_48").sort("card_id")
    print(
        f"  only D0M {base.filter(pl.col('found') & ~pl_['found']).height}, "
        f"only PL {pl_.filter(pl.col('found') & ~base['found']).height}"
    )


def print_alarms(*, out: Path) -> dict[str, tuple[int, int]]:
    """Unexplained alarm counts by even and odd month, with the composition counts."""
    alarms = pl.read_parquet(out / "alarms.parquet")
    thresholds = json.loads((out / "thresholds.json").read_text())
    panel = load_panel()
    plain = prepare(residuals=panel["residual"], window=48)
    candidates = sum(len(c) for c in paired_alarms(inputs=plain, prior=flat_prior()).values())
    print(f"\nthresholds {thresholds}; candidate onsets at or above 2.5 noise units: {candidates}")
    counts = {}
    for setting in DETECTORS:
        part = alarms.filter(pl.col("setting") == setting)
        unexplained = part.filter(~pl.col("logged_matched"))
        counts[setting] = (
            unexplained.filter(pl.col("month") % 2 == 0).height,
            unexplained.filter(pl.col("month") % 2 == 1).height,
        )
        print(
            f"{setting}: unexplained alarms even months {counts[setting][0]}, "
            f"odd {counts[setting][1]}; "
            f"all alarms {part.height}, near a logged closure {part['logged_matched'].sum()}"
        )
    return counts


def verdict(
    *,
    shapes: dict[str, dict[str, tuple[float, float, float]]],
    pooled: dict[str, tuple[float, float, float]],
    alarms: dict[str, tuple[int, int]],
) -> str:
    """Apply the pre-registered decision rule."""
    point, low, high = pooled["PL-D0M"]
    if high < SUPPORT_AT_LEAST:
        return "refuted"
    base = alarms["D0M_48"][0]
    quiet = all(alarms[s][0] <= ALARM_TOLERANCE * base for s in ("PL_48", "PF_48"))
    if low > 0 and point >= SUPPORT_AT_LEAST and quiet:
        return "supported" if pooled["PF-D0M"][1] > 0 else "partly supported"
    return "inconclusive"


def main() -> None:
    """Print the tables and the verdict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    out = parser.parse_args().out
    data = population(out=out)
    one = data.filter(pl.col("shape") == "S1", pl.col("setting") == "D0M_48")
    print(f"{one.height} transfers; {one.filter(pl.col('weight') <= 0).height} move no MW")
    full = labelled_duration_prior()
    held = labelled_duration_prior(
        exclude_cards=frozenset(pl.read_parquet(out / "real_results.parquet")["card_id"].to_list())
    )
    for hours in (24, 48, 72, 168, 336):
        print(f"PL prior at {hours} hours: {full[int(hours * 2 - GRID[0])]:+.3f}")
    print(
        f"largest change of the prior without the scored transfers: {np.abs(full - held).max():.3f}"
    )
    shapes = print_shapes(data=data)
    frames = [wide(data=data, shape=s) for s in NON_MATCHING]
    pooled = {
        "PL-D0M": contrast(frames=frames, arm="fl", base="f0"),
        "PF-D0M": contrast(frames=frames, arm="ff", base="f0"),
        "PE-D0M": contrast(frames=frames, arm="fe", base="f0"),
        "PL-PF": contrast(frames=frames, arm="fl", base="ff"),
    }
    print(f"\nPooled over {NON_MATCHING}:")
    for name, value in pooled.items():
        print(f"  {name}: {fmt(*value)}")
    print_by_step(data=data)
    print_big_steps(data=data)
    print_real(out=out)
    alarms = print_alarms(out=out)
    print(f"\nVERDICT: {verdict(shapes=shapes, pooled=pooled, alarms=alarms)}")


if __name__ == "__main__":
    main()
