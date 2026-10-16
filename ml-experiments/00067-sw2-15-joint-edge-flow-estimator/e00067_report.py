"""Print every number the write-up of experiment 67 quotes, and the verdict the decision rule gives.

Reads the tables that `e00067_run.py` wrote (EF) and experiment 65's stored tables (D0M and
LOGIT_SAME with the PE end rule, on the same transfers, seeds, and shapes).

    uv run python ml-experiments/00067-*/e00067_report.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00067_common import ML_EXPERIMENTS_DATA_DIR, WORK_DIR, stamp_months, stamp_year_months
from e00067_inject import HEADLINE_BAND
from e00067_profile import NON_MATCHING, SHAPE_NAMES

EXPERIMENT_58_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00058-sw2-8b-donor-step-recut-and-stronger-veto"
)
EXPERIMENT_65_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00065-sw2-14-combined-single-series-and-neighbour-score"
)
N_RESAMPLES: Final[int] = 20_000
N_CONTRASTS: Final[int] = 2
LEVEL: Final[float] = 1 - 0.05 / N_CONTRASTS
"""Two-sided level of the intervals: Bonferroni across the two planned contrasts (97.5%)."""
SEED: Final[int] = 6700
SUPPORT_AT_LEAST: Final[float] = 0.05
REFUTED_BELOW: Final[float] = 0.02
ALARM_MARGIN: Final[int] = 10
STEP_EDGES: Final[tuple[float, ...]] = (0.0, 2.5, 4.0, 6.0, 10.0, float("inf"))
ARMS: Final[dict[str, str]] = {"EF": "EF_PE", "D0M": "D0M_PE", "LOGIT_SAME": "LOGIT_SAME_PE"}
"""Each arm's name in the tables, all with the strongest-opposite-step end rule."""


def population(*, out: Path) -> pl.DataFrame:
    """One row per transfer and shape, with each arm's `found` as a column."""
    events = pl.read_parquet(EXPERIMENT_58_DIR / "events.parquet")
    onset = events["onset"].to_numpy()
    events = events.with_columns(
        month=pl.Series(stamp_months()[onset]), year_month=pl.Series(stamp_year_months()[onset])
    ).filter(
        pl.col("size_band") == HEADLINE_BAND[0],
        pl.col("duration_band") == HEADLINE_BAND[1],
        pl.col("month") % 2 == 0,
        pl.col("set") != "holiday",
    )
    stored = pl.read_parquet(EXPERIMENT_65_DIR / "event_results.parquet")
    weight = (
        stored.filter(pl.col("shape") == "S1", pl.col("setting") == "D0M_TR")
        .select("event_id", weight=pl.col("moved_mw"))
        .unique("event_id")
    )
    step = (
        stored.filter(pl.col("setting") == "D0M_PE")
        .select("event_id", "shape", "donor_step_z")
        .unique(["event_id", "shape"])
    )
    found = pl.concat(
        [
            stored.filter(pl.col("setting").is_in(["D0M_PE", "LOGIT_SAME_PE"])).select(
                "event_id", "shape", "setting", "found"
            ),
            pl.read_parquet(out / "event_results.parquet").select(
                "event_id", "shape", "setting", "found"
            ),
        ]
    ).pivot(on="setting", index=["event_id", "shape"], values="found")
    return (
        found.join(events, on="event_id", validate="m:1")
        .join(weight, on="event_id", validate="m:1")
        .join(step, on=["event_id", "shape"], validate="1:1")
        .rename({v: k for k, v in ARMS.items()})
        .sort("shape", "event_id")
    )


def interval(*, resampled: np.ndarray) -> tuple[float, float]:
    """The two-sided interval of resampled statistics at `LEVEL`."""
    tail = 100 * (1 - LEVEL) / 2
    low, high = np.percentile(resampled, [tail, 100 - tail])
    return float(low), float(high)


def month_sums(*, frame: pl.DataFrame, found: str) -> tuple[np.ndarray, np.ndarray]:
    """Per calendar year-month: the weight-times-found sum and the weight sum."""
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


def contrast(*, frames: list[pl.DataFrame], arm: str, base: str) -> tuple[float, float, float]:
    """Point value and interval of the shape-averaged MW-weighted recall difference."""
    months = np.unique(frames[0]["year_month"].to_numpy())
    picks = np.random.default_rng(SEED).integers(0, len(months), size=(N_RESAMPLES, len(months)))
    point = float(diffs(frames=frames, arm=arm, base=base, picks=None)[0])
    low, high = interval(resampled=diffs(frames=frames, arm=arm, base=base, picks=picks))
    return point, low, high


def fmt(point: float, low: float, high: float) -> str:
    """A point value with its interval."""
    return f"{point:+.3f} ({low:+.3f} to {high:+.3f})"


def recall(*, frame: pl.DataFrame, column: str) -> tuple[float, float]:
    """MW-weighted and unweighted recall of one arm."""
    w, f = frame["weight"].to_numpy(), frame[column].to_numpy().astype(float)
    return float((w * f).sum() / w.sum()), float(f.mean())


def print_recall(*, data: pl.DataFrame) -> None:
    """Recall of every arm for each shape and pooled, with the share of the moved MW missed."""
    print("\nMW-weighted recall | unweighted recall (even months); missed MW = 1 - first number")
    for shape in NON_MATCHING:
        frame = data.filter(pl.col("shape") == shape)
        cells = ["{} {:.3f}|{:.3f}".format(arm, *recall(frame=frame, column=arm)) for arm in ARMS]
        print(f"{shape} {SHAPE_NAMES[shape]}, n={frame.height}: " + "  ".join(cells))
    frames = [data.filter(pl.col("shape") == s) for s in NON_MATCHING]
    cells = []
    for arm in ARMS:
        value = np.mean([recall(frame=f, column=arm)[0] for f in frames])
        unweighted = np.mean([recall(frame=f, column=arm)[1] for f in frames])
        cells.append(f"{arm} {value:.3f}|{unweighted:.3f} (missed {1 - value:.3f})")
    print("pooled S2-S4: " + "  ".join(cells))


def print_by_step(*, data: pl.DataFrame) -> None:
    """Found / transfers by donor step in noise units, split by metered receiver."""
    print("\nFound / transfers by donor step in noise units; arms: " + " | ".join(ARMS))
    for shape in NON_MATCHING:
        frame = data.filter(pl.col("shape") == shape)
        for flag, label in ((True, "metered"), (False, "unmetered")):
            part = frame.filter((pl.col("metered_receivers") > 0) == flag)
            cells = []
            for low, high in itertools.pairwise(STEP_EDGES):
                bin_ = part.filter(pl.col("donor_step_z").abs().is_between(low, high, "left"))
                counts = "|".join(str(bin_[a].sum()) for a in ARMS)
                cells.append(f"[{low:g},{high:g}) {counts}/{bin_.height}")
            print(f"{shape} {label:9s} " + "; ".join(cells))


def print_real() -> None:
    """The labelled real transfers of more than 1 day found by each arm, by onset parity."""
    mine = pl.read_parquet(WORK_DIR / "real_results.parquet")
    theirs = pl.read_parquet(EXPERIMENT_65_DIR / "real_results.parquet")
    real = pl.concat([theirs.filter(pl.col("setting").is_in(["D0M_PE", "LOGIT_SAME_PE"])), mine])
    real = real.filter(pl.col("hours") > 24)
    print(
        f"\nLabelled real transfers of more than 1 day: {mine.filter(pl.col('hours') > 24).height}"
    )
    for name, setting in ARMS.items():
        part = real.filter(pl.col("setting") == setting)
        even, odd = part.filter(pl.col("month") % 2 == 0), part.filter(pl.col("month") % 2 == 1)
        print(
            f"  {name}: found {part['found'].sum()}, onset found {part['onset_found'].sum()}; "
            f"even-month onsets {even['found'].sum()}/{even.height}, "
            f"odd {odd['found'].sum()}/{odd.height}"
        )


def print_alarms(*, out: Path) -> dict[str, int]:
    """Unexplained alarm counts on the real residual by odd and even month."""
    counts = {}
    ef = pl.read_parquet(out / "alarms.parquet").filter(~pl.col("explained"))
    theirs = pl.read_parquet(EXPERIMENT_65_DIR / "alarms.parquet").filter(
        pl.col("setting") == "D0M_PE", ~pl.col("logged_matched")
    )
    for name, part in (("EF", ef), ("D0M", theirs)):
        counts[name] = part.filter(pl.col("month") % 2 == 0).height
        print(
            f"{name}: unexplained alarms even months {counts[name]}, "
            f"odd {part.filter(pl.col('month') % 2 == 1).height}"
        )
    return counts


def print_tuning(*, out: Path) -> None:
    """The tuning grid: each cell's matched lambda, alarm count, and odd-month recall."""
    tuning = json.loads((out / "tuning.json").read_text())
    print("\nTuning cells (fused ratio, slack multiplier, lambda, odd alarms, recall):")
    for cell in tuning["cells"]:
        print(
            f"  {cell['fused_ratio']}, {cell['slack_multiplier']}, "
            f"{cell.get('level', float('nan')):.3f}, {cell.get('odd_unexplained')}, "
            f"{cell['recall']:.3f} (n={cell.get('n_transfers')})"
        )
    print(f"  chosen: {tuning['chosen']}")


def verdict(*, c1: tuple[float, float, float], alarms: dict[str, int]) -> str:
    """Apply the pre-registered decision rule."""
    point, low, high = c1
    if alarms["EF"] > alarms["D0M"] + ALARM_MARGIN:
        return "inconclusive (even-month alarm cap broken)"
    if high < REFUTED_BELOW:
        return "refuted"
    if low > 0 and point >= SUPPORT_AT_LEAST:
        return "supported"
    if low > 0:
        return "small gain"
    return "inconclusive"


def main() -> None:
    """Print the tables and the verdict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    out = parser.parse_args().out
    data = population(out=out)
    for arm in ARMS:
        if data[arm].null_count() or data.height != 620 * len(NON_MATCHING):
            raise SystemExit(f"STOP: {arm} is missing for some of the 620 x 3 transfers")
    one = data.filter(pl.col("shape") == "S2")
    print(f"{one.height} transfers; {one.filter(pl.col('weight') <= 0).height} move no MW")
    print_recall(data=data)
    frames = [data.filter(pl.col("shape") == s) for s in NON_MATCHING]
    c1 = contrast(frames=frames, arm="EF", base="D0M")
    c2 = contrast(frames=frames, arm="EF", base="LOGIT_SAME")
    print(f"\nC1 EF-D0M: {fmt(*c1)} at level {LEVEL:.4f}")
    print(f"C2 EF-LOGIT_SAME (no verdict): {fmt(*c2)}")
    print_by_step(data=data)
    print_real()
    alarms = print_alarms(out=out)
    print_tuning(out=out)
    print(f"\nEXPERIMENT VERDICT: {verdict(c1=c1, alarms=alarms)}")


if __name__ == "__main__":
    main()
