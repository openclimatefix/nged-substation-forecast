"""Print every number the write-up of experiment 60 quotes, and the verdict the decision rule gives.

Reads the tables that `e00060_run.py` wrote: MW-weighted recall of D0M and D3 for each injection
shape with month-resampled intervals (C1 pooled over S2, S3 and S4; C2 on S1), recall by donor
step in noise units and metered receiver, the labelled real transfers (C3), and alarm counts.

    uv run python ml-experiments/00060-*/e00060_report.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00060_common import (
    ML_EXPERIMENTS_DATA_DIR,
    WORK_DIR,
    load_panel,
    stamp_months,
    stamp_year_months,
)
from e00060_detectors import prepare
from e00060_inject import HEADLINE_BAND
from e00060_profile import NON_MATCHING, SHAPE_NAMES, SHAPES, prepare_profile

EXPERIMENT_58_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00058-sw2-8b-donor-step-recut-and-stronger-veto"
)
N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 1 - 0.05 / 3
"""Two-sided level of the intervals: Bonferroni across the three contrasts C1 to C3 (98.33%)."""
SEED: Final[int] = 6000
SUPPORT_AT_LEAST: Final[float] = 0.01
ALARM_TOLERANCE: Final[float] = 1.10
STEP_EDGES: Final[tuple[float, ...]] = (0.0, 2.0, 4.0, 6.0, 10.0, float("inf"))
DETECTORS: Final[tuple[str, str]] = ("D0M_48", "D3_48")


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
    """One row per transfer with both detectors' `found` for one shape."""
    one = data.filter(pl.col("shape") == shape)
    d0 = one.filter(pl.col("setting") == "D0M_48").select(
        "event_id", "weight", "year_month", "metered_receivers", "donor_step_z", f0="found"
    )
    d3 = one.filter(pl.col("setting") == "D3_48").select("event_id", f3="found")
    return d0.join(d3, on="event_id", validate="1:1").sort("event_id")


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


def diffs(*, frames: list[pl.DataFrame], picks: np.ndarray | None) -> np.ndarray:
    """MW-weighted recall, D3 minus D0M, averaged over shapes; `picks` resamples months."""
    total = 0.0
    for frame in frames:
        wf3, w = month_sums(frame=frame, found="f3")
        wf0, _ = month_sums(frame=frame, found="f0")
        if picks is None:
            total = total + (wf3.sum() - wf0.sum()) / w.sum()
        else:
            total = total + (wf3[picks].sum(axis=1) - wf0[picks].sum(axis=1)) / w[picks].sum(axis=1)
    return np.atleast_1d(total / len(frames))


def contrast(*, frames: list[pl.DataFrame]) -> tuple[float, float, float]:
    """Point value and interval of the shape-averaged MW-weighted recall difference."""
    months = np.unique(frames[0]["year_month"].to_numpy())
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(months), size=(N_RESAMPLES, len(months)))
    point = float(diffs(frames=frames, picks=None)[0])
    low, high = interval(resampled=diffs(frames=frames, picks=picks))
    return point, low, high


def recall_rows(*, frame: pl.DataFrame) -> dict[str, float]:
    """Weighted and unweighted recall of both detectors for one set of transfers."""
    w = frame["weight"].to_numpy()
    out = {}
    for name in ("f0", "f3"):
        found = frame[name].to_numpy().astype(float)
        out[f"{name}_mw"] = float((w * found).sum() / w.sum())
        out[f"{name}_n"] = float(found.mean())
    return out


def print_shapes(*, data: pl.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Per-shape table; returns each shape's difference and interval."""
    found: dict[str, tuple[float, float, float]] = {}
    print("\nMW-weighted recall (share of the moved MW in found transfers), even months")
    for shape in SHAPES:
        frame = wide(data=data, shape=shape)
        rec = recall_rows(frame=frame)
        found[shape] = contrast(frames=[frame])
        point, low, high = found[shape]
        print(
            f"{shape} {SHAPE_NAMES[shape]:22s} n={frame.height} D0M {rec['f0_mw']:.3f} "
            f"D3 {rec['f3_mw']:.3f} diff {point:+.3f} ({low:+.3f} to {high:+.3f}); "
            f"missed MW D0M {1 - rec['f0_mw']:.3f} D3 {1 - rec['f3_mw']:.3f}; "
            f"unweighted D0M {rec['f0_n']:.3f} D3 {rec['f3_n']:.3f}"
        )
        metered = frame.filter(pl.col("metered_receivers") > 0)
        rec = recall_rows(frame=metered)
        point, low, high = contrast(frames=[metered])
        print(
            f"   metered receiver only n={metered.height} D0M {rec['f0_mw']:.3f} "
            f"D3 {rec['f3_mw']:.3f} diff {point:+.3f} ({low:+.3f} to {high:+.3f})"
        )
    print("\nMean correlation of the moved MW with the donor's expected power over the transfer")
    for shape in SHAPES:
        corr = data.filter(pl.col("shape") == shape, pl.col("setting") == "D0M_48")["corr_expected"]
        print(f"{shape} {SHAPE_NAMES[shape]:22s} {corr.drop_nans().mean():+.3f}")
    return found


def print_by_step(*, data: pl.DataFrame) -> None:
    """Standard table: recall by donor step in noise units, split by metered receiver."""
    print("\nRecall by donor step in noise units (count found / transfers), D0M | D3")
    for shape in SHAPES:
        frame = wide(data=data, shape=shape)
        for flag, label in ((True, "metered"), (False, "unmetered")):
            part = frame.filter((pl.col("metered_receivers") > 0) == flag)
            cells = []
            for low, high in itertools.pairwise(STEP_EDGES):
                bin_ = part.filter(pl.col("donor_step_z").abs().is_between(low, high, "left"))
                cells.append(
                    f"[{low:g},{high:g}) {bin_['f0'].sum()}|{bin_['f3'].sum()}/{bin_.height}"
                )
            print(f"{shape} {label:9s} " + "; ".join(cells))


def print_statistic_similarity() -> None:
    """How far D3's step differs from D0's on the real residual, over all stamps and series."""
    panel = load_panel()
    plain = prepare(residuals=panel["residual"], window=48)
    profile = prepare_profile(residuals=panel["residual"], expected=panel["expected"], window=48)
    x, y = [], []
    for s in plain.shift:
        ok = np.isfinite(plain.shift[s]) & np.isfinite(profile.shift[s])
        x.append(plain.shift[s][ok] / plain.scale[s])
        y.append(profile.shift[s][ok] / profile.scale[s])
    x, y = np.concatenate(x), np.concatenate(y)
    big = np.abs(x) > 3
    print(
        f"\nD0 and D3 step in noise units, all stamps: correlation {np.corrcoef(x, y)[0, 1]:.3f}; "
        f"where D0 exceeds 3: correlation {np.corrcoef(x[big], y[big])[0, 1]:.3f}, "
        f"mean ratio D3/D0 {np.mean(y[big] / x[big]):.3f}"
    )


def print_big_steps(*, data: pl.DataFrame) -> None:
    """Whether D0M's misses at donor steps of 6 noise units or more lack an alarm or an end."""
    print("\nD0M on transfers whose donor step is 6 noise units or more")
    for shape in SHAPES:
        part = data.filter(
            pl.col("shape") == shape,
            pl.col("setting") == "D0M_48",
            pl.col("donor_step_z").abs() >= 6,
        )
        n = part.height
        found = part["found"].sum()
        wrong_end = part.filter(pl.col("onset_found") & ~pl.col("found")).height
        print(
            f"{shape}: {n} transfers, found {found}, onset found but end wrong {wrong_end}, "
            f"no alarm at the onset {n - found - wrong_end}"
        )


def print_real(*, out: Path) -> None:
    """C3: the labelled real transfers of more than 1 day, found by each detector."""
    real = pl.read_parquet(out / "real_results.parquet").filter(pl.col("hours") > 24)
    d0 = real.filter(pl.col("setting") == "D0M_48").sort("card_id")
    d3 = real.filter(pl.col("setting") == "D3_48").sort("card_id")
    both = d0.join(d3, on="card_id", suffix="_d3", validate="1:1")
    n = both.height
    print(f"\nC3: labelled real transfers of more than 1 day: {n}")
    for vis in ("conserved", "one_sided"):
        part = both.filter(pl.col("visibility") == vis)
        print(
            f"  {vis}: {part.height}; D0M {part['found'].sum()}, D3 {part['found_d3'].sum()}, "
            f"only D0M {part.filter(pl.col('found') & ~pl.col('found_d3')).height}, "
            f"only D3 {part.filter(~pl.col('found') & pl.col('found_d3')).height}"
        )
    print(
        f"  onset found (within 2 hours): D0M {both['onset_found'].sum()}, "
        f"D3 {both['onset_found_d3'].sum()} of {n}"
    )
    w = both["moved_mw_p50"].fill_null(0).to_numpy()
    for name in ("found", "found_d3"):
        f = both[name].to_numpy().astype(float)
        print(f"  MW-weighted recall ({name}): {(w * f).sum() / w.sum():.3f}; count {int(f.sum())}")


def print_alarms(*, out: Path) -> tuple[int, int]:
    """Even-month unexplained alarm counts of the real residual, and the odd-month counts."""
    alarms = pl.read_parquet(out / "alarms.parquet")
    curves = pl.read_parquet(out / "curves.parquet")
    thresholds = json.loads((out / "thresholds.json").read_text())
    counts = {}
    for setting in DETECTORS:
        part = alarms.filter(pl.col("setting") == setting, ~pl.col("logged_matched"))
        counts[setting] = (
            part.filter(pl.col("month") % 2 == 0).height,
            part.filter(pl.col("month") % 2 == 1).height,
        )
    print(f"\nthresholds {thresholds}")
    for detector, setting in (("D0M", "D0M_48"), ("D3", "D3_48")):
        row = curves.filter(
            pl.col("detector") == detector, pl.col("threshold") == thresholds[setting]
        )
        print(
            f"{setting}: odd-month unexplained alarms {counts[setting][1]} "
            f"(tuning curve {row['odd_unexplained'].item()}); even-month {counts[setting][0]}"
        )
    return counts["D0M_48"][0], counts["D3_48"][0]


def verdict(*, pooled: tuple[float, float, float], shapes: dict, alarms: tuple[int, int]) -> str:
    """Apply the pre-registered decision rule."""
    point, low, high = pooled
    d0_alarms, d3_alarms = alarms
    if high < SUPPORT_AT_LEAST:
        return "refuted"
    if low > 0 and point >= SUPPORT_AT_LEAST and d3_alarms <= ALARM_TOLERANCE * d0_alarms:
        below = [s for s in NON_MATCHING if shapes[s][2] < 0]
        return "partly supported" if below else "supported"
    return "inconclusive"


def main() -> None:
    """Print the tables and the verdict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    out = parser.parse_args().out
    data = population(out=out)
    print(f"{data.filter(pl.col('shape') == 'S1', pl.col('setting') == 'D0M_48').height} transfers")
    shapes = print_shapes(data=data)
    pooled = contrast(frames=[wide(data=data, shape=s) for s in NON_MATCHING])
    print(
        f"\nC1 pooled over {NON_MATCHING}: {pooled[0]:+.3f} ({pooled[1]:+.3f} to {pooled[2]:+.3f})"
    )
    print(f"C2 (S1): {shapes['S1'][0]:+.3f} ({shapes['S1'][1]:+.3f} to {shapes['S1'][2]:+.3f})")
    print_by_step(data=data)
    print_big_steps(data=data)
    print_statistic_similarity()
    print_real(out=out)
    alarms = print_alarms(out=out)
    print(f"\nVERDICT: {verdict(pooled=pooled, shapes=shapes, alarms=alarms)}")


if __name__ == "__main__":
    main()
