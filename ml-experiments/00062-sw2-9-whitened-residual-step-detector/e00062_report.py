"""Print every number the write-up of experiment 62 quotes, and the verdict the decision rule gives.

Reads the tables that `e00062_run.py` wrote: MW-weighted recall of D0M and W0 for each injection
shape with month-resampled intervals (C1 pooled over S2, S3 and S4 at 24 hours; C2 the same at 4
hours), recall by donor step in noise units and metered receiver, the labelled real transfers (C3),
alarm counts, and how much the filter changes the residual.

    uv run python ml-experiments/00062-*/e00062_report.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00062_common import (
    ML_EXPERIMENTS_DATA_DIR,
    WORK_DIR,
    load_panel,
    stamp_months,
    stamp_year_months,
)
from e00062_detectors import prepare
from e00062_inject import HEADLINE_BAND
from e00062_profile import NON_MATCHING, SHAPE_NAMES, SHAPES
from e00062_whiten import fit_coefficients, prepare_whitened, whiten

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
HEADLINE_WINDOW: Final[int] = 48
"""The 24-hour window, in half-hours."""
WINDOWS: Final[tuple[int, ...]] = (48, 8)


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


def wide(*, data: pl.DataFrame, shape: str, window: int = HEADLINE_WINDOW) -> pl.DataFrame:
    """One row per transfer with both detectors' `found` for one shape and window."""
    one = data.filter(pl.col("shape") == shape)
    d0 = one.filter(pl.col("setting") == f"D0M_{window}").select(
        "event_id", "weight", "year_month", "metered_receivers", "donor_step_z", f0="found"
    )
    w0 = one.filter(pl.col("setting") == f"W0_{window}").select("event_id", fw="found")
    return d0.join(w0, on="event_id", validate="1:1").sort("event_id")


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
    """MW-weighted recall, W0 minus D0M, averaged over shapes; `picks` resamples months."""
    total = 0.0
    for frame in frames:
        wfw, w = month_sums(frame=frame, found="fw")
        wf0, _ = month_sums(frame=frame, found="f0")
        if picks is None:
            total = total + (wfw.sum() - wf0.sum()) / w.sum()
        else:
            total = total + (wfw[picks].sum(axis=1) - wf0[picks].sum(axis=1)) / w[picks].sum(axis=1)
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
    for name in ("f0", "fw"):
        found = frame[name].to_numpy().astype(float)
        out[f"{name}_mw"] = float((w * found).sum() / w.sum())
        out[f"{name}_n"] = float(found.mean())
    return out


def print_shapes(
    *, data: pl.DataFrame, window: int = HEADLINE_WINDOW
) -> dict[str, tuple[float, float, float]]:
    """Per-shape table; returns each shape's difference and interval."""
    found: dict[str, tuple[float, float, float]] = {}
    print(f"\nWindow {window}: MW-weighted recall (share of moved MW found), even months")
    for shape in SHAPES:
        frame = wide(data=data, shape=shape, window=window)
        rec = recall_rows(frame=frame)
        found[shape] = contrast(frames=[frame])
        point, low, high = found[shape]
        print(
            f"{shape} {SHAPE_NAMES[shape]:22s} n={frame.height} D0M {rec['f0_mw']:.3f} "
            f"W0 {rec['fw_mw']:.3f} diff {point:+.3f} ({low:+.3f} to {high:+.3f}); "
            f"missed MW D0M {1 - rec['f0_mw']:.3f} W0 {1 - rec['fw_mw']:.3f}; "
            f"unweighted D0M {rec['f0_n']:.3f} W0 {rec['fw_n']:.3f}"
        )
        metered = frame.filter(pl.col("metered_receivers") > 0)
        rec = recall_rows(frame=metered)
        point, low, high = contrast(frames=[metered])
        print(
            f"   metered receiver only n={metered.height} D0M {rec['f0_mw']:.3f} "
            f"W0 {rec['fw_mw']:.3f} diff {point:+.3f} ({low:+.3f} to {high:+.3f})"
        )
    print("\nMean correlation of the moved MW with the donor's expected power over the transfer")
    for shape in SHAPES:
        corr = data.filter(pl.col("shape") == shape, pl.col("setting") == "D0M_48")["corr_expected"]
        print(f"{shape} {SHAPE_NAMES[shape]:22s} {corr.drop_nans().mean():+.3f}")
    return found


def print_by_step(*, data: pl.DataFrame, window: int = HEADLINE_WINDOW) -> None:
    """Standard table: recall by donor step in noise units, split by metered receiver."""
    print("\nRecall by donor step in noise units (count found / transfers), D0M | W0")
    for shape in SHAPES:
        frame = wide(data=data, shape=shape, window=window)
        for flag, label in ((True, "metered"), (False, "unmetered")):
            part = frame.filter((pl.col("metered_receivers") > 0) == flag)
            cells = []
            for low, high in itertools.pairwise(STEP_EDGES):
                bin_ = part.filter(pl.col("donor_step_z").abs().is_between(low, high, "left"))
                cells.append(
                    f"[{low:g},{high:g}) {bin_['f0'].sum()}|{bin_['fw'].sum()}/{bin_.height}"
                )
            print(f"{shape} {label:9s} " + "; ".join(cells))


def lag_one_autocorrelation(*, values: np.ndarray, lag: int = 1) -> float:
    """Correlation of a series with itself `lag` half-hours later, over odd months."""
    odd = np.isin(stamp_months(), [1, 3, 5, 7, 9, 11])[lag:]
    ok = odd & np.isfinite(values[lag:]) & np.isfinite(values[:-lag])
    return float(np.corrcoef(values[lag:][ok], values[:-lag][ok])[0, 1])


def print_filter_effect() -> None:
    """Post hoc: what the filter does to the residual and to the 24-hour step, by series."""
    panel = load_panel()
    coefficients = fit_coefficients(residuals=panel["residual"])
    plain = prepare(residuals=panel["residual"], window=48)
    white = prepare_whitened(residuals=panel["residual"], coefficients=coefficients, window=48)
    print(f"\n{len(plain.shift)} series; filter effect on the real residual (odd-month noise)")
    rows = []
    for s, r in panel["residual"].items():
        w = whiten(residual=r, coefficients=coefficients[s])
        x, y = plain.shift[s], white.shift[s]
        ok = np.isfinite(x) & np.isfinite(y)
        rows.append(
            (
                float(np.mean(coefficients[s])),
                lag_one_autocorrelation(values=r),
                lag_one_autocorrelation(values=w),
                lag_one_autocorrelation(values=r, lag=48),
                lag_one_autocorrelation(values=w, lag=48),
                float(np.mean(coefficients[s] >= 0.98)),
                float(np.nanstd(w) / np.nanstd(r)),
                white.scale[s] / plain.scale[s],
                float(np.corrcoef(x[ok], y[ok])[0, 1]),
            )
        )
    table = np.array(rows)
    names = (
        "mean AR coefficient",
        "lag-1 autocorrelation, raw",
        "lag-1 autocorrelation, whitened",
        "lag-48 autocorrelation, raw",
        "lag-48 autocorrelation, whitened",
        "share of the 48 coefficients at the clip",
        "residual std, whitened over raw",
        "24-hour step noise scale, whitened over raw (MW)",
        "correlation of raw and whitened 24-hour step",
    )
    for name, col in zip(names, table.T, strict=True):
        print(f"{name}: median {np.median(col):.3f} (min {col.min():.3f}, max {col.max():.3f})")
    # The same ratio of step size to noise for an ideal constant transfer is the signal gain.
    print(
        "ratio of the whitened step's size to its noise scale for a constant 1 MW step, "
        "relative to the raw step's: "
        + ", ".join(
            f"{(1 - c.mean()) / (white.scale[s] / plain.scale[s]):.2f}"
            for s, c in coefficients.items()
        )
    )


def print_big_steps(*, data: pl.DataFrame) -> None:
    """Whether misses at donor steps of 6 noise units or more lack an alarm or an end."""
    print("\nTransfers whose donor step is 6 noise units or more (24-hour D0M noise units)")
    for setting in ("D0M_48", "W0_48"):
        for shape in SHAPES:
            part = data.filter(
                pl.col("shape") == shape,
                pl.col("setting") == setting,
                pl.col("donor_step_z").abs() >= 6,
            )
            n = part.height
            found = part["found"].sum()
            wrong_end = part.filter(pl.col("onset_found") & ~pl.col("found")).height
            print(
                f"{setting} {shape}: {n} transfers, found {found}, onset found but end wrong "
                f"{wrong_end}, no alarm at the onset {n - found - wrong_end}"
            )


def print_real(*, out: Path, window: int = HEADLINE_WINDOW) -> None:
    """C3: the labelled real transfers of more than 1 day, found by each detector."""
    real = pl.read_parquet(out / "real_results.parquet").filter(pl.col("hours") > 24)
    d0 = real.filter(pl.col("setting") == f"D0M_{window}").sort("card_id")
    w0 = real.filter(pl.col("setting") == f"W0_{window}").sort("card_id")
    both = d0.join(w0, on="card_id", suffix="_w0", validate="1:1")
    n = both.height
    print(f"\nWindow {window}, labelled real transfers of more than 1 day: {n}")
    for vis in ("conserved", "one_sided"):
        part = both.filter(pl.col("visibility") == vis)
        print(
            f"  {vis}: {part.height}; D0M {part['found'].sum()}, W0 {part['found_w0'].sum()}, "
            f"only D0M {part.filter(pl.col('found') & ~pl.col('found_w0')).height}, "
            f"only W0 {part.filter(~pl.col('found') & pl.col('found_w0')).height}"
        )
    print(
        f"  onset found (within 2 hours): D0M {both['onset_found'].sum()}, "
        f"W0 {both['onset_found_w0'].sum()} of {n}"
    )
    w = both["moved_mw_p50"].fill_null(0).to_numpy()
    for name in ("found", "found_w0"):
        f = both[name].to_numpy().astype(float)
        print(f"  MW-weighted recall ({name}): {(w * f).sum() / w.sum():.3f}; count {int(f.sum())}")


def print_alarms(*, out: Path) -> dict[str, tuple[int, int]]:
    """Even-month and odd-month unexplained alarm counts of the real residual, by setting."""
    alarms = pl.read_parquet(out / "alarms.parquet")
    curves = pl.read_parquet(out / "curves.parquet")
    thresholds = json.loads((out / "thresholds.json").read_text())
    counts = {}
    print(f"\nthresholds {thresholds}")
    for setting, threshold in thresholds.items():
        detector, window = setting.split("_")
        part = alarms.filter(pl.col("setting") == setting, ~pl.col("logged_matched"))
        counts[setting] = (
            part.filter(pl.col("month") % 2 == 0).height,
            part.filter(pl.col("month") % 2 == 1).height,
        )
        row = curves.filter(
            pl.col("detector") == detector,
            pl.col("window") == int(window),
            pl.col("threshold") == threshold,
        )
        print(
            f"{setting}: odd-month unexplained alarms {counts[setting][1]} "
            f"(tuning curve {row['odd_unexplained'].item()}); even-month {counts[setting][0]}"
        )
    return counts


def verdict(
    *, pooled: tuple[float, float, float], shapes: dict, alarms: dict[str, tuple[int, int]]
) -> str:
    """Apply the pre-registered decision rule."""
    point, low, high = pooled
    d0_alarms, w0_alarms = alarms["D0M_48"][0], alarms["W0_48"][0]
    if high < SUPPORT_AT_LEAST:
        return "refuted"
    if low > 0 and point >= SUPPORT_AT_LEAST and w0_alarms <= ALARM_TOLERANCE * d0_alarms:
        below = [s for s in NON_MATCHING if shapes[s][2] < 0]
        return "partly supported" if below else "supported"
    return "inconclusive"


def main() -> None:
    """Print the tables and the verdict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    out = parser.parse_args().out
    data = population(out=out)
    one = data.filter(pl.col("shape") == "S1", pl.col("setting") == "D0M_48")
    if one.height != 620:
        raise SystemExit(f"STOP: {one.height} transfers, the plan has 620")
    print(f"{one.height} transfers; {one.filter(pl.col('weight') <= 0).height} move no MW")
    pooled_by_window = {}
    shapes_by_window = {}
    for window in WINDOWS:
        shapes_by_window[window] = print_shapes(data=data, window=window)
        pooled_by_window[window] = contrast(
            frames=[wide(data=data, shape=s, window=window) for s in NON_MATCHING]
        )
        point, low, high = pooled_by_window[window]
        print(
            f"\nWindow {window}, pooled over {NON_MATCHING}: "
            f"{point:+.3f} ({low:+.3f} to {high:+.3f})"
        )
        s1 = shapes_by_window[window]["S1"]
        print(f"S1 alone: {s1[0]:+.3f} ({s1[1]:+.3f} to {s1[2]:+.3f})")
        print_by_step(data=data, window=window)
        print_real(out=out, window=window)
    print_big_steps(data=data)
    print_filter_effect()
    alarms = print_alarms(out=out)
    verdict_word = verdict(
        pooled=pooled_by_window[HEADLINE_WINDOW],
        shapes=shapes_by_window[HEADLINE_WINDOW],
        alarms=alarms,
    )
    print(f"\nVERDICT: {verdict_word}")


if __name__ == "__main__":
    main()
