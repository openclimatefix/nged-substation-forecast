"""Print every number the write-up of experiment 68 quotes, and the verdict the decision rule gives.

Reads the tables that `e00068_run.py loop` wrote. C1 is MW-weighted recall in world B (injected
transfers in the training data), round 3 minus round 0, averaged over S2, S3, and S4. C2 is the
relative change in the forecast's mean absolute error in world A on even months outside logged
switching and labelled transfers. Both are read at 97.5% from resamples of whole calendar months.

    uv run python ml-experiments/00068-*/e00068_report.py
"""

from __future__ import annotations

import itertools
import json
from typing import Final

import numpy as np
import polars as pl
from e00068_common import WORK_DIR

N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 1 - 0.05 / 2
"""Two-sided level of the intervals: Bonferroni across the two verdict contrasts C1 and C2."""
SEED: Final[int] = 6800
RECALL_AT_LEAST: Final[float] = 0.02
ERROR_AT_MOST: Final[float] = -0.01
ALARM_TOLERANCE: Final[float] = 1.10
STEP_EDGES: Final[tuple[float, ...]] = (0.0, 2.0, 4.0, 6.0, 10.0, float("inf"))
ROUNDS: Final[tuple[int, ...]] = (0, 1, 2, 3)
SHAPES: Final[tuple[str, ...]] = ("S2", "S3", "S4")
LONG_HOURS: Final[float] = 7 * 24


def interval(*, resampled: np.ndarray) -> tuple[float, float]:
    """Two-sided interval of resampled statistics at `LEVEL`."""
    tail = 100 * (1 - LEVEL) / 2
    low, high = np.percentile(resampled, [tail, 100 - tail])
    return float(low), float(high)


def fmt(value: tuple[float, float, float], *, percent: bool = False) -> str:
    """A point value with its interval."""
    scale = 100 if percent else 1
    unit = "%" if percent else ""
    point, low, high = (v * scale for v in value)
    return f"{point:+.3f}{unit} ({low:+.3f}{unit} to {high:+.3f}{unit})"


def event_frame(*, data: pl.DataFrame, world: str, name: str) -> pl.DataFrame:
    """One round's rows for one world."""
    return data.filter(pl.col("world") == world, pl.col("round") == name).sort("event_id", "shape")


def recall_contrast(
    *, data: pl.DataFrame, world: str, arm: str, base: str, subset: pl.Expr | None = None
) -> tuple[float, float, float]:
    """Shape-averaged MW-weighted recall of `arm` minus `base`, with a month-resampled interval."""
    a = event_frame(data=data, world=world, name=arm)
    b = event_frame(data=data, world=world, name=base)
    joined = a.join(
        b.select("event_id", "shape", found_base="found"), on=["event_id", "shape"], validate="1:1"
    )
    if subset is not None:
        joined = joined.filter(subset)
    months = np.sort(joined["year_month"].unique().to_numpy())
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(months), size=(N_RESAMPLES, len(months)))
    total_point, total_draw = 0.0, np.zeros(N_RESAMPLES)
    for shape in SHAPES:
        part = joined.filter(pl.col("shape") == shape)
        grouped = (
            part.with_columns(
                wa=pl.col("moved_mw") * pl.col("found").cast(pl.Float64),
                wb=pl.col("moved_mw") * pl.col("found_base").cast(pl.Float64),
            )
            .group_by("year_month")
            .agg(pl.col("wa").sum(), pl.col("wb").sum(), pl.col("moved_mw").sum().alias("w"))
        )
        lookup = {int(m): i for i, m in enumerate(grouped["year_month"].to_numpy())}
        wa, wb, w = (np.zeros(len(months)) for _ in range(3))
        for j, m in enumerate(months):
            if int(m) in lookup:
                i = lookup[int(m)]
                wa[j], wb[j], w[j] = grouped["wa"][i], grouped["wb"][i], grouped["w"][i]
        total_point += (wa.sum() - wb.sum()) / w.sum()
        total_draw += (wa[picks].sum(axis=1) - wb[picks].sum(axis=1)) / w[picks].sum(axis=1)
    low, high = interval(resampled=total_draw / len(SHAPES))
    return total_point / len(SHAPES), low, high


def recall(*, frame: pl.DataFrame) -> tuple[float, float]:
    """MW-weighted and unweighted recall of one table, pooled with equal weight over shapes."""
    weighted, plain = [], []
    for shape in SHAPES:
        part = frame.filter(pl.col("shape") == shape)
        w = part["moved_mw"].to_numpy()
        f = part["found"].to_numpy().astype(float)
        weighted.append((w * f).sum() / w.sum())
        plain.append(f.mean())
    return float(np.mean(weighted)), float(np.mean(plain))


def error_contrast(
    *, errors: pl.DataFrame, arm: str, base: str
) -> tuple[float, float, float, float, float]:
    """Relative change in MAE (arm vs base) on even months, equal weight per series.

    Returns the point value, its interval, and the two absolute MAEs (points of capacity).
    """
    even = errors.filter(pl.col("year_month") % 100 % 2 == 0)
    months = np.sort(even["year_month"].unique().to_numpy())
    series = np.sort(even["series"].unique().to_numpy())

    def cube(name: str) -> tuple[np.ndarray, np.ndarray]:
        part = even.filter(pl.col("round") == name)
        total = np.zeros((len(series), len(months)))
        count = np.zeros_like(total)
        si = {int(s): i for i, s in enumerate(series)}
        mi = {int(m): i for i, m in enumerate(months)}
        for s, m, e, n in part.select("series", "year_month", "abs_error_sum", "n").iter_rows():
            total[si[s], mi[m]], count[si[s], mi[m]] = e, n
        return total, count

    ta, na = cube(arm)
    tb, nb = cube(base)

    def mae(total: np.ndarray, count: np.ndarray, picks: np.ndarray | None) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            if picks is None:
                return np.atleast_1d(np.nanmean(total.sum(axis=1) / count.sum(axis=1)))
            return np.nanmean(total[:, picks].sum(axis=2) / count[:, picks].sum(axis=2), axis=0)

    point_a, point_b = mae(ta, na, None)[0], mae(tb, nb, None)[0]
    rng = np.random.default_rng(SEED + 1)
    picks = rng.integers(0, len(months), size=(N_RESAMPLES, len(months)))
    draws = mae(ta, na, picks) / mae(tb, nb, picks) - 1
    low, high = interval(resampled=draws)
    return point_a / point_b - 1, low, high, float(point_a), float(point_b)


def print_recall(*, data: pl.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Recall by round in each world, and the contrasts."""
    out: dict[str, tuple[float, float, float]] = {}
    for world in ("B_in_training", "A_into_residual", "A_into_residual_round0_threshold"):
        print(f"\n{world}: MW-weighted recall (unweighted), pooled over S2, S3, S4")
        names = sorted(data.filter(pl.col("world") == world)["round"].unique().to_list())
        for name in names:
            frame = event_frame(data=data, world=world, name=name)
            weighted, plain = recall(frame=frame)
            print(
                f"  {name}: {weighted:.3f} ({plain:.3f}); missed MW {1 - weighted:.3f}; "
                f"n={frame['event_id'].n_unique()}"
            )
        worlds_rounds = [f"{world[0]}_r{k}" for k in ROUNDS]
        for base, arm in itertools.pairwise(worlds_rounds):
            if arm in names and base in names:
                value = recall_contrast(data=data, world=world, arm=arm, base=base)
                out[f"{world}:{arm}-{base}"] = value
                print(f"  {arm} minus {base}: {fmt(value)}")
        if f"{world[0]}_r3" in names and f"{world[0]}_r0" in names:
            value = recall_contrast(
                data=data, world=world, arm=f"{world[0]}_r3", base=f"{world[0]}_r0"
            )
            out[f"{world}:r3-r0"] = value
            print(f"  r3 minus r0: {fmt(value)}")
            long_only = recall_contrast(
                data=data,
                world=world,
                arm=f"{world[0]}_r3",
                base=f"{world[0]}_r0",
                subset=pl.col("hours") > LONG_HOURS,
            )
            short_only = recall_contrast(
                data=data,
                world=world,
                arm=f"{world[0]}_r3",
                base=f"{world[0]}_r0",
                subset=pl.col("hours") <= LONG_HOURS,
            )
            first = event_frame(data=data, world=world, name=f"{world[0]}_r0")
            n_long = int((first["hours"] > LONG_HOURS).sum())
            print(f"  transfer rows lasting over 7 days: {n_long}")
            print(
                f"  r3 minus r0, transfers over 7 days: {fmt(long_only)}; "
                f"7 days or less: {fmt(short_only)}"
            )
    value = recall_contrast(data=data, world="A_into_residual", arm="A_r1", base="A_rc1")
    out["C3_recall"] = value
    print(f"\nC3, world A recall, round 1 minus random-mask control: {fmt(value)}")
    return out


def print_by_step(*, data: pl.DataFrame) -> None:
    """Standard table: recall by donor step in noise units, split by metered receiver."""
    for world, first, last in (
        ("B_in_training", "B_r0", "B_r3"),
        ("A_into_residual", "A_r0", "A_r3"),
    ):
        print(
            f"\nRecall by donor step in noise units, {world}: found/transfers at {first} | {last}"
        )
        for flag, label in ((True, "metered"), (False, "unmetered")):
            cells = []
            for low, high in itertools.pairwise(STEP_EDGES):
                counts = []
                for name in (first, last):
                    frame = event_frame(data=data, world=world, name=name).filter(
                        (pl.col("metered_receivers") > 0) == flag,
                        pl.col("donor_step_z").abs().is_between(low, high, "left"),
                    )
                    counts.append((frame["found"].sum(), frame.height))
                cells.append(f"[{low:g},{high:g}) {counts[0][0]}|{counts[1][0]}/{counts[0][1]}")
            print(f"  {label:9s} " + "; ".join(cells))


def print_error(*, errors: pl.DataFrame) -> dict[str, tuple[float, float, float]]:
    """C2 and the by-round errors."""
    out: dict[str, tuple[float, float, float]] = {}
    print("\nForecast error in world A, even months, outside logged and labelled switching")
    for arm, base in (("A_r3", "A_r0"), ("A_r1", "A_r0"), ("A_r2", "A_r1"), ("A_r3", "A_r2")):
        change, low, high, mae_a, mae_b = error_contrast(errors=errors, arm=arm, base=base)
        out[f"{arm}-{base}"] = (change, low, high)
        print(
            f"  {arm} vs {base}: MAE {mae_a * 100:.4f} vs {mae_b * 100:.4f} points of capacity; "
            f"relative change {fmt((change, low, high), percent=True)}"
        )
    change, low, high, mae_a, mae_b = error_contrast(errors=errors, arm="A_r1", base="A_rc1")
    out["C3_error"] = (change, low, high)
    print(
        f"  C3, A_r1 vs random-mask control A_rc1: MAE {mae_a * 100:.4f} vs {mae_b * 100:.4f}; "
        f"relative change {fmt((change, low, high), percent=True)}"
    )
    all_months = errors.filter(pl.col("round") == "A_r0")
    print(f"  half-hours scored per round (even and odd months): {int(all_months['n'].sum()):,}")
    return out


def print_real() -> None:
    """The labelled real transfers of more than 1 day found in each round."""
    real = pl.read_parquet(WORK_DIR / "real_results.parquet").filter(pl.col("hours") > 24)
    print("\nLabelled real transfers of more than 24 hours (in the training data and the mask)")
    for name in sorted(real["round"].unique().to_list()):
        part = real.filter(pl.col("round") == name)
        w = part["moved_mw_p50"].fill_null(0).to_numpy()
        f = part["found"].to_numpy().astype(float)
        print(
            f"  {name}: found {int(f.sum())} of {part.height}; onset found "
            f"{part['onset_found'].sum()}; MW-weighted recall {(w * f).sum() / w.sum():.3f}"
        )


def print_stats(*, stats: dict) -> dict[str, dict]:
    """Per-round threshold, scale, and alarm counts."""
    print("\nRound statistics")
    for name, row in sorted(stats.items()):
        print(
            f"  {name}: threshold {row['threshold']}, median noise scale "
            f"{row['median_scale']:.4f} MW, odd-month unexplained {row['odd_unexplained']}, "
            f"even-month unexplained {row['even_unexplained']}, intervals {row['n_intervals']}, "
            f"masked half-hours {row['masked_half_hours']:,}, odd-month unexplained at round 0's "
            f"threshold {row.get('odd_unexplained_at_round0_threshold', '-')}"
        )
    return stats


def verdict(
    *,
    recall_c1: tuple[float, float, float],
    error_c2: tuple[float, float, float],
    stats: dict,
    check_ok: bool,
) -> str:
    """Apply the pre-registered decision rule."""
    if not check_ok:
        return "invalid"
    point_r, low_r, high_r = recall_c1
    point_e, low_e, high_e = error_c2
    if high_r < RECALL_AT_LEAST and low_e > ERROR_AT_MOST:
        return "refuted"
    base = stats["A_r0"]["even_unexplained"]
    quiet = all(
        abs(stats[f"A_r{k}"]["even_unexplained"] / base - 1) <= ALARM_TOLERANCE - 1
        for k in (1, 2, 3)
    )
    c1 = low_r > 0 and point_r >= RECALL_AT_LEAST
    c2 = high_e < 0 and point_e <= ERROR_AT_MOST
    if c1 and c2 and quiet:
        return "supported"
    if (c1 != c2) and quiet:
        return "partly supported"
    if low_r > 0 or high_e < 0:
        return "small gain" if quiet else "inconclusive"
    return "inconclusive"


def main() -> None:
    """Print the tables and the verdict."""
    data = pl.read_parquet(WORK_DIR / "event_results.parquet")
    errors = pl.read_parquet(WORK_DIR / "errors.parquet")
    stats = print_stats(stats=json.loads((WORK_DIR / "round_stats.json").read_text()))
    recalls = print_recall(data=data)
    print_by_step(data=data)
    errs = print_error(errors=errors)
    print_real()
    print(
        "\nVERDICT: "
        + verdict(
            recall_c1=recalls["B_in_training:r3-r0"],
            error_c2=errs["A_r3-A_r0"],
            stats=stats,
            check_ok=True,
        )
    )


if __name__ == "__main__":
    main()
