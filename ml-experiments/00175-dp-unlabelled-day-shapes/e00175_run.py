"""Classify the shape of the forecast's error on the 37 unlabelled worst days (experiment 175).

Reads experiment 174's day list and arm D's residuals, applies the pre-registered rules of
`e00175_classify`, checks interconnected neighbours for an opposite-signed step, and tallies the
detectors. Prints the report and writes the tables under `WORK_DIR`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00175-*/e00175_run.py > printed_report.txt
"""

import math
import os
from datetime import date
from typing import Final

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import e00175_classify as rules
import e00175_common as common
import numpy as np
import polars as pl

N_SHIFT_DRAWS: Final[int] = 1_000
SHIFT_DAYS: Final[tuple[int, int]] = (1, 90)
DETECTOR_WINDOW_READINGS: Final[int] = 6
"""Three hours either side of the substation's edge."""
DETECTOR_SETTINGS: Final[tuple[str, ...]] = ("R2_M48", "R0_M48")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)


def wilson(*, hits: int, n: int) -> tuple[float, float]:
    """Return the 95% Wilson interval of a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    z = 1.96
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def scale_by_position(*, residual: np.ndarray, local_dates: np.ndarray) -> np.ndarray:
    """Return, for every grid position, the day-before spread of the position's local day."""
    days = list(np.unique(local_dates))
    spreads = rules.day_before_spreads(residual=residual, local_dates=local_dates, days=days)
    lookup = np.array([spreads[d] for d in days])
    index = np.searchsorted(np.array(days), local_dates)
    return lookup[index]


def neighbour_hit(
    *,
    residual: np.ndarray,
    scale: np.ndarray,
    edge_position: int,
    sign: float,
) -> bool:
    """Return whether a neighbour shows an opposite-signed step near `edge_position`."""
    return rules.neighbour_opposite_step(
        residual=residual,
        edge_position=edge_position,
        scale=float(scale[edge_position]),
        sign=sign,
    )


def detector_tables() -> dict[str, pl.DataFrame]:
    """Return the stored real detector onsets: RO-NC settings and the pair alarm."""
    intervals = pl.read_parquet(common.E147_INTERVALS).filter(
        pl.col("setting").is_in(list(DETECTOR_SETTINGS))
    )
    tables = {
        setting: intervals.filter(pl.col("setting") == setting).select("series", "onset")
        for setting in DETECTOR_SETTINGS
    }
    tables["pair_alarm_W8"] = pl.read_parquet(common.E156_PAIR_ALARMS).select(
        "series", onset=pl.col("stamp")
    )
    return tables


def fired(
    *, table: pl.DataFrame, grid: common.ResidualGrid, series: int, edge_position: int
) -> bool:
    """Return whether a detector onset for `series` lies within 3 hours of the edge."""
    onsets = table.filter(pl.col("series") == series)["onset"].to_list()
    return any(
        abs(grid.position_of_stamp(stamp=int(o)) - edge_position) <= DETECTOR_WINDOW_READINGS
        for o in onsets
    )


def main() -> None:
    """Run the classification, the neighbour check, and the tallies."""
    days = common.unlabelled_days()
    sites = tuple(sorted(days["site"].unique().to_list()))
    arm_d = common.arm_d_residuals(sites=sites)
    bg2 = common.bg2_residuals()
    grid = common.ResidualGrid(residual=arm_d)
    link = common.neighbours()
    ids = common.series_ids()
    detectors = detector_tables()
    neighbour_scale = {
        label: scale_by_position(residual=bg2[label], local_dates=grid.local_dates) for label in bg2
    }

    rows: list[dict] = []
    for site in sites:
        residual = arm_d[site]
        site_days: list[date] = days.filter(pl.col("site") == site)["local_date"].to_list()
        spreads = rules.day_before_spreads(
            residual=residual, local_dates=grid.local_dates, days=site_days
        )
        for day in site_days:
            positions = grid.positions_of(day=day)
            scale = spreads[day]
            edges = rules.step_candidates(residual=residual, positions=positions, scale=scale)
            shape = rules.classify_day(
                day=residual[positions], hours=grid.local_hours[positions], edges=edges, scale=scale
            )
            row: dict = {
                "site": site,
                "local_date": day,
                "shape": shape,
                "spread": scale,
                "n_valid": int(np.isfinite(residual[positions]).sum()),
                "n_edges": len(edges),
            }
            if edges:
                primary = max(edges, key=lambda edge: edge.strength)
                metered = sorted(n for n in link[site] if n in bg2)
                hits = [
                    n
                    for n in metered
                    if any(
                        neighbour_hit(
                            residual=bg2[n],
                            scale=neighbour_scale[n],
                            edge_position=e.position,
                            sign=e.shift,
                        )
                        for e in edges
                    )
                ]
                row |= {
                    "edge_position": primary.position,
                    "edge_shift": primary.shift,
                    "edge_strength": primary.strength,
                    "returns": rules.returns_within_day(primary=primary, edges=edges),
                    "persists_7d": rules.persists(
                        residual=residual,
                        local_dates=grid.local_dates,
                        day=day,
                        shift=primary.shift,
                    ),
                    "n_neighbours": len(link[site]),
                    "n_metered_neighbours": len(metered),
                    "neighbour_step": bool(hits) if metered else None,
                    "neighbours_with_step": ",".join(hits),
                }
                for name, table in detectors.items():
                    row[f"fired_{name}"] = fired(
                        table=table, grid=grid, series=ids[site], edge_position=primary.position
                    )
            rows.append(row)
    table = pl.DataFrame(rows, infer_schema_length=None).sort(["site", "local_date"])
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(common.WORK_DIR / "day_shapes.parquet")

    print(table)
    n = table.height
    print(f"\nDays: {n}")
    counts = table.group_by("shape").len().sort("shape")
    for shape, count in counts.iter_rows():
        low, high = wilson(hits=count, n=n)
        print(f"  {shape}: {count} ({count / n:.3f}, 95% Wilson {low:.3f} to {high:.3f})")
    print("\nBy site:")
    print(
        table.group_by(["site", "shape"])
        .len()
        .sort(["site", "shape"])
        .pivot(on="shape", index="site", values="len")
        .fill_null(0)
    )
    blocks = table.filter(pl.col("shape") == "a_block")
    n_a = blocks.height
    print(f"\nClass (a) days: {n_a} of {n} (rule: at least {math.ceil(n / 2)})")
    with_neighbour = blocks.filter(pl.col("n_metered_neighbours") > 0)
    hits = int(with_neighbour["neighbour_step"].sum())
    low, high = wilson(hits=hits, n=with_neighbour.height)
    print(
        f"Class (a) days with a metered neighbour: {with_neighbour.height}; "
        f"opposite-signed neighbour step on {hits} "
        f"({hits / max(with_neighbour.height, 1):.3f}, Wilson {low:.3f} to {high:.3f}); "
        f"rule: at least a third"
    )
    print(f"Class (a) days with no metered neighbour: {n_a - with_neighbour.height}")
    for name in detectors:
        k = int(blocks[f"fired_{name}"].sum())
        print(f"Detector {name} fired within 3 h of the edge: {k} of {n_a}")
    print(f"Block returns within the day: {int(blocks['returns'].sum())} of {n_a}")
    persist = blocks["persists_7d"]
    print(
        f"Persists over the next 7 days: {int((persist == True).sum())}; "  # noqa: E712
        f"reverts: {int((persist == False).sum())}; unknown: {persist.null_count()}"  # noqa: E712
    )

    chance(
        blocks=with_neighbour,
        grid=grid,
        link=link,
        bg2=bg2,
        scale=neighbour_scale,
        arm_d=arm_d,
        edge_table=table,
        observed=hits / max(with_neighbour.height, 1),
    )
    print("\nExperiment 172's known-answer control recovered 89.2% of injected days (bar 90%).")


def chance(
    *,
    blocks: pl.DataFrame,
    grid: common.ResidualGrid,
    link: dict[str, set[str]],
    bg2: dict[str, np.ndarray],
    scale: dict[str, np.ndarray],
    arm_d: dict[str, np.ndarray],
    edge_table: pl.DataFrame,
    observed: float,
) -> None:
    """Print the chance rate of an opposite-signed neighbour step, by shifting neighbours."""
    rng = np.random.default_rng(common.SEED)
    steps_per_day = 48
    edges_by_day: list[tuple[list[str], list[tuple[int, float]]]] = []
    for row in blocks.iter_rows(named=True):
        site, day = row["site"], row["local_date"]
        positions = grid.positions_of(day=day)
        own_scale = rules.day_before_spreads(
            residual=arm_d[site], local_dates=grid.local_dates, days=[day]
        )[day]
        edges = rules.step_candidates(residual=arm_d[site], positions=positions, scale=own_scale)
        metered = sorted(n for n in link[site] if n in bg2)
        edges_by_day.append((metered, [(e.position, e.shift) for e in edges]))
    rates = []
    for _ in range(N_SHIFT_DRAWS):
        hit_days = 0
        shifts = {n: int(rng.integers(*SHIFT_DAYS)) * steps_per_day for n in bg2}
        for metered, edges in edges_by_day:
            hit = False
            for n in metered:
                shifted = np.roll(bg2[n], shifts[n])
                shifted_scale = np.roll(scale[n], shifts[n])
                if any(
                    neighbour_hit(residual=shifted, scale=shifted_scale, edge_position=p, sign=s)
                    for p, s in edges
                ):
                    hit = True
                    break
            hit_days += hit
        rates.append(hit_days / max(len(edges_by_day), 1))
    arr = np.array(rates)
    print(
        f"\nChance rate of an opposite-signed neighbour step (neighbours shifted by whole days, "
        f"{N_SHIFT_DRAWS} draws): mean {arr.mean():.3f}, 95% range "
        f"{np.quantile(arr, 0.025):.3f} to {np.quantile(arr, 0.975):.3f}; "
        f"share of draws at or above the observed {observed:.3f}: {(arr >= observed).mean():.3f}"
    )


if __name__ == "__main__":
    main()
