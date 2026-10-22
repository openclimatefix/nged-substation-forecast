"""Classify the shape of the forecast's error on the 37 unlabelled worst days (experiment 175).

Reads experiment 174's day list and arm D's residuals, applies the pre-registered rules of
`e00175_classify`, checks interconnected neighbours for an opposite-signed step, and tallies the
detectors. Repeats the classifier on ordinary days and on days with injected shapes. Prints the
report and writes the tables under `WORK_DIR`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00175-*/e00175_run.py > printed_report.txt
"""

import math
import os
from dataclasses import dataclass
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
N_ORDINARY_PER_SITE: Final[int] = 40
ORDINARY_QUANTILES: Final[tuple[float, float]] = (0.25, 0.75)
N_INJECTED_PER_CLASS: Final[int] = 40
STEPS_PER_DAY: Final[int] = 48
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)


@dataclass(frozen=True)
class Context:
    """Everything the per-day analysis reads.

    Attributes:
        grid: The half-hour grid and arm D's residuals.
        bg2: BG2's residual per D-label.
        bg2_scale: The day-before spread at every grid position, per BG2 D-label.
        link: Each D-label's interconnected D-labels.
        ids: Each D-label's time-series number.
        detectors: Stored onsets, per detector table.
    """

    grid: common.ResidualGrid
    bg2: dict[str, np.ndarray]
    bg2_scale: dict[str, np.ndarray]
    link: dict[str, set[str]]
    ids: dict[str, int]
    detectors: dict[str, pl.DataFrame]


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
    return lookup[np.searchsorted(np.array(days), local_dates)]


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


def fired(*, ctx: Context, table: str, series: int, positions: range) -> bool:
    """Return whether a stored onset for `series` lies at one of the grid `positions`."""
    onsets = ctx.detectors[table].filter(pl.col("series") == series)["onset"].to_list()
    wanted = set(positions)
    return any(ctx.grid.position_of_stamp(stamp=int(o)) in wanted for o in onsets)


def neighbour_steps(*, ctx: Context, site: str, edges: list[rules.Edge]) -> tuple[bool, list[str]]:
    """Return whether any metered neighbour steps against any edge, and which neighbours."""
    metered = sorted(n for n in ctx.link[site] if n in ctx.bg2)
    hits = [
        n
        for n in metered
        if any(
            rules.neighbour_opposite_step(
                residual=ctx.bg2[n],
                edge_position=e.position,
                scale=float(ctx.bg2_scale[n][e.position]),
                sign=e.shift,
            )
            for e in edges
        )
    ]
    return bool(hits), hits


def analyse_day(*, ctx: Context, residual: np.ndarray, site: str, day: date, scale: float) -> dict:
    """Classify one (site, day); for a block, add the neighbour and detector checks."""
    grid = ctx.grid
    positions = grid.positions_of(day=day)
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
        "midday_part": rules.midday_share(
            day=residual[positions], hours=grid.local_hours[positions]
        ),
    }
    if edges:
        primary = max(edges, key=lambda edge: edge.strength)
        metered = [n for n in ctx.link[site] if n in ctx.bg2]
        hit, which = neighbour_steps(ctx=ctx, site=site, edges=edges)
        near = range(primary.position - DETECTOR_WINDOW_READINGS, primary.position + 7)
        row |= {
            "edge_position": primary.position,
            "edge_shift": primary.shift,
            "edge_strength": primary.strength,
            "returns": rules.returns_within_day(primary=primary, edges=edges),
            "persists_7d": rules.persists(
                residual=residual, local_dates=grid.local_dates, day=day, shift=primary.shift
            ),
            "n_neighbours": len(ctx.link[site]),
            "n_metered_neighbours": len(metered),
            "neighbour_step": hit if metered else None,
            "neighbours_with_step": ",".join(which),
        }
        for name in ctx.detectors:
            row[f"fired_{name}"] = fired(ctx=ctx, table=name, series=ctx.ids[site], positions=near)
    return row


def run_days(*, ctx: Context, days: pl.DataFrame, residuals: dict[str, np.ndarray]) -> pl.DataFrame:
    """Analyse every (site, local_date) of `days` against `residuals`."""
    rows = []
    for site in sorted(days["site"].unique().to_list()):
        site_days: list[date] = days.filter(pl.col("site") == site)["local_date"].to_list()
        spreads = rules.day_before_spreads(
            residual=residuals[site], local_dates=ctx.grid.local_dates, days=site_days
        )
        rows.extend(
            analyse_day(ctx=ctx, residual=residuals[site], site=site, day=d, scale=spreads[d])
            for d in site_days
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort(["site", "local_date"])


def ordinary_days(*, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return 40 random days per site from the middle half of its days by excess loss."""
    rng = np.random.default_rng(common.SEED)
    frames = []
    for site in sites:
        table = common.day_excess(site=site)
        low, high = (
            table["excess"].quantile(ORDINARY_QUANTILES[0]),
            table["excess"].quantile(ORDINARY_QUANTILES[1]),
        )
        pool = table.filter(pl.col("excess").is_between(low, high))["local_date"].to_list()
        chosen = rng.choice(len(pool), size=min(N_ORDINARY_PER_SITE, len(pool)), replace=False)
        frames.append(pl.DataFrame({"site": site, "local_date": [pool[i] for i in sorted(chosen)]}))
    return pl.concat(frames)


def injected_control(
    *, ctx: Context, ordinary: pl.DataFrame, residuals: dict[str, np.ndarray]
) -> pl.DataFrame:
    """Inject one shape into ordinary days and return the share classified as intended."""
    rng = np.random.default_rng(common.SEED + 1)
    out = []
    for intended in ("a_block", "b_offset", "c_ramp_or_midday", "d_spike"):
        pool = ordinary.sample(n=min(N_INJECTED_PER_CLASS, ordinary.height), seed=common.SEED)
        hits = 0
        for site, day in pool.iter_rows():
            residual = residuals[site].copy()
            scale = rules.day_before_spreads(
                residual=residual, local_dates=ctx.grid.local_dates, days=[day]
            )[day]
            positions = ctx.grid.positions_of(day=day)
            if intended == "a_block":
                start = int(rng.integers(8, 28))
                residual[positions[start : start + 12]] += 3.0 * scale
            elif intended == "b_offset":
                residual[positions] += 2.0 * scale
            elif intended == "c_ramp_or_midday":
                residual[positions] += np.linspace(0.0, 4.0 * scale, positions.size)
            else:
                start = int(rng.integers(8, 40))
                residual[positions[start : start + 3]] += 6.0 * scale
            row = analyse_day(ctx=ctx, residual=residual, site=site, day=day, scale=scale)
            hits += row["shape"] == intended
        out.append({"injected": intended, "n": pool.height, "classified_as_intended": hits})
    return pl.DataFrame(out)


def shape_counts(*, table: pl.DataFrame, title: str) -> None:
    """Print each shape's count with its Wilson interval."""
    n = table.height
    print(f"\n{title}: {n} days")
    for shape, count in table.group_by("shape").len().sort("shape").iter_rows():
        low, high = wilson(hits=count, n=n)
        print(f"  {shape}: {count} ({count / n:.3f}, 95% Wilson {low:.3f} to {high:.3f})")


def block_summary(*, blocks: pl.DataFrame, label: str) -> float:
    """Print the neighbour, detector, return, and persistence tallies; return the neighbour rate."""
    n_a = blocks.height
    with_neighbour = blocks.filter(pl.col("n_metered_neighbours") > 0)
    hits = int(with_neighbour["neighbour_step"].sum())
    rate = hits / max(with_neighbour.height, 1)
    low, high = wilson(hits=hits, n=with_neighbour.height)
    print(
        f"\n{label}: class (a) days {n_a}; with a metered neighbour {with_neighbour.height}; "
        f"opposite-signed neighbour step on {hits} ({rate:.3f}, Wilson {low:.3f} to {high:.3f})"
    )
    for name in [c for c in blocks.columns if c.startswith("fired_")]:
        print(f"  {name}: {int(blocks[name].sum())} of {n_a}")
    print(f"  returns within the day: {int(blocks['returns'].sum())} of {n_a}")
    persist = blocks["persists_7d"]
    print(
        f"  persists over the next 7 days: {int((persist == True).sum())}; "  # noqa: E712
        f"reverts: {int((persist == False).sum())}; unknown: {persist.null_count()}"  # noqa: E712
    )
    return rate


def chance_rate(*, ctx: Context, blocks: pl.DataFrame, residuals: dict[str, np.ndarray]) -> None:
    """Print the rate of an opposite-signed neighbour step when each neighbour is rolled by days."""
    rng = np.random.default_rng(common.SEED + 2)
    cases = []
    for site, day in blocks.select("site", "local_date").iter_rows():
        scale = rules.day_before_spreads(
            residual=residuals[site], local_dates=ctx.grid.local_dates, days=[day]
        )[day]
        edges = rules.step_candidates(
            residual=residuals[site], positions=ctx.grid.positions_of(day=day), scale=scale
        )
        metered = sorted(n for n in ctx.link[site] if n in ctx.bg2)
        if metered:
            cases.append((metered, edges))
    rates = []
    for _ in range(N_SHIFT_DRAWS):
        shifts = {
            n: int(rng.integers(SHIFT_DAYS[0], SHIFT_DAYS[1] + 1)) * STEPS_PER_DAY for n in ctx.bg2
        }
        hit_days = sum(
            any(
                rules.neighbour_opposite_step(
                    residual=np.roll(ctx.bg2[n], shifts[n]),
                    edge_position=e.position,
                    scale=float(np.roll(ctx.bg2_scale[n], shifts[n])[e.position]),
                    sign=e.shift,
                )
                for n in metered
                for e in edges
            )
            for metered, edges in cases
        )
        rates.append(hit_days / max(len(cases), 1))
    arr = np.array(rates)
    pl.DataFrame({"rate": arr}).write_parquet(common.WORK_DIR / "chance.parquet")
    print(
        f"\nChance rate of an opposite-signed neighbour step (each neighbour rolled by whole "
        f"days, {N_SHIFT_DRAWS} draws, {len(cases)} days): mean {arr.mean():.3f}, "
        f"95% range {np.quantile(arr, 0.025):.3f} to {np.quantile(arr, 0.975):.3f}"
    )


def main() -> None:
    """Run the classification, the neighbour check, the controls, and the tallies."""
    days = common.unlabelled_days()
    sites = tuple(sorted(days["site"].unique().to_list()))
    residuals = common.arm_d_residuals(sites=sites)
    bg2 = common.bg2_residuals()
    grid = common.ResidualGrid(residual=residuals)
    ctx = Context(
        grid=grid,
        bg2=bg2,
        bg2_scale={
            label: scale_by_position(residual=bg2[label], local_dates=grid.local_dates)
            for label in bg2
        },
        link=common.neighbours(),
        ids=common.series_ids(),
        detectors=detector_tables(),
    )
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)

    for name, onsets in ctx.detectors.items():
        for site in sites:
            n_onsets = onsets.filter(pl.col("series") == ctx.ids[site]).height
            print(f"{name} onsets for {site} over the whole stored history: {n_onsets}")
    table = run_days(ctx=ctx, days=days, residuals=residuals)
    table.write_parquet(common.WORK_DIR / "day_shapes.parquet")
    print(table)
    shape_counts(table=table, title="Unlabelled worst days")
    print("\nBy site:")
    print(
        table.group_by(["site", "shape"])
        .len()
        .pivot(on="shape", index="site", values="len")
        .fill_null(0)
        .sort("site")
    )
    midday = table.filter(pl.col("shape") == "c_ramp_or_midday")
    print(
        f"Class (c) days with at least half the squared residual 10:00-16:00: "
        f"{int((midday['midday_part'] >= 0.5).sum())} of {midday.height}"
    )
    observed = block_summary(blocks=table.filter(pl.col("shape") == "a_block"), label="Unlabelled")
    print(f"Rule: class (a) at least {math.ceil(table.height / 2)} of {table.height}")
    chance_rate(ctx=ctx, blocks=table.filter(pl.col("shape") == "a_block"), residuals=residuals)

    ordinary = ordinary_days(sites=sites)
    ordinary_table = run_days(ctx=ctx, days=ordinary, residuals=residuals)
    ordinary_table.write_parquet(common.WORK_DIR / "ordinary_day_shapes.parquet")
    shape_counts(table=ordinary_table, title="Ordinary days (middle half by excess loss)")
    for name in ctx.detectors:
        with_onset = sum(
            fired(
                ctx=ctx,
                table=name,
                series=ctx.ids[site],
                positions=range(
                    int(ctx.grid.positions_of(day=day)[0]) - DETECTOR_WINDOW_READINGS,
                    int(ctx.grid.positions_of(day=day)[-1]) + DETECTOR_WINDOW_READINGS + 1,
                ),
            )
            for site, day in ordinary.iter_rows()
        )
        print(f"Ordinary days with a {name} onset within 3 hours of the day: {with_onset} of 200")
    block_summary(blocks=ordinary_table.filter(pl.col("shape") == "a_block"), label="Ordinary")
    print("\nKnown-answer control (shape injected into ordinary days):")
    print(injected_control(ctx=ctx, ordinary=ordinary, residuals=residuals))
    print(f"\nObserved neighbour rate on unlabelled class (a) days: {observed:.3f}")
    print("Experiment 172's known-answer control recovered 89.2% of injected days (bar 90%).")


if __name__ == "__main__":
    main()
