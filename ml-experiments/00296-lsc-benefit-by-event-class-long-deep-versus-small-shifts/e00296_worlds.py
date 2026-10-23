"""Injected worlds with one shift class per event: a grid of duration by depth, constant and shaped.

Each world holds events at a single Primary substation, at least 7 days apart and away from every
logged closure. An event adds (receiver) or removes (donor) a share of the site's load: the share
is `depth` times the site's median power in the rows' period (constant), or `depth` times the site's
own power at each half-hour (demand-shaped). The worlds are the only shifts this experiment scores,
so they cover shifts under 25% of load and under a day, which experiment 141's worlds do not.

    uv run --no-sync python e00296_worlds.py
"""

from __future__ import annotations

import json
from typing import Final

import e00296_fed as fed
import numpy as np
import polars as pl
from e00296_common import BLOCK, GRID_LENGTH, WORK_DIR, load_panel, near_log_masks

WORLDS: Final = WORK_DIR / "worlds"
FIRST_STAMP: Final[int] = 45282 + 7 * BLOCK
"""Seven days after the first replicate row's target time."""
LAST_STAMP: Final[int] = 78813
QUIET_STAMPS: Final[int] = 7 * BLOCK
EVENTS_PER_SITE_CELL: Final[int] = 4
"""Four events per site, so 24 per cell across the six sites."""
DEPTHS: Final[tuple[float, ...]] = (0.05, 0.10, 0.25, 0.50)
DURATION_BINS: Final[tuple[tuple[str, float, float], ...]] = (
    ("2-6 h", 2.0, 6.0),
    ("6-24 h", 6.0, 24.0),
    ("1-3 d", 24.0, 72.0),
    ("3-7 d", 72.0, 168.0),
    ("7-30 d", 168.0, 720.0),
    ("over 30 d", 720.0, 1440.0),
)
SHAPES: Final[tuple[str, ...]] = ("constant", "demand-shaped")
SITE_SERIES: Final[dict[str, int]] = fed.SERIES_OF_SITE
SEED: Final[int] = 296_000


def draw_site_events(*, site: str, rng: np.random.Generator) -> list[dict]:
    """Every event of one site: class, sign and length, not yet placed."""
    events = []
    for shape in SHAPES:
        for depth in DEPTHS:
            for name, low, high in DURATION_BINS:
                for k in range(EVENTS_PER_SITE_CELL):
                    hours = float(rng.uniform(low, high))
                    events.append(
                        {
                            "site": site,
                            "shape": shape,
                            "depth": depth,
                            "duration_bin": name,
                            "hours": hours,
                            "length": max(4, round(hours * 2)),
                            "sign": 1.0 if (k % 2 == 0) else -1.0,
                        }
                    )
    return events


def pack(*, events: list[dict], quiet: np.ndarray, rng: np.random.Generator) -> list[list[dict]]:
    """Place events into worlds so that no two events of one world come within 7 days."""
    order = list(rng.permutation(len(events)))
    order.sort(key=lambda i: -events[i]["length"])
    worlds: list[list[dict]] = []
    for i in order:
        event = dict(events[i])
        for world in worlds:
            for _ in range(300):
                onset = int(rng.integers(FIRST_STAMP, LAST_STAMP - event["length"] - 2 * BLOCK))
                end = onset + event["length"]
                lo, hi = onset - QUIET_STAMPS, end + QUIET_STAMPS
                if quiet[max(lo, 0) : hi].any():
                    continue
                if any(lo < o["end"] and o["onset"] < hi for o in world):
                    continue
                event.update(onset=onset, end=end)
                world.append(event)
                break
            else:
                continue
            break
        else:
            for _ in range(2000):
                onset = int(rng.integers(FIRST_STAMP, LAST_STAMP - event["length"] - 2 * BLOCK))
                end = onset + event["length"]
                if quiet[max(onset - QUIET_STAMPS, 0) : end + QUIET_STAMPS].any():
                    continue
                event.update(onset=onset, end=end)
                worlds.append([event])
                break
            else:
                raise RuntimeError(f"cannot place an event of {event['length']} stamps")
    return worlds


def main() -> None:
    """Write one npz of the moved megawatts and one table of events per site and world."""
    WORLDS.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    logged = near_log_masks(pad_stamps=QUIET_STAMPS)
    tables = []
    for si, site in enumerate(sorted(SITE_SERIES)):
        series = SITE_SERIES[site]
        rng = np.random.default_rng([SEED, si])
        power = panel["power"][series]
        median = float(np.nanmedian(power[FIRST_STAMP:LAST_STAMP]))
        fill = np.where(np.isfinite(power), power, median)
        events = draw_site_events(site=site, rng=rng)
        worlds = pack(events=events, quiet=logged[series], rng=rng)
        for wi, world in enumerate(worlds):
            delta = np.zeros(GRID_LENGTH)
            for ei, e in enumerate(world):
                stamps = np.arange(e["onset"], e["end"])
                base = median if e["shape"] == "constant" else fill[stamps]
                delta[stamps] += e["sign"] * e["depth"] * base
                e.update(world=SEED + si * 100 + wi, event=ei + 1, series=series)
            np.savez(WORLDS / f"{site}_{SEED + si * 100 + wi}.npz", delta=delta)
            tables.extend(world)
        print(site, len(events), "events in", len(worlds), "worlds; median MW", round(median, 2))
    frame = pl.DataFrame(tables)
    frame.write_parquet(WORLDS / "events.parquet")
    census = (
        frame.group_by(["shape", "duration_bin", "depth"])
        .len()
        .sort(["shape", "duration_bin", "depth"])
    )
    print(census)
    (WORLDS / "worlds.json").write_text(json.dumps(sorted(frame["world"].unique().to_list())))


if __name__ == "__main__":
    main()
