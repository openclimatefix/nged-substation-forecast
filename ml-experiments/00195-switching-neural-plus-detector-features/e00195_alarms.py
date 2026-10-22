# ruff: noqa: E501
"""Run a trained window network over the whole record and turn its outputs into alarms.

An onset alarm is a window whose onset probability passes the threshold, located at the peak of its
onset map; alarms of one series within `NMS` half-hours are merged (highest probability kept). End
alarms are built the same way from the end head. An onset alarm takes the earliest end alarm after
it on the same series, within 30 days, as its end (a hindsight rule: the end alarm needs the 5 days
after the end that the window can hold).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
import torch
from e00195_common import GRID_LENGTH, SERIES_IDS, TUNING_MONTHS, WINDOW, WORK_DIR, stamp_months
from e00195_net import REGION, Context, WindowNet, window_inputs

STRIDE: Final[int] = 12
"""Half-hours between window origins (6 hours)."""
NMS: Final[int] = 8
MAX_EVENT: Final[int] = 30 * 48
MIN_P: Final[float] = 0.02
BUDGET: Final[int] = 80
ONSET_TOLERANCE: Final[int] = 4
END_MINIMUM: Final[int] = 12
END_FRACTION: Final[float] = 0.10


def load_model(*, fold: int, seed: int = 0, device: str = "cpu") -> WindowNet:
    """The best-epoch weights of one fold's network."""
    model = WindowNet()
    model.load_state_dict(torch.load(WORK_DIR / "models" / f"fold{fold}_seed{seed}.pt"))
    return model.to(device).eval()


@dataclass
class Candidates:
    """Every merged onset and end candidate over the record, before any threshold."""

    onset: pl.DataFrame
    """Columns series, stamp, p, explained, odd."""
    end: pl.DataFrame


def _merge(*, rows: list[tuple[int, int, float]]) -> pl.DataFrame:
    """Keep the highest-probability candidate within `NMS` half-hours on each series."""
    table = pl.DataFrame(
        rows, schema={"series": pl.Int64, "stamp": pl.Int64, "p": pl.Float64}, orient="row"
    ).sort("p", descending=True)
    kept: dict[int, list[int]] = {}
    keep_rows = []
    for series, stamp, p in table.iter_rows():
        if any(abs(stamp - k) <= NMS for k in kept.setdefault(series, [])):
            continue
        kept[series].append(stamp)
        keep_rows.append((series, stamp, p))
    return pl.DataFrame(
        keep_rows, schema={"series": pl.Int64, "stamp": pl.Int64, "p": pl.Float64}, orient="row"
    )


def candidates(
    *, model: WindowNet, ctx: Context, z: np.ndarray | None = None, device: str = "cpu"
) -> Candidates:
    """Run the network over every series' windows (stride 6 hours) and merge the peaks."""
    z = ctx.z if z is None else z
    onset_rows: list[tuple[int, int, float]] = []
    end_rows: list[tuple[int, int, float]] = []
    starts_all = np.arange(0, GRID_LENGTH - WINDOW + 1, STRIDE)
    for i, series in enumerate(SERIES_IDS):
        finite = np.concatenate([[0], np.cumsum(np.isfinite(z[i]))])
        share = (finite[starts_all + WINDOW] - finite[starts_all]) / WINDOW
        starts = starts_all[share >= 0.2]
        for chunk in np.array_split(starts, max(len(starts) // 1500, 1)):
            if not len(chunk):
                continue
            x = window_inputs(ctx=ctx, targets=np.full(len(chunk), i), starts=chunk, z=z)
            with torch.no_grad():
                alarms, maps = model(torch.tensor(x, device=device))
            p = torch.sigmoid(alarms).cpu().numpy()
            maps = maps.cpu().numpy()
            on = maps[:, 0, REGION[0] : REGION[1]].argmax(axis=1) + REGION[0]
            en = maps[:, 1, REGION[0] : REGION[1]].argmax(axis=1) + REGION[0]
            for k, start in enumerate(chunk):
                if p[k, 0] >= MIN_P:
                    onset_rows.append((series, int(start + on[k]), float(p[k, 0])))
                if p[k, 1] >= MIN_P:
                    end_rows.append((series, int(start + en[k]), float(p[k, 1])))
    months = stamp_months()
    index = {s: i for i, s in enumerate(SERIES_IDS)}

    def decorate(table: pl.DataFrame) -> pl.DataFrame:
        stamps = table["stamp"].to_numpy()
        explained = np.array(
            [ctx.explained[index[s], t] for s, t in zip(table["series"], stamps, strict=True)]
        )
        odd = np.isin(months[np.clip(stamps, 0, GRID_LENGTH - 1)], list(TUNING_MONTHS))
        return table.with_columns(explained=pl.Series(explained), odd=pl.Series(odd))

    return Candidates(onset=decorate(_merge(rows=onset_rows)), end=decorate(_merge(rows=end_rows)))


def threshold_for_budget(*, table: pl.DataFrame, budget: int = BUDGET) -> float:
    """Probability above which exactly `budget` unexplained odd-month candidates remain."""
    pool = table.filter(pl.col("odd"), ~pl.col("explained"))["p"].to_numpy()
    ordered = np.sort(pool)[::-1]
    return float(ordered[budget]) + 1e-12 if len(ordered) > budget else MIN_P


def end_tolerance(*, onset: int, end: int) -> float:
    """The allowed end error: the larger of 6 hours and 10% of the duration, in half-hours."""
    return max(float(END_MINIMUM), END_FRACTION * (end - onset))


def alarm_intervals(
    *, found: Candidates, onset_threshold: float, end_threshold: float
) -> dict[int, list[tuple[int, int | None, float]]]:
    """Per series, (onset, end or None, p) for each onset alarm above the threshold."""
    ends = found.end.filter(pl.col("p") > end_threshold)
    ends_by = {s: np.sort(g["stamp"].to_numpy()) for (s,), g in ends.group_by("series")}
    out: dict[int, list[tuple[int, int | None, float]]] = {}
    for series, stamp, p in (
        found.onset.filter(pl.col("p") > onset_threshold).select("series", "stamp", "p").iter_rows()
    ):
        later = ends_by.get(series, np.array([], dtype=int))
        later = later[(later > stamp + 4) & (later <= stamp + MAX_EVENT)]
        out.setdefault(series, []).append((stamp, int(later[0]) if later.size else None, p))
    return out


def match(
    *,
    intervals: dict[int, list[tuple[int, int | None, float]]],
    series: tuple[int, ...],
    onset: int,
    end: int,
) -> dict[str, float | bool]:
    """Whether an event was found (onset within 2 hours and end within tolerance) and the onset error."""
    tolerance = end_tolerance(onset=onset, end=end)
    onset_found, found, errors = False, False, []
    for s in series:
        for alarm_onset, alarm_end, _ in intervals.get(s, []):
            gap = abs(alarm_onset - onset)
            if gap <= 48:
                errors.append(gap)
            if gap <= ONSET_TOLERANCE:
                onset_found = True
                if alarm_end is not None and abs(alarm_end - end) <= tolerance:
                    found = True
    return {
        "onset_found": onset_found,
        "found": found,
        "onset_error": float(min(errors)) if errors else float("nan"),
    }
