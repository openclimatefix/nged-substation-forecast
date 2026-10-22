"""Experiment 245: do the false steps of experiment 207 coincide with load shifts between Primaries?

Reads (private, stored) the monthly capacity paths of experiment 207, the arm B residual panel
that the false-alarm check of experiment 130 put on one half-hourly grid, and the alarm tables of
experiments 130 and 195. Writes the event table and the matched controls to the private folder.

Run: OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00245-lsc-false-steps-meter-regime-audit/e00245_audit.py
"""

import pickle
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
from studies.dp_calendar import bank_holiday_names
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ID: Final[str] = "00245-lsc-false-steps-meter-regime-audit"
OUT_DIR: Final = ML_EXPERIMENTS_DATA_DIR / ID
E207_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00207-dp-meter-regime-steps-e2-step-capacity-known-answer"
)
ARRAYS: Final = ML_EXPERIMENTS_DATA_DIR / "briefs" / "sw_false_alarms" / "work" / "arrays.pkl"
E130_ALARMS: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00130-ro-nc-detector-on-the-gam-k-residual" / "B" / "alarms.parquet"
)
E195_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00195-switching-neural-plus-detector-features"

GRID_START: Final = datetime(2022, 1, 1, tzinfo=UTC)
GRID_LENGTH: Final[int] = 78_816  # 2022-01-01 to 2026-07-01, as experiment 130's grid
DAY: Final[int] = 48
WINDOW_HALF: Final[int] = 15 * DAY  # a month window is 30 days
SHIFT_WINDOW: Final[int] = 48  # one day either side, as the false-alarm check
OWN_Z_MIN: Final[float] = 3.0
NEIGHBOUR_Z_MIN: Final[float] = 2.0
BALANCE_BAND: Final[tuple[float, float]] = (0.4, 2.5)
NEIGHBOUR_HALF: Final[int] = 12  # +/- 6 hours
ALARM_HALF: Final[int] = 48  # +/- 24 hours
SIZE_MATCH: Final[float] = 0.3
N_CONTROLS: Final[int] = 200
PLACEBO_STEP_MW: Final[float] = 0.5
PLACEBO_YEARS: Final[tuple[int, ...]] = (2022, 2023, 2024)
CONTROL_YEARS: Final[tuple[int, ...]] = (2022, 2023, 2024, 2025)
MIN_PRESENT: Final[float] = 0.75
TUNING_MONTHS: Final[tuple[int, ...]] = (1, 3, 5, 7, 9, 11)
ADJACENCY: Final[dict[int, tuple[int, ...]]] = {
    3: (15, 16, 26),
    7: (24,),
    8: (10, 14, 24),
    9: (11, 24),
    10: (8, 12, 13, 14, 24, 25),
    11: (9, 25),
    12: (10, 13, 14, 16, 25),
    13: (10, 12, 14, 16, 25),
    14: (8, 10, 12, 13, 15, 16),
    15: (3, 14, 17, 18),
    16: (3, 12, 13, 14, 26),
    17: (15, 18, 26),
    18: (15, 17),
    24: (7, 8, 9, 10, 25),
    25: (10, 11, 12, 13, 24),
    26: (3, 16, 17),
}  # copied from experiment 130
SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")


def level_shift(*, values: np.ndarray, window: int) -> np.ndarray:
    """Return, at each stamp, mean of the next `window` stamps minus mean of the previous ones."""
    total = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values))])
    count = np.concatenate([[0], np.cumsum(np.isfinite(values))])
    t = np.arange(GRID_LENGTH)
    lo, hi = np.clip(t - window, 0, GRID_LENGTH), np.clip(t + window, 0, GRID_LENGTH)
    n_before, n_after = count[t] - count[lo], count[hi] - count[t]
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = (total[hi] - total[t]) / n_after - (total[t] - total[lo]) / n_before
    shift[(n_before < MIN_PRESENT * window) | (n_after < MIN_PRESENT * window)] = np.nan
    return shift


def robust_scale(*, shift: np.ndarray) -> float:
    """Return 1.4826 times the MAD of the shift over odd months."""
    stamps = pl.datetime_range(
        GRID_START, GRID_START + timedelta(minutes=30 * (GRID_LENGTH - 1)), "30m", eager=True
    )
    months = stamps.dt.month().to_numpy()
    selected = shift[np.isin(months, TUNING_MONTHS) & np.isfinite(shift)]
    return float(1.4826 * np.median(np.abs(selected - np.median(selected))))


def stamp_of(*, year: int, month: int) -> int:
    """Index of 00:00 UTC on the first of the month on the grid."""
    return int((datetime(year, month, 1, tzinfo=UTC) - GRID_START).total_seconds() // 1800)


def false_step_events() -> pl.DataFrame:
    """Return experiment 207's false steps (step path, weight 0.3, no injection), one row each.

    The rule is experiment 207's: a change above 0.5 MW between two summer months of a year, or
    the net change from the previous August to the April of the year (`kind` "winter"). One row
    per site, year, and boundary, with the number of the 5 row sets that show it.
    """
    paths = pl.concat(
        [pl.read_parquet(f) for f in sorted((E207_DIR / "work").glob("paths_*.parquet"))]
    )
    base = paths.filter(pl.col("target") == 0, pl.col("arm") == "step_0.3")
    rows = []
    for (site, farm), group in base.group_by("site", "farm"):
        ordered = group.sort("month")
        scale = float(ordered["scale_mw_per_point"][0])
        key = {
            (1970 + int(m) // 12, int(m) % 12 + 1): float(v)
            for m, v in zip(ordered["month"], ordered["level"], strict=True)
        }
        for y in PLACEBO_YEARS:
            for m in (5, 6, 7, 8):
                if (y, m) in key and (y, m - 1) in key:
                    change = (key[y, m] - key[y, m - 1]) * scale
                    if abs(change) > PLACEBO_STEP_MW:
                        rows.append((site, farm, y, m, change, "summer"))
            if (y, 4) in key and (y - 1, 8) in key:
                change = (key[y, 4] - key[y - 1, 8]) * scale
                if abs(change) > PLACEBO_STEP_MW:
                    rows.append((site, farm, y, 4, change, "winter"))
    frame = pl.DataFrame(
        rows, schema=["site", "farm", "year", "month", "change_mw", "kind"], orient="row"
    )
    return (
        frame.group_by("site", "year", "month", "kind")
        .agg(n_row_sets=pl.len(), change_mw=pl.col("change_mw").median())
        .sort("site", "year", "month")
    )


class Panel:
    """Residual steps, readings and alarms on experiment 130's grid."""

    def __init__(self) -> None:
        """Load the panel and the alarm tables."""
        arrays = pickle.loads(ARRAYS.read_bytes())
        self.site_series = {d["site"]: s for s, d in arrays.items()}
        self.shift: dict[int, np.ndarray] = {}
        self.scale: dict[int, float] = {}
        self.raw: dict[int, np.ndarray] = {}
        self.reason: dict[int, np.ndarray] = {}
        for s, d in arrays.items():
            if d["resid"] is None or not np.isfinite(d["resid"]).any():
                continue
            self.shift[s] = level_shift(values=d["resid"], window=SHIFT_WINDOW)
            self.scale[s] = robust_scale(shift=self.shift[s])
            self.raw[s], self.reason[s] = d["raw"], d["reason"]
        ro = pl.read_parquet(E130_ALARMS).filter(pl.col("setting") == "XL")
        p195 = pl.concat([pl.read_parquet(f) for f in sorted(E195_DIR.glob("alarms_f*.parquet"))])
        self.alarms = {
            "RO-NC": ro,
            "Pair alarm": p195.filter(pl.col("arm") == "Pair alarm", pl.col("source") == "p"),
            "Learned score": p195.filter(pl.col("arm") == "F logistic"),
        }
        self.alarm_stamps = {
            name: {s: np.unique(g["stamp"].to_numpy()) for (s,), g in frame.group_by("series")}
            for name, frame in self.alarms.items()
        }
        holidays = set(bank_holiday_names())
        days = np.array(
            [(GRID_START + timedelta(minutes=30 * i)).date() for i in range(0, GRID_LENGTH, DAY)]
        )
        self.holiday = np.repeat(np.array([d in holidays for d in days]), DAY)[:GRID_LENGTH]

    def window_tags(self, *, series: int, centre: int) -> dict:
        """Score one 30-day window at a Primary: own step, balanced neighbour, alarms, tags."""
        lo, hi = max(centre - WINDOW_HALF, 0), min(centre + WINDOW_HALF, GRID_LENGTH)
        z = self.shift[series][lo:hi] / self.scale[series]
        out = {"own_z": np.nan, "own_mw": np.nan, "t0": -1}
        if not np.isfinite(z).any():
            return out | {"visible": False, "balanced": False, "any_alarm": False}
        k = int(np.nanargmax(np.abs(z)))
        t0 = lo + k
        own_mw = float(self.shift[series][t0])
        sign = 1.0 if own_mw > 0 else -1.0
        out |= {
            "own_z": float(z[k]),
            "own_mw": own_mw,
            "t0": t0,
            "visible": bool(abs(z[k]) >= OWN_Z_MIN),
        }
        balanced, best_opp = False, 0.0
        for n in ADJACENCY.get(series, ()):
            if n not in self.shift:
                continue
            seg = self.shift[n][max(t0 - NEIGHBOUR_HALF, 0) : t0 + NEIGHBOUR_HALF + 1]
            if not np.isfinite(seg).any():
                continue
            opp_z = float(np.nanmax(-sign * seg) / self.scale[n])
            best_opp = max(best_opp, opp_z)
            ratio = float(np.nanmax(-sign * seg)) / abs(own_mw) if own_mw else np.nan
            if opp_z >= NEIGHBOUR_Z_MIN and BALANCE_BAND[0] <= ratio <= BALANCE_BAND[1]:
                balanced = True
        out["balanced"] = bool(balanced and out["visible"])
        out["best_opp_z"] = best_opp
        anyalarm = False
        for name, by_series in self.alarm_stamps.items():
            stamps = by_series.get(series, np.array([], dtype=int))
            hit = bool(len(stamps) and np.abs(stamps - t0).min() <= ALARM_HALF)
            out[f"alarm_{name}"] = hit
            hit_wide = bool(len(stamps) and ((stamps >= lo) & (stamps < hi)).any())
            out[f"wide_{name}"] = hit_wide
            anyalarm |= hit
        out["any_alarm"] = bool(anyalarm and out["visible"])
        wide = out["visible"] and any(out[f"wide_{n}"] for n in self.alarm_stamps)
        out["any_wide"] = bool(wide)
        a, b = max(t0 - DAY, 0), min(t0 + DAY, GRID_LENGTH)
        raw, reason = self.raw[series][a:b], self.reason[series][a:b]
        out["flagged_or_zero"] = bool((reason != "").any() or (raw == 0).any())
        out["bank_holiday"] = bool(self.holiday[a:b].any())
        return out


def main() -> None:
    """Write the event table and the matched controls."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events = false_step_events()
    panel = Panel()
    rng = np.random.default_rng(245)
    event_rows, control_rows = [], []
    localised = events.filter(pl.col("kind") == "summer")
    for e in events.iter_rows(named=True):
        series = panel.site_series[e["site"]]
        if e["kind"] == "winter":
            event_rows.append({**e, "series": series, "explained": None})
            continue
        centre = stamp_of(year=e["year"], month=e["month"])
        tags = panel.window_tags(series=series, centre=centre)
        explained = bool(tags["balanced"] or tags["any_alarm"])
        event_rows.append({**e, "series": series, "centre": centre, "explained": explained, **tags})
    # matched controls
    own_boundaries = {
        site: [
            stamp_of(year=r["year"], month=r["month"])
            for r in localised.filter(pl.col("site") == site).iter_rows(named=True)
        ]
        for site in SITES
    }
    for ev in event_rows:
        if ev["kind"] != "summer":
            continue
        series = ev["series"]
        cands = []
        for y in CONTROL_YEARS:
            base = stamp_of(year=y, month=ev["month"])
            for k in range(-15, 16):
                c = base + k * DAY
                if all(abs(c - b) > 31 * DAY for b in own_boundaries[ev["site"]]):
                    cands.append(c)
        rng.shuffle(cands)
        target = abs(ev["own_mw"]) if np.isfinite(ev["own_mw"]) else np.nan
        kept = []
        for c in cands:
            if c + WINDOW_HALF >= GRID_LENGTH:
                continue
            t = panel.window_tags(series=series, centre=c)
            if not np.isfinite(t["own_mw"]):
                continue
            if abs(abs(t["own_mw"]) - target) <= SIZE_MATCH * target:
                kept.append(
                    {
                        "site": ev["site"],
                        "year": ev["year"],
                        "month": ev["month"],
                        "centre": c,
                        "explained": bool(t["balanced"] or t["any_alarm"]),
                        **t,
                    }
                )
            if len(kept) >= N_CONTROLS:
                break
        control_rows.extend(kept)
        ev["n_controls"] = len(kept)
    ev_frame = pl.DataFrame(event_rows, infer_schema_length=None)
    ev_frame.write_parquet(OUT_DIR / "events.parquet")
    pl.DataFrame(control_rows, infer_schema_length=None).write_parquet(OUT_DIR / "controls.parquet")
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    print(ev_frame)


if __name__ == "__main__":
    main()
