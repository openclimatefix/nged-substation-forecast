# ruff: noqa: E501, PLR0915
"""RO-NC's candidates and score, the pair-alarm statistic, and the features of every candidate.

Copies the pieces of experiments 82, 130, and 156 that this experiment needs, so that the scripts
import only their own folder. A candidate is a local peak of a step statistic of one series' residual
(a difference of the mean residual in the next window and the previous one). Nothing here reads a
switching log or a label; `explained` and `d8_drop` are flags attached for the callers to use.
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00249_common import (
    ADJACENCY,
    BLANK_SERIES,
    D8_DROPS,
    DAY,
    GAM_PANEL,
    GRID_LENGTH,
    ML_EXPERIMENTS_DATA_DIR,
    TUNING_MONTHS,
    explained_masks,
    holiday_flag,
    stamp_hour_weekday,
    stamp_index,
    stamp_months,
)
from scipy.ndimage import maximum_filter1d

FEATURES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00006-stage1a-demand-input-feature-sweep"
    / "inputs"
    / "features.parquet"
)
FITS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00071-sw2-18-refit-offset-and-gap-bridging" / "offset_fits"
)
MODELS_82: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00082-sw2-18-19-combined-refit-offset-and-same-sign-count"
    / "models.pkl"
)
METADATA_PATH: Final[Path] = ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "metadata.parquet"

W2, W4, W6, W24 = 4, 8, 12, 48
"""Window lengths in half-hours: 2, 4, 6, and 24 hours."""
MIN_PRESENT: Final[float] = 0.75
CANDIDATE_Z: Final[float] = 2.5
"""RO-NC's candidate threshold in noise units (experiment 130); the loosened list the re-ranker reads."""
PAIR_Z: Final[float] = 3.0
PAIR_PARTNER_Z: Final[float] = 2.0
NEIGHBOUR_Z: Final[float] = 2.0
SAME_SIGN_Z: Final[float] = 2.0
STRONG_Z: Final[float] = 4.0
CAP_DAYS: Final[int] = 14
RETURN_Z_MIN: Final[float] = 1.5
FIRST_SHARE: Final[float] = 0.5
TIE_BREAK: Final[float] = 1e-3
RAW_BEFORE: Final[int] = 2 * DAY
RAW_AFTER: Final[int] = 4 * DAY
RAW_LENGTH: Final[int] = RAW_BEFORE + RAW_AFTER
RAW_CLIP: Final[float] = 15.0
Arrays = dict[int, np.ndarray]

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "own_z2",
    "own_z4",
    "own_z6",
    "own_z24",
    "persist",
    "step_mw24",
    "opp_z24",
    "opp_z6",
    "opp_offset_h",
    "opp_ratio",
    "imbalance",
    "n_opposite",
    "n_same",
    "n_neighbours",
    "n_metered_opposite",
    "mva_neighbour",
    "log_sigma_mw",
    "refit_offset",
    "hour_sin",
    "hour_cos",
    "season_sin",
    "season_cos",
    "weekend",
    "holiday",
    "recent_alarms",
)
"""Everything the learned scores may read; no series identifier and no Primary."""
import os  # noqa: E402

_DROP = {x for x in os.environ.get("E195_DROP", "").split(",") if x}
if _DROP:
    FEATURE_NAMES = tuple(f for f in FEATURE_NAMES if f not in _DROP)  # ty: ignore[invalid-assignment]
"""The post hoc ablation sets E195_DROP to a comma-separated list of features to remove."""
RONC_FEATURES: Final[tuple[str, ...]] = ("x1", "x2", "n_neighbours", "mva_neighbour", "n_same")


def level_shift(*, values: np.ndarray, window: int) -> np.ndarray:
    """Mean of `values[t:t+window]` minus mean of `values[t-window:t]`; NaN where under 75% present."""
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
    """1.4826 times the MAD of `shift` over odd (tuning) months."""
    months = stamp_months()
    selected = shift[np.isin(months, list(TUNING_MONTHS)) & np.isfinite(shift)]
    return float(1.4826 * np.median(np.abs(selected - np.median(selected))))


@cache
def refit_zone() -> np.ndarray:
    """True within 24 hours of the monthly refit of the first residual model (00:00 UTC on the 1st)."""
    index = stamp_index()
    first = np.flatnonzero(
        (index.dt.day().to_numpy() == 1)
        & (index.dt.hour().to_numpy() == 0)
        & (index.dt.minute().to_numpy() == 0)
    )
    zone = np.zeros(GRID_LENGTH, dtype=bool)
    for stamp in first:
        zone[max(stamp - W24 + 1, 0) : stamp + W24] = True
    return zone


def load_offset_panel() -> dict[str, Arrays]:
    """`expected` and `residual` of the convex model refitted on the 16th (experiment 130's panel_16)."""
    stored = pickle.loads((GAM_PANEL.parent / "panel_16.pkl").read_bytes())["frozen"]
    blank = np.full(GRID_LENGTH, np.nan)
    return {
        "expected": {s: (blank if s == BLANK_SERIES else a) for s, a in stored["expected"].items()},
        "residual": {s: (blank if s == BLANK_SERIES else a) for s, a in stored["residual"].items()},
    }


EXPECTED_FLOOR_SHARE: Final[float] = 0.1
PRE_STAMPS: Final[int] = 7 * DAY
CAP_STAMPS: Final[int] = 60 * DAY
MIN_BLOCK_PRESENT: Final[float] = 0.5


def relative_residual(*, residual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Residual divided by expected power, with expected floored at 10% of its median."""
    floor = EXPECTED_FLOOR_SHARE * np.nanmedian(expected)
    return residual / np.maximum(expected, floor)


def tracked_relative(
    *, panel: dict, offset_panel: dict, residual: Arrays | None = None, offset: Arrays | None = None
) -> Arrays:
    """The relative residual the level tracker reads: the second model's inside a refit zone."""
    zone = refit_zone()
    residual = panel["residual"] if residual is None else residual
    offset = offset_panel["residual"] if offset is None else offset
    out: Arrays = {}
    for series, values in residual.items():
        first = relative_residual(residual=values, expected=panel["expected"][series])
        if series in offset:
            second = relative_residual(
                residual=offset[series], expected=offset_panel["expected"][series]
            )
            first = np.where(zone & np.isfinite(second), second, first)
        out[series] = first
    return out


def _block_mean(*, q: np.ndarray, start: int) -> float:
    block = q[start : start + DAY]
    present = np.isfinite(block)
    return float(block[present].mean()) if present.mean() >= MIN_BLOCK_PRESENT else float("nan")


def level_track(*, q: np.ndarray, onset: int, window: int = W24) -> tuple[float, int]:
    """Experiment 130's level tracker: the onset step and the end stamp of the event."""
    lo = max(onset - PRE_STAMPS, 0)
    before = q[lo:onset]
    pre = float(np.nanmedian(before)) if np.isfinite(before).any() else 0.0
    after = q[onset : onset + window]
    if not np.isfinite(after).any():
        return float("nan"), onset + window
    rho = float(np.nanmean(after)) - pre
    last_good, start = onset, onset
    while start - onset < CAP_STAMPS and start < GRID_LENGTH:
        mean = _block_mean(q=q, start=start)
        if np.isfinite(mean):
            step = mean - pre
            if np.sign(step) == np.sign(rho) and abs(step) >= 0.5 * abs(rho):
                last_good = start + DAY
            else:
                break
        start += DAY
    return rho, (onset + window if last_good == onset else last_good)


@cache
def mva_series() -> frozenset[int]:
    """Series metered only in MVA (a magnitude with no direction)."""
    metadata = pl.read_parquet(METADATA_PATH, columns=["time_series_id", "units"])
    return frozenset(
        metadata.filter(pl.col("units").cast(pl.String) == "MVA")["time_series_id"].to_list()
    )


@dataclass(frozen=True)
class Scales:
    """Noise scales fixed on the uninjected residual, so injected events do not move thresholds."""

    sigma: dict[int, float]
    x24: dict[int, float]
    w: dict[int, dict[int, float]]
    """Per window (4, 8, 12): per series scale of the plain-residual step statistic."""


@dataclass
class Stats:
    """Step statistics of one residual panel (clean or injected)."""

    residual: Arrays
    shift_x: Arrays
    """24-hour step statistic with the refit offset (first model outside the refit zone, second inside it)."""
    z_first: Arrays
    z_second: Arrays
    shift_w: dict[int, Arrays]
    scales: Scales
    z_x: Arrays
    z_w: dict[int, Arrays]
    tracker: Arrays
    """Relative residual read by the level tracker."""


def make_scales(*, residual: Arrays, offset: Arrays, sigma: dict[int, float]) -> Scales:
    """Scales from the clean panels."""
    stats = _shifts(residual=residual, offset=offset)
    x24 = {s: robust_scale(shift=v) for s, v in stats["x"].items()}
    w = {win: {s: robust_scale(shift=v) for s, v in stats[win].items()} for win in (W2, W4, W6)}
    return Scales(sigma=sigma, x24=x24, w=w)


def _shifts(*, residual: Arrays, offset: Arrays) -> dict:
    zone = refit_zone()
    out: dict = {"x": {}, "first": {}, "second": {}, W2: {}, W4: {}, W6: {}}
    for s, values in residual.items():
        first = level_shift(values=values, window=W24)
        second = level_shift(values=offset.get(s, np.full(GRID_LENGTH, np.nan)), window=W24)
        out["first"][s], out["second"][s] = first, second
        out["x"][s] = np.where(zone, second, first)
        for win in (W2, W4, W6):
            out[win][s] = level_shift(values=values, window=win)
    return out


def make_stats(
    *, residual: Arrays, offset: Arrays, scales: Scales, panel: dict, offset_panel: dict
) -> Stats:
    """Step statistics of a (possibly injected) residual panel, in the clean panel's scales."""
    raw = _shifts(residual=residual, offset=offset)
    z_x = {s: v / scales.x24[s] for s, v in raw["x"].items()}
    z_w = {win: {s: v / scales.w[win][s] for s, v in raw[win].items()} for win in (W2, W4, W6)}
    return Stats(
        residual=residual,
        shift_x=raw["x"],
        z_first={s: v / scales.x24[s] for s, v in raw["first"].items()},
        z_second={s: v / scales.x24[s] for s, v in raw["second"].items()},
        shift_w={win: raw[win] for win in (W2, W4, W6)},
        scales=scales,
        z_x=z_x,
        z_w=z_w,
        tracker=tracked_relative(
            panel=panel, offset_panel=offset_panel, residual=residual, offset=offset
        ),
    )


def _peaks(*, z: np.ndarray, threshold: float, half: int) -> np.ndarray:
    """Stamps where |z| is at least `threshold` and the largest within +/- `half`."""
    magnitude = np.nan_to_num(np.abs(z))
    top = maximum_filter1d(magnitude, size=2 * half + 1, mode="constant", cval=0.0)
    return np.flatnonzero((magnitude >= threshold) & (magnitude == top))


@dataclass
class CandidateSet:
    """Candidates with features, flags, and the raw residual windows around each."""

    table: pl.DataFrame
    windows: np.ndarray
    """Shape (candidates, 4, RAW_LENGTH), float32: own z, own present, neighbour z, neighbour present."""


def _neighbour_view(*, stats: Stats, series: int, t: int, sign: float) -> dict[str, float | int]:
    """What the connected neighbours did near stamp `t`, oriented so that the own step is positive."""
    sx = stats.scales.x24
    best24, same, n_opp, n_metered_opp = 0.0, 0, 0, 0
    opp_mw_sum, best_name = 0.0, None
    best6, best6_offset = 0.0, 0.0
    neighbours = [n for n in ADJACENCY.get(series, ()) if n in stats.shift_x]
    mva = mva_series()
    for n in neighbours:
        segment = stats.shift_x[n][max(t - 1, 0) : t + 2]
        if np.isfinite(segment).any():
            opposite = float(np.nanmax(-sign * segment))
            same_step = float(np.nanmax(sign * segment))
            opp_z = opposite / sx[n]
            best24 = max(best24, opp_z)
            same += int(same_step / sx[n] > SAME_SIGN_Z)
            if opp_z > NEIGHBOUR_Z:
                n_opp += 1
                opp_mw_sum += opposite
                n_metered_opp += int(n not in mva)
                if best_name is None or opp_z > best24 - 1e-12:
                    best_name = n
        seg6 = stats.z_w[W6][n][max(t - 8, 0) : t + 9]
        if np.isfinite(seg6).any():
            oriented = -sign * seg6
            k = int(np.nanargmax(oriented))
            if oriented[k] > best6:
                best6, best6_offset = float(oriented[k]), abs(k - (t - max(t - 8, 0))) / 2.0
    return {
        "opp_z24": best24,
        "opp_z6": best6,
        "opp_offset_h": best6_offset,
        "n_opposite": n_opp,
        "n_same": same,
        "n_neighbours": len(neighbours),
        "n_metered_opposite": n_metered_opp,
        "mva_neighbour": float(any(n in mva for n in neighbours)),
        "opp_mw_sum": opp_mw_sum,
        "best_neighbour": -1 if best_name is None else best_name,
    }


def candidates(*, stats: Stats, explained: dict[int, np.ndarray] | None = None) -> CandidateSet:
    """All RO-NC candidates (source `x`) and extra pair-alarm candidates (source `p`) with features."""
    explained = explained_masks() if explained is None else explained
    months = stamp_months()
    hour, weekday = stamp_hour_weekday()
    holiday = holiday_flag()
    day_of_year = stamp_index().dt.convert_time_zone("Europe/London").dt.ordinal_day().to_numpy()
    drops = pl.read_parquet(D8_DROPS)
    d8 = np.zeros(GRID_LENGTH, dtype=bool)
    for first, last in drops.select("first_stamp", "last_stamp").iter_rows():
        d8[max(first - DAY, 0) : min(last + DAY, GRID_LENGTH - 1) + 1] = True
    sigma = stats.scales.sigma
    rows: list[dict] = []
    windows: list[np.ndarray] = []
    strong = {s: _peaks(z=stats.z_x[s], threshold=STRONG_Z, half=W24) for s in stats.z_x}
    for series in stats.z_x:
        if not np.isfinite(sigma.get(series, np.nan)):
            continue
        zx, z6 = stats.z_x[series], stats.z_w[W6][series]
        x_peaks = _peaks(z=zx, threshold=CANDIDATE_Z, half=W24)
        p_all = _peaks(z=z6, threshold=PAIR_Z, half=W6)
        p_peaks = np.array(
            [p for p in p_all if not (np.abs(x_peaks - p) <= 4).any()] if len(x_peaks) else p_all,
            dtype=int,
        )
        res = stats.residual[series]
        for source, peaks in (("x", x_peaks), ("p", p_peaks)):
            for peak in peaks:
                t = int(peak)
                lead = zx[t] if source == "x" else z6[t]
                sign = 1.0 if lead >= 0 else -1.0
                view = _neighbour_view(stats=stats, series=series, t=t, sign=sign)
                if source == "p" and view["opp_z6"] < PAIR_PARTNER_Z:
                    continue
                own24 = sign * float(np.nan_to_num(zx[t]))
                step_mw = sign * float(np.nan_to_num(stats.shift_x[series][t]))
                after = res[t + W24 : t + 2 * W24]
                before = res[max(t - W24 + 2, 0) : max(t - 2, 1)]
                if (
                    np.isfinite(after).sum() > 8
                    and np.isfinite(before).sum() > 8
                    and step_mw > 1e-3
                ):
                    persist = float(
                        np.clip(
                            sign * (np.nanmean(after) - np.nanmean(before)) / step_mw, -1.0, 2.0
                        )
                    )
                else:
                    persist = 0.0
                ratio = (
                    float(np.clip(view["opp_mw_sum"] / step_mw, 0.0, 5.0))
                    if step_mw > 1e-3
                    else 0.0
                )
                strong_before = strong[series]
                recent = int(((strong_before < t) & (strong_before >= t - 7 * DAY)).sum())
                refit = sign * float(
                    np.nan_to_num(stats.z_first[series][t] - stats.z_second[series][t])
                )
                row = {
                    "series": series,
                    "stamp": t,
                    "source": source,
                    "sign": sign,
                    "x1": abs(float(np.nan_to_num(zx[t]))),
                    "x2": view["opp_z24"],
                    "own_z2": sign * float(np.nan_to_num(stats.z_w[W2][series][t])),
                    "own_z4": sign * float(np.nan_to_num(stats.z_w[W4][series][t])),
                    "own_z6": sign * float(np.nan_to_num(z6[t])),
                    "own_z24": own24,
                    "persist": persist,
                    "step_mw24": step_mw,
                    "opp_z24": view["opp_z24"],
                    "opp_z6": view["opp_z6"],
                    "opp_offset_h": view["opp_offset_h"],
                    "opp_ratio": ratio,
                    "imbalance": float(np.clip(abs(1.0 - ratio), 0.0, 3.0)),
                    "n_opposite": float(view["n_opposite"]),
                    "n_same": float(view["n_same"]),
                    "n_neighbours": float(view["n_neighbours"]),
                    "n_metered_opposite": float(view["n_metered_opposite"]),
                    "mva_neighbour": float(view["mva_neighbour"]),
                    "log_sigma_mw": float(np.log(sigma[series])),
                    "refit_offset": float(np.clip(refit, -10, 10)),
                    "hour_sin": float(np.sin(2 * np.pi * hour[t] / 24)),
                    "hour_cos": float(np.cos(2 * np.pi * hour[t] / 24)),
                    "season_sin": float(np.sin(2 * np.pi * day_of_year[t] / 365.25)),
                    "season_cos": float(np.cos(2 * np.pi * day_of_year[t] / 365.25)),
                    "weekend": float(weekday[t] >= 5),
                    "holiday": float(holiday[t]),
                    "recent_alarms": float(recent),
                    "end_est": end_stamp(stats=stats, series=series, onset=t, sign=sign),
                    "explained": bool(explained[series][t]),
                    "odd": bool(months[t] in TUNING_MONTHS),
                    "d8_drop": bool(d8[t]),
                }
                rows.append(row)
                window = np.zeros((4, RAW_LENGTH), dtype=np.float32)
                lo, hi = t - RAW_BEFORE, t + RAW_AFTER
                a, b = max(lo, 0), min(hi, GRID_LENGTH)
                own = sign * res[a:b] / sigma[series]
                window[0, a - lo : b - lo] = np.clip(np.nan_to_num(own), -RAW_CLIP, RAW_CLIP)
                window[1, a - lo : b - lo] = np.isfinite(own)
                nb = int(view["best_neighbour"])
                if nb >= 0 and np.isfinite(sigma.get(nb, np.nan)):
                    other = sign * stats.residual[nb][a:b] / sigma[nb]
                    window[2, a - lo : b - lo] = np.clip(np.nan_to_num(other), -RAW_CLIP, RAW_CLIP)
                    window[3, a - lo : b - lo] = np.isfinite(other)
                windows.append(window)
    table = pl.DataFrame(rows, infer_schema_length=None)
    return CandidateSet(
        table=table,
        windows=np.stack(windows) if windows else np.zeros((0, 4, RAW_LENGTH), np.float32),
    )


@dataclass(frozen=True)
class Logistic:
    """RO-NC's fitted logistic score (experiment 82's pickle, class of the same shape)."""

    features: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float

    def score(self, *, table: pl.DataFrame) -> np.ndarray:
        """The log-odds plus the tie-break."""
        x = (table.select(self.features).to_numpy().astype(float) - self.mean) / self.std
        return x @ self.coef + self.intercept + TIE_BREAK * (table["x1"] + table["x2"]).to_numpy()


class _Renamed(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        if name == "Logistic":
            return Logistic
        return super().find_class(module, name)


@cache
def ronc_score_model() -> Logistic:
    """RO-NC's score (stored as `XL`)."""
    return _Renamed(io.BytesIO(MODELS_82.read_bytes())).load()["XL"]


def ronc_scores(*, table: pl.DataFrame) -> np.ndarray:
    """RO-NC's log-odds per candidate; NaN for pair-source candidates, which RO-NC never proposes."""
    out = ronc_score_model().score(table=table)
    return np.where(table["source"].to_numpy() == "x", out, -np.inf)


def pair_scores(*, table: pl.DataFrame) -> np.ndarray:
    """The pair-alarm statistic: the smaller of the own and the opposite neighbour's 6-hour step."""
    return np.minimum(table["own_z6"].to_numpy(), table["opp_z6"].to_numpy())


def end_stamp(*, stats: Stats, series: int, onset: int, sign: float) -> int:
    """RO-NC's end: the level tracker's end, replaced by the first-return rule when one is found."""
    _, tracked = level_track(q=stats.tracker[series], onset=onset)
    z = stats.z_x[series]
    stop = min(onset + CAP_DAYS * DAY + 1, GRID_LENGTH)
    ret = -sign * z[onset + DAY : stop]
    need = max(RETURN_Z_MIN, FIRST_SHARE * abs(float(np.nan_to_num(z[onset]))))
    usable = np.where(np.isfinite(ret) & (ret >= need), ret, -np.inf)
    hits = np.flatnonzero(np.isfinite(usable))
    if hits.size == 0:
        return tracked
    first = int(hits[0])
    best = first + int(np.argmax(usable[first : first + W24 + 1]))
    return int(onset + DAY + best)
