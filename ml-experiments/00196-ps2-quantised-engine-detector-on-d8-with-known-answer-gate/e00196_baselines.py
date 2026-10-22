"""RO-NC and experiment 181's block detector on the same injections as the engine detector.

Both are single-series step detectors on the residual (reading minus expected power). The step
statistic, its odd-month noise scale, the peak rule, and the operating points are copied from
experiments 82 and 181; the neighbour-count check uses the other panel series' residuals. The
refit-offset variant (a second residual model refitted on the 16th) is not in the stored panel, so
alarms within 24 hours of a refit on the 1st are suppressed instead, as in experiment 82's PF14.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00196-*/e00196_baselines.py
"""

import json

import numpy as np
from e00196_common import DATA_DIR, g2_output, key_of, load_panel
from e00196_inject import Injection, build, synthetic

N = 78816
TUNING_MONTHS = {1, 3, 5, 7, 9, 11}
NEIGHBOUR_Z = 2.0
BALANCE_RANGE = (0.4, 2.5)
ADJACENCY = {
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
}
RONC_POINTS = (
    ("D0_48_4.0", 48, 4.0),
    ("D1_48_2.5", 48, 2.5),
    ("D0_8_6.0", 8, 6.0),
    ("D1_8_3.75", 8, 3.75),
)
BLOCK_LENGTHS = (12, 24, 48, 96, 240)
BLOCK_THRESHOLD = 4.371440390284409
"""Experiment 181's S0 threshold, fixed there on odd months."""
TOLERANCE = 4
"""Half-hours (2 hours) within which an alarm counts as at a spell's onset or end."""


def month_index() -> np.ndarray:
    """UTC calendar month (1 to 12) of every grid stamp."""
    base = np.datetime64("2022-01-01T00:00")
    return (base + np.arange(N) * np.timedelta64(30, "m")).astype("datetime64[M]").astype(
        int
    ) % 12 + 1


MONTHS = month_index()


def level_shift(*, values: np.ndarray, window: int) -> np.ndarray:
    """Mean of the next `window` stamps minus the mean of the previous `window` stamps."""
    total = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values))])
    count = np.concatenate([[0], np.cumsum(np.isfinite(values))])
    t = np.arange(N)
    lo, hi = np.clip(t - window, 0, N), np.clip(t + window, 0, N)
    nb, na = count[t] - count[lo], count[hi] - count[t]
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = (total[hi] - total[t]) / na - (total[t] - total[lo]) / nb
    shift[(nb < 0.75 * window) | (na < 0.75 * window)] = np.nan
    return shift


def mad_scale(*, x: np.ndarray) -> float:
    """1.4826 MAD over odd calendar months."""
    sel = x[np.isin(MONTHS, list(TUNING_MONTHS)) & np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(sel - np.median(sel))))


def local_peaks(*, z: np.ndarray, threshold: float, window: int) -> np.ndarray:
    """Stamps where |z| exceeds the threshold and is the largest within +/- window."""
    mag = np.nan_to_num(np.abs(z))
    return np.array(
        [
            t
            for t in np.flatnonzero(mag > threshold)
            if mag[t] == mag[max(t - window, 0) : t + window + 1].max()
        ],
        dtype=int,
    )


def refit_stamps() -> np.ndarray:
    """Stamps at 00:00 UTC on the 1st of each month."""
    base = np.datetime64("2022-01-01T00:00")
    t = base + np.arange(N) * np.timedelta64(30, "m")
    return np.flatnonzero(
        (
            t.astype("datetime64[D]").astype("datetime64[M]").astype("datetime64[D]")
            == t.astype("datetime64[D]")
        )
        & (t == t.astype("datetime64[D]"))
    )


def ronc_alarms(
    *,
    residuals: dict[int, np.ndarray],
    scales: dict,
    series: int,
    window: int,
    threshold: float,
    neighbours: bool,
) -> np.ndarray:
    """Alarm stamps of D0 (own step only) or D1 (an opposite-signed neighbour step balances it)."""
    shifts = {s: level_shift(values=v, window=window) for s, v in residuals.items()}
    z = shifts[series] / scales[(series, window)]
    refits = refit_stamps() if window >= 48 else np.array([], dtype=int)
    out = []
    for t in local_peaks(z=z, threshold=threshold, window=window):
        if len(refits) and (np.abs(refits - t) < window).any():
            continue
        if neighbours:
            own = shifts[series][t]
            partners = 0.0
            n_partner = 0
            for nb in ADJACENCY.get(series, ()):
                seg = shifts[nb][max(t - 1, 0) : t + 2]
                if not np.isfinite(seg).any():
                    continue
                step = float(seg[np.nanargmax(-np.sign(own) * seg)])
                if -np.sign(own) * step / scales[(nb, window)] > NEIGHBOUR_Z:
                    partners += step
                    n_partner += 1
            ratio = -partners / own
            if not n_partner or not BALANCE_RANGE[0] <= ratio <= BALANCE_RANGE[1]:
                continue
        out.append(t)
    return np.array(out, dtype=int)


def edge_contrasts(*, values: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Block mean minus the mean before it and minus the mean after it."""
    flank = min(length, 48)
    cum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values))])
    cnt = np.concatenate([[0], np.cumsum(np.isfinite(values))])
    t = np.arange(N)

    def wmean(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = np.clip(lo, 0, N), np.clip(hi, 0, N)
        k = cnt[hi] - cnt[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (cum[hi] - cum[lo]) / k, k

    inside, ni = wmean(t, t + length)
    before, nb = wmean(t - flank, t)
    after, na = wmean(t + length, t + length + flank)
    ok = (ni >= 0.75 * length) & (nb >= 0.75 * flank) & (na >= 0.75 * flank)
    up, down = inside - before, inside - after
    up[~ok], down[~ok] = np.nan, np.nan
    return up, down


def block_z(
    *, values: np.ndarray, scales: dict[int, float] | None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Signed block score of largest magnitude over the lengths, its length, and the scales used."""
    zs, used = [], {}
    for length in BLOCK_LENGTHS:
        up, down = edge_contrasts(values=values, length=length)
        same_up, same_down = (up > 0) & (down > 0), (up < 0) & (down < 0)
        score = np.where(
            same_up, np.minimum(up, down), np.where(same_down, np.maximum(up, down), 0.0)
        )
        score[~np.isfinite(up) | ~np.isfinite(down)] = np.nan
        scale = mad_scale(x=up) if scales is None else scales[length]
        used[length] = scale
        zs.append(np.nan_to_num(score / scale))
    stack = np.stack(zs)
    pick = np.argmax(np.abs(stack), axis=0)
    return np.take_along_axis(stack, pick[None], axis=0)[0], np.array(BLOCK_LENGTHS)[pick], used


def spells(*, inj: Injection) -> list[tuple[int, int]]:
    """Runs of consecutive running, scored half-hours: (first stamp, last stamp inclusive)."""
    run = inj.scored & (inj.truth_units >= 1)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], run.astype(int), [0]])))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def coverage(*, inj: Injection, alarms: np.ndarray, lengths: np.ndarray | None = None) -> dict:
    """Share of spells with an alarm at onset or end, and of running half-hours in found spells."""
    sp = spells(inj=inj)
    found, hh_found, hh_total, hh_block = 0, 0, 0, 0
    run = inj.scored & (inj.truth_units >= 1)
    covered = np.zeros(N, dtype=bool)
    if lengths is not None:
        for t in alarms:
            covered[t : t + int(lengths[t])] = True
    for a, b in sp:
        near = (np.abs(alarms - a) <= TOLERANCE) | (np.abs(alarms - b - 1) <= TOLERANCE)
        hh_total += b - a + 1
        if near.any():
            found += 1
            hh_found += b - a + 1
    hh_block = int((run & covered).sum()) if lengths is not None else None
    return {
        "spells": len(sp),
        "spells_found": found,
        "spell_recall": found / max(len(sp), 1),
        "half_hour_recall_found_spells": hh_found / max(hh_total, 1),
        "half_hour_recall_block_span": None if hh_block is None else hh_block / max(hh_total, 1),
        "alarms_in_windows": int(np.isin(alarms, np.flatnonzero(inj.window_id >= 0)).sum()),
    }


def main() -> None:
    """Score both detectors on K1, K2 and the synthetic injections."""
    panel = load_panel()
    key = key_of(panel=panel, label="D8")
    residuals = {k: np.asarray(v["resid"], dtype=float) for k, v in panel.items()}
    expected = np.asarray(panel[key]["expected"], dtype=float)
    y = np.asarray(panel[key]["power"], dtype=float)
    g2 = g2_output(n=N)
    scales = {}
    for window in {w for _, w, _ in RONC_POINTS}:
        for k, v in residuals.items():
            scales[(k, window)] = mad_scale(x=level_shift(values=v, window=window))
    _, _, block_scales = block_z(values=residuals[key], scales=None)
    arms = {
        "K1": build(y=y, g2=g2, with_k2=False),
        "K2": build(y=y, g2=g2, with_k1=False),
        "synthetic_x0.5": synthetic(y=y, g2=g2, rate=0.5, seed=196),
        "synthetic_x2.0": synthetic(y=y, g2=g2, rate=2.0, seed=196),
    }
    results: dict = {}
    for name, inj in arms.items():
        res = residuals.copy()
        res[key] = inj.y - expected
        out: dict = {}
        for label, window, thr in RONC_POINTS:
            alarms = ronc_alarms(
                residuals=res,
                scales=scales,
                series=key,
                window=window,
                threshold=thr,
                neighbours=label.startswith("D1"),
            )
            out[f"RO-NC {label}"] = coverage(inj=inj, alarms=alarms)
        z, length, _ = block_z(values=res[key], scales=block_scales)
        peaks = local_peaks(z=z, threshold=BLOCK_THRESHOLD, window=48)
        out["block S0"] = coverage(inj=inj, alarms=peaks, lengths=length)
        results[name] = out
        print(
            name,
            {k: round(v["half_hour_recall_found_spells"], 3) for k, v in out.items()},
            flush=True,
        )
    (DATA_DIR / "baseline_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
