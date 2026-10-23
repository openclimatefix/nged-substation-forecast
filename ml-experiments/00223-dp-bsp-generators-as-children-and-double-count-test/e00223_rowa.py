"""Row A of experiment 223: does a BSP's remainder carry each metered generator?

The remainder is the BSP's reading minus the weighted sum of its child Primaries' readings in MW,
with experiment 204's arm (d) weights and signed flows. A hour-of-day-by-month profile is removed
from the remainder and from the generator, and the remainder is regressed on the generator with no
intercept. A generator that the BSP carries gives -1, and one behind a Primary gives 0. The
interval resamples whole days. Each Primary's reading is regressed on each generator the same way.
The known-answer check plants a synthetic generator (a real generator's output shifted by 364
days) in the remainder or behind a Primary and checks the regression classifies it correctly.
"""

import json
from datetime import UTC, datetime

import e00223_common as common
import e00223_generators as gens
import numpy as np
from studies.dp_features import FeatureCache, naive_utc

N_RESAMPLES = 20_000
ALPHA = 0.05
SEED = 223
MIN_DAYS = 60
SHIFT_ROWS = 364 * 48
CUTOFF = datetime(2025, 7, 1, tzinfo=UTC)


def _remainder(*, cache: FeatureCache, bsp: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the BSP's remainder in MW and each child's weighted reading in MW."""
    weights = common.weights_for(cache=cache, bsp=bsp, variant="fold")
    reading = common._megawatts(cache=cache, site=bsp)
    parts = {
        child: weights[child] * common._megawatts(cache=cache, site=child, variant="fold")
        for child in common.BSPS[bsp]
    }
    r = reading.copy()
    for part in parts.values():
        r = r - part
    return r, parts


def _rows(*, cache: FeatureCache, bsp: str) -> np.ndarray:
    ok = cache.end_times < naive_utc(moment=CUTOFF)
    for site in (bsp, *common.BSPS[bsp]):
        ok &= cache.sites[site].in_pool & ~cache.sites[site].switching_mask
    return ok


def _cells(*, stamps: np.ndarray) -> np.ndarray:
    months = stamps.astype("datetime64[M]").astype(int) % 12
    slots = ((stamps - stamps.astype("datetime64[D]")) / np.timedelta64(30, "m")).astype(int)
    return months * 48 + slots


def _demean(*, values: np.ndarray, cells: np.ndarray) -> np.ndarray:
    sums = np.bincount(cells, weights=values, minlength=576)
    counts = np.bincount(cells, minlength=576)
    return values - (sums / np.maximum(counts, 1))[cells]


def regress(
    *, y: np.ndarray, x: np.ndarray, usable: np.ndarray, stamps: np.ndarray, level: float
) -> dict:
    """Regress `y` on `x` through the origin after removing the month-by-slot profile.

    Returns the slope, the day-resampled interval at `level`, and the number of days.
    """
    keep = usable & np.isfinite(y) & np.isfinite(x)
    if keep.sum() < MIN_DAYS * 48:
        return {"slope": None, "low": None, "high": None, "days": int(keep.sum() // 48)}
    cells = _cells(stamps=stamps[keep])
    yy = _demean(values=y[keep], cells=cells)
    xx = _demean(values=x[keep], cells=cells)
    days = stamps[keep].astype("datetime64[D]")
    _, inverse = np.unique(days, return_inverse=True)
    n_days = inverse.max() + 1
    sxy = np.bincount(inverse, weights=xx * yy, minlength=n_days)
    sxx = np.bincount(inverse, weights=xx * xx, minlength=n_days)
    slope = float(sxy.sum() / sxx.sum())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_days, size=(N_RESAMPLES, n_days))
    draws = sxy[idx].sum(axis=1) / sxx[idx].sum(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return {"slope": slope, "low": float(low), "high": float(high), "days": int(n_days)}


def regress_joint(
    *,
    y: np.ndarray,
    columns: dict[str, np.ndarray],
    usable: np.ndarray,
    stamps: np.ndarray,
    level: float,
) -> dict:
    """Regress `y` on all `columns` at once, with day-resampled intervals for each slope."""
    names = list(columns)
    x = np.column_stack([columns[n] for n in names])
    keep = usable & np.isfinite(y) & np.isfinite(x).all(axis=1)
    if keep.sum() < MIN_DAYS * 48:
        return {}
    cells = _cells(stamps=stamps[keep])
    yy = _demean(values=y[keep], cells=cells)
    xx = np.column_stack([_demean(values=x[keep, j], cells=cells) for j in range(len(names))])
    _, inverse = np.unique(stamps[keep].astype("datetime64[D]"), return_inverse=True)
    n_days = inverse.max() + 1
    k = len(names)
    xtx = np.zeros((n_days, k, k))
    xty = np.zeros((n_days, k))
    for a in range(k):
        xty[:, a] = np.bincount(inverse, weights=xx[:, a] * yy, minlength=n_days)
        for b in range(k):
            xtx[:, a, b] = np.bincount(inverse, weights=xx[:, a] * xx[:, b], minlength=n_days)
    beta = np.linalg.solve(xtx.sum(0), xty.sum(0))
    rng = np.random.default_rng(SEED)
    draws = np.empty((N_RESAMPLES, k))
    for start in range(0, N_RESAMPLES, 1000):
        idx = rng.integers(0, n_days, size=(1000, n_days))
        draws[start : start + 1000] = np.linalg.solve(xtx[idx].sum(1), xty[idx].sum(1)[..., None])[
            ..., 0
        ]
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail], axis=0)
    return {
        n: {
            "slope": float(beta[j]),
            "low": float(low[j]),
            "high": float(high[j]),
            "days": int(n_days),
        }
        for j, n in enumerate(names)
    }


def classify(*, result: dict) -> str:
    """Return `carried by the BSP`, `inside a Primary`, `cannot tell`, or `no data`."""
    if result["slope"] is None:
        return "no data"
    holds_carried = result["low"] <= -1.0 <= result["high"]
    holds_zero = result["low"] <= 0.0 <= result["high"]
    if holds_carried and not holds_zero:
        return "carried by the BSP"
    if holds_zero and not holds_carried:
        return "inside a Primary"
    return "cannot tell"


def units(*, series: dict[str, np.ndarray]) -> dict[str, dict]:
    """Return the tested units: each non-solar EHV generator, and the EHV solar group per BSP."""
    out: dict[str, dict] = {}
    solar: dict[str, list[str]] = {}
    for label, spec in gens.GENERATORS.items():
        if spec["voltage"] != "EHV":
            continue
        if spec["kind"] == "solar":
            solar.setdefault(spec["bsp"], []).append(label)
        else:
            out[label] = {"bsp": spec["bsp"], "members": [label]}
    for bsp, members in solar.items():
        out[f"solar_{bsp}"] = {"bsp": bsp, "members": members}
    return out


def unit_series(*, series: dict[str, np.ndarray], members: list[str]) -> np.ndarray:
    """Return the sum of members' outputs, `NaN` where any member lacks a reading."""
    return np.sum([series[m] for m in members], axis=0)


def main() -> None:
    """Run Row A and its known-answer check, and write `work/rowa.json`."""
    cache = common.load_cache()
    series = gens.load_generators(cache=cache)
    stamps = cache.end_times
    tested = units(series=series)
    level = 1.0 - ALPHA / len(tested)
    result: dict = {
        "children": gens.child_table(cache=cache, series=series),
        "level": level,
        "tests": len(tested),
        "units": {},
        "primary_check": {},
        "known_answer": {},
        "joint": {},
    }
    for bsp in common.BSPS:
        r, parts = _remainder(cache=cache, bsp=bsp)
        usable = _rows(cache=cache, bsp=bsp)
        for name, spec in tested.items():
            if spec["bsp"] != bsp:
                continue
            x = unit_series(series=series, members=spec["members"])
            fit = regress(y=r, x=x, usable=usable, stamps=stamps, level=level)
            fit["class"] = classify(result=fit)
            result["units"][name] = {**spec, **fit}
            # Second check: each child Primary's weighted reading on the generator.
            checks = {}
            for child, part in parts.items():
                one = regress(y=part, x=x, usable=usable, stamps=stamps, level=level)
                checks[child] = one
            result["primary_check"][name] = checks
            # Known-answer check: plant a shifted copy as a carried or as an inside-a-Primary unit.
            shifted = np.concatenate([np.full(SHIFT_ROWS, np.nan), x[:-SHIFT_ROWS]])
            carried = regress(
                y=r - np.nan_to_num(shifted), x=shifted, usable=usable, stamps=stamps, level=level
            )
            inside = regress(y=r, x=shifted, usable=usable, stamps=stamps, level=level)
            result["known_answer"][name] = {
                "planted_in_remainder": {**carried, "class": classify(result=carried)},
                "planted_behind_primary": {**inside, "class": classify(result=inside)},
            }
        columns = {
            name: unit_series(series=series, members=spec["members"])
            for name, spec in tested.items()
            if spec["bsp"] == bsp
        }
        result["joint"][bsp] = regress_joint(
            y=r, columns=columns, usable=usable, stamps=stamps, level=level
        )
    path = common.WORK_DIR / "rowa.json"
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    common.write_json_atomic(path=path, content=result)
    for name, r in result["units"].items():
        print(name, r["bsp"], r["class"], r["slope"], r["low"], r["high"], r["days"])
    print(json.dumps(result["known_answer"], indent=0)[:2000])


if __name__ == "__main__":
    main()
