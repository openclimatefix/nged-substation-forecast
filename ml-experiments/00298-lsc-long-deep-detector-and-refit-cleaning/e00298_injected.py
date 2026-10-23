# ruff: noqa: PLW2901, PERF102, B007
"""Part 1 on injected shifts: run every variant at its tuned threshold on the uncleaned worlds.

The worlds are those of `e00298_worlds.py --stage detect`: the node's daily residual after a
monthly refit with the transfer in the training data. Each variant runs at the threshold that gave
2 false alarms per Primary-year in odd months on the real record (`chosen_k.json`). A world counts
as found when an alarm span overlaps the transfer's days (or the day either side). A placebo run
shifts the neighbours' residuals by whole weeks, which gives the chance level of D to F.

    uv run --no-sync python e00298_injected.py
"""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00298_common as common
import e00298_detect as dt
import e00298_real as real
import e00298_worlds as worlds
import numpy as np
import polars as pl

PLACEBO_EXTRA_WEEKS: Final[tuple[int, ...]] = (1, 2, 3, 4)
"""Placebo shifts are the transfer's length in whole weeks plus 1 to 4 weeks, either way."""
SAME_K: Final[dict[str, str]] = {
    "D-A3": "A3",
    "D-A7": "A7",
    "D-C": "C",
    "F-A3": "A3",
    "F-A7": "A7",
    "F-C": "C",
}

_CLEAN: dict[str, dt.Stream] = {}
_PAIR_NOISE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


def variants_to_run() -> dict[str, dt.Variant]:
    """Every family at its tuned k, plus each wrapper at its base's k."""
    chosen = json.loads((common.EXPERIMENT_DIR / "chosen_k.json").read_text())
    out = {}
    for family, kwargs in real.FAMILIES:
        out[family] = dt.Variant(k=chosen[family], **kwargs)
    for family, base in SAME_K.items():
        kwargs = dict(next(kw for f, kw in real.FAMILIES if f == family))
        out[f"{family}@{base}k"] = dt.Variant(k=chosen[base], **kwargs)
    return out


def _pair_noise(
    *, site: str, others: dict[str, dt.Stream]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Clean-history pair noise per neighbour, cached across worlds."""
    out = {}
    for name, other in others.items():
        key = (site, name)
        if key not in _PAIR_NOISE:
            _PAIR_NOISE[key] = dt.pair_noise_of(a=_CLEAN[site].residual, b=_CLEAN[name].residual)
        out[name] = _PAIR_NOISE[key]
    return out


def world_rows(path: Path) -> list[dict]:
    """Score every world of one job file at every variant (and placebo for D to F)."""
    frame = pl.read_parquet(path)
    variants = variants_to_run()
    rows = []
    for (site, onset, tag), group in frame.group_by(["site", "onset", "tag"]):
        site = str(site)
        day0 = int((np.datetime64(onset.replace(tzinfo=None), "D") - dt.DAY0).astype(int))
        resid = _CLEAN[site].residual.copy()
        resid[day0:] = np.nan
        days = group["day"].to_numpy()
        resid[days] = group["residual_mw"].to_numpy()
        noise = worlds.noise_after(
            residual=resid, base_noise=_CLEAN[site].noise, start=day0, stop=int(days.max()) + 1
        )
        stream = dt.Stream(residual=resid, noise=noise, level=_CLEAN[site].level)
        others = {n: s for n, s in _CLEAN.items() if n in common.NEIGHBOURS.get(site, ())}
        parts = tag.split("_")
        size, shape, dur, met = int(parts[0][1:]) / 100, parts[1], parts[2], parts[3] == "met"
        step = dur == "step"
        duration = None if step else int(dur[:-1])
        last_known = int(days.max())
        first_true = day0
        last_true = last_known if duration is None else day0 + duration - 1
        partner = None
        if met and "partner_mw" in group.columns:
            partner_resid = np.full(dt.N_DAYS, np.nan)
            clean_name = next(
                n
                for n in common.NEIGHBOURS[site]
                if n in _CLEAN and np.isfinite(group["partner_mw"].to_numpy()).any()
            )
            partner = clean_name
            partner_resid[days] = group["partner_mw"].to_numpy()
            if frame.filter(pl.col("tag") == tag).height:
                pass
        for name, variant in variants.items():
            runs = [("true", others)]
            if variant.kind in "DEF":
                base = -(-((last_true - first_true + 1) + 7) // 7)
                weeks_list = [sg * (base + e) for e in PLACEBO_EXTRA_WEEKS for sg in (-1, 1)]
                runs += [(f"placebo{w}", _shift_others(others=others, weeks=w)) for w in weeks_list]
            for label, nb in runs:
                if (
                    variant.kind in "DEF"
                    and met
                    and partner is not None
                    and "partner_mw" in group.columns
                ):
                    nb = _with_partner(
                        nb=nb,
                        site=site,
                        partner_resid=group,
                        days=days,
                        weeks=0 if label == "true" else int(label[7:]),
                    )
                spans = dt.detect(
                    variant=variant,
                    stream=stream,
                    others=nb,
                    first=max(day0 - 35, 40),
                    last=last_known,
                    pair_noise=_pair_noise(site=site, others=nb) if variant.kind == "E" else None,
                )
                hit = [
                    s
                    for s in spans
                    if s.sign == 1 and s.start <= last_true + 1 and s.end >= first_true - 1
                ]
                first = min(hit, key=lambda s: s.fire) if hit else None
                rows.append(
                    {
                        "site": site,
                        "onset": onset,
                        "tag": tag,
                        "size": size,
                        "shape": shape,
                        "duration": -1 if step else duration,
                        "metered": met,
                        "variant": name,
                        "run": "true" if label == "true" else "placebo",
                        "found": first is not None,
                        "alarm_lag_h": (first.fire - first_true) * 24.0 if first else np.nan,
                        "onset_err_h": (first.start - first_true) * 24.0 if first else np.nan,
                        "end_err_h": (first.end - last_true) * 24.0
                        if first and not step
                        else np.nan,
                        "n_spans_outside": sum(1 for s in spans if s not in hit),
                        "explained": first.explained if first else np.nan,
                    }
                )
    return rows


def _shift_others(*, others: dict[str, dt.Stream], weeks: int) -> dict[str, dt.Stream]:
    """Return the neighbours' streams shifted by `weeks` whole weeks."""
    return {
        n: dt.Stream(
            residual=np.roll(s.residual, 7 * weeks),
            noise=np.roll(s.noise, 7 * weeks),
            level=s.level,
        )
        for n, s in others.items()
    }


def _with_partner(*, nb, site, partner_resid, days, weeks) -> dict[str, dt.Stream]:  # noqa: ANN001
    """Replace the partner neighbour's stream with the world's partner residual (shifted)."""
    # The partner is the neighbour with the most power; recover it from the stored column order.
    partner_name = _PARTNER[site]
    if partner_name not in nb and weeks == 0:
        return nb
    arr = np.full(dt.N_DAYS, np.nan)
    arr[days] = partner_resid["partner_mw"].to_numpy()
    base = _CLEAN[partner_name]
    full = base.residual.copy()
    first_day = int(days.min())
    full[first_day:] = np.nan
    full[days] = arr[days]
    out = dict(nb)
    out[partner_name] = dt.Stream(
        residual=np.roll(full, 7 * weeks) if weeks else full, noise=base.noise, level=base.level
    )
    return out


_PARTNER: dict[str, str] = {}


def _init() -> None:
    """Build the clean streams and the partner choice inside a worker."""
    _CLEAN.update(real.build_streams())
    cache = common.load_cache()
    for site in common.INJECT_SITES:
        options = [n for n in common.NEIGHBOURS.get(site, ()) if n in _CLEAN]
        if options:
            power = {
                n: float(np.nanmean(cache.sites[n].y) * cache.sites[n].table.effective_capacity_mw)
                for n in options
            }
            _PARTNER[site] = max(power, key=lambda name: power[name])


def main() -> None:
    """Score all worlds of the detect stage in parallel and write the table."""
    files = sorted((common.WORLDS_DIR / "detect").glob("streams_*.parquet"))
    context = multiprocessing.get_context("spawn")
    rows = []
    with ProcessPoolExecutor(max_workers=10, mp_context=context, initializer=_init) as pool:
        for chunk in pool.map(world_rows, files):
            rows.extend(chunk)
    pl.DataFrame(rows).write_parquet(common.EXPERIMENT_DIR / "injected_detection.parquet")
    print(f"{len(rows)} rows from {len(files)} files")


if __name__ == "__main__":
    main()
