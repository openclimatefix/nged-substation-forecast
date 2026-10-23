# ruff: noqa: PLR0915, PLR0912, PERF401, D107
"""Injected worlds: add a load transfer to one Primary, refit monthly, and clean or not.

A world is (site, onset, scenario). The node receives `size` times its own power (`share`) or its
mean power over the year before the onset (`fixed`) for `duration` days, or for ever (`permanent`).
A metered world takes the same megawatts from the neighbour with the most power, so that variants
D to F can confirm it. The monthly cuts run from the onset to four months after the end (fifteen
months for a permanent step), plus a cut at the end of the transfer, as in experiments 274 and 290.

Modes: `none` (no cleaning), `oracle` (the true shift days are excluded from the refit), or a
detector variant (`A3`, `C`, `D-A3`, ... with its threshold), whose flagged days known before the
cut are excluded, capped at 60 days in a row.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
        uv run --no-sync python e00298_worlds.py --stage detect --workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Final

import e00298_common as common
import e00298_detect as dt
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from scipy.special import ndtri
from studies.dp_ens_weather import replace_site
from studies.dp_features import FeatureCache, naive_utc

DURATIONS: Final[tuple[int, ...]] = (1, 3, 6, 7, 14, 30, 90)
SIZES: Final[tuple[float, ...]] = (0.10, 0.25, 0.50)
STEP_SIZES: Final[tuple[float, ...]] = (0.10, 0.25)
SCORE_ONSETS: Final[tuple[datetime, ...]] = tuple(
    datetime(y, m, 1, tzinfo=UTC) for y, m in ((2022, 12), (2023, 6), (2023, 12), (2024, 6))
)
TUNE_ONSETS: Final[tuple[datetime, ...]] = tuple(
    datetime(y, m, 1, tzinfo=UTC) for y, m in ((2023, 1), (2023, 7), (2024, 1), (2024, 7))
)
MONTHS_AFTER_END: Final[int] = 4
STEP_MONTHS: Final[int] = 15
CAP_DAYS: Final[int] = 60
AFTER_WINDOW: Final[tuple[tuple[str, float, float], ...]] = (
    ("AFTER_0_2", 0.0, 2.0),
    ("AFTER_2_10", 2.0, 10.0),
    ("AFTER_2_33", 2.0, 33.0),
)


@dataclass(frozen=True)
class Scenario:
    """One injected transfer. `duration` is in days; `None` is a permanent step."""

    size: float
    shape: str
    duration: int | None
    metered: bool = False

    @property
    def tag(self) -> str:
        """A file-safe label."""
        dur = "step" if self.duration is None else f"{self.duration}d"
        return f"s{int(self.size * 100)}_{self.shape}_{dur}_{'met' if self.metered else 'unm'}"


def scenarios(*, stage: str) -> list[Scenario]:
    """Return the scenarios of a stage: the full grid, plus metered, plus permanent steps."""
    grid = [Scenario(s, shape, d) for s in SIZES for shape in ("share", "fixed") for d in DURATIONS]
    metered = [Scenario(0.25, "share", d, True) for d in DURATIONS]
    steps = [Scenario(s, "share", None) for s in STEP_SIZES]
    steps += [replace(sc, metered=True) for sc in steps]
    return [*grid, *metered, *steps]


def cut_schedule(*, onset: datetime, end: datetime | None) -> tuple[list[datetime], list[datetime]]:
    """Return (cuts, stops): monthly cuts from the onset plus a cut at the end of the transfer."""
    last = (
        common.add_months(moment=onset, months=STEP_MONTHS)
        if end is None
        else common.add_months(moment=end, months=MONTHS_AFTER_END)
    )
    last = min(last, common.LAST_CUT)
    cuts = common.month_starts(first=onset, last=last)
    if end is not None and end <= common.LAST_CUT and end not in cuts:
        cuts.append(end)
    cuts = sorted(cuts)
    stops = [*cuts[1:], common.add_months(moment=cuts[-1], months=1)]
    return cuts, stops


def true_days(*, onset: datetime, end: datetime | None) -> tuple[int, int]:
    """Return (first, last) grid days of the transfer; `last` is the final day for a step."""
    first = int((np.datetime64(onset.replace(tzinfo=None), "D") - dt.DAY0).astype(int))
    if end is None:
        return first, dt.N_DAYS - 1
    return first, int((np.datetime64(end.replace(tzinfo=None), "D") - dt.DAY0).astype(int)) - 1


LEVELS: Final[np.ndarray] = np.arange(1, 100) / 100.0
Z_LEVELS: Final[np.ndarray] = ndtri(LEVELS)


def gaussian_scores(
    *, centre: np.ndarray, sd: float, actual: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (CRPS, threshold-weighted CRPS) per row of a Gaussian forecast, by 99 quantiles.

    The threshold-weighted score uses the chaining function max(x, threshold), which puts all the
    weight above the threshold (the upper tail, where curtailment matters).
    """
    quantiles = centre[:, None] + sd * Z_LEVELS[None, :]

    def pinball(q: np.ndarray, y: np.ndarray) -> np.ndarray:
        error = y[:, None] - q
        return np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error)

    crps = 2.0 * pinball(quantiles, actual).mean(axis=1)
    tw = 2.0 * pinball(np.maximum(quantiles, threshold), np.maximum(actual, threshold)).mean(axis=1)
    return crps, tw


def noise_after(
    *, residual: np.ndarray, base_noise: np.ndarray, start: int, stop: int
) -> np.ndarray:
    """Return `base_noise` with entries `start`..`stop - 1` recomputed from `residual`."""
    out = base_noise.copy()
    for d in range(start, stop):
        window = residual[max(0, d - dt.NOISE_DAYS) : d]
        window = window[np.isfinite(window)]
        if len(window) >= dt.MIN_NOISE_DAYS:
            out[d] = 1.4826 * np.median(np.abs(window - np.median(window)))
    return out


class JobContext:
    """What a (site, onset) job shares across its worlds: the cache, clean streams, clean fits."""

    def __init__(self, *, cache: FeatureCache, site: str, onset: datetime) -> None:
        self.cache, self.site, self.onset = cache, site, onset
        panel = pl.read_parquet(common.PANEL_PATH)
        arrays = dt.panel_arrays(panel=panel)
        self.clean = {
            s: dt.make_stream(residual=a["residual"], level=dt.rolling_level(power=a["power"]))
            for s, a in arrays.items()
        }
        self.partner_name = self._pick_partner()
        self.clean_fits: dict[tuple[str, datetime, datetime], np.ndarray] = {}
        self.row_day = dt.day_index(days=(cache.end_times - np.timedelta64(30, "m")))
        self.partner_cache: dict[str, dict] = {}
        before = (cache.end_times <= naive_utc(moment=onset)) & (
            cache.end_times > naive_utc(moment=common.add_months(moment=onset, months=-24))
        )
        self.threshold = float(np.nanquantile(cache.sites[site].y[before], 0.90))

    def _pick_partner(self) -> str | None:
        """The metered neighbour with the most power, or `None`."""
        options = [n for n in common.NEIGHBOURS.get(self.site, ()) if n in self.clean]
        if not options:
            return None
        power = {
            n: float(
                np.nanmean(self.cache.sites[n].y) * self.cache.sites[n].table.effective_capacity_mw
            )
            for n in options
        }
        return max(power, key=lambda name: power[name])

    def clean_forecast(
        self, *, site: str, cut: datetime, stop: datetime, index: np.ndarray
    ) -> np.ndarray:
        """The clean-trained forecast of `index` (cached across scenarios and modes)."""
        key = (site, cut, stop)
        if key not in self.clean_fits:
            predicted = common.forecast_month(cache=self.cache, site=site, cut=cut, index=index)
            self.clean_fits[key] = (
                predicted if predicted is not None else np.full(len(index), np.nan)
            )
        return self.clean_fits[key]


def delta_points(
    *, ctx: JobContext, site: str, scenario: Scenario, end: datetime | None, sign: float
) -> np.ndarray:
    """Return the injected change of `site`'s power in points, on the grid, for the node's MW."""
    features = ctx.cache.sites[ctx.site]
    stamps = ctx.cache.end_times
    active = stamps > naive_utc(moment=ctx.onset)
    if end is not None:
        active = active & (stamps <= naive_utc(moment=end))
    node = np.zeros_like(features.y)
    if scenario.shape == "share":
        node[active] = scenario.size * features.y[active]
    else:
        before = (stamps <= naive_utc(moment=ctx.onset)) & (
            stamps > naive_utc(moment=common.add_months(moment=ctx.onset, months=-12))
        )
        node[active] = scenario.size * float(np.nanmean(features.y[before]))
    if site == ctx.site:
        return node
    megawatts = node * features.table.effective_capacity_mw / 100.0
    return sign * megawatts / ctx.cache.sites[site].table.effective_capacity_mw * 100.0


def window_labels(
    *, times: np.ndarray, onset: datetime, end: datetime | None
) -> dict[str, np.ndarray]:
    """Row masks for each scoring label."""
    t0 = naive_utc(moment=onset)
    out = {"ALL": np.ones(len(times), dtype=bool), "BEFORE": times <= t0}
    if end is None:
        month = np.timedelta64(30, "D")
        block = ((times - t0) // month).astype(int)
        out["DURING"] = times > t0
        for b in range(STEP_MONTHS + 1):
            out[f"STEP_B{b}"] = (times > t0) & (block == b)
        return out
    t1 = naive_utc(moment=end)
    out["DURING"] = (times > t0) & (times <= t1)
    for name, lo, hi in AFTER_WINDOW:
        out[name] = (times > t1 + np.timedelta64(int(lo * 86400), "s")) & (
            times <= t1 + np.timedelta64(int(hi * 86400), "s")
        )
    out["LATER"] = times > t1 + np.timedelta64(33 * 86400, "s")
    return out


def run_world(
    *,
    ctx: JobContext,
    scenario: Scenario,
    mode: str,
    variant: dt.Variant | None,
    control: bool = False,
) -> tuple[list[dict], pl.DataFrame]:
    """Run one world in one mode and return (per-window records, daily streams)."""
    cache, site, onset = ctx.cache, ctx.site, ctx.onset
    features = cache.sites[site]
    end = (
        None if control or scenario.duration is None else onset + timedelta(days=scenario.duration)
    )
    if control:
        cuts = common.month_starts(first=datetime(2022, 1, 1, tzinfo=UTC), last=common.LAST_CUT)
        stops = [*cuts[1:], common.add_months(moment=cuts[-1], months=1)]
        onset_eff = cuts[0]
    else:
        cuts, stops = cut_schedule(onset=onset, end=end)
        onset_eff = onset
    zero = np.zeros_like(features.y)
    delta = (
        zero if control else delta_points(ctx=ctx, site=site, scenario=scenario, end=end, sign=1.0)
    )
    y_node = features.y + delta
    partner = ctx.partner_name if (scenario.metered and not control) else None
    partner_stream = None
    if partner is not None:
        partner_stream = _partner_stream(
            ctx=ctx, scenario=scenario, end=end, partner=partner, cuts=cuts, stops=stops
        )
    day0 = int((np.datetime64(onset_eff.replace(tzinfo=None), "D") - dt.DAY0).astype(int))
    resid = ctx.clean[site].residual.copy()
    resid[day0:] = np.nan
    base_noise = ctx.clean[site].noise
    level = ctx.clean[site].level
    others = {n: s for n, s in ctx.clean.items() if n in common.NEIGHBOURS.get(site, ())}
    if partner_stream is not None and partner is not None:
        others[partner] = dt.Stream(
            residual=partner_stream, noise=ctx.clean[partner].noise, level=ctx.clean[partner].level
        )
    true_first, true_last = (0, -1) if control else true_days(onset=onset, end=end)
    records: list[dict] = []
    flagged_total = flagged_true = flagged_outside = 0
    spans: list[dt.Span] = []
    for cut, stop in zip(cuts, stops, strict=True):
        index = common.target_rows(cache=cache, site=site, cut=cut, stop=stop)
        if len(index) == 0:
            continue
        cut_day = int((np.datetime64(cut.replace(tzinfo=None), "D") - dt.DAY0).astype(int))
        exclude_days = np.zeros(dt.N_DAYS, dtype=bool)
        if mode == "oracle" and not control:
            exclude_days[true_first : min(true_last, cut_day - 1) + 1] = True
            run = np.cumsum(exclude_days) - np.maximum.accumulate(
                np.where(~exclude_days, np.cumsum(exclude_days), 0)
            )
            exclude_days &= run <= CAP_DAYS
        elif variant is not None and cut_day > day0:
            noise = noise_after(residual=resid, base_noise=base_noise, start=day0, stop=cut_day)
            stream = dt.Stream(residual=resid, noise=noise, level=level)
            spans = dt.detect(
                variant=variant,
                stream=stream,
                others=others,
                first=max(day0 - 35, 40),
                last=cut_day - 1,
            )
            exclude_days = dt.flagged_days(spans=spans, cut_day=cut_day, cap=CAP_DAYS)
        excluded_rows = exclude_days[ctx.row_day]
        if excluded_rows.any():
            training = features.default_rows & ~excluded_rows
            world_cache = replace_site(cache=cache, site=site, y=y_node, default_rows=training)
        else:
            world_cache = replace_site(cache=cache, site=site, y=y_node)
        recheck_index = (
            common.target_rows(cache=cache, site=site, cut=onset_eff, stop=cut)
            if variant is not None and exclude_days.any() and cut_day > day0
            else np.zeros(0, dtype=int)
        )
        both = common.forecast_many(
            cache=world_cache, site=site, cut=cut, indexes=[index, recheck_index]
        )
        if both is None:
            continue
        predicted = both[0]
        recheck_overlap = float("nan")
        if len(recheck_index):
            days_r = common.utc_day_of(cache=cache, index=recheck_index)
            capacity = features.table.effective_capacity_mw / 100.0
            day_r, resid_r = common.daily_means(
                days=days_r, values=(y_node[recheck_index] - both[1]) * capacity
            )
            resid2 = resid.copy()
            resid2[dt.day_index(days=day_r)] = resid_r
            noise2 = noise_after(residual=resid2, base_noise=base_noise, start=day0, stop=cut_day)
            assert variant is not None
            spans2 = dt.detect(
                variant=variant,
                stream=dt.Stream(residual=resid2, noise=noise2, level=level),
                others=others,
                first=max(day0 - 35, 40),
                last=cut_day - 1,
            )
            flag2 = dt.flagged_days(spans=spans2, cut_day=cut_day, cap=CAP_DAYS)
            recheck_overlap = float((exclude_days & flag2).sum() / max(1, exclude_days.sum()))
        clean_pred = ctx.clean_forecast(site=site, cut=cut, stop=stop, index=index)
        truth = features.y[index]
        observed = y_node[index]
        sd = 1.4826 * float(
            np.nanmedian(np.abs(truth - clean_pred - np.nanmedian(truth - clean_pred)))
        )
        sd = max(sd, 1e-3)
        crps, tw = gaussian_scores(centre=predicted, sd=sd, actual=truth, threshold=ctx.threshold)
        days = common.utc_day_of(cache=cache, index=index)
        day_unique, resid_days = common.daily_means(
            days=days, values=(observed - predicted) * features.table.effective_capacity_mw / 100.0
        )
        resid[dt.day_index(days=day_unique)] = resid_days
        labels = window_labels(times=cache.end_times[index], onset=onset_eff, end=end)
        for label, rows in labels.items():
            if not rows.any():
                continue
            records.append(
                {
                    "label": label,
                    "cut": cut,
                    "n_rows": int(rows.sum()),
                    "bias": float(np.mean(predicted[rows] - clean_pred[rows])),
                    "transfer": float(np.mean(delta[index][rows])),
                    "mae_vs_nra": float(np.mean(np.abs(predicted[rows] - truth[rows]))),
                    "mae_clean_vs_nra": float(np.mean(np.abs(clean_pred[rows] - truth[rows]))),
                    "mae_vs_observed": float(np.mean(np.abs(predicted[rows] - observed[rows]))),
                    "mean_error_vs_nra": float(np.mean(predicted[rows] - truth[rows])),
                    "crps_vs_nra": float(np.mean(crps[rows])),
                    "twcrps_vs_nra": float(np.mean(tw[rows])),
                    "n_flagged": int(exclude_days.sum()),
                    "recheck_overlap": recheck_overlap,
                }
            )
        flagged_total += int(exclude_days[cut_day - 31 : cut_day].sum()) if cut_day > 31 else 0
    ever = (
        dt.flagged_days(spans=spans, cut_day=dt.N_DAYS)
        if spans
        else np.zeros(dt.N_DAYS, dtype=bool)
    )
    near = np.zeros(dt.N_DAYS, dtype=bool)
    if true_last >= true_first:
        near[max(0, true_first - 3) : true_last + 4] = True
    flagged_true = int((ever & near).sum())
    flagged_outside = int((ever & ~near).sum())
    window = slice(day0, dt.N_DAYS)
    streams = pl.DataFrame(
        {"day": np.arange(dt.N_DAYS)[window], "residual_mw": resid[window]}
        | ({"partner_mw": partner_stream[window]} if partner_stream is not None else {})
    ).filter(pl.col("residual_mw").is_not_nan() & pl.col("residual_mw").is_not_null())
    for r in records:
        r["flagged_days_true"], r["flagged_days_outside"] = flagged_true, flagged_outside
    return records, streams


def _partner_stream(*, ctx, scenario, end, partner, cuts, stops) -> np.ndarray:  # noqa: ANN001
    """The partner's daily residual when it loses the node's megawatts, fitted at the same cuts."""
    key = f"{scenario.tag}_{ctx.onset:%Y%m}"
    if key in ctx.partner_cache:
        return ctx.partner_cache[key]
    delta = delta_points(ctx=ctx, site=partner, scenario=scenario, end=end, sign=-1.0)
    y = ctx.cache.sites[partner].y + delta
    world_cache = replace_site(cache=ctx.cache, site=partner, y=y)
    capacity = ctx.cache.sites[partner].table.effective_capacity_mw / 100.0
    resid = ctx.clean[partner].residual.copy()
    day0 = int((np.datetime64(ctx.onset.replace(tzinfo=None), "D") - dt.DAY0).astype(int))
    resid[day0:] = np.nan
    for cut, stop in zip(cuts, stops, strict=True):
        index = common.target_rows(cache=ctx.cache, site=partner, cut=cut, stop=stop)
        if len(index) == 0:
            continue
        predicted = common.forecast_month(cache=world_cache, site=partner, cut=cut, index=index)
        if predicted is None:
            continue
        days, values = common.daily_means(
            days=common.utc_day_of(cache=ctx.cache, index=index),
            values=(y[index] - predicted) * capacity,
        )
        resid[dt.day_index(days=days)] = values
    ctx.partner_cache[key] = resid
    return resid


_CACHE: FeatureCache | None = None
_ARGS: dict = {}


def _job(job: tuple[str, datetime]) -> str:
    torch.set_num_threads(1)
    site, onset = job
    assert _CACHE is not None
    stage = _ARGS["stage"]
    out_dir = common.WORLDS_DIR / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{site}_{onset:%Y%m}.parquet"
    if target.exists():
        return f"{target.name}: already done"
    ctx = JobContext(cache=_CACHE, site=site, onset=onset)
    rows, streams = [], []
    modes = _ARGS["modes"]
    control = stage == "control"
    for scenario in [Scenario(0.0, "share", None)] if control else scenarios(stage=stage):
        if scenario.metered and ctx.partner_name is None:
            continue
        for mode, variant in modes:
            records, stream = run_world(
                ctx=ctx, scenario=scenario, mode=mode, variant=variant, control=control
            )
            for r in records:
                rows.append(
                    {
                        "site": site,
                        "onset": onset,
                        "tag": scenario.tag,
                        "mode": mode,
                        "size": scenario.size,
                        "shape": scenario.shape,
                        "duration": -1 if scenario.duration is None else scenario.duration,
                        "metered": scenario.metered,
                        "partner": ctx.partner_name or "",
                        **r,
                    }
                )
            if mode == "none":
                streams.append(
                    stream.with_columns(
                        site=pl.lit(site), onset=pl.lit(onset), tag=pl.lit(scenario.tag)
                    )
                )
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    pl.DataFrame(rows).write_parquet(tmp)
    if streams:
        pl.concat(streams, how="diagonal").write_parquet(out_dir / f"streams_{target.name}")
    tmp.rename(target)
    return f"{target.name}: {len(rows)} rows"


def main() -> None:
    """Run every (site, onset) job in parallel; finished jobs are skipped on rerun."""
    global _CACHE  # noqa: PLW0603
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("detect", "clean", "control"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sites", nargs="+", default=list(common.INJECT_SITES))
    parser.add_argument(
        "--variants", default="", help="JSON list of {kind,k,w,base,label} for stage clean"
    )
    args = parser.parse_args()
    common.WORLDS_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE = common.load_cache()
    if args.stage == "detect":
        _ARGS.update(stage="detect", modes=[("none", None)])
        onsets = (*SCORE_ONSETS, *TUNE_ONSETS)
    else:
        specs = json.loads(args.variants)
        modes = [("none", None), ("oracle", None)]
        modes += [
            (
                s["label"],
                dt.Variant(kind=s["kind"], k=s["k"], w=s.get("w", 3), base=s.get("base", "A")),
            )
            for s in specs
        ]
        if args.stage == "control":
            modes = [m for m in modes if m[0] != "oracle"]
        _ARGS.update(stage=args.stage, modes=modes)
        onsets = (datetime(2022, 1, 1, tzinfo=UTC),) if args.stage == "control" else SCORE_ONSETS
    jobs = [(site, onset) for site in args.sites for onset in onsets]
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
