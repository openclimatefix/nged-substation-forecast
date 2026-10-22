# ruff: noqa: E501, PERF401, PLR0912, PLR0915
"""Fit and score the cooling hinge on experiment 155's arm D at the stored Primaries.

Reads arm D's forecasts, experiment 111's stored ensemble temperatures, and the feature cache's ERA5
temperature (only for the known-answer injection). Writes `results.json` under `WORK_DIR` and
prints every number the write-up quotes. Run it from the worktree root as

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00198-*/e00198_fit.py
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import e00198_common as common
import numpy as np
import polars as pl
from scipy.stats import rankdata
from studies.dp_features import FeatureCache

N_PLACEBO: Final[int] = 200
N_BOOTSTRAP: Final[int] = 20000
CONFIDENCE: Final[float] = 0.99
BLOCK_DAYS: Final[int] = 14
RUN_ZERO: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
SUPPORTED_BAR: Final[float] = -0.01
INJECT_MW_PER_DEGREE: Final[float] = 0.1
INJECT_TAU: Final[float] = 20.0
SEED: Final[int] = 198


@dataclass
class SiteData:
    """One Primary's arm D rows with the stored ensemble temperature joined on."""

    site: str
    run: np.ndarray
    k: np.ndarray
    lead: np.ndarray
    band: np.ndarray
    hour: np.ndarray
    date_index: np.ndarray
    origin_month: np.ndarray
    origin_day: np.ndarray
    actual: np.ndarray
    mean: np.ndarray
    quantiles: np.ndarray
    period: np.ndarray
    temperature_mean: np.ndarray
    ens_mean_t: np.ndarray
    ens_t: np.ndarray
    capacity_mw: float
    valid_stamp: np.ndarray
    valid_month: np.ndarray
    day_of_year: np.ndarray
    y_points: np.ndarray


def load_site(*, site: str, cache: FeatureCache) -> SiteData:
    """Return arm D's rows (all periods, switching events removed) at a site."""
    frames = []
    for period in ("tuning", "replicate", "scored"):
        frames.append(
            common.read_arm_d(site=site, period=period).with_columns(period=pl.lit(period))
        )
    frame = (
        pl.concat(frames)
        .filter(~pl.col("switching_mask"))
        .with_columns(
            origin_month=pl.col("power_fcst_init_time").dt.year() * 100
            + pl.col("power_fcst_init_time").dt.month(),
            origin_day=(
                pl.col("power_fcst_init_time") - pl.lit(common.RUN_ZERO_UTC)
            ).dt.total_days(),
        )
    )
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    hour = (valid - valid.astype("datetime64[D]")).astype("timedelta64[m]").astype(float) / 60.0
    stored = common.read_ens_store(site=site)
    temperature = stored["temperature"]
    run = frame["run"].to_numpy().astype(int)
    k = np.rint(frame["lead_hours"].to_numpy() * 2).astype(int)
    features = cache.sites[site]
    origin64 = (
        np.datetime64(common.RUN_ZERO_UTC.replace(tzinfo=None), "us")
        + run.astype("timedelta64[D]").astype("timedelta64[us]")
        + np.timedelta64(9, "h")
    )
    assert np.all(valid == origin64 + k.astype("timedelta64[m]").astype("timedelta64[us]") * 30), (
        "k is not half-hours after the run's origin"
    )
    assert np.all(k < temperature.shape[2]), "lead beyond the stored ensemble horizon"
    position = np.searchsorted(cache.end_times, valid)
    assert np.all(position < len(cache.end_times))
    assert np.all(cache.end_times[position] == valid)
    y_points = features.y[position]
    check = np.isfinite(y_points) & np.isfinite(frame["actual"].to_numpy())
    assert np.allclose(y_points[check], frame["actual"].to_numpy()[check], atol=1e-3), (
        "arm D actual is not the cache's points of capacity"
    )
    return SiteData(
        site=site,
        run=run,
        k=k,
        lead=frame["lead_hours"].to_numpy(),
        band=np.minimum((hour // 4).astype(int), common.N_BANDS - 1),
        hour=hour,
        date_index=valid.astype("datetime64[D]").astype(int),
        origin_month=frame["origin_month"].to_numpy(),
        origin_day=frame["origin_day"].to_numpy(),
        actual=frame["actual"].to_numpy().astype(float),
        mean=frame["mean"].to_numpy().astype(float),
        quantiles=frame.select(list(common.QUANTILES)).to_numpy().astype(float),
        period=frame["period"].to_numpy(),
        temperature_mean=np.nanmean(temperature, axis=1),
        ens_mean_t=np.nanmean(temperature, axis=1)[run, k],
        ens_t=temperature,
        capacity_mw=features.table.effective_capacity_mw,
        valid_stamp=valid,
        valid_month=frame["valid_time"].dt.month().to_numpy(),
        day_of_year=frame["valid_time"].dt.ordinal_day().to_numpy(),
        y_points=y_points,
    )


def hinge(*, data: SiteData, tau: float, shift: int | np.ndarray = 0) -> np.ndarray:
    """Return the member-mean hinge max(0, T - tau) of every row, read `shift` runs earlier.

    Rows whose shifted source run is outside the store get `NaN`.
    """
    source = data.run - shift
    valid = (source >= 0) & (source < data.ens_t.shape[0])
    out = np.full(len(data.run), np.nan)
    members = data.ens_t[source[valid], :, data.k[valid]]
    out[valid] = np.nanmean(np.maximum(0.0, members - tau), axis=1)
    return out


def design(*, data: SiteData, h: np.ndarray) -> np.ndarray:
    """Return the hinge times a one-hot of the 4-hour band of the valid time."""
    out = np.zeros((len(h), common.N_BANDS))
    out[np.arange(len(h)), data.band] = h
    return out


def fit_coefficients(*, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the ridge least-squares coefficients."""
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
    gram = x[finite].T @ x[finite] + common.RIDGE * np.eye(x.shape[1])
    return np.linalg.solve(gram, x[finite].T @ y[finite])


def row_loss(*, data: SiteData, shift: np.ndarray) -> np.ndarray:
    """Return the quantile score (twice the mean pinball loss over 11 quantiles) of every row."""
    error = data.actual[:, None] - (data.quantiles + shift[:, None])
    levels = common.LEVELS[None, :]
    return 2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error), axis=1)


def first_scored_valid(*, data: SiteData) -> np.datetime64:
    """Return the earliest valid time of any scored-year row."""
    return data.valid_stamp[data.period == "scored"].min()


def fit_rows(*, data: SiteData, months: tuple[int, ...]) -> np.ndarray:
    """Return the mask of non-scored rows whose origin month is in `months`.

    Rows whose valid time reaches the first scored valid time are dropped, because their actuals are
    scored rows' actuals (leads run to 168 hours).
    """
    before = data.valid_stamp < first_scored_valid(data=data)
    return np.isin(data.origin_month, months) & (data.period != "scored") & before


def scored_rows(*, data: SiteData) -> np.ndarray:
    """Return the mask of scored-year summer rows (valid month June to September)."""
    return (data.period == "scored") & np.isin(data.valid_month, list(common.SUMMER_MONTHS))


FIT_MONTH_KEYS: Final[tuple[int, ...]] = tuple(y * 100 + m for y, m in common.FIT_MONTHS)


def eligible_taus(*, sites: list[SiteData]) -> dict[float, dict]:
    """Return, per tau, the median-site day and daytime half-hour counts above tau in the fit months."""
    out = {}
    for tau in common.TAUS:
        days, halves = [], []
        for data in sites:
            rows = fit_rows(data=data, months=FIT_MONTH_KEYS) & (data.lead <= 24)
            above = data.ens_mean_t[rows] >= tau
            day_max = (
                pl.DataFrame({"d": data.date_index[rows], "t": data.ens_mean_t[rows]})
                .group_by("d")
                .agg(pl.col("t").max())
            )
            days.append(int((day_max["t"] >= tau).sum()))
            daytime = (data.hour[rows] >= 10) & (data.hour[rows] < 18)
            halves.append(
                int(
                    np.unique(
                        np.stack([data.date_index[rows], (data.hour[rows] * 2).astype(int)]).T[
                            above & daytime
                        ],
                        axis=0,
                    ).shape[0]
                )
            )
        out[tau] = {
            "median_days": float(np.median(days)),
            "median_half_hours": float(np.median(halves)),
            "days": days,
            "half_hours": halves,
            "eligible": bool(
                np.median(days) >= common.MIN_DAYS_ABOVE_TAU
                and np.median(halves) >= common.MIN_HALF_HOURS_ABOVE_TAU
            ),
        }
    return out


def lomo(*, sites: list[SiteData], tau: float) -> dict[int, float]:
    """Return the held-out mean loss difference (cooling minus arm D) per fit month."""
    out = {}
    for held in FIT_MONTH_KEYS:
        diffs = []
        for data in sites:
            h = hinge(data=data, tau=tau)
            test = fit_rows(data=data, months=(held,))
            low, high = data.valid_stamp[test].min(), data.valid_stamp[test].max()
            train = fit_rows(data=data, months=tuple(m for m in FIT_MONTH_KEYS if m != held))
            train &= ~((data.valid_stamp >= low) & (data.valid_stamp <= high))
            x = design(data=data, h=h)
            coef = fit_coefficients(x=x[train], y=(data.actual - data.mean)[train])
            shift = np.nan_to_num(x @ coef)
            base = row_loss(data=data, shift=np.zeros(len(h)))
            new = row_loss(data=data, shift=shift)
            diffs.append((new - base)[test])
        out[held] = float(np.concatenate(diffs).mean())
    return out


def fit_and_diff(
    *,
    sites: list[SiteData],
    tau: float,
    fit_shift: int | dict[str, np.ndarray] = 0,
    score_shift: int | dict[str, np.ndarray] = 0,
) -> dict[str, np.ndarray]:
    """Fit on the four fit months and return per-row loss differences on the scored summer rows."""
    diffs, blocks, masks = [], [], []
    for data in sites:
        h_fit = hinge(
            data=data,
            tau=tau,
            shift=fit_shift[data.site] if isinstance(fit_shift, dict) else fit_shift,
        )
        h_score = hinge(
            data=data,
            tau=tau,
            shift=score_shift[data.site] if isinstance(score_shift, dict) else score_shift,
        )
        train = fit_rows(data=data, months=FIT_MONTH_KEYS)
        coef = fit_coefficients(
            x=design(data=data, h=h_fit)[train], y=(data.actual - data.mean)[train]
        )
        shift = np.nan_to_num(design(data=data, h=h_score) @ coef)
        rows = scored_rows(data=data) & np.isfinite(h_score)
        base = row_loss(data=data, shift=np.zeros(len(shift)))
        new = row_loss(data=data, shift=shift)
        diffs.append((new - base)[rows])
        blocks.append(data.origin_day[rows] // BLOCK_DAYS)
        masks.append((data, rows, coef, shift, base, new))
    return {"diff": np.concatenate(diffs), "block": np.concatenate(blocks), "detail": masks}  # ty: ignore[invalid-return-type]


def block_interval(
    *, diff: np.ndarray, block: np.ndarray, seed: int = SEED
) -> tuple[float, float, float]:
    """Return the mean difference and its paired 99% interval from a bootstrap over blocks."""
    ids, inverse = np.unique(block, return_inverse=True)
    sums = np.bincount(inverse, weights=diff)
    counts = np.bincount(inverse).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(N_BOOTSTRAP, len(ids)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(means, [(1 - CONFIDENCE) / 2, 1 - (1 - CONFIDENCE) / 2])
    return float(diff.mean()), float(low), float(high)


def pipeline(*, sites: list[SiteData], label: str, taus: tuple[float, ...] = common.TAUS) -> dict:
    """Run the pre-registered pipeline: eligibility, tau choice, sign check, C1 and its variants."""
    counts = {t: v for t, v in eligible_taus(sites=sites).items() if t in taus}
    print(
        f"[{label}] eligibility (median site): "
        + "; ".join(
            f"tau {t:g}: {v['median_days']:.0f} days, {v['median_half_hours']:.0f} half-hours, "
            f"{'ok' if v['eligible'] else 'DROPPED'}"
            for t, v in counts.items()
        )
    )
    held = {t: lomo(sites=sites, tau=t) for t, v in counts.items() if v["eligible"]}
    if not held:
        return {"counts": counts, "eligible": False}
    mean_held = {t: float(np.mean(list(v.values()))) for t, v in held.items()}
    tau = min(mean_held, key=lambda t: mean_held[t])
    n_negative = sum(1 for v in held[tau].values() if v < 0)
    print(f"[{label}] leave-one-month-out mean held-out difference by tau: {mean_held}")
    print(
        f"[{label}] chosen tau {tau:g}; held-out gain negative in {n_negative} of 4 months: {held[tau]}"
    )
    result = fit_and_diff(sites=sites, tau=tau)
    mean, low, high = block_interval(diff=result["diff"], block=result["block"])
    print(
        f"[{label}] C1 (cooling minus arm D, summer scored rows, 14-day blocks): {mean:+.4f} [{low:+.4f}, {high:+.4f}] n={len(result['diff'])} blocks={len(np.unique(result['block']))}"
    )
    return {
        "counts": counts,
        "held": held,
        "tau": tau,
        "sign_ok": n_negative >= 3,
        "c1": [mean, low, high],
        "result": result,
    }


def hot_day_mask(*, data: SiteData, rows: np.ndarray) -> np.ndarray:
    """Return, for the scored summer rows, True where the row's valid date is a hot day.

    A hot day is a date whose ensemble-mean temperature, read from lead 24 hours or less, reaches
    `HOT_DAY_C` at some half-hour.
    """
    short = rows & (data.lead <= 24)
    frame = pl.DataFrame({"d": data.date_index[short], "t": data.ens_mean_t[short]})
    hot = set(
        frame.group_by("d").agg(pl.col("t").max()).filter(pl.col("t") >= common.HOT_DAY_C)["d"]
    )
    return np.isin(data.date_index[rows], list(hot))


def spells(*, dates: list[int]) -> list[list[int]]:
    """Group sorted day numbers into runs of consecutive days."""
    out: list[list[int]] = []
    for d in sorted(dates):
        if out and d == out[-1][-1] + 1:
            out[-1].append(d)
        else:
            out.append([d])
    return out


def tail_report(*, data: SiteData, shift: np.ndarray, rows: np.ndarray, hot: np.ndarray) -> dict:
    """Return coverage and pinball at q01, q05, q95, q99 on hot and normal scored summer days."""
    out = {}
    for name, mask in (("hot", hot), ("normal", ~hot)):
        sel = np.where(rows)[0][mask]
        for q, level in (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99)):
            col = common.QUANTILES.index(q)
            forecast = data.quantiles[sel, col] + shift[sel]
            err = data.actual[sel] - forecast
            out[f"{name}_{q}_coverage"] = (
                float(np.mean(data.actual[sel] <= forecast)) if len(sel) else float("nan")
            )
            out[f"{name}_{q}_pinball"] = (
                float(np.mean(np.maximum(level * err, (level - 1) * err)))
                if len(sel)
                else float("nan")
            )
        out[f"{name}_rows"] = len(sel)
    return out


def partial_spearman(*, x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Return the Spearman correlation of x and y after removing the rank of `control` from both."""
    rx, ry, rc = rankdata(x), rankdata(y), rankdata(control)
    basis = np.stack([np.ones_like(rc), rc], axis=1)
    resid_x = rx - basis @ np.linalg.lstsq(basis, rx, rcond=None)[0]
    resid_y = ry - basis @ np.linalg.lstsq(basis, ry, rcond=None)[0]
    return float(np.corrcoef(resid_x, resid_y)[0, 1])


def correlations(*, sites: list[SiteData]) -> dict[str, float]:
    """Return per-Primary partial Spearman of afternoon residual with the tau-20 hinge.

    One row per (valid date, half-hour), the shortest-lead forecast of it; afternoon is 12:00 to
    18:00 UTC; days are those whose ensemble-mean day maximum (lead 24 hours or less) is 22 degrees
    or more; the control is the day of year.
    """
    out: dict[str, float] = {}
    for data in sites:
        month = data.valid_month
        base = (data.lead <= 24) & np.isin(month, list(common.SUMMER_MONTHS))
        frame = pl.DataFrame({"d": data.date_index, "t": data.ens_mean_t})
        warm = set(
            frame.filter(pl.Series(base))
            .group_by("d")
            .agg(pl.col("t").max())
            .filter(pl.col("t") >= 22.0)["d"]
        )
        rows = base & (data.hour >= 12) & (data.hour < 18) & np.isin(data.date_index, list(warm))
        index = np.where(rows)[0]
        key = data.valid_stamp[index].astype("datetime64[m]").astype(np.int64)
        first = np.unique(key, return_index=True)[1]
        pick = index[first]
        h = hinge(data=data, tau=INJECT_TAU)[pick]
        residual = (data.actual - data.mean)[pick]
        good = np.isfinite(h) & np.isfinite(residual)
        out[data.site] = partial_spearman(
            x=h[good], y=residual[good], control=data.day_of_year[pick][good].astype(float)
        )
        out[f"{data.site}__rows"] = float(good.sum())
    return out


def block_shifts(*, data: SiteData, rng: np.random.Generator, offsets: list[int]) -> np.ndarray:
    """Return a per-row run shift: one random offset per 14-day block of origin days."""
    blocks = data.origin_day // BLOCK_DAYS
    chosen = {int(b): int(rng.choice(offsets)) for b in np.unique(blocks)}
    return np.array([chosen[int(b)] for b in blocks])


def placebo_gains(*, sites: list[SiteData], tau: float) -> np.ndarray:
    """Return C1 under temperatures laid from other years' same calendar weeks, refitted each draw.

    Each 14-day block of runs reads the ensemble temperature of a randomly chosen week 51 to 53
    weeks away (364 plus or minus 14 days times 0 to 2, in the direction that stays in the store):
    fit rows (2024) read 2025 temperatures, scored rows (2025 and 2026) read 2024 and 2025. Rows
    whose source is outside the store are dropped from both arms. Tau is held at the real choice.
    """
    rng = np.random.default_rng(SEED)
    gains = []
    for _ in range(N_PLACEBO):
        fit_shift = {
            d.site: block_shifts(
                data=d, rng=rng, offsets=[-364 + 14 * j for j in (-2, -1, 0, 1, 2)]
            )
            for d in sites
        }
        score_shift = {
            d.site: block_shifts(data=d, rng=rng, offsets=[364 + 14 * j for j in (-2, -1, 0, 1, 2)])
            for d in sites
        }
        result = fit_and_diff(sites=sites, tau=tau, fit_shift=fit_shift, score_shift=score_shift)
        gains.append(float(np.nanmean(result["diff"])))
    return np.array(gains)


def inject(*, data: SiteData, cache: FeatureCache) -> SiteData:
    """Return a copy whose actuals include a synthetic afternoon cooling load driven by ERA5.

    The load is 0.1 MW per degree above 20 degrees, shaped by a bell centred on 15:00 UTC (standard
    deviation 3 hours), converted to points of effective capacity.
    """
    from dataclasses import replace

    position = np.searchsorted(cache.end_times, data.valid_stamp)
    era5 = cache.sites[data.site].temperature_c[position]
    points_per_degree = INJECT_MW_PER_DEGREE / data.capacity_mw * 100.0
    bell = np.exp(-0.5 * ((data.hour - 15.0) / 3.0) ** 2)
    load = points_per_degree * np.maximum(0.0, era5 - INJECT_TAU) * bell
    return replace(data, actual=data.actual + np.nan_to_num(load))


def oracle_temperature(*, data: SiteData, cache: FeatureCache) -> SiteData:
    """Return a copy whose every ensemble member is the realised ERA5 temperature (post hoc)."""
    from dataclasses import replace

    stored = common.read_ens_store(site=data.site)
    position = np.searchsorted(cache.end_times, stored["origins"])
    steps = np.arange(data.ens_t.shape[2])
    index = np.minimum(position[:, None] + steps[None, :], len(cache.end_times) - 1)
    era5 = cache.sites[data.site].temperature_c[index]
    return replace(data, ens_t=era5[:, None, :].astype(np.float32))


def lead_band_c1(*, result: dict) -> dict[str, list[float]]:
    """Return C1 per lead band."""
    out = {}
    for name, low, high in (("0.5-48 h", 0.0, 48.0), ("48-168 h", 48.0, 168.1)):
        diffs, blocks = [], []
        for data, rows, _coef, _shift, base, new in result["detail"]:
            lead = data.lead[rows]
            keep = (lead > low) & (lead <= high)
            diffs.append((new - base)[rows][keep])
            blocks.append((data.origin_day[rows] // BLOCK_DAYS)[keep])
        out[name] = list(block_interval(diff=np.concatenate(diffs), block=np.concatenate(blocks)))
    return out


def main() -> None:
    """Run the pre-registered analysis and write `results.json`."""
    cache = common.load_cache()
    sites = [load_site(site=s, cache=cache) for s in common.DECISION_SITES]
    sites_d9 = load_site(site="D9", cache=cache)
    real = pipeline(sites=sites, label="five decision Primaries")
    results: dict = {"counts": real["counts"]}
    if not real.get("tau"):
        print("No eligible tau; verdict capped at inconclusive.")
        common.write_json_atomic(path=common.WORK_DIR / "results.json", content=results)
        return
    tau, result = real["tau"], real["result"]
    results.update(tau=tau, held=real["held"], sign_ok=real["sign_ok"], c1=real["c1"])
    results["c1_by_lead"] = lead_band_c1(result=result)
    print("C1 by lead band:", results["c1_by_lead"])
    # Hot days and the hottest spell.
    hot_all, dates_hot, hinge_by_date = [], set(), {}
    for data, rows, _c, _s, _b, _n in result["detail"]:
        hot = hot_day_mask(data=data, rows=rows)
        hot_all.append(hot)
        dates_hot |= set(data.date_index[rows][hot].tolist())
        h = hinge(data=data, tau=tau)[rows]
        for d, value in zip(data.date_index[rows].tolist(), np.nan_to_num(h).tolist(), strict=True):
            hinge_by_date[d] = hinge_by_date.get(d, 0.0) + value
    hot_spells = spells(dates=sorted(dates_hot))
    results["hot_days"] = len(dates_hot)
    results["hot_spells"] = len(hot_spells)
    print(
        f"Hot days (any Primary, day max >= {common.HOT_DAY_C} C): {len(dates_hot)} in {len(hot_spells)} spells"
    )
    diff = result["diff"]
    hot_concat = np.concatenate(hot_all)
    if len(dates_hot) >= 10 and len(hot_spells) >= 3:
        results["c1_hot"] = list(
            block_interval(diff=diff[hot_concat], block=result["block"][hot_concat])
        )
        print("C1 on hot days:", results["c1_hot"], f"rows={int(hot_concat.sum())}")
    else:
        results["c1_hot"] = None
        print(
            f"Hot-day contrast is descriptive only: mean {diff[hot_concat].mean():+.4f} on {int(hot_concat.sum())} rows"
            if hot_concat.any()
            else "No hot-day rows."
        )
    warm_dates: set[int] = set()
    for data, rows, *_ in result["detail"]:
        short = rows & (data.lead <= 24)
        frame = pl.DataFrame({"d": data.date_index[short], "t": data.ens_mean_t[short]})
        warm_dates |= set(
            frame.group_by("d").agg(pl.col("t").max()).filter(pl.col("t") >= tau)["d"].to_list()
        )
    warm_spells = spells(dates=sorted(warm_dates))
    spell_totals = [sum(hinge_by_date.get(d, 0.0) for d in sp) for sp in warm_spells]
    top = warm_spells[int(np.argmax(spell_totals))]
    results["warm_spell_count"] = len(warm_spells)
    print(
        f"Spells of days with an ensemble-mean day maximum >= tau {tau:g}: {len(warm_spells)}; the hottest spans {len(top)} days"
    )
    drop = np.concatenate(
        [np.isin(data.date_index[rows], top) for data, rows, *_ in result["detail"]]
    )
    results["hottest_spell"] = [min(top), max(top)]
    results["c1_without_hottest_spell"] = list(
        block_interval(diff=diff[~drop], block=result["block"][~drop])
    )
    print(
        f"C1 without the hottest spell (day numbers {min(top)} to {max(top)}):",
        results["c1_without_hottest_spell"],
    )
    # Tails.
    tails = {}
    for data, rows, _c, shift, _b, _n in result["detail"]:
        hot = hot_day_mask(data=data, rows=rows)
        tails[data.site] = {
            "arm_d": tail_report(data=data, shift=np.zeros(len(shift)), rows=rows, hot=hot),
            "cooling": tail_report(data=data, shift=shift, rows=rows, hot=hot),
        }
    results["tails"] = tails
    # Absolute scores.
    abs_scores = {}
    for data, rows, _c, _s, base, new in result["detail"]:
        abs_scores[data.site] = {
            "arm_d": float(base[rows].mean()),
            "cooling": float(new[rows].mean()),
        }
    results["absolute_scores"] = abs_scores
    print("Absolute summer scores:", abs_scores)
    # Mean error beside the quantile score (check for trigger 3), median minus mean per site.
    mean_error = {}
    for data, rows, _c, shift, _b, _n in result["detail"]:
        err_base = float(np.mean(np.abs(data.actual[rows] - data.mean[rows])))
        err_new = float(np.mean(np.abs(data.actual[rows] - data.mean[rows] - shift[rows])))
        median_minus_mean = float(
            np.mean(data.quantiles[rows, common.QUANTILES.index("q50")] - data.mean[rows])
        )
        mean_error[data.site] = {
            "mae_arm_d": err_base,
            "mae_cooling": err_new,
            "median_minus_mean": median_minus_mean,
        }
    results["mean_error"] = mean_error
    print("Mean-forecast error and median-minus-mean:", mean_error)
    # Per-site C1 (leave-one-site-out, trigger 4).
    per_site = {}
    for index, data in enumerate(sites):
        others = [s for j, s in enumerate(sites) if j != index]
        r = fit_and_diff(sites=others, tau=tau)
        per_site[f"without_{data.site}"] = list(block_interval(diff=r["diff"], block=r["block"]))
    results["leave_one_site_out"] = per_site
    print("Leave one site out:", per_site)
    # Correlations.
    results["correlations"] = correlations(sites=[*sites, sites_d9])
    print("Partial Spearman, residual vs tau-20 hinge:", results["correlations"])
    # Placebo.
    placebo = placebo_gains(sites=sites, tau=tau)
    results["placebo"] = {
        "p05": float(np.quantile(placebo, 0.05)),
        "median": float(np.median(placebo)),
        "p95": float(np.quantile(placebo, 0.95)),
        "n": len(placebo),
    }
    results["c1_below_placebo_p05"] = bool(real["c1"][0] < results["placebo"]["p05"])
    print(
        "Placebo gains:",
        results["placebo"],
        "C1 below placebo 5th percentile:",
        results["c1_below_placebo_p05"],
    )
    # Known answer at D10.
    d10 = next(s for s in sites if s.site == "D10")
    d10_injected = inject(data=d10, cache=cache)
    injected = pipeline(sites=[d10_injected], label="known answer at D10")
    if injected.get("tau") is not None:
        data, rows, _coef, shift, *_ = injected["result"]["detail"][0]
        load = d10_injected.actual - d10.actual
        hot_rows = rows & (load > 0)
        fitted = float(np.nansum(shift[hot_rows]))
        injected_sum = float(np.nansum(load[hot_rows]))
        by_lead = {}
        for name, low, high in (("0.5-48 h", 0.0, 48.0), ("48-168 h", 48.0, 168.1)):
            sel = hot_rows & (data.lead > low) & (data.lead <= high)
            by_lead[name] = float(np.nansum(shift[sel]) / np.nansum(load[sel]))
        results["known_answer"] = {
            "tau": injected["tau"],
            "c1": injected["c1"],
            "fitted_sum": fitted,
            "injected_sum": injected_sum,
            "ratio": fitted / injected_sum if injected_sum else float("nan"),
            "ratio_by_lead": by_lead,
            "hot_rows": int(hot_rows.sum()),
        }
        results["known_answer"]["passes"] = bool(
            abs(results["known_answer"]["ratio"] - 1) <= 0.25 and injected["c1"][2] < SUPPORTED_BAR
        )
    else:
        results["known_answer"] = {"passes": False, "reason": "no eligible tau"}
    print("Known answer:", results["known_answer"])
    # Post hoc diagnostics of the failed known-answer check (labelled post hoc in the write-up).
    post_hoc = {}
    for name, variant, taus in (
        ("tau_20_forced", d10_injected, (INJECT_TAU,)),
        (
            "oracle_era5_temperature",
            oracle_temperature(data=d10_injected, cache=cache),
            common.TAUS,
        ),
        ("oracle_era5_tau_20", oracle_temperature(data=d10_injected, cache=cache), (INJECT_TAU,)),
    ):
        run_ = pipeline(sites=[variant], label=f"post hoc known answer: {name}", taus=taus)
        if run_.get("tau") is None:
            post_hoc[name] = None
            continue
        data, rows, _coef, shift, *_ = run_["result"]["detail"][0]
        load = d10_injected.actual - d10.actual
        hot = rows & (load > 0)
        post_hoc[name] = {
            "tau": run_["tau"],
            "c1": run_["c1"],
            "sign_ok": run_["sign_ok"],
            "ratio": float(np.nansum(shift[hot]) / np.nansum(load[hot])),
        }
        print("Post hoc", name, post_hoc[name])
    results["post_hoc_known_answer"] = post_hoc
    c1 = results["c1"]
    conditions = {
        "interval_below_bar": c1[2] < SUPPORTED_BAR,
        "below_zero_without_hottest_spell": results["c1_without_hottest_spell"][2] < 0,
        "below_placebo_p05": results["c1_below_placebo_p05"],
        "known_answer": results["known_answer"]["passes"],
        "sign_check": results["sign_ok"],
    }
    if not (results["sign_ok"] and results["known_answer"]["passes"]):
        verdict = "inconclusive (a control or the sign check failed)"
    elif all(conditions.values()):
        verdict = "supported"
    elif (
        c1[2] < 0
        and conditions["below_zero_without_hottest_spell"]
        and conditions["below_placebo_p05"]
    ):
        verdict = "small gain"
    elif c1[1] > 0:
        verdict = "refuted"
    else:
        verdict = "not supported"
    results["conditions"], results["forecast_verdict"] = conditions, verdict
    print("Conditions:", conditions, "-> forecast verdict:", verdict)
    common.write_json_atomic(path=common.WORK_DIR / "results.json", content=results)


if __name__ == "__main__":
    main()
