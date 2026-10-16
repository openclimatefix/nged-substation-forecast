"""Known-answer injections for experiment 00063: S1 (solar farm output) and S2 (transfers).

Both keep the baseline frozen: the injected series replaces the metered power and the joint fit is
refitted on all months (no month held out), with the mean of u taken over all months. A fit that
refit the XGBoost baseline on every injection would need about 1,300 more XGBoost fits.

**S1.** The 15-minute-to-half-hour AC output of each real solar farm (experiment 00045's farms A to
F, each divided by its own 99th percentile) is multiplied by the injected capacity, 10%, 25%, or
50% of the Primary's mean season-year constant capacity, and subtracted from the metered power
(embedded solar lowers the net power). The recovered added capacity is the change in `k` times the
Primary's mean capacity on the fitted rows. Rows where the farm has no output in daylight (a
dropout or a dropped farm-week) leave both fits.

**S2.** On each Primary, 40 intervals of 6, 12, 24, or 72 hours start at random outside every
logged and labelled switching period, half in April to September. Each interval adds `beta` times
the real net power of another MW-metered series (the donor, drawn at random among those with data)
to the Primary, with `beta` in 0.1, 0.25, 0.5. A further 40 placebo intervals add nothing
(`beta = 0`). The interval is given to the fit as one more transfer column, and the interval's
rows, night included, join the fit.

    uv run python ml-experiments/00063-*/e00063_inject.py
"""

from __future__ import annotations

import itertools
import time
from typing import Final

import e00063_common as common
import e00063_fit as fit
import numpy as np
import polars as pl

FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
SHARES: Final[tuple[float, ...]] = (0.10, 0.25, 0.50)
"""S1: injected capacity as a share of the Primary's mean capacity."""
DURATIONS_HOURS: Final[tuple[int, ...]] = (6, 12, 24, 72)
TRANSFER_SHARES: Final[tuple[float, ...]] = (0.1, 0.25, 0.5)
N_INTERVALS: Final[int] = 40
"""S2: injected intervals per Primary, and also placebo intervals per Primary."""
MIN_OK_SHARE: Final[float] = 0.9
"""An S2 interval needs this share of its half-hours present, unblocked, and with a donor value."""
MAX_ATTEMPTS: Final[int] = 400


def _farm_output(*, design: fit.Design, farm: str) -> np.ndarray:
    """Return the farm's normalised output at the design's stamps (NaN where none)."""
    farm_rows = (
        pl.scan_parquet(common.FARM_HALF_HOURLY_PATH)
        .filter(pl.col("farm") == farm)
        .select(time=pl.col("time").dt.replace_time_zone(None), value=pl.col("farm_norm"))
        .collect()
    )
    stamps = pl.DataFrame({"time": design.time})
    joined = stamps.join(farm_rows, on="time", how="left", maintain_order="left")
    return joined["value"].cast(pl.Float64).fill_null(np.nan).to_numpy()


def run_s1(*, design: fit.Design) -> pl.DataFrame:
    """Inject each farm at each share into one Primary and refit on all, summer, and winter rows."""
    anomalies = fit.anomaly(design=design, exclude_month=None)
    base_rows = fit.fit_rows_mask(design=design, capacity="const", include_transfer=True)
    seasons = {
        "all": np.ones(design.n_rows, dtype=bool),
        "summer": design.summer,
        "winter": ~design.summer,
    }
    out: list[dict[str, float | int | str]] = []
    for farm in FARMS:
        output = _farm_output(design=design, farm=farm)
        farm_valid = np.isfinite(output) | ~design.daylight
        injected_unit = np.where(np.isfinite(output), output, 0.0)
        for season, season_rows in seasons.items():
            rows = base_rows & farm_valid & season_rows
            k0, _, used = fit.fit_arm(
                design=design,
                anomalies=anomalies,
                capacity="const",
                include_transfer=True,
                rows=rows,
            )
            mean_capacity = float(design.capacity["const"][used].mean())
            for share in SHARES:
                injected_mw = share * mean_capacity
                k1, _, used_1 = fit.fit_arm(
                    design=design,
                    anomalies=anomalies,
                    capacity="const",
                    include_transfer=True,
                    rows=rows,
                    target=design.y - injected_mw * injected_unit,
                )
                recovered_mw = (k1 - k0) * mean_capacity
                out.append(
                    {
                        "series": design.series,
                        "farm": farm,
                        "share": share,
                        "season": season,
                        "injected_mw": injected_mw,
                        "recovered_mw": recovered_mw,
                        "ratio": recovered_mw / injected_mw,
                        "n_rows": int(used_1.sum()),
                    }
                )
    return pl.DataFrame(out)


def _interval_rows(
    *,
    design: fit.Design,
    start: int,
    n_half_hours: int,
    ok: np.ndarray,
) -> np.ndarray | None:
    """Return the mask of the interval's usable rows, or `None` when too few rows are usable."""
    first = design.time[start]
    last = first + np.timedelta64(30 * n_half_hours, "m")
    stop = int(np.searchsorted(design.time, last, side="left"))
    mask = np.zeros(design.n_rows, dtype=bool)
    mask[start:stop] = ok[start:stop]
    return mask if mask.sum() >= MIN_OK_SHARE * n_half_hours else None


def run_s2(*, design: fit.Design) -> pl.DataFrame:
    """Inject transfers and placebo intervals into one Primary and refit for the recovered share."""
    rng = np.random.default_rng(design.series * 1000 + 63)
    anomalies = fit.anomaly(design=design, exclude_month=None)
    base_rows = fit.fit_rows_mask(design=design, capacity="const", include_transfer=True)
    ok = (
        design.eligible
        & ~design.blocked
        & ~design.in_label
        & np.isfinite(anomalies)
        & np.isfinite(design.nr["q50"])
        & np.isfinite(design.capacity["const"])
    )
    donors = [s for s in common.SERIES_IDS if s != design.series]
    combos = list(itertools.product(DURATIONS_HOURS, TRANSFER_SHARES))
    plans: list[tuple[str, int, float]] = [
        ("injected", *combos[i % len(combos)]) for i in range(N_INTERVALS)
    ] + [("placebo", DURATIONS_HOURS[i % len(DURATIONS_HOURS)], 0.0) for i in range(N_INTERVALS)]
    out: list[dict[str, float | int | str]] = []
    for kind in ("injected", "placebo"):
        kind_plans = [p for p in plans if p[0] == kind]
        summer_flags = rng.permutation(
            [True] * (len(kind_plans) // 2) + [False] * (len(kind_plans) - len(kind_plans) // 2)
        )
        for (_, duration, beta), summer in zip(kind_plans, summer_flags, strict=True):
            candidates = np.flatnonzero(ok & (design.summer == summer))
            for _ in range(MAX_ATTEMPTS):
                start = int(rng.choice(candidates))
                n_half_hours = 2 * duration
                rows = _interval_rows(design=design, start=start, n_half_hours=n_half_hours, ok=ok)
                if rows is None:
                    continue
                donor = int(rng.choice(donors))
                donor_power = design.power_mw[donor]
                usable = rows & np.isfinite(donor_power)
                if usable.sum() < MIN_OK_SHARE * n_half_hours:
                    continue
                x = np.where(usable, donor_power, 0.0)
                target = design.y + beta * x
                _, shares, _ = fit.fit_arm(
                    design=design,
                    anomalies=anomalies,
                    capacity="const",
                    include_transfer=True,
                    rows=base_rows,
                    target=target,
                    extra_columns=x[:, None],
                    extra_rows=usable,
                )
                beta_hat = float(shares[-1])
                out.append(
                    {
                        "series": design.series,
                        "kind": kind,
                        "beta": beta,
                        "duration_hours": duration,
                        "season": "summer" if summer else "winter",
                        "donor": donor,
                        "n_rows": int(usable.sum()),
                        "beta_hat": beta_hat,
                        "error": beta_hat - beta,
                    }
                )
                break
            else:
                out.append(
                    {
                        "series": design.series,
                        "kind": kind,
                        "beta": beta,
                        "duration_hours": duration,
                        "season": "summer" if summer else "winter",
                        "donor": -1,
                        "n_rows": 0,
                        "beta_hat": float("nan"),
                        "error": float("nan"),
                    }
                )
    return pl.DataFrame(out)


def main() -> None:
    """Run S1 and S2 for every allowed series and write `s1.parquet` and `s2.parquet`."""
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    s1_tables, s2_tables = [], []
    for series in common.SERIES_IDS:
        design = fit.build_design(series=series)
        if design is None:
            continue
        start = time.time()
        s1_tables.append(run_s1(design=design))
        s2_tables.append(run_s2(design=design))
        print(f"series {series}: S1 and S2 done in {time.time() - start:.0f} s", flush=True)
    pl.concat(s1_tables).write_parquet(common.RESULTS_DIR / "s1.parquet")
    pl.concat(s2_tables).write_parquet(common.RESULTS_DIR / "s2.parquet")


if __name__ == "__main__":
    main()
