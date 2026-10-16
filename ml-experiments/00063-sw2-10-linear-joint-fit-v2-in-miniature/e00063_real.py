"""R1: how much of the residual inside each real labelled event the transfer regressor explains.

For each real visible event of experiment 00041 that names a donor and a receiver, both metered in
MW, and for each of the two that is an allowed Primary, the residual of the joint fit (the metered
power minus the baseline without radiation minus the fitted solar term, with `k` from the
all-months fit over every row from 2022-01, the first month of the baseline files) is regressed
on the transfer regressor with an intercept inside the event. The statistic is the R-squared of
that regression. The same statistic is computed on 20 placebo windows
per event: the event's window shifted by a whole number of weeks, so the time of day and the length
are unchanged, into rows that lie outside every logged closure and labelled event. The regressor of
a placebo window is the same pair of series at the shifted time. R1 is descriptive: each event has
one parameter and no held-out data.

    uv run python ml-experiments/00063-*/e00063_real.py
"""

from __future__ import annotations

from typing import Final

import e00063_common as common
import e00063_fit as fit
import numpy as np
import polars as pl

N_PLACEBOS: Final[int] = 20
MAX_SHIFT_WEEKS: Final[int] = 52
MIN_EVENT_ROWS: Final[int] = 6
MIN_PLACEBO_MATCH: Final[float] = 0.8
"""A placebo window needs this share of the event's half-hours present, unblocked, and defined."""
WEEK: Final[np.timedelta64] = np.timedelta64(7 * 24 * 60, "m")


def _r_squared(*, residual: np.ndarray, regressor: np.ndarray) -> tuple[float, float]:
    """Return the R-squared and slope of `residual` on `regressor` with an intercept."""
    centred_x = regressor - regressor.mean()
    centred_r = residual - residual.mean()
    total = float((centred_r**2).sum())
    spread = float((centred_x**2).sum())
    if total <= 0.0 or spread <= 0.0:
        return float("nan"), float("nan")
    slope = float((centred_x * centred_r).sum() / spread)
    explained = slope**2 * spread
    return explained / total, slope


def _residual(*, design: fit.Design) -> np.ndarray:
    """Return `y - baseline - k * (-C * a)`, with `k` from the all-months fit."""
    anomalies = fit.anomaly(design=design, exclude_month=None)
    rows = fit.fit_rows_mask(design=design, capacity="const", include_transfer=True)
    k, _, _ = fit.fit_arm(
        design=design,
        anomalies=anomalies,
        capacity="const",
        include_transfer=True,
        rows=rows,
    )
    return design.y - design.nr["q50"] + k * design.capacity["const"] * anomalies


def _shifted_index(*, design: fit.Design, index: np.ndarray, weeks: int) -> np.ndarray | None:
    """Return the rows `weeks` later than `index`, or `None` when any stamp is missing."""
    wanted = design.time[index] + weeks * WEEK
    position = np.searchsorted(design.time, wanted)
    position = np.minimum(position, design.n_rows - 1)
    found = design.time[position] == wanted
    return position if found.all() else None


def _skip_reason(*, event: common.Event) -> str | None:
    """Return why an event cannot give a regressor, or `None` when it can."""
    if event.donor is None:
        return "no donor named"
    if event.side_b is None:
        return "one-sided closure, no second series"
    if not {event.side_a, event.side_b} <= set(common.SERIES_IDS):
        return "a side is not one of the 8 MW-metered Primaries"
    if event.donor not in (event.side_a, event.side_b):
        return "donor is neither side of the closure"
    return None


def _placebo_windows(
    *,
    design: fit.Design,
    event_rows: np.ndarray,
    residual: np.ndarray,
    regressor: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[int, float]]:
    """Return up to `N_PLACEBOS` pairs of (shift in weeks, R-squared) of shifted event windows."""
    clean = design.eligible_event & ~design.blocked & ~design.in_label
    candidates = [w for w in range(-MAX_SHIFT_WEEKS, MAX_SHIFT_WEEKS + 1) if w != 0]
    rng.shuffle(candidates)
    found: list[tuple[int, float]] = []
    for weeks in candidates:
        if len(found) >= N_PLACEBOS:
            break
        shifted = _shifted_index(design=design, index=event_rows, weeks=weeks)
        if shifted is None:
            continue
        ok = clean[shifted] & np.isfinite(residual[shifted]) & np.isfinite(regressor[shifted])
        if ok.sum() < max(MIN_EVENT_ROWS, MIN_PLACEBO_MATCH * event_rows.size):
            continue
        placebo_r2, _ = _r_squared(residual=residual[shifted][ok], regressor=regressor[shifted][ok])
        if np.isfinite(placebo_r2):
            found.append((weeks, placebo_r2))
    return found


def _pair_row(
    *,
    event: common.Event,
    series: int,
    n_rows: int,
    r_squared: float,
    share: float,
    placebo: np.ndarray,
) -> dict[str, float | int | str | bool]:
    """Return one event-Primary pair's table row."""
    has_placebo = placebo.size > 0
    return {
        "card_id": event.card_id,
        "kind": event.kind,
        "series": series,
        "role": "donor" if event.donor == series else "receiver",
        "n_rows": n_rows,
        "r_squared": r_squared,
        "share": share,
        "placebo_n": int(placebo.size),
        "placebo_median": float(np.median(placebo)) if has_placebo else float("nan"),
        "placebo_p95": float(np.percentile(placebo, 95)) if has_placebo else float("nan"),
        "exceeds_placebo_p95": bool(has_placebo and r_squared > np.percentile(placebo, 95)),
        "placebo_rank": float((placebo < r_squared).mean()) if has_placebo else float("nan"),
    }


def run_r1() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Compute R1 for every usable (event, Primary) pair.

    Returns:
        The per-pair table (`card_id`, `kind`, `series`, `role`, `n_rows`, `r_squared`, `share`,
        `placebo_n`, `placebo_median`, `placebo_p95`, `exceeds_placebo_p95`, `placebo_rank`), the
        per-placebo table (`card_id`, `series`, `shift_weeks`, `r_squared`), and the skipped
        events with their reasons.
    """
    rng = np.random.default_rng(63)
    designs = {
        s: fit.build_design(series=s, window_start=common.R1_WINDOW_START)
        for s in common.SERIES_IDS
    }
    residuals = {s: _residual(design=d) for s, d in designs.items() if d is not None}
    pairs: list[dict[str, float | int | str | bool]] = []
    placebos: list[dict[str, float | int | str]] = []
    skipped: list[dict[str, str]] = []
    for event in common.real_visible_events():
        reason = _skip_reason(event=event)
        if reason is not None:
            skipped.append({"card_id": event.card_id, "kind": event.kind, "reason": reason})
            continue
        for series in (event.side_a, event.side_b):
            assert series is not None
            design = designs[series]
            other = event.counterparty(series=series)
            assert other is not None
            if design is None:
                reason = f"series {series} capacity status not allowed"
                skipped.append({"card_id": event.card_id, "kind": event.kind, "reason": reason})
                continue
            sign = 1.0 if event.donor == other else -1.0
            regressor = sign * design.power_mw[other]
            residual = residuals[series]
            mask = common.event_mask(times=design.time, event=event)
            index = np.flatnonzero(
                mask & design.eligible_event & np.isfinite(residual) & np.isfinite(regressor)
            )
            if index.size < MIN_EVENT_ROWS:
                reason = f"series {series}: only {index.size} usable half-hours"
                skipped.append({"card_id": event.card_id, "kind": event.kind, "reason": reason})
                continue
            r_squared, share = _r_squared(residual=residual[index], regressor=regressor[index])
            windows = _placebo_windows(
                design=design,
                event_rows=np.flatnonzero(mask),
                residual=residual,
                regressor=regressor,
                rng=rng,
            )
            placebos.extend(
                {
                    "card_id": event.card_id,
                    "series": series,
                    "shift_weeks": weeks,
                    "r_squared": placebo_r2,
                }
                for weeks, placebo_r2 in windows
            )
            pairs.append(
                _pair_row(
                    event=event,
                    series=series,
                    n_rows=int(index.size),
                    r_squared=r_squared,
                    share=share,
                    placebo=np.array([r2 for _, r2 in windows]),
                )
            )
    return pl.DataFrame(pairs), pl.DataFrame(placebos), pl.DataFrame(skipped)


def main() -> None:
    """Compute R1 and write `r1_pairs.parquet`, `r1_placebos.parquet`, and `r1_skipped.parquet`."""
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs, placebos, skipped = run_r1()
    pairs.write_parquet(common.RESULTS_DIR / "r1_pairs.parquet")
    placebos.write_parquet(common.RESULTS_DIR / "r1_placebos.parquet")
    skipped.write_parquet(common.RESULTS_DIR / "r1_skipped.parquet")
    print(f"R1: {pairs.height} event-Primary pairs, {skipped.height} skipped")


if __name__ == "__main__":
    main()
