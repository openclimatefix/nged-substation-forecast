"""The joint ordinary-least-squares fit of experiment 00063 and its leave-one-month-out scoring.

For a Primary p with metered power y, a baseline b (the median of a leave-one-month-out XGBoost
model), the capacity C of the week, the CAMS unit PV shape u, and its anomaly a (u minus the mean
of u at the same calendar month and half-hour of the day over the training months), the fit is

    y - b = k * (-C * a) + sum_j beta_j * x_j * 1[t in I_j] + error

with one scale `k` on the capacity and one share `beta_j` per labelled transfer `j`: `x_j` is the
power of the donor (the receiver gains) or, when p is the donor, minus the power of the receiver.
The fit has no intercept. The held-out forecast of the median is `b + k * (-C * a)`: transfer terms
are zero outside switching, so the scored rows test the solar term only.

Arms scored on the same rows (`run_arms`): the frozen v1 baseline with radiation (`v1`), the
baseline without radiation (`control`), and the joint fit with the season-year constant capacity
(`c1`), with the scale fixed at 1 (`c2`), with the weekly capacity (`c3`), without the transfer
terms (`c4`), and with the previous season-year's constant capacity (`c5`).

    uv run python ml-experiments/00063-*/e00063_fit.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

import e00063_common as common
import numpy as np
import polars as pl

N_CELLS: Final[int] = 12 * 48
"""Calendar months by half-hours of the day: the cells of the mean of u."""
ARM_NAMES: Final[tuple[str, ...]] = ("v1", "control", "c1", "c2", "c3", "c4", "c5")


@dataclass
class Design:
    """One Primary's half-hours as numpy arrays, with the row roles the fits need.

    Every array has one entry per half-hour (`n` rows), sorted by time, except where stated.

    Attributes:
        series: The `time_series_id`.
        norm_mw: The capacity that errors are normalised by.
        time: Stamps, `datetime64[us]`, UTC.
        month_id: Year times 12 plus month minus 1.
        cell: Calendar month (0 to 11) times 48 plus half-hour (0 to 47), the cell of the mean of u.
        summer: Whether the month is April to September.
        y: Metered power, MW.
        v1: Medians of the frozen baseline with radiation (`v1_q10`, `v1_q50`, `v1_q90`), MW.
        nr: Medians of the baseline without radiation (`nr_q10`, `nr_q50`, `nr_q90`), MW.
        u: CAMS unit PV shape.
        capacity: Capacity in MW by variant: `const`, `weekly`, and `previous` (NaN when none).
        daylight: Zenith under 75 degrees and a positive CAMS shape.
        eligible: The row filters of `e00063_common.load_series`.
        eligible_event: The rows that may lie inside a labelled event: `eligible`, or with
            `E63_EVENT_ROWS=any_week` also the rows of `unobserved` weeks.
        blocked: Inside a logged closure (padded 24 hours).
        in_label: Inside any labelled visible event that touches the series.
        member: Shape (n, J). Row i is inside event j and event j has a usable regressor.
        regressor: Shape (n, J). The transfer regressor (MW), zero outside the event.
        event_ids: The card id of each of the J usable events.
        power_mw: The metered power of every series in `common.SERIES_IDS` at these stamps, MW
            (NaN where missing).
    """

    series: int
    norm_mw: float
    time: np.ndarray
    month_id: np.ndarray
    cell: np.ndarray
    summer: np.ndarray
    y: np.ndarray
    v1: dict[str, np.ndarray]
    nr: dict[str, np.ndarray]
    u: np.ndarray
    capacity: dict[str, np.ndarray]
    daylight: np.ndarray
    eligible: np.ndarray
    eligible_event: np.ndarray
    blocked: np.ndarray
    in_label: np.ndarray
    member: np.ndarray
    regressor: np.ndarray
    event_ids: list[str] = field(default_factory=list)
    power_mw: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        """Return the number of half-hours."""
        return self.y.shape[0]

    @property
    def solar_rows(self) -> np.ndarray:
        """Return the solar-term rows: eligible daylight rows outside blocks and labels."""
        return self.eligible & self.daylight & ~self.blocked & ~self.in_label

    @property
    def transfer_rows(self) -> np.ndarray:
        """Return the eligible rows inside a labelled event with a usable regressor."""
        return self.eligible_event & self.member.any(axis=1)


def build_design(*, series: int, window_start: datetime = common.WINDOW_START) -> Design | None:
    """Build a Primary's `Design`, or `None` when its capacity status is not allowed.

    Args:
        series: A `time_series_id` in `common.SERIES_IDS`.
        window_start: The first stamp of the design.

    Returns:
        The design, with the labelled events' regressors taken from the frozen power files.
    """
    rows = common.load_series(series=series, window_start=window_start)
    if rows.status not in common.ALLOWED_STATUSES:
        return None
    frame = rows.frame
    power = common.power_wide()
    times = common.times_of(frame=frame)
    power_here = frame.select("valid_time").join(
        power, on="valid_time", how="left", maintain_order="left"
    )
    in_label = np.zeros(frame.height, dtype=bool)
    members: list[np.ndarray] = []
    regressors: list[np.ndarray] = []
    ids: list[str] = []
    for event in common.real_visible_events():
        if not event.touches(series=series):
            continue
        mask = common.event_mask(times=times, event=event)
        in_label |= mask
        other = event.counterparty(series=series)
        usable = (
            event.donor is not None
            and event.donor in common.SERIES_IDS
            and other in common.SERIES_IDS
            and event.donor in (series, other)
        )
        if not usable:
            continue
        assert other is not None
        sign = 1.0 if event.donor == other else -1.0
        regressor_here = sign * power_here[f"p{other}"]
        x = np.where(mask, regressor_here.fill_null(np.nan).to_numpy(), 0.0)
        member = mask & np.isfinite(x)
        if not member.any():
            continue
        members.append(member)
        regressors.append(np.where(member, x, 0.0))
        ids.append(event.card_id)
    n_events = len(ids)

    def _array(name: str) -> np.ndarray:
        return frame[name].cast(pl.Float64).fill_null(np.nan).to_numpy()

    return Design(
        series=series,
        norm_mw=rows.capacity_norm_mw,
        time=times,
        month_id=frame["month_id"].to_numpy(),
        cell=(
            (frame["cal_month"].cast(pl.Int32) - 1) * 48 + frame["half_hour"].cast(pl.Int32)
        ).to_numpy(),
        summer=frame["cal_month"].is_in(list(common.SUMMER_MONTHS)).to_numpy(),
        y=_array("obs_mw"),
        v1={q: _array(f"v1_{q}") for q in ("q10", "q50", "q90")},
        nr={q: _array(f"nr_{q}") for q in ("q10", "q50", "q90")},
        u=_array("u"),
        capacity={
            "const": _array("capacity_const"),
            "weekly": _array("capacity_weekly"),
            "previous": _array("capacity_previous"),
        },
        daylight=frame["daylight"].fill_null(False).to_numpy(),
        eligible=frame["eligible"].to_numpy(),
        eligible_event=frame[
            "eligible_any_week" if common.EVENT_ROWS_ANY_WEEK else "eligible"
        ].to_numpy(),
        blocked=frame["blocked"].fill_null(True).to_numpy(),
        in_label=in_label,
        member=np.column_stack(members) if n_events else np.zeros((frame.height, 0), dtype=bool),
        regressor=np.column_stack(regressors) if n_events else np.zeros((frame.height, 0)),
        event_ids=ids,
        power_mw={s: power_here[f"p{s}"].fill_null(np.nan).to_numpy() for s in common.SERIES_IDS},
    )


def cell_means(*, design: Design, exclude_month: int | None) -> np.ndarray:
    """Return the mean of u in each (calendar month, half-hour) cell, NaN for an empty cell.

    The mean runs over every eligible row of the design outside month `exclude_month` (the whole
    series when `None`): night rows count, because u is zero there.

    Args:
        design: The design.
        exclude_month: A `month_id` to leave out, or `None`.

    Returns:
        An array of `N_CELLS` means.
    """
    keep = design.eligible & np.isfinite(design.u)
    if exclude_month is not None:
        keep &= design.month_id != exclude_month
    total = np.bincount(design.cell[keep], weights=design.u[keep], minlength=N_CELLS)
    count = np.bincount(design.cell[keep], minlength=N_CELLS)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def anomaly(*, design: Design, exclude_month: int | None) -> np.ndarray:
    """Return u minus its cell mean for every row (NaN where the cell has no training row)."""
    return design.u - cell_means(design=design, exclude_month=exclude_month)[design.cell]


def solve(
    *, solar_column: np.ndarray, transfer_columns: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray]:
    """Solve the least-squares fit and return the scale `k` and the transfer shares.

    Args:
        solar_column: `-C * a` on the fitted rows, shape (m,).
        transfer_columns: The transfer regressors on the fitted rows, shape (m, J).
        target: `y - b` on the fitted rows, shape (m,).

    Returns:
        `k`, and the J shares (a share whose event has no fitted row is zero).
    """
    matrix = np.column_stack([solar_column, transfer_columns])
    coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
    return float(coefficients[0]), coefficients[1:]


def fit_rows_mask(*, design: Design, capacity: str, include_transfer: bool) -> np.ndarray:
    """Return the rows a fit uses before a month is held out, for one capacity variant."""
    rows = design.solar_rows | design.transfer_rows if include_transfer else design.solar_rows
    return rows & np.isfinite(design.capacity[capacity])


def fit_arm(
    *,
    design: Design,
    anomalies: np.ndarray,
    capacity: str,
    include_transfer: bool,
    rows: np.ndarray,
    target: np.ndarray | None = None,
    extra_columns: np.ndarray | None = None,
    extra_rows: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit `k` and the shares on `rows` (plus `extra_rows`) and return them with the row mask used.

    Args:
        design: The design.
        anomalies: The anomaly of u for every row.
        capacity: `const`, `weekly`, or `previous`.
        include_transfer: Whether the labelled events' columns enter the fit.
        rows: The rows allowed (from `fit_rows_mask`, with any month already left out).
        target: The series to fit, defaults to `y - nr_q50`.
        extra_columns: Further regressor columns of shape (n, E) (an injected transfer), or `None`.
        extra_rows: Further rows (shape (n,)) to fit, such as an injected interval's rows.

    Returns:
        `k`, the coefficients of the labelled events' columns followed by those of the extra
        columns, and the mask of rows the fit used.
    """
    target = design.y - design.nr["q50"] if target is None else target - design.nr["q50"]
    used = rows & np.isfinite(anomalies) & np.isfinite(target)
    if extra_rows is not None:
        extra_ok = (
            extra_rows
            & design.eligible
            & ~design.blocked
            & ~design.in_label
            & np.isfinite(anomalies)
            & np.isfinite(target)
            & np.isfinite(design.capacity[capacity])
        )
        used = used | extra_ok
    columns = [design.regressor] if include_transfer else [np.zeros((design.n_rows, 0))]
    if extra_columns is not None:
        columns.append(extra_columns)
    transfer = np.column_stack(columns)
    solar_column = -design.capacity[capacity][used] * anomalies[used]
    k, shares = solve(
        solar_column=solar_column, transfer_columns=transfer[used], target=target[used]
    )
    return k, shares, used


def scored_rows(*, design: Design) -> np.ndarray:
    """Return the rows C1 to C4 are scored on, before the anomaly of the held-out month is known."""
    return (
        design.solar_rows
        & np.isfinite(design.capacity["const"])
        & np.isfinite(design.capacity["weekly"])
        & np.isfinite(design.nr["q50"])
        & np.isfinite(design.v1["q50"])
    )


def _pinball(*, quantile: float, observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    error = observed - predicted
    return np.maximum(quantile * error, (quantile - 1.0) * error)


def run_arms(*, design: Design) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Score every arm by leaving one calendar month out at a time.

    A month is held out when it holds scored rows. For each, the mean of u excludes the month, the
    scale `k` of each fitted arm is fitted on the other months, and the held-out month's rows are
    forecast as `b + k * (-C * a)`. A row is scored when its anomaly is defined: it needs another
    year's rows in the same calendar month and half-hour.

    Args:
        design: The design.

    Returns:
        The per-row table (`series`, `valid_time`, `month_id`, `summer`, the observed power and
        every arm's median and absolute error in MW, and the pinball losses at q10 and q90 of the
        `v1`, `control`, and `c1` arms) and the per-fold table (`series`, `month_id`, `n_fit_rows`,
        `n_events_in_fit`, and each fitted arm's `k`).
    """
    scored = scored_rows(design=design)
    months = np.unique(design.month_id[scored])
    outputs: list[pl.DataFrame] = []
    folds: list[dict[str, float | int]] = []
    fit_masks = {
        ("const", True): fit_rows_mask(design=design, capacity="const", include_transfer=True),
        ("weekly", True): fit_rows_mask(design=design, capacity="weekly", include_transfer=True),
        ("const", False): fit_rows_mask(design=design, capacity="const", include_transfer=False),
        ("previous", True): fit_rows_mask(
            design=design, capacity="previous", include_transfer=True
        ),
    }
    for month in months:
        anomalies = anomaly(design=design, exclude_month=int(month))
        held = scored & (design.month_id == month) & np.isfinite(anomalies)
        if not held.any():
            continue
        in_training = design.month_id != month
        fold: dict[str, float | int] = {"series": design.series, "month_id": int(month)}
        adjust: dict[str, np.ndarray] = {}
        for arm, (capacity, include_transfer) in {
            "c1": ("const", True),
            "c3": ("weekly", True),
            "c4": ("const", False),
            "c5": ("previous", True),
        }.items():
            rows = fit_masks[(capacity, include_transfer)] & in_training
            k, _, used = fit_arm(
                design=design,
                anomalies=anomalies,
                capacity=capacity,
                include_transfer=include_transfer,
                rows=rows,
            )
            fold[f"k_{arm}"] = k
            if arm == "c1":
                fold["n_fit_rows"] = int(used.sum())
                fold["n_events_in_fit"] = int(
                    (design.member[used].any(axis=0)).sum() if design.member.shape[1] else 0
                )
            adjust[arm] = -k * design.capacity[capacity] * anomalies
        adjust["c2"] = -design.capacity["const"] * anomalies
        folds.append(fold)
        index = np.flatnonzero(held)
        outputs.append(_month_table(design=design, index=index, adjust=adjust))
    table = pl.concat(outputs) if outputs else pl.DataFrame()
    return table, pl.DataFrame(folds)


def _month_table(
    *, design: Design, index: np.ndarray, adjust: dict[str, np.ndarray]
) -> pl.DataFrame:
    """Return the per-row scores of the rows `index` of one held-out month."""
    y = design.y[index]
    medians = {
        "v1": design.v1["q50"][index],
        "control": design.nr["q50"][index],
        **{arm: design.nr["q50"][index] + adjust[arm][index] for arm in ("c1", "c2", "c3", "c4")},
        "c5": design.nr["q50"][index] + adjust["c5"][index],
    }
    columns: dict[str, np.ndarray] = {
        "series": np.full(index.size, design.series),
        "valid_time": design.time[index],
        "month_id": design.month_id[index],
        "summer": design.summer[index],
        "observed_mw": y,
    }
    for arm, predicted in medians.items():
        columns[f"median_{arm}"] = predicted
        columns[f"abs_error_{arm}"] = np.abs(y - predicted)
    shifts = {
        "v1": np.zeros(index.size),
        "control": np.zeros(index.size),
        "c1": adjust["c1"][index],
    }
    bases = {"v1": design.v1, "control": design.nr, "c1": design.nr}
    for arm, shift in shifts.items():
        for quantile, name in ((0.1, "q10"), (0.9, "q90")):
            columns[f"pinball_{name}_{arm}"] = _pinball(
                quantile=quantile, observed=y, predicted=bases[arm][name][index] + shift
            )
    return (
        pl.DataFrame(columns)
        .with_columns(valid_time=pl.col("valid_time").dt.replace_time_zone("UTC"))
        .fill_nan(None)
    )


def row_counts(*, design: Design) -> dict[str, int | str]:
    """Return how many rows survive each filter, for the rows-per-series table."""
    scored = scored_rows(design=design)
    return {
        "series": design.series,
        "window_rows": design.n_rows,
        "eligible": int(design.eligible.sum()),
        "eligible_daylight": int((design.eligible & design.daylight).sum()),
        "blocked_daylight": int((design.eligible & design.daylight & design.blocked).sum()),
        "labelled_daylight": int((design.eligible & design.daylight & design.in_label).sum()),
        "scoreable": int(scored.sum()),
        "events_with_regressor": len(design.event_ids),
    }


def main() -> None:
    """Score every allowed series and write the per-row, per-fold, and row-count tables."""
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row_tables, fold_tables, counts = [], [], []
    for series in common.SERIES_IDS:
        design = build_design(series=series)
        if design is None:
            skipped = common.load_series(series=series)
            print(
                f"series {series}: capacity status {skipped.status!r} is not in "
                f"{common.ALLOWED_STATUSES}, skipped"
            )
            counts.append(
                {
                    "series": series,
                    "window_rows": skipped.frame.height,
                    "eligible": 0,
                    "eligible_daylight": 0,
                    "blocked_daylight": 0,
                    "labelled_daylight": 0,
                    "scoreable": 0,
                    "events_with_regressor": 0,
                    "scored": 0,
                }
            )
            continue
        table, folds = run_arms(design=design)
        counts.append(row_counts(design=design) | {"scored": table.height})
        print(f"series {series}: {table.height} scored rows in {folds.height} held-out months")
        if table.height:
            row_tables.append(table)
            fold_tables.append(folds)
    pl.concat(row_tables).write_parquet(common.RESULTS_DIR / "c_rows.parquet")
    pl.concat(fold_tables, how="diagonal").write_parquet(common.RESULTS_DIR / "c_folds.parquet")
    pl.DataFrame(counts).write_parquet(common.RESULTS_DIR / "row_counts.parquet")


if __name__ == "__main__":
    main()
