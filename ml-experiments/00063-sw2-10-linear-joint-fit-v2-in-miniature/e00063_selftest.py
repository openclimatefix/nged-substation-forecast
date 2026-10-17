"""Self-test of experiment 00063's joint fit on synthetic data with a known scale and shares.

Checks, with no file read:

1. the anomaly of u has zero mean in every (calendar month, half-hour) cell;
2. the ordinary-least-squares fit recovers a known scale `k` and known transfer shares exactly
   (to 1e-9) from noise-free synthetic power built with the same anomaly;
3. the fit does not recover a transfer share from rows outside its event, and a placebo column of
   zeros returns a zero share;
4. leave-one-month-out scoring of power that follows a known scale recovers it to within 10%
   (the held-out anomaly is computed without the held-out month, which attenuates the estimate
   slightly), scores the control arm at the exact baseline, and scores the fixed-scale arm as
   `b - C * a` exactly.

    uv run python ml-experiments/00063-*/e00063_selftest.py
"""

from __future__ import annotations

from typing import Final

import e00063_fit as fit
import numpy as np

K_TRUE: Final[float] = 0.8
SHARES_TRUE: Final[tuple[float, float]] = (0.3, -0.15)
YEARS: Final[int] = 6


def synthetic_design(*, rng: np.random.Generator) -> fit.Design:
    """Build a noise-free design: 6 years, 5 days per month, 48 half-hours per day."""
    stamps, month_id, cell = [], [], []
    for year in range(2018, 2018 + YEARS):
        for month in range(1, 13):
            for day in range(1, 6):
                base = np.datetime64(f"{year}-{month:02d}-{day:02d}T00:00", "us")
                for half_hour in range(48):
                    stamps.append(base + np.timedelta64(30 * half_hour, "m"))
                    month_id.append(year * 12 + month - 1)
                    cell.append((month - 1) * 48 + half_hour)
    n = len(stamps)
    cell_array = np.array(cell)
    half_hour = cell_array % 48
    daylight = (half_hour >= 14) & (half_hour < 34)
    shape = np.where(daylight, 0.3 + 0.6 * np.sin(np.pi * (half_hour - 14) / 20.0), 0.0)
    u = np.where(daylight, shape + rng.normal(0.0, 0.1, n), 0.0)
    capacity = 2.0 + rng.uniform(0.0, 1.0, n)
    baseline = 5.0 + rng.normal(0.0, 1.0, n)
    time = np.array(stamps)
    # Two labelled events, each a stretch of rows with a donor-power regressor.
    member = np.zeros((n, 2), dtype=bool)
    member[1000:1200, 0] = True
    member[7000:7300, 1] = True
    regressor = np.where(member, rng.uniform(2.0, 6.0, (n, 2)), 0.0)
    in_label = member.any(axis=1)
    blocked = np.zeros(n, dtype=bool)
    blocked[3000:3100] = True
    return fit.Design(
        series=0,
        norm_mw=10.0,
        time=time,
        month_id=np.array(month_id),
        cell=cell_array,
        summer=np.zeros(n, dtype=bool),
        y=baseline.copy(),
        v1=dict.fromkeys(("q10", "q50", "q90"), baseline + 0.1),
        nr={q: baseline.copy() for q in ("q10", "q50", "q90")},
        u=u,
        capacity={"const": capacity, "weekly": capacity.copy(), "previous": capacity.copy()},
        daylight=daylight,
        eligible=np.ones(n, dtype=bool),
        eligible_event=np.ones(n, dtype=bool),
        blocked=blocked,
        in_label=in_label,
        member=member,
        regressor=regressor,
        event_ids=["event_0", "event_1"],
    )


def main() -> None:
    """Run every check; an assertion error is a failure."""
    rng = np.random.default_rng(0)
    design = synthetic_design(rng=rng)

    anomalies = fit.anomaly(design=design, exclude_month=None)
    for cell in range(fit.N_CELLS):
        in_cell = design.cell == cell
        assert abs(anomalies[in_cell].mean()) < 1e-12, f"cell {cell} anomaly mean"
    print("PASS: the anomaly of u has zero mean in all 576 (month, half-hour) cells")

    true_shift = -design.capacity["const"] * anomalies
    design.y = design.nr["q50"] + K_TRUE * true_shift + design.regressor @ np.array(SHARES_TRUE)
    rows = fit.fit_rows_mask(design=design, capacity="const", include_transfer=True)
    k, shares, used = fit.fit_arm(
        design=design,
        anomalies=anomalies,
        capacity="const",
        include_transfer=True,
        rows=rows,
    )
    assert abs(k - K_TRUE) < 1e-9, f"k {k}"
    assert np.allclose(shares, SHARES_TRUE, atol=1e-9), f"shares {shares}"
    assert not used[3000:3100].any(), "blocked rows without a label must not be fitted"
    print(f"PASS: OLS recovers k = {k:.12f} and shares {np.round(shares, 12)} exactly")

    # A placebo interval (zero share) returns zero; a share cannot leak into a wrong column.
    placebo_x = np.zeros((design.n_rows, 1))
    placebo_x[12000:12200, 0] = rng.uniform(2.0, 6.0, 200)
    k_p, shares_p, _ = fit.fit_arm(
        design=design,
        anomalies=anomalies,
        capacity="const",
        include_transfer=True,
        rows=rows,
        extra_columns=placebo_x,
        extra_rows=placebo_x[:, 0] != 0.0,
    )
    assert abs(shares_p[-1]) < 1e-9, shares_p
    assert abs(k_p - K_TRUE) < 1e-9, k_p
    injected = design.y + 0.25 * placebo_x[:, 0]
    _, shares_i, _ = fit.fit_arm(
        design=design,
        anomalies=anomalies,
        capacity="const",
        include_transfer=True,
        rows=rows,
        target=injected,
        extra_columns=placebo_x,
        extra_rows=placebo_x[:, 0] != 0.0,
    )
    assert abs(shares_i[-1] - 0.25) < 1e-9, shares_i
    print(
        "PASS: a zero-share interval returns 0 and an injected share of 0.25 is recovered exactly"
    )

    # Leave-one-month-out: power that follows the scale K_TRUE on the held-out anomaly's cell mean.
    # The mean of u excludes the held-out month, so the estimate is slightly attenuated.
    design.y = design.nr["q50"] + K_TRUE * true_shift
    table, folds = fit.run_arms(design=design)
    k_hat = float(np.median(folds["k_c1"].to_numpy()))
    assert abs(k_hat / K_TRUE - 1.0) < 0.10, f"median fold k {k_hat}"
    assert np.allclose(table["median_control"].to_numpy(), design.nr["q50"][_index(design, table)])
    fixed = table["median_c2"].to_numpy()
    anomaly_by_row = _fold_anomaly(design=design, table=table)
    expected = (
        design.nr["q50"][_index(design, table)]
        - design.capacity["const"][_index(design, table)] * anomaly_by_row
    )
    assert np.allclose(fixed, expected, atol=1e-9), "the fixed-scale arm is b - C * a"
    c1_error = float(np.mean(table["abs_error_c1"].to_numpy()))
    control_error = float(np.mean(table["abs_error_control"].to_numpy()))
    assert c1_error < control_error, (c1_error, control_error)
    print(
        f"PASS: leave-one-month-out median k {k_hat:.3f} vs true {K_TRUE}; "
        f"c1 error {c1_error:.4f} < "
        f"control error {control_error:.4f}; the control and fixed-scale arms are exact"
    )


def _index(design: fit.Design, table) -> np.ndarray:  # noqa: ANN001
    """Return the design row of each table row."""
    return np.searchsorted(design.time, table["valid_time"].dt.replace_time_zone(None).to_numpy())


def _fold_anomaly(*, design: fit.Design, table) -> np.ndarray:  # noqa: ANN001
    """Return each table row's anomaly with its own month left out of the mean."""
    out = np.empty(table.height)
    index = _index(design, table)
    months = table["month_id"].to_numpy()
    for month in np.unique(months):
        anomalies = fit.anomaly(design=design, exclude_month=int(month))
        sel = months == month
        out[sel] = anomalies[index[sel]]
    return out


if __name__ == "__main__":
    main()
