"""Run every detector over the grid, fix the false-alarm budget on tuning, and score validation.

Writes `curve.parquet`, `chosen.parquet`, `event_hits.parquet`, and `alarms_<detector>_<w>.parquet`
under the experiment's data folder, and prints the numbers the write-up quotes.
"""

from itertools import product

import numpy as np
import polars as pl
from e00037_common import WORK_DIR, load_arrays
from e00037_score import (
    BUDGET_PER_SERIES_MONTH,
    FINE_THRESHOLD_GRID,
    N_PLACEBO_SHIFTS,
    N_RESAMPLES,
    THRESHOLD_GRID,
    alarm_rate,
    bootstrap_recall,
    hits,
    near_log_masks,
    operation_members,
    placebo_shifts,
    run_grid,
    series_month_cells,
    unexplained_counts,
    weighted_recall,
)


def _rate_interval(*, unexplained: pl.DataFrame, cells: pl.DataFrame) -> tuple[float, float]:
    """95% interval of the alarm rate, resampling series-month cells."""
    joined = cells.join(unexplained, on=["series", "month_index"], how="left").fill_null(0)
    n, months = joined["n"].to_numpy().astype(float), joined["series_months"].to_numpy()
    index = np.random.default_rng(1).integers(0, len(n), size=(N_RESAMPLES, len(n)))
    rates = n[index].sum(axis=1) / months[index].sum(axis=1)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def main() -> None:  # noqa: PLR0915
    """Score the grid and write the tables."""
    panel = pl.read_parquet(WORK_DIR / "panel.parquet")
    residuals = load_arrays(panel=panel, column="res_mw")
    intervals = pl.read_parquet(WORK_DIR / "intervals.parquet")
    oracle = pl.read_parquet(WORK_DIR / "oracle_real.parquet").filter("primary")
    members = operation_members(intervals=intervals)
    near = near_log_masks(intervals=intervals, series_ids=list(residuals))
    cells = {s: series_month_cells(residuals=residuals, split=s) for s in ("tuning", "validation")}
    grid = run_grid(residuals=residuals, grid=FINE_THRESHOLD_GRID)
    registered = {(d, w, h) for d, hs in THRESHOLD_GRID.items() for w in (8, 48) for h in hs}
    confirmed = oracle.filter("confirmed", pl.col("split") == "validation")
    logged = oracle.filter(pl.col("split") == "validation")
    print(
        f"validation confirmed {confirmed.height} "
        f"(two-sided {confirmed.filter(pl.col('kind') == 'two_sided').height}),"
        f" logged >=2h {logged.height}"
    )
    curve = []
    for (detector, window, threshold), table in grid.items():
        table.write_parquet(WORK_DIR / f"alarms_{detector}_{window}_{threshold}.parquet")
        record = {
            "detector": detector,
            "window": window,
            "h": threshold,
            "pre_registered": (detector, window, threshold) in registered,
        }
        for split in ("tuning", "validation"):
            un = unexplained_counts(alarm_table=table, near=near, split=split)
            record[f"rate_{split}"] = alarm_rate(unexplained=un, cells=cells[split])
        scored = hits(events=confirmed, alarm_table=table, members=members)
        weight = scored["transfer_mw"].to_numpy()
        record["recall_weighted"] = weighted_recall(hit=scored["hit"].to_numpy(), weight=weight)
        record["recall_unweighted"] = float(scored["hit"].mean())  # ty: ignore[invalid-argument-type]
        record["n_alarms"] = table.height
        curve.append(record)
    curve_frame = pl.DataFrame(curve)
    curve_frame.write_parquet(WORK_DIR / "curve.parquet")
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    print(curve_frame)
    chosen_rows, hit_rows = [], []
    for grid_name, detector, window in product(("pre_registered", "fine"), ("D0", "D1"), (8, 48)):
        sub = curve_frame.filter(pl.col("detector") == detector, pl.col("window") == window)
        if grid_name == "pre_registered":
            sub = sub.filter("pre_registered")
        ok = sub.filter(pl.col("rate_tuning") <= BUDGET_PER_SERIES_MONTH).sort("h")
        budget_met = ok.height > 0
        h = float(ok["h"][0]) if budget_met else float(sub["h"].max())  # ty: ignore[invalid-argument-type]
        table = grid[(detector, window, h)]
        un = unexplained_counts(alarm_table=table, near=near, split="validation")
        rate_lo, rate_hi = _rate_interval(unexplained=un, cells=cells["validation"])
        scored = hits(events=confirmed, alarm_table=table, members=members)
        weight = scored["transfer_mw"].to_numpy()
        hit = scored["hit"].to_numpy().astype(float)
        interval = bootstrap_recall(hit_a=hit, hit_b=None, weight=weight)
        logged_scored = hits(events=logged, alarm_table=table, members=members)
        placebo = []
        for seed in range(N_PLACEBO_SHIFTS):
            shifted = hits(
                events=confirmed,
                alarm_table=table,
                members=members,
                shifts=placebo_shifts(n_events=confirmed.height, seed=seed),
            )
            placebo.append(weighted_recall(hit=shifted["hit"].to_numpy(), weight=weight))
        by_band = {band: float(g["hit"].mean()) for (band,), g in scored.group_by(["size_band"])}  # ty: ignore[invalid-argument-type]
        chosen_rows.append(
            {
                "grid": grid_name,
                "detector": detector,
                "window": window,
                "h": h,
                "budget_met": budget_met,
                "rate_tuning": float(sub.filter(pl.col("h") == h)["rate_tuning"][0]),
                "rate_validation": float(sub.filter(pl.col("h") == h)["rate_validation"][0]),
                "rate_lo": rate_lo,
                "rate_hi": rate_hi,
                "recall_weighted": weighted_recall(hit=hit, weight=weight),
                "recall_lo": interval["lo"],
                "recall_hi": interval["hi"],
                "recall_unweighted": float(hit.mean()),
                "recall_two_sided": float(
                    scored.filter(pl.col("kind") == "two_sided")["hit"].mean()  # ty: ignore[invalid-argument-type]
                ),
                "recall_one_sided": float(
                    scored.filter(pl.col("kind") == "one_sided")["hit"].mean()  # ty: ignore[invalid-argument-type]
                ),
                "recall_all_logged": float(logged_scored["hit"].mean()),  # ty: ignore[invalid-argument-type]
                "placebo_weighted_mean": float(np.mean(placebo)),
                "placebo_weighted_p95": float(np.percentile(placebo, 95)),
                "median_delay": scored.filter("hit")["delay"].median(),
                **{f"recall_band_{k}": v for k, v in by_band.items()},
            }
        )
        hit_rows.append(
            scored.with_columns(
                grid=pl.lit(grid_name),
                detector=pl.lit(detector),
                window=pl.lit(window),
                h=pl.lit(h),
            )
        )
    chosen = pl.DataFrame(chosen_rows, infer_schema_length=None)
    chosen.write_parquet(WORK_DIR / "chosen.parquet")
    pl.concat(hit_rows, how="diagonal").write_parquet(WORK_DIR / "event_hits.parquet")
    print(chosen)
    every = pl.concat(hit_rows, how="diagonal")
    for grid_name in ("pre_registered", "fine"):
        d0 = every.filter(
            (pl.col("grid") == grid_name) & (pl.col("detector") == "D0") & (pl.col("window") == 8)
        )
        d1 = every.filter(
            (pl.col("grid") == grid_name) & (pl.col("detector") == "D1") & (pl.col("window") == 8)
        )
        weight = d1["transfer_mw"].to_numpy()
        diff = bootstrap_recall(
            hit_a=d1["hit"].to_numpy().astype(float),
            hit_b=d0["hit"].to_numpy().astype(float),
            weight=weight,
        )
        point = weighted_recall(hit=d1["hit"].to_numpy(), weight=weight) - weighted_recall(
            hit=d0["hit"].to_numpy(), weight=weight
        )
        print(
            f"[{grid_name}] D1 - D0 weighted recall (w=8, budget): {point:+.3f} "
            f"[{diff['diff_lo']:+.3f}, {diff['diff_hi']:+.3f}]"
        )
        keep = np.arange(len(weight)) != weight.argmax()
        print(
            f"[{grid_name}] largest event weight share {weight.max() / weight.sum():.2f}; "
            "D1 recall "
            f"without it {weighted_recall(hit=d1['hit'].to_numpy()[keep], weight=weight[keep]):.3f}"
        )


if __name__ == "__main__":
    main()
