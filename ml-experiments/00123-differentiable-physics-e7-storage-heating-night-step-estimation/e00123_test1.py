"""Tests 0, 1, and 3 of experiment 123: feasibility, reproduction, and the rank check.

Test 1 prints the timestamp convention, the window rows, and 20 nights recomputed by hand, checks
that no row is on or after 2026-07-01, and checks that the stored B0 residuals of experiment 114
reproduce its tuning-year quantile score at four decimal places. It reads no estimate.
"""

import json
from datetime import datetime, timedelta
from typing import cast

import e00123_common as common
import e00123_estimate as estimate
import numpy as np
import polars as pl

QUANTILE_COLUMNS: tuple[str, ...] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
QUANTILE_LEVELS: tuple[float, ...] = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.40,
    0.50,
    0.60,
    0.75,
    0.90,
    0.95,
    0.99,
)
STORED_B0_TUNING_SCORES: dict[str, float] = {
    "D5": 2.5656, "D10": 7.0380, "D11": 3.3068, "D14": 2.1622, "D16": 2.3242,
}  # fmt: skip
"""Experiment 114's printed tuning-year B0 quantile score per site (points)."""
HAND_CHECK_NIGHTS: int = 20


def print_convention() -> None:
    """Print how a half-hour's stamp maps to its start, and the nine slots of one night."""
    print("Convention: `time` is the END of the half-hour; a half-hour starting at s is stamped")
    print(
        "s + 30 min, as in experiments 114 and 121 (their code subtracts 30 min to get the start)."
    )
    print(
        "Window rows (start): 00:30, 01:00, 01:30, 02:00, 02:30; baseline (start): 22:00 to 23:30."
    )
    slots = common.slot_table(clock="utc", evenings=[__import__("datetime").date(2022, 1, 12)])
    print(slots.sort("role", "slot"))


def hand_check(*, table: pl.DataFrame, power: pl.DataFrame, site: str) -> list[str]:
    """Recompute `d` on 20 tuning-winter GMT nights straight from the raw readings."""
    lookup = dict(zip(power["time"].to_list(), power["power"].to_list(), strict=True))
    nights = (
        table.filter(
            (pl.col("site") == site)
            & (pl.col("season") == "gmt")
            & (pl.col("winter") == common.TUNING_WINTER)
            & pl.col("d").is_not_null()
        )
        .sort("evening")
        .gather_every(7)
        .head(HAND_CHECK_NIGHTS)
    )
    lines = []
    for row in nights.iter_rows(named=True):
        evening = row["evening"]
        base = common.datetime.combine(evening, common.time.min).replace(tzinfo=common.UTC)
        baseline = [lookup[base + timedelta(hours=22, minutes=30 * k + 30)] for k in range(4)]
        window = [lookup[base + timedelta(hours=24, minutes=30 * k + 60)] for k in range(5)]
        by_hand = float(np.mean(window) - np.mean(baseline))
        gap = abs(by_hand - row["d"])
        lines.append(
            f"{site} {evening}: table d {row['d']:.6f}, by hand {by_hand:.6f}, gap {gap:.1e}"
        )
        if gap > 1e-4:
            msg = f"night {evening} of {site} differs from the hand computation"
            raise AssertionError(msg)
    return lines


def check_b0_reproduction(*, infos: dict[str, common.SeriesInfo]) -> list[str]:
    """Check the stored B0 tuning predictions reproduce experiment 114's printed scores."""
    lines = []
    for site, stored in STORED_B0_TUNING_SCORES.items():
        info = infos[site]
        frame = pl.read_parquet(
            common.EXPERIMENT_114_DIR / "tuning_b0" / "predictions.parquet"
        ).filter((pl.col("time_series_id") == info.time_series_id) & ~pl.col("switching_mask"))
        actual = frame["actual"].to_numpy().astype(np.float64)
        quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
        error = actual[:, None] - quantiles
        levels = np.array(QUANTILE_LEVELS)[None, :]
        score = float(
            np.mean(2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error), axis=1))
        )
        lines.append(
            f"{site}: recomputed {score:.4f}, stored {stored:.4f}, gap {abs(score - stored):.1e}"
        )
        if abs(score - stored) > 1e-4:
            msg = f"B0 tuning quantile score of {site} does not reproduce experiment 114's"
            raise AssertionError(msg)
    return lines


def main() -> None:
    """Run tests 0, 1, and 3 and write `test1.json`."""
    infos = common.series_infos()
    table = pl.read_parquet(common.WORK_DIR / "night_tables.parquet")
    print_convention()
    ids = [infos["D10"].time_series_id]
    power = common.load_power(time_series_ids=ids)
    if cast("datetime", power["time"].max()) >= common.HOLDOUT_START:
        msg = "a reading on or after 2026-07-01 was read"
        raise AssertionError(msg)
    print("Readings read: latest", power["time"].max(), "(before 2026-07-01).")
    print("\n".join(hand_check(table=table, power=power, site="D10")))
    print("\nB0 reproduction (experiment 114 stored predictions, tuning year):")
    print("\n".join(check_b0_reproduction(infos=infos)))
    print("\nTest 0 and test 3, on the design winters 2023-24 and 2024-25 only:")
    report: dict[str, dict[str, float | str]] = {}
    for site in common.FITTED_SERIES:
        nights = estimate.load_nights(
            table=table,
            site=site,
            capacity_mw=infos[site].effective_capacity_mw,
            winters=common.DESIGN_WINTERS,
            summer_years=common.SUMMER_YEARS,
        )
        shares = (
            table.filter(
                (pl.col("site") == site)
                & (pl.col("season") == "gmt")
                & ~pl.col("excluded")
                & pl.col("winter").is_in(list(common.DESIGN_WINTERS))
            )
            .group_by("winter")
            .agg(share=pl.col("d").is_not_null().mean())
            .sort("winter")["share"]
            .to_list()
        )
        lowest = min(float(share) for share in shares) if shares else 0.0
        check = estimate.rank_check(nights=nights)
        ok = (
            len(nights.winters) >= 2
            and lowest >= common.MIN_PRESENT_SHARE
            and check["rank"] == check["columns"]
            and check["vif_cold"] <= 10.0
        )
        print(
            f"{site}: winters {nights.winters}, lowest per-winter present share {lowest:.3f}, "
            f"rank {check['rank']:.0f}/{check['columns']:.0f}, smallest singular value "
            f"{check['smallest_singular_value']:.3f}, VIF of c {check['vif_cold']:.2f}, "
            f"{'kept' if ok else 'DROPPED'}"
        )
        report[site] = {**check, "present_share": lowest, "kept": str(ok)}
    (common.WORK_DIR / "test1.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
