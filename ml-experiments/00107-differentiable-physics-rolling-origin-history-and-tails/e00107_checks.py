"""The blocking known-answer checks of experiment 107, run before any scored window is read.

1. **Part 2 reproduces.** At the 2025-07-01 cut, the three fitted-capacity arms (`g1_fc`,
   `g_all_fc`, `g_all_d1_fc`) with Part 2's replica spread match experiment 104's stored
   `*_centred` predictions for July to September 2025.
2. **No lookahead, per cut.** Every stored row's valid time lies in the cut's 3-month window, every
   D1 knot is on or before its cut, and no spread-pool row has a valid time after the origin it
   serves.
3. **The pool rule.** Changing any actual or any predicted mean after origin `c` leaves the fitted
   S0, S1 (its tercile cut points included), and S2 unchanged, and changing a pool row does change
   them, so the test can fail.
4. **Pinned capacity source** per cut and site, beside the `-fc` fit's capacity (printed).

    uv run python ml-experiments/00107-*/e00107_checks.py
"""

import json
import sys
from datetime import UTC, datetime

import e00107_assemble as assemble
import e00107_common as common
import numpy as np
import polars as pl
from e00107_spread import ConformalSpread, LevelTercileSpread
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    QUANTILE_COLUMNS,
    DriftType,
    GamSpec,
    ResidualSpread,
    TrainingWindow,
    make_layout,
    replica_residuals,
)

TOLERANCE: float = 1e-6
FAILURES: list[str] = []


def report(*, name: str, passed: bool, detail: str) -> None:
    """Print one check's outcome and remember a failure."""
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)
    if not passed:
        FAILURES.append(name)


def check_part_2_reproduces(*, cache: FeatureCache) -> None:
    """Compare the `-fc` arms' July to September 2025 predictions with experiment 104's."""
    cut = datetime(2025, 7, 1, tzinfo=UTC)
    tuned = json.loads(common.TUNED_PATH.read_text())
    fits = assemble.read_fits(kind="gam", variant="base").filter(pl.col("origin") == "2025-07-01")
    arms = {
        "g1_fc": ("g1_centred", 12),
        "g_all_fc": ("g_all_centred", None),
        "g_all_d1_fc": ("g_all_d1_centred", None),
    }
    drift: dict[str, DriftType] = {"g1_fc": "none", "g_all_fc": "none", "g_all_d1_fc": "d1"}
    positions = assemble.grid_positions(cache=cache, frame=fits)
    for arm, (stored, months) in arms.items():
        old = pl.read_parquet(common.PART_2_PREDICTIONS_DIR / stored / "predictions.parquet")
        worst_mean = 0.0
        worst_quantile = 0.0
        n_rows = 0
        for site in common.ALL_SITES:
            if site == "D9":
                continue
            rows = fits.filter(pl.col("arm") == arm, pl.col("site") == site)
            if rows.is_empty():
                continue
            spec = GamSpec(
                drift=drift[arm],
                ridge=tuned["ridge"]["chosen"],
                drift_step=tuned["drift_step"]["chosen"],
            )
            replica = replica_residuals(
                cache=cache,
                site=site,
                spec=spec,
                cut=cut,
                window=TrainingWindow(months=months),
                season_years=common.SEASON_YEARS,
            )
            spread = ResidualSpread.fit(
                residuals=replica.residuals, band=replica.band, quarter=replica.quarter, centre=True
            )
            row_positions = positions[
                (fits["arm"] == arm).to_numpy() & (fits["site"] == site).to_numpy()
            ]
            quantiles = spread.quantiles(
                mean=rows["mean"].to_numpy(),
                band=cache.calendar.band[row_positions],
                quarter=cache.calendar.quarter[row_positions],
            )
            mine = rows.select("time_series_id", "valid_time").with_columns(
                mean_new=rows["mean"].cast(pl.Float32),
                **{
                    f"{column}_new": pl.Series(np.sort(quantiles, axis=1)[:, position]).cast(
                        pl.Float32
                    )
                    for position, column in enumerate(QUANTILE_COLUMNS)
                },
            )
            joined = mine.join(old, on=["time_series_id", "valid_time"], how="inner")
            n_rows += joined.height
            worst_mean = max(
                worst_mean,
                float(joined.select((pl.col("mean_new") - pl.col("mean")).abs().max()).item()),
            )
            for column in QUANTILE_COLUMNS:
                worst_quantile = max(
                    worst_quantile,
                    float(
                        joined.select((pl.col(f"{column}_new") - pl.col(column)).abs().max()).item()
                    ),
                )
        report(
            name=f"1. Part 2 reproduces ({arm})",
            passed=n_rows > 0 and worst_mean <= TOLERANCE and worst_quantile <= TOLERANCE,
            detail=(
                f"{n_rows} rows; max |mean diff| {worst_mean:.2e}, "
                f"max |quantile diff| {worst_quantile:.2e}"
            ),
        )


def check_no_lookahead(*, cache: FeatureCache) -> None:
    """Check each stored row's window, each D1 knot, and each pool's valid times."""
    bad_rows = 0
    for kind in ("gam", "xgb"):
        for variant in ("base", "d10_masked"):
            folder = common.WORK_DIR / kind / variant
            for file in sorted(folder.glob("*.parquet")):
                frame = pl.read_parquet(file)
                cut = datetime.fromisoformat(frame["origin"][0]).replace(tzinfo=UTC)
                end = common.window_end(cut=cut)
                outside = frame.filter(
                    (pl.col("valid_time") <= cut)
                    | (pl.col("valid_time") > end)
                    | (pl.col("origin") != common.cut_label(cut=cut))
                )
                bad_rows += outside.height
    report(
        name="2a. every stored row is in (cut, cut + 3 months]",
        passed=bad_rows == 0,
        detail=f"{bad_rows} rows outside",
    )
    wrong_knots = 0
    for cut in common.CUTS:
        for site in common.sites_for_cut(cut=cut):
            features = cache.sites[site]
            layout = make_layout(
                spec=GamSpec(drift="d1"),
                season_years=common.SEASON_YEARS,
                cut=cut,
                earliest=cache.end_times[np.flatnonzero(features.default_rows)[0]],
            )
            cut_day = common.to_64(moment=cut).astype("datetime64[s]").astype(np.int64) / 86400.0
            wrong_knots += int(layout.knot_days.max() != cut_day)
    report(
        name="2b. no D1 knot after its cut",
        passed=wrong_knots == 0,
        detail=f"{wrong_knots} of {len(common.CUTS)} layouts have a last knot other than the cut",
    )
    fits = assemble.read_fits(kind="gam", variant="base")
    worst = 0
    for cut in common.SCORED_ORIGINS:
        for site in common.sites_for_cut(cut=cut):
            rows = fits.filter(pl.col("arm") == "g1", pl.col("site") == site)
            if rows.is_empty() or (
                site == "D9" and cut < common.add_months(moment=common.D9_FIRST_CUT, months=12)
            ):
                continue
            pool = assemble.pool_rows(arm_rows=rows, cut=cut, drop_jump_window=False)
            worst = max(worst, int((pool["valid_time"] > cut).sum()))
    report(
        name="2c. no spread-pool row after the origin it serves",
        passed=worst == 0,
        detail=f"{worst} offending rows (pool_rows also raises on one)",
    )


def check_pool_rule(*, cache: FeatureCache) -> None:
    """Check that data after an origin leaves the fitted spreads unchanged.

    Perturbing the newest pool window must move them, which shows the test can fail.
    """
    fits = assemble.read_fits(kind="gam", variant="base").filter(
        pl.col("arm") == "g_all_d1", pl.col("site") == "D14"
    )
    positions = assemble.grid_positions(cache=cache, frame=fits)
    fits = fits.with_columns(
        band=pl.Series(cache.calendar.band[positions].astype(np.int64)),
        quarter=pl.Series(cache.calendar.quarter[positions].astype(np.int64)),
        residual=pl.col("actual") - pl.col("mean"),
    )
    generator = np.random.default_rng(0)

    def fitted(*, frame: pl.DataFrame, cut: datetime) -> dict[str, np.ndarray]:
        pool = assemble.pool_rows(arm_rows=frame, cut=cut, drop_jump_window=False)
        residual = (pool["actual"] - pool["mean"]).to_numpy()
        band, quarter = pool["band"].to_numpy(), pool["quarter"].to_numpy()
        newest = (
            pool["origin"] == common.cut_label(cut=assemble.previous_cuts(cut=cut)[-1])
        ).to_numpy()
        s0 = ResidualSpread.fit(residuals=residual, band=band, quarter=quarter, centre=True)
        s1 = LevelTercileSpread.fit(
            residuals=residual, predicted=pool["mean"].to_numpy(), band=band, quarter=quarter
        )
        s2 = ConformalSpread.fit(residuals=residual, band=band, quarter=quarter, newest=newest)
        return {
            "s0": s0.table,
            "s1_table": s1.table,
            "s1_cut_points": s1.cut_points,
            "s2_shift": s2.shift,
            "s2_base": s2.base.table,
        }

    unchanged = 0
    total = 0
    sensitive = True
    for cut in common.SCORED_ORIGINS:
        before = fitted(frame=fits, cut=cut)
        late = pl.col("valid_time") > cut
        noise = generator.normal(size=fits.height)
        perturbed = fits.with_columns(
            actual=pl.when(late)
            .then(pl.col("actual") + 5.0 * pl.Series(noise))
            .otherwise(pl.col("actual")),
            mean=pl.when(late)
            .then(pl.col("mean") + 5.0 * pl.Series(noise[::-1].copy()))
            .otherwise(pl.col("mean")),
        )
        after = fitted(frame=perturbed, cut=cut)
        for key, value in before.items():
            total += 1
            unchanged += int(np.array_equal(value, after[key]))
        pool_index = (fits["valid_time"] <= cut) & (
            fits["valid_time"] > common.add_months(moment=cut, months=-3)
        )
        touched = fits.with_columns(
            actual=pl.when(pool_index)
            .then(pl.col("actual") + 5.0 * pl.Series(noise))
            .otherwise(pl.col("actual")),
        )
        changed = fitted(frame=touched, cut=cut)
        sensitive &= not np.array_equal(before["s0"], changed["s0"])
    report(
        name="3. pool rule: data after the origin leaves S0, S1, S2 unchanged",
        passed=unchanged == total,
        detail=f"{unchanged} of {total} fitted tables identical (12 origins, 5 tables each)",
    )
    report(
        name="3b. the pool-rule test can fail: perturbing the newest pool window moves S0",
        passed=sensitive,
        detail="S0 changed at every origin" if sensitive else "S0 did not change at some origin",
    )


def print_capacity_sources() -> None:
    """Print the pinned capacity's source per cut and site, beside the fitted capacity."""
    sources = assemble.capacity_sources()
    print("\n4. Pinned capacity source per cut and site (held level beside the -fc fit's)")
    with pl.Config(tbl_rows=200, tbl_cols=10, fmt_str_lengths=30):
        print(sources)
    counts = sources.group_by("pinned_source").len().sort("pinned_source")
    print(counts)
    report(
        name="4. capacity sources printed",
        passed=sources.height > 0,
        detail=f"{sources.height} cut-site rows",
    )


def main() -> None:
    """Run the four checks and exit nonzero if any failed."""
    cache = common.load_cache()
    check_part_2_reproduces(cache=cache)
    check_no_lookahead(cache=cache)
    check_pool_rule(cache=cache)
    print_capacity_sources()
    if FAILURES:
        print(f"\nFAILED: {FAILURES}")
        sys.exit(1)
    print("\nAll blocking checks passed.")


if __name__ == "__main__":
    main()
