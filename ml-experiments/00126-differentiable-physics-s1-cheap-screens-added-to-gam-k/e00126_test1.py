"""Known-answer test 1 of experiment 126: every arm reproduces T-all where it must.

Each check is a limit in which an arm has to equal T-all (or B0), so a failure means the arm's
columns or penalties are wrong, not that the data disagree.

1. B0 built from this folder's code equals `studies.dp_gam.fit_gam` to 1e-8 relative in the
   parameters, at every decision site and the replicate and scored cuts.
2. T-all's scored-year mean forecast reproduces experiment 114's stored `scored_t_all` mean to
   1e-4 points.
3. Each arm that appends columns (A-mono, A-free, S-ewma, S-mm, D-sch, E-sol, W-stn), with the
   appended columns set to zero, has T-all's fitted values to 1e-5 points. C-wind and C-hw with
   the wind and humidity terms removed have T-all's design matrix exactly. W-mix's two blocks sum
   to T-all's heating block, so equal coefficients reproduce T-all.
4. T-all's heating columns carry a non-negativity bound whose lower end is zero, so the heating
   slope can fall to zero (A-mono's injected flattening needs this).
5. No training row ends after its cut, and W-basis reads only rows before its cut.
6. The pieces of a split column partition the rows.

    uv run --no-sync python ml-experiments/00126-*/e00126_test1.py
"""

import sys
from typing import Any, Final

import e00126_common as common
import e00126_design as design
import e00126_pipeline as pipeline
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import assert_no_lookahead, fit_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

PARAMETER_TOLERANCE: Final[float] = 1e-8
FITTED_VALUE_TOLERANCE_POINTS: Final[float] = 1e-5
STORED_TOLERANCE_POINTS: Final[float] = 1e-4
TEST_CUTS: Final[tuple[str, ...]] = ("replicate", "scored")
APPENDING_ARMS: Final[tuple[str, ...]] = (
    "a_mono",
    "a_free",
    "s_ewma",
    "s_mm",
    "d_sch",
    "e_sol",
    "w_stn",
)
PENALTY_WEIGHT: Final[float] = 1.0
STORED_114: Final = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00114-differentiable-physics-n3-n4-interactions-and-heating-threshold"
    / "predictions"
    / "scored_t_all"
    / "predictions.parquet"
)


def _context(
    *, cache: FeatureCache, vintages: pl.DataFrame, site: str, cut_label: str, arm: str
) -> design.ArmContext:
    """Return an arm's extra arrays at a cut."""
    return pipeline.arm_context(
        cache=cache, vintages=vintages, site=site, cut=common.FIT_CUTS[cut_label], arm_name=arm
    )


def check_b0(*, cache: FeatureCache, vintages: pl.DataFrame, site: str, label: str) -> list[str]:
    """Run checks 1 and 5 at one site and cut and return the failures."""
    failures: list[str] = []
    cut = common.FIT_CUTS[label]
    base = design.build_base(cache=cache, vintages=vintages, site=site, cut=cut)
    assert_no_lookahead(cache=cache, index=base.index, cut=cut)
    if cache.end_times[base.index].max() > common.to_64(moment=cut):
        failures.append(f"{site} {label}: a training row ends after the cut")
    context = design.ArmContext(aux=design.site_aux(cache=cache, site=site))
    fit = design.fit_arm(cache=cache, base=base, arm=design.B0, context=context)
    reference = fit_gam(
        cache=cache,
        site=site,
        specs=[common.base_spec()],
        cut=cut,
        window=common.base_window(),
        season_years=common.SEASON_YEARS,
        pinned=pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut),
    )
    relative = float(
        np.abs(reference.theta[0] - fit.theta).max() / np.abs(reference.theta[0]).max()
    )
    print(f"{site} {label}: B0 against fit_gam, max relative difference {relative:.2e}")
    if relative > PARAMETER_TOLERANCE:
        failures.append(f"{site} {label}: B0 differs from fit_gam by {relative:.2e}")
    for factors in (("daytype",), ("subband",), ("daytype", "subband")):
        masks = design.piece_masks(cache=cache, index=base.index, factors=factors)
        if not np.array_equal(masks.sum(axis=1), np.ones(base.n)):
            failures.append(f"{site} {label}: pieces of {factors} do not partition the rows")
    return failures


def check_t_all(*, cache: FeatureCache, vintages: pl.DataFrame, site: str) -> list[str]:
    """Run checks 2 and 4 at one site and return the failures."""
    run = pipeline.fit_year(
        cache=cache,
        vintages=vintages,
        site=site,
        fit_label="scored",
        arm_name="t_all",
        penalty_weight=PENALTY_WEIGHT,
    )
    assert run is not None
    stored = pl.read_parquet(STORED_114).filter(
        pl.col("time_series_id") == cache.sites[site].table.time_series_id
    )
    mine = pl.DataFrame(
        {"valid_time": cache.end_times[run.index], "mean_mine": run.mean}
    ).with_columns(pl.col("valid_time").dt.replace_time_zone("UTC"))
    joined = stored.join(mine, on="valid_time", how="inner")
    gap = float((joined["mean"].cast(pl.Float64) - joined["mean_mine"]).abs().max())
    print(f"{site}: T-all against 114's stored mean, {joined.height} rows, max gap {gap:.2e}")
    failures = []
    if joined.height != run.index.size or gap > STORED_TOLERANCE_POINTS:
        failures.append(f"{site}: T-all differs from 114's stored forecast by {gap:.2e}")
    base = design.build_base(cache=cache, vintages=vintages, site=site, cut=run.cut)
    arm = pipeline.arm_spec(arm_name="t_all", penalty_weight=PENALTY_WEIGHT)
    penalty = design.arm_penalty(layout=base.layout, arm=arm, n_rows=base.n)
    if not penalty.constrained.any() or run.fit.theta[penalty.constrained].min() < 0.0:
        failures.append(f"{site}: T-all's bounded columns are not non-negative")
    return failures


def check_arms(*, cache: FeatureCache, vintages: pl.DataFrame, site: str) -> list[str]:
    """Run check 3 at one site and return the failures."""
    failures: list[str] = []
    label = "scored"
    cut = common.FIT_CUTS[label]
    base = design.build_base(cache=cache, vintages=vintages, site=site, cut=cut)
    t_all = pipeline.arm_spec(arm_name="t_all", penalty_weight=PENALTY_WEIGHT)
    context = _context(cache=cache, vintages=vintages, site=site, cut_label=label, arm="t_all")
    reference = design.fit_arm(cache=cache, base=base, arm=t_all, context=context)
    index = common.scoring_index(
        cache=cache, site=site, after=cut, through=common.year_end(cut=cut)
    )
    expected = design.predict_arm(cache=cache, base=base, fit=reference, index=index)
    original = design.extra_block
    for name in APPENDING_ARMS:
        arm = pipeline.arm_spec(arm_name=name, penalty_weight=PENALTY_WEIGHT)
        arm_context = _context(cache=cache, vintages=vintages, site=site, cut_label=label, arm=name)

        def zeros(  # noqa: ANN202
            *,
            cache: FeatureCache,
            site: str,
            index: np.ndarray,
            arm: Any,
            context: Any,
            _o: Any = original,
        ):
            block = _o(cache=cache, site=site, index=index, arm=arm, context=context)
            return None if block is None else np.zeros_like(block)

        design.extra_block = zeros
        try:
            fit = design.fit_arm(cache=cache, base=base, arm=arm, context=arm_context)
            values = design.predict_arm(cache=cache, base=base, fit=fit, index=index)
        finally:
            design.extra_block = original
        gap = float(np.abs(values - expected).max())
        print(f"{site}: {name} with zeroed columns against T-all, max gap {gap:.2e}")
        if gap > FITTED_VALUE_TOLERANCE_POINTS:
            failures.append(f"{site}: {name} with zeroed columns differs from T-all by {gap:.2e}")
    x0 = base.x0
    heating = base.layout.slices["heating"]
    features = cache.sites[site]
    band_hot = design._band_hot(cache=cache, index=base.index)
    own = design.heating_hinges(
        smoothed=features.smoothed_temperature_c[base.index], band_hot=band_hot
    )
    gap = float(np.abs(own - x0[:, heating]).max())
    print(f"{site}: C arms' hinge on plain temperature against B0's heating columns, gap {gap:.2e}")
    if gap > FITTED_VALUE_TOLERANCE_POINTS:
        failures.append(f"{site}: the C arms' hinge differs from B0's heating block by {gap:.2e}")
    weight = design.cold_weight(temperature_c=features.smoothed_temperature_c[base.index, 1])
    mix = x0[:, heating] * weight[:, None] + x0[:, heating] * (1.0 - weight)[:, None]
    gap = float(np.abs(mix - x0[:, heating]).max())
    if gap > FITTED_VALUE_TOLERANCE_POINTS:
        failures.append(f"{site}: W-mix's blocks do not sum to the heating block ({gap:.2e})")
    return failures


def main() -> None:
    """Run every check and exit non-zero if any fails."""
    common.configure_threads()
    cache = common.load_cache()
    vintages = read_vintages()
    failures: list[str] = []
    for site in common.DECISION_SITES:
        for label in TEST_CUTS:
            failures += check_b0(cache=cache, vintages=vintages, site=site, label=label)
        failures += check_t_all(cache=cache, vintages=vintages, site=site)
        failures += check_arms(cache=cache, vintages=vintages, site=site)
    if failures:
        print("\nTEST 1 FAILED")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)
    print("\nTEST 1 PASSED")


if __name__ == "__main__":
    main()
