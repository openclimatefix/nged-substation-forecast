r"""Experiment 365: a rule on the meter readings alone that chooses between two stored fits.

For each cell of experiment 358 (site, cut, injection level, reactive setting), compute the 1st
percentile of the training readings on sunny midday half-hours. If it is below `THRESHOLD_POINTS`,
the rule chooses the fit through the meter (L0), otherwise the naive fit (N). The rule's score is
the stored score of the chosen arm. A census applies the same statistic to the 8 MVA-only Primaries.

    PYTHONPATH=ml-experiments/00110-*:ml-experiments/00358-* \
        uv run --no-sync python ml-experiments/00365-*/e00365_rule.py
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import e00110_arms as arms  # ty: ignore[unresolved-import]
import e00110_common as common  # ty: ignore[unresolved-import]
import e00110_inject as inject  # ty: ignore[unresolved-import]
import e00358_known_answer as known  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import fit_gam, pinned_solar_points, predict_gam
from studies.dp_operator import tan_phi_from_power_factor
from studies.dp_pinned_capacity import pinned_capacity, read_vintages
from studies.sources import ML_EXPERIMENTS_DATA_DIR

THRESHOLD_POINTS: Final[float] = 5.0
PERCENTILE: Final[float] = 1.0
SLUG: Final[str] = "mva-fold-readings-only-reverse-flow-rule"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00365-{SLUG}" / "work"
KNOWN_DIR: Final[Path] = known.OUT_DIR


def statistic(
    *, cache: FeatureCache, site: str, reading: np.ndarray, cut: Any
) -> tuple[float, int]:
    """Return the 1st percentile of the training sunny-midday readings and their count."""
    features = cache.sites[site]
    ends = cache.end_times
    rows = np.flatnonzero(
        (ends <= common.to_64(moment=cut)) & np.isfinite(reading) & ~features.switching_mask
    )
    sunny = known._sunny(cache=cache, site=site, index=rows)
    values = reading[rows][sunny]
    if len(values) == 0:
        return float("nan"), 0
    return float(np.percentile(values, PERCENTILE)), len(values)


def cell(job: tuple[str, str, float, str]) -> dict[str, Any]:
    """Return the rule's statistic for one cell of experiment 358."""
    site, cut_name, level, reactive = job
    cut = known.CUTS[cut_name]
    cache = common.load_cache()
    part_4 = arms.read_part_4_tuned()
    spec_b = arms.base_specs(part_4=part_4)["b"]
    window = arms.arm_definitions(part_4=part_4, tuned=arms.read_tuned())["lq"].window
    features = cache.sites[site]
    pinned = pinned_capacity(vintages=read_vintages(), table=features.table, cut=cut)
    scale = inject.midday_median_absolute(cache=cache, site=site, before=cut)
    signed_cache = (
        cache
        if level == 0.0
        else inject.inject_farm(
            cache=cache, site=site, farm=known.FARM, capacity_points=level * scale
        )
    )
    signed = signed_cache.sites[site].y
    if reactive == "Q0":
        reading = np.abs(signed)
    else:
        fit_b = fit_gam(
            cache=cache,
            site=site,
            specs=[replace(spec_b)],
            cut=cut,
            window=window,
            season_years=common.SEASON_YEARS,
            pinned=pinned,
        )
        rows = np.flatnonzero(np.isfinite(features.y) & np.isfinite(features.pv_unit_shape))
        demand = np.full(len(cache.end_times), np.nan)
        demand[rows] = predict_gam(cache=cache, fit=fit_b, index=rows) + pinned_solar_points(
            cache=cache, features=features, pinned=pinned, index=rows
        )
        tan_phi = tan_phi_from_power_factor(power_factor=known.POWER_FACTOR)
        reading = (
            inject.inject_meter(
                cache=signed_cache, site=site, signed=signed, demand=demand, tan_phi=tan_phi
            )
            .sites[site]
            .y
        )
    value, count = statistic(cache=cache, site=site, reading=reading, cut=cut)
    return {
        "site": site,
        "cut": cut_name,
        "level": level,
        "reactive": reactive,
        "p1_training_sunny": value,
        "n_training_sunny": count,
        "chosen": "l0" if value < THRESHOLD_POINTS else "n",
    }


def census() -> list[dict[str, Any]]:
    """Return the statistic at the 8 MVA-only Primaries at the scored cut."""
    cache = common.load_cache()
    rows = []
    for site in common.MVA_SITES:
        value, count = statistic(
            cache=cache, site=site, reading=cache.sites[site].y, cut=known.CUTS["scored"]
        )
        rows.append(
            {
                "site": site,
                "p1_training_sunny": value,
                "n_training_sunny": count,
                "chosen": "l0" if value < THRESHOLD_POINTS else "n",
            }
        )
    return rows


def main() -> None:
    """Compute the statistic, join the stored scores, and print the numbers."""
    WORK.mkdir(parents=True, exist_ok=True)
    jobs = [
        (site, cut, level, reactive)
        for cut in known.CUTS
        for level in known.LEVELS
        for reactive in known.REACTIVE
        for site in known.SITES
    ]
    stats = pl.DataFrame([cell(job) for job in jobs])
    stored = pl.DataFrame(
        [json.loads(path.read_text()) for path in sorted(KNOWN_DIR.glob("*.json"))],
        infer_schema_length=None,
    )
    joined = (
        stats.join(stored, on=["site", "cut", "level", "reactive"])
        .with_columns(
            acc_chosen=pl.when(pl.col("chosen") == "l0")
            .then(pl.col("acc_l0_sunny"))
            .otherwise(pl.col("acc_n_sunny")),
            cap_chosen=pl.when(pl.col("chosen") == "l0")
            .then(pl.col("capacity_l0"))
            .otherwise(pl.col("capacity_n")),
        )
        .with_columns(
            gap_to_signed=pl.col("acc_s_sunny") - pl.col("acc_chosen"),
            cap_ratio=pl.col("cap_chosen") / pl.col("capacity_s"),
        )
    )
    joined.write_parquet(WORK / "rule.parquet")
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    for cut in known.CUTS:
        for reactive in known.REACTIVE:
            part = joined.filter(pl.col("cut") == cut, pl.col("reactive") == reactive).sort(
                "level", "site"
            )
            print(f"\n== {cut} cut, {reactive} ==")
            print(
                part.select(
                    "level",
                    "site",
                    pl.col("p1_training_sunny").round(2),
                    "n_training_sunny",
                    "chosen",
                    pl.col("acc_s_sunny").round(3),
                    pl.col("acc_l0_sunny").round(3),
                    pl.col("acc_n_sunny").round(3),
                    pl.col("acc_chosen").round(3),
                    pl.col("gap_to_signed").round(3),
                    pl.col("cap_ratio").round(2),
                )
            )
            for level in known.LEVELS:
                sub = part.filter(pl.col("level") == level)
                print(
                    f"level {level}: sites within 0.02 of the signed fit:",
                    int((sub["gap_to_signed"].abs() <= 0.02).sum()),
                    "of",
                    len(sub),
                )
    print("\n== census at the 8 MVA-only Primaries (scored cut, not scored) ==")
    census_table = pl.DataFrame(census())
    census_table.write_parquet(WORK / "census.parquet")
    print(census_table.with_columns(pl.col("p1_training_sunny").round(2)))


if __name__ == "__main__":
    main()
