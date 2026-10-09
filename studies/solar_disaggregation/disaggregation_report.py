"""Write every number the solar-disaggregation page quotes to `report.md`, and print the report.

Run after `stage1_single_sites.py`, `stage1b_synthetic_recovery.py`,
`stage2_synthetic_separation.py`, and `stage3_real_aggregates.py`:
`uv run python studies/solar_disaggregation/disaggregation_report.py`.
"""

from collections.abc import Callable
from typing import Final

import numpy as np
import polars as pl
from inputs import OUTPUT_DIR, PURE_PV_BMUS, WEATHER_PATH, grid_point_ids, hourly_cams

BOOTSTRAPS: Final[int] = 2000
SEED: Final[int] = 20261008
CLIP_GAIN_THRESHOLD: Final[float] = 0.02
"""A site's AC capacity counts as identified when allowing clipping lowers the fit's loss by at
least this share against a fit that cannot clip."""
BEST_ARMS: Final[tuple[str, ...]] = ("free", "fixed", "tracker")


def markdown_table(*, frame: pl.DataFrame, decimals: int = 2) -> str:
    """Render a frame as a markdown table, rounding floats to `decimals` places.

    Args:
        frame: The rows to render.
        decimals: The decimal places of each float.

    Returns:
        The table, with a header row, a rule, and one row per frame row.
    """
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = []
    for row in frame.iter_rows():
        cells = []
        for value in row:
            if value is None:
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.{decimals}f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows])


def site_interval(
    *, values: np.ndarray, statistic: Callable[[np.ndarray], float] = np.mean
) -> tuple[float, float, float]:
    """Return a statistic and its 95% interval from resampling whole sites.

    Args:
        values: One value per site.
        statistic: The function reducing a resample to one number.

    Returns:
        The statistic of the values, and the 2.5th and 97.5th percentiles of its bootstrap draws.
    """
    rng = np.random.default_rng(SEED)
    draws = [
        statistic(values[rng.integers(0, len(values), size=len(values))]) for _ in range(BOOTSTRAPS)
    ]
    return (
        float(statistic(values)),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def stage1_capacity() -> tuple[str, dict[str, float]]:
    """Return the stage 1 capacity table and the P1 summary, as markdown.

    Returns:
        The table, and the P1 summary numbers.
    """
    baselines = pl.read_parquet(OUTPUT_DIR / "stage1_baselines.parquet")
    fits = pl.read_parquet(OUTPUT_DIR / "stage1_fits.parquet").filter(pl.col("fold") == -1)
    arms = fits.filter(pl.col("arm").is_in(["free", "fixed", "tracker", "free_no_clip"])).pivot(
        on="arm", index="bmu", values=["loss", "ac_capacity_mw"]
    )
    table = (
        baselines.join(arms, on="bmu")
        .with_columns(
            clip_gain=1 - pl.col("loss_free") / pl.col("loss_free_no_clip"),
            pure_pv=pl.col("bmu").is_in(list(PURE_PV_BMUS)),
        )
        .with_columns(identified=pl.col("clip_gain") >= CLIP_GAIN_THRESHOLD)
    )
    estimates = {
        "physical, free orientation": "ac_capacity_mw_free",
        "physical, fixed orientation": "ac_capacity_mw_fixed",
        "envelope, cosine shape": "envelope_cosine_mw",
        "envelope, CAMS own point": "envelope_cams_mw",
        "99th percentile of output": "p99_output_mw",
        "largest output": "max_output_mw",
    }
    shown = table.select(
        "bmu",
        "technology",
        pl.col("generation_capacity_mw").alias("generation_capacity"),
        pl.col("repd_installed_capacity_mw").alias("repd"),
        pl.col("lccc_contract_capacity_mw").alias("cfd"),
        *[pl.col(column).alias(name) for name, column in estimates.items()],
        (pl.col("ac_capacity_mw_free") < pl.col("p99_output_mw")).alias("fitted AC below p99"),
        pl.col("clip_gain").mul(100).alias("clip_gain_pct"),
        "identified",
    ).sort("bmu")
    lines = [
        "## Stage 1: AC capacity at the 11 single-site BMUs (MW), capacity hidden from every fit",
        "",
    ]
    lines.append(markdown_table(frame=shown, decimals=1))
    lines.append("")
    summary: dict[str, float] = {}
    groups = {
        "BMUs whose AC capacity is identified": table.filter(pl.col("identified")),
        "BMUs whose AC capacity is not identified": table.filter(~pl.col("identified")),
        "all 11 BMUs": table,
        "pure PV": table.filter(pl.col("pure_pv")),
        "hybrid": table.filter(~pl.col("pure_pv")),
    }
    rows = []
    for group, frame in groups.items():
        row: dict[str, object] = {"group": group, "bmus": frame.height}
        for name, column in estimates.items():
            error = (frame[column] / frame["generation_capacity_mw"] - 1).abs().to_numpy()
            row[name] = float(error.mean()) * 100 if len(error) else None
        rows.append(row)
    lines += [
        "## P1: mean absolute error of the capacity estimate against Generation Capacity (%)",
        "",
        markdown_table(frame=pl.DataFrame(rows), decimals=1),
        "",
    ]
    # Paired differences in absolute relative error: negative means the physical fit is closer.
    physical = (table["ac_capacity_mw_free"] / table["generation_capacity_mw"] - 1).abs().to_numpy()
    contrast_rows = []
    for name in (
        "envelope, cosine shape",
        "envelope, CAMS own point",
        "99th percentile of output",
        "largest output",
    ):
        other = (table[estimates[name]] / table["generation_capacity_mw"] - 1).abs().to_numpy()
        diff = (physical - other) * 100
        mean, low, high = site_interval(values=diff)
        contrast_rows.append(
            {
                "physical free fit minus": name,
                "all 11 BMUs: points [low, high]": f"{mean:.1f} [{low:.1f}, {high:.1f}]",
            }
        )
        identified = table["identified"].to_numpy()
        if identified.sum() > 2:
            mean_i, low_i, high_i = site_interval(values=diff[identified])
            contrast_rows[-1]["identified BMUs: points [low, high]"] = (
                f"{mean_i:.1f} [{low_i:.1f}, {high_i:.1f}]"
            )
    lines += [
        (
            "## P1 contrasts: error of the physical free-orientation fit minus the comparator's "
            "error (percentage points of Generation Capacity; negative favours the physical fit)"
        ),
        "",
        markdown_table(frame=pl.DataFrame(contrast_rows)),
        "",
    ]
    doubtful = table.filter(pl.col("bmu").is_in(["C__LSTAT020", "C__ESTAT019"])).select(
        "bmu",
        pl.col("generation_capacity_mw").alias("generation_capacity"),
        pl.col("lccc_contract_capacity_mw").alias("cfd"),
        pl.col("repd_installed_capacity_mw").alias("repd"),
        pl.col("ac_capacity_mw_free").alias("physical, free orientation"),
        pl.col("envelope_cams_mw").alias("envelope, CAMS own point"),
        pl.col("max_output_mw").alias("largest output"),
    )
    lines += [
        (
            "## Sensitivity: the two BMUs whose Generation Capacity differs from their CfD or "
            "REPD capacity (MW)"
        ),
        "",
        markdown_table(frame=doubtful, decimals=1),
        "",
    ]
    return "\n".join(lines), summary


def stage1_heldout() -> str:
    """Return the P2 table and the parameter tables, as markdown.

    Returns:
        The markdown.
    """
    predictions = pl.read_parquet(OUTPUT_DIR / "stage1_predictions.parquet")
    capacities = pl.read_parquet(OUTPUT_DIR / "stage1_baselines.parquet").select(
        "bmu", "generation_capacity_mw"
    )
    scored = (
        predictions.join(capacities, on="bmu")
        .with_columns(
            error=(pl.col("output_mw") - pl.col("predicted_mw")).abs()
            / pl.col("generation_capacity_mw")
        )
        .group_by("bmu", "arm", "fold")
        .agg(pl.col("error").mean().alias("nmae"), pl.len().alias("rows"))
    )
    by_site = (
        scored.group_by("bmu", "arm")
        .agg(pl.col("nmae").mean())
        .pivot(on="arm", index="bmu", values="nmae")
        .sort("bmu")
    )
    lines = [
        (
            "## Stage 1: held-out error of the fitted plants (mean absolute error, % of "
            "Generation Capacity)"
        ),
        "",
    ]
    lines.append(markdown_table(frame=by_site.with_columns(pl.exclude("bmu") * 100), decimals=2))
    lines.append("")
    summary_rows = []
    for arm in BEST_ARMS:
        mean, low, high = site_interval(values=by_site[arm].to_numpy() * 100)
        summary_rows.append(
            {"arm": arm, "mean over sites": mean, "interval low": low, "interval high": high}
        )
    lines += [
        "### Mean over the 11 BMUs, with a 95% interval from resampling BMUs",
        "",
        markdown_table(frame=pl.DataFrame(summary_rows)),
        "",
    ]
    contrast_rows = []
    fold_means = (
        scored.group_by("arm", "fold")
        .agg(pl.col("nmae").mean())
        .pivot(on="arm", index="fold", values="nmae")
        .sort("fold")
    )
    for other in ("fixed", "tracker"):
        diff = (by_site["free"] - by_site[other]).to_numpy() * 100
        mean, low, high = site_interval(values=diff)
        fold_diff = ((fold_means["free"] - fold_means[other]) * 100).to_list()
        contrast_rows.append(
            {
                "free orientation minus": other,
                "points [low, high]": f"{mean:.2f} [{low:.2f}, {high:.2f}]",
                "BMUs where free is better": f"{int((diff < 0).sum())} of {len(diff)}",
                "folds where free is better": f"{sum(d < 0 for d in fold_diff)} of 4",
                "per-fold differences": ", ".join(f"{d:.2f}" for d in fold_diff),
            }
        )
    lines += [
        (
            "## P2: held-out error of the free-orientation fit minus the comparator's "
            "(percentage points of Generation Capacity; negative favours free orientation)"
        ),
        "",
        markdown_table(frame=pl.DataFrame(contrast_rows)),
        "",
    ]
    fits = pl.read_parquet(OUTPUT_DIR / "stage1_fits.parquet")
    full = fits.filter((pl.col("arm") == "free") & (pl.col("fold") == -1)).select(
        "bmu", "tilt_deg", "azimuth_deg", "dc_ac_ratio", "ac_capacity_mw"
    )
    spread = (
        fits.filter((pl.col("arm") == "free") & (pl.col("fold") >= 0))
        .group_by("bmu")
        .agg(
            pl.col("tilt_deg").std().alias("tilt sd across folds"),
            pl.col("azimuth_deg").std().alias("azimuth sd across folds"),
            pl.col("dc_ac_ratio").std().alias("DC:AC sd across folds"),
        )
    )
    lines += [
        (
            "## Stage 1: fitted parameters, free orientation (all data) and their spread across "
            "the four held-out folds"
        ),
        "",
        markdown_table(frame=full.join(spread, on="bmu").sort("bmu"), decimals=1),
        "",
    ]
    offsets = (
        fits.filter(pl.col("arm") == "offset_scan")
        .pivot(on="offset", index="bmu", values="loss")
        .sort("bmu")
    )
    best = offsets.with_columns(
        best_offset=pl.concat_list(pl.exclude("bmu")).list.arg_min() - 2
    ).select("bmu", "best_offset")
    lines += [
        (
            "## Timestamp alignment: the output shifted by n half-hours, loss of the "
            "fixed-orientation fit"
        ),
        "",
        markdown_table(frame=offsets.join(best, on="bmu"), decimals=2),
        "",
    ]
    return "\n".join(lines)


def stage1_recovery() -> str:
    """Return the synthetic parameter-recovery table, as markdown.

    Returns:
        The markdown.
    """
    frame = pl.read_parquet(OUTPUT_DIR / "stage1b_recovery.parquet").with_columns(
        tilt_error=pl.col("fitted_tilt_deg") - pl.col("true_tilt_deg"),
        azimuth_error=pl.col("fitted_azimuth_deg") - pl.col("true_azimuth_deg"),
        ratio_error=pl.col("fitted_dc_ac_ratio") - pl.col("true_dc_ac_ratio"),
        ac_error_percent=(pl.col("fitted_ac_capacity_mw") / pl.col("true_ac_capacity_mw") - 1)
        * 100,
    )
    rows = [
        {
            "parameter": name,
            "mean error": float(frame[column].mean()),  # ty: ignore[invalid-argument-type]
            "mean absolute error": float(frame[column].abs().mean()),  # ty: ignore[invalid-argument-type]
            "largest absolute error": float(frame[column].abs().max()),  # ty: ignore[invalid-argument-type]
            "correlation of fitted with true": float(np.corrcoef(frame[fitted], frame[true])[0, 1])
            if fitted
            else None,
        }
        for name, column, fitted, true in (
            ("tilt (degrees)", "tilt_error", "fitted_tilt_deg", "true_tilt_deg"),
            ("azimuth (degrees)", "azimuth_error", "fitted_azimuth_deg", "true_azimuth_deg"),
            ("DC:AC ratio", "ratio_error", "fitted_dc_ac_ratio", "true_dc_ac_ratio"),
            ("AC capacity (%)", "ac_error_percent", "", ""),
        )
    ]
    lines = [
        (
            f"## Stage 1b: recovery of known parameters from simulated output ({frame.height} "
            "simulated plants, a different irradiance chain from the fit's)"
        ),
        "",
        markdown_table(frame=pl.DataFrame(rows), decimals=2),
        "",
    ]
    return "\n".join(lines)


PRIMARY: Final[tuple[str, str]] = ("separation_cams_regional", "seasonal")
"""The method and baseline flexibility that the plan names for P3."""
P3_COMPARATORS: Final[dict[str, tuple[str, str]]] = {
    "envelope-scaled curve (census method)": ("envelope_scaled", "none"),
    "separation with clear-sky irradiance (no cloud)": ("separation_clear_sky", "seasonal"),
    "separation with the GB-mean of 18 CAMS points": ("separation_cams_gb_mean", "seasonal"),
    "separation with the solar sites' own CAMS points (oracle regressor)": (
        "separation_cams_own_sites",
        "seasonal",
    ),
    "separation with the non-solar part known (oracle baseline)": ("oracle_baseline", "none"),
}
NON_SOLAR_PLANNED: Final[tuple[str, ...]] = (
    "offshore_wind",
    "gas_peaker",
    "gas_baseload",
    "pumped_storage",
    "battery_lakeside",
    "battery_dollymans",
)
"""The six non-solar halves with no PV behind them."""


def _planned(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the planned scenarios: planned solar sets, six non-solar BMUs, no injected demand."""
    return frame.filter(
        pl.col("planned")
        & pl.col("non_solar").is_in(NON_SOLAR_PLANNED)
        & ~pl.col("weather_demand")
        & (pl.col("share") > 0)
    )


def _month_array(
    *, monthly: pl.DataFrame, method: tuple[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-scenario, per-month absolute-error sums and row counts for one method.

    The sums are scaled by the truth's 99th percentile.
    """
    frame = (
        _planned(frame=monthly)
        .filter((pl.col("method") == method[0]) & (pl.col("flexibility") == method[1]))
        .sort("solar_set", "non_solar", "share", "month_index")
    )
    scenarios = frame.select("solar_set", "non_solar", "share").unique(maintain_order=True).height
    errors = (frame["abs_error_sum"] / frame["truth_p99_mw"]).to_numpy().reshape(scenarios, 12)
    rows = frame["rows"].to_numpy().reshape(scenarios, 12)
    return errors, rows


def _nmae_from_months(*, errors: np.ndarray, rows: np.ndarray, chosen: np.ndarray) -> float:
    """Return the mean over scenarios of the normalised mean absolute error on the chosen months."""
    return float(np.mean(errors[:, chosen].sum(axis=1) / rows[:, chosen].sum(axis=1)))


POST_HOC_METHODS: Final[dict[str, tuple[str, str]]] = {
    "calendar-baseline separation (planned)": PRIMARY,
    "difference separation (post hoc)": ("difference_cams_regional", "lag8"),
    "difference separation, clear-sky irradiance (post hoc)": ("difference_clear_sky", "lag8"),
    "envelope-scaled curve (census method)": ("envelope_scaled", "none"),
}
"""The methods stage 2 scores, with the fit that stage 2c adds as a separate row."""
OTHER_GENERATION_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "wind and gas": ("offshore_wind", "gas_peaker", "gas_baseload"),
    "pumped storage": ("pumped_storage",),
    "batteries": ("battery_lakeside", "battery_dollymans"),
}
"""The non-solar halves, grouped by how they relate to the sun."""


def _stage2_row(*, label: str, sub: pl.DataFrame) -> dict[str, object]:
    """Summarise a set of stage 2 scenarios for one method."""
    return {
        "method": label,
        "scenarios": sub.height,
        "mean error (% of solar p99)": float(sub["nmae_of_solar_p99"].mean()) * 100,  # ty: ignore[invalid-argument-type]
        "mean energy ratio": float(sub["energy_ratio"].mean()),  # ty: ignore[invalid-argument-type]
        "mean correlation": float(sub["correlation"].mean()),  # ty: ignore[invalid-argument-type]
        "false solar p99 in the no-solar aggregates (MW of 100): mean": None,
    }


def _confound_and_control_tables(*, scores: pl.DataFrame) -> list[str]:
    """Return the tables of weather-correlated confounds and of the no-solar controls."""
    planned = scores.filter(pl.col("planned") & pl.col("non_solar").is_in(NON_SOLAR_PLANNED))
    physical = pl.read_parquet(OUTPUT_DIR / "stage2c_physical_fit.parquet").filter(
        (pl.col("prior") == "none") & (pl.col("regressor") == "regional")
    )
    rows: list[dict[str, object]] = []
    for group, members in OTHER_GENERATION_GROUPS.items():
        for demand in (False, True):
            if demand and group != "wind and gas":
                continue
            scenario = pl.col("non_solar").is_in(members) & (pl.col("share") > 0)
            scenario &= pl.col("weather_demand") if demand else ~pl.col("weather_demand")
            suffix = ", with a demand that rises on dull days" if demand else ""
            for label, (method, flexibility) in POST_HOC_METHODS.items():
                sub = planned.filter(
                    scenario & (pl.col("method") == method) & (pl.col("flexibility") == flexibility)
                )
                row = _stage2_row(label=label, sub=sub)
                row.pop("false solar p99 in the no-solar aggregates (MW of 100): mean")
                rows.append({"other generation": group + suffix, **row})
            sub = physical.filter(scenario)
            row = _stage2_row(label="physical fit to changes (post hoc)", sub=sub)
            row.pop("false solar p99 in the no-solar aggregates (MW of 100): mean")
            rows.append({"other generation": group + suffix, **row})
    lines = [
        (
            "## Weather-correlated confounds: error, energy, and shape of the recovered solar "
            "(planned solar halves, all three solar shares)"
        ),
        "",
        markdown_table(frame=pl.DataFrame(rows)),
        "",
    ]
    control_rows: list[dict[str, object]] = []
    for demand in (False, True):
        sub_scores = planned.filter((pl.col("share") == 0.0) & (pl.col("weather_demand") == demand))
        for label, (method, flexibility) in POST_HOC_METHODS.items():
            sub = sub_scores.filter(
                (pl.col("method") == method) & (pl.col("flexibility") == flexibility)
            )
            control_rows.append(
                {
                    "no-solar aggregate": "with a demand that rises on dull days"
                    if demand
                    else "plain",
                    "method": label,
                    "aggregates": sub.height,
                    "mean recovered p99 (MW of 100)": float(sub["estimate_p99_mw"].mean()),  # ty: ignore[invalid-argument-type]
                    "largest recovered p99 (MW of 100)": float(sub["estimate_p99_mw"].max()),  # ty: ignore[invalid-argument-type]
                    "aggregates under the 2 MW limit": int((sub["estimate_p99_mw"] < 2.0).sum()),
                }
            )
    lines += [
        (
            "## P4: negative control, an aggregate with no solar (the planned limit is 2% of the "
            "100 MW aggregate's 99th percentile)"
        ),
        "",
        markdown_table(frame=pl.DataFrame(control_rows)),
        "",
    ]
    return lines


def stage2_tables() -> str:
    """Return the P3 and P4 tables, the confound tables, and the supplier extra, as markdown.

    Returns:
        The markdown.
    """
    scores = pl.read_parquet(OUTPUT_DIR / "stage2_scores.parquet")
    monthly = pl.read_parquet(OUTPUT_DIR / "stage2_monthly.parquet")
    lines = [
        (
            "## P3: recovery of the solar series from a synthetic sum (mean absolute error as % "
            "of the solar part's 99th-percentile output; smaller is better)"
        ),
        "",
    ]
    methods = {"separation with regional CAMS (primary)": PRIMARY, **P3_COMPARATORS}
    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, 12, size=12) for _ in range(BOOTSTRAPS)]
    arrays = {
        name: _month_array(monthly=monthly, method=method) for name, method in methods.items()
    }
    rows = []
    for share in (None, 0.1, 0.25, 0.5):
        row: dict[str, object] = {
            "solar share of aggregate p99": "all" if share is None else f"{share:.0%}"
        }
        for name, method in methods.items():
            subset = monthly if share is None else monthly.filter(pl.col("share") == share)
            frame = (
                _planned(frame=subset)
                .filter((pl.col("method") == method[0]) & (pl.col("flexibility") == method[1]))
                .sort("solar_set", "non_solar", "share", "month_index")
            )
            n = frame.select("solar_set", "non_solar", "share").unique().height
            errors = (frame["abs_error_sum"] / frame["truth_p99_mw"]).to_numpy().reshape(n, 12)
            counts = frame["rows"].to_numpy().reshape(n, 12)
            point = _nmae_from_months(errors=errors, rows=counts, chosen=np.arange(12)) * 100
            boots = [_nmae_from_months(errors=errors, rows=counts, chosen=d) * 100 for d in draws]
            row[name] = (
                f"{point:.1f} [{np.quantile(boots, 0.025):.1f}, {np.quantile(boots, 0.975):.1f}]"
            )
        rows.append(row)
    lines += [markdown_table(frame=pl.DataFrame(rows)), ""]
    primary_errors, primary_rows = arrays["separation with regional CAMS (primary)"]
    diff_rows = []
    for name in P3_COMPARATORS:
        errors, counts = arrays[name]
        point = (
            _nmae_from_months(errors=primary_errors, rows=primary_rows, chosen=np.arange(12))
            - _nmae_from_months(errors=errors, rows=counts, chosen=np.arange(12))
        ) * 100
        boots = [
            (
                _nmae_from_months(errors=primary_errors, rows=primary_rows, chosen=d)
                - _nmae_from_months(errors=errors, rows=counts, chosen=d)
            )
            * 100
            for d in draws
        ]
        diff_rows.append(
            {
                "primary separation minus": name,
                "points [low, high]": (
                    f"{point:.2f} [{np.quantile(boots, 0.025):.2f}, "
                    f"{np.quantile(boots, 0.975):.2f}]"
                ),
            }
        )
    lines += [
        (
            "### Paired differences over all planned scenarios (percentage points; negative "
            "favours the primary separation; 95% interval from resampling whole calendar months)"
        ),
        "",
        markdown_table(frame=pl.DataFrame(diff_rows)),
        "",
    ]
    # Spread across scenarios for the primary and the envelope method.
    spread = (
        _planned(frame=scores)
        .filter(
            ((pl.col("method") == PRIMARY[0]) & (pl.col("flexibility") == PRIMARY[1]))
            | (pl.col("method") == "envelope_scaled")
        )
        .group_by("method", "non_solar")
        .agg(pl.col("nmae_of_solar_p99").mean().mul(100), pl.col("energy_ratio").mean())
        .sort("non_solar", "method")
    )
    lines += [
        "### Mean error and energy ratio by non-solar half (planned solar sets, all shares)",
        "",
        markdown_table(frame=spread),
        "",
    ]
    by_set = (
        _planned(frame=scores)
        .filter(
            pl.col("method").is_in([PRIMARY[0], "envelope_scaled"])
            & pl.col("flexibility").is_in([PRIMARY[1], "none"])
        )
        .group_by("method", "solar_set")
        .agg(pl.col("nmae_of_solar_p99").mean().mul(100), pl.col("energy_ratio").mean())
        .sort("solar_set", "method")
    )
    lines += ["### Mean error and energy ratio by solar half", "", markdown_table(frame=by_set), ""]
    flexibility = (
        _planned(frame=scores)
        .filter(pl.col("method") == PRIMARY[0])
        .group_by("flexibility")
        .agg(pl.col("nmae_of_solar_p99").mean().mul(100), pl.col("energy_ratio").mean())
        .sort("flexibility")
    )
    lines += [
        "### Exploratory: baseline flexibility, primary regressor",
        "",
        markdown_table(frame=flexibility),
        "",
    ]
    lines += _confound_and_control_tables(scores=scores)
    # Supplier extra.
    supplier = (
        scores.filter(
            (pl.col("non_solar") == "supplier_totalenergies")
            & (pl.col("share") > 0)
            & ~pl.col("weather_demand")
            & (
                ((pl.col("method") == PRIMARY[0]) & (pl.col("flexibility") == PRIMARY[1]))
                | (pl.col("method") == "envelope_scaled")
            )
        )
        .group_by("method", "share")
        .agg(
            pl.col("nmae_of_solar_p99").mean().mul(100),
            pl.col("energy_ratio").mean(),
            pl.col("estimate_p99_mw").mean(),
        )
        .sort("share", "method")
    )
    lines += [
        (
            "## Extra: a supplier BMU as the non-solar half (its true solar content is unknown, "
            "so errors are against the added solar only)"
        ),
        "",
        markdown_table(frame=supplier),
        "",
    ]
    return "\n".join(lines)


def decision_rule_tables() -> str:
    """Return the stage 0 tables: which aggregate BMUs fit the envelope model, by capacity.

    Returns:
        The markdown.
    """
    frame = pl.read_parquet(OUTPUT_DIR / "stage0_decision_rule.parquet").sort("bmu")
    with_output = frame.filter(pl.col("behaviour") != "no positive output")
    rows = []
    for behaviour in ("clean", "structured"):
        subset = with_output.filter(pl.col("behaviour") == behaviour)
        rows.append(
            {
                "behaviour": behaviour,
                "BMUs": subset.height,
                "Generation Capacity (MW)": float(subset["generation_capacity_mw"].sum()),
                "envelope estimate, cosine (MW)": float(subset["envelope_cosine_mw"].sum()),
                "envelope estimate, all-GB CAMS mean (MW)": float(subset["envelope_cams_mw"].sum()),
            }
        )
    totals = {
        "behaviour": "all with output",
        "BMUs": with_output.height,
        "Generation Capacity (MW)": float(with_output["generation_capacity_mw"].sum()),
        "envelope estimate, cosine (MW)": float(with_output["envelope_cosine_mw"].sum()),
        "envelope estimate, all-GB CAMS mean (MW)": float(with_output["envelope_cams_mw"].sum()),
    }
    lines = [
        (
            "## Stage 0: the decision rule on the 26 aggregate BMUs (clean: under 5% of "
            "half-hours negative and night output under 5% of the 99th percentile)"
        ),
        "",
        markdown_table(frame=pl.DataFrame([*rows, totals]), decimals=1),
        "",
        markdown_table(
            frame=frame.select(
                "bmu",
                "behaviour",
                pl.col("generation_capacity_mw").alias("generation_capacity"),
                "p99_output_mw",
                pl.col("negative_share").mul(100).alias("negative_half_hours_pct"),
                pl.col("night_ratio").mul(100).alias("night_output_pct_of_p99"),
            ),
            decimals=1,
        ),
        "",
    ]
    return "\n".join(lines)


def ridge_tables() -> str:
    """Return the stage 2b tables: level and shape of the calendar-baseline separation by ridge.

    Returns:
        The markdown.
    """
    frame = pl.read_parquet(OUTPUT_DIR / "stage2b_level_and_shape.parquet")
    with_solar = (
        frame.filter(pl.col("share") > 0)
        .group_by("ridge")
        .agg(
            pl.col("energy_ratio").mean().alias("energy ratio (recovered / true)"),
            pl.col("correlation").mean().alias("correlation with true solar"),
            pl.col("nmae").mean().mul(100).alias("error (% of solar p99)"),
            pl.col("nmae_rescaled")
            .mean()
            .mul(100)
            .alias("error after rescaling to the true energy (% of solar p99)"),
        )
        .sort("ridge")
    )
    controls = (
        frame.filter(pl.col("share") == 0)
        .group_by("ridge")
        .agg(
            pl.col("estimate_p99_mw").mean().alias("mean false solar p99 (MW of 100)"),
            pl.col("estimate_p99_mw").max().alias("largest false solar p99 (MW of 100)"),
        )
        .sort("ridge")
    )
    lines = [
        (
            "## Stage 2b: the calendar-baseline separation's level, shape, and false solar by "
            "ridge penalty (exploratory; 1e-3 is the planned value)"
        ),
        "",
        markdown_table(frame=with_solar.join(controls, on="ridge"), decimals=3),
        "",
    ]
    return "\n".join(lines)


def physical_tables() -> str:
    """Return the stage 2c tables: detection, parameter errors, and series errors.

    Returns:
        The markdown.
    """
    both_skies = pl.read_parquet(OUTPUT_DIR / "stage2c_physical_fit.parquet")
    everything = both_skies.filter(pl.col("regressor") == "regional")
    frame = everything.filter(pl.col("prior") == "none")
    no_solar = frame.filter((pl.col("share") == 0) & ~pl.col("weather_demand"))
    threshold = float(no_solar["variance_explained"].max())  # ty: ignore[invalid-argument-type]
    detected = pl.col("variance_explained") > threshold
    lines = [
        (
            "## Stage 2c: the physical fit to changes, regional sky (detection "
            f"threshold {threshold:.5f}: the largest variance explained over the "
            f"{frame.filter((pl.col('share') == 0) & ~pl.col('weather_demand')).height} no-solar "
            "aggregates)"
        ),
        "",
    ]
    with_solar = frame.filter((pl.col("share") > 0) & ~pl.col("weather_demand")).with_columns(
        tilt_error=(pl.col("fitted_tilt_deg") - pl.col("reference_tilt_deg")).abs(),
        azimuth_error=(pl.col("fitted_azimuth_deg") - pl.col("reference_azimuth_deg")).abs(),
        ratio_error=(pl.col("fitted_dc_ac_ratio") - pl.col("reference_dc_ac_ratio")).abs(),
        ac_error=(pl.col("fitted_ac_mw") / pl.col("reference_ac_mw") - 1).abs() * 100,
        detected=detected,
    )
    by_share = (
        with_solar.group_by("share")
        .agg(
            pl.len().alias("aggregates"),
            pl.col("detected").mean().mul(100).alias("detected (%)"),
            pl.col("variance_explained").median().alias("median variance explained"),
        )
        .sort("share")
    )
    lines += ["### Detection by solar share", "", markdown_table(frame=by_share, decimals=3), ""]
    group_columns = {
        "tilt_error": "median tilt error (degrees)",
        "azimuth_error": "median azimuth error (degrees)",
        "ratio_error": "median DC:AC error",
        "ac_error": "median AC capacity error (%)",
        "nmae_of_solar_p99": "median series error (% of solar p99)",
        "energy_ratio": "median energy ratio",
        "correlation": "median correlation",
    }

    def summarise(*, by: str) -> pl.DataFrame:
        return (
            with_solar.filter(pl.col("detected"))
            .group_by(by)
            .agg(
                pl.len().alias("detected aggregates"),
                *[pl.col(c).median().alias(label) for c, label in group_columns.items()],
            )
            .sort(by)
        )

    with_percent = lambda frame_: frame_.with_columns(  # noqa: E731
        (pl.col("median series error (% of solar p99)") * 100).alias(
            "median series error (% of solar p99)"
        )
    )
    lines += [
        "### Parameter and series errors of the detected aggregates, by other generation",
        "",
        markdown_table(frame=with_percent(summarise(by="non_solar")), decimals=2),
        "",
        "### By solar share",
        "",
        markdown_table(frame=with_percent(summarise(by="share")), decimals=2),
        "",
        "### By solar half",
        "",
        markdown_table(frame=with_percent(summarise(by="solar_set")), decimals=2),
        "",
    ]
    weather = (
        frame.filter(pl.col("weather_demand"))
        .group_by("share")
        .agg(
            pl.len().alias("aggregates"),
            pl.col("variance_explained").max().alias("largest variance explained"),
            (pl.col("variance_explained") > threshold).sum().alias("detected"),
        )
        .sort("share")
    )
    lines += [
        "### Control: a demand term that rises on dull days (15 MW at its peak)",
        "",
        markdown_table(frame=weather, decimals=4),
        "",
    ]
    lines += _headline_and_prior_tables(everything=everything, threshold=threshold)
    lines += _regressor_tables(both_skies=both_skies)
    return "\n".join(lines)


def _median(values: pl.Series) -> float:
    """Return the median of `values`, or NaN when the series is empty."""
    result = values.median()
    return float("nan") if result is None else float(result)  # ty: ignore[invalid-argument-type]


def _percent(flags: pl.Series) -> float:
    """Return the percentage of `flags` that are true."""
    return 100.0 * float(flags.mean())  # ty: ignore[invalid-argument-type]


def _quantile(values: pl.Series, quantile: float) -> float:
    """Return a quantile of `values`, or NaN when the series is empty."""
    result = values.quantile(quantile)
    return float("nan") if result is None else float(result)


def _regressor_tables(*, both_skies: pl.DataFrame) -> list[str]:
    """Return the stage 2c tables that compare the regional sky with the GB-mean sky.

    The GB-mean sky is the one `stage3b_real_controls.py` gives every real BMU. Each sky has its
    own detection thresholds, the largest value over the plain no-solar aggregates.

    Args:
        both_skies: The stage 2c rows of both skies and every prior.

    Returns:
        The markdown lines.
    """
    frame = both_skies.filter(pl.col("prior") == "none")
    plain_no_solar = frame.filter((pl.col("share") == 0) & ~pl.col("weather_demand"))
    thresholds = plain_no_solar.group_by("regressor").agg(
        pl.col("variance_explained").max().alias("ve_threshold"),
        pl.col("cloud_increment").max().alias("increment_threshold"),
    )
    scored = frame.join(thresholds, on="regressor").with_columns(
        detected_by_ve=pl.col("variance_explained") > pl.col("ve_threshold"),
        detected_by_increment=pl.col("cloud_increment") > pl.col("increment_threshold"),
        ac_ratio=pl.col("fitted_ac_mw") / pl.col("reference_ac_mw"),
    )
    detection_rows = []
    for (regressor, share, demand), sub in scored.group_by(
        "regressor", "share", "weather_demand", maintain_order=True
    ):
        detection_rows.append(
            {
                "regressor": regressor,
                "solar share": share,
                "dull-day demand": demand,
                "aggregates": sub.height,
                "median variance explained": _median(sub["variance_explained"]),
                "median clear-sky variance explained": _median(sub["variance_explained_clear_sky"]),
                "median cloud increment (primary)": _median(sub["cloud_increment"]),
                "detected by cloud increment (%)": 100.0
                * float(sub["detected_by_increment"].mean()),  # ty: ignore[invalid-argument-type]
                "detected by variance explained (%)": _percent(sub["detected_by_ve"]),
            }
        )
    detection = pl.DataFrame(detection_rows).sort("regressor", "dull-day demand", "solar share")
    threshold_table = thresholds.sort("regressor").rename(
        {
            "ve_threshold": "largest variance explained, plain no-solar aggregates",
            "increment_threshold": "largest cloud increment, plain no-solar aggregates (primary)",
        }
    )
    lines = [
        "## Stage 2c by sky: regional (3 nearest points) and GB mean (all 18 points)",
        "",
        "### Detection thresholds, one per sky",
        "",
        markdown_table(frame=threshold_table, decimals=4),
        "",
        (
            "### Detection by sky, solar share, and dull-day demand. The cloud increment is the "
            "variance explained minus the variance a clear-sky fit explains, and is the primary "
            "statistic"
        ),
        "",
        markdown_table(frame=detection, decimals=3),
        "",
    ]
    headline_rows = []
    quartile_rows = []
    errors = {
        "tilt error (degrees)": (pl.col("fitted_tilt_deg") - pl.col("reference_tilt_deg")).abs(),
        "azimuth error (degrees)": (
            pl.col("fitted_azimuth_deg") - pl.col("reference_azimuth_deg")
        ).abs(),
        "DC:AC error": (pl.col("fitted_dc_ac_ratio") - pl.col("reference_dc_ac_ratio")).abs(),
        "AC capacity error (%)": (pl.col("ac_ratio") - 1).abs() * 100,
    }
    with_solar = scored.filter((pl.col("share") > 0) & ~pl.col("weather_demand")).with_columns(
        **errors, series_error=pl.col("nmae_of_solar_p99") * 100
    )
    for regressor in ("regional", "gb_mean"):
        for group, members in OTHER_GENERATION_GROUPS.items():
            in_group = (pl.col("regressor") == regressor) & pl.col("non_solar").is_in(members)
            sub = with_solar.filter(in_group & (pl.col("share") >= 0.25) & pl.col("detected_by_ve"))
            headline_rows.append(
                {
                    "regressor": regressor,
                    "other generation": group,
                    "detected aggregates (25% and 50% shares, by variance explained)": sub.height,
                    "median series error (% of solar p99)": _median(sub["series_error"]),
                    "median energy ratio": _median(sub["energy_ratio"]),
                    "median correlation": _median(sub["correlation"]),
                    **{name: _median(sub[name]) for name in errors},
                }
            )
            for share in (0.1, 0.25, 0.5):
                every = with_solar.filter(in_group & (pl.col("share") == share))
                quartile_rows.append(
                    {
                        "regressor": regressor,
                        "other generation": group,
                        "solar share": share,
                        "aggregates": every.height,
                        "median fitted/reference AC": _median(every["ac_ratio"]),
                        "lower quartile": _quantile(every["ac_ratio"], 0.25),
                        "upper quartile": _quantile(every["ac_ratio"], 0.75),
                    }
                )
    lines += [
        "### Headline errors by sky: the detected aggregates with a 25% or 50% solar share",
        "",
        markdown_table(frame=pl.DataFrame(headline_rows), decimals=2),
        "",
        (
            "### Capacity recovery: fitted AC capacity over the reference AC capacity, every "
            "aggregate with solar and no dull-day demand (detected or not)"
        ),
        "",
        markdown_table(frame=pl.DataFrame(quartile_rows), decimals=2),
        "",
    ]
    return lines


def wind_clearness_correlation() -> float:
    """Return the correlation of daily mean ERA5 wind speed with the daily clearness index.

    The wind speed at 100 m is the mean over the 18 grid points. The clearness index of a day is
    its summed all-sky irradiance over its summed clear-sky irradiance, both CAMS means over the
    same 18 points. A day is a UTC day.

    Returns:
        The Pearson correlation over the days present in both series.
    """
    wind = (
        pl.read_parquet(WEATHER_PATH)
        .group_by("time")
        .agg(pl.col("wind_speed_100m_m_s").mean())
        .with_columns(day=pl.col("time").dt.truncate("1d"))
        .group_by("day")
        .agg(pl.col("wind_speed_100m_m_s").mean())
    )
    all_sky = hourly_cams(point_ids=grid_point_ids())
    clear = hourly_cams(point_ids=grid_point_ids(), column="clear_sky_ghi_w_m2")
    hours = all_sky.join(clear, on="time", suffix="_clear").with_columns(
        day=pl.col("time").dt.truncate("1d")
    )
    clearness = hours.group_by("day").agg(
        (pl.col("ghi_w_m2").sum() / pl.col("ghi_w_m2_clear").sum()).alias("clearness_index")
    )
    joined = wind.join(clearness, on="day").filter(pl.col("clearness_index").is_finite())
    return float(np.corrcoef(joined["wind_speed_100m_m_s"], joined["clearness_index"])[0, 1])


def _headline_and_prior_tables(*, everything: pl.DataFrame, threshold: float) -> list[str]:
    """Return the headline table of the physical fit and the table of fits with priors."""
    errors = {
        "tilt error (degrees)": (pl.col("fitted_tilt_deg") - pl.col("reference_tilt_deg")).abs(),
        "azimuth error (degrees)": (
            pl.col("fitted_azimuth_deg") - pl.col("reference_azimuth_deg")
        ).abs(),
        "DC:AC error": (pl.col("fitted_dc_ac_ratio") - pl.col("reference_dc_ac_ratio")).abs(),
        "AC capacity error (%)": (pl.col("fitted_ac_mw") / pl.col("reference_ac_mw") - 1).abs()
        * 100,
    }
    base = everything.filter((pl.col("share") > 0) & ~pl.col("weather_demand")).with_columns(
        **errors,
        series_error=pl.col("nmae_of_solar_p99") * 100,
        detected=pl.col("variance_explained") > threshold,
    )
    headline_rows = []
    for group, members in OTHER_GENERATION_GROUPS.items():
        sub = base.filter(
            (pl.col("prior") == "none")
            & pl.col("non_solar").is_in(members)
            & (pl.col("share") >= 0.25)
            & pl.col("detected")
        )
        headline_rows.append(
            {
                "other generation": group,
                "detected aggregates (25% and 50% shares)": sub.height,
                "median series error (% of solar p99)": _median(sub["series_error"]),
                "median energy ratio": _median(sub["energy_ratio"]),
                "median correlation": _median(sub["correlation"]),
                **{name: _median(sub[name]) for name in errors},
            }
        )
    lines = [
        "### Headline: the detected aggregates with a 25% or 50% solar share, by other generation",
        "",
        markdown_table(frame=pl.DataFrame(headline_rows), decimals=2),
        "",
    ]
    prior_rows = []
    for prior in ("none", "loose", "tight"):
        controls = everything.filter(
            (pl.col("prior") == prior) & (pl.col("share") == 0) & ~pl.col("weather_demand")
        )
        prior_threshold = float(controls["variance_explained"].max())  # ty: ignore[invalid-argument-type]
        for group, members in OTHER_GENERATION_GROUPS.items():
            sub = base.filter((pl.col("prior") == prior) & pl.col("non_solar").is_in(members))
            hit = sub.filter(pl.col("variance_explained") > prior_threshold)
            prior_rows.append(
                {
                    "prior": prior,
                    "other generation": group,
                    "aggregates": sub.height,
                    "detected (%)": 100.0 * hit.height / sub.height,
                    **{f"median {name}": _median(sub[name]) for name in errors},
                    "median series error (% of solar p99)": _median(sub["series_error"]),
                    "median energy ratio": _median(sub["energy_ratio"]),
                }
            )
    lines += [
        (
            "### Exploratory: priors on tilt, azimuth, and DC:AC ratio (all aggregates with solar, "
            "all shares, no weather demand; detection uses each prior's own no-solar threshold)"
        ),
        "",
        markdown_table(frame=pl.DataFrame(prior_rows), decimals=2),
        "",
    ]
    return lines


def real_tables() -> str:
    """Return the stage 3 and 3b tables, as markdown.

    Returns:
        The markdown.
    """
    controls = pl.read_parquet(OUTPUT_DIR / "stage3b_real_controls.parquet")
    grouped = (
        controls.group_by("group", "fuel")
        .agg(
            pl.len().alias("BMUs"),
            pl.col("variance_explained").median().alias("median"),
            pl.col("variance_explained").min().alias("lowest"),
            pl.col("variance_explained").max().alias("highest"),
            pl.col("variance_explained_clear_sky").median().alias("clear-sky median"),
            pl.col("variance_explained_clear_sky").min().alias("clear-sky lowest"),
            pl.col("variance_explained_clear_sky").max().alias("clear-sky highest"),
            pl.col("cloud_increment").median().alias("increment median (primary)"),
            pl.col("cloud_increment").min().alias("increment lowest"),
            pl.col("cloud_increment").max().alias("increment highest"),
        )
        .sort("group", "fuel")
    )
    positive = controls.filter(pl.col("group") == "positive control").sort("bmu")
    lines = [
        (
            "## Stage 3b: share of four-hour output changes explained by the fitted plant, real "
            "BMUs of known and unknown technology"
        ),
        "",
        markdown_table(frame=grouped, decimals=4),
        "",
        "### The known solar BMUs under the same protocol (all-GB CAMS mean, no position)",
        "",
        markdown_table(
            frame=positive.select(
                "bmu",
                "p99_abs_output_mw",
                "separation_capacity_mw",
                "ac_mw",
                "dc_ac_ratio",
                "tilt_deg",
                "azimuth_deg",
                "variance_explained",
                "variance_explained_clear_sky",
                "cloud_increment",
            ),
            decimals=2,
        ),
        "",
    ]
    separations = pl.read_parquet(OUTPUT_DIR / "stage3_separations.parquet")
    stage0 = pl.read_parquet(OUTPUT_DIR / "stage0_decision_rule.parquet").select(
        "bmu", "generation_capacity_mw", "p99_output_mw", "envelope_cosine_mw", "envelope_cams_mw"
    )
    replica = controls.filter(pl.col("group") == "calendar replica").select(
        "bmu", pl.col("cloud_increment").alias("replica_cloud_increment")
    )
    gb_mean = controls.filter(pl.col("group") == "aggregate").select(
        "bmu", pl.col("cloud_increment").alias("gb_mean_cloud_increment")
    )
    table = (
        separations.join(stage0, on="bmu")
        .join(gb_mean, on="bmu", how="left")
        .join(replica, on="bmu", how="left")
        .sort("physical_ac_mw", descending=True)
    )
    shown = table.select(
        "bmu",
        "gsp_group",
        pl.col("generation_capacity_mw").alias("generation_capacity"),
        pl.col("p99_output_mw").alias("p99_output"),
        pl.col("physical_ac_mw").alias("physical_ac"),
        pl.col("physical_dc_ac_ratio").alias("physical_dc_ac_ratio"),
        pl.col("physical_tilt_deg").alias("tilt_deg"),
        pl.col("physical_azimuth_deg").alias("azimuth_deg"),
        pl.col("physical_variance_explained").alias("variance_explained"),
        pl.col("physical_variance_explained_clear_sky").alias("clear_sky_variance_explained"),
        pl.col("physical_cloud_increment").alias("cloud_increment"),
        "gb_mean_cloud_increment",
        "replica_cloud_increment",
        pl.col("regional_difference_capacity_mw").alias("difference_capacity"),
        pl.col("regional_difference_capacity_low_mw").alias("difference_low"),
        pl.col("regional_difference_capacity_high_mw").alias("difference_high"),
        pl.col("regional_levels_capacity_mw").alias("calendar_baseline_capacity"),
        "envelope_cosine_mw",
        "envelope_cams_mw",
    )
    lines += [
        (
            "## Stage 3: the aggregate BMUs, GSP-group regional irradiance (MW; the interval "
            "resamples whole calendar months)"
        ),
        "",
        markdown_table(frame=shown, decimals=1),
        "",
    ]
    controls_increment = controls.filter(
        pl.col("group").is_in(["negative control", "calendar replica"])
    )["cloud_increment"]
    bar = float(controls_increment.max())  # ty: ignore[invalid-argument-type]
    bar_without_nuclear = float(
        controls.filter(
            pl.col("group").is_in(["negative control", "calendar replica"])
            & (pl.col("fuel") != "NUCLEAR")
        )["cloud_increment"].max()  # ty: ignore[invalid-argument-type]
    )
    aggregate_increments = controls.filter(pl.col("group") == "aggregate")["cloud_increment"]
    solar_increments = controls.filter(pl.col("group") == "positive control")["cloud_increment"]
    lines += [
        "### Cloud increment against the controls",
        "",
        markdown_table(
            frame=pl.DataFrame(
                [
                    {
                        "largest increment of any non-solar control or calendar replica": bar,
                        "largest increment excluding nuclear controls": bar_without_nuclear,
                        "aggregate BMUs above that (GB-mean sky)": int(
                            (aggregate_increments > bar_without_nuclear).sum()
                        ),
                        "single-site solar BMUs above that": int(
                            (solar_increments > bar_without_nuclear).sum()
                        ),
                        "non-solar controls": controls.filter(
                            pl.col("group") == "negative control"
                        ).height,
                        "calendar replicas": controls.filter(
                            pl.col("group") == "calendar replica"
                        ).height,
                        "aggregate BMUs above it (GB-mean sky)": int(
                            (aggregate_increments > bar).sum()
                        ),
                        "aggregate BMUs above it (regional sky, stage 3)": int(
                            (table["physical_cloud_increment"] > bar).sum()
                        ),
                        "single-site solar BMUs above it": int((solar_increments > bar).sum()),
                    }
                ]
            ),
            decimals=4,
        ),
        "",
    ]
    detected = table.filter(pl.col("physical_variance_explained") > 0.5)
    total_rows = []
    for label, frame in (
        ("all 25 aggregate BMUs with output", table),
        ("those with variance explained above 0.5", detected),
    ):
        total_rows.append(
            {
                "group": label,
                "BMUs": frame.height,
                "Generation Capacity (MW)": float(frame["generation_capacity_mw"].sum()),
                "envelope, cosine (MW)": float(frame["envelope_cosine_mw"].sum()),
                "envelope, all-GB CAMS mean (MW)": float(frame["envelope_cams_mw"].sum()),
                "physical fit AC capacity (MW)": float(frame["physical_ac_mw"].sum()),
                "difference separation (MW)": float(frame["regional_difference_capacity_mw"].sum()),
                "physical fit solar energy (GWh)": float(frame["physical_solar_energy_mwh"].sum())
                / 1000.0,
            }
        )
    lines += [
        "### Totals",
        "",
        markdown_table(frame=pl.DataFrame(total_rows), decimals=1),
        "",
    ]
    return "\n".join(lines)


def wind_clearness_section() -> str:
    """Return the correlation of daily wind speed with the daily clearness index, as markdown."""
    return (
        "## Wind and cloud\n\n"
        "Pearson correlation of the daily mean ERA5 wind speed at 100 m with the daily clearness "
        f"index, over the 18 grid points: {wind_clearness_correlation():.3f}"
    )


def placebo_section() -> str:
    """Return the stage 3c table: the cloud increment against a placebo sky, and by season."""
    placebo = pl.read_parquet(OUTPUT_DIR / "stage3c_placebo.parquet")
    return "\n\n".join(
        [
            "## Stage 3c: placebo sky and season split of the cloud increment",
            markdown_table(frame=placebo, decimals=3),
        ]
    )


def main() -> None:
    """Write `report.md` and print it."""
    sections = [
        decision_rule_tables(),
        stage1_capacity()[0],
        stage1_heldout(),
        stage1_recovery(),
        stage2_tables(),
        ridge_tables(),
        physical_tables(),
        wind_clearness_section(),
        real_tables(),
        placebo_section(),
    ]
    text = "# Solar disaggregation: report\n\n" + "\n\n".join(sections) + "\n"
    (OUTPUT_DIR / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
