"""Build the season-year capacity table of embedded solar behind each Primary, with no refit.

    POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        uv run python studies/embedded_pv_capacity/build_capacity_table.py

Reads experiment 00040's stored arm vectors, rebuilds the cross-arm capacity `C`, its
block-bootstrap interval, the misfit flag, and the bound, checks them against 00040's stored
scores, fits the growth rate per Primary, and prints every number the page quotes into `report.md`.
"""

import math
from typing import Final

import embedded_pv_inputs as inputs
import numpy as np
import polars as pl
from scipy import stats

from studies import pv_capacity_misfit as misfit


def real_shape_vectors(*, shape: str) -> dict[tuple[str, int], np.ndarray]:
    """Return the arm vector of one confound shape on each real Primary-year's own rows."""
    table = pl.read_parquet(inputs.REAL_VECTORS_PATH).filter(pl.col("kind") == shape)
    return {
        (r["site"], r["year"]): np.array([r[a] for a in misfit.ALL_FITS])
        for r in table.iter_rows(named=True)
    }


EARTH_RADIUS_KM: Final[float] = 6371.0
MIN_DISTINCT_YEARS_PER_RESAMPLE: Final[int] = 3
DAILY_OVER_C_REFERENCE: Final[float] = 0.92


def status_of(
    *, site: str, flagged: bool, flagged_primary: bool, bound_amplitude: float | None
) -> str:
    """Return the table status of a Primary-year.

    A Primary is a flagged Primary when the misfit test flags more than half of its season-years.
    Every season-year of a flagged Primary is `flagged`, including one the test happened not to
    flag, because the unflagged year is then more likely a missed detection than a clean year.
    """
    if site in inputs.LEFT_OUT_PRIMARIES:
        return "left-out"
    if flagged or flagged_primary:
        return "flagged"
    if bound_amplitude is not None and bound_amplitude > inputs.CLEAN_BOUND_AMPLITUDE:
        return "low-power"
    return "clean"


def days_per_season_year() -> pl.DataFrame:
    """Count the days with at least ten daylight half-hours of metered power per Primary-year."""
    return (
        pl.scan_parquet(inputs.FRAME_PATH)
        .filter(
            pl.col("site").is_in(list(inputs.PRIMARY_NAMES))
            & (pl.col("zenith") < inputs.MAX_ZENITH_DEG)
            & pl.col("power").is_not_null()
        )
        .with_columns(local=pl.col("time").dt.convert_time_zone("Europe/London"))
        .with_columns(
            date=pl.col("local").dt.date(),
            year=misfit.season_year_of(timestamps=pl.col("time").dt.offset_by("-15m")),
        )
        .group_by("site", "year", "date")
        .agg(n=pl.len())
        .filter(pl.col("n") >= inputs.MIN_DAYLIGHT_ROWS_PER_DAY)
        .group_by("site", "year")
        .agg(n_days=pl.len())
        .collect()
    )


def build_rows(
    *, detector: misfit.Detector, scores: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the table (one row per Primary-year) and the per-draw capacities."""
    day_leakage = real_shape_vectors(shape="dayL")
    boot = pl.read_parquet(inputs.BOOTSTRAP_PATH)
    stored = {(r["site"], r["year"]): r for r in scores.iter_rows(named=True)}
    flag_share = (
        scores.group_by("site").agg(share=pl.col("flagged").mean()).to_dict(as_series=False)
    )
    flagged_primaries = {
        site
        for site, share in zip(flag_share["site"], flag_share["share"], strict=True)
        if share > inputs.FLAGGED_PRIMARY_SHARE
    }
    rows, draw_rows = [], []
    for (site, year), vector in sorted(detector.real.items()):
        ratios = detector.ratios(site=site, excluded=frozenset())
        assert ratios is not None
        capacity = detector.capacity(site=site, year=year, excluded=frozenset())
        t_real = float(misfit.capacity_and_misfit(vectors=vector[None, :], ratios=ratios)[1][0])
        tau = detector.tau(site=site, year=year, excluded=frozenset())
        draws = (
            boot.filter((pl.col("site") == site) & (pl.col("year") == year))
            .sort("draw")
            .select(misfit.ALL_FITS)
            .to_numpy()
        )
        draw_c, draw_daily = misfit.block_bootstrap_capacities(draws=draws, ratios=ratios)
        bound = misfit.shape_bound(
            detector=detector, site=site, year=year, leakage=day_leakage[site, year]
        )
        power_at_quarter = bound.power_by_amplitude[misfit.POWER_AMPLITUDE_INDEX]
        flagged = bool(t_real > tau)
        daily = float(vector[misfit.DAILY_FIT_INDEX] / ratios[misfit.DAILY_FIT_INDEX])
        old = stored[site, year]
        to_c_units = capacity / old["headline"]
        rows.append(
            {
                "primary": inputs.PRIMARY_NAMES[site],
                "site": site,
                "season_year": year,
                "capacity_mw": capacity,
                "capacity_low_mw": float(np.quantile(draw_c, inputs.INTERVAL_QUANTILES[0])),
                "capacity_high_mw": float(np.quantile(draw_c, inputs.INTERVAL_QUANTILES[1])),
                "capacity_daily_fit_mw": daily,
                "daily_over_six_arm": daily / capacity,
                "misfit_T": t_real,
                "misfit_threshold": tau,
                "flagged": flagged,
                "bound_dayL_relative": bound.bound,
                "bound_amplitude": bound.bound_amplitude,
                "power_dayL_025": power_at_quarter,
                "stored_bound_any_shape": old["B"],
                "status": status_of(
                    site=site,
                    flagged=flagged,
                    flagged_primary=site in flagged_primaries,
                    bound_amplitude=bound.bound_amplitude,
                ),
                "library_low_mw": old["range_low"] * to_c_units
                if site in flagged_primaries
                else None,
                "library_high_mw": old["range_high"] * to_c_units
                if site in flagged_primaries
                else None,
                "check_C_diff": abs(capacity - old["C"]),
                "check_tau_diff": abs(tau - old["tau"]),
                "check_T_diff": abs(t_real - old["T"]),
                "check_flag_equal": flagged == old["flagged"],
                "check_B_dayL_diff": abs(bound.bound - old["B_dayL"]),
                "check_power_dayL_diff": abs(
                    bound.power_by_amplitude[misfit.POWER_AMPLITUDE_INDEX] - old["power_dayL"]
                ),
            }
        )
        draw_rows.extend(
            {"site": site, "season_year": year, "draw": i, "capacity_mw": c, "daily_fit_mw": d}
            for i, (c, d) in enumerate(zip(draw_c, draw_daily, strict=True))
        )
    table = pl.DataFrame(rows).join(
        days_per_season_year().rename({"year": "season_year"}),
        on=["site", "season_year"],
        how="left",
    )
    return table, pl.DataFrame(draw_rows)


def linear_slope_per_year(*, years: np.ndarray, capacity: np.ndarray) -> float:
    """Return the fitted growth in MW a year of a straight line through the capacities."""
    return float(np.polyfit(years, capacity, 1)[0])


def growth_ols_interval(*, years: np.ndarray, capacity: np.ndarray) -> tuple[float, float]:
    """Return the 95% t interval of the annual growth rate from the log-linear least-squares fit."""
    fit = stats.linregress(years, np.log(capacity))
    half = stats.t.ppf(0.975, years.size - 2) * fit.stderr
    return float(np.expm1(fit.slope - half)), float(np.expm1(fit.slope + half))


def growth_per_year(*, years: np.ndarray, capacity: np.ndarray) -> float:
    """Return the fitted annual growth rate of a log-linear trend, as a fraction per year."""
    slope = np.polyfit(years, np.log(capacity), 1)[0]
    return float(np.expm1(slope))


def trend_rows(*, table: pl.DataFrame, draws: pl.DataFrame) -> pl.DataFrame:
    """Fit the growth rate per Primary with at least six clean or low-power season-years.

    For each Primary, `growth_*` is the fitted rate on the point capacity, `season_*` is the 95%
    interval of a bootstrap over season-years, and `sampling_*` is the 95% interval of the rate
    fitted to each bootstrap draw's capacities taken across the season-years in turn. Draw `d` of
    each season-year is an independent resample, so the second interval holds the sampling noise
    alone. The same three numbers are fitted to the daily fit.
    """
    rng = np.random.default_rng(inputs.TREND_BOOTSTRAP_SEED)
    out = []
    usable = table.filter(pl.col("status").is_in(["clean", "low-power"]))
    for site, unsorted in usable.group_by("site", maintain_order=True):
        group = unsorted.sort("season_year")
        if group.height < inputs.MIN_YEARS_FOR_TREND:
            continue
        years = group["season_year"].to_numpy().astype(float)
        picks = [
            pick
            for pick in (
                rng.integers(0, years.size, years.size) for _ in range(inputs.TREND_BOOTSTRAP_DRAWS)
            )
            if np.unique(pick).size >= MIN_DISTINCT_YEARS_PER_RESAMPLE
        ]
        for estimator, column, draw_column in (
            ("six-arm C", "capacity_mw", "capacity_mw"),
            ("daily fit", "capacity_daily_fit_mw", "daily_fit_mw"),
        ):
            values = group[column].to_numpy()
            point = growth_per_year(years=years, capacity=values)
            resampled = [
                growth_per_year(years=years[pick], capacity=values[pick]) for pick in picks
            ]
            per_draw = np.column_stack(
                [
                    draws.filter((pl.col("site") == site[0]) & (pl.col("season_year") == int(y)))
                    .sort("draw")[draw_column]
                    .to_numpy()
                    for y in years
                ]
            )
            sampling = [growth_per_year(years=years, capacity=row) for row in per_draw]
            out.append(
                {
                    "primary": inputs.PRIMARY_NAMES[site[0]],
                    "site": site[0],
                    "estimator": estimator,
                    "n_season_years": int(years.size),
                    "first_season_year": int(years[0]),
                    "growth_per_year": point,
                    "mw_per_year": linear_slope_per_year(years=years, capacity=values),
                    "growth_if_fixed_offset": growth_per_year(
                        years=years, capacity=values - inputs.OFFSET_SHARE * values.mean()
                    ),
                    "ols_low": growth_ols_interval(years=years, capacity=values)[0],
                    "ols_high": growth_ols_interval(years=years, capacity=values)[1],
                    "growth_without_last_year": growth_per_year(
                        years=years[:-1], capacity=values[:-1]
                    ),
                    "n_resamples_used": len(picks),
                    "season_low": float(np.quantile(resampled, inputs.INTERVAL_QUANTILES[0])),
                    "season_high": float(np.quantile(resampled, inputs.INTERVAL_QUANTILES[1])),
                    "sampling_low": float(np.quantile(sampling, inputs.INTERVAL_QUANTILES[0])),
                    "sampling_high": float(np.quantile(sampling, inputs.INTERVAL_QUANTILES[1])),
                }
            )
    return pl.DataFrame(out)


def bias_rows(*, detector: misfit.Detector) -> pl.DataFrame:
    """Return the noise-free bias of `C` and of the daily fit for each undetected shape.

    For every unit of a confound shape, the truth vector is the Primary's recovery ratios and the
    load is scaled so that its M1 leakage is `amplitude` times the truth.
    """
    out = []
    for unit in detector.units:
        ratios = detector.ratios(site=unit.site, excluded=frozenset())
        assert ratios is not None
        for shape in inputs.BIAS_SHAPES:
            for amplitude in inputs.BIAS_AMPLITUDES:
                cross_arm, daily = misfit.undetected_bias(
                    ratios=ratios, pv=ratios, leakage=unit.shapes[shape], amplitude=amplitude
                )
                out.append(
                    {
                        "site": unit.site,
                        "farm": unit.farm,
                        "season_year": unit.year,
                        "shape": shape,
                        "amplitude": amplitude,
                        "bias_cross_arm": cross_arm,
                        "bias_daily_fit": daily,
                    }
                )
    return pl.DataFrame(out)


def calibration_report() -> list[str]:
    """Return report lines on how the capacity estimate behaves on injected solar-farm output.

    The rows are experiment 00040's `pure` scenarios: one farm's output added to a Primary's
    demand, scored with that farm left out of the recovery ratios.
    """
    pure = pl.read_parquet(inputs.UNIT_SCORES_PATH).filter(pl.col("shape") == "pure")
    ratio = pure["median_ratio"]
    per_farm = (
        pure.group_by("farm")
        .agg(
            median_ratio=pl.col("median_ratio").median(),
            false_positive_rate=pl.col("power").mean(),
            n_units=pl.len(),
        )
        .sort("farm")
    )
    detector = misfit.load_detector(
        unit_vectors_path=inputs.UNIT_VECTORS_PATH,
        real_vectors_path=inputs.REAL_VECTORS_PATH,
        bootstrap_path=inputs.BOOTSTRAP_PATH,
    )
    baseline = []
    for unit in detector.units:
        ratios = detector.ratios(site=unit.site, excluded=frozenset({unit.farm}))
        assert ratios is not None
        capacity = float(misfit.capacity_and_misfit(vectors=unit.pv[None, :], ratios=ratios)[0][0])
        daily = float(unit.pv[misfit.DAILY_FIT_INDEX] / ratios[misfit.DAILY_FIT_INDEX])
        baseline.append({"farm": unit.farm, "daily_over_c": daily / capacity})
    baseline_frame = pl.DataFrame(baseline)
    per_farm_daily = (
        baseline_frame.group_by("farm")
        .agg(median_daily_over_c=pl.col("daily_over_c").median())
        .sort("farm")
    )
    daily_over_c = baseline_frame["daily_over_c"]
    with pl.Config(tbl_rows=20, float_precision=3):
        per_farm_text = str(per_farm.join(per_farm_daily, on="farm"))
    return [
        "## Transfer from solar farms: C over truth on injected farm output (00040's `pure` rows)",
        "",
        (
            f"All {pure.height} units: median {ratio.median():.3f}, 10th to 90th percentile "
            f"{ratio.quantile(0.1):.3f} to {ratio.quantile(0.9):.3f}; false-positive rate "
            f"{pure['power'].mean():.3f}."
        ),
        "",
        "```",
        per_farm_text,
        "```",
        "",
        "## Daily fit over C on injected farm output, noise-free (leave-own-farm-out ratios)",
        "",
        (
            f"All {daily_over_c.len()} units: median {daily_over_c.median():.3f}, 10th to 90th "
            f"percentile {daily_over_c.quantile(0.1):.3f} to "
            f"{daily_over_c.quantile(0.9):.3f}; share below {DAILY_OVER_C_REFERENCE}: "
            f"{(daily_over_c < DAILY_OVER_C_REFERENCE).sum() / daily_over_c.len():.3f}."
        ),
        "",
    ]


def report(
    *, table: pl.DataFrame, trend: pl.DataFrame, bias: pl.DataFrame, draws: pl.DataFrame
) -> str:
    """Return the report text: every number the page quotes about the table."""
    lines = ["# Capacity table report", ""]
    checks = table.select(
        pl.col("check_C_diff").max().alias("C"),
        pl.col("check_tau_diff").max().alias("tau"),
        pl.col("check_T_diff").max().alias("T"),
        pl.col("check_B_dayL_diff").max().alias("B_dayL"),
        pl.col("check_power_dayL_diff").max().alias("power_dayL"),
        pl.col("check_flag_equal").all().alias("flags_equal"),
    )
    lines += [
        "## P1: reproduction of experiment 00040's stored scores (largest absolute difference)",
        "",
        "```",
        str(checks),
        "```",
        "",
    ]
    same = table.filter(
        (pl.col("stored_bound_any_shape") - pl.col("bound_dayL_relative")).abs()
        > inputs.REPRODUCTION_TOLERANCE
    ).height
    lines += [
        (
            f"Primary-years where 00040's all-shape bound differs from the day-long bound: {same} "
            f"of {table.height}."
        ),
        "",
    ]
    cols = [
        "primary",
        "season_year",
        "n_days",
        "status",
        "capacity_mw",
        "capacity_low_mw",
        "capacity_high_mw",
        "capacity_daily_fit_mw",
        "daily_over_six_arm",
        "misfit_T",
        "misfit_threshold",
        "bound_dayL_relative",
        "bound_amplitude",
        "power_dayL_025",
        "library_low_mw",
        "library_high_mw",
    ]
    with pl.Config(tbl_rows=60, tbl_cols=30, tbl_width_chars=250, float_precision=3):
        lines += [
            "## Capacity table",
            "",
            "```",
            str(table.select(cols).sort("primary", "season_year")),
            "```",
            "",
        ]
    status = table.group_by("status").agg(n=pl.len()).sort("status")
    lines += ["## Status counts", "", "```", str(status), "```", ""]
    latest = table.sort("season_year").group_by("site").last().sort("site")
    with pl.Config(tbl_rows=20, tbl_cols=30, tbl_width_chars=250, float_precision=3):
        lines += ["## 2025/26", "", "```", str(latest.select(cols)), "```", ""]
    interval = table.filter(pl.col("status").is_in(["clean", "low-power"])).select(
        (
            (pl.col("capacity_high_mw") - pl.col("capacity_low_mw")) / 2 / pl.col("capacity_mw")
        ).alias("half_width")
    )["half_width"]
    lines += [
        "## Sampling interval half-width over capacity, clean and low-power rows",
        "",
        (
            f"Median {interval.median():.3f}, 10th percentile {interval.quantile(0.1):.3f}, "
            f"90th percentile {interval.quantile(0.9):.3f}, n = {interval.len()}."
        ),
        "",
    ]
    lower = (pl.col("capacity_mw") - pl.col("capacity_low_mw")) / pl.col("capacity_mw")
    upper = (pl.col("capacity_high_mw") - pl.col("capacity_mw")) / pl.col("capacity_mw")
    widths = (
        table.filter(pl.col("season_year") == inputs.LAST_SEASON_YEAR)
        .filter(pl.col("status").is_in(["clean", "low-power"]))
        .select("primary", below_share=lower, above_share=upper)
        .sort("primary")
    )
    with pl.Config(tbl_rows=20, float_precision=3):
        lines += [
            "## 2025/26 sampling interval, share of C below and above the capacity",
            "",
            "```",
            str(widths),
            "```",
            "",
        ]
    unflagged = table.filter(pl.col("status").is_in(["clean", "low-power"]))
    ratio = unflagged["daily_over_six_arm"]
    kirton = table.filter(pl.col("site") == inputs.KIRTON)["daily_over_six_arm"]
    lines += [
        "## Daily fit over six-arm C",
        "",
        (
            f"Clean and low-power Primary-years (so not Kirton or Wrangle T1): n = {ratio.len()}, "
            f"median "
            f"{ratio.median():.3f}, 10th to 90th percentile {ratio.quantile(0.1):.3f} to "
            f"{ratio.quantile(0.9):.3f}, above 1 in {int((ratio > 1).sum())}."
        ),
        (
            f"Kirton, all {kirton.len()} years: median {kirton.median():.3f}, range "
            f"{kirton.min():.3f} to {kirton.max():.3f}, above 1 in {int((kirton > 1).sum())}."
        ),
        "",
    ]
    with pl.Config(tbl_rows=20, tbl_cols=30, tbl_width_chars=250, float_precision=3):
        lines += ["## Growth rate per year", "", "```", str(trend), "```", ""]
    summary = (
        bias.group_by("shape", "amplitude")
        .agg(
            pl.col("bias_cross_arm").median().alias("bias_C"),
            pl.col("bias_daily_fit").median().alias("bias_daily_fit"),
            pl.len().alias("n_units"),
        )
        .with_columns(
            daily_over_c_with_load=(1 + pl.col("bias_daily_fit")) / (1 + pl.col("bias_C"))
        )
        .sort("shape", "amplitude")
    )
    with pl.Config(tbl_rows=20, float_precision=3):
        lines += [
            "## Noise-free bias of C and of the daily fit, median over units",
            "",
            "```",
            str(summary),
            "```",
            "",
        ]
    lines += calibration_report()
    lines += [f"Bootstrap draws per Primary-year: {draws['draw'].n_unique()}.", ""]
    return "\n".join(lines)


def distance_report() -> list[str]:
    """Return report lines on how far apart the eight Primary substations are."""
    sites = (
        pl.read_parquet(inputs.SITES_PATH)
        .filter(pl.col("site").is_in(list(inputs.PRIMARY_NAMES)))
        .select("site", "latitude", "longitude")
        .to_dicts()
    )
    distances = []
    for i, a in enumerate(sites):
        for b in sites[i + 1 :]:
            lat1, lat2 = math.radians(a["latitude"]), math.radians(b["latitude"])
            dlon = math.radians(b["longitude"] - a["longitude"])
            cosine = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(
                dlon
            )
            distances.append(EARTH_RADIUS_KM * math.acos(min(1.0, cosine)))
    return [
        "## Distance between the eight Primary substations",
        "",
        f"Largest pairwise distance {max(distances):.1f} km over {len(distances)} pairs.",
        "",
    ]


def markdown_table(*, table: pl.DataFrame) -> str:
    """Return the 2025/26 rows of the table as a Markdown table, for the page."""
    latest = (
        table.filter(pl.col("season_year") == inputs.LAST_SEASON_YEAR)
        .filter(pl.col("status") != "left-out")
        .sort("capacity_mw", descending=True, nulls_last=True)
    )
    lines = [
        (
            "| Primary substation | Status | Capacity C (MW) | 95% interval (MW) "
            "| Bound B (strength) | Daily fit divided by C |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for r in latest.iter_rows(named=True):
        if r["status"] == "flagged":
            lines.append(
                f"| {r['primary']} | flagged | none | range {r['library_low_mw']:.1f} to "
                f"{r['library_high_mw']:.1f} | not given | {r['daily_over_six_arm']:.2f} |"
            )
        else:
            lines.append(
                f"| {r['primary']} | {r['status']} | {r['capacity_mw']:.2f} | "
                f"{r['capacity_low_mw']:.2f} to {r['capacity_high_mw']:.2f} | "
                f"{r['bound_dayL_relative']:.2f} ({r['bound_amplitude']:g}) | "
                f"{r['daily_over_six_arm']:.2f} |"
            )
    return "\n".join(lines)


def refresh_page(*, table: pl.DataFrame) -> None:
    """Rewrite the generated table between the markers of the page, if the page exists."""
    if not inputs.PAGE_PATH.exists():
        return
    text = inputs.PAGE_PATH.read_text()
    start, end = text.index(inputs.TABLE_START_MARKER), text.index(inputs.TABLE_END_MARKER)
    block = f"{inputs.TABLE_START_MARKER}\n\n{markdown_table(table=table)}\n\n"
    inputs.PAGE_PATH.write_text(text[:start] + block + text[end:])


def main() -> None:
    """Build and write the table, the draws, the trend, the bias table, and the report."""
    inputs.STUDY_DIR.mkdir(parents=True, exist_ok=True)
    detector = misfit.load_detector(
        unit_vectors_path=inputs.UNIT_VECTORS_PATH,
        real_vectors_path=inputs.REAL_VECTORS_PATH,
        bootstrap_path=inputs.BOOTSTRAP_PATH,
    )
    scores = pl.read_parquet(inputs.REAL_SCORES_PATH)
    table, draws = build_rows(detector=detector, scores=scores)
    trend = trend_rows(table=table, draws=draws)
    bias = bias_rows(detector=detector)
    table.write_parquet(inputs.TABLE_PATH)
    draws.write_parquet(inputs.DRAWS_PATH)
    trend.write_parquet(inputs.TREND_PATH)
    bias.write_parquet(inputs.BIAS_PATH)
    text = report(table=table, trend=trend, bias=bias, draws=draws)
    text += "\n".join(distance_report())
    inputs.REPORT_PATH.write_text(text)
    refresh_page(table=table)
    inputs.CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.drop([c for c in table.columns if c.startswith("check_")]).write_csv(inputs.CSV_PATH)
    print(text)


if __name__ == "__main__":
    main()
