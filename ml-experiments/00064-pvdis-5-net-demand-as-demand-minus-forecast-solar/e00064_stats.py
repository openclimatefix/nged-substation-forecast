"""Score the arms of experiment 00064, read the pre-registered contrasts, and print every number.

Reads `work/arms/<arm>.parquet` (see `e00064_assemble.py`) and writes under `work/results/`:

- `absolute.parquet`: each arm's mean quantile score and error of the median, with 95% intervals,
  and the stored reference arms `climatology` and `manual_heuristic`.
- `verdict_family.parquet`: contrasts N (arm minus `d0`) and T (arm minus its placebo twin) of the
  four arms `sf`, `a`, `am`, and `b`, and the gain family (N over leads under 48 hours), all at
  98.75% (Bonferroni across the four arms) from 50,000 resamples of whole ISO weeks of forecast
  start.
- `exploratory.parquet`: every other contrast and split, at 95%.
- `effective_weights.parquet`, `coverage.parquet`, `by_month.parquet`, `temperature.parquet`.

Scored rows are the daylight half-hours of April to September with a CAMS `u` and an observed power,
on the five verdict series. The score is in points of capacity (1 point is 1% of the series'
effective capacity). The decision rule is `verdict_rule()`:

- An arm is non-inferior when the upper end of its N interval is below +0.10 points.
- An arm carries solar information when the upper end of its T interval is below zero.
- `supported` when at least one arm is both; `refuted` when no arm is non-inferior and every
  arm's N interval lies wholly above zero; `inconclusive` otherwise.

    uv run python ml-experiments/00064-*/e00064_stats.py
"""

from typing import Final

import e00064_common as common
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

NON_INFERIORITY_MARGIN: Final[float] = 0.10
"""Points of capacity: the upper end of N must be below this for an arm to be non-inferior."""
GAIN_MARGIN: Final[float] = -0.10
"""Points of capacity: a gain is reported when the upper end of the gain read is below this."""
VERDICT_ARMS: Final[tuple[str, ...]] = ("sf", "a", "am", "b")
TWIN_OF: Final[dict[str, str]] = {arm: f"{arm}_twin" for arm in VERDICT_ARMS}
VERDICT_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / len(VERDICT_ARMS))
"""98.75: a 5% familywise level split over the four arms."""
EXPLORATORY_LEVEL: Final[float] = 95.0
VERDICT_RESAMPLES: Final[int] = 50_000
EXPLORATORY_RESAMPLES: Final[int] = 10_000
REFERENCE_ARM: Final[str] = "d0"
BASE_ARM: Final[str] = "d1l"
SCORED_ARMS: Final[tuple[str, ...]] = (
    "d0",
    "d1l",
    "d1l_ens_shift",
    "sf",
    "sf_member",
    "sf_twin",
    "sf_temp",
    "a",
    "a_twin",
    "am",
    "am_twin",
    "b",
    "b_member",
    "b_twin",
)
STORED_REFERENCE_ARMS: Final[tuple[str, ...]] = ("climatology", "manual_heuristic")
COVERAGE_ARMS: Final[tuple[str, ...]] = ("d0", "d1l", "sf", "a", "am", "b")
GAIN_BANDS: Final[tuple[str, ...]] = ("0-24 h", "1-2 d")
"""The lead bands that make up leads under 48 hours."""
BAND_NAMES: Final[tuple[str, ...]] = tuple(name for name, _low, _high in common.LEAD_BANDS)
CSI_BINS: Final[tuple[str, ...]] = ("cloudy", "middle", "sunny")
SUNNIEST_QUANTILE: Final[float] = 0.9
VERDICT_SCOPE: Final[pl.Expr] = pl.lit(value=True)
"""Every row of the context is a scored row of a verdict series."""


def temperature_table() -> pl.DataFrame:
    """Return the member-mean 2 m temperature of every validation run, series, and valid time."""
    ids = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))["time_series_id"]
    return pl.concat(
        [
            pl.read_parquet(path, columns=[*common.KEYS, "temperature_2m"])
            .filter(pl.col("time_series_id").is_in(ids.implode()))
            .group_by(*common.KEYS)
            .agg(temperature_mean=pl.col("temperature_2m").mean())
            for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
        ]
    )


def context_table() -> pl.DataFrame:
    """Return the scored rows of the verdict series with the facts the splits and reads use.

    Returns:
        One row per run, valid time, and series with `KEYS`, `series_label`, `band` (name of the
        lead band), `valid_month` (local calendar month), `lead_hours`, `clear_sky_index`,
        `csi_bin`, `sunniest` (the pooled top decile of the clear-sky index), `applied_shift` (the
        capacity over the effective capacity times the ensemble-mean calibrated anomaly), and
        `cell_temperature` (member-mean air temperature plus 25 K times the calibrated shape).
    """
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    arrays = common.load_series_arrays(series=series)
    rows = common.scored_rows(series=series, arrays=arrays).filter(pl.col("scored"))
    capacity = common.capacities().join(
        series.select("site", "effective_capacity_mw"), on="site", validate="1:1"
    )
    climatology = pl.read_parquet(common.E46_SOLAR_DIR / "climatology.parquet").select(
        "site", "month", "hh", "clim_cams"
    )
    anomaly = pl.read_parquet(common.INPUTS_DIR / "val_keys.parquet").select(
        *common.KEYS, "anomaly_mean_calibrated"
    )
    threshold = rows["clear_sky_index"].quantile(SUNNIEST_QUANTILE)
    return (
        rows.join(capacity, on="site", validate="m:1")
        .join(climatology, on=["site", "month", "hh"], how="left", validate="m:1")
        .join(anomaly, on=list(common.KEYS), how="left", validate="m:1")
        .join(temperature_table(), on=list(common.KEYS), how="left", validate="m:1")
        .with_columns(
            applied_shift=pl.col("capacity_mw")
            * pl.col("anomaly_mean_calibrated").fill_null(0.0)
            / pl.col("effective_capacity_mw"),
            cell_temperature=pl.col("temperature_mean")
            + common.CELL_TEMPERATURE_RISE_K
            * (
                pl.col("clim_cams").fill_null(0.0)
                + pl.col("anomaly_mean_calibrated").fill_null(0.0)
            ),
        )
        .with_columns(
            series_label=pl.col("site"),
            band_name=pl.col("band").map_elements(BAND_NAMES.__getitem__, return_dtype=pl.String),
            csi_bin=pl.when(pl.col("clear_sky_index") >= common.SUNNY_INDEX)
            .then(pl.lit("sunny"))
            .when(pl.col("clear_sky_index") < common.CLOUDY_INDEX)
            .then(pl.lit("cloudy"))
            .when(pl.col("clear_sky_index").is_not_null())
            .then(pl.lit("middle")),
            sunniest=pl.col("clear_sky_index") >= threshold,
        )
        .select(
            *common.KEYS,
            "series_label",
            band=pl.col("band_name"),
            valid_month=pl.col("month"),
            lead_hours=pl.col("lead_hours"),
            clear_sky_index=pl.col("clear_sky_index"),
            csi_bin=pl.col("csi_bin"),
            sunniest=pl.col("sunniest").fill_null(value=False),
            applied_shift=pl.col("applied_shift"),
            cell_temperature=pl.col("cell_temperature"),
        )
    )


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `qs` and `mae` in points of capacity to a frame of quantiles and `actual`."""
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    return frame.with_columns(
        qs=pl.Series(common.pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0),
        mae=pl.Series(np.abs(quantiles[:, common.MEDIAN] - actual) * 100.0),
    )


def load_losses(*, context: pl.DataFrame) -> pl.DataFrame:
    """Return every scored arm's per-row losses on the context rows, resampled by ISO week.

    Args:
        context: `context_table()`.

    Returns:
        The frame `experiment_store.losses_for_bootstrap` builds, with the context columns, `qs`,
        `mae`, the nine quantiles, and `actual`; its `month` column holds the ISO week of the
        forecast start (`2025-W27`), so every bootstrap resamples whole weeks.
    """
    arm_losses = {
        arm: with_losses(
            frame=pl.read_parquet(common.ARMS_DIR / f"{arm}.parquet").join(
                context, on=list(common.KEYS), how="inner", validate="m:1"
            )
        )
        for arm in SCORED_ARMS
    }
    losses = experiment_store.losses_for_bootstrap(arm_losses=arm_losses)
    return losses.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%G-W%V"))


def read_contrast(
    *,
    losses: pl.DataFrame,
    contrast_id: str,
    treatment: str,
    reference: str,
    scope_name: str,
    scope: pl.Expr,
    level: float,
    n_resamples: int,
) -> dict[str, str | float | int]:
    """Return one contrast's mean paired difference and its week-block interval.

    Args:
        losses: `load_losses()`.
        contrast_id: The contrast's label.
        treatment: The arm whose score is compared.
        reference: The arm it is compared against.
        scope_name: What the scope is, for the printed table.
        scope: A Polars expression selecting the rows.
        level: The interval's coverage in percent.
        n_resamples: How many resamples.

    Returns:
        A row with the difference, bounds, half-width, rows per seed, blocks, and seed spread.
    """
    scoped = losses.filter(scope)
    differences, weeks = bootstrap.paired_differences(
        losses=scoped, treatment=treatment, reference=reference, metric="qs"
    )
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=scoped,
        treatment=treatment,
        reference=reference,
        metric="qs",
        level=level,
        n_resamples=n_resamples,
    )
    return {
        "contrast": contrast_id,
        "treatment": treatment,
        "reference": reference,
        "scope": scope_name,
        "level": level,
        "n_resamples": n_resamples,
        "difference": float(differences.mean()),
        "lower": lower,
        "upper": upper,
        "half_width": (upper - lower) / 2.0,
        "seed_spread": float(differences.mean(axis=1).std()),
        "rows_per_seed": differences.shape[1],
        "weeks": len(np.unique(weeks)),
    }


def verdict_family(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Read contrasts N and T of every verdict arm and the gain family, at 98.75%.

    Args:
        losses: `load_losses()`.

    Returns:
        One row per read, labelled `N`, `T`, or `gain` in `contrast`.
    """
    gain_scope = pl.col("band").is_in(list(GAIN_BANDS))
    rows = []
    for arm in VERDICT_ARMS:
        for contrast_id, reference, scope_name, scope in (
            ("N", REFERENCE_ARM, "all leads", VERDICT_SCOPE),
            ("T", TWIN_OF[arm], "all leads", VERDICT_SCOPE),
            ("gain", REFERENCE_ARM, "leads under 48 h", gain_scope),
        ):
            rows.append(
                read_contrast(
                    losses=losses,
                    contrast_id=contrast_id,
                    treatment=arm,
                    reference=reference,
                    scope_name=scope_name,
                    scope=scope,
                    level=VERDICT_LEVEL,
                    n_resamples=VERDICT_RESAMPLES,
                )
            )
    return pl.DataFrame(rows)


def verdict_rule(*, family: pl.DataFrame) -> tuple[str, dict[str, str]]:
    """Apply the pre-registered rule, returning exactly one verdict and each arm's reading.

    The mapping is: `supported` if at least one arm has N upper below +0.10 and T upper below 0;
    otherwise `refuted` if no arm has N upper below +0.10 and every arm has N lower above 0;
    otherwise `inconclusive`.

    Args:
        family: `verdict_family()`.

    Returns:
        The verdict, and for each arm a sentence naming which reads it passed.
    """
    readings: dict[str, str] = {}
    passes = []
    non_inferior = []
    wholly_above = []
    for arm in VERDICT_ARMS:
        n = family.filter((pl.col("treatment") == arm) & (pl.col("contrast") == "N")).row(
            0, named=True
        )
        t = family.filter((pl.col("treatment") == arm) & (pl.col("contrast") == "T")).row(
            0, named=True
        )
        is_non_inferior = float(n["upper"]) < NON_INFERIORITY_MARGIN
        carries_information = float(t["upper"]) < 0.0
        non_inferior.append(is_non_inferior)
        wholly_above.append(float(n["lower"]) > 0.0)
        passes.append(is_non_inferior and carries_information)
        if is_non_inferior and carries_information:
            readings[arm] = "passes: non-inferior and beats its placebo twin"
        elif is_non_inferior:
            readings[arm] = "matches d0, but the solar term adds nothing (does not beat its twin)"
        elif carries_information:
            readings[arm] = "beats its twin, but is not shown non-inferior to d0"
        else:
            readings[arm] = "neither non-inferior nor better than its twin"
    if any(passes):
        return "supported", readings
    if not any(non_inferior) and all(wholly_above):
        return "refuted", readings
    return "inconclusive", readings


def absolute_scores(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return each arm's mean score and median error on the scored rows, with 95% intervals."""
    rows = []
    for arm in (a for a in SCORED_ARMS if a in set(losses["arm"].unique().to_list())):
        row: dict[str, str | float | int] = {"arm": arm}
        for metric in ("qs", "mae"):
            interval = bootstrap.bootstrap_absolute(losses=losses, arm=arm, metric=metric)
            row |= {
                metric: interval["value"],
                f"{metric}_lower": interval["lower_95"],
                f"{metric}_upper": interval["upper_95"],
            }
        rows.append(row)
    return pl.DataFrame(rows)


def stored_reference_scores(*, context: pl.DataFrame) -> pl.DataFrame:
    """Return the stored `climatology` and `manual_heuristic` arms' mean score on the scored rows.

    These arms come from experiment 00033 and hold a few rows fewer than the other arms, so they
    are scored on the rows they share with the context and not paired in the bootstrap.

    Args:
        context: `context_table()`.

    Returns:
        Columns `arm`, `qs`, `mae`, and `rows` (per seed), and `missing_rows` (context rows the
        arm lacks).
    """
    rows = []
    for arm in STORED_REFERENCE_ARMS:
        frame = with_losses(
            frame=pl.read_parquet(common.ARMS_DIR / f"{arm}.parquet").join(
                context, on=list(common.KEYS), how="inner", validate="m:1"
            )
        )
        per_seed = frame.filter(pl.col("seed") == frame["seed"].min()).height
        rows.append(
            {
                "arm": arm,
                "qs": float(np.mean(frame["qs"].to_numpy())),
                "mae": float(np.mean(frame["mae"].to_numpy())),
                "rows": per_seed,
                "missing_rows": context.height - per_seed,
            }
        )
    return pl.DataFrame(rows)


def exploratory_contrasts(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Read the exploratory contrasts and splits at 95%.

    The reads are: each arm against `d1l`; `a` against `am` overall and in the sunniest decile;
    each member shift against its ensemble-mean shift; the temperature derating; each verdict arm's
    contrast N by lead band, by clear-sky-index bin, in the sunniest decile, and by series; and
    each arm against its twin by lead band.

    Args:
        losses: `load_losses()`.

    Returns:
        One row per read, in the shape of `read_contrast`.
    """
    sunniest = pl.col("sunniest")
    specs: list[tuple[str, str, str, str, pl.Expr]] = []
    specs += [("arm_vs_d1l", arm, BASE_ARM, "all leads", VERDICT_SCOPE) for arm in VERDICT_ARMS]
    specs += [
        ("sf_vs_fixed_member_shift", "sf", "d1l_ens_shift", "all leads", VERDICT_SCOPE),
        ("A_vs_AM", "a", "am", "all leads", VERDICT_SCOPE),
        ("A_vs_AM", "a", "am", "sunniest decile", sunniest),
        ("member_vs_mean", "sf_member", "sf", "all leads", VERDICT_SCOPE),
        ("member_vs_mean", "b_member", "b", "all leads", VERDICT_SCOPE),
        ("temperature_derating", "sf_temp", "sf", "all leads", VERDICT_SCOPE),
    ]
    for arm in VERDICT_ARMS:
        specs.append(("N_sunniest_decile", arm, REFERENCE_ARM, "sunniest decile", sunniest))
        for band in BAND_NAMES:
            band_scope = pl.col("band") == band
            specs.append(("N_by_band", arm, REFERENCE_ARM, f"band {band}", band_scope))
            specs.append(("T_by_band", arm, TWIN_OF[arm], f"band {band}", band_scope))
        for csi_bin in CSI_BINS:
            specs.append(
                ("N_by_csi", arm, REFERENCE_ARM, f"csi {csi_bin}", pl.col("csi_bin") == csi_bin)
            )
        for site in common.SITES:
            specs.append(
                (
                    "N_by_series",
                    arm,
                    REFERENCE_ARM,
                    f"series {site}",
                    pl.col("series_label") == site,
                )
            )
    available = set(losses["arm"].unique().to_list())
    return pl.DataFrame(
        [
            read_contrast(
                losses=losses,
                contrast_id=contrast_id,
                treatment=treatment,
                reference=reference,
                scope_name=scope_name,
                scope=scope,
                level=EXPLORATORY_LEVEL,
                n_resamples=EXPLORATORY_RESAMPLES,
            )
            for contrast_id, treatment, reference, scope_name, scope in specs
            if treatment in available and reference in available
        ]
    )


def effective_weights(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return each arm's effective weight: the slope of its median minus `d1l`'s on `-shift`.

    The regressor is minus the capacity over the effective capacity times the ensemble-mean
    calibrated anomaly, so an arm that subtracts the solar term with weight `w` has an effective
    weight of `w`. The slope is through the origin, pooled over seeds, overall and by lead band.

    Args:
        losses: `load_losses()`.

    Returns:
        Columns `arm`, `scope`, `effective_weight`, and `rows`.
    """
    base = losses.filter(pl.col("arm") == BASE_ARM).select(
        "site", "time", "seed", base_median=pl.col("q50")
    )
    rows = []
    for arm in (
        a for a in SCORED_ARMS if a != BASE_ARM and a in set(losses["arm"].unique().to_list())
    ):
        paired = (
            losses.filter(pl.col("arm") == arm)
            .join(base, on=["site", "time", "seed"], validate="m:1")
            .with_columns(
                delta=pl.col("q50") - pl.col("base_median"), regressor=-pl.col("applied_shift")
            )
        )
        scopes = [("all leads", paired)] + [
            (f"band {band}", paired.filter(pl.col("band") == band)) for band in BAND_NAMES
        ]
        for scope_name, part in scopes:
            denominator = float((part["regressor"] ** 2).sum())
            rows.append(
                {
                    "arm": arm,
                    "scope": scope_name,
                    "effective_weight": float((part["delta"] * part["regressor"]).sum())
                    / denominator
                    if denominator > 0
                    else float("nan"),
                    "rows": part.height,
                }
            )
    return pl.DataFrame(rows)


def coverage_table(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return the fraction of actuals below the 0.05 and 0.95 quantiles in sunny and cloudy rows."""
    return (
        losses.filter(
            pl.col("arm").is_in(COVERAGE_ARMS) & pl.col("csi_bin").is_in(["sunny", "cloudy"])
        )
        .group_by("arm", "csi_bin")
        .agg(
            below_q05=(pl.col("actual") < pl.col("q05")).mean(),
            below_q95=(pl.col("actual") < pl.col("q95")).mean(),
            rows=pl.len(),
        )
        .sort("csi_bin", "arm")
    )


def by_month_table(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return the mean paired difference of N and T of each verdict arm in each valid month."""
    pieces = []
    for arm in VERDICT_ARMS:
        for contrast_id, reference in (("N", REFERENCE_ARM), ("T", TWIN_OF[arm])):
            paired = losses.filter(pl.col("arm") == arm).join(
                losses.filter(pl.col("arm") == reference).select(
                    "site", "time", "seed", reference=pl.col("qs")
                ),
                on=["site", "time", "seed"],
                validate="m:1",
            )
            pieces.append(
                paired.group_by("valid_month")
                .agg(difference=(pl.col("qs") - pl.col("reference")).mean(), rows=pl.len())
                .with_columns(contrast=pl.lit(f"{contrast_id} {arm}"))
                .sort("valid_month")
            )
    return pl.concat(pieces)


def temperature_read(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return the slope of the sunny-row quantile score on cell temperature for `sf` and `sf_temp`.

    Args:
        losses: `load_losses()`.

    Returns:
        Columns `arm`, `slope_points_per_kelvin`, `rows`, and `share_removed` (one minus the slope
        over `sf`'s slope; null for `sf`).
    """
    rows = []
    for arm in ("d1l", "sf", "sf_temp"):
        part = losses.filter(
            (pl.col("arm") == arm)
            & (pl.col("csi_bin") == "sunny")
            & pl.col("cell_temperature").is_not_null()
        )
        temperature = part["cell_temperature"].to_numpy()
        score = part["qs"].to_numpy()
        centred = temperature - temperature.mean()
        rows.append(
            {
                "arm": arm,
                "slope_points_per_kelvin": float((centred * (score - score.mean())).sum())
                / float((centred**2).sum()),
                "rows": part.height,
            }
        )
    table = pl.DataFrame(rows)
    sf_slope = float(table.filter(pl.col("arm") == "sf")["slope_points_per_kelvin"][0])
    return table.with_columns(
        share_removed=pl.when(pl.col("arm") == "sf_temp")
        .then(1.0 - pl.col("slope_points_per_kelvin") / sf_slope)
        .otherwise(None)
    )


def print_section(*, title: str, table: pl.DataFrame) -> None:
    """Print a titled table with every row and column shown."""
    with pl.Config(tbl_rows=300, tbl_cols=30, fmt_float="mixed", tbl_width_chars=220):
        print(f"\n{title}")
        print(table)


def main() -> None:
    """Print the absolute scores, the verdict and gain families, and the exploratory reads."""
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    context = context_table()
    losses = load_losses(context=context)
    first = losses.filter((pl.col("arm") == BASE_ARM) & (pl.col("seed") == 0))
    print(
        f"scored rows per seed on the five verdict series: {first.height:,}; "
        f"seeds {sorted(losses['seed'].unique().to_list())}; "
        f"forecast-start weeks {losses['month'].n_unique()}; "
        f"valid months {sorted(first['valid_month'].unique().to_list())}"
    )
    print(f"arms scored: {sorted(losses['arm'].unique().to_list())}")
    absolute = absolute_scores(losses=losses)
    print_section(title="Absolute scores (points of capacity; smaller is better)", table=absolute)
    reference = stored_reference_scores(context=context)
    print_section(title="Stored reference arms of experiment 00033", table=reference)
    family = verdict_family(losses=losses)
    print_section(
        title=f"Verdict family and gain family ({VERDICT_LEVEL}% intervals, week blocks)",
        table=family.select(
            "contrast",
            "treatment",
            "reference",
            "scope",
            "difference",
            "lower",
            "upper",
            "half_width",
            "seed_spread",
            "rows_per_seed",
            "weeks",
        ),
    )
    verdict, readings = verdict_rule(family=family)
    print("\nVerdict rule: supported if some arm has N upper < +0.10 and T upper < 0;")
    print("refuted if no arm has N upper < +0.10 and every arm has N lower > 0; else inconclusive")
    for arm, reading in readings.items():
        print(f"  {arm}: {reading}")
    gains = family.filter((pl.col("contrast") == "gain") & (pl.col("upper") < GAIN_MARGIN))
    gained = gains["treatment"].to_list() or "no arm"
    print(f"Gain reported (upper end below {GAIN_MARGIN}) for: {gained}")
    print(f"VERDICT: {verdict}")
    exploratory = exploratory_contrasts(losses=losses)
    print_section(
        title="Exploratory contrasts (95%; week blocks; negative favours the treatment)",
        table=exploratory.select(
            "contrast",
            "treatment",
            "reference",
            "scope",
            "difference",
            "lower",
            "upper",
            "seed_spread",
            "rows_per_seed",
        ),
    )
    weights = effective_weights(losses=losses)
    print_section(title="Effective weights (compare with experiment 51's slopes)", table=weights)
    fitted = pl.read_parquet(common.WEIGHTS_FILE).filter(pl.col("variant") == "all")
    print_section(
        title="Fitted weights w (mean over series by lead band and season)",
        table=fitted.group_by("band", "season")
        .agg(
            slope_oracle=pl.col("slope_oracle").mean(), slope_applied=pl.col("slope_applied").mean()
        )
        .sort("band", "season"),
    )
    coverage = coverage_table(losses=losses)
    print_section(title="Coverage: fraction of actuals below q05 and q95", table=coverage)
    by_month = by_month_table(losses=losses)
    print_section(
        title="Mean paired difference by local valid month (the sign check)",
        table=by_month.pivot(on="valid_month", index="contrast", values="difference"),
    )
    temperature = temperature_read(losses=losses)
    print_section(title="Sunny-row score against cell temperature", table=temperature)
    for name, table in (
        ("absolute", absolute),
        ("verdict_family", family),
        ("exploratory", exploratory),
        ("effective_weights", weights),
        ("coverage", coverage),
        ("by_month", by_month),
        ("temperature", temperature),
        ("stored_reference", reference),
    ):
        table.write_parquet(common.RESULTS_DIR / f"{name}.parquet")
    print(f"\nresults written to {common.RESULTS_DIR}")


if __name__ == "__main__":
    main()
