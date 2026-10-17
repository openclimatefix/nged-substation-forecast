"""Score the arms of experiment 00074, read the pre-registered contrasts, and print every number.

Reads `work/arms/<arm>.parquet` (see `e00074_assemble.py`) and writes under `work/results/`:

- `absolute.parquet`: each arm's mean quantile score and error of the median, with 95% intervals.
- `planned.parquet`: the six planned contrasts C1 to C6, at 99.17% (5% split across six) from
  50,000 resamples of whole ISO weeks of forecast start, on the daylight April-to-September rows.
- `exploratory.parquet`: every other contrast and split, at 95%.
- `coverage.parquet`: the share of actuals below the 0.05 and 0.95 quantiles, sunny and cloudy.
- `analogue_counts.parquet`: the share of validation cases with 0, 1, 2, and 3 or more analogues.

The decision rule is `verdict_rule()`, the one in the plan's front matter. The score is in points of
capacity (1 point is 1% of the series' effective capacity).

    uv run python ml-experiments/00074-*/e00074_stats.py
"""

import argparse
from typing import Final

import e00074_common as common
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

PLANNED_ARMS: Final[tuple[str, ...]] = ("R0", "E", "E_twin", "A1", "A3", "A5", "A3E", "A3S")
CONTEXT_ARMS: Final[tuple[str, ...]] = ("R0_rad",)
PLANNED_CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("C1", "E", "R0"),
    ("C2", "A3", "R0"),
    ("C3", "A3", "E"),
    ("C4", "A3E", "E"),
    ("C5", "E", "E_twin"),
    ("C6", "A3", "A3S"),
)
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / len(PLANNED_CONTRASTS))
"""99.17: a 5% familywise level split over the six planned contrasts."""
EXPLORATORY_LEVEL: Final[float] = 95.0
PLANNED_RESAMPLES: Final[int] = 50_000
EXPLORATORY_RESAMPLES: Final[int] = 10_000
PORTING_THRESHOLD: Final[float] = -0.05
"""Points of capacity: the gain of the analogues on top of the offset must clear this to port."""
GAIN_BANDS: Final[tuple[str, ...]] = ("0-24 h", "1-2 d")
BAND_NAMES: Final[tuple[str, ...]] = tuple(name for name, _low, _high in common.LEAD_BANDS)
CSI_BINS: Final[tuple[str, ...]] = ("cloudy", "middle", "sunny")
SUNNIEST_QUANTILE: Final[float] = 0.9
EVERYTHING: Final[pl.Expr] = pl.lit(value=True)


def context_table(*, night: bool) -> pl.DataFrame:
    """Return the scored rows of the five series with the facts the splits use.

    Args:
        night: Return the April-to-September rows after dark (not daylight, with an observed power)
            when true, and the scored daylight rows when false.

    Returns:
        One row per run, valid time, and series with `KEYS`, `series_label`, `band`, `valid_month`,
        `lead_hours`, `clear_sky_index`, `csi_bin`, and `sunniest`.
    """
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    arrays = common.load_series_arrays(series=series)
    rows = common.scored_rows(series=series, arrays=arrays)
    if night:
        rows = rows.filter(
            ~pl.col("daylight")
            & pl.col("actual").is_finite()
            & pl.col("month").is_in(list(common.SCORED_MONTHS))
            & (pl.col("valid_time") < common.HOLDOUT)
        )
    else:
        rows = rows.filter(pl.col("scored"))
    threshold = rows["clear_sky_index"].quantile(SUNNIEST_QUANTILE)
    return rows.with_columns(
        csi_bin=pl.when(pl.col("clear_sky_index") >= common.SUNNY_INDEX)
        .then(pl.lit("sunny"))
        .when(pl.col("clear_sky_index") < common.CLOUDY_INDEX)
        .then(pl.lit("cloudy"))
        .when(pl.col("clear_sky_index").is_not_null())
        .then(pl.lit("middle")),
        sunniest=(pl.col("clear_sky_index") >= threshold).fill_null(value=False),
        band_name=pl.col("band").map_elements(BAND_NAMES.__getitem__, return_dtype=pl.String),
    ).select(
        *common.KEYS,
        series_label=pl.col("site"),
        band=pl.col("band_name"),
        valid_month=pl.col("month"),
        lead_hours=pl.col("lead_hours"),
        clear_sky_index=pl.col("clear_sky_index"),
        csi_bin=pl.col("csi_bin"),
        sunniest=pl.col("sunniest"),
    )


def with_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `qs` and `mae` in points of capacity to a frame of quantiles and `actual`."""
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    return frame.with_columns(
        qs=pl.Series(common.pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0),
        mae=pl.Series(np.abs(quantiles[:, common.MEDIAN] - actual) * 100.0),
    )


def load_losses(
    *, context: pl.DataFrame, arms: tuple[str, ...], prefix: str = "", seeds: tuple[int, ...] = (0,)
) -> pl.DataFrame:
    """Return every named arm's per-row losses on the context rows, resampled by ISO week.

    Args:
        context: `context_table()`.
        arms: The arms to read.
        prefix: The file-name prefix (`ref_` for the context arms of experiment 64).
        seeds: The seeds to keep (the context arms of experiment 64 hold seeds 0 to 2).

    Returns:
        The frame `experiment_store.losses_for_bootstrap` builds, with the context columns, `qs`,
        `mae`, the nine quantiles, and `actual`; its `month` column holds the ISO week of the
        forecast start (`2025-W27`), so every bootstrap resamples whole weeks.
    """
    arm_losses = {
        arm: with_losses(
            frame=pl.read_parquet(common.ARMS_DIR / f"{prefix}{arm}.parquet")
            .filter(pl.col("seed").is_in(list(seeds)))
            .join(context, on=list(common.KEYS), how="inner", validate="m:1")
        )
        for arm in arms
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
    """Return one contrast's mean paired difference and its week-block interval."""
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
        "rows_per_seed": differences.shape[1],
        "weeks": len(np.unique(weeks)),
    }


def planned_contrasts(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Read C1 to C6 at 99.17%."""
    return pl.DataFrame(
        [
            read_contrast(
                losses=losses,
                contrast_id=contrast_id,
                treatment=treatment,
                reference=reference,
                scope_name="all leads",
                scope=EVERYTHING,
                level=PLANNED_LEVEL,
                n_resamples=PLANNED_RESAMPLES,
            )
            for contrast_id, treatment, reference in PLANNED_CONTRASTS
        ]
    )


def verdict_rule(*, planned: pl.DataFrame) -> tuple[str, str]:
    """Apply the pre-registered rule and return the verdict with the reads that decided it.

    `refuted` when C3's upper end is below zero (the analogues beat the explicit forecast) or C4's
    upper end is below -0.05 (the analogues add a porting-worthy gain on top of it). Otherwise,
    when C1's upper end is below zero and C3's lower end is above zero: `small gain` if C4's upper
    end is below zero, and `supported` if C4's lower end is at or above -0.05. `inconclusive`
    otherwise.

    Args:
        planned: `planned_contrasts()`.

    Returns:
        The verdict and a sentence naming the reads.
    """
    read = {r["contrast"]: r for r in planned.iter_rows(named=True)}
    c1, c3, c4 = read["C1"], read["C3"], read["C4"]
    facts = (
        f"C1 upper {c1['upper']:+.3f}, C3 lower {c3['lower']:+.3f} and upper {c3['upper']:+.3f}, "
        f"C4 lower {c4['lower']:+.3f} and upper {c4['upper']:+.3f}"
    )
    if c3["upper"] < 0.0:
        return "refuted", f"the analogues beat the explicit forecast ({facts})"
    if c4["upper"] < PORTING_THRESHOLD:
        return "refuted", f"the analogues add a porting-worthy gain on top of it ({facts})"
    if c1["upper"] < 0.0 and c3["lower"] > 0.0:
        if c4["upper"] < 0.0:
            return "small gain", f"the analogues help a little on top of the offset ({facts})"
        if c4["lower"] >= PORTING_THRESHOLD:
            return "supported", f"a gain of 0.05 points from the analogues is ruled out ({facts})"
    return "inconclusive", f"the reads do not meet either rule ({facts})"


def absolute_scores(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return each arm's mean score and median error on the scored rows, with 95% intervals."""
    rows = []
    for arm in losses["arm"].unique(maintain_order=True).to_list():
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


def exploratory_contrasts(*, losses: pl.DataFrame, night_losses: pl.DataFrame) -> pl.DataFrame:
    """Read the exploratory contrasts and splits at 95%.

    The reads are: `A1` and `A5` against `A3`; `A3E` against `A3`; `R0_rad` against `R0`; the
    planned contrasts C1 to C6 over leads under 48 hours, in the sunniest decile, by lead band, by
    clear-sky-index bin, and by series; and the night rows for `A3` against `A3S` and `R0`.

    Args:
        losses: `load_losses()` for the daylight rows.
        night_losses: `load_losses()` for the night rows.

    Returns:
        One row per read, in the shape of `read_contrast`.
    """
    specs: list[tuple[str, str, str, str, pl.Expr]] = [
        ("K", "A1", "A3", "all leads", EVERYTHING),
        ("K", "A5", "A3", "all leads", EVERYTHING),
        ("both_vs_A3", "A3E", "A3", "all leads", EVERYTHING),
        ("radiation_context", "R0_rad", "R0", "all leads", EVERYTHING),
    ]
    for contrast_id, treatment, reference in PLANNED_CONTRASTS:
        specs.append(
            (
                f"{contrast_id}_under_48h",
                treatment,
                reference,
                "leads under 48 h",
                pl.col("band").is_in(list(GAIN_BANDS)),
            )
        )
        specs.append(
            (f"{contrast_id}_sunniest", treatment, reference, "sunniest decile", pl.col("sunniest"))
        )
        specs.extend(
            (f"{contrast_id}_by_band", treatment, reference, f"band {band}", pl.col("band") == band)
            for band in BAND_NAMES
        )
        specs.extend(
            (
                f"{contrast_id}_by_csi",
                treatment,
                reference,
                f"csi {csi_bin}",
                pl.col("csi_bin") == csi_bin,
            )
            for csi_bin in CSI_BINS
        )
        specs.extend(
            (
                f"{contrast_id}_by_series",
                treatment,
                reference,
                f"series {site}",
                pl.col("series_label") == site,
            )
            for site in common.SITES
        )
        specs.extend(
            (
                f"{contrast_id}_by_month",
                treatment,
                reference,
                f"month {month}",
                pl.col("valid_month") == month,
            )
            for month in common.SCORED_MONTHS
        )
    rows = [
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
    ]
    rows += [
        read_contrast(
            losses=night_losses,
            contrast_id=contrast_id,
            treatment=treatment,
            reference=reference,
            scope_name="night rows",
            scope=EVERYTHING,
            level=EXPLORATORY_LEVEL,
            n_resamples=EXPLORATORY_RESAMPLES,
        )
        for contrast_id, treatment, reference in (
            ("night_A3_vs_A3S", "A3", "A3S"),
            ("night_A3_vs_R0", "A3", "R0"),
            ("night_E_vs_R0", "E", "R0"),
        )
    ]
    return pl.DataFrame(rows)


def coverage_table(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return the share of actuals below the 0.05 and 0.95 quantiles, sunny and cloudy rows."""
    return (
        losses.filter(pl.col("csi_bin").is_in(["sunny", "cloudy"]))
        .group_by("arm", "csi_bin")
        .agg(
            below_q05=(pl.col("actual") < pl.col("q05")).mean(),
            below_q95=(pl.col("actual") < pl.col("q95")).mean(),
            rows=pl.len(),
        )
        .sort("arm", "csi_bin")
    )


def stored_reference_scores(*, context: pl.DataFrame) -> pl.DataFrame:
    """Return the mean score of experiment 00033's `climatology` and `manual_heuristic`.

    These arms lack 40 of the context rows, so they are scored on the rows they share with the
    context, without an interval.
    """
    rows = []
    for arm in ("climatology", "manual_heuristic"):
        frame = with_losses(
            frame=pl.read_parquet(common.ARMS_DIR / f"ref_{arm}.parquet").join(
                context, on=list(common.KEYS), how="inner", validate="m:1"
            )
        )
        rows.append(
            {
                "arm": arm,
                "qs": float(np.mean(frame["qs"].to_numpy())),
                "mae": float(np.mean(frame["mae"].to_numpy())),
            }
        )
    return pl.DataFrame(rows)


def analogue_counts() -> pl.DataFrame:
    """Return the share of validation cases with 0, 1, 2, and 3 or more matched analogues."""
    frames = [
        pl.read_parquet(path, columns=["a3_an_count", "ensemble_member"]).filter(
            pl.col("ensemble_member") == 0
        )
        for path in sorted((common.WORK_DIR / "val").glob("*/init=*.parquet"))
    ]
    counts = pl.concat(frames)["a3_an_count"].cast(pl.Int32).clip(upper_bound=3)
    return counts.value_counts(normalize=True).sort("a3_an_count").rename({"a3_an_count": "count"})


def main() -> None:
    """Compute and print every read."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--tag", default="seed0", help="the folder under work/results/")
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    results_dir = common.RESULTS_DIR / args.tag
    results_dir.mkdir(parents=True, exist_ok=True)
    day = context_table(night=False)
    night = context_table(night=True)
    print(f"daylight scored rows: {day.height:,}; night rows: {night.height:,}")
    losses = load_losses(context=day, arms=(*PLANNED_ARMS, *CONTEXT_ARMS), seeds=seeds)
    night_losses = load_losses(context=night, arms=("R0", "E", "A3", "A3S"), seeds=seeds)
    absolute = absolute_scores(losses=losses)
    planned = planned_contrasts(losses=losses)
    verdict, reason = verdict_rule(planned=planned)
    exploratory = exploratory_contrasts(losses=losses, night_losses=night_losses)
    coverage = coverage_table(losses=losses)
    counts = analogue_counts()
    refs = pl.concat(
        [
            absolute_scores(
                losses=load_losses(context=day, arms=("d0", "d1l", "am"), prefix="ref_")
            ),
            stored_reference_scores(context=day),
        ],
        how="diagonal",
    )
    for name, table in (
        ("absolute", absolute),
        ("planned", planned),
        ("exploratory", exploratory),
        ("coverage", coverage),
        ("analogue_counts", counts),
        ("reference_absolute", refs),
    ):
        common.write_atomically(frame=table, target=results_dir / f"{name}.parquet")
    with pl.Config(tbl_rows=300, tbl_cols=20, tbl_width_chars=220):
        print("Absolute scores (points of capacity, 95% intervals)")
        print(absolute)
        print("Context arms of experiment 64 (frozen models; d0, d1l and am carry the correction)")
        print(refs)
        print(f"Planned contrasts at {PLANNED_LEVEL:.2f}%")
        print(planned)
        print(f"Verdict: {verdict}: {reason}")
        print("Exploratory contrasts at 95%")
        print(exploratory)
        print("Coverage of the 0.05 and 0.95 quantiles")
        print(coverage)
        print("Share of validation cases by number of matched analogues (A3)")
        print(counts)


if __name__ == "__main__":
    main()
