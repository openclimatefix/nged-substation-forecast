"""Score each arm with a timing-tolerant metric, at a run of temporal tolerances.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**Mean absolute error charges a forecast twice for a peak it places half an hour late: once for the
power it predicted that did not arrive, and once for the power that arrived unpredicted.** A
forecast that gets a spike's magnitude right and its timing wrong therefore scores worse than a
forecast that never predicts a spike at all, which is the behaviour the beam/diffuse split is meant
to buy. The Fractions Skill Score measures how much of that penalty is timing rather than
magnitude, by asking whether each arm put a threshold exceedance *somewhere near* the right hour.

The score is computed on the forecasts the runs already wrote. `run_experiment.py` stores
`signed_error_capped_mw` as `capped_point - actual`, so adding the metered power back recovers the
capped point forecast exactly.

Run it with `uv run python
studies/beam_diffuse_split/fractions_skill_score.py --source ukv`.
"""

import argparse
import sys
from typing import Final

import numpy as np
import polars as pl
from sources import SOURCE_CHOICES, STUDY_DATA_DIR
from studies.fractions_skill_score import fss_from, monthly_components, on_a_complete_hourly_grid

WINDOW_HOURS: Final[tuple[int, ...]] = (1, 3, 5, 7, 9)
"""The temporal tolerances the score is reported at, in hours, widest last.

A window of 1 hour is the point score, which carries the full double penalty and is the comparison
every other column is read against. Each wider window centres on the same hour and forgives a
displacement of up to half the window either side, so 3 hours forgives an hour and 9 hours forgives
four. The target NGED forecasts is half-hourly, so a tolerance beyond a few hours is
uninteresting: a spike moved 4 hours is a different day's weather, not a timing error.
"""

BOOTSTRAP_DRAWS: Final[int] = 1000
"""How many monthly block resamples the interval on an FSS difference is built from."""

CONFIDENCE_PERCENTILES: Final[tuple[float, float]] = (2.5, 97.5)
"""The percentiles of the bootstrap distribution reported as a 95% interval."""

REPORTED_ARMS: Final[dict[str, tuple[str, ...]]] = {
    "xgboost": ("A_global_only", "B_erbs", "C_era5_split", "D_direct_fraction"),
    "physics": ("P_A_global_only", "P_B_erbs", "P_C_source_split", "P_E_blended"),
}
"""The arms each instrument's table prints, in the order it prints them.

The two separation-model arms the main report carries — `B_disc` and `B_learned` — are left out
because the double penalty is a claim about the published split, and the headline contrast that
claim rests on is the source's own split against the Erbs split.
"""

HEADLINE_CONTRAST: Final[dict[str, tuple[str, str]]] = {
    "xgboost": ("C_era5_split", "B_erbs"),
    "physics": ("P_C_source_split", "P_B_erbs"),
}
"""Each instrument's treatment and reference arm, whose FSS difference gets a bootstrap interval."""


def _capped_forecasts(*, source: str, suffix: str, instrument: str) -> pl.DataFrame:
    """Rebuild each arm's capped point forecast from its stored signed error.

    Args:
        source: The irradiance source the run used.
        suffix: Distinguishes a variant build from the main one for the same source, and is empty
            for the main one.
        instrument: Which instrument's results to read, `xgboost` or `physics`.

    Returns:
        One row per site, hour, arm, and seed, carrying the metered power and the forecast.

    Raises:
        FileNotFoundError: If that run or its dataset has not been produced.
    """
    stem = "results" if instrument == "xgboost" else "physics"
    losses_path = (
        STUDY_DATA_DIR / f"beam_diffuse_{stem}_{source}{suffix}" / "per_row_losses.parquet"
    )
    dataset_path = STUDY_DATA_DIR / f"beam_diffuse_dataset_{source}{suffix}.parquet"
    for path in (losses_path, dataset_path):
        if not path.exists():
            msg = f"{path} missing; run the {instrument} instrument on {source} first"
            raise FileNotFoundError(msg)

    losses = pl.read_parquet(losses_path).filter(
        (pl.col("setting") == "primary") & (pl.col("target") == "power_mw")
    )
    observed = pl.read_parquet(dataset_path).select("site", "time", "power_mw")
    return losses.join(observed, on=["site", "time"], how="inner").with_columns(
        forecast_mw=pl.col("power_mw") + pl.col("signed_error_capped_mw")
    )


def _bootstrap_fss_difference(
    *, treatment: pl.DataFrame, reference: pl.DataFrame, generator: np.random.Generator
) -> dict[str, float]:
    """Interval the difference of two arms' scores by resampling whole months.

    Both arms are resampled on the *same* drawn months, so the paired structure that makes the
    difference precise is preserved. The two arms saw the same weather, and the interval should
    reflect the shared weather rather than treat the arms as independent runs.

    Args:
        treatment: The treatment arm's per-month components.
        reference: The reference arm's per-month components.
        generator: The random source the month draws come from.

    Returns:
        The point difference and its 95% interval.
    """
    paired = treatment.join(reference, on="month", how="inner", suffix="_reference")
    columns = paired.select(
        "squared_difference", "reference", "squared_difference_reference", "reference_reference"
    ).to_numpy()
    point = fss_from(
        squared_difference=float(columns[:, 0].sum()), reference=float(columns[:, 1].sum())
    ) - fss_from(
        squared_difference=float(columns[:, 2].sum()), reference=float(columns[:, 3].sum())
    )

    draws = generator.integers(0, len(columns), size=(BOOTSTRAP_DRAWS, len(columns)))
    sums = columns[draws].sum(axis=1)
    differences = np.array(
        [
            fss_from(squared_difference=float(row[0]), reference=float(row[1]))
            - fss_from(squared_difference=float(row[2]), reference=float(row[3]))
            for row in sums
        ]
    )
    lower, upper = np.nanpercentile(differences, CONFIDENCE_PERCENTILES)
    return {"difference": point, "lower_95": float(lower), "upper_95": float(upper)}


def main() -> int:
    """Print the score for every arm, at every tolerance, and interval the headline contrast.

    Returns:
        The process exit status.

    Raises:
        ValueError: If no arm this instrument reports is present in the run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE_CHOICES, default="open-meteo")
    parser.add_argument("--instrument", choices=("xgboost", "physics"), default="xgboost")
    parser.add_argument("--suffix", default="", help="Selects a variant build of the same source.")
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.9,
        help="The per-site quantile of metered power an hour must exceed to count.",
    )
    arguments = parser.parse_args()

    forecasts = _capped_forecasts(
        source=arguments.source,
        suffix=arguments.suffix,
        instrument=arguments.instrument,
    )
    thresholds = (
        forecasts.filter(pl.col("seed") == pl.col("seed").min())
        .group_by("site")
        .agg(threshold_mw=pl.col("power_mw").quantile(arguments.threshold_quantile))
    )
    present = forecasts["arm"].unique().to_list()
    arms = [arm for arm in REPORTED_ARMS[arguments.instrument] if arm in present]
    if not arms:
        msg = (
            f"none of the {arguments.instrument} arms {REPORTED_ARMS[arguments.instrument]} "
            f"is in this run, which holds {sorted(present)}"
        )
        raise ValueError(msg)
    seeds = sorted(forecasts["seed"].unique().to_list())

    components: dict[tuple[str, int, int], pl.DataFrame] = {}
    for arm in arms:
        for seed in seeds:
            gridded = on_a_complete_hourly_grid(
                frame=forecasts.filter((pl.col("arm") == arm) & (pl.col("seed") == seed)),
                thresholds=thresholds,
            )
            for window in WINDOW_HOURS:
                components[arm, seed, window] = monthly_components(
                    gridded=gridded, window_hours=window
                )

    label = f"{arguments.source}{arguments.suffix}"
    lines = [
        f"### Fractions Skill Score — {label}, {arguments.instrument}",
        "",
        (
            f"Threshold: each site's {arguments.threshold_quantile:.0%} quantile of metered power. "
            f"Scores are averaged over {len(seeds)} seeds."
        ),
        "",
        "| Arm | " + " | ".join(f"±{(w - 1) // 2}h" for w in WINDOW_HOURS) + " |",
        "|---|" + "---|" * len(WINDOW_HOURS),
    ]
    for arm in arms:
        cells = []
        for window in WINDOW_HOURS:
            scores = [
                fss_from(
                    squared_difference=float(
                        components[arm, seed, window]["squared_difference"].sum()
                    ),
                    reference=float(components[arm, seed, window]["reference"].sum()),
                )
                for seed in seeds
            ]
            cells.append(f"{float(np.nanmean(scores)):.4f}")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    treatment_arm, reference_arm = HEADLINE_CONTRAST[arguments.instrument]
    if treatment_arm in arms and reference_arm in arms:
        lines += [
            "",
            f"{treatment_arm} − {reference_arm}, pooled over seeds:",
            "",
            "| Tolerance | ΔFSS | 95% interval | Excludes zero? |",
            "|---|---|---|---|",
        ]
        generator = np.random.default_rng(0)
        for window in WINDOW_HOURS:
            treatment = (
                pl.concat([components[treatment_arm, seed, window] for seed in seeds])
                .group_by("month")
                .agg(pl.col("squared_difference").sum(), pl.col("reference").sum())
            )
            reference = (
                pl.concat([components[reference_arm, seed, window] for seed in seeds])
                .group_by("month")
                .agg(pl.col("squared_difference").sum(), pl.col("reference").sum())
            )
            interval = _bootstrap_fss_difference(
                treatment=treatment, reference=reference, generator=generator
            )
            excludes = interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0
            lines.append(
                f"| ±{(window - 1) // 2}h | {interval['difference']:+.4f} | "
                f"[{interval['lower_95']:+.4f}, {interval['upper_95']:+.4f}] | "
                f"{'**yes**' if excludes else 'no'} |"
            )
    lines.append("")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
