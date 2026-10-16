"""Read experiment 00046's arms on summer midday rows, by lead day (exploratory, post hoc).

The maintainer asked for a read of the arms where solar matters most: valid times 09:00 to 15:00 UTC
inclusive (the stamps are period-ending), in June to August and, as a second read, April to
September, at the 5 verdict series and again without D9. Lead bands are measured from each run's
`power_fcst_init_time`, the same origin as `common.LEAD_BANDS`, but with their own edges:
day+1 is 24 to 48 hours, day+2 is 48 to 72 hours, and day+3 is 72 to 96 hours. The existing bands
(0-24 h, 1-2 d, 2-6 d) do not share the day+2 and day+3 edges, so they are not used.

For each lead band, row set, and series set the script prints the absolute quantile score of five
arms, the contrasts C3, C4, and C6 with 95% month-block and seed bootstrap intervals and the number
of blocks, the coverage of the 0.05 and 0.95 quantiles in sunny and cloudy rows, and the
correlation of the ensemble-mean ENS anomaly with the CAMS anomaly. It writes `midday.parquet` and
`midday_coverage.parquet` under `work/results/`.

    uv run python ml-experiments/00046-*/e00046_midday.py
"""

from datetime import time
from typing import Final

import e00046_common as common
import e00046_stats as stats
import numpy as np
import polars as pl

from studies import bootstrap

MIDDAY_START: Final[time] = time(9, 0)
MIDDAY_END: Final[time] = time(15, 0)
DAY_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("day+1 (24-48 h)", 24.0, 48.0),
    ("day+2 (48-72 h)", 48.0, 72.0),
    ("day+3 (72-96 h)", 72.0, 96.0),
)
ROW_SETS: Final[dict[str, tuple[int, ...]]] = {
    "June-August": (6, 7, 8),
    "April-September": (4, 5, 6, 7, 8, 9),
}
SERIES_SETS: Final[dict[str, tuple[str, ...]]] = {
    "5 verdict series": common.VERDICT_SITES,
    "without D9": tuple(site for site in common.VERDICT_SITES if site != stats.D9),
}
ARMS: Final[tuple[str, ...]] = (
    "d0",
    "d1l",
    "d1l_ens_shift",
    "d1l_cams_shift",
    "d1l_ens_shift_fitted",
)
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C3": stats.CONTRASTS["C3"],
    "C4": stats.CONTRASTS["C4"],
    "C6": stats.CONTRASTS["C6"],
}
RESAMPLES: Final[int] = 50_000
LEVEL: Final[float] = 95.0


def midday_frame(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Return the losses on 09:00 to 15:00 UTC valid times with the lead day of each row."""
    clock = pl.col("valid_time").dt.time()
    edges = [low for _name, low, _high in DAY_BANDS]
    names = [name for name, _low, _high in DAY_BANDS]
    in_bands = (pl.col("lead_hours") >= edges[0]) & (pl.col("lead_hours") < DAY_BANDS[-1][2])
    return (
        losses.filter(clock.is_between(MIDDAY_START, MIDDAY_END) & in_bands)
        .with_columns(
            day_band=pl.col("lead_hours").map_elements(
                lambda hours: names[int(np.searchsorted(edges, hours, side="right")) - 1],
                return_dtype=pl.String,
            )
        )
        .filter(pl.col("series_label").is_in(list(common.VERDICT_SITES)))
    )


def ens_correlations(*, rows: pl.DataFrame) -> float:
    """Return the correlation of the ensemble-mean ENS anomaly with the CAMS anomaly on the rows.

    Args:
        rows: One arm and seed of the midday frame, carrying `series_label`,
            `power_fcst_init_time`, and `valid_time`.

    Returns:
        Pearson's r over the rows that have both anomalies.
    """
    ens = (
        pl.concat(
            [pl.read_parquet(path) for path in (common.SOLAR_DIR / "ens_val").glob("*.parquet")]
        )
        .group_by("site", "power_fcst_init_time", "valid_time")
        .agg(ens_mean=pl.col("anomaly").mean())
    )
    cams = pl.read_parquet(common.SOLAR_DIR / "cams_anomaly.parquet")
    joined = (
        rows.select(pl.col("series_label").alias("site"), "power_fcst_init_time", "valid_time")
        .join(ens, on=["site", "power_fcst_init_time", "valid_time"])
        .join(cams, on=["site", "valid_time"])
    )
    return float(
        np.corrcoef(joined["ens_mean"].to_numpy(), joined["anomaly_cams"].to_numpy())[0, 1]
    )


def main() -> None:
    """Print and save the midday reads."""
    losses = midday_frame(losses=stats.load_losses())
    score_rows, contrast_rows, coverage_rows = [], [], []
    for row_name, months in ROW_SETS.items():
        for series_name, sites in SERIES_SETS.items():
            base = pl.col("series_label").is_in(list(sites)) & pl.col("valid_month").is_in(
                list(months)
            )
            for band_name, _low, _high in DAY_BANDS:
                scope = base & (pl.col("day_band") == band_name)
                part = losses.filter(scope)
                one = part.filter((pl.col("arm") == "d1l") & (pl.col("seed") == 0))
                label = {"row_set": row_name, "series": series_name, "band": band_name}
                correlation = ens_correlations(rows=one)
                for arm in ARMS:
                    interval = bootstrap.bootstrap_absolute(losses=part, arm=arm, metric="qs")
                    score_rows.append(
                        label
                        | {
                            "arm": arm,
                            "qs": interval["value"],
                            "lower": interval["lower_95"],
                            "upper": interval["upper_95"],
                            "rows_per_seed": one.height,
                            "ens_cams_correlation": correlation,
                        }
                    )
                for arm in ("d0", "d1l", "d1l_ens_shift"):
                    for sky, test in (
                        ("sunny", pl.col("csi_bin") == "sunny"),
                        ("cloudy", pl.col("csi_bin") == "cloudy"),
                    ):
                        sub = part.filter((pl.col("arm") == arm) & test)
                        coverage_rows.append(
                            label
                            | {
                                "arm": arm,
                                "sky": sky,
                                "below_q05": float(
                                    np.mean(sub["actual"].to_numpy() < sub["q05"].to_numpy())
                                ),
                                "below_q95": float(
                                    np.mean(sub["actual"].to_numpy() < sub["q95"].to_numpy())
                                ),
                                "rows": sub.height,
                            }
                        )
                for contrast_id, (treatment, reference) in CONTRASTS.items():
                    read = stats.read_contrast(
                        losses=losses,
                        contrast_id=contrast_id,
                        treatment=treatment,
                        reference=reference,
                        scope_name=f"{row_name}, {series_name}, {band_name}",
                        scope=scope,
                        level=LEVEL,
                        n_resamples=RESAMPLES,
                        scored_only=False,
                    )
                    contrast_rows.append(label | read)
    scores = pl.DataFrame(score_rows)
    contrasts = pl.DataFrame(contrast_rows)
    coverage = pl.DataFrame(coverage_rows)
    with pl.Config(tbl_rows=300, tbl_cols=30, fmt_float="mixed", tbl_width_chars=220):
        print("Midday (09:00-15:00 UTC valid) quantile scores, points of capacity")
        print(scores.drop("ens_cams_correlation"))
        print("\nCorrelation of the ensemble-mean ENS anomaly with the CAMS anomaly")
        print(
            scores.select(
                "row_set", "series", "band", "rows_per_seed", "ens_cams_correlation"
            ).unique(maintain_order=True)
        )
        print("\nContrasts, 95% month-block and seed bootstrap (negative favours the treatment)")
        print(
            contrasts.select(
                "row_set",
                "series",
                "band",
                "contrast",
                "difference",
                "lower",
                "upper",
                "rows_per_seed",
                "blocks",
            )
        )
        print("\nCoverage of the 0.05 and 0.95 quantiles")
        print(coverage)
    scores.write_parquet(common.RESULTS_DIR / "midday.parquet")
    contrasts.write_parquet(common.RESULTS_DIR / "midday_contrasts.parquet")
    coverage.write_parquet(common.RESULTS_DIR / "midday_coverage.parquet")


if __name__ == "__main__":
    main()
