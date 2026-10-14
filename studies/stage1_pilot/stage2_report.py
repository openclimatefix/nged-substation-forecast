"""Print study 2b's tables from the saved per-row scores.

Reads `data/studies/per_study/stage2_pilot/scores/rows_seed<s>.parquet` and writes, under
`data/studies/per_study/stage2_pilot/results/`: `report.md` (every table the README quotes),
`levels.parquet`, and `contrasts.parquet`. Series appear only as anonymous labels in any per-series
output; this report prints none.

    uv run python studies/stage1_pilot/stage2_report.py
"""

import itertools
from pathlib import Path
from typing import Final

import polars as pl
import stage1_inputs as inputs
import stage1_stats as stats
from stage2_common import STAGE2_ROOT

SCORE_DIR: Final[Path] = STAGE2_ROOT / "scores"
RESULT_DIR: Final[Path] = STAGE2_ROOT / "results"
BANDS: Final[tuple[str, ...]] = ("all", *(name for name, _, _ in inputs.LEAD_BANDS))
TECHNOLOGIES: Final[tuple[str, ...]] = ("solar", "wind")
METRICS: Final[dict[str, str]] = {
    "crps": "CRPS estimate (points of capacity; smaller is better)",
    "abs_error": "Mean absolute error of the median (points of capacity; smaller is better)",
    "cover_80": "Coverage of the 10 to 90 interval (% of cases; nominal 80)",
    "cover_90": "Coverage of the 5 to 95 interval (% of cases; nominal 90)",
    "pinball_90": "Pinball loss at level 0.9 (points of capacity; smaller is better; exploratory)",
    "pinball_95": "Pinball loss at level 0.95 (points of capacity; smaller is better; exploratory)",
    "brier_90": (
        "Brier score of exceeding the series' training-period 90th percentile of power "
        "(100 times the squared error of the probability; smaller is better; exploratory)"
    ),
    "brier_95": (
        "Brier score of exceeding the series' training-period 95th percentile of power "
        "(100 times the squared error of the probability; smaller is better; exploratory)"
    ),
}
TAIL_METRICS: Final[tuple[str, ...]] = ("pinball_90", "pinball_95", "brier_90", "brier_95")
TAIL_CONTRASTS: Final[tuple[str, ...]] = ("Q1", "Q2", "Q3", "X5", "X6", "X7", "X11")
"""Contrasts also computed on the tail metrics; the others are read on CRPS and MAE only."""
RELIABILITY_ARMS: Final[tuple[str, ...]] = ("a_ens_quantile", "b_mixture_full", "c_stage2_full_blp")
RELIABILITY_BINS: Final[tuple[float, ...]] = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001)
BASE_ARMS: Final[tuple[str, ...]] = (
    "a_ens_quantile",
    "b_mixture_full",
    "b_mixture_short",
    "c_stage2_full",
    "c_stage2_short",
)
ARMS: Final[tuple[str, ...]] = (
    *[a for base in BASE_ARMS for a in (base, f"{base}_blp")],
    "climatology_q",
)
PLANNED: Final[dict[str, tuple[str, str]]] = {
    "Q1": ("c_stage2_full_blp", "b_mixture_full"),
    "Q2": ("c_stage2_full_blp", "a_ens_quantile"),
    "Q3": ("b_mixture_full", "a_ens_quantile"),
}
"""The planned contrasts as (treatment, reference), on the CRPS estimate and on the MAE."""
EXPLORATORY: Final[dict[str, tuple[str, str]]] = {
    "X1: c_stage2_full_blp minus c_stage2_full (the blend)": ("c_stage2_full_blp", "c_stage2_full"),
    "X2: b_mixture_full_blp minus b_mixture_full (the blend)": (
        "b_mixture_full_blp",
        "b_mixture_full",
    ),
    "X3: a_ens_quantile_blp minus a_ens_quantile (the blend)": (
        "a_ens_quantile_blp",
        "a_ens_quantile",
    ),
    "X4: c_stage2_full minus b_mixture_full (calibrator, no blend)": (
        "c_stage2_full",
        "b_mixture_full",
    ),
    "X5: c_stage2_short minus c_stage2_full (training length, stage 2)": (
        "c_stage2_short",
        "c_stage2_full",
    ),
    "X6: b_mixture_short minus b_mixture_full (training length, mixture)": (
        "b_mixture_short",
        "b_mixture_full",
    ),
    "X7: c_stage2_short_blp minus c_stage2_full_blp (training length, stage 2 and blend)": (
        "c_stage2_short_blp",
        "c_stage2_full_blp",
    ),
    "X8: c_stage2_full_blp minus climatology_q": ("c_stage2_full_blp", "climatology_q"),
    "X9: b_mixture_full minus climatology_q": ("b_mixture_full", "climatology_q"),
    "X10: a_ens_quantile minus climatology_q": ("a_ens_quantile", "climatology_q"),
    "X11: b_mixture_short minus a_ens_quantile": ("b_mixture_short", "a_ens_quantile"),
}
"""Contrasts that were not planned: all are exploratory."""


def load_rows(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Read every seed's rows and attach technology and lead band."""
    rows = pl.concat([pl.read_parquet(SCORE_DIR / f"rows_seed{s}.parquet") for s in inputs.SEEDS])
    band = pl.lit("none")
    for name, low, high in reversed(inputs.LEAD_BANDS):
        band = (
            pl.when((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
            .then(pl.lit(name))
            .otherwise(band)
        )
    return rows.join(
        sites.select("time_series_id", "technology"), on="time_series_id"
    ).with_columns(
        band=band,
        brier_90=100.0 * (pl.col("p_exceed_90") - pl.col("o_exceed_90")) ** 2,
        brier_95=100.0 * (pl.col("p_exceed_95") - pl.col("o_exceed_95")) ** 2,
    )


def scope(*, rows: pl.DataFrame, arm: str, technology: str, band: str) -> pl.DataFrame:
    """Return one arm's rows within a technology and lead band (`all` keeps every band)."""
    selected = rows.filter((pl.col("arm") == arm) & (pl.col("technology") == technology))
    return selected if band == "all" else selected.filter(pl.col("band") == band)


def table(*, header: list[str], body: list[list[str]]) -> list[str]:
    """Format a Markdown table."""
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *["| " + " | ".join(row) + " |" for row in body],
        "",
    ]


def reliability(*, rows: pl.DataFrame) -> list[str]:
    """Return reliability tables: forecast exceedance probability against observed frequency."""
    out = [
        "## Reliability of the exceedance probabilities (exploratory)",
        "",
        (
            "Each cell is the mean forecast probability, the observed frequency, and the number of "
            "cases (pooled over the three seeds and the series of the technology), by forecast "
            "probability bin. A reliable forecast has the first two numbers equal."
        ),
        "",
    ]
    saved = []
    for threshold in ("90", "95"):
        for technology in TECHNOLOGIES:
            for band in BANDS:
                body = []
                for low, high in itertools.pairwise(RELIABILITY_BINS):
                    cells = [f"[{low:.2f}, {min(high, 1.0):.2f})"]
                    for arm in RELIABILITY_ARMS:
                        part = scope(rows=rows, arm=arm, technology=technology, band=band).filter(
                            (pl.col(f"p_exceed_{threshold}") >= low)
                            & (pl.col(f"p_exceed_{threshold}") < high)
                        )
                        n = part.height
                        forecast = part[f"p_exceed_{threshold}"].mean() if n else float("nan")
                        observed = part[f"o_exceed_{threshold}"].mean() if n else float("nan")
                        saved.append(
                            {
                                "threshold": threshold,
                                "technology": technology,
                                "band": band,
                                "arm": arm,
                                "bin_low": low,
                                "forecast": forecast,
                                "observed": observed,
                                "cases": n,
                            }
                        )
                        cells.append(f"{forecast:.3f} / {observed:.3f} ({n:,})" if n else "-")
                    body.append(cells)
                out += [f"### exceeding the {threshold}th percentile, {technology}, {band}", ""]
                out += table(header=["forecast probability bin", *RELIABILITY_ARMS], body=body)
    pl.DataFrame(saved).write_parquet(RESULT_DIR / "reliability.parquet")
    return out


def main() -> None:
    """Compute every table and write the report."""
    sites = inputs.site_table()
    rows = load_rows(sites=sites)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    report = ["# Stage 2 pilot results", ""]
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    duplicates = rows.group_by("arm", "seed", *keys).len().filter(pl.col("len") > 1).height
    counts = (
        rows.filter(pl.col("seed") == 0)
        .group_by("arm", "technology")
        .agg(rows=pl.len())
        .sort("arm", "technology")
    )
    report += ["## Rows scored per arm and fitting seed", ""]
    report += table(
        header=["arm", "technology", "scored half-hours"],
        body=[[r["arm"], r["technology"], f"{r['rows']:,}"] for r in counts.iter_rows(named=True)],
    )
    report += [f"Duplicate (arm, seed, case) keys: {duplicates}.", ""]
    if duplicates:
        msg = "duplicate scored cases"
        raise ValueError(msg)

    level_rows = []
    for metric, title in METRICS.items():
        report += [f"## {title}", ""]
        for technology in TECHNOLOGIES:
            report += [f"### {technology}", ""]
            body = []
            for arm in ARMS:
                cells = []
                for band in BANDS:
                    part = scope(rows=rows, arm=arm, technology=technology, band=band)
                    interval = stats.level(table=stats.monthly(rows=part, value=metric))
                    level_rows.append(
                        {
                            "metric": metric,
                            "arm": arm,
                            "technology": technology,
                            "band": band,
                            "value": interval.value,
                            "lower": interval.lower,
                            "upper": interval.upper,
                        }
                    )
                    cells.append(
                        f"{interval.value:.2f} ({interval.lower:.2f} to {interval.upper:.2f})"
                    )
                body.append([arm, *cells])
            report += table(header=["arm", *BANDS], body=body)
    pl.DataFrame(level_rows).write_parquet(RESULT_DIR / "levels.parquet")

    contrast_rows = []
    report += [
        "## Paired contrasts",
        "",
        (
            "Treatment minus reference, in points of capacity (negative: the treatment has the "
            "smaller error). `months` is the number of the 12 validation months in which the "
            "difference is negative. Q1 to Q3 are planned; every other row is exploratory. A "
            "planned contrast is material when the mean difference is at most -0.15, the 95% "
            "interval lies wholly below zero, and at least 9 of 12 months are negative."
        ),
        "",
    ]
    for metric in ("crps", "abs_error", *TAIL_METRICS):
        for technology in TECHNOLOGIES:
            report += [f"### {metric}, {technology}", ""]
            body = []
            for name, (treatment, reference) in {**PLANNED, **EXPLORATORY}.items():
                if metric in TAIL_METRICS and name.split(":")[0] not in TAIL_CONTRASTS:
                    continue
                for band in BANDS:
                    a = scope(rows=rows, arm=treatment, technology=technology, band=band)
                    b = scope(rows=rows, arm=reference, technology=technology, band=band)
                    result = stats.contrast(
                        treatment=stats.monthly(rows=a, value=metric),
                        reference=stats.monthly(rows=b, value=metric),
                    )
                    planned = name in PLANNED
                    contrast_rows.append(
                        {
                            "metric": metric,
                            "contrast": name,
                            "planned": planned,
                            "technology": technology,
                            "band": band,
                            "difference": result.difference,
                            "lower": result.lower,
                            "upper": result.upper,
                            "months_better": result.months_better,
                            "material": result.material and planned,
                        }
                    )
                    body.append(
                        [
                            name,
                            band,
                            f"{result.difference:+.3f}",
                            f"{result.lower:+.3f} to {result.upper:+.3f}",
                            f"{result.months_better} of {result.n_months}",
                            ("yes" if result.material else "no") if planned else "-",
                        ]
                    )
            report += table(
                header=[
                    "contrast",
                    "lead band",
                    "difference",
                    "95% interval",
                    "months",
                    "material",
                ],
                body=body,
            )
    pl.DataFrame(contrast_rows).write_parquet(RESULT_DIR / "contrasts.parquet")

    report += reliability(rows=rows)
    params_path = SCORE_DIR / "blend_params.parquet"
    if params_path.exists():
        params = (
            pl.read_parquet(params_path)
            .group_by("arm", "technology", "band")
            .agg(
                alpha=pl.col("alpha").mean(),
                beta=pl.col("beta").mean(),
                share_unchanged=((pl.col("alpha") == 1.0) & (pl.col("beta") == 1.0)).mean(),
            )
            .sort("arm", "technology", "band")
        )
        report += ["## Learned blend parameters (mean over months and seeds)", ""]
        report += table(
            header=["arm", "technology", "band", "alpha", "beta", "share of fits left at 1 and 1"],
            body=[
                [
                    r["arm"],
                    r["technology"],
                    r["band"],
                    f"{r['alpha']:.2f}",
                    f"{r['beta']:.2f}",
                    f"{r['share_unchanged']:.2f}",
                ]
                for r in params.iter_rows(named=True)
            ],
        )
    (RESULT_DIR / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
