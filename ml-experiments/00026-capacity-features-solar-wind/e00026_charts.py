"""Draw experiment 00026's figures from the saved results, as SVG, and print the null shares.

1. The CRPS contrasts with intervals, one panel per technology.
2. Each farm's mean CRPS difference of arms A2, A3, and A7 against A0.

The null shares are those of each feature's daily table over the validation year (cutoff days
2025-07-01 to 2026-06-30), by series technology and feature.
The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00026-*/e00026_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00026_features as features
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00026_common import WORK_DIR, site_table

RESULTS_DIR: Final[Path] = WORK_DIR / "results"
CHART_DIR: Final[Path] = WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 520
WRAP_CHARACTERS: Final[int] = 100
SCOPE: Final[str] = (
    "6 solar farms and 3 wind farms; 53 forecast runs on Tuesdays, July 2025 to June 2026; "
    "pooled mixture of 51 ENS members, 3 seeds."
)
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "Single feature minus no feature",
    "C2": "Single feature minus shuffled copy",
    "C3": "Three features minus shuffled copies",
    "C4": "Three features minus single feature",
    "C5": "Single feature + shape feature minus single",
    "C6": "Single feature + last-year feature minus single",
}
ARM_LABELS: Final[dict[str, str]] = {
    "a2": "Single feature",
    "a3": "Three features",
    "a7": "Shuffled single feature",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """The contrasts, one panel per technology."""
    contrasts = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").with_columns(
        label=pl.col("contrast") + ": " + pl.col("contrast").replace_strict(CONTRAST_LABELS)
    )
    panels = []
    for technology in ("solar", "wind"):
        part = contrasts.filter(pl.col("technology") == technology).sort("contrast")
        labels = part["label"].to_list()
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(color="#292b2b", strokeDash=[4, 3])
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        base = alt.Chart(part).encode(
            y=alt.Y("label:N", sort=labels, title=None, axis=alt.Axis(labelLimit=330)),
            color=alt.condition(
                "datum.contrast == 'C1' || datum.contrast == 'C2'",
                alt.value(ocf.BRAND_ORANGE),
                alt.value(ocf.DATA_BLUE),
            ),
        )
        points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("difference:Q", title="CRPS difference (points of capacity; negative = better)")
        )
        bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bars, points).properties(
                width=PLOT_WIDTH, height=24 * part.height, title=f"{technology.title()} farms"
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="A recent effective-capacity feature does not lower the CRPS detectably",
            lines=[
                SCOPE,
                (
                    "Dot: estimate. Line: 98.75% interval from re-drawing whole months and seeds. "
                    "Orange: the two contrasts that decide the verdict. Black dashed line: no "
                    "change."
                ),
            ],
        )
    )


def figure_2() -> alt.TopLevelMixin:
    """Each farm's CRPS difference against the arm without the feature."""
    farms = pl.read_parquet(RESULTS_DIR / "per_farm.parquet").with_columns(
        label=pl.col("arm").replace_strict(ARM_LABELS)
    )
    labels = list(ARM_LABELS.values())
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#292b2b", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    points = (
        alt.Chart(farms)
        .mark_point(filled=True, size=60)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="CRPS minus the no-feature arm's (points of capacity; negative = better)",
            ),
            y=alt.Y("site:N", title="Farm"),
            color=alt.Color(
                "label:N",
                sort=labels,
                scale=alt.Scale(domain=labels, range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, "#5a5a5a"]),
                legend=alt.Legend(title=None, orient="top", columns=3),
            ),
        )
    )
    return alt.layer(zero, points).properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            text="One solar farm gains clearly; solar farm A and wind farm W3 lose",
            lines=[
                SCOPE,
                "Each dot is one farm's mean CRPS difference against the no-feature arm, 3 seeds.",
            ],
        ),
    )


def null_shares() -> pl.DataFrame:
    """Return each feature's null share over the validation year, for each series."""
    sites = site_table()
    start = int(features.day_index(times=np.array(["2025-07-01"], dtype="datetime64[us]"))[0])
    end = int(features.day_index(times=np.array(["2026-07-01"], dtype="datetime64[us]"))[0])
    rows = []
    for row in sites.iter_rows(named=True):
        table = np.load(WORK_DIR / "tables" / f"{row['time_series_id']}.npy")[start:end]
        for name, share in zip(
            features.FEATURE_NAMES[row["technology"]], np.isnan(table).mean(axis=0), strict=True
        ):
            rows.append({"site": row["site"], "technology": row["technology"], "feature": name})
            rows[-1]["null_share"] = float(share)
    return pl.DataFrame(rows)


def main() -> None:
    """Save both figures and print the null shares."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    figure_1().save(CHART_DIR / "figure_1.svg")
    figure_2().save(CHART_DIR / "figure_2.svg")
    shares = null_shares()
    print(
        shares.group_by("technology", "feature", maintain_order=True).agg(
            mean=pl.col("null_share").mean().round(3), max=pl.col("null_share").max().round(3)
        )
    )


if __name__ == "__main__":
    main()
