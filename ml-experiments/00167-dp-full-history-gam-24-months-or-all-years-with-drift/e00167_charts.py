"""Draw the figures of experiment 167's write-up from the scored predictions.

uv run --no-sync python ml-experiments/00167-*/e00167_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00167_common as common
import e00167_score as score
import plotting.ocf_theme as ocf
import polars as pl
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 680
WRAP: Final[int] = 110
GREY: Final[str] = "#808080"
ARM_LABELS: Final[dict[str, str]] = {
    "m24": "24 months, no drift",
    "all": "all years, no drift",
    "all_d1": "all years, drift",
    "m24_d1": "24 months, drift",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "24 months, no drift": GREY,
    "all years, no drift": ocf.DATA_BLUE,
    "all years, drift": ocf.BRAND_ORANGE,
    "24 months, drift": ocf.DATA_SKY,
}
SITE_LABELS: Final[tuple[str, ...]] = common.DECISION_SITES


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def figure_per_origin(*, wide: pl.DataFrame) -> alt.TopLevelMixin:
    """The three headline contrasts at each origin, with approximate 95% intervals."""
    wanted = {
        "C1": "C1: all years with drift minus 24 months",
        "C3": "C3: drift minus no drift, both on all years",
        "E2": "E2 (post hoc): all years minus 24 months, both with drift",
    }
    rows = []
    for contrast in score.CONTRASTS:
        if contrast.name not in wanted:
            continue
        table = score.per_origin(
            wide=wide, contrast=contrast, metric="score", sites=common.DECISION_SITES
        )
        rows.extend(
            {
                "contrast": wanted[contrast.name],
                "origin": row["origin"],
                "estimate": row["estimate"],
                "lower": row["estimate"] - 1.96 * row["se"],
                "upper": row["estimate"] + 1.96 * row["se"],
            }
            for row in table.iter_rows(named=True)
        )
    data = pl.DataFrame(rows)
    panels = []
    for name in wanted.values():
        base = alt.Chart(data.filter(pl.col("contrast") == name)).encode(
            y=alt.Y("origin:N", title="Refit date", sort=None)
        )
        rule = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black", strokeDash=[4, 3])
            .encode(x="x:Q")
        )
        bars = base.mark_rule(color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lower:Q",
                title="Quantile score difference (points of capacity; below 0 is better)",
            ),
            x2="upper:Q",
        )
        dots = base.mark_point(color=ocf.BRAND_ORANGE, filled=True, size=45).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            (rule + bars + dots).properties(
                width=WIDTH - 120,
                height=170,
                title=alt.TitleParams(name, anchor="start", fontSize=12),
            )
        )
    return (
        alt.vconcat(*panels)
        .properties(
            title=_title(
                text="Drift lowers the score at most refits; extra years add a small further gain",
                subtitle=(
                    "Dot: contrast at one refit, pooled over D5, D10, D11, D14, and D16. Line: "
                    "approximate 95% interval from resampling 13 week blocks. Dashed rule: no "
                    "difference. Score is twice the mean pinball loss over 11 quantiles."
                ),
            )
        )
        .configure_view(stroke=None)
    )


def figure_absolute(*, wide: pl.DataFrame) -> alt.TopLevelMixin:
    """Each arm's mean quantile score at each decision site."""
    rows = score.usable(wide=wide, sites=SITE_LABELS, origins=score.origin_labels())
    data = pl.concat(
        [
            rows.group_by("site")
            .agg(pl.col(f"score__{arm}").mean().alias("score"))
            .with_columns(arm=pl.lit(ARM_LABELS[arm]))
            for arm in score.ARMS
        ]
    )
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            y=alt.Y("site:N", title=None, sort=list(SITE_LABELS)),
            x=alt.X(
                "score:Q",
                title="Mean quantile score over 12 refits (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=2),
            ),
        )
        .properties(
            width=WIDTH - 80,
            height=190,
            title=_title(
                text="Drift lowers the score at 4 of 5 substations; D14 is the exception",
                subtitle=(
                    "Mean quantile score of each arm at each substation, over the 12 quarterly "
                    "refits, on past weather (ERA5 and CAMS), outside the switching events."
                ),
            ),
        )
    )


def figure_per_site(*, wide: pl.DataFrame) -> alt.TopLevelMixin:
    """The drift and history contrasts at each substation."""
    names = {
        "E1": "Drift, on 24 months",
        "C3": "Drift, on all years",
        "C2": "All years, no drift (vs 24 months)",
        "E2": "All years, with drift (vs 24 months)",
    }
    rows = []
    for contrast in score.CONTRASTS:
        if contrast.name not in names:
            continue
        for site in SITE_LABELS:
            value = score.pooled(
                wide=wide,
                contrast=contrast,
                metric="score",
                sites=[site],
                origins=score.origin_labels(),
            )
            rows.append({"contrast": names[contrast.name], "site": site, **value})
    data = pl.DataFrame(rows)
    colours = [GREY, ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY]
    base = alt.Chart(data).encode(
        y=alt.Y("site:N", title=None, sort=list(SITE_LABELS)),
        color=alt.Color(
            "contrast:N",
            scale=alt.Scale(domain=list(names.values()), range=colours),
            legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=2),
        ),
        yOffset="contrast:N",
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="black", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    bars = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q", title="Quantile score difference (points of capacity; below 0 is better)"
        ),
        x2="upper:Q",
    )
    dots = base.mark_point(filled=True, size=50).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
    return (rule + bars + dots).properties(
        width=WIDTH - 80,
        height=260,
        title=_title(
            text="Drift helps at every substation but D14, and D10 gains most",
            subtitle=(
                "Dot: contrast at one substation over the 12 refits. Line: 98.33% interval from "
                "resampling whole calendar months. Dashed rule: no difference."
            ),
        ),
    )


def figure_tails(*, scored: pl.DataFrame) -> alt.TopLevelMixin:
    """Coverage of the four tail quantiles by arm."""
    keep = scored.filter(pl.col("site").is_in(list(SITE_LABELS)), ~pl.col("switching_mask"))
    rows = []
    for arm in score.ARMS:
        group = keep.filter(pl.col("arm") == arm)
        actual = group["actual"]
        for column in score.TAIL_LEVELS:
            level = QUANTILE_LEVELS[QUANTILE_COLUMNS.index(column)]
            rows.append(
                {
                    "arm": ARM_LABELS[arm],
                    "level": column,
                    "nominal": level,
                    "coverage": float((actual <= group[column]).mean()),  # ty: ignore[invalid-argument-type]
                }
            )
    data = pl.DataFrame(rows)
    nominal = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_tick(color="black", thickness=2, size=30)
        .encode(y=alt.Y("level:N", title=None, sort=list(score.TAIL_LEVELS)), x="nominal:Q")
    )
    dots = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            y=alt.Y("level:N", title=None, sort=list(score.TAIL_LEVELS)),
            x=alt.X("coverage:Q", title="Share of measured values at or below the quantile"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=2),
            ),
        )
    )
    return (nominal + dots).properties(
        width=WIDTH - 80,
        height=170,
        title=_title(
            text="Every arm has tails that are too narrow; drift narrows the gap",
            subtitle=(
                "Dot: share of measured values at or below each quantile, over the 12 refits at "
                "the five substations. Black tick: the nominal share."
            ),
        ),
    )


def main() -> None:
    """Draw every figure."""
    scored = pl.read_parquet(common.WORK_DIR / "scored.parquet")
    wide = score.build_wide(scored=scored)
    _save(figure_per_origin(wide=wide), name="figure_01_per_origin")
    _save(figure_absolute(wide=wide), name="figure_02_absolute")
    _save(figure_per_site(wide=wide), name="figure_03_per_site")
    _save(figure_tails(scored=scored), name="figure_04_tails")


if __name__ == "__main__":
    main()
