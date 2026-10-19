"""Draw experiment 00100's figures from the saved results, as SVG.

1. The change in coverage of the 5% to 95% band against `plain11`, on real rows with a masked lag
   and on injected cases.
2. The change in mean band width, on real masked rows, real unmasked rows, and injected cases.
3. The change in quantile score, on the same three groups.
4. Each arm's absolute coverage of the 5% to 95% band, on the same three groups.
5. Each arm's coverage of the 5% to 95% band on real masked rows by lead band.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00100-*/e00100_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00100_common as common
import plotting.ocf_theme as ocf
import polars as pl

PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
OUT_DIR: Final[Path] = common.CHART_DIR
SCOPE: Final[str] = (
    "16 Primary substations, 292 daily forecast runs, July 2025 to June 2026. Real rows are "
    "scored only where the target lies outside every labelled event and logged closure."
)
ARM_LABELS: Final[dict[str, str]] = {
    "plain11": "plain11: all rows, no masking",
    "mask": "mask: events removed, lags masked",
    "drop10": "drop10: mask + dropout, 10% of rows",
    "drop30": "drop30: mask + dropout, 30% of rows",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "plain11": "#555555",
    "mask": ocf.DATA_BLUE,
    "drop10": ocf.BRAND_ORANGE,
    "drop30": ocf.DATA_PURPLE,
}
GROUP_LABELS: Final[dict[str, str]] = {
    "masked": "Real rows with a masked lag (37,394)",
    "unmasked": "Real rows without a masked lag",
    "injected": "Injected transfers (family A)",
}
INJECTED_ARMS: Final[dict[str, str]] = {
    "mask_inj": "mask",
    "drop10_inj": "drop10",
    "drop30_inj": "drop30",
    "plain": "plain11",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _rule(
    *, value: float, colour: str = "#999999", domain: tuple[float, float] | None = None
) -> alt.Chart:
    scale = alt.Scale(domain=list(domain)) if domain else alt.Scale()
    return (
        alt.Chart(pl.DataFrame({"x": [value]}))
        .mark_rule(color=colour)
        .encode(x=alt.X("x:Q", scale=scale))  # ty: ignore[unresolved-attribute]
    )


def _real(*, metric: str, subset: str) -> pl.DataFrame:
    """Return the real-row contrasts of each arm against `plain11` for a metric and subset."""
    return (
        pl.read_parquet(common.RESULTS_DIR / "real_contrasts.parquet")
        .filter(
            (pl.col("reference") == "plain11")
            & (pl.col("metric") == metric)
            & (pl.col("subset") == subset)
            & (pl.col("scope") == "all leads")
            & (pl.col("level_label") == "95")
            & pl.col("id").str.starts_with("D")
        )
        .with_columns(group=pl.lit(subset), arm=pl.col("treatment"))
    )


def _injected(*, metric: str) -> pl.DataFrame:
    """Return the injected-case contrasts of each arm against `plain11` on contaminated lags."""
    return (
        pl.read_parquet(common.RESULTS_DIR / "injected_contrasts.parquet")
        .filter(
            (pl.col("reference") == "plain")
            & (pl.col("treatment").is_in(["mask_inj", "drop10_inj", "drop30_inj"]))
            & (pl.col("metric") == metric)
            & (pl.col("family") == "A")
            & (pl.col("target") == "all")
            & (pl.col("level_label") == "95")
        )
        .with_columns(
            group=pl.lit("injected"), arm=pl.col("treatment").replace_strict(INJECTED_ARMS)
        )
    )


def _forest(
    *, data: pl.DataFrame, groups: list[str], axis_title: str, zero: float
) -> alt.VConcatChart:
    panels = []
    colours = alt.Scale(
        domain=list(ARM_LABELS)[1:], range=[ARM_COLOURS[arm] for arm in list(ARM_LABELS)[1:]]
    )
    for group in groups:
        part = data.filter(pl.col("group") == group).with_columns(
            label=pl.col("arm").replace_strict(ARM_LABELS)
        )
        base = alt.Chart(part).encode(
            y=alt.Y(
                "label:N",
                sort=[ARM_LABELS[arm] for arm in list(ARM_LABELS)[1:]],
                title=None,
                axis=alt.Axis(labelLimit=300),
            ),
            color=alt.Color("arm:N", scale=colours, legend=None),
        )
        panels.append(
            alt.layer(
                _rule(value=zero),
                base.mark_rule().encode(x="lower:Q", x2="upper:Q"),  # ty: ignore[unresolved-attribute]
                base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X("difference:Q", title=axis_title)
                ),
            ).properties(width=PLOT_WIDTH, height=70, title=GROUP_LABELS[group])
        )
    return alt.vconcat(*panels)


def figure_1() -> alt.TopLevelMixin:
    """Coverage change of the 5% to 95% band."""
    data = pl.concat(
        [_real(metric="cov90", subset="masked"), _injected(metric="cov90")], how="diagonal"
    )
    return _forest(
        data=data,
        groups=["masked", "injected"],
        axis_title="Change in 5%-95% band coverage vs plain11 (nominal 0.90)",
        zero=0.0,
    ).properties(
        title=_title(
            text="Random dropout did not widen the band enough to move its coverage",
            lines=[
                SCOPE,
                "Dot: estimate. Line: 95% interval from re-drawing whole months. Zero: plain11.",
            ],
        )
    )


def figure_2() -> alt.TopLevelMixin:
    """Band width change."""
    data = pl.concat(
        [
            _real(metric="w90", subset="masked"),
            _real(metric="w90", subset="unmasked"),
            _injected(metric="w90"),
        ],
        how="diagonal",
    )
    return _forest(
        data=data,
        groups=["masked", "unmasked", "injected"],
        axis_title="Change in mean band width vs plain11 (points of capacity; positive is wider)",
        zero=0.0,
    ).properties(
        title=_title(
            text="Dropout did not widen the band where a lag was masked, and narrowed it elsewhere",
            lines=[SCOPE, "Dot: estimate. Line: 95% interval from re-drawing whole months."],
        )
    )


def figure_3() -> alt.TopLevelMixin:
    """Quantile score change."""
    data = pl.concat(
        [
            _real(metric="qs", subset="masked"),
            _real(metric="qs", subset="unmasked"),
            _injected(metric="qs"),
        ],
        how="diagonal",
    )
    return _forest(
        data=data,
        groups=["masked", "unmasked", "injected"],
        axis_title="Change in quantile score vs plain11 (points of capacity; below zero better)",
        zero=0.0,
    ).properties(
        title=_title(
            text="Masking lags cost accuracy on injected transfers; dropout recovered most of it",
            lines=[SCOPE, "Dot: estimate. Line: 95% interval from re-drawing whole months."],
        )
    )


def figure_4() -> alt.TopLevelMixin:
    """Absolute coverage of the 5% to 95% band by arm and group."""
    real = pl.read_parquet(common.RESULTS_DIR / "real_contrasts.parquet").filter(
        (pl.col("metric") == "cov90")
        & (pl.col("scope") == "all leads")
        & (pl.col("level_label") == "95")
        & (pl.col("reference") == "plain11")
        & pl.col("id").str.starts_with("D")
        & pl.col("subset").is_in(["masked", "unmasked"])
    )
    rows = [
        {"group": subset[0], "arm": "plain11", "coverage": part["reference_mean"][0]}
        for subset, part in real.group_by("subset")
    ]
    rows += [
        {"group": row["subset"], "arm": row["treatment"], "coverage": row["treatment_mean"]}
        for row in real.iter_rows(named=True)
        if row["treatment"] in ARM_LABELS
    ]
    injected = pl.read_parquet(common.RESULTS_DIR / "injected_contrasts.parquet").filter(
        (pl.col("metric") == "cov90")
        & (pl.col("family") == "A")
        & (pl.col("target") == "all")
        & (pl.col("level_label") == "95")
        & (pl.col("reference") == "plain")
        & pl.col("treatment").is_in(list(INJECTED_ARMS))
    )
    rows += [
        {
            "group": "injected",
            "arm": INJECTED_ARMS[row["treatment"]],
            "coverage": row["treatment_mean"],
        }
        for row in injected.iter_rows(named=True)
    ]
    rows.append({"group": "injected", "arm": "plain11", "coverage": injected["reference_mean"][0]})
    data = pl.DataFrame(rows).unique(["group", "arm"])
    panels = []
    for group in ("masked", "unmasked", "injected"):
        part = data.filter(pl.col("group") == group).with_columns(
            label=pl.col("arm").replace_strict(ARM_LABELS)
        )
        base = alt.Chart(part).encode(
            y=alt.Y(
                "label:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=300)
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=None,
            ),
        )
        panels.append(
            alt.layer(
                _rule(value=0.9, colour=ocf.DATA_GREEN, domain=(0.7, 1.0)),
                base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X(
                        "coverage:Q",
                        title="Share of actuals inside the 5%-95% band (nominal 0.90)",
                        scale=alt.Scale(domain=[0.7, 1.0]),
                    )
                ),
            ).properties(width=PLOT_WIDTH, height=90, title=GROUP_LABELS[group])
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="Coverage of the 5%-95% band stayed near 0.80 on real rows with a masked lag",
            lines=[SCOPE, "Dot: mean over cases. Green line: nominal 0.90."],
        )
    )


def figure_5() -> alt.TopLevelMixin:
    """Coverage on real masked rows by lead band."""
    contrasts = pl.read_parquet(common.RESULTS_DIR / "real_contrasts.parquet").filter(
        (pl.col("metric") == "cov90")
        & (pl.col("subset") == "masked")
        & (pl.col("level_label") == "95")
        & pl.col("id").str.starts_with("D")
        & (pl.col("scope") != "all leads")
    )
    rows = [
        {"scope": row["scope"], "arm": row["reference"], "coverage": row["reference_mean"]}
        for row in contrasts.filter(pl.col("reference") == "plain11").iter_rows(named=True)
    ]
    rows += [
        {"scope": row["scope"], "arm": row["treatment"], "coverage": row["treatment_mean"]}
        for row in contrasts.filter(pl.col("reference") == "plain11").iter_rows(named=True)
    ]
    rows += [
        {"scope": row["scope"], "arm": "drop30", "coverage": row["treatment_mean"]}
        for row in contrasts.filter(
            (pl.col("treatment") == "drop30") & (pl.col("reference") == "drop10")
        ).iter_rows(named=True)
    ]
    data = (
        pl.DataFrame(rows)
        .unique(["scope", "arm"])
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "scope:N",
                sort=[band[0] for band in common.LEAD_BANDS],
                title="Lead time",
            ),
            y=alt.Y(
                "coverage:Q",
                title="Share of actuals inside the 5%-95% band",
                scale=alt.Scale(domain=[0.6, 1.0], zero=False),
            ),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=list(ARM_LABELS.values()),
                    range=[ARM_COLOURS[arm] for arm in ARM_LABELS],
                ),
                legend=alt.Legend(orient="top", columns=1, title=None, labelLimit=400),
            ),
        )
    )
    return alt.layer(_rule_h(0.9), chart).properties(
        width=PLOT_WIDTH,
        height=220,
        title=_title(
            text="Coverage on rows with a masked lag fell with lead time in every arm",
            lines=[SCOPE, "Dot: mean over cases. Green line: nominal 0.90."],
        ),
    )


def _rule_h(value: float) -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"y": [value]})).mark_rule(color=ocf.DATA_GREEN).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )


def main() -> None:
    """Write the figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, draw in enumerate((figure_1, figure_2, figure_3, figure_4, figure_5), start=1):
        draw().save(OUT_DIR / f"figure_{index}.svg")
        print(f"figure {index} written")


if __name__ == "__main__":
    main()
