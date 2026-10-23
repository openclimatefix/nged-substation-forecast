"""Draw experiment 00285's figures as SVG and print the numbers the write-up quotes.

Reads `work/score.json` and `work/rows.parquet`. Afterwards optimise with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python e00285_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00285_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00285-wind-effective-wind-adapters"
)
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
GREY: Final[str] = "#999999"
SCOPE: Final[str] = (
    "3 wind farms, 53 Tuesday runs, 3 seeds, July 2025 to June 2026, rows kept by the mask."
)
PART_A_LABELS: Final[dict[str, str]] = {
    "era5_X": "XGBoost (the target to match)",
    "era5_A2": "A2 shear term",
    "era5_A3": "A3 direction sectors",
    "era5_A23": "A2 and A3 together",
    "era5_A4": "A4 neural adapter",
    "era5_A5": "A5 neural adapter, no hour of day",
    "era5_A6": "A6 blind adapter (100 m speed only)",
}
PART_B_LABELS: Final[dict[str, str]] = {
    "B1": "B1 mapping by lead day",
    "B2": "B2 mapping by lead day and sector",
    "B3": "B3 neural mapping",
    "B4": "B4 offset table",
    "B5_B1": "B5 offset table after B1",
    "B5_B2": "B5 offset table after B2",
    "B5_B3": "B5 offset table after B3",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _intervals(
    *,
    data: pl.DataFrame,
    order: list[str],
    colour: str,
    xtitle: str,
    height: int,
    title: alt.TitleParams,
) -> alt.TopLevelMixin:
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=280))
    )
    points = base.mark_point(filled=True, size=55, color=colour).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("estimate:Q", title=xtitle)
    )
    bars = base.mark_rule(color=colour).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(width=WIDTH, height=height, title=title)


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """Part A: gain over the physics core of each adapter and of XGBoost, kept rows."""
    rows = []
    for key, label in PART_A_LABELS.items():
        estimate, lower, upper = score["contrasts"][f"era5_P - {key}"]["kept | qs"]
        rows.append({"arm": label, "estimate": estimate, "lower": lower, "upper": upper})
    return _intervals(
        data=pl.DataFrame(rows),
        order=list(PART_A_LABELS.values()),
        colour=ocf.DATA_BLUE,
        xtitle="Core score minus the arm's (points of capacity; positive: arm better)",
        height=190,
        title=_title(
            text="A neural wind-speed adapter matches XGBoost's lead over the physics core",
            lines=[
                "Gain over the physics core with ERA5 as the weather. Grey line: same as the "
                "core. Dot: estimate. Line: 95% interval from re-drawing whole months and seeds. "
                + SCOPE
            ],
        ),
    )


def figure_2(*, rows: pl.DataFrame) -> alt.TopLevelMixin:
    """Part A: gain over the physics core by 100 m wind speed bin."""
    names = {"era5_X": "XGBoost", "era5_A23": "A2 and A3", "era5_A4": "A4 neural adapter"}
    kept = rows.filter(~pl.col("masked")).with_columns(
        speed_bin=pl.col("speed_100m").floor().clip(0, 20).cast(pl.Int32)
    )
    pieces = []
    for key, label in names.items():
        pieces.append(
            kept.group_by("speed_bin")
            .agg(
                n=pl.len(),
                gain=(pl.col("qs__era5_P") - pl.col(f"qs__{key}")).mean(),
            )
            .with_columns(arm=pl.lit(label))
            .filter(pl.col("n") >= 100)
        )
    data = pl.concat(pieces)
    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("speed_bin:Q", title="ERA5 100 m wind speed (m/s, 1 m/s bins)"),
            y=alt.Y("gain:Q", title="Gain over the physics core (points of capacity)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(names.values()),
                    range=[ocf.BRAND_ORANGE, ocf.DATA_GREEN, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=GREY).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, chart).properties(
        width=WIDTH,
        height=230,
        title=_title(
            text="The gain sits at wind speeds where the power curve is steep",
            lines=[
                "Mean of the physics core's quantile score minus the arm's, positive: arm better. "
                "Grey line: same as the core. " + SCOPE
            ],
        ),
    )


def figure_3(*, score: dict, model: str) -> alt.TopLevelMixin:
    """Part B: gain over raw ensemble members for each correction, kept rows."""
    rows = []
    for key, label in PART_B_LABELS.items():
        found = score["contrasts"].get(f"{model}_B0 - {model}_{key}")
        if found is None:
            continue
        estimate, lower, upper = found["kept | qs"]
        rows.append({"arm": label, "estimate": estimate, "lower": lower, "upper": upper})
    name = "physics core" if model == "core" else "XGBoost"
    return _intervals(
        data=pl.DataFrame(rows),
        order=list(PART_B_LABELS.values()),
        colour=ocf.BRAND_ORANGE if model == "core" else ocf.DATA_BLUE,
        xtitle="Raw-member score minus the arm's (points of capacity; positive: arm better)",
        height=190,
        title=_title(
            text=f"Gain over raw ensemble members, {name}",
            lines=[
                "51 ensemble members as the weather. Grey line: same as raw members. Dot: "
                "estimate. Line: 95% interval from re-drawing whole months and seeds. " + SCOPE
            ],
        ),
    )


def figure_4(*, score: dict) -> alt.TopLevelMixin:
    """Part B: share of observed power below the median forecast, per arm."""
    rows = []
    for model, model_label in (("core", "Physics core"), ("xgb", "XGBoost")):
        for arm in ("B0", "B1", "B2", "B3", "B4", "B5_B1", "B5_B2", "B5_B3"):
            entry = score["scores"].get(f"{model}_{arm}")
            if entry is None:
                continue
            label = "B0 raw members" if arm == "B0" else PART_B_LABELS[arm]
            rows.append(
                {"arm": label, "model": model_label, "share": entry["kept"]["below_median"]}
            )
    data = pl.DataFrame(rows)
    order = ["B0 raw members", *PART_B_LABELS.values()]
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=280)),
        x=alt.X(
            "share:Q",
            title="Share of observed power below the median forecast (50% is calibrated)",
            scale=alt.Scale(domain=[0.3, 0.6]),
        ),
        color=alt.Color(
            "model:N",
            scale=alt.Scale(
                domain=["Physics core", "XGBoost"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
        yOffset="model:N",
    )
    half = alt.Chart(pl.DataFrame({"x": [0.5]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(half, base.mark_point(filled=True, size=60)).properties(  # ty: ignore[invalid-argument-type]
        width=WIDTH,
        height=260,
        title=_title(
            text="Correcting the ensemble's wind raises the share of power below the median",
            lines=[
                "Share of kept scored rows whose observed power is below the arm's median "
                "forecast, over 3 seeds. Grey line: calibrated median. " + SCOPE
            ],
        ),
    )


def main() -> None:
    """Write the figures and print the quoted numbers."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    rows = pl.read_parquet(common.WORK_DIR / "rows.parquet")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1": figure_1(score=score),
        "figure_2": figure_2(rows=rows),
        "figure_3": figure_3(score=score, model="core"),
        "figure_4": figure_3(score=score, model="xgb"),
        "figure_5": figure_4(score=score),
    }
    for name, chart in figures.items():
        chart.save(DOCS_DIR / f"{name}.svg")
        print("wrote", name)


if __name__ == "__main__":
    main()
