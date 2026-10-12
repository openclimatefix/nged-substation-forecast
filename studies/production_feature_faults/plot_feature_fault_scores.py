"""Plot paired forecast effects and descriptive gain shares from the completed study reports."""

import argparse
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
import vl_convert as vlc
from plot_production_features import FEATURE_LABELS

TEXT: Final[dict[str, tuple[str, str]]] = {
    "scores": (
        "Forecast effects of candidate feature corrections",
        (
            "31 series; July 2025–June 2026 weekly runs; 51 weather members. Three fitting seeds. "
            "Equal series weight; changes in capacity percentage points, negative improves. "
            "Bars: 95% whole-month bootstrap intervals, with one seed shared by compared arms. "
            "Each conditional contribution compares all corrections with one correction removed; "
            "contributions are not additive. Corrections: short-wave reconstruction, "
            "wind-direction wrap, and guarded windchill. Masking makes precipitation type "
            "or ordinal weekday missing. CRPS: fair continuous ranked probability score. "
            "MAE: mean absolute error of the ensemble mean."
        ),
    ),
    "subgroups": (
        "Combined corrections: effects by series type and forecast lead",
        (
            "July 2025–June 2026 weekly runs; three fitting seeds, 51 weather members. "
            "All corrections minus baseline; fair continuous ranked probability score (CRPS), "
            "in capacity percentage points. "
            "Equal series weight within each group; 95% month-and-seed bootstrap intervals. "
            "Exploratory subgroup comparisons; negative improves. Lead is time from power "
            "forecast issuance to its forecasted time. Corrections: radiation reconstruction, "
            "wind-direction wrap, and guarded windchill."
        ),
    ),
    "importance": (
        "Training gain describes model usage, not forecast benefit",
        (
            "Controlled setting; April 2024–June 2025 training; 31 per-series GPU models. "
            "Total gain normalised within each model, then averaged over series and three seeds. "
            "Bars: minimum to maximum of the three seed means. Zero includes unused features. "
            "Correlation can redistribute gain. Paired forecast comparisons measure "
            "changes in forecast error."
        ),
    ),
}
SETTINGS: Final[list[str]] = ["Production shape, column subsampling 1", "More regularised"]
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE]


def forest(data: pl.DataFrame, *, labels: str, title: str) -> alt.LayerChart:
    """Show signed effects with bootstrap intervals and redundant setting markers."""
    data = data.with_columns(
        setting_label=pl.col("setting").replace(
            {"controlled": SETTINGS[0], "sensitivity": SETTINGS[1]}
        )
    )
    base = alt.Chart(data).encode(
        y=alt.Y(f"{labels}:N", title=None, axis=alt.Axis(labelLimit=400)),
        yOffset=alt.YOffset("setting_label:N"),
        color=alt.Color(
            "setting_label:N", scale=alt.Scale(domain=SETTINGS, range=COLOURS), title=None
        ),
    )
    bars = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower_95:Q", title=title, scale=alt.Scale(zero=False)),
        x2="upper_95:Q",
    )
    dots = base.mark_point(filled=True, size=65).encode(  # ty: ignore[unresolved-attribute]
        x="difference:Q",
        shape=alt.Shape("setting_label:N", legend=None),
    )
    zero = (
        alt.Chart(pl.DataFrame({"zero": [0]}))
        .mark_rule(color="#555555")
        .encode(  # ty: ignore[unresolved-attribute]
            x="zero:Q",
        )
    )
    return (zero + bars + dots).properties(width=520, height=30 * data[labels].n_unique())


def importance_chart(directory: Path) -> alt.LayerChart:
    """Average model gain over series before showing the range across fitting seeds."""
    parts = [
        pl.read_parquet(path)
        for path in sorted((directory / "scores").glob("controlled_*_importance.parquet"))
    ]
    gains = pl.concat(parts).filter(pl.col("arm").is_in(["baseline", "all_fixed"]))
    checks = gains.group_by("arm", "seed", "feature").len()
    if set(gains["arm"]) != {"baseline", "all_fixed"} or set(gains["seed"]) != {0, 1, 2}:
        raise ValueError("importance requires both arms and all three seeds")
    if checks["len"].min() != 31 or checks["len"].max() != 31 or checks.height != 2 * 3 * 24:
        raise ValueError("importance requires exactly 31 models for every feature, arm, and seed")
    seeds = gains.group_by("arm", "seed", "feature").agg(pl.col("share").mean())
    summary = (
        seeds.group_by("arm", "feature")
        .agg(
            mean=100 * pl.col("share").mean(),
            low=100 * pl.col("share").min(),
            high=100 * pl.col("share").max(),
        )
        .with_columns(
            feature=pl.col("feature").replace(FEATURE_LABELS),
            arm=pl.col("arm").replace({"baseline": "Baseline", "all_fixed": "All corrections"}),
        )
    )
    base = alt.Chart(summary).encode(
        y=alt.Y(
            "feature:N",
            title=None,
            sort=alt.SortField("mean", order="descending"),
            axis=alt.Axis(labelLimit=330),
        ),
        yOffset="arm:N",
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=["Baseline", "All corrections"], range=COLOURS),
            title=None,
        ),
    )
    bars = base.mark_rule().encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dots = base.mark_point(filled=True).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("mean:Q", title="Mean share of training total gain (%)"),
        shape=alt.Shape("arm:N", legend=None),
    )
    return (bars + dots).properties(width=520, height=650)


def main() -> None:
    """Emit figure text for review, or render the reviewed score and importance figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png-dir", type=Path, required=True)
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    args.png_dir.mkdir(parents=True, exist_ok=True)
    (args.png_dir / "score_figure_text.txt").write_text(
        "\n\n".join(f"{key}\n{title}\n{subtitle}" for key, (title, subtitle) in TEXT.items()) + "\n"
    )
    if args.text_only:
        return
    data = pl.read_parquet(args.report / "contrasts.parquet").filter(
        pl.col("setting").is_in(["controlled", "sensitivity"])
    )
    overall = data.filter(pl.col("scope") == "All types, all leads")
    scores = alt.vconcat(
        *[
            forest(
                overall.filter(pl.col("metric") == metric),
                labels="contrast",
                title=("Fair CRPS" if metric == "crps" else "Ensemble-mean MAE")
                + " change (percentage points of capacity)",
            )
            for metric in ("crps", "mae")
        ]
    )
    groups = data.filter(
        (pl.col("scope") != "All types, all leads")
        & (pl.col("contrast") == "All corrections (planned)")
        & (pl.col("metric") == "crps")
    )
    plots = {
        "scores": scores,
        "subgroups": forest(
            groups, labels="scope", title="Fair CRPS change (percentage points of capacity)"
        ),
        "importance": importance_chart(args.fits),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for key, plot in plots.items():
        title, subtitle = TEXT[key]
        chart = plot.properties(
            title=alt.Title(
                textwrap.wrap(title, 65),
                subtitle=textwrap.wrap(subtitle, 105),
                anchor="start",
                fontSize=18,
                subtitleFontSize=11,
            )
        ).configure_legend(orient="top", labelLimit=400)
        spec = chart.to_dict()
        (args.output / f"production_feature_faults_{key}.svg").write_text(vlc.vegalite_to_svg(spec))
        (args.png_dir / f"{key}.png").write_bytes(vlc.vegalite_to_png(spec, scale=2))
        print(f"rendered {key}", flush=True)


if __name__ == "__main__":
    main()
