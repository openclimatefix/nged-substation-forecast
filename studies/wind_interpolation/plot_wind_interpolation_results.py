"""Draw the focused interpolation comparison from saved results, without fitting.

Use `--text-only` to write figure text for review before rendering. The default run writes
optimised SVGs, PNG previews, and compact numerical evidence into `docs/studies/assets/`.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import polars as pl
import vl_convert as vlc
from plotting.ocf_theme import BRAND_ORANGE, DATA_BLUE, DATA_GREEN
from studies.sources import study_dir_for
from wind_interpolation_experiment import METHODS, interpolate

OUTPUT: Final[Path] = study_dir_for(study="wind_interpolation")
ASSETS: Final[Path] = Path("docs/studies/assets")
NAMES: Final[list[str]] = ["Linear", "PCHIP", "Makima"]
COLOURS: Final[list[str]] = [BRAND_ORANGE, DATA_BLUE, DATA_GREEN]
TEXT: Final[dict[str, tuple[str, list[str]]]] = {
    "power": (
        "Does smoother component interpolation help wind-power forecasts?",
        [
            "Three UK wind farms · control forecast · July 2025–June 2026",
            "Inputs: speed and bearing at 10 m and 100 m",
            "Dots: means; bars: 95% month-and-seed intervals",
            "Error unit: percentage points of each farm's maximum training power",
        ],
    ),
    "curves": (
        "Smooth curves add assumptions between the same wind samples",
        [
            "Synthetic instantaneous wind · points: native samples; lines: interpolation",
            "u: eastward wind; v: northward wind · no between-sample truth is assumed",
        ],
    ),
    "stations": (
        "Does smoothing reduce observed station wind-speed error?",
        [
            "15 UK weather stations · control forecast · April 2024–December 2025",
            "Measured nominal 10 m wind · forecast speed averaged over the same 10 minutes",
            "Dots: means; bars: 95% whole-month intervals · post-hoc observational check",
        ],
    ),
}
CAPTIONS: Final[dict[str, str]] = {
    "power": (
        "Mean absolute error (MAE) is pooled over the matched farm/forecast rows. Day 1: leads "
        "24–47.5 h; day 7: leads 168–191.5 h. Primary: 500 depth-6 trees; sensitivity: 1,200 "
        "depth-4 trees. Each setting is fitted three times using different random seeds. "
        "Intervals are computed by resampling whole validation months and selecting one paired "
        "random-seed fit. Four comparisons are evaluated separately under each setting. Bars "
        "are nominal 95% intervals; the report also gives 98.75% intervals correcting the four "
        "comparisons within each setting. Zero: same error as linear. Lower differences are "
        "better."
    ),
    "curves": (
        "Linear is orange and solid; PCHIP is blue and dashed; Makima is green and dotted. "
        "PCHIP means piecewise cubic Hermite interpolating polynomial. Makima is modified "
        "Akima interpolation. All methods reproduce the native sample points. The smooth "
        "methods use neighbouring slopes. Speed and bearing are derived afterwards in every "
        "method."
    ),
    "stations": (
        "Mean absolute error (MAE) is pooled over matched station/forecast rows. Forecast "
        "speed is averaged over the observation's ten-minute window, from 20 to 10 minutes "
        "before its timestamp. Day 1 uses averaging windows inside leads 24–48 h; day 7 uses "
        "windows inside leads 168–192 h. The four exploratory comparisons cover two "
        "interpolation methods and two forecast days. The report also gives 98.75% intervals "
        "correcting those four comparisons. Zero means the same error as linear. Lower "
        "differences are better. No models were fitted."
    ),
}


def _export(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Optimise the SVG before rendering its PNG preview."""
    svg = ASSETS / f"wind-interpolation-{name}.svg"
    chart.save(svg)
    subprocess.run(
        ["npx", "--yes", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(svg)],
        check=True,
        capture_output=True,
    )
    svg.with_suffix(".png").write_bytes(vlc.svg_to_png(svg.read_text(), scale=1.5))


def _styled(*, chart: alt.TopLevelMixin, name: str) -> alt.TopLevelMixin:
    """Apply readable text and the reviewed title/subtitle."""
    title, subtitle = TEXT[name]
    return (
        chart.properties(
            title=alt.TitleParams(text=title, subtitle=subtitle, anchor="start", fontSize=17)
        )
        .configure_axis(labelFontSize=12, titleFontSize=12)
        .configure_legend(labelFontSize=12, titleFontSize=12, orient="top")
        .configure_view(stroke=None)
    )


def _power(*, output: Path) -> alt.LayerChart:
    """Show the four planned differences under both fixed XGBoost settings."""
    rows = pl.read_csv(output / "contrasts.csv").with_columns(
        label=pl.col("band").replace_strict({"day1": "Day 1", "day7": "Day 7"})
        + " / "
        + pl.col("method").replace_strict({"pchip": "PCHIP", "makima": "Makima"}),
        setting=pl.col("setting").replace_strict(
            {"primary": "Primary (depth 6)", "sensitivity": "Sensitivity (depth 4)"}
        ),
        method=pl.col("method").replace_strict({"pchip": "PCHIP", "makima": "Makima"}),
    )
    base = alt.Chart(rows).encode(
        y=alt.Y(
            "label:N",
            title=None,
            sort=["Day 1 / PCHIP", "Day 1 / Makima", "Day 7 / PCHIP", "Day 7 / Makima"],
        ),
        yOffset="setting:N",
        color=alt.Color(
            "method:N", title="Smooth method", scale=alt.Scale(domain=NAMES[1:], range=COLOURS[1:])
        ),
    )
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower_95:Q", title="MAE minus linear (percentage points; lower is better)"),
        x2="upper_95:Q",
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x="difference:Q",
        shape=alt.Shape(
            "setting:N", title="Tree setting", scale=alt.Scale(range=["circle", "diamond"])
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"zero": [0]}))
        .mark_rule(color="#292B2B", strokeDash=[3, 3])
        .encode(x="zero:Q")  # ty: ignore[unresolved-attribute]
    )
    return (zero + bars + points).properties(width=500, height=230)


def _curves() -> alt.VConcatChart:
    """Illustrate the assumptions using synthetic component samples."""
    x = np.arange(0, 19, 3, dtype=float)
    samples = np.array([[-2, 3, 8, 7, -3, -7, 0], [-10, -8, -3, 5, 7, 1, -6]], dtype=float)
    targets = np.arange(0, 18.05, 0.05)
    rows = []
    for method, name in zip(METHODS, NAMES, strict=True):
        values = interpolate(values=samples, x=x, targets=targets, method=method)
        for component, y in zip(
            ("Eastward component u", "Northward component v"), values, strict=True
        ):
            rows.extend(
                {"lead": float(t), "value": float(value), "component": component, "method": name}
                for t, value in zip(targets, y, strict=True)
            )
    anchors = pl.DataFrame(
        {
            "lead": np.tile(x, 2),
            "value": samples.flatten(),
            "component": np.repeat(["Eastward component u", "Northward component v"], len(x)),
        }
    )
    frame = pl.DataFrame(rows)
    panels = []
    for component in ("Eastward component u", "Northward component v"):
        line = (
            alt.Chart(frame.filter(pl.col("component") == component))
            .mark_line(strokeWidth=2.5)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "lead:Q", title="Forecast lead (hours)", axis=alt.Axis(values=[0, 6, 12, 18])
                ),
                y=alt.Y("value:Q", title="Wind component (m/s)"),
                color=alt.Color(
                    "method:N", title="Interpolation", scale=alt.Scale(domain=NAMES, range=COLOURS)
                ),
                strokeDash=alt.StrokeDash(
                    "method:N",
                    title=None,
                    scale=alt.Scale(domain=NAMES, range=[[1, 0], [6, 3], [2, 3]]),
                ),
            )
        )
        dots = (
            alt.Chart(anchors.filter(pl.col("component") == component))
            .mark_point(color="#292B2B", filled=True, size=35)
            .encode(x="lead:Q", y="value:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append((line + dots).properties(width=570, height=130, title=component))
    return alt.vconcat(*panels, spacing=18)


def _stations(*, output: Path) -> alt.LayerChart:
    """Show smooth-minus-linear speed error against measured MIDAS wind."""
    rows = pl.read_csv(output / "midas" / "contrasts.csv").with_columns(
        label=pl.col("band").replace_strict({"day1": "Day 1", "day7": "Day 7"})
        + " / "
        + pl.col("method").replace_strict({"pchip": "PCHIP", "makima": "Makima"}),
        method=pl.col("method").replace_strict({"pchip": "PCHIP", "makima": "Makima"}),
    )
    base = alt.Chart(rows).encode(
        y=alt.Y(
            "label:N",
            title=None,
            sort=["Day 1 / PCHIP", "Day 1 / Makima", "Day 7 / PCHIP", "Day 7 / Makima"],
        ),
        color=alt.Color(
            "method:N", title="Smooth method", scale=alt.Scale(domain=NAMES[1:], range=COLOURS[1:])
        ),
    )
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower_95:Q", title="Speed MAE minus linear (m/s; lower is better)"),
        x2="upper_95:Q",
    )
    points = base.mark_point(filled=True, size=70).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"zero": [0]}))
        .mark_rule(color="#292B2B", strokeDash=[3, 3])
        .encode(x="zero:Q")  # ty: ignore[unresolved-attribute]
    )
    return (zero + bars + points).properties(width=500, height=180)


def main() -> None:
    """Write reviewed figure text, or render plots and publish compact evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    text = "\n\n".join(
        f"Figure {index}: {name}\n{title}\n" + "\n".join(subtitle) + "\n" + CAPTIONS[name]
        for index, (name, (title, subtitle)) in enumerate(TEXT.items(), start=1)
    )
    review_path = Path(".claude/worktrees/review/figure-text-final.md")
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(text + "\n")
    if args.text_only:
        return
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("power", _power(output=args.output)),
        ("curves", _curves()),
        ("stations", _stations(output=args.output)),
    ):
        _export(chart=_styled(chart=chart, name=name), name=name)
    provenance = json.loads((args.output / "provenance.json").read_text())
    provenance.pop("nwp_path")
    evidence = {
        "provenance": provenance,
        "absolute_errors": pl.read_csv(args.output / "absolute.csv").to_dicts(),
        "contrasts": pl.read_csv(args.output / "contrasts.csv").to_dicts(),
        "per_site": pl.read_csv(args.output / "per-site.csv").to_dicts(),
        "withheld_native": pl.read_csv(args.output / "withheld-native.csv").to_dicts(),
        "midas": {
            "provenance": json.loads((args.output / "midas" / "provenance.json").read_text()),
            "absolute_errors": pl.read_csv(args.output / "midas" / "absolute.csv").to_dicts(),
            "contrasts": pl.read_csv(args.output / "midas" / "contrasts.csv").to_dicts(),
        },
    }
    (ASSETS / "wind-interpolation-results.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
