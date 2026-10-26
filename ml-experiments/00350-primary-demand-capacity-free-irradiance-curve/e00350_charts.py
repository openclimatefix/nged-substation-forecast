# ruff: noqa: E501
"""Draw experiment 350's figures as SVG and print every number the page quotes beyond the report.

Reads `report.json` (written by `e00350_score.py`), the arms' stored predictions, and the fitted
curves. Primaries are the anonymised labels D5 to D16. Optimise afterwards with
`npx svgo@4 --multipass --precision=1 --final-newline f.svg`.

    PYTHONPATH=ml-experiments/00136-dp-bg-stage1-assembly-on-the-ensemble-mean:ml-experiments/00155-dp-bg-stage2-assembly-night-offset-and-error-correction:ml-experiments/00317-primary-demand-default-1-at-7-to-10-days:ml-experiments/00350-primary-demand-capacity-free-irradiance-curve \
        uv run --no-sync python ml-experiments/00350-primary-demand-capacity-free-irradiance-curve/e00350_charts.py
"""

import json
import pickle
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00136_common as c136  # ty: ignore[unresolved-import]
import e00317_score as s317  # ty: ignore[unresolved-import]
import e00350_common as common
import e00350_curve as curve
import e00350_rows as rows_module
import e00350_score as score
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.dp_gam import derated_unit_shape
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

OUT: Final[Path] = Path("docs/ml-experiments/00350-primary-demand-capacity-free-irradiance-curve")
WIDTH: Final[int] = 520
WRAP: Final[int] = 95
SCOPE: Final[str] = (
    "Six Primaries (D5, D9, D10, D11, D14, D16), 2025-07 to 2026-06, sunny daylight half-hours "
    "(sun above 5 degrees, ensemble-mean shortwave in the top quarter of the Primary's daylight half-hours)."
)
ARM_COLOURS: Final[dict[str, str]] = {
    "B (shared curve)": ocf.BRAND_ORANGE,
    "C (curve per Primary)": ocf.DATA_BLUE,
    "A (pinned capacity)": ocf.DATA_PURPLE,
}
GRID: Final[np.ndarray] = np.linspace(0.0, 1000.0, 41)


def title(*, text: str, lines: list[str]) -> alt.TitleParams:
    """Return a wrapped title."""
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def zero_rules(*, with_threshold: bool) -> alt.LayerChart:
    """Return the dashed rule at zero, and optionally the +0.05 non-inferiority bound."""
    values = [0.0, 0.05] if with_threshold else [0.0]
    return (
        alt.Chart(pl.DataFrame({"x": values}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#555555")
        .encode(x="x:Q")
    )


def figure_1(*, report: dict) -> None:
    """Headline: B - A and C - A on sunny daylight half-hours at the two lead ranges."""
    rows = []
    for lead in ("0.5-240 h", "24-240 h"):
        for pair, label in (("B-A", "B (shared curve)"), ("C-A", "C (curve per Primary)")):
            c = report["contrasts"][f"sunny daylight|{lead}|{pair}"]
            rows.append(
                {
                    "row": f"{label}, lead {lead}",
                    "arm": label,
                    "mean": c["diff"],
                    "lo": c["lo95"],
                    "hi": c["hi95"],
                }
            )
            print(f"fig1 {label} {lead}: {c['diff']:+.3f} [{c['lo95']:+.3f}, {c['hi95']:+.3f}]")
    frame = pl.DataFrame(rows)
    order = frame["row"].to_list()
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(
            domain=["B (shared curve)", "C (curve per Primary)"],
            range=[ARM_COLOURS["B (shared curve)"], ARM_COLOURS["C (curve per Primary)"]],
        ),
        legend=None,
    )
    y = alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=400))
    bars = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=4)
        .encode(
            y=y,
            x=alt.X(
                "lo:Q",
                title="Quantile score minus arm A's (points of capacity; negative is better)",
            ),
            x2="hi:Q",
            color=colour,
        )
    )
    dots = alt.Chart(frame).mark_point(filled=True, size=70).encode(y=y, x="mean:Q", color=colour)  # ty: ignore[unresolved-attribute]
    chart = alt.layer(bars, dots, zero_rules(with_threshold=True)).properties(
        width=WIDTH - 140,
        height=150,
        title=title(
            text="A learned irradiance curve against the pinned capacity: no case for dropping the capacity",
            lines=[
                SCOPE,
                "Dot: estimate. Line: 95% interval from resampling calendar months. Dashed lines: same as arm A (0) and the +0.05 bound the simplicity case needs the upper end to stay under.",
            ],
        ),
    )
    chart.save(OUT / "figure_1_headline.svg")


def figure_2(*, report: dict) -> None:
    """Per-Primary B - A and C - A on its own rows, sunny daylight, 24 to 240 hours."""
    rows = []
    for site in common.SCORED_SITES:
        entry = report["leave_one_out"][f"{site}|24-240 h"]
        for key, label in (("B_alone", "B (shared curve)"), ("C_alone", "C (curve per Primary)")):
            c = entry[key]
            rows.append(
                {"site": site, "arm": label, "mean": c["diff"], "lo": c["lo95"], "hi": c["hi95"]}
            )
            print(f"fig2 {site} {label}: {c['diff']:+.3f} [{c['lo95']:+.3f}, {c['hi95']:+.3f}]")
    frame = pl.DataFrame(rows)
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(
            domain=["B (shared curve)", "C (curve per Primary)"],
            range=[ARM_COLOURS["B (shared curve)"], ARM_COLOURS["C (curve per Primary)"]],
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    y = alt.Y("site:N", sort=list(common.SCORED_SITES), title=None)
    offset = alt.YOffset("arm:N")
    bars = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=3)
        .encode(
            y=y,
            yOffset=offset,
            x=alt.X(
                "lo:Q",
                title="Quantile score minus arm A's (points of capacity; negative is better)",
            ),
            x2="hi:Q",
            color=colour,
        )
    )
    dots = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60)
        .encode(y=y, yOffset=offset, x="mean:Q", color=colour)
    )
    chart = alt.layer(bars, dots, zero_rules(with_threshold=False)).properties(
        width=WIDTH,
        height=230,
        title=title(
            text="The whole gain is at one Primary, D5",
            lines=[
                "Each Primary scored alone, sunny daylight half-hours, lead 24 to 240 hours. Line: 95% interval from resampling calendar months. Dashed line: same as arm A.",
            ],
        ),
    )
    chart.save(OUT / "figure_2_per_primary.svg")


def joined_rows() -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return the scored rows that have both references, and each arm's quantiles on them."""
    cache = c136.load_cache()
    arms = {arm: rows_module.load_arm(arm=arm, period="scored") for arm in common.ARMS}
    base = rows_module.add_sun_flags(frame=arms["A"], cache=cache)
    manual = s317.read_reference(name="ref_manual_heuristic", tag="mh")
    clim = s317.read_reference(name="ref_climatology", tag="cl")
    frame = base.with_row_index("row")
    joined = frame.join(manual, on=s317.common.KEYS, how="left", validate="1:1").join(
        clim, on=s317.common.KEYS, how="left", validate="1:1"
    )
    keep = (joined["mh_q50"].is_not_null() & joined["cl_q50"].is_not_null()).to_numpy()
    index = joined["row"].to_numpy()[keep]
    joined = joined.filter(pl.Series(keep))
    quantiles = {
        arm: arms[arm].select(list(common.QUANTILES)).to_numpy().astype(np.float64)[index]
        for arm in common.ARMS
    }
    print(f"joined_rows: {joined.height:,} rows with both references")
    return joined, quantiles


def month_rows(*, joined: pl.DataFrame, quantiles: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return the rows with each arm's quantile score and the contrasts against arm A."""
    actual = joined["actual"].to_numpy().astype(np.float64)
    columns = {
        f"qs_{arm}": 2.0 * score.pin(actual=actual, q=quantiles[arm]).mean(axis=1)
        for arm in common.ARMS
    }
    out = joined.select(
        "site", "power_fcst_init_time", "lead_hours", "sunny_daylight"
    ).with_columns(**{k: pl.Series(v) for k, v in columns.items()})
    return out.with_columns(
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        b_minus_a=pl.col("qs_B") - pl.col("qs_A"),
    )


def figure_3(*, rows: pl.DataFrame) -> None:
    """B - A by initialisation month on sunny daylight half-hours at lead 24 hours or more."""
    sunny = rows.filter(pl.col("sunny_daylight"), pl.col("lead_hours") >= 24.0)
    frames = []
    for label, part in (
        ("All six Primaries", sunny),
        ("Without D5", sunny.filter(pl.col("site") != "D5")),
    ):
        grouped = (
            part.group_by("month")
            .agg(n=pl.len(), diff=pl.col("b_minus_a").mean())
            .sort("month")
            .with_columns(series=pl.lit(label))
        )
        frames.append(grouped)
        for row in grouped.iter_rows(named=True):
            print(f"fig3 {label} {row['month']}: B-A {row['diff']:+.3f} (n={row['n']:,})")
    frame = pl.concat(frames)
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("month:N", title="Month the forecast was issued"),
            y=alt.Y(
                "diff:Q",
                title="Arm B minus arm A (points of capacity; negative is better)",
                scale=alt.Scale(domain=[-1.0, 1.0]),
            ),
            xOffset="series:N",
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["All six Primaries", "Without D5"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=title(
                text="The curve's gain changes sign from month to month",
                lines=[
                    "Arm B minus arm A on sunny daylight half-hours, lead 24 to 240 hours, by the month the forecast was issued. November to January have no sunny rows in the scored data.",
                ],
            ),
        )
    )
    chart.save(OUT / "figure_3_by_month.svg")


def tails(*, joined: pl.DataFrame, quantiles: dict[str, np.ndarray]) -> None:
    """Print coverage and pinball at q01, q05, q95, and q99 for arms A, B, and C."""
    actual = joined["actual"].to_numpy().astype(np.float64)
    lead = joined["lead_hours"].to_numpy()
    sunny = joined["sunny_daylight"].to_numpy()
    for name, mask in (("all rows", lead >= 24.0), ("sunny daylight", sunny & (lead >= 24.0))):
        for arm in common.ARMS:
            q = quantiles[arm][mask]
            a = actual[mask]
            pinball = score.pin(actual=a, q=q).mean(axis=0)
            parts = []
            for level, column in ((1, 0), (5, 1), (95, 9), (99, 10)):
                cover = float((a <= q[:, column]).mean())
                parts.append(f"q{level:02d} coverage {cover:.4f} pinball {pinball[column]:.4f}")
            print(f"tails {name}, 24-240 h, arm {arm}: " + "; ".join(parts))


def figure_4() -> None:
    """Fitted curves of arms B and C beside the pinned shape, per Primary."""
    cache = c136.load_cache()
    vintages = read_vintages()
    rows = []
    for site in common.SCORED_SITES:
        features = cache.sites[site]
        last = pickle.loads((common.FITS_DIR / "A" / f"{site}.pkl").read_bytes())[-1]
        pinned = pinned_capacity(vintages=vintages, table=features.table, cut=last.cut)
        index = np.flatnonzero(
            np.isfinite(features.ghi_w_m2)
            & np.isfinite(features.pv_unit_shape)
            & np.isfinite(features.temperature_c)
            & (features.elevation_deg > 5.0)
        )
        shape = derated_unit_shape(features=features, index=index)
        irradiance = features.ghi_w_m2[index]
        bins = np.digitize(irradiance, GRID)
        median_shape = np.array(
            [
                np.median(shape[bins == k]) if (bins == k).sum() >= 20 else np.nan
                for k in range(1, len(GRID) + 1)
            ]
        )
        a_curve = pinned.held_points * median_shape
        top = int(np.flatnonzero(np.isfinite(a_curve))[-1])
        print(
            f"fig4 {site}: pinned capacity {pinned.held_points:.1f} points; arm A at {GRID[top]:.0f} W/m2 {a_curve[top]:.1f}"
        )
        for k, g in enumerate(GRID):
            if np.isfinite(a_curve[k]):
                rows.append(
                    {
                        "site": site,
                        "arm": "A (pinned capacity)",
                        "irradiance": g,
                        "points": float(a_curve[k]),
                    }
                )
        for arm, label in (("B", "B (shared curve)"), ("C", "C (curve per Primary)")):
            fits = pickle.loads((common.FITS_DIR / arm / f"{site}.pkl").read_bytes())
            theta = fits[-1].theta[-curve.N_RAMPS :]
            values = curve.ramps(ghi_w_m2=GRID) @ theta
            print(f"fig4 {site} {label}: 1000 W/m2 {values[-1]:.1f}")
            rows.extend(
                {"site": site, "arm": label, "irradiance": float(g), "points": float(v)}
                for g, v in zip(GRID, values, strict=True)
            )
    frame = pl.DataFrame(rows)
    line = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("irradiance:Q", title="Irradiance (W/m2)"),
            y=alt.Y("points:Q", title=None),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=150, height=130)
        .facet(facet=alt.Facet("site:N", sort=list(common.SCORED_SITES), title=None), columns=3)
        .properties(
            title=title(
                text="The fitted curves, and the pinned capacity times the unit solar shape",
                lines=[
                    "Vertical axis: points of capacity subtracted from demand. Fits at the last monthly cut (2026-06-01). Arm A: the pinned capacity times the median temperature-derated unit shape in each irradiance bin.",
                ],
            )
        )
    )
    chart = line
    chart.save(OUT / "figure_4_curves.svg")


def main() -> None:
    """Draw the four figures and print the extra numbers."""
    report = json.loads((common.REPORT_DIR / "report.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    figure_1(report=report)
    figure_2(report=report)
    joined, quantiles = joined_rows()
    figure_3(rows=month_rows(joined=joined, quantiles=quantiles))
    figure_4()
    tails(joined=joined, quantiles=quantiles)


if __name__ == "__main__":
    main()
