"""Draw the figures of experiment 119's write-up from the stored result files.

Every figure carries its own scope, unit, and direction in its title and subtitle. SVGs go to the
write-up's folder and PNGs to a scratch folder for the render check.

    uv run python ml-experiments/00119-*/e00119_charts.py
"""

import json
import math
import textwrap
from pathlib import Path
from typing import Final, cast

import altair as alt
import e00119_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
GREY: Final[str] = "#808080"
YEAR_LABELS: Final[dict[str, str]] = {
    "F1": "F1: fitted on 2024-04 to 2025-06, scored 2025-07 to 2026-06",
    "F2": "F2: fitted on 2025-07 to 2026-06, scored 2024-07 to 2025-06",
}
YEAR_COLOURS: Final[list[str]] = [ocf.BRAND_ORANGE, ocf.DATA_BLUE]
ARM_NAMES: Final[dict[str, str]] = {
    "ENC-c": "ENC-c: full patch, corrects GAM-K",
    "ENC-c0": "ENC-c0: centre cell only",
    "ENC-cps": "ENC-cps: distant site's patch",
    "ENC-cpt": "ENC-cpt: patch from another date",
    "ENC-r": "ENC-r: full patch, replaces weather",
    "ENC-r0": "ENC-r0: centre cell only",
}


def title(*, text: str, subtitle: str) -> alt.TitleParams:
    """Return a left-aligned title with a wrapped subtitle."""
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def save(chart: alt.TopLevelMixin, *, name: str) -> None:
    """Write a chart as SVG into the docs folder and PNG into the scratch folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name, flush=True)


def load(name: str) -> dict:
    """Return a stored JSON result."""
    return json.loads((common.WORK_DIR / name).read_text())


def dot_interval(
    *,
    table: pl.DataFrame,
    heading: str,
    subtitle: str,
    x_title: str,
    order: list[str],
    height_per_row: int = 22,
    label_width: int = 250,
) -> alt.TopLevelMixin:
    """Draw dots with 99% intervals per row, coloured by direction (year)."""
    colour = alt.Color(
        "year:N",
        scale=alt.Scale(domain=list(YEAR_LABELS.values()), range=YEAR_COLOURS),
        legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=420),
    )
    y = alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=label_width))
    offset = alt.YOffset("year:N", scale=alt.Scale(domain=list(YEAR_LABELS.values())))
    base = alt.Chart(table).encode(y=y, yOffset=offset, color=colour)
    rule = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "low:Q",
            scale=alt.Scale(
                domain=[
                    float(cast(float, table["low"].min())) - 0.006,
                    float(cast(float, table["high"].max())) + 0.006,
                ]
            ),
        ),
        x2="high:Q",
    )
    pad = 0.006
    domain = [
        float(cast(float, table["low"].min())) - pad,
        float(cast(float, table["high"].max())) + pad,
    ]
    dot = base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("estimate:Q", title=x_title, scale=alt.Scale(domain=domain))
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (zero + rule + dot).properties(
        width=WIDTH - label_width,
        height=height_per_row * len(order) * 2,
        title=title(text=heading, subtitle=subtitle),
    )


def contrast_rows(*, key: str, scope: str, names: dict[str, str]) -> list[dict]:
    """Return table rows of contrast estimates and intervals for both directions."""
    rows = []
    for direction, label in YEAR_LABELS.items():
        result = load(f"results_{direction}.json")["contrasts"]
        for contrast, name in names.items():
            item = result[contrast]["by_scope"][scope]
            rows.append(
                {
                    "row": name,
                    "year": label,
                    "estimate": item["difference"],
                    "low": item["lower"],
                    "high": item["upper"],
                }
            )
    return rows


def figure_headline() -> alt.TopLevelMixin:
    """The five planned contrasts, both years."""
    names = {
        "C1": "C1: ENC-c minus GAM-K (decision)",
        "C2": "C2: ENC-c minus centre cell only",
        "C3": "C3: ENC-c minus patch from another date",
        "C3b": "C3b: ENC-c minus distant site's patch",
        "C4": "C4: ENC-r minus GAM-K (labelled)",
    }
    table = pl.DataFrame(contrast_rows(key="c", scope="0.5-168 h", names=names))
    return dot_interval(
        table=table,
        order=list(names.values()),
        heading=(
            "The weather encoder lowers GAM-K's score, "
            "but the centre cell does as much as the patch"
        ),
        subtitle=(
            "Dot: difference in mean quantile score (points of capacity) over the five decision "
            "substations, leads 0.5 to 168 hours, outside switching events. Line: 99% interval "
            "from resampling whole calendar months. Dashed line: no difference. Below zero favours "
            "the first arm. All five contrasts are planned. The known-answer gate failed, so none "
            "carries a label."
        ),
        x_title=(
            "Difference in quantile score (points of capacity; more negative favours the first arm)"
        ),
    )


def figure_decomposition() -> alt.TopLevelMixin:
    """Each arm minus GAM-K, both years."""
    rows = []
    for direction, label in YEAR_LABELS.items():
        extra = load(f"extra_{direction}.json")["arm_vs_k0"]
        for arm, name in ARM_NAMES.items():
            item = extra[arm]["0.5-168 h"]
            rows.append(
                {
                    "row": name,
                    "year": label,
                    "estimate": item["difference"],
                    "low": item["lower"],
                    "high": item["upper"],
                }
            )
    order = [ARM_NAMES[a] for a in ("ENC-cpt", "ENC-c0", "ENC-cps", "ENC-c", "ENC-r0", "ENC-r")]
    return dot_interval(
        table=pl.DataFrame(rows),
        order=order,
        heading="Most of the gain survives when the patch comes from another date",
        subtitle=(
            "Dot: each arm's mean quantile score minus GAM-K's (points of capacity), five decision "
            "substations, leads 0.5 to 168 hours, outside switching events. Line: 99% interval "
            "from resampling whole calendar months. Arms reading the wrong date's weather "
            "(ENC-cpt) still gain, so part of the gain recalibrates GAM-K's quantiles and "
            "carries no information from the weather of that day."
        ),
        x_title="Arm minus GAM-K, quantile score (points of capacity; more negative is better)",
    )


def figure_scopes() -> alt.TopLevelMixin:
    """C1 by lead band, season, and day type."""
    scopes = {
        "0.5-168 h": "All leads (0.5 to 168 h)",
        "D+1 (15.5-39 h)": "Day ahead (15.5 to 39 h)",
        "0-24 h": "0 to 24 h",
        "24-48 h": "24 to 48 h",
        "48-144 h": "48 to 144 h",
        "144-168 h": "144 to 168 h",
        "winter (Nov-Mar)": "Winter (November to March)",
        "winter evening peak (16-20 UTC)": "Winter evening peak (16 to 20 UTC)",
        "summer (Apr-Oct)": "Summer (April to October)",
        "working days": "Working days",
        "weekends and bank holidays": "Weekends and bank holidays",
    }
    rows = []
    for direction, label in YEAR_LABELS.items():
        result = load(f"results_{direction}.json")["contrasts"]["C1"]["by_scope"]
        for scope, name in scopes.items():
            item = result[scope]
            rows.append(
                {
                    "row": name,
                    "year": label,
                    "estimate": item["difference"],
                    "low": item["lower"],
                    "high": item["upper"],
                }
            )
    return dot_interval(
        table=pl.DataFrame(rows),
        order=list(scopes.values()),
        heading=(
            "The gain is smaller in winter than over the whole year, and does not turn into a loss"
        ),
        subtitle=(
            "C1, ENC-c minus GAM-K: mean quantile score difference (points of capacity) over the "
            "five decision substations by lead band, season (run date's month), and day type. "
            "Line: 99% interval from resampling whole calendar months. Each direction holds one "
            "winter, so no winter row is a finding by itself."
        ),
        x_title="ENC-c minus GAM-K, quantile score (points of capacity; more negative is better)",
        height_per_row=14,
    )


def figure_hard_days() -> alt.TopLevelMixin:
    """C1 and C3 on the three hard-day subsets, both years."""
    names = {
        "coldest": "coldest 5%",
        "showery": "showery 5%",
        "largest_error": "largest errors",
    }
    rows = []
    for direction, label in YEAR_LABELS.items():
        result = load(f"results_{direction}.json")["contrasts"]
        for contrast, pair in (("C1", "ENC-c minus GAM-K"), ("C3", "ENC-c minus other date")):
            for key, name in names.items():
                item = result[contrast]["hard_days"][key]
                rows.append(
                    {
                        "row": f"{name}: {pair} ({item['n_days']} days)",
                        "year": label,
                        "estimate": item["difference"],
                        "low": item["lower"],
                        "high": item["upper"],
                    }
                )
    order = list(dict.fromkeys(row["row"] for row in rows))
    return dot_interval(
        table=pl.DataFrame(rows),
        order=order,
        heading="On the hardest days the intervals are wide and carry no label",
        subtitle=(
            "Dot: mean quantile score difference (points of capacity) on days chosen before any "
            "score was read (the number of days is in each label, from the five decision "
            "substations). Line: 99% interval from resampling 7-day blocks. Every subset has under "
            "60 independent days a year, so all rows are descriptive. The largest-error subset is "
            "chosen on GAM-K's own outcomes and flatters any arm that differs from it."
        ),
        x_title="Difference in quantile score (points of capacity; more negative favours ENC-c)",
        height_per_row=12,
        label_width=300,
    )


def figure_gate() -> alt.TopLevelMixin:
    """Share of the exactly recoverable injection recovered, run 1 against run 2."""
    run1 = json.loads((Path(__file__).resolve().parent / "gate_run1.json").read_text())[
        "injections"
    ]
    run2 = load("gate.json")["injections"]
    rows = []
    for label, runs in (("Run 1 (as first written)", run1), ("Run 2 (redefined)", run2)):
        for name, item in runs.items():
            share = item["recovered_share"]
            if math.isnan(share):  # NaN: the sparse ENC-r injection could not be sized
                continue
            rows.append(
                {
                    "injection": name,
                    "run": label,
                    "share": share,
                    "sites": item["sites_recovering_half"],
                }
            )
    table = pl.DataFrame(rows)
    order = list(dict.fromkeys(r["injection"] for r in rows))
    bars = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("injection:N", sort=order, title=None),
            yOffset="run:N",
            x=alt.X(
                "share:Q", title="Share of the exactly recoverable gain that the encoder recovered"
            ),
            color=alt.Color(
                "run:N",
                scale=alt.Scale(
                    domain=["Run 1 (as first written)", "Run 2 (redefined)"],
                    range=[ocf.DATA_BLUE_LIGHT, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    rules = (
        alt.Chart(pl.DataFrame({"x": [0.2, 0.5]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (rules + bars).properties(
        width=WIDTH - 200,
        height=24 * len(order),
        title=title(
            text=(
                "The gate's controls recovered the injected signal too, "
                "and ENC-r barely recovered it"
            ),
            subtitle=(
                "A known function of the patch was added to the target on the 2024-04 to 2025-06 "
                "fitting year (never the scored year). Bar: share of the gain an exact recovery "
                "would give, at the five decision substations. Dashed lines: the 50% an encoder "
                "must reach and the 20% a control must stay under. Controls ENC-c0 and ENC-cpt "
                "cannot see the injected signal in principle and should be near zero."
            ),
        ),
    )


def figure_tails() -> alt.TopLevelMixin:
    """Coverage at the four tail levels for K0, ENC-c, and ENC-cpt."""
    rows = []
    for direction in YEAR_LABELS:
        tails = load(f"results_{direction}.json")["tails"]
        for arm, name in (("K0", "GAM-K"), ("ENC-c", "ENC-c"), ("ENC-cpt", "ENC-cpt")):
            rows.extend(
                {
                    "level": f"q{round(item['level'] * 100):02d}",
                    "arm": name,
                    "year": direction,
                    "gap": item["gap"] * 100,
                }
                for item in tails[arm]["0.5-168 h"]
            )
    table = pl.DataFrame(rows)
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "gap:Q",
                title="Coverage minus nominal level (percentage points; zero is calibrated)",
            ),
            y=alt.Y("level:N", sort=["q01", "q05", "q95", "q99"], title=None),
            yOffset="arm:N",
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["GAM-K", "ENC-c", "ENC-cpt"],
                    range=[ocf.DATA_BLUE_LIGHT, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            shape=alt.Shape("year:N", legend=alt.Legend(title="Direction", orient="top")),
        )
        .properties(width=WIDTH - 80, height=160)
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (zero + chart).properties(
        title=title(
            text=(
                "The encoder moves the tail quantiles towards their nominal levels, "
                "and so does the wrong-date patch"
            ),
            subtitle=(
                "Share of observations at or below each tail quantile minus its nominal level "
                "(percentage points), five decision substations, leads 0.5 to 168 hours. Dots on "
                "the dashed line are calibrated. F1 and F2 are the two fitting directions."
            ),
        )
    )


def figure_month_hour() -> alt.TopLevelMixin:
    """C1 and the wrong-date placebo by calendar month and by hour band."""
    month_rows, hour_rows = [], []
    for direction in YEAR_LABELS:
        extra = load(f"extra_{direction}.json")
        for month, arms in extra["by_month"].items():
            month_rows.extend(
                {"month": month, "arm": arm, "gain": arms[arm], "year": direction}
                for arm in ("ENC-c", "ENC-cpt")
            )
        for band, arms in extra["by_hour"].items():
            hour_rows.extend(
                {"band": band, "arm": arm, "gain": arms[arm], "year": direction}
                for arm in ("ENC-c", "ENC-cpt")
            )
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=["ENC-c", "ENC-cpt"], range=[ocf.BRAND_ORANGE, ocf.DATA_PURPLE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    months = (
        alt.Chart(pl.DataFrame(month_rows).sort("month"))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("month:O", title="Calendar month of the run (both directions' scored years)"),
            y=alt.Y("gain:Q", title="Arm minus GAM-K (points; negative is better)"),
            color=colour,
        )
        .properties(width=WIDTH - 80, height=170)
    )
    hours = (
        alt.Chart(pl.DataFrame(hour_rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "band:N",
                sort=["night 0-6", "morning 6-11", "midday 11-16", "evening 16-24"],
                title="Hour of the valid time (UTC)",
            ),
            xOffset="arm:N",
            y=alt.Y("mean(gain):Q", title="Arm minus GAM-K (points; negative is better)"),
            color=colour,
        )
        .properties(width=WIDTH - 80, height=150)
    )
    return alt.vconcat(months, hours).properties(
        title=title(
            text=(
                "The gain is largest at midday and in spring, "
                "and the wrong-date patch follows the same pattern"
            ),
            subtitle=(
                "Mean quantile score difference from GAM-K (points of capacity), five decision "
                "substations. Top: by calendar month, both directions' scored years joined. "
                "Bottom: by hour of the valid time, averaged over both directions."
            ),
        )
    )


def figure_absolute() -> alt.TopLevelMixin:
    """Absolute scores by lead band for GAM-0, GAM-K, ENC-c, and ENC-cpt."""
    rows = []
    for direction in YEAR_LABELS:
        results = load(f"results_{direction}.json")["absolute"]
        extra = load(f"extra_{direction}.json")["gam0_by_lead"]
        for band in ("0-24 h", "24-48 h", "48-144 h", "144-168 h"):
            rows.append(
                {
                    "band": band,
                    "arm": "GAM-0 (no Kalman filter)",
                    "score": extra[band]["gam0"],
                    "year": direction,
                }
            )
            for arm, name in (("K0", "GAM-K"), ("ENC-c", "ENC-c"), ("ENC-cpt", "ENC-cpt")):
                rows.append(
                    {"band": band, "arm": name, "score": results[arm][band], "year": direction}
                )
    domain = ["GAM-0 (no Kalman filter)", "GAM-K", "ENC-cpt", "ENC-c"]
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "band:N",
                sort=["0-24 h", "24-48 h", "48-144 h", "144-168 h"],
                title="Lead band (hours after the forecast origin)",
            ),
            y=alt.Y(
                "score:Q",
                title="Mean quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=domain,
                    range=[ocf.DATA_SKY, ocf.DATA_BLUE_LIGHT, ocf.DATA_PURPLE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top", columns=2),
            ),
            strokeDash=alt.StrokeDash("year:N", legend=alt.Legend(title="Direction", orient="top")),
        )
        .properties(width=WIDTH - 80, height=240)
    )
    return chart.properties(
        title=title(
            text="Scores rise with lead, and the encoder's gain is small beside that rise",
            subtitle=(
                "Mean quantile score (points of capacity) by lead band on the rows outside "
                "switching events at the five decision substations. F1 and F2 are the two "
                "fitting directions (solid and dashed). GAM-0 is experiment 111's convex model "
                "without the Kalman filter, on the same rows."
            ),
        )
    )


def figure_seeds() -> alt.TopLevelMixin:
    """The five seeds' absolute scores for ENC-c and ENC-r, both directions."""
    rows = []
    for direction in YEAR_LABELS:
        result = load(f"results_{direction}.json")
        for arm in ("ENC-c", "ENC-r"):
            for seed, value in zip(
                (11, 12, 13, 14, 15), result["per_seed_absolute"][arm], strict=True
            ):
                rows.append({"arm": arm, "year": direction, "seed": str(seed), "score": value})
        rows.append(
            {
                "arm": "GAM-K",
                "year": direction,
                "seed": "none",
                "score": result["absolute"]["K0"]["0.5-168 h"],
            }
        )
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "score:Q",
                title="Mean quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("year:N", title=None),
            yOffset="arm:N",
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["GAM-K", "ENC-c", "ENC-r"],
                    range=[ocf.DATA_BLUE_LIGHT, ocf.BRAND_ORANGE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=WIDTH - 80, height=130)
    )
    return chart.properties(
        title=title(
            text="Five fitting seeds agree to within 0.01 points, far less than the gain",
            subtitle=(
                "Mean quantile score of each of the seeds 11 to 15, over the five decision "
                "substations, leads 0.5 to 168 hours. Each dot is one seed's own forecast; the "
                "headline arms average the five seeds' quantiles. GAM-K has no seed."
            ),
        )
    )


def main() -> None:
    """Draw every figure."""
    for name, function in (
        ("headline", figure_headline),
        ("decomposition", figure_decomposition),
        ("scopes", figure_scopes),
        ("hard_days", figure_hard_days),
        ("gate", figure_gate),
        ("tails", figure_tails),
        ("month_hour", figure_month_hour),
        ("absolute", figure_absolute),
        ("seeds", figure_seeds),
    ):
        save(function(), name=name)


if __name__ == "__main__":
    main()
