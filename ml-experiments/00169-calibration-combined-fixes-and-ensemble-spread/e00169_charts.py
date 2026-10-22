"""Draw the figures of experiment 169's write-up from `work/report/report.json`.

uv run --no-sync python ml-experiments/00169-*/e00169_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00169_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
REPORT: Final[Path] = common.WORK_DIR / "report" / "report.json"
FIT: Final[Path] = common.WORK_DIR / "fit_parameters.json"
WIDTH: Final[int] = 680
GREY: Final[str] = "#808080"
ARM_COLOURS: Final[dict[str, str]] = {
    "I: smooth table": GREY,
    "A: I + cold-day widening": ocf.BRAND_ORANGE,
    "AT: A + lower tail of T": ocf.DATA_GREEN,
    "SP: A + ensemble spread": ocf.DATA_BLUE,
    "SP0: SP without spread terms": ocf.DATA_SKY,
}
ARM_KEYS: Final[dict[str, str]] = {
    "ref": "I: smooth table",
    "a": "A: I + cold-day widening",
    "at": "AT: A + lower tail of T",
    "sp": "SP: A + ensemble spread",
    "sp0": "SP0: SP without spread terms",
}
COLD_KEY: Final[str] = "cold nights (day mean below 2 C)"
TITLE_1: Final[str] = (
    "Adding the spread terms leaves cold-night q95 coverage where the intercept alone put it"
)
TITLE_2: Final[str] = "The spread terms (C5) add almost nothing beyond the fitted intercept (C6)"
TITLE_3: Final[str] = (
    "The fitted scale raises q95 coverage on every subset and leaves q05 coverage above 0.05"
)
TITLE_4: Final[str] = "The fitted scale rises with shortwave-radiation spread, most at the q75"

SUBSETS: Final[tuple[str, ...]] = (
    "all leads",
    "winter nights",
    "cold nights (day mean below 2 C)",
    "bank holidays",
    "Christmas (24 Dec to 1 Jan)",
)


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, 110),
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


def figure_coverage(*, data: dict) -> alt.TopLevelMixin:
    """Cold-night q95 coverage by arm, pooled, with the nominal and pass lines."""
    rows = []
    for period, label in (("replicate", "replicate months"), ("scored", "scored year")):
        for row in data[period]["tables"]["tail"][COLD_KEY]:
            if row["arm"] in ARM_KEYS:
                rows.append(  # noqa: PERF401
                    {"arm": ARM_KEYS[row["arm"]], "period": label, "coverage": row["cov_q95"]}
                )
    points = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=120)
        .encode(
            y=alt.Y("arm:N", sort=list(ARM_COLOURS), title=None, axis=alt.Axis(labelLimit=200)),
            x=alt.X(
                "coverage:Q",
                scale=alt.Scale(domain=[0.80, 0.96]),
                title="share of cold-night readings at or under the q95",
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=None,
            ),
        )
    )
    labels = points.mark_text(dy=-14, fontSize=11).encode(text=alt.Text("coverage:Q", format=".3f"))
    pass_line = alt.Chart().mark_rule(strokeDash=[4, 3], color="#444444").encode(x=alt.datum(0.90))  # ty: ignore[unresolved-attribute]
    nominal = alt.Chart().mark_rule(color="#444444").encode(x=alt.datum(0.95))  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(points, labels, pass_line, nominal)
        .properties(width=WIDTH / 2 - 120, height=170)
        .facet(
            column=alt.Column("period:N", sort=["replicate months", "scored year"], title=None),
            data=pl.DataFrame(rows),
        )
        .properties(
            title=_title(
                text=TITLE_1,
                subtitle=(
                    "Pooled over D5, D10, D11, D14, and D16. "
                    "Dashed line: pass line 0.90. Solid line: nominal 0.95."
                ),
            )
        )
    )


def figure_contrasts(*, data: dict) -> alt.TopLevelMixin:
    """All-row and cold-night quantile-score contrasts with their intervals."""
    names = {
        "C1": "C1 A minus I (assembly)",
        "C2": "C2 SP minus A (spread model)",
        "C4": "C4 AT minus A (lower tail of T)",
        "C5": "C5 SP minus SP0 (spread terms)",
        "C6": "C6 SP0 minus A (intercept)",
    }
    rows = []
    for period, label in (("replicate", "replicate months"), ("scored", "scored year")):
        for key, name in names.items():
            for scope, scope_label in (("all", "all rows"), (COLD_KEY, "cold nights")):
                c = data[period]["contrasts"][key][scope]["qs"]
                rows.append(
                    {
                        "contrast": name,
                        "period": label,
                        "scope": scope_label,
                        "estimate": c["estimate"],
                        "low": c["lower"],
                        "high": c["upper"],
                    }
                )
    base = alt.Chart().encode(
        y=alt.Y("contrast:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=250)),
        yOffset=alt.YOffset("period:N", sort=["replicate months", "scored year"]),
        color=alt.Color(
            "period:N",
            scale=alt.Scale(
                domain=["replicate months", "scored year"], range=[ocf.DATA_SKY, ocf.BRAND_ORANGE]
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    layer = alt.layer(
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="quantile-score change (points; negative is better)"),
            x2="high:Q",
        ),
        base.mark_point(filled=True, size=70).encode(x="estimate:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH / 2 - 80, height=140)
    return (
        layer.facet(
            column=alt.Column("scope:N", sort=["all rows", "cold nights"], title=None),
            data=pl.DataFrame(rows),
        )
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text=TITLE_2,
                subtitle="Paired month-resampled 95% intervals.",
            )
        )
    )


def figure_subsets(*, data: dict) -> alt.TopLevelMixin:
    """The q05 and q95 coverage of each arm on all rows, winter nights, and the holiday subsets."""
    rows = []
    for subset in SUBSETS:
        for row in data["scored"]["tables"]["tail"][subset]:
            if row["arm"] not in ARM_KEYS:
                continue
            for level in ("q05", "q95"):
                rows.append(  # noqa: PERF401
                    {
                        "subset": subset.replace(" (day mean below 2 C)", ""),
                        "arm": ARM_KEYS[row["arm"]],
                        "level": level,
                        "coverage": row[f"cov_{level}"],
                    }
                )
    labels = [s.replace(" (day mean below 2 C)", "") for s in SUBSETS]
    base = alt.Chart().encode(
        y=alt.Y("subset:N", sort=labels, title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X("coverage:Q", scale=alt.Scale(domain=[0.0, 1.0]), title="coverage"),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top", columns=2),
        ),
        yOffset=alt.YOffset("arm:N", sort=list(ARM_COLOURS)),
    )
    layers = []
    for level, nominal_value, domain in (("q05", 0.05, [0.0, 0.15]), ("q95", 0.95, [0.8, 1.0])):
        points = base.encode(
            x=alt.X("coverage:Q", scale=alt.Scale(domain=domain), title="coverage")
        ).mark_point(filled=True, size=50)
        rule = (
            alt.Chart(pl.DataFrame({"nominal": [nominal_value]}))
            .mark_rule(color="#444444")
            .encode(x="nominal:Q")  # ty: ignore[unresolved-attribute]
        )
        layers.append(
            alt.layer(
                points,  # ty: ignore[invalid-argument-type]
                rule,
                data=pl.DataFrame([r for r in rows if r["level"] == level]),
            ).properties(width=WIDTH / 2 - 120, height=200, title=f"{level} coverage")
        )
    return alt.hconcat(*layers).properties(
        title=_title(
            text=TITLE_3,
            subtitle="Scored year, five decision substations. Solid line: nominal coverage.",
        )
    )


def figure_scale(*, data: dict) -> alt.TopLevelMixin:
    """The fitted scale of each level's offset against the shortwave-radiation spread."""
    fit = json.loads(FIT.read_text())
    knots = fit["knots"]
    z2 = np.linspace(0.0, 2.0, 81)
    rows = []
    for level, coefficients in fit["coefficients"].items():
        scale = np.full_like(z2, coefficients[0])
        for k, b in zip(knots["k2"], coefficients[1 + len(knots["k1"]) :], strict=True):
            scale += b * np.maximum(z2 - k, 0.0)
        for x, y in zip(z2, scale, strict=True):
            rows.append(
                {
                    "level": f"q{round(float(level) * 100):02d}",
                    "spread": x * knots["scale2"],
                    "scale": y,
                }
            )
    order = [f"q{round(float(level) * 100):02d}" for level in fit["coefficients"]]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_line()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("spread:Q", title="spread of 24-hour mean shortwave radiation (W/m²)"),
            y=alt.Y("scale:Q", title="multiplier on the quantile's offset from the mean"),
            color=alt.Color("level:N", sort=order, title=None),
        )
        .properties(
            width=WIDTH - 120,
            height=240,
            title=_title(
                text=TITLE_4,
                subtitle=(
                    "Fitted on the tuning months, temperature spread at or below its 90th "
                    "percentile."
                ),
            ),
        )
    )


def main() -> None:
    """Draw every figure."""
    data = json.loads(REPORT.read_text())
    _save(figure_coverage(data=data), name="figure_1_cold_night_coverage")
    _save(figure_contrasts(data=data), name="figure_2_contrasts")
    _save(figure_subsets(data=data), name="figure_3_coverage_by_subset")
    _save(figure_scale(data=data), name="figure_4_spread_scale")


if __name__ == "__main__":
    main()
