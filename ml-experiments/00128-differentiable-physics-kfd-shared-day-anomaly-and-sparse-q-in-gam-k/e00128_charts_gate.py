"""Draw the figures of experiment 128's write-up and print every number the page quotes.

The experiment stopped at the known-answer gate, so every figure here comes from the stored gate,
control, and hyperparameter files. Only D-labels appear. SVGs go to the write-up's folder, PNGs to a
scratch folder for the render check.

    uv run python ml-experiments/00128-*/e00128_charts_gate.py figures
    uv run python ml-experiments/00128-*/e00128_charts_gate.py numbers
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00128_common as common
import e00128_gate as gate
import e00128_gate_report as report
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
WIDTH: Final[int] = 680
WRAP: Final[int] = 110
GREY: Final[str] = "#808080"
ARMS: Final[tuple[str, ...]] = ("GK", "KD", "KS", "KDS")
ARM_COLOURS: Final[dict[str, str]] = {
    "GK": ocf.DATA_BLUE,
    "KD": ocf.DATA_SKY,
    "KS": ocf.DATA_GREEN,
    "KDS": ocf.BRAND_ORANGE,
}
REAL: Final[str] = "Real residuals resampled"
AR1: Final[str] = "AR(1) noise (the filter's own noise model)"
BAND_LABELS: Final[tuple[str, ...]] = (
    "Night (00:00 to 06:00)",
    "Morning (06:00 to 11:00)",
    "Midday (11:00 to 16:00)",
    "Evening (16:00 to 24:00)",
)
SUPPORT_ARMS: Final[dict[str, str]] = {
    "KDS-1n": "Threshold 1 nat (KDS-1n)",
    "KDS": "Threshold 2 nats (KDS, the pre-registered rule)",
    "KDS-4n": "Threshold 4 nats (KDS-4n)",
}
P_SHIFT_CRITERION: Final[float] = 0.1
EXAMPLE_DRAW: Final[int] = 0
EXAMPLE_SITE: Final[str] = "D5"
EXAMPLE_SCENARIO: Final[str] = "S1x3"


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


def _gate_rows() -> pl.DataFrame:
    stored = json.loads(gate.GATE_JSON.read_text())
    return pl.DataFrame(
        [
            {
                "reading": row["reading"],
                "arm": row["arm"],
                "scenario": row["scenario"],
                "form": REAL if row["form"] == "real" else AR1,
                "rate": row["rate"],
                "low": max(row["rate"] - row["half_width"], 0.0),
                "high": min(row["rate"] + row["half_width"], 1.0),
            }
            for row in stored["rows"]
        ]
    )


def figure_reading(*, reading: str, title: str, subtitle: str, bar: bool) -> alt.TopLevelMixin:
    """Pass rate of one gate reading by arm and noise form, one panel per injection size."""
    table = _gate_rows().filter(pl.col("reading") == reading, pl.col("scenario") != "S0")
    panels = []
    for scenario, label in (
        ("S1", "Injected change as in experiment 118 (S1)"),
        ("S1x3", "Injected change three times larger (S1x3)"),
    ):
        part = table.filter(pl.col("scenario") == scenario)
        base = alt.Chart(part).encode(
            y=alt.Y("arm:N", sort=list(ARMS), title=None),
            yOffset=alt.YOffset("form:N", sort=[REAL, AR1]),
            color=alt.Color(
                "form:N",
                scale=alt.Scale(domain=[REAL, AR1], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(title=None, orient="top", labelLimit=400),
            ),
        )
        layers = [
            base.mark_rule(strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("low:Q", scale=alt.Scale(domain=[0, 1])), x2="high:Q"
            ),
            base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("rate:Q", title="Share of draws that pass")
            ),
        ]
        if bar and scenario == "S1x3":
            bar_value = float(
                part.filter(pl.col("arm") == "KDS", pl.col("form") == AR1)["rate"][0]
                * report.RELATIVE_BAR
            )
            layers.append(
                alt.Chart(pl.DataFrame({"x": [bar_value]}))  # ty: ignore[unresolved-attribute]
                .mark_rule(color="#333333", strokeDash=[4, 3])
                .encode(x="x:Q")
            )
        panels.append(alt.layer(*layers).properties(width=WIDTH - 90, height=130, title=label))
    return alt.vconcat(*panels).properties(title=_title(text=title, subtitle=subtitle))


def figure_p_shift() -> alt.TopLevelMixin:
    """Day-state noise fitted on the real stream and on band-shifted streams, per substation."""
    stored = json.loads((common.WORK_DIR / "control_p_shift.json").read_text())
    rows = []
    for site, entry in stored["sites"].items():
        rows.append({"site": site, "fit": "Real stream", "sigma_d": entry["unshifted_sigma_d"]})
        rows.append(
            {
                "site": site,
                "fit": "Bands shifted by random days (median of 20)",
                "sigma_d": entry["median_shifted_sigma_d"],
            }
        )
    table = pl.DataFrame(rows)
    order = sorted(stored["sites"], key=lambda s: int(s[1:]))
    points = (
        alt.Chart(table)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("site:N", sort=order, title=None),
            x=alt.X(
                "sigma_d:Q",
                title="Fitted day-state noise (points per day)",
                scale=alt.Scale(domain=[0, 4]),
                axis=alt.Axis(values=[0, 1, 2, 3, 4]),
            ),
            color=alt.Color(
                "fit:N",
                scale=alt.Scale(
                    domain=["Real stream", "Bands shifted by random days (median of 20)"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top", labelLimit=400),
            ),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [P_SHIFT_CRITERION]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#333333", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (points + rule).properties(
        width=WIDTH - 90,
        height=200,
        title=_title(
            text=(
                "Shifting the hour bands apart lowers the day state's fitted noise, "
                "but not to near zero"
            ),
            subtitle=(
                "Each substation's day-state noise (sigma_d) fitted by KD. The control shifts each "
                "hour band by its own random number of days, which destroys any structure shared "
                "between bands. Dashed line: 0.1, the level the shifted fits had to fall below. "
                "None did."
            ),
        ),
    )


def figure_support() -> alt.TopLevelMixin:
    """Which hour bands keep a nonzero level variance, per substation and threshold."""
    rows = []
    for arm, label in SUPPORT_ARMS.items():
        stored = json.loads((common.WORK_DIR / "hyper" / f"{arm}.json").read_text())
        for site in common.DECISION_SITES:
            for band, active in enumerate(stored["sites"][site]["active"]):
                rows.append(
                    {
                        "threshold": label,
                        "site": site,
                        "band": BAND_LABELS[band],
                        "state": "Level may move" if active else "Level held fixed (variance 0)",
                    }
                )
    table = pl.DataFrame(rows)
    order = sorted(common.DECISION_SITES, key=lambda s: int(s[1:]))
    chart = (
        alt.Chart(table)
        .mark_rect(stroke="white", strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "band:N",
                sort=list(BAND_LABELS),
                title=None,
                axis=alt.Axis(labelAngle=0, labelLimit=150),
            ),
            y=alt.Y("site:N", sort=order, title=None),
            color=alt.Color(
                "state:N",
                scale=alt.Scale(
                    domain=["Level may move", "Level held fixed (variance 0)"],
                    range=[ocf.DATA_BLUE, "#E6E6E6"],
                ),
                legend=alt.Legend(title=None, orient="top", labelLimit=400),
            ),
        )
        .properties(width=WIDTH - 90, height=110)
        .facet(
            row=alt.Row(
                "threshold:N",
                sort=list(SUPPORT_ARMS.values()),
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=11),
            ),
        )
    )
    return chart.properties(
        title=_title(
            text="At 2 nats, D11 keeps no hour band's level free to move",
            subtitle=(
                "The sparse search lets an hour band's level drift only if that raises the "
                "likelihood by at least the threshold. Blue: level may move. Grey: held fixed. "
                "Fitted on the odd months of 2024-07 to 2025-06."
            ),
        )
    )


def _example_traces() -> pl.DataFrame:
    truth_1, k_star = report.site_truth(EXAMPLE_SITE)
    amplitude = gate.SCENARIOS[EXAMPLE_SCENARIO]
    records = report.load_records()
    treated = records[("real", EXAMPLE_SCENARIO, EXAMPLE_SITE, EXAMPLE_DRAW)]
    base = records[("real", "S0", EXAMPLE_SITE, EXAMPLE_DRAW)]
    days = np.arange(1, gate.READ_WINDOW[1] + 1)
    rows = [
        {
            "arm": "Injected change",
            "day": int(d),
            "value": float(amplitude * truth_1[d]),
            "k": k_star,
        }
        for d in days
    ]
    for arm in ARMS:
        mid = report.paired(treated=treated, base=base, arm=arm, name="mid")
        rows += [
            {"arm": arm, "day": int(d), "value": float(mid[d]), "k": k_star}
            for d in days
            if np.isfinite(mid[d])
        ]
    return pl.DataFrame(rows)


def figure_example() -> alt.TopLevelMixin:
    """The injected change and each arm's midday level on one example draw."""
    table = _example_traces()
    panels = []
    for arm in ARMS:
        truth = table.filter(pl.col("arm") == "Injected change").with_columns(
            series=pl.lit("Injected change")
        )
        own = table.filter(pl.col("arm") == arm).with_columns(
            series=pl.lit("Filtered midday level")
        )
        both = pl.concat([truth, own])
        panels.append(
            alt.Chart(both)
            .mark_line(clip=True, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "day:Q",
                    title="Days since the injection began",
                    axis=alt.Axis(values=[0, 50, 100, 150, 200]),
                    scale=alt.Scale(domain=[0, 200]),
                ),
                y=alt.Y("value:Q", title="Points of capacity"),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=["Injected change", "Filtered midday level"],
                        range=["#333333", ARM_COLOURS[arm]],
                    ),
                    legend=alt.Legend(title=None, orient="top", labelLimit=400)
                    if arm == "GK"
                    else None,
                ),
            )
            .properties(width=WIDTH - 90, height=110, title=arm)
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text=(
                "On one draw of real residuals, every arm's midday level falls short of the "
                "injected change at its peak"
            ),
            subtitle=(
                "Substation D5, first draw of real residuals resampled, change three times the "
                "experiment 118 size (S1x3). Black: the injected change (a drop in the residual "
                "that grows with solar output). Colour: the arm's midday level on the injected "
                "stream minus its level on the same noise without the injection. The draw was "
                "fixed in advance, not chosen."
            ),
        )
    )


def figures() -> None:
    """Draw and save the five figures."""
    _save(
        figure_reading(
            reading="1",
            title=(
                "With real residuals, KDS tracks the injected change in 35% of draws; "
                "its bar was 48%"
            ),
            subtitle=(
                "Share of draws in which the midday level recovers at least 70% of the injected "
                "change and correlates with it at 0.70 or more (higher is better). Dot: share. "
                "Line: 95% interval. Dashed line: the bar KDS had to reach, 0.8 times its rate "
                "under the filter's own noise model."
            ),
            bar=True,
        ),
        name="figure_01_gate_tracking",
    )
    _save(
        figure_reading(
            reading="2",
            title=(
                "With real residuals, KD fails the cross-talk reading in most draws, and KDS "
                "passes in only 57% at S1x3"
            ),
            subtitle=(
                "Share of draws in which neither the night level nor the day state moves by 25% "
                "or more of the injected change's peak (higher is better). Dot: share. Line: 95% "
                "interval."
            ),
            bar=True,
        ),
        name="figure_02_gate_cross_talk",
    )
    _save(figure_example(), name="figure_03_example_draw")
    _save(figure_p_shift(), name="figure_04_p_shift")
    _save(figure_support(), name="figure_05_sparse_support")


def numbers() -> None:
    """Print every number the write-up quotes that the gate report does not already print."""
    gate_stored = json.loads(gate.GATE_JSON.read_text())
    print("decision scenario", gate_stored["decision_scenario"], gate_stored["s1_underpowered"])
    print("false movement", json.dumps(gate_stored["false_movement"], indent=1))
    print("rule", json.dumps(gate_stored["rule"], indent=1))
    print("-- power (peak of injected change in posterior sds at S1; peak in points)")
    for site, entry in sorted(json.loads(gate.POWER_JSON.read_text()).items()):
        print(
            site,
            round(entry["primary_in_sds"], 2),
            round(entry["peak_s1"], 2),
            round(3 * entry["primary_in_sds"], 2),
        )
    print("-- P-shift")
    shift = json.loads((common.WORK_DIR / "control_p_shift.json").read_text())
    for site, entry in sorted(shift["sites"].items()):
        print(
            site,
            round(entry["unshifted_sigma_d"], 2),
            round(entry["median_shifted_sigma_d"], 2),
            entry["share_shifted_below_0.1"],
            entry["ok"],
        )
    print("-- P-premise")
    premise = json.loads((common.WORK_DIR / "control_p_premise.json").read_text())
    for site in common.DECISION_SITES:
        print(
            site,
            round(premise[site]["cross_band_correlation"], 3),
            round(premise[site]["lag1_autocorrelation"], 3),
        )
    print(json.dumps(premise["summary"]))
    print("-- P-ident")
    ident = json.loads((common.WORK_DIR / "control_p_ident.json").read_text())
    print({k: v for k, v in ident.items() if k != "outputs"})
    print("-- P-sparse")
    sparse = json.loads((common.WORK_DIR / "control_p_sparse.json").read_text())
    print(json.dumps(sparse))
    print("-- P1, P2, P-day")
    for name in ("p1", "p2", "p_day"):
        stored = json.loads((common.WORK_DIR / f"control_{name}.json").read_text())
        passed = stored.get("passed")
        if passed is None:
            passed = all(v["passed"] for v in stored.values() if isinstance(v, dict))
        print(name, passed)
    print("-- support (active bands) by arm")
    for arm in ("KS", "KDS", "KDS-floor", "KDS-1n", "KDS-4n"):
        stored = json.loads((common.WORK_DIR / "hyper" / f"{arm}.json").read_text())
        for site in common.DECISION_SITES:
            entry = stored["sites"][site]
            print(arm, site, entry["active"], [round(q, 4) for q in entry["params"]["q"]])
    print("-- KD and KDS fitted day state")
    for arm in ("KD", "KDS"):
        stored = json.loads((common.WORK_DIR / "hyper" / f"{arm}.json").read_text())
        for site in common.DECISION_SITES:
            params = stored["sites"][site]["params"]
            print(arm, site, round(params["sigma_d"], 2), round(params["phi_d"], 2))
    print("-- fit status", (common.WORK_DIR / "fit_status.json").read_text())
    print("-- example draw")
    truth_1, k_star = report.site_truth(EXAMPLE_SITE)
    amplitude = gate.SCENARIOS[EXAMPLE_SCENARIO]
    records = report.load_records()
    treated = records[("real", EXAMPLE_SCENARIO, EXAMPLE_SITE, EXAMPLE_DRAW)]
    base = records[("real", "S0", EXAMPLE_SITE, EXAMPLE_DRAW)]
    print("read day", k_star, "truth at read day", round(amplitude * truth_1[k_star], 2))
    for arm in ARMS:
        mid = report.paired(treated=treated, base=base, arm=arm, name="mid")
        recent = mid[k_star - gate.READ_AVERAGE_DAYS + 1 : k_star + 1]
        print(arm, "level at read day", round(float(np.nanmean(recent)), 2))


def main() -> None:
    """Draw the figures or print the numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("figures", "numbers"))
    args = parser.parse_args()
    if args.step == "figures":
        figures()
    else:
        numbers()


if __name__ == "__main__":
    main()
