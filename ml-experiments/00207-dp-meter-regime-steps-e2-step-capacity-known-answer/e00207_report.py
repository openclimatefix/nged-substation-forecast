"""Print the numbers of experiment 207 and draw its figures as SVG into the docs folder.

Only Primaries (D-labels) and anonymised farms (A to F, as ratios) appear. Afterwards optimise each
SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00207-*/e00207_report.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00207_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00207-dp-meter-regime-steps-e2-step-capacity-known-answer"
)
PNG_DIR: Final[Path] = Path("/tmp/e00207_png")
WIDTH: Final[int] = 600
WRAP: Final[int] = 100
SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The five decision Primaries. D9 has no readings before 2025, so it cannot hold a 2022 step."""
MAIN: Final[str] = "step_0.3"
ARM_LABELS: Final[dict[str, str]] = {
    "two_stage": "Season-year steps (experiment 129)",
    "smooth": "Smooth monthly path (experiment 157)",
    "step_0.3": "Step path, weight 0.3 (pre-registered)",
    "step_3": "Step path, weight 3 (sensitivity)",
}
ARM_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]
N_BOOT: Final[int] = 2000
PLACEBO_YEARS: Final[tuple[int, ...]] = (2022, 2023, 2024)


def load_recovery() -> pl.DataFrame:
    """Return every Primary's recovery table."""
    return pl.concat([pl.read_parquet(common.WORK_DIR / f"recovery_{s}.parquet") for s in SITES])


def load_paths() -> pl.DataFrame:
    """Return every Primary's monthly level paths."""
    return pl.concat([pl.read_parquet(common.WORK_DIR / f"paths_{s}.parquet") for s in SITES])


def median_interval(*, values: np.ndarray, seed: int = 207) -> tuple[float, float, float]:
    """Return the median and a 95% bootstrap interval over injections."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True)
    low, high = np.percentile(np.median(draws, axis=1), [2.5, 97.5])
    return float(np.median(values)), float(low), float(high)


def recovery_summary(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return the median size ratio with interval, and the share of dates within tolerance."""
    rows = []
    for (site, arm), group in table.group_by("site", "arm"):
        median, low, high = median_interval(values=group["size_ratio"].to_numpy())
        rows.append(
            {
                "site": site,
                "arm": arm,
                "median": median,
                "low": low,
                "high": high,
                "date_ok": float((group["date_error_days"] <= common.DATE_TOLERANCE_DAYS).mean()),  # ty: ignore[invalid-argument-type]
                "n": group.height,
            }
        )
    return pl.DataFrame(rows).sort("arm", "site")


def placebo_steps(*, paths: pl.DataFrame) -> pl.DataFrame:
    """Count month-to-month changes above `PLACEBO_STEP_MW` in the base path per year."""
    rows = []
    base = paths.filter(pl.col("target") == 0, pl.col("arm").str.starts_with("step_"))
    for (site, farm, arm), group in base.group_by("site", "farm", "arm"):
        ordered = group.sort("month")
        level = ordered["level"].to_numpy()
        month = ordered["month"].to_numpy()
        change = np.abs(np.diff(level)) * float(ordered["scale_mw_per_point"][0])
        year = 1970 + month[1:] // 12
        rows += [
            {
                "site": site,
                "farm": farm,
                "arm": arm,
                "year": y,
                "false_steps": int(((change > common.PLACEBO_STEP_MW) & (year == y)).sum()),
                "largest_mw": float(change[year == y].max()),
            }
            for y in PLACEBO_YEARS
        ]
    return pl.DataFrame(rows)


def gate_table(*, summary: pl.DataFrame, placebo: pl.DataFrame) -> pl.DataFrame:
    """Return, per step arm and Primary, whether the size, date, and placebo tests pass."""
    false = placebo.group_by("site", "arm").agg(false_per_year=pl.col("false_steps").mean())
    low, high = common.SIZE_BAND
    return (
        summary.filter(pl.col("arm").str.starts_with("step_"))
        .join(false, on=["site", "arm"])
        .with_columns(
            size_ok=pl.col("median").is_between(low, high),
            date_pass=pl.col("date_ok") >= 0.75,
            placebo_ok=pl.col("false_per_year") < 1.0,
        )
        .with_columns(all_ok=pl.col("size_ok") & pl.col("date_pass") & pl.col("placebo_ok"))
        .sort("arm", "site")
    )


def print_numbers(*, table: pl.DataFrame, paths: pl.DataFrame) -> None:
    """Print the numbers the write-up quotes."""
    summary = recovery_summary(table=table)
    placebo = placebo_steps(paths=paths)
    gates = gate_table(summary=summary, placebo=placebo)
    with pl.Config(tbl_rows=80, float_precision=3, tbl_cols=14, tbl_width_chars=200):
        print(summary.filter(pl.col("arm").is_in(list(ARM_LABELS))))
        print(gates)
        print(
            gates.group_by("arm").agg(
                size=pl.col("size_ok").sum(),
                date=pl.col("date_pass").sum(),
                placebo=pl.col("placebo_ok").sum(),
                all_three=pl.col("all_ok").sum(),
            )
        )
        print(
            placebo.group_by("arm")
            .agg(
                mean=pl.col("false_steps").mean(),
                worst_year=pl.col("false_steps").max(),
                largest_mw=pl.col("largest_mw").max(),
            )
            .sort("arm")
        )
        print(
            table.filter(pl.col("arm") == MAIN)
            .group_by("share", "direction")
            .agg(
                median=pl.col("size_ratio").median(),
                date_ok=(pl.col("date_error_days") <= 31).mean(),
            )
            .sort("share", "direction")
        )
        print(
            table.filter(pl.col("arm") == MAIN)
            .group_by("site")
            .agg(
                n_rows=pl.col("n_rows").median(),
                injected_mw_low=(pl.col("injected_points") * pl.col("scale_mw_per_point")).min(),
                injected_mw_high=(pl.col("injected_points") * pl.col("scale_mw_per_point")).max(),
            )
            .sort("site")
        )
        negative = (
            paths.filter(pl.col("arm").str.starts_with("step_"))
            .group_by("arm")
            .agg(lowest_level=pl.col("level").min(), n_negative=(pl.col("level") < -1e-6).sum())
        )
        print(negative)
        print(
            table.filter(pl.col("arm") == MAIN)
            .group_by("farm")
            .agg(median=pl.col("size_ratio").median())
            .sort("farm")
        )


def save_chart(chart: alt.TopLevelMixin, *, name: str) -> None:
    """Write the chart as an SVG into the docs folder and as a PNG for checking."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)


def chart_title(*, text: str, subtitle: str) -> alt.TitleParams:
    """Return a left-aligned title with the subtitle wrapped to the plot width."""
    return alt.TitleParams(
        text=text, subtitle=textwrap.wrap(subtitle, WRAP), anchor="start", subtitleColor="#444444"
    )


def figure_recovery(*, summary: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the size ratio per Primary and arm, with the gate band."""
    data = summary.filter(pl.col("arm").is_in(list(ARM_LABELS))).with_columns(
        label=pl.col("arm").replace(ARM_LABELS)
    )
    y_scale = alt.Scale(domain=[0.0, 1.4])
    band = (
        alt.Chart(pl.DataFrame({"low": [common.SIZE_BAND[0]], "high": [common.SIZE_BAND[1]]}))
        .mark_rect(color=ocf.DATA_GREEN, opacity=0.2)
        .encode(y=alt.Y("low:Q", scale=y_scale), y2="high:Q")  # ty: ignore[unresolved-attribute]
    )
    base = alt.Chart(data).encode(
        x=alt.X("site:N", sort=list(SITES), title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("label:N", sort=list(ARM_LABELS.values())),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=list(ARM_LABELS.values()), range=ARM_COLOURS),
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
        ),
    )
    bars = base.mark_rule(strokeWidth=2.5).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("low:Q", title="Recovered step over injected step", scale=y_scale), y2="high:Q"
    )
    points = base.mark_point(filled=True, size=45).encode(y=alt.Y("median:Q", scale=y_scale))  # ty: ignore[unresolved-attribute]
    return alt.layer(band, bars, points).properties(
        width=WIDTH,
        height=260,
        title=chart_title(
            text="Monthly capacity paths recover an injected step; season-year steps do not",
            subtitle=(
                "Dot: median over 20 injections (5 real farms, 2 sizes, on and off) at each"
                " Primary. Line: 95% bootstrap interval over the injections. Green band: the"
                " pre-registered gate, 0.75 to 1.25. Summer rows (April to August) before"
                " 2025-07-01."
            ),
        ),
    )


def figure_tradeoff(*, table: pl.DataFrame, paths: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the false-step rate and the size ratio against the penalty weight."""
    placebo = (
        placebo_steps(paths=paths).group_by("site", "arm").agg(value=pl.col("false_steps").mean())
    )
    size = (
        table.filter(pl.col("arm").str.starts_with("step_"))
        .group_by("site", "arm")
        .agg(value=pl.col("size_ratio").median())
    )
    parts = []
    for name, frame in (("False steps per placebo year", placebo), ("Median size ratio", size)):
        parts.append(frame.with_columns(panel=pl.lit(name)))
    data = pl.concat(parts).with_columns(
        weight=pl.col("arm").str.replace("step_", "").cast(pl.Float64)
    )
    data = data.filter(pl.col("weight") < 300)

    def panel(name: str, rule: float, title: str) -> alt.TopLevelMixin:
        sub = data.filter(pl.col("panel") == name)
        lines = (
            alt.Chart(sub)
            .mark_line(point=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("weight:Q", scale=alt.Scale(type="log"), title="Total-variation weight"),
                y=alt.Y("value:Q", title=title),
                color=alt.Color(
                    "site:N",
                    sort=list(SITES),
                    scale=alt.Scale(domain=list(SITES), range=list(ocf.DATA_COLOURS[: len(SITES)])),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
        )
        mark = (
            alt.Chart(pl.DataFrame({"y": [rule]}))
            .mark_rule(color="#808080", strokeDash=[4, 3])
            .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        )
        return (lines + mark).properties(width=WIDTH // 2 - 40, height=220)

    return alt.hconcat(
        panel("False steps per placebo year", 1.0, "False steps above 0.5 MW per year"),  # ty: ignore[invalid-argument-type]
        panel("Median size ratio", 1.0, "Recovered step over injected step"),  # ty: ignore[invalid-argument-type]
    ).properties(
        title=chart_title(
            text="A heavier penalty removes the false steps and, from weight 30, the real step",
            subtitle=(
                "Left: steps above 0.5 MW in the no-injection fit, per year, mean over 2022 to"
                " 2024 and five row sets (dashed line: the gate's 1). Right: median recovered"
                " step (dashed line: 1). Weight 300 removes every step and is not drawn."
            ),
        )
    )


def figure_paths(*, paths: pl.DataFrame, table: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the injected-minus-base path of farm A at D10 for one on-step and one off-step."""
    rows = table.filter(
        pl.col("site") == "D10",
        pl.col("farm") == "A",
        pl.col("share") == common.INJECTED_SHARES[1],
        pl.col("arm") == MAIN,
    ).with_row_index("n")
    pieces = []
    for direction in common.DIRECTIONS:
        info = rows.filter(pl.col("direction") == direction).row(0, named=True)
        target = (
            1 + common.INJECTED_SHARES.index(info["share"]) * 2 + common.DIRECTIONS.index(direction)
        )
        step_month = (info["step_date"].year - 1970) * 12 + info["step_date"].month - 1
        mine = paths.filter(pl.col("site") == "D10", pl.col("farm") == "A")
        for arm in ARM_LABELS:
            one = mine.filter(pl.col("arm") == arm)
            base = one.filter(pl.col("target") == 0).sort("month")["level"].to_numpy()
            injected = one.filter(pl.col("target") == target).sort("month")
            months = injected["month"].to_numpy()
            scale = float(injected["scale_mw_per_point"][0])
            pieces.append(
                pl.DataFrame(
                    {
                        "month": months,
                        "change_mw": (injected["level"].to_numpy() - base) * scale,
                        "label": ARM_LABELS[arm],
                        "panel": f"{direction}-step, {info['step_date']:%B %Y}",
                        "step_month": step_month,
                        "true_mw": info["injected_points"]
                        * scale
                        * (1 if direction == "on" else -1),
                    }
                )
            )
    data = pl.concat(pieces).with_columns(
        year=1970 + pl.col("month") // 12, calendar_month=pl.col("month") % 12 + 1
    )
    data = data.with_columns(date=pl.date(pl.col("year"), pl.col("calendar_month"), 1)).filter(
        pl.col("year").is_between(2021, 2025)
    )
    lines = (
        alt.Chart(data)
        .mark_line()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("date:T", title=None),
            y=alt.Y("change_mw:Q", title="Fitted capacity, injected minus base (MW)"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=list(ARM_LABELS.values()), range=ARM_COLOURS),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
            ),
        )
        .properties(width=WIDTH // 2 - 40, height=220)
    )
    return lines.facet(column=alt.Column("panel:N", title=None)).properties(
        title=chart_title(
            text="All three monthly paths jump at the right month; the season-year path does not",
            subtitle=(
                "Farm A injected at 60% of the pinned capacity at Primary D10 (fitted capacity with"
                " the injection minus the fit without). The farm output is subtracted from the"
                " step month on (left) or before it (right)."
            ),
        )
    )


def main() -> None:
    """Print the numbers and write the figures."""
    table = load_recovery()
    paths = load_paths()
    print_numbers(table=table, paths=paths)
    summary = recovery_summary(table=table)
    save_chart(figure_recovery(summary=summary), name="figure_1")
    save_chart(figure_tradeoff(table=table, paths=paths), name="figure_2")
    save_chart(figure_paths(paths=paths, table=table), name="figure_3")


if __name__ == "__main__":
    main()
