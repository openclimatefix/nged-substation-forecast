"""Draw the figures of experiment 201's write-up from `work/score.json` and `work/preds/`.

uv run --no-sync python ml-experiments/00201-*/e00201_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00201_common as common
import e00201_data as data
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
WIDTH: Final[int] = 640
ARM_NAMES: Final[dict[int, str]] = {
    0: "0 blind: lead, calendar, site only (post hoc)",
    1: "1 convex model's own inputs",
    2: "2 plus calendar and holiday flags",
    3: "3 plus ensemble-mean weather",
    4: "4 plus lagged power and site",
}
ARM_COLOURS: Final[dict[int, str]] = {
    0: "#808080",
    1: ocf.DATA_SKY,
    2: ocf.DATA_BLUE,
    3: ocf.BRAND_ORANGE,
    4: ocf.DATA_PURPLE,
}
GROUPS: Final[dict[str, str]] = {
    "all": "all rows",
    "d_plus_1": "day D+1",
    "cold_nights": "cold nights",
    "sunny_daylight": "sunny daylight",
    "economy_7_nights": "Economy 7 nights",
    "bank_holidays": "bank holidays",
    "christmas": "Christmas",
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, 105),
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


def forest(*, score: dict, groups: list[str], name: str, title: str, subtitle: str) -> None:
    """Draw the contrast of each arm against arm D, with 97.5% intervals, for some groups."""
    rows = [
        {
            "group": GROUPS[g],
            "arm": ARM_NAMES[a],
            "estimate": score[f"arm{a}"]["contrast"][g][0],
            "low": score[f"arm{a}"]["contrast"][g][1],
            "high": score[f"arm{a}"]["contrast"][g][2],
        }
        for g in groups
        for a in ARM_NAMES
    ]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("group:N", sort=[GROUPS[g] for g in groups], title=None),
        yOffset=alt.YOffset("arm:N", sort=list(ARM_NAMES.values())),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(ARM_NAMES.values()), range=list(ARM_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=500),
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = (
        zero
        + base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Quantile score minus arm D (points; below zero is better)"),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=40).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH, height=40 * len(groups) + 40, title=_title(text=title, subtitle=subtitle)
    )
    _save(chart, name=name)


def shift_profiles() -> None:
    """Draw the mean median shift of arms 1 and 3 by temperature, half-hour, and day type."""
    frame = data.load_rows().filter(pl.col("period") == "scored")
    rows = []
    for arm in (1, 3):
        preds = pl.concat(
            [
                pl.read_parquet(common.WORK_DIR / "preds" / f"arm{arm}_seed{s}.parquet")
                for s in range(3)
            ]
        )
        shift = preds["delta"].to_numpy().reshape(3, -1).mean(axis=0)
        scale = np.exp(preds["log_scale"].to_numpy().reshape(3, -1).mean(axis=0))
        view = frame.with_columns(
            pl.Series("shift", shift),
            pl.Series("scale", scale),
            temp_bin=(pl.col("drv_temp_c") / 3).floor() * 3,
            day=pl.col("dow").replace_strict(
                {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            ),
        )
        for variable, column in (
            ("driver temperature (C, 3-degree bins)", "temp_bin"),
            ("local half-hour of day", "local_hh"),
            ("day of week", "day"),
            ("lead (hours, 24-hour bins)", "lead_bin"),
        ):
            if column == "lead_bin":
                view = view.with_columns(lead_bin=(pl.col("lead_hours") / 24).floor() * 24)
            for key, value, _scale in (
                view.group_by(column)
                .agg(pl.col("shift").mean().alias("shift"), pl.col("scale").mean().alias("scale"))
                .iter_rows()
            ):
                rows.append(
                    {
                        "arm": ARM_NAMES[arm],
                        "variable": variable,
                        "value": str(key) if column == "day" else float(key),
                        "shift": value,
                    }
                )
    for variable in sorted({row["variable"] for row in rows}):
        part = [row for row in rows if row["variable"] == variable]
        numeric = variable != "day of week"
        chart = (
            alt.Chart(alt.Data(values=part))
            .mark_line(point=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "value:Q" if numeric else "value:N",
                    title=str(variable),
                    sort=None if numeric else ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                ),
                y=alt.Y("shift:Q", title="Mean shift of the median (points)"),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(
                        domain=[ARM_NAMES[1], ARM_NAMES[3]],
                        range=[ARM_COLOURS[1], ARM_COLOURS[3]],
                    ),
                    legend=alt.Legend(
                        title=None, orient="top", direction="vertical", labelLimit=500
                    ),
                ),
            )
            .properties(width=WIDTH, height=180)
        )
        slug = str(variable).split(" (")[0].replace(" ", "_")
        _save(chart, name=f"figure_shift_by_{slug}")


def main() -> None:
    """Draw every figure."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    forest(
        score=score,
        groups=["all", "d_plus_1"],
        name="figure_1_headline",
        title="A small network lowers arm D's score by about 0.1 points, with or without weather",
        subtitle="Scored year, six substations, 97.5% month-blocked intervals, mean of 3 seeds.",
    )
    forest(
        score=score,
        groups=["cold_nights", "sunny_daylight", "economy_7_nights", "bank_holidays", "christmas"],
        name="figure_2_special_days",
        title="Every arm scores worse than arm D on bank holidays and Christmas",
        subtitle="Same contrasts on the special groups (few days: wide intervals).",
    )
    shift_profiles()


if __name__ == "__main__":
    main()
