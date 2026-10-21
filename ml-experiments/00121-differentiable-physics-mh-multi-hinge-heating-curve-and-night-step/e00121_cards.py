"""Draw experiment 121's substation cards (provisional) and print the NS-UT step by winter.

**One card per decision substation (D-labels).** Each card shows five sample weeks of the scored
year, fixed before any score was read: the week starting on the second Monday of July, October,
January, and April, and the coldest Monday-to-Sunday week (lowest mean of the daily mean ERA5
temperature at the substation's cell). Each panel draws observed power, the median forecasts of
B0-K, MH-K, and NS-U-K, and the 80% band of NS-U-K. The script also prints NS-UT's fitted monthly
step levels as a mean per winter (November to February), from the stored coefficients.

    OMP_NUM_THREADS=1 uv run python ml-experiments/00121-*/e00121_cards.py
"""

import json
import pickle
from datetime import date, timedelta
from typing import Final

import altair as alt
import e00121_charts as charts
import e00121_common as common
import e00121_report as report
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

SEASON_MONTHS: Final[dict[str, tuple[int, int]]] = {
    "summer (July 2025)": (2025, 7),
    "autumn (October 2025)": (2025, 10),
    "winter (January 2026)": (2026, 1),
    "spring (April 2026)": (2026, 4),
}
ARMS: Final[dict[str, str]] = {
    "b0_k": "B0-K (no new term)",
    "mh_k": "MH-K (five hinges)",
    "ns_u_k": "NS-U-K (UTC night levels)",
}


def second_monday(*, year: int, month: int) -> date:
    """Return the date of the second Monday of a month."""
    first = date(year, month, 1)
    offset = (7 - first.weekday()) % 7
    return first + timedelta(days=offset + 7)


def coldest_week_start(*, site: str) -> date:
    """Return the Monday of the scored-year week with the lowest mean daily temperature."""
    days = report.cold_days().filter(
        (pl.col("site_name") == site)
        & (pl.col("local_date") >= date(2025, 7, 7))
        & (pl.col("local_date") < date(2026, 6, 29))
    )
    days = days.with_columns(
        monday=pl.col("local_date") - pl.duration(days=pl.col("local_date").dt.weekday() - 1)
    )
    weeks = days.group_by("monday").agg(pl.col("tmean").mean(), pl.len().alias("n"))
    weeks = weeks.filter(pl.col("n") == 7).sort("tmean")
    return weeks["monday"][0]


def card(*, site: str) -> alt.TopLevelMixin:
    """Return one substation's card."""
    starts = {name: second_monday(year=y, month=m) for name, (y, m) in SEASON_MONTHS.items()}
    starts["coldest week"] = coldest_week_start(site=site)
    rows = []
    for arm, label in ARMS.items():
        frame = report.load_arm(year="scored", arm=arm, drop_switching=False).filter(
            pl.col("site_name") == site
        )
        for panel, start in starts.items():
            week = frame.filter(
                (pl.col("local_date") >= start) & (pl.col("local_date") < start + timedelta(days=7))
            ).sort("valid_time")
            rows.extend(
                {
                    "panel": f"{panel}, from {start:%d %b %Y}",
                    "series": label,
                    "day": (row["local_date"] - start).days
                    + (row["local_start"].hour * 60 + row["local_start"].minute) / 1440.0,
                    "forecast": row["q50"],
                    "low": row["q10"],
                    "high": row["q90"],
                    "observed": row["actual"],
                }
                for row in week.iter_rows(named=True)
            )
    frame = pl.DataFrame(rows)
    order = [f"{name}, from {start:%d %b %Y}" for name, start in starts.items()]
    x = alt.X(
        "day:Q",
        title="Days since Monday 00:00 (local)",
        scale=alt.Scale(domain=[0, 7]),
        axis=alt.Axis(values=list(range(8))),
    )
    band = (
        alt.Chart()
        .transform_filter(alt.datum.series == ARMS["ns_u_k"])
        .mark_area(opacity=0.2, color=ocf.DATA_BLUE)
        .encode(x=x, y=alt.Y("low:Q", scale=alt.Scale(zero=False)), y2="high:Q")  # ty: ignore[unresolved-attribute]
    )
    lines = (
        alt.Chart()
        .mark_line(strokeWidth=1.5)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("forecast:Q", title="Power (points of capacity)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=list(ARMS.values()), range=[charts.GREY, ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(title=None, orient="top", labelLimit=300),
            ),
        )
    )
    truth = (
        alt.Chart()
        .transform_filter(alt.datum.series == ARMS["b0_k"])
        .mark_line(strokeWidth=1.5, color=charts.DARK, strokeDash=[4, 2])
        .encode(x=x, y="observed:Q")  # ty: ignore[unresolved-attribute]
    )
    return (
        alt.layer(band, lines, truth)
        .properties(width=190, height=120)
        .facet(facet=alt.Facet("panel:N", sort=order, title=None), data=frame, columns=3)
        .properties(
            title=charts._title(
                text=f"Provisional: substation {site} on a screening run, scored year 2025-26",
                subtitle=(
                    "Fitted to past weather, day ahead, after the filter. Dashed black: observed "
                    "power. Lines: median forecasts. Blue band: 80% interval of NS-U-K. Weeks "
                    "fixed before any score was read; the coldest week has the lowest mean daily "
                    "temperature of the year."
                ),
            )
        )
    )


def step_by_winter() -> None:
    """Print NS-UT's fitted monthly levels as a mean per winter, from the stored coefficients."""
    cards = pickle.loads((common.WORK_DIR / "coefficients.pkl").read_bytes())
    capacity = json.loads((common.WORK_DIR / "nightstep.json").read_text())["capacity_mw"]
    out = []
    print("\n== NS-UT fitted step by winter (points; MW in brackets) ==")
    for site in common.DECISION_SITES:
        levels: dict[str, list[float]] = {}
        for label in ("tuning", "replicate", "scored"):
            theta = cards[site]["ns_ut"].get(label)
            if theta is None:
                continue
            knots = theta[-(common.WINDOW_MONTHS + 1) :]
            cut = common.FIT_CUTS[label]
            for k, level in enumerate(knots):
                moment = np.datetime64(cut.replace(tzinfo=None), "D") - np.timedelta64(
                    round(common.WINDOW_MONTHS * 30.4375), "D"
                )
                month = (moment + np.timedelta64(round(k * 30.4375), "D")).astype("datetime64[M]")
                number = int(month.astype(int) % 12) + 1
                year = int(month.astype("datetime64[Y]").astype(int)) + 1970
                if number in (11, 12, 1, 2):
                    winter = year if number >= 7 else year - 1
                    levels.setdefault(f"winter {winter}-{winter + 1 - 2000}", []).append(
                        float(level)
                    )
        summary = {
            k: (round(float(np.mean(v)), 2), round(float(np.mean(v)) * capacity[site] / 100.0, 3))
            for k, v in levels.items()
        }
        print(site, summary)
        out.append({"site": site, **summary})
    (common.WORK_DIR / "ns_ut_step_by_winter.json").write_text(json.dumps(out, indent=2) + "\n")


def main() -> None:
    """Draw every card and print the NS-UT step by winter."""
    for site in common.DECISION_SITES:
        charts._save(card(site=site), name=f"card_{site}")
    step_by_winter()


if __name__ == "__main__":
    main()
