"""Post hoc checks on the CERRA wind-levels study: how much of the gain is a second height.

One-off throwaway script for the write-up of
<https://github.com/openclimatefix/nged-substation-forecast/issues/957>. The science review of
`cerra_wind_levels.py`'s results found that its planned contrast 3 (four heights against 100 m
alone) mixes two effects, a second height and further heights. Every number here is **exploratory
and post hoc**: the contrasts below were chosen after the results were seen, so they carry no
Bonferroni interval.

**Contrasts.** `levels_50_to_150` minus `speed_10m_100m`, `levels_all` minus `speed_10m_100m`,
`speed_10m_100m` minus `speed_10m`, and `levels_50_to_150` minus `mean_near_100m`, each at both
hyperparameter settings and, at the primary setting, at each farm. All are computed from
`cerra_wind_levels.py`'s saved `losses.parquet`, so this script fits nothing.

**Share of the gain.** For each of `speed_10m_100m`, `levels_50_to_150` and `levels_all`, the fall
in mean absolute error against `speed_100m`, and `speed_10m_100m`'s fall as a share of each of the
other two. The share is a ratio of point estimates and has no interval.

**Era check at one month.** The height-level product's documentation does not give the dates
where its production streams join, so the script reports the step statistic of
`cerra_wind_levels.era_step_table` at 2021-07, the month after the documented pause in production,
for each height's ratio to 100 m and its own level.

Run it with `uv run python studies/past_weather/cerra_wind_levels_shear.py`, after
`cerra_wind_levels.py`. It writes `report.md` and `intervals.parquet` to its own folder,
`cerra_wind_levels_post_hoc`, and stops while either exists. `report.md` there holds
`cerra_wind_levels.py`'s report followed by the post hoc sections, and `intervals.parquet` holds
both scripts' intervals, so `check_page_numbers.py` can check the whole write-up against one report.
"""

import logging
import sys
from typing import Final

import polars as pl
from cerra_wind_levels import (
    OUTPUT_DIR as SOURCE_DIR,
)
from cerra_wind_levels import (
    PRIMARY_SETTING,
    SENSITIVITY_SETTING,
    SPEED_100M,
    SPEED_COLUMNS,
    Contrast,
    _contrast_lines,
    interval_record,
    monthly_steps,
    read_wind,
)
from studies.guards import refuse_to_overwrite
from studies.pv_dataset import wind_sites
from studies.sources import CERRA_WIND_LEVELS_POST_HOC_DIR
from weather_products import _mae

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

OUTPUT_DIR: Final = CERRA_WIND_LEVELS_POST_HOC_DIR
"""Where this script writes, separate from `cerra_wind_levels.py`'s folder."""

TWO_HEIGHTS: Final[str] = "speed_10m_100m"
"""The arm with a second height, the 10 m speed beside the 100 m speed."""

SHEAR_CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast("levels_50_to_150", TWO_HEIGHTS, "four heights against two", False),
    Contrast("levels_all", TWO_HEIGHTS, "five heights against two", False),
    Contrast(TWO_HEIGHTS, "speed_10m", "two heights against the 10 m speed alone", False),
    Contrast("levels_50_to_150", "mean_near_100m", "four heights against their mean", False),
)
"""The post hoc contrasts."""

ONE_HEIGHT: Final[str] = "speed_100m"
"""The arm with the 100 m speed alone."""

GAIN_ARMS: Final[tuple[str, ...]] = (TWO_HEIGHTS, "levels_50_to_150", "levels_all")
"""The arms whose fall in error against 100 m alone the share table reports."""

ERA_MONTH: Final[str] = "2021-07"
"""The month after the documented pause in production, where the step statistic is reported."""


def _records(*, losses: pl.DataFrame, farms: list[str]) -> pl.DataFrame:
    """Compute each post hoc contrast at both settings, and at each farm at the primary setting.

    Args:
        losses: `cerra_wind_levels.py`'s saved losses.
        farms: The anonymised farm labels.

    Returns:
        One record per contrast and scope, as `interval_record` returns.
    """
    rows = []
    for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING):
        at_setting = losses.filter(pl.col("setting") == setting)
        for contrast in SHEAR_CONTRASTS:
            rows.append(
                interval_record(losses=at_setting, contrast=contrast, setting=setting, scope="all")
            )
            if setting == PRIMARY_SETTING:
                rows += [
                    interval_record(
                        losses=at_setting.filter(pl.col("site") == farm),
                        contrast=contrast,
                        setting=setting,
                        scope=farm,
                    )
                    for farm in farms
                ]
    return pl.DataFrame(rows)


def _share_lines(*, losses: pl.DataFrame) -> list[str]:
    """Render each arm's fall in error against 100 m alone, and the second height's share of it.

    Args:
        losses: `cerra_wind_levels.py`'s saved losses.

    Returns:
        Markdown lines, header included.
    """
    lines = [
        (
            "| Setting | Arm | Fall in error against speed_100m (pp of capacity) "
            "| Share of the fall that speed_10m_100m gives |"
        ),
        "|---|---|---|---|",
    ]
    for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING):
        at_setting = losses.filter(pl.col("setting") == setting)
        reference = _mae(losses=at_setting, arm=ONE_HEIGHT)
        two = reference - _mae(losses=at_setting, arm=TWO_HEIGHTS)
        for arm in GAIN_ARMS:
            fall = reference - _mae(losses=at_setting, arm=arm)
            lines.append(f"| {setting} | {arm} | {fall:.3f} | {100.0 * two / fall:.0f}% |")
    return lines


def _era_lines() -> list[str]:
    """Render the step statistic at `ERA_MONTH` for each height's ratio to 100 m and its level.

    Returns:
        Markdown lines, header included.
    """
    wind = read_wind(sites=wind_sites())
    monthly = (
        wind.with_columns(month=pl.col("time").dt.strftime("%Y-%m"))
        .group_by("month")
        .agg(pl.col(SPEED_COLUMNS).mean())
        .sort("month")
    )
    months = monthly["month"].to_list()
    lines = [f"| Test | Height | z at {ERA_MONTH} |", "|---|---|---|"]
    for test in ("ratio", "level"):
        for column in SPEED_COLUMNS:
            if test == "ratio" and column == SPEED_100M:
                continue
            series = monthly[column].to_numpy()
            if test == "ratio":
                series = series / monthly[SPEED_100M].to_numpy()
            steps, standard_error = monthly_steps(values=series, months=months)
            height = column.removeprefix("wind_speed_")
            lines.append(f"| {test} | {height} | {steps[ERA_MONTH] / standard_error:+.2f} |")
    return lines


def main() -> int:
    """Compute the post hoc contrasts from the saved losses and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = [OUTPUT_DIR / "report.md", OUTPUT_DIR / "intervals.parquet"]
    refuse_to_overwrite(paths=paths)
    losses = pl.read_parquet(SOURCE_DIR / "losses.parquet")
    farms = sorted(losses["site"].unique().to_list())
    records = _records(losses=losses, farms=farms)
    lines = [
        (SOURCE_DIR / "report.md").read_text().rstrip(),
        "",
        "## Post hoc checks (exploratory)",
        "",
        (
            "Every contrast here was chosen after the results were seen, so none carries a "
            "Bonferroni interval."
        ),
        "",
        "#### Post hoc contrasts, all farms",
        "",
        *_contrast_lines(records=records.filter(pl.col("scope") == "all")),
        "",
        "#### Post hoc contrasts, per farm (primary setting)",
        "",
        *_contrast_lines(records=records.filter(pl.col("scope") != "all")),
        "",
        "#### Share of the gain that a second height gives",
        "",
        *_share_lines(losses=losses),
        "",
        f"#### Step statistic at {ERA_MONTH}",
        "",
        *_era_lines(),
        "",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # `value`, `lower` and `upper` are the columns `studies.page_numbers` reads.
    pl.concat([pl.read_parquet(SOURCE_DIR / "intervals.parquet"), records]).with_columns(
        value=pl.col("difference_pp"),
        lower=pl.col("lower_95_pp"),
        upper=pl.col("upper_95_pp"),
    ).write_parquet(paths[1])
    paths[0].write_text("\n".join(lines))
    _LOG.info("wrote %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
