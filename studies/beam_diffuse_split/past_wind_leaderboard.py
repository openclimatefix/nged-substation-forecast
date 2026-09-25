"""One leaderboard and one set of contrasts against ERA5 for the four headline past-wind row sets.

The past-wind page scores four row sets, each on its own common rows: the main rows (five products
on 50,734 farm-hours), the ICON-DREAM-EU rows (six products on 50,041), the ECMWF rows (HRES and ENS
day 0 beside five products, from December 2024), and the nearest-weather-station rows (arms from a
Met Office station, from ERA5's 10 m wind, and from UKV, to December 2025). This script reads each
row set's saved `pooled` losses, without refitting anything, and writes for each

- every arm's own mean absolute error with a 95% interval,
- every arm's mean absolute error minus the reference arm's with a 95% interval, and
- every planned contrast the row set's report prints, the first arm minus the second (4 main, 2
  ICON-DREAM-EU, 3 ECMWF, 2 station), most of them not against the reference arm,

all from the same month-and-seed resampling `weather_products.py` uses. The reference arm is
`era5_wind`, except in the station block, where it is `era5_10m_wind`: the station block contrasts
the station's 10 m wind with ERA5's 10 m wind, not with ERA5's hub-height wind.

**The script stops before writing anything unless every number a row set's report already prints
is reproduced at the report's own precision**: each arm's error and interval, each printed contrast
against the reference arm on the same rows, and each planned contrast, at both settings where the
second is saved. A planned contrast the report prints and this script does not list, or the
reverse, stops the script too. It writes `report.md` and `intervals.parquet` into
`WIND_LEADERBOARD_DIR`, which it refuses to overwrite. `--check-only` reads and verifies but writes
nothing.

The generic scoring, checking, and writing code is `past_solar_leaderboard.py`'s. Each row set here
declares only where its report prints things: the heading above its table of errors, the sections
that print the planned contrasts and the second setting, and the sections whose contrasts come from
another fit.

Run it with `uv run studies/beam_diffuse_split/past_wind_leaderboard.py`. No generator's name,
identifier, or coordinates appears in its output.
"""

import sys
from typing import Final, NamedTuple

import past_solar_leaderboard as leaderboard
from sources import STUDY_DATA_DIR, UPDATE_OUTPUT_DIR, WIND_LEADERBOARD_DIR
from studies.charts import BlockArm, PlannedContrast, ProductFamily

REFERENCE_ARM: Final[str] = "era5_wind"
"""The arm every contrast is taken against, in every block but the station block."""

STATION_REFERENCE_ARM: Final[str] = "era5_10m_wind"
"""The station block's reference arm: ERA5's 10 m wind, the height the station measures at."""

ARM_LABELS: Final[dict[str, tuple[str, ProductFamily]]] = {
    "era5_wind": ("ERA5", "reanalysis"),
    "ukv_wind": ("UKV", "weather model"),
    "icon_d2_wind": ("ICON-D2", "weather model"),
    "icon_eu_wind": ("ICON-EU", "weather model"),
    "icon_global_wind": ("ICON global", "weather model"),
    "icon_dream_eu_wind": ("ICON-DREAM-EU", "reanalysis"),
    "hres_wind": ("ECMWF HRES", "weather model"),
    "ens_mean_day0_wind": ("ECMWF ENS day-0 mean", "weather model"),
    "station_wind": ("Nearest station", "station observations"),
    "era5_10m_wind": ("ERA5 10 m", "reanalysis"),
    "ukv_station_wind": ("UKV + nearest station", "station observations"),
    "ukv_padded_wind": ("UKV + its own 80 m wind", "weather model"),
}
"""Each arm's row label and product family, which sets its colour in `studies.charts`."""


class BlockSetting(NamedTuple):
    """What a block of the past-wind leaderboard states about its row set, beyond its arms.

    Attributes:
        hub_height: The wind heights the block's arms carry, for the figures' captions.
        reference_name: What the block's contrasts are against, as a zero rule and an axis title
            name it.
        note: A caveat about the block that its figures' captions state, or the empty string.
    """

    hub_height: str
    reference_name: str
    note: str = ""


def _arm(*, arm: str, reference: bool = False, label: str | None = None) -> BlockArm:
    """Return the block arm for one arm name, labelled and coloured from `ARM_LABELS`.

    `label` replaces the arm's label from `ARM_LABELS`, where a block needs to tell two ERA5 arms
    apart.
    """
    default_label, family = ARM_LABELS[arm]
    return BlockArm(arm=arm, label=label or default_label, family=family, reference=reference)


def _arms(
    *, arms: tuple[str, ...], reference_arm: str, labels: dict[str, str] | None = None
) -> tuple[BlockArm, ...]:
    """Return one block arm per name, the reference arm marked as such.

    `labels` maps an arm name to the label this block gives it instead of its `ARM_LABELS` label.
    """
    labels = labels or {}
    return tuple(
        _arm(arm=arm, reference=arm == reference_arm, label=labels.get(arm)) for arm in arms
    )


def _without_reference(*, arms: tuple[BlockArm, ...], reference_arm: str) -> tuple[BlockArm, ...]:
    """Drop the reference arm, which the zero rule stands for in a contrast against it."""
    return tuple(arm for arm in arms if arm.arm != reference_arm)


def _planned(
    *, arms: tuple[BlockArm, ...], pairs: tuple[tuple[str, str], ...]
) -> tuple[PlannedContrast, ...]:
    """Return each (first arm, second arm) pair as a `PlannedContrast`, first minus second."""
    by_arm = {arm.arm: arm for arm in arms}
    return tuple(PlannedContrast(by_arm[first], by_arm[second]) for first, second in pairs)


STATION_LABELS: Final[dict[str, str]] = {"era5_wind": "ERA5 100\u00a0m"}
"""Labels the station block gives an arm instead of its `ARM_LABELS` label.

The station block scores ERA5's 10 m wind as its reference beside ERA5's 100 m wind, so both cannot
be labelled `ERA5`.
"""

MAIN_ARMS: Final[tuple[BlockArm, ...]] = _arms(
    arms=("era5_wind", "ukv_wind", "icon_d2_wind", "icon_eu_wind", "icon_global_wind"),
    reference_arm=REFERENCE_ARM,
)
DREAM_ARMS: Final[tuple[BlockArm, ...]] = _arms(
    arms=(*(arm.arm for arm in MAIN_ARMS), "icon_dream_eu_wind"), reference_arm=REFERENCE_ARM
)
ECMWF_ARMS: Final[tuple[BlockArm, ...]] = _arms(
    arms=(*(arm.arm for arm in MAIN_ARMS), "hres_wind", "ens_mean_day0_wind"),
    reference_arm=REFERENCE_ARM,
)
STATION_ARMS: Final[tuple[BlockArm, ...]] = _arms(
    arms=(
        "station_wind",
        "era5_10m_wind",
        "era5_wind",
        "ukv_wind",
        "ukv_station_wind",
        "ukv_padded_wind",
    ),
    reference_arm=STATION_REFERENCE_ARM,
    labels=STATION_LABELS,
)
"""The arms each block scores, in the order its report lists them where the report has an order.

The ECMWF block leaves out the three alternative ENS interpolations its report also scores, and
the station block leaves out the mean-of-three-stations arm and the three post-review arms, each
of which the report marks exploratory.
"""

MAIN_PLANNED: Final[tuple[PlannedContrast, ...]] = _planned(
    arms=MAIN_ARMS,
    pairs=(
        ("icon_eu_wind", "era5_wind"),
        ("ukv_wind", "era5_wind"),
        ("icon_eu_wind", "ukv_wind"),
        ("icon_d2_wind", "icon_eu_wind"),
    ),
)
DREAM_PLANNED: Final[tuple[PlannedContrast, ...]] = _planned(
    arms=DREAM_ARMS,
    pairs=(("icon_dream_eu_wind", "era5_wind"), ("icon_dream_eu_wind", "icon_eu_wind")),
)
ECMWF_PLANNED: Final[tuple[PlannedContrast, ...]] = _planned(
    arms=ECMWF_ARMS,
    pairs=(
        ("hres_wind", "ukv_wind"),
        ("ens_mean_day0_wind", "ukv_wind"),
        ("hres_wind", "era5_wind"),
    ),
)
STATION_PLANNED: Final[tuple[PlannedContrast, ...]] = _planned(
    arms=STATION_ARMS,
    pairs=(("station_wind", "era5_10m_wind"), ("ukv_station_wind", "ukv_padded_wind")),
)
"""The planned contrasts of each row set, each the first arm's error minus the second's.

Each set is the one its row set's report prints under its planned-contrast heading, and
`score_row_set` stops unless the two agree. The main report's table of deciding contrasts also
prints ICON-D2 minus UKV, which the report labels exploratory; the main row set's
`exploratory_in_planned` declares it as such.
"""

POST_HOC_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("icon_eu_wind", "era5_wind"),
    ("icon_eu_wind", "ukv_wind"),
    ("icon_d2_wind", "icon_eu_wind"),
)
"""The main row set's planned contrasts that use the 80 m ICON arms, chosen after the first run.

The study plan specified the 100 m ICON arms. The report names these three contrasts as planned,
and the report and the figures mark them post hoc.
"""

DECIDING_SECTION: Final[str] = "Deciding contrasts, named before the run"
"""The heading of the main and ICON-DREAM-EU reports' table of planned contrasts."""

ECMWF_PLANNED_SECTION: Final[str] = "Planned contrasts P1 to P3, pooled over three farms"
"""The start of the two ECMWF report headings that print P1 to P3, at each setting."""

STATION_PLANNED_SECTION: Final[str] = "Planned contrasts S1 and S2 at the"
"""The start of the two station report headings that print S1 and S2, at each setting."""

ROW_SETS: Final[tuple[leaderboard.RowSet, ...]] = (
    leaderboard.RowSet(
        key="main",
        label="Main rows",
        directory=STUDY_DATA_DIR / "beam_diffuse_wind_products",
        printed_column="All sites",
        arm_suffix="_wind",
        leaderboard_arms=MAIN_ARMS,
        contrast_arms=_without_reference(arms=MAIN_ARMS, reference_arm=REFERENCE_ARM),
        planned_contrasts=MAIN_PLANNED,
        reference_arm=REFERENCE_ARM,
        intervals="none",
        planned_section=DECIDING_SECTION,
        second_planned_section="Sensitivity",
        second_scope="second setting",
        second_section="Sensitivity",
        other_fit_sections=(),
        exploratory_in_planned=(("icon_d2_wind", "ukv_wind"),),
        post_hoc_contrasts=POST_HOC_CONTRASTS,
        planned_heading=leaderboard.POST_HOC_PLANNED_CONTRAST_SECTION,
        hours_unit="farm-hours",
    ),
    leaderboard.RowSet(
        key="icon_dream_eu",
        label="ICON-DREAM-EU rows",
        directory=UPDATE_OUTPUT_DIR / "wind_icon_dream",
        printed_column="All sites",
        arm_suffix="_wind",
        leaderboard_arms=DREAM_ARMS,
        contrast_arms=_without_reference(arms=DREAM_ARMS, reference_arm=REFERENCE_ARM),
        planned_contrasts=DREAM_PLANNED,
        reference_arm=REFERENCE_ARM,
        intervals="table",
        planned_section=DECIDING_SECTION,
        second_planned_section="The same two contrasts at the second hyperparameter setting",
        second_scope="all",
        second_section="The same two contrasts at the second hyperparameter setting",
        other_fit_sections=(
            "The same two contrasts at the second hyperparameter setting",
            "Three published (unrefit) contrasts",
        ),
        hours_unit="farm-hours",
    ),
    leaderboard.RowSet(
        key="ecmwf",
        label="ECMWF rows",
        directory=UPDATE_OUTPUT_DIR / "ens_hres_past_wind",
        printed_column="MAE (pp of capacity)",
        arm_suffix="",
        leaderboard_arms=ECMWF_ARMS,
        contrast_arms=_without_reference(arms=ECMWF_ARMS, reference_arm=REFERENCE_ARM),
        planned_contrasts=ECMWF_PLANNED,
        reference_arm=REFERENCE_ARM,
        printed_decimals=4,
        leaderboard_section="Absolute error of every arm, pooled over three farms, primary setting",
        intervals="table",
        planned_section=f"{ECMWF_PLANNED_SECTION}, primary setting",
        second_planned_section=f"{ECMWF_PLANNED_SECTION}, second setting",
        second_scope="all",
        second_section=f"{ECMWF_PLANNED_SECTION}, second setting",
        other_fit_sections=(
            f"{ECMWF_PLANNED_SECTION}, second setting",
            "The same two contrasts, second setting",
        ),
        hours_unit="farm-hours",
    ),
    leaderboard.RowSet(
        key="station",
        label="Weather-station rows",
        directory=UPDATE_OUTPUT_DIR / "station_wind_arms",
        printed_column="MAE (pp of capacity)",
        arm_suffix="",
        leaderboard_arms=STATION_ARMS,
        contrast_arms=_without_reference(arms=STATION_ARMS, reference_arm=STATION_REFERENCE_ARM),
        planned_contrasts=STATION_PLANNED,
        reference_arm=STATION_REFERENCE_ARM,
        reference_label="ERA5's 10 m wind",
        leaderboard_section="Leaderboard at the primary setting",
        intervals="table",
        planned_section=f"{STATION_PLANNED_SECTION} primary setting",
        second_planned_section=f"{STATION_PLANNED_SECTION} second setting",
        second_scope="all",
        second_section=f"{STATION_PLANNED_SECTION} second setting",
        other_fit_sections=(f"{STATION_PLANNED_SECTION} second setting",),
        wide_contrast_tables=True,
        hours_unit="farm-hours",
    ),
)
"""The four headline row sets, in the order the leaderboard stacks them."""

DREAM_NOTE: Final[str] = (
    "planned contrasts were written after the main block's five products were scored."
)
"""The caveat the ICON-DREAM-EU block's figures state."""

BLOCK_SETTINGS: Final[dict[str, BlockSetting]] = {
    "main": BlockSetting(hub_height="100 m; ICON 80 m", reference_name="ERA5"),
    "icon_dream_eu": BlockSetting(
        hub_height="100 m; ICON 80 m; ICON-DREAM-EU 96 m",
        reference_name="ERA5",
        note=DREAM_NOTE,
    ),
    "ecmwf": BlockSetting(hub_height="100 m; ICON 80 m", reference_name="ERA5"),
    "station": BlockSetting(
        hub_height="10 m station and ERA5 arm; 100 m others", reference_name="ERA5's 10\u00a0m wind"
    ),
}
"""What each block's label and axis state beyond its dates and row count, by row set `key`."""

REPORT_TITLE: Final[str] = "Past-wind leaderboard and contrasts against ERA5"
"""The heading of the past-wind leaderboard's `report.md`."""

REPORT_INTRODUCTION: Final[str] = (
    "Every number is recomputed from the saved `pooled` losses of four row sets, by resampling "
    "whole months and a fitting seed. Each row set is scored on its own common farm-hours, so a "
    "value is comparable within a row set and not across row sets. Mean absolute error is a "
    "percentage of each farm's 99th-percentile output. Every contrast is against ERA5's "
    "hub-height wind, except in the weather-station rows, where it is against ERA5's 10 m wind. "
    "A contrast is planned where its row set's report names it before the run, and exploratory "
    "otherwise. `Second setting` is the same contrast at the second hyperparameter setting, "
    "shown only for planned contrasts and contrasts near the 5% line (an interval bound within "
    "20% of the interval's width from zero), and only where both arms have saved second-setting "
    "losses."
)
"""The paragraph under the report's heading."""


def main() -> int:
    """Score the four past-wind row sets, check them against their reports, and write them."""
    return leaderboard.run(
        row_sets=ROW_SETS,
        output_dir=WIND_LEADERBOARD_DIR,
        title=REPORT_TITLE,
        introduction=REPORT_INTRODUCTION,
        description=__doc__,
    )


if __name__ == "__main__":
    sys.exit(main())
