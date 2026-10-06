"""Pin where every study folder lives, and that scripts do not join a folder's name themselves.

Every folder under `data/studies/` is named once, in `studies.sources`, so that moving a folder
changes one constant. These tests hold the constants to the paths the scripts used before they
existed, and fail on any script that joins one of those folder names onto a path with `/` and a
string literal. The scan does not see an f-string, `joinpath`, `os.path.join`, `Path(a, b)`, or a
name held in a variable.
"""

import ast
from pathlib import Path
from typing import Final

import pytest
from studies.sources import (
    DOWNLOADS_DIR,
    NFC_BATCH_DIRS,
    NFC_DIR,
    NFC_STAMP_GLOB,
    NFC_STUDY_DIR,
    PER_STUDY_DIR,
    REPO_DATA_DIR,
    SCRATCH_DIR,
    STUDIES_DATA_DIR,
    TRIAL_AREA_BOX_PATH,
    nfc_batch_dir_for,
    per_study_relative,
    previous_runs_product_dir_for,
    product_dir_for,
    site_points_dir_for,
    study_dir_for,
)

from studies import sources

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

PRODUCT_FOLDERS: Final[dict[str, str]] = {
    "ERA5_PRODUCT_DIR": "downloads/reanalysis/ERA5",
    "ERA5_WIND_2019_2023_PRODUCT_DIR": "downloads/reanalysis/ERA5-WIND-2019-2023",
    "CAMS_PRODUCT_DIR": "downloads/reanalysis/CAMS",
    "ENS_PRODUCT_DIR": "downloads/NWP/ENS_SITE_EXTRACT",
    "CERRA_PRODUCT_DIR": "downloads/reanalysis/CERRA",
    "NORA3_PRODUCT_DIR": "downloads/reanalysis/NORA3",
    "NORA3_10M_PRODUCT_DIR": "downloads/reanalysis/NORA3_10m",
    "ICON_DREAM_EU_PRODUCT_DIR": "downloads/reanalysis/ICON-DREAM-EU",
    "MIDAS_OPEN_PRODUCT_DIR": "downloads/observations/MIDAS-OPEN",
    "SARAH_3_PRODUCT_DIR": "downloads/observations/SARAH-3",
    "ECMWF_IFS_HRES_PRODUCT_DIR": "downloads/NWP/OPEN-METEO-PREVIOUS-RUNS/ECMWF-IFS-HRES",
    "ECMWF_IFS_SINGLE_RUNS_PRODUCT_DIR": "downloads/NWP/ECMWF-IFS-SINGLE-RUNS",
    "ECMWF_AIFS_PRODUCT_DIR": "downloads/NWP/ECMWF-AIFS",
    "ECMWF_AIFS_ENS_PRODUCT_DIR": "downloads/NWP/ECMWF-AIFS-ENS",
    "GEFS_WINDOW_DIR": "downloads/NWP/windows/GEFS_window_2024-11-01_None",
    "GFS_PRODUCT_DIR": "downloads/NWP/GFS",
    "GFS_WINDOW_DIR": "downloads/NWP/windows/GFS_window_2025-07-01_2025-07-02",
    "WEATHERNEXT3_PRODUCT_DIR": "downloads/NWP/WeatherNext3",
    "UKV_CEDA_T120_PRODUCT_DIR": "weather/UKV-CEDA-T120",
    "OPEN_METEO_ENSEMBLE_MEANS_PRODUCT_DIR": "downloads/NWP/OPEN-METEO-ENSEMBLE-MEANS",
}
"""Each product constant against its folder under `data/studies/`.

The UKV-on-CEDA T120 store has not moved, so its row still names `weather/`.
"""

PREVIOUS_RUNS_FOLDERS: Final[tuple[str, ...]] = (
    "AROME-FRANCE",
    "ARPEGE-EUROPE",
    "DMI-HARMONIE-AROME",
    "ECMWF-IFS-025",
    "ECMWF-IFS-HRES",
    "GFS-SEAMLESS",
    "ICON-D2",
    "ICON-EU",
    "ICON-GLOBAL",
    "KNMI-HARMONIE-AROME",
    "UKV",
)
"""The eleven Open-Meteo Previous Runs products, as the folders they were in under `weather/`."""

OLD_STUDY_FOLDERS: Final[dict[str, str]] = {
    "STUDY_DATA_DIR": "beam_diffuse_split",
    "ENS_FORECAST_HORIZONS_DIR": "ens_forecast_horizons",
    "ENS_FORECAST_HORIZONS_DAY4_DIR": "ens_forecast_horizons_day4",
    "OPEN_METEO_ENSEMBLE_MEANS_DIR": "open_meteo_ensemble_means",
    "OPEN_METEO_ENS_GAP_DIR": "open_meteo_ens_gap",
    "ICON_EU_COMPARE_DIR": "icon_eu_compare",
    "ERA5_WIND_COMPARE_DIR": "era5_wind_compare",
    "ENS_BACKFILL_PILOT_DIR": "ens_backfill_pilot",
    "CERRA_WIND_LEVELS_DIR": "per_study/cerra_wind/levels",
    "CERRA_WIND_LEVELS_POST_HOC_DIR": "per_study/cerra_wind/levels_post_hoc",
    "CERRA_WIND_LEVELS_SHEAR_DIR": "per_study/cerra_wind/shear",
    "CERRA_WIND_DIRECTION_DIR": "per_study/cerra_wind/direction",
    "UKV_CEDA_BLENDS_DIR": "ukv_ceda_blends",
    "UKV_CEDA_BLENDS_RUN15_DIR": "ukv_ceda_blends_run15",
    "NFC_STUDY_DIR": "per_study/nwp_forecast_comparison",
    "NFC_DIR": "per_study/nwp_forecast_comparison/original",
    "NFC_AIFS_DIR": "per_study/nwp_forecast_comparison/aifs",
    "NFC_AIFS_BLENDS_DIR": "per_study/nwp_forecast_comparison/aifs_blends",
    "NFC_AIFS_EXTRA_DAYS_DIR": "per_study/nwp_forecast_comparison/aifs_extra_days",
    "NFC_DAY4_SHARED_DIR": "per_study/nwp_forecast_comparison/day4_shared",
    "NFC_DAY5_AIFS_WN3_DIR": "per_study/nwp_forecast_comparison/day5_aifs_wn3",
    "NFC_LEADERBOARD_BY_DAY_DIR": "per_study/nwp_forecast_comparison/leaderboard_by_day",
    "NFC_LEADERBOARD_BY_DAY_FIG3_DIR": "per_study/nwp_forecast_comparison/leaderboard_by_day_fig3",
    "NFC_LEADS_DIR": "per_study/nwp_forecast_comparison/leads",
    "NFC_LEADS_DAY10_DIR": "per_study/nwp_forecast_comparison/leads_day10",
    "NFC_LEADS_DAY10B_DIR": "per_study/nwp_forecast_comparison/leads_day10b",
    "NFC_LEADS_DAY10C_DIR": "per_study/nwp_forecast_comparison/leads_day10c",
    "NFC_LEADS_DAY10D_DIR": "per_study/nwp_forecast_comparison/leads_day10d",
    "NFC_P4_SEEDS_DIR": "per_study/nwp_forecast_comparison/p4_seeds",
    "NFC_PRODUCT_BLENDS_DIR": "per_study/nwp_forecast_comparison/product_blends",
    "NFC_PRODUCT_BLENDS_REPORT_DIR": "per_study/nwp_forecast_comparison/product_blends_report",
    "NFC_VS_ENS_DOTS_DIR": "per_study/nwp_forecast_comparison/vs_ens_dots",
    "NFC_VS_ENS_DOTS_ALL_DAYS_DIR": "per_study/nwp_forecast_comparison/vs_ens_dots_all_days",
    "NFC_VS_ENS_DOTS_BLENDS_DIR": "per_study/nwp_forecast_comparison/vs_ens_dots_blends",
    "NFC_VS_ENS_DOTS_BLENDS_FINAL_DIR": (
        "per_study/nwp_forecast_comparison/vs_ens_dots_blends_final"
    ),
    "NFC_VS_ENS_DOTS_FINAL_DIR": "per_study/nwp_forecast_comparison/vs_ens_dots_final",
    "NFC_WN3_DIR": "per_study/nwp_forecast_comparison/wn3",
    "NFC_WN3_EXTRA_DAYS_DIR": "per_study/nwp_forecast_comparison/wn3_extra_days",
}
"""Each study-folder constant against its folder under `data/studies/`."""

NFC_BATCH_CONSTANTS: Final[frozenset[str]] = frozenset(
    constant
    for constant in OLD_STUDY_FOLDERS
    if constant.startswith("NFC_") and constant != "NFC_STUDY_DIR"
)
"""The batch folders of the NWP forecast comparison, whose names (`aifs`, `leads`, `wn3`, ...) are
too common to flag in a scan for a hand-written folder name."""

HAND_WRITTEN_FOLDER_NAMES: Final[frozenset[str]] = frozenset(
    {
        *(Path(folder).name for folder in PRODUCT_FOLDERS.values()),
        *(
            Path(folder).name
            for constant, folder in OLD_STUDY_FOLDERS.items()
            if constant not in NFC_BATCH_CONSTANTS
        ),
        "nwp_forecast_comparison",
        *PREVIOUS_RUNS_FOLDERS,
        "downloads",
        "reanalysis",
        "observations",
        "windows",
        "site_points",
        "per_study",
        "weather",
        "NGED-ANM",
        "_trial_area_box.json",
    }
)
"""Folder names that only `studies.sources` may join onto a path."""

SCANNED_FOLDERS: Final[tuple[Path, ...]] = (
    REPO_ROOT / "studies",
    REPO_ROOT / "packages" / "studies" / "src",
)
FROZEN_FOLDER: Final[str] = "era_fold_design"


@pytest.mark.parametrize(("constant", "folder"), PRODUCT_FOLDERS.items())
def test_each_product_constant_is_the_folder_the_data_was_moved_to(constant: str, folder: str):
    assert getattr(sources, constant) == REPO_DATA_DIR / "studies" / folder


@pytest.mark.parametrize("model", PREVIOUS_RUNS_FOLDERS)
def test_each_previous_runs_product_has_a_folder_of_its_own_under_the_previous_runs_folder(
    model: str,
):
    expected = DOWNLOADS_DIR / "NWP" / "OPEN-METEO-PREVIOUS-RUNS" / model

    assert product_dir_for(product=model) == expected
    assert previous_runs_product_dir_for(product=model) == expected
    assert site_points_dir_for(product=model) == expected / "site_points"


@pytest.mark.parametrize(("constant", "folder"), OLD_STUDY_FOLDERS.items())
def test_each_study_constant_is_the_folder_the_study_has_now(constant: str, folder: str):
    assert getattr(sources, constant) == REPO_DATA_DIR / "studies" / folder


def test_the_remaining_constants_are_the_paths_the_data_was_moved_to():
    assert STUDIES_DATA_DIR == REPO_DATA_DIR / "studies"
    assert DOWNLOADS_DIR == STUDIES_DATA_DIR / "downloads"
    assert PER_STUDY_DIR == STUDIES_DATA_DIR
    assert sources.WEATHER_DATA_DIR == STUDIES_DATA_DIR / "weather"
    assert sources.ANM_DATA_DIR == DOWNLOADS_DIR / "observations" / "NGED-ANM"
    assert SCRATCH_DIR == STUDIES_DATA_DIR / "_scratch"
    assert TRIAL_AREA_BOX_PATH == STUDIES_DATA_DIR / "weather" / "_trial_area_box.json"


def test_the_per_site_frames_of_a_product_sit_in_its_site_points_folder():
    ukv = DOWNLOADS_DIR / "NWP" / "OPEN-METEO-PREVIOUS-RUNS" / "UKV" / "site_points"

    assert sources.point_output_path_for(source="ukv") == ukv / "beam_diffuse_ukv.parquet"
    assert sources.temperature_site_b_path_for(source="icon-d2") == (
        DOWNLOADS_DIR
        / "NWP"
        / "OPEN-METEO-PREVIOUS-RUNS"
        / "ICON-D2"
        / "site_points"
        / "temperature_2m_site_b.parquet"
    )
    assert sources.point_output_path_for(source="sarah-3") == (
        DOWNLOADS_DIR / "observations" / "SARAH-3" / "site_points" / "beam_diffuse_sarah-3.parquet"
    )
    assert sources.ERA5_SITE_POINTS_DIR == sources.ERA5_PRODUCT_DIR / "site_points"
    assert sources.CAMS_SITE_POINTS_DIR == sources.CAMS_PRODUCT_DIR / "site_points"
    assert sources.ENS_SITE_POINTS_DIR == sources.ENS_PRODUCT_DIR / "site_points"


def test_a_window_of_dates_is_filed_under_the_windows_folder():
    name = "GEFS_window_2099-01-01_2099-01-02"

    assert product_dir_for(product=name) == DOWNLOADS_DIR / "NWP" / "windows" / name


def test_an_unknown_product_name_raises_rather_than_naming_a_stray_folder():
    with pytest.raises(ValueError, match="unknown product"):
        product_dir_for(product="ICON-D3")
    with pytest.raises(ValueError, match="not an Open-Meteo Previous Runs product"):
        previous_runs_product_dir_for(product="ERA5")


def test_every_product_the_scripts_name_is_filed_under_downloads_except_the_ukv_ceda_stores():
    names = (
        *sources.PREVIOUS_RUNS_PRODUCTS,
        *sources.NWP_PRODUCT_NAMES,
        *sources.REANALYSIS_PRODUCT_NAMES,
        *sources.OBSERVATION_PRODUCT_NAMES,
        "ENS",
        "WeatherNext3_trial_area",
    )

    assert all(DOWNLOADS_DIR in product_dir_for(product=name).parents for name in names)
    assert product_dir_for(product="UKV-CEDA-T120").parent == sources.WEATHER_DATA_DIR


def test_the_batch_folders_are_the_twenty_two_siblings_of_the_original_batch():
    assert len(NFC_BATCH_DIRS) == 22
    assert len(set(NFC_BATCH_DIRS)) == 22
    assert NFC_DIR not in NFC_BATCH_DIRS
    assert {folder.parent for folder in NFC_BATCH_DIRS} == {NFC_STUDY_DIR}
    assert NFC_DIR.parent == NFC_STUDY_DIR
    assert NFC_STUDY_DIR == STUDIES_DATA_DIR / "per_study" / "nwp_forecast_comparison"
    assert nfc_batch_dir_for(batch="wn3") == sources.NFC_WN3_DIR


def test_every_batch_constant_is_listed_among_the_batch_folders():
    batch_constants = {
        value
        for name, value in vars(sources).items()
        if name.startswith("NFC_")
        and name.endswith("_DIR")
        and name not in {"NFC_DIR", "NFC_STUDY_DIR"}
    }

    assert batch_constants == set(NFC_BATCH_DIRS)


def test_the_name_helpers_join_one_folder_name_onto_their_layer():
    assert product_dir_for(product="ICON-D2") == sources.PREVIOUS_RUNS_DIR / "ICON-D2"
    assert previous_runs_product_dir_for(product="ICON-D2") == product_dir_for(product="ICON-D2")
    assert study_dir_for(study="x") == PER_STUDY_DIR / "x"
    assert per_study_relative(folder=study_dir_for(study="x")) == Path("x")


def test_the_stamp_glob_reads_every_batch_folder_and_nothing_else(tmp_path: Path):
    study = tmp_path / NFC_STUDY_DIR.relative_to(PER_STUDY_DIR)
    for folder in (study / "a", study / "b", tmp_path / "ukv_ceda_blends"):
        folder.mkdir(parents=True)
        (folder / "solar_x_losses.json").write_text("{}")
    (study / "a" / "superseded").mkdir()
    (study / "a" / "superseded" / "solar_y_losses.json").write_text("{}")

    found = sorted(path.parent.name for path in tmp_path.glob(NFC_STAMP_GLOB))

    assert found == ["a", "b"]


@pytest.mark.skipif(
    not STUDIES_DATA_DIR.exists(),
    reason="the private study data is not in this checkout",
)
def test_every_folder_constant_names_a_folder_that_exists_on_disk():
    constants = {**PRODUCT_FOLDERS, **OLD_STUDY_FOLDERS}
    missing = [name for name in constants if not getattr(sources, name).exists()]

    assert missing == []
    assert TRIAL_AREA_BOX_PATH.exists()


@pytest.mark.skipif(
    not STUDIES_DATA_DIR.exists(),
    reason="the private study data is not in this checkout",
)
def test_the_stamp_glob_finds_72_stamps_on_disk_and_no_batch_folder_is_left_at_the_old_paths():
    stamps = {path.resolve() for path in PER_STUDY_DIR.glob(NFC_STAMP_GLOB)}

    assert len(stamps) == 72
    assert sorted(STUDIES_DATA_DIR.glob("nwp_forecast_comparison_*")) == []


def _joined_folder_names(*, path: Path) -> list[tuple[int, str]]:
    """Return every line where a script joins one of `HAND_WRITTEN_FOLDER_NAMES` onto a path."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
            continue
        name = node.right.value if isinstance(node.right, ast.Constant) else None
        if isinstance(name, str) and name in HAND_WRITTEN_FOLDER_NAMES:
            found.append((node.lineno, name))
    return found


def test_no_script_joins_a_data_folder_name_onto_a_path_with_a_slash_and_a_literal():
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}: / {name!r}"
        for root in SCANNED_FOLDERS
        for path in sorted(root.rglob("*.py"))
        if FROZEN_FOLDER not in path.parts and path.name != "sources.py"
        for line, name in _joined_folder_names(path=path)
    ]

    assert offenders == []


def test_the_scan_finds_a_hand_written_folder_name(tmp_path: Path):
    script = tmp_path / "script.py"
    script.write_text('DIR = ROOT / "studies" / "weather" / "ERA5"\nOK = ROOT / "reports"\n')

    assert _joined_folder_names(path=script) == [(1, "ERA5"), (1, "weather")]
