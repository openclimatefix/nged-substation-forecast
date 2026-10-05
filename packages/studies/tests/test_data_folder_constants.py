"""Pin where every study folder lives, and that no script spells a folder's name itself.

Every folder under `data/studies/` is named once, in `studies.sources`, so that moving a folder
changes one constant. These tests hold the constants to the paths the scripts used before they
existed, and fail on any script that joins a folder name onto a path by hand.
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
    PER_STUDY_DIR,
    REPO_DATA_DIR,
    SCRATCH_DIR,
    STUDIES_DATA_DIR,
    TRIAL_AREA_BOX_PATH,
    nfc_batch_dir_for,
    per_study_relative,
    previous_runs_product_dir_for,
    product_dir_for,
    study_dir_for,
)

from studies import sources

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

OLD_PRODUCT_FOLDERS: Final[dict[str, str]] = {
    "ERA5_PRODUCT_DIR": "ERA5",
    "ERA5_WIND_2019_2023_PRODUCT_DIR": "ERA5-WIND-2019-2023",
    "CAMS_PRODUCT_DIR": "CAMS",
    "ENS_PRODUCT_DIR": "ENS",
    "CERRA_PRODUCT_DIR": "CERRA",
    "NORA3_PRODUCT_DIR": "NORA3",
    "NORA3_10M_PRODUCT_DIR": "NORA3_10m",
    "ICON_DREAM_EU_PRODUCT_DIR": "ICON-DREAM-EU",
    "MIDAS_OPEN_PRODUCT_DIR": "MIDAS-OPEN",
    "SARAH_3_PRODUCT_DIR": "SARAH-3",
    "ECMWF_IFS_HRES_PRODUCT_DIR": "ECMWF-IFS-HRES",
    "ECMWF_IFS_SINGLE_RUNS_PRODUCT_DIR": "ECMWF-IFS-SINGLE-RUNS",
    "ECMWF_AIFS_PRODUCT_DIR": "ECMWF-AIFS",
    "ECMWF_AIFS_ENS_PRODUCT_DIR": "ECMWF-AIFS-ENS",
    "GEFS_WINDOW_DIR": "GEFS_window_2024-11-01_None",
    "GFS_PRODUCT_DIR": "GFS",
    "GFS_WINDOW_DIR": "GFS_window_2025-07-01_2025-07-02",
    "WEATHERNEXT3_PRODUCT_DIR": "WeatherNext3_trial_area",
    "UKV_CEDA_T120_PRODUCT_DIR": "UKV-CEDA-T120",
    "OPEN_METEO_ENSEMBLE_MEANS_PRODUCT_DIR": "OPEN-METEO-ENSEMBLE-MEANS",
}
"""Each weather-product constant against the folder name under `data/studies/weather/` that the
scripts spelled out before the constants existed."""

OLD_STUDY_FOLDERS: Final[dict[str, str]] = {
    "STUDY_DATA_DIR": "beam_diffuse_split",
    "ENS_FORECAST_HORIZONS_DIR": "ens_forecast_horizons",
    "ENS_FORECAST_HORIZONS_DAY4_DIR": "ens_forecast_horizons_day4",
    "OPEN_METEO_ENSEMBLE_MEANS_DIR": "open_meteo_ensemble_means",
    "OPEN_METEO_ENS_GAP_DIR": "open_meteo_ens_gap",
    "ICON_EU_COMPARE_DIR": "icon_eu_compare",
    "ERA5_WIND_COMPARE_DIR": "era5_wind_compare",
    "ENS_BACKFILL_PILOT_DIR": "ens_backfill_pilot",
    "CERRA_WIND_LEVELS_DIR": "cerra_wind_levels",
    "CERRA_WIND_LEVELS_POST_HOC_DIR": "cerra_wind_levels_post_hoc",
    "CERRA_WIND_LEVELS_SHEAR_DIR": "cerra_wind_levels_shear",
    "CERRA_WIND_DIRECTION_DIR": "cerra_wind_direction",
    "UKV_CEDA_BLENDS_DIR": "ukv_ceda_blends",
    "UKV_CEDA_BLENDS_RUN15_DIR": "ukv_ceda_blends_run15",
    "NFC_DIR": "nwp_forecast_comparison",
    "NFC_AIFS_DIR": "nwp_forecast_comparison_aifs",
    "NFC_AIFS_BLENDS_DIR": "nwp_forecast_comparison_aifs_blends",
    "NFC_AIFS_EXTRA_DAYS_DIR": "nwp_forecast_comparison_aifs_extra_days",
    "NFC_DAY4_SHARED_DIR": "nwp_forecast_comparison_day4_shared",
    "NFC_DAY5_AIFS_WN3_DIR": "nwp_forecast_comparison_day5_aifs_wn3",
    "NFC_LEADERBOARD_BY_DAY_DIR": "nwp_forecast_comparison_leaderboard_by_day",
    "NFC_LEADERBOARD_BY_DAY_FIG3_DIR": "nwp_forecast_comparison_leaderboard_by_day_fig3",
    "NFC_LEADS_DIR": "nwp_forecast_comparison_leads",
    "NFC_LEADS_DAY10_DIR": "nwp_forecast_comparison_leads_day10",
    "NFC_LEADS_DAY10B_DIR": "nwp_forecast_comparison_leads_day10b",
    "NFC_LEADS_DAY10C_DIR": "nwp_forecast_comparison_leads_day10c",
    "NFC_LEADS_DAY10D_DIR": "nwp_forecast_comparison_leads_day10d",
    "NFC_P4_SEEDS_DIR": "nwp_forecast_comparison_p4_seeds",
    "NFC_PRODUCT_BLENDS_DIR": "nwp_forecast_comparison_product_blends",
    "NFC_PRODUCT_BLENDS_REPORT_DIR": "nwp_forecast_comparison_product_blends_report",
    "NFC_VS_ENS_DOTS_DIR": "nwp_forecast_comparison_vs_ens_dots",
    "NFC_VS_ENS_DOTS_ALL_DAYS_DIR": "nwp_forecast_comparison_vs_ens_dots_all_days",
    "NFC_VS_ENS_DOTS_BLENDS_DIR": "nwp_forecast_comparison_vs_ens_dots_blends",
    "NFC_VS_ENS_DOTS_BLENDS_FINAL_DIR": "nwp_forecast_comparison_vs_ens_dots_blends_final",
    "NFC_VS_ENS_DOTS_FINAL_DIR": "nwp_forecast_comparison_vs_ens_dots_final",
    "NFC_WN3_DIR": "nwp_forecast_comparison_wn3",
    "NFC_WN3_EXTRA_DAYS_DIR": "nwp_forecast_comparison_wn3_extra_days",
}
"""Each study-folder constant against its folder name under `data/studies/` before the constants."""

HAND_WRITTEN_FOLDER_NAMES: Final[frozenset[str]] = frozenset(
    {
        *OLD_PRODUCT_FOLDERS.values(),
        *OLD_STUDY_FOLDERS.values(),
        "weather",
        "anm",
        "_trial_area_box.json",
    }
)
"""Folder names that only `studies.sources` may join onto a path."""

SCANNED_FOLDERS: Final[tuple[Path, ...]] = (
    REPO_ROOT / "studies",
    REPO_ROOT / "packages" / "studies" / "src",
)
FROZEN_FOLDER: Final[str] = "era_fold_design"


@pytest.mark.parametrize(("constant", "folder"), OLD_PRODUCT_FOLDERS.items())
def test_each_product_constant_is_the_path_the_scripts_spelled_out_before(
    constant: str, folder: str
):
    assert getattr(sources, constant) == REPO_DATA_DIR / "studies" / "weather" / folder


@pytest.mark.parametrize(("constant", "folder"), OLD_STUDY_FOLDERS.items())
def test_each_study_constant_is_the_path_the_scripts_spelled_out_before(constant: str, folder: str):
    assert getattr(sources, constant) == REPO_DATA_DIR / "studies" / folder


def test_the_remaining_constants_are_the_paths_the_scripts_spelled_out_before():
    assert STUDIES_DATA_DIR == REPO_DATA_DIR / "studies"
    assert sources.WEATHER_DATA_DIR == REPO_DATA_DIR / "studies" / "weather"
    assert sources.ANM_DATA_DIR == REPO_DATA_DIR / "studies" / "anm"
    assert SCRATCH_DIR == REPO_DATA_DIR / "_scratch"
    assert TRIAL_AREA_BOX_PATH == REPO_DATA_DIR / "studies" / "weather" / "_trial_area_box.json"
    assert sources.point_output_path_for(source="ukv") == (
        REPO_DATA_DIR / "studies" / "weather" / "UKV" / "beam_diffuse_ukv.parquet"
    )


def test_the_two_new_layers_equal_the_studies_folder_until_the_data_moves():
    assert DOWNLOADS_DIR == STUDIES_DATA_DIR
    assert PER_STUDY_DIR == STUDIES_DATA_DIR


def test_the_batch_folders_are_the_twenty_two_siblings_of_the_original_batch():
    assert len(NFC_BATCH_DIRS) == 22
    assert len(set(NFC_BATCH_DIRS)) == 22
    assert NFC_DIR not in NFC_BATCH_DIRS
    assert {folder.parent for folder in NFC_BATCH_DIRS} == {PER_STUDY_DIR}
    assert all(folder.name.startswith("nwp_forecast_comparison_") for folder in NFC_BATCH_DIRS)
    assert nfc_batch_dir_for(batch="wn3") == sources.NFC_WN3_DIR


def test_every_batch_constant_is_listed_among_the_batch_folders():
    batch_constants = {
        value
        for name, value in vars(sources).items()
        if name.startswith("NFC_") and name.endswith("_DIR") and name != "NFC_DIR"
    }

    assert batch_constants == set(NFC_BATCH_DIRS)


def test_the_name_helpers_join_one_folder_name_onto_their_layer():
    assert product_dir_for(product="ICON-D2") == sources.WEATHER_DATA_DIR / "ICON-D2"
    assert previous_runs_product_dir_for(product="ICON-D2") == product_dir_for(product="ICON-D2")
    assert study_dir_for(study="x") == PER_STUDY_DIR / "x"
    assert per_study_relative(folder=study_dir_for(study="x")) == Path("x")


def test_the_stamp_glob_reads_every_batch_folder_and_nothing_else(tmp_path: Path):
    for folder in ("nwp_forecast_comparison_a", "ukv_ceda_blends", "nwp_forecast_comparison"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "solar_x_losses.json").write_text("{}")
    (tmp_path / "nwp_forecast_comparison_a" / "superseded").mkdir()
    (tmp_path / "nwp_forecast_comparison_a" / "superseded" / "solar_y_losses.json").write_text("{}")

    found = sorted(path.parent.name for path in tmp_path.glob(NFC_STAMP_GLOB))

    assert found == ["nwp_forecast_comparison_a"]


@pytest.mark.skipif(
    not (STUDIES_DATA_DIR / "weather").exists(),
    reason="the private study data is not in this checkout",
)
def test_every_folder_constant_names_a_folder_that_exists_on_disk():
    constants = {**OLD_PRODUCT_FOLDERS, **OLD_STUDY_FOLDERS}
    missing = [name for name in constants if not getattr(sources, name).exists()]

    assert missing == []
    assert TRIAL_AREA_BOX_PATH.exists()


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


def test_no_script_joins_a_data_folder_name_onto_a_path_by_hand():
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
