"""Tests for the pure functions of `studies/beam_diffuse_split/reanalysis_past_wind.py`.

Every frame is synthetic, so no test needs `data/`. Each test is built to fail on the bug it exists
for: a CERRA row set that is not one main-study hour in three, a block whose arms differ in width
(which hands the wider arm a free win), a fresh run that overwrites a published output, and a NORA3
block that runs without its 10 m speed.
"""

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import polars as pl
import pytest

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
SCRIPT_DIR: Final[Path] = REPO_ROOT / "studies" / "beam_diffuse_split"
DAY: Final[datetime] = datetime(2025, 6, 1, tzinfo=UTC)


def _load() -> ModuleType:
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "reanalysis_past_wind", SCRIPT_DIR / "reanalysis_past_wind.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wind = _load()


def _hours(*, days: int, step: int = 1, sites: tuple[str, ...] = ("W1", "W2")) -> pl.DataFrame:
    times = [DAY + timedelta(hours=hour) for hour in range(0, 24 * days, step)]
    return pl.DataFrame(
        [{"site": site, "time": time} for site in sites for time in times],
        schema={"site": pl.String, "time": pl.Datetime("us", "UTC")},
    )


def test_cerra_keeps_one_main_hour_in_three():
    base = _hours(days=2).with_columns(power_mw=1.0)
    values = _hours(days=2, step=3).with_columns(speed_hub_cerra=5.0)

    kept = wind.select_product_hours(base=base, wind=values, spec=wind.SPECS["cerra"])

    assert base.height == 96
    assert kept.height == 32
    assert kept["time"].dt.hour().unique().sort().to_list() == [0, 3, 6, 9, 12, 15, 18, 21]


def test_a_cerra_value_off_the_three_hourly_step_raises():
    base = _hours(days=1)
    values = _hours(days=1).filter(pl.col("time").dt.hour() != 4).with_columns(speed_hub_cerra=5.0)

    with pytest.raises(ValueError, match="not on its 3-hourly step"):
        wind.select_product_hours(base=base, wind=values, spec=wind.SPECS["cerra"])


def test_the_product_rows_stop_at_the_products_last_hour():
    spec = wind.SPECS["nora3"]
    times = [spec.end - timedelta(hours=1), spec.end]
    values = pl.DataFrame(
        {"site": ["W1", "W1"], "time": times, "speed_hub_nora3": [5.0, 6.0]},
        schema={"site": pl.String, "time": pl.Datetime("us", "UTC"), "speed_hub_nora3": pl.Float64},
    )
    base = values.select("site", "time")

    kept = wind.select_product_hours(base=base, wind=values, spec=spec)

    assert kept["time"].to_list() == [spec.end - timedelta(hours=1)]


@pytest.mark.parametrize("product", ["cerra", "nora3"])
@pytest.mark.parametrize("include_exploratory_arms", [True, False])
def test_every_arm_has_the_same_width_with_and_without_direction(
    product: str, *, include_exploratory_arms: bool
):
    spec = wind.SPECS[product]
    widths = {}
    for with_direction in (True, False):
        features = wind.arm_features(
            spec=spec,
            with_direction=with_direction,
            include_exploratory_arms=include_exploratory_arms,
        )
        wind.check_block_widths(features=features)
        widths[with_direction] = {len(columns) for columns in features.values()}

    assert widths == {True: {7}, False: {5}}


def test_without_direction_no_arm_has_a_direction_column():
    features = wind.arm_features(
        spec=wind.SPECS["cerra"], with_direction=False, include_exploratory_arms=True
    )

    assert not [
        column for columns in features.values() for column in columns if "direction" in column
    ]
    assert "speed_hub_era5" in features["era5_wind"]


def test_an_arm_that_lost_its_direction_pair_is_rejected():
    features = wind.arm_features(
        spec=wind.SPECS["cerra"], with_direction=True, include_exploratory_arms=False
    )
    features["cerra_wind"] = tuple(c for c in features["cerra_wind"] if "direction" not in c)

    with pytest.raises(ValueError, match="equal counts are required"):
        wind.check_block_widths(features=features)
    with pytest.raises(ValueError, match="equal counts are required"):
        wind.jobs(features=features)


def test_jobs_fit_every_arm_at_both_settings():
    features = wind.arm_features(
        spec=wind.SPECS["nora3"], with_direction=True, include_exploratory_arms=True
    )

    job_list = wind.jobs(features=features)

    assert len(job_list) == 2 * len(features)
    assert {(arm, setting) for arm, setting, *_ in job_list} == {
        (arm, setting) for arm in features for setting in ("pooled", "sensitivity")
    }
    assert all("colsample_bytree" not in job[4] for job in job_list)


@pytest.mark.parametrize("product", ["cerra", "nora3"])
def test_planned_contrasts_pair_the_new_product_with_era5_and_the_leader(product: str):
    features = wind.arm_features(
        spec=wind.SPECS[product], with_direction=True, include_exploratory_arms=False
    )

    planned = wind.PLANNED_CONTRASTS[product]

    assert planned == (
        (f"{product}_wind", "era5_wind"),
        (f"{product}_wind", f"{wind.LEADING_MAIN_PRODUCT}_wind"),
    )
    assert all(arm in features for pair in planned for arm in pair)


def test_no_planned_contrast_is_listed_as_exploratory():
    for product, spec in wind.SPECS.items():
        exploratory = wind.exploratory_contrasts(spec=spec, include_exploratory_arms=True)

        assert not set(exploratory) & set(wind.PLANNED_CONTRASTS[product])


def test_direction_becomes_a_sine_and_a_cosine_at_the_scored_height():
    spec = wind.SPECS["nora3"]
    raw = pl.DataFrame(
        {
            "site": ["W1"],
            "time": [DAY],
            "wind_speed_10m": [3.0],
            "wind_speed_100m": [8.0],
            "wind_direction_100m": [90.0],
        }
    )

    with_direction = wind.product_wind_columns(
        raw=raw, spec=spec, with_direction=True, include_exploratory_arms=False
    )
    without = wind.product_wind_columns(
        raw=raw.drop("wind_direction_100m"),
        spec=spec,
        with_direction=False,
        include_exploratory_arms=False,
    )

    assert with_direction.columns == [
        "site",
        "time",
        "speed_hub_nora3",
        "speed_10m_nora3",
        "direction_sin_nora3",
        "direction_cos_nora3",
    ]
    assert with_direction["direction_sin_nora3"].to_list() == pytest.approx([1.0])
    assert with_direction["direction_cos_nora3"].to_list() == pytest.approx([0.0], abs=1e-12)
    assert without.columns == ["site", "time", "speed_hub_nora3", "speed_10m_nora3"]


def test_cerra_direction_is_used_only_when_every_needed_file_exists(tmp_path: Path):
    for height in (100, 75):
        (tmp_path / wind.CERRA_DIRECTION_FILES[height]).touch()

    assert wind.cerra_has_direction(directory=tmp_path, heights=[100, 75])
    assert not wind.cerra_has_direction(directory=tmp_path, heights=[100, 75, 150])
    assert not wind.cerra_has_direction(directory=tmp_path / "empty", heights=[100])


def test_nora3_without_its_10m_file_raises_a_clear_error(tmp_path: Path):
    main = tmp_path / "NORA3_wind.parquet"
    main.touch()
    cells = pl.DataFrame({"site": ["W1"], "y_index": [1], "x_index": [1]})

    with pytest.raises(FileNotFoundError, match=r"10 m file .* has not been downloaded"):
        wind.read_nora3(
            cells=cells, heights=[100], path=main, path_10m=tmp_path / "missing.parquet"
        )


def _paths(*, directory: Path) -> Any:
    return wind.OutputPaths(
        losses=directory / "losses.parquet",
        fingerprint=directory / "losses.fingerprint",
        report=directory / "report.md",
    )


@pytest.mark.parametrize("existing", ["losses", "fingerprint", "report"])
def test_a_fresh_run_refuses_to_overwrite_any_output_before_fitting(tmp_path: Path, existing: str):
    paths = _paths(directory=tmp_path)
    getattr(paths, existing).write_text("published")
    fitted = []

    def fit(**_: object) -> pl.DataFrame:
        fitted.append(True)
        return pl.DataFrame({"arm": ["a"]})

    with pytest.raises(FileExistsError, match="superseded"):
        wind.fit_and_save(
            frame=pl.DataFrame(), job_list=[], fingerprint="abc", paths=paths, fit=fit
        )

    assert not fitted
    assert getattr(paths, existing).read_text() == "published"


def test_a_fresh_run_writes_the_losses_and_the_fingerprint(tmp_path: Path):
    paths = _paths(directory=tmp_path / "wind_cerra")

    losses = wind.fit_and_save(
        frame=pl.DataFrame(),
        job_list=[],
        fingerprint="abc",
        paths=paths,
        fit=lambda **_: pl.DataFrame({"arm": ["a"]}),
    )

    assert losses.height == 1
    assert pl.read_parquet(paths.losses).height == 1
    assert paths.fingerprint.read_text() == "abc"
    assert not paths.report.exists()
