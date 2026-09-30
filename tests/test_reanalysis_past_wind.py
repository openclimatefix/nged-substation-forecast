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


@pytest.mark.parametrize(("product", "kept"), [("cerra", 32), ("nora3", 96)])
def test_the_row_count_follows_the_products_own_step(product: str, kept: int):
    spec = wind.SPECS[product]
    base = _hours(days=2).with_columns(power_mw=1.0)
    values = _hours(days=2, step=spec.step_hours).with_columns(speed_hub=5.0)

    assert wind.select_product_hours(base=base, wind=values, spec=spec).height == kept


def test_hourly_values_are_rejected_for_a_three_hourly_product_and_kept_for_an_hourly_one():
    base = _hours(days=1)
    hourly = _hours(days=1).with_columns(speed_hub=5.0)

    assert wind.select_product_hours(base=base, wind=hourly, spec=wind.SPECS["nora3"]).height == 48
    with pytest.raises(ValueError, match="not on its 3-hourly step"):
        wind.select_product_hours(base=base, wind=hourly, spec=wind.SPECS["cerra"])


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
@pytest.mark.parametrize("with_extra_arms", [True, False])
def test_every_arm_has_the_same_width_with_and_without_direction(
    product: str, *, with_extra_arms: bool
):
    spec = wind.SPECS[product]
    widths = {}
    for with_direction in (True, False):
        features = wind.arm_features(
            spec=spec,
            with_direction=with_direction,
            extra_heights_m=spec.extra_heights_m if with_extra_arms else (),
        )
        wind.check_block_widths(features=features)
        widths[with_direction] = {len(columns) for columns in features.values()}

    assert widths == {True: {7}, False: {5}}


def test_without_direction_no_arm_has_a_direction_column():
    features = wind.arm_features(
        spec=wind.SPECS["cerra"],
        with_direction=False,
        extra_heights_m=wind.SPECS["cerra"].extra_heights_m,
    )

    assert not [
        column for columns in features.values() for column in columns if "direction" in column
    ]
    assert "speed_hub_era5" in features["era5_wind"]


def test_an_arm_that_lost_its_direction_pair_is_rejected():
    features = wind.arm_features(spec=wind.SPECS["cerra"], with_direction=True, extra_heights_m=())
    features["cerra_wind"] = tuple(c for c in features["cerra_wind"] if "direction" not in c)

    with pytest.raises(ValueError, match="equal counts are required"):
        wind.check_block_widths(features=features)
    with pytest.raises(ValueError, match="equal counts are required"):
        wind.jobs(features=features)


def test_jobs_fit_every_arm_at_both_settings():
    features = wind.arm_features(
        spec=wind.SPECS["nora3"],
        with_direction=True,
        extra_heights_m=wind.SPECS["nora3"].extra_heights_m,
    )

    job_list = wind.jobs(features=features)

    assert len(job_list) == 2 * len(features)
    assert {(arm, setting) for arm, setting, *_ in job_list} == {
        (arm, setting) for arm in features for setting in ("pooled", "sensitivity")
    }
    assert all("colsample_bytree" not in job[4] for job in job_list)


@pytest.mark.parametrize("product", ["cerra", "nora3"])
def test_planned_contrasts_pair_the_new_product_with_era5_and_the_leader(product: str):
    features = wind.arm_features(spec=wind.SPECS[product], with_direction=True, extra_heights_m=())

    planned = wind.PLANNED_CONTRASTS[product]

    assert planned == (
        (f"{product}_wind", "era5_wind"),
        (f"{product}_wind", f"{wind.LEADING_MAIN_PRODUCT}_wind"),
    )
    assert all(arm in features for pair in planned for arm in pair)


def test_no_planned_contrast_is_listed_as_exploratory():
    for product, spec in wind.SPECS.items():
        exploratory = wind.exploratory_contrasts(spec=spec, extra_heights_m=spec.extra_heights_m)

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
        raw=raw, spec=spec, with_direction=True, extra_heights_m=()
    )
    without = wind.product_wind_columns(
        raw=raw.drop("wind_direction_100m"),
        spec=spec,
        with_direction=False,
        extra_heights_m=(),
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


def _direction_files(*, directory: Path, heights: tuple[int, ...]) -> None:
    for height in heights:
        (directory / wind.CERRA_DIRECTION_FILES[height]).touch()


def test_with_direction_the_planned_files_alone_keep_the_planned_arms_direction(tmp_path: Path):
    spec = wind.SPECS["cerra"]
    _direction_files(directory=tmp_path, heights=(100,))

    extra, omitted = wind.plan_extra_heights(spec=spec, directory=tmp_path, with_direction=True)
    features = wind.arm_features(spec=spec, with_direction=True, extra_heights_m=extra)

    assert (extra, omitted) == ((), (75, 150))
    assert "direction_sin_cerra" in features["cerra_wind"]
    assert "direction_sin_era5" in features["era5_wind"]
    assert set(features) == {f"{p}_wind" for p in (*wind.MAIN_PRODUCTS, "cerra")}


def test_with_direction_each_exploratory_height_is_kept_only_if_its_file_exists(tmp_path: Path):
    _direction_files(directory=tmp_path, heights=(100, 150))

    extra, omitted = wind.plan_extra_heights(
        spec=wind.SPECS["cerra"], directory=tmp_path, with_direction=True
    )

    assert (extra, omitted) == ((150,), (75,))


def test_with_direction_a_missing_planned_file_raises_a_clear_error(tmp_path: Path):
    _direction_files(directory=tmp_path, heights=(75, 150))

    with pytest.raises(FileNotFoundError, match="CERRA_WITH_DIRECTION is True"):
        wind.plan_extra_heights(spec=wind.SPECS["cerra"], directory=tmp_path, with_direction=True)


def test_without_direction_no_file_is_needed_and_every_exploratory_height_stays(tmp_path: Path):
    spec = wind.SPECS["cerra"]

    extra, omitted = wind.plan_extra_heights(spec=spec, directory=tmp_path, with_direction=False)

    assert (extra, omitted) == (spec.extra_heights_m, ())


def test_nora3_never_looks_for_cerra_direction_files(tmp_path: Path):
    spec = wind.SPECS["nora3"]

    assert wind.plan_extra_heights(spec=spec, directory=tmp_path, with_direction=True) == (
        spec.extra_heights_m,
        (),
    )


@pytest.mark.parametrize("constant", [True, False])
def test_the_constant_decides_cerras_direction_and_never_nora3s(
    monkeypatch: pytest.MonkeyPatch, *, constant: bool
):
    monkeypatch.setattr(wind, "CERRA_WITH_DIRECTION", constant)

    assert wind.with_direction_for(spec=wind.SPECS["cerra"]) is constant
    assert wind.with_direction_for(spec=wind.SPECS["nora3"]) is True


def test_nora3_without_its_10m_file_raises_a_clear_error(tmp_path: Path):
    main = tmp_path / "NORA3_wind.parquet"
    main.touch()
    cells = pl.DataFrame({"site": ["W1"], "y_index": [1], "x_index": [1]})

    with pytest.raises(FileNotFoundError, match=r"10 m file .* has not been downloaded"):
        wind.read_nora3(
            cells=cells, heights=[100], path=main, path_10m=tmp_path / "missing.parquet"
        )


def test_nora3_keeps_an_hour_the_10m_file_lacks_as_a_null_so_the_missing_value_check_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    main_file, file_10m = tmp_path / "main.parquet", tmp_path / "ten.parquet"
    main_file.touch()
    file_10m.touch()
    hours = [DAY, DAY + timedelta(hours=1)]
    main = pl.DataFrame({"site": ["W1"] * 2, "time": hours, "wind_speed_100m": [8.0, 9.0]})
    surface = pl.DataFrame({"site": ["W1"], "time": hours[:1], "wind_speed_10m": [3.0]})
    monkeypatch.setattr(
        wind, "read_nora3_wind", lambda *, path, **_: main if path == main_file else surface
    )

    joined = wind.read_nora3(cells=pl.DataFrame(), heights=[100], path=main_file, path_10m=file_10m)

    assert joined.height == 2
    assert joined["wind_speed_10m"].to_list() == [3.0, None]


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


def _main_rows() -> pl.DataFrame:
    """One row per site per day at 00 UTC from September 2024 to June 2026, with feature columns."""
    start = datetime(2024, 9, 1, tzinfo=UTC)
    times = pl.datetime_range(
        start, datetime(2026, 6, 30, tzinfo=UTC), interval="1d", time_zone="UTC", eager=True
    )
    return (
        pl.DataFrame({"time": times})
        .join(pl.DataFrame({"site": ["W1", "W2"]}), how="cross")
        .with_columns(
            month=pl.col("time").dt.strftime("%Y-%m"),
            hour_of_day=pl.col("time").dt.hour(),
            day_of_year=pl.col("time").dt.ordinal_day(),
            power_mw=1.0,
        )
    )


def _product_values(*, base: pl.DataFrame, spec: Any) -> pl.DataFrame:
    """Every arm's speed columns, as `assemble_rows` receives them from `product_wind_columns`."""
    columns = []
    for key in (*wind.MAIN_PRODUCTS, spec.key):
        speed, _, _, surface = wind._wind_columns(product=key)
        columns += [pl.lit(6.0).alias(speed), pl.lit(3.0).alias(surface)]
    return base.select("site", "time").with_columns(*columns)


def _cerra_features() -> dict[str, tuple[str, ...]]:
    return wind.arm_features(spec=wind.SPECS["cerra"], with_direction=False, extra_heights_m=())


def test_assemble_rows_cuts_covering_folds_and_reports_the_share_seen_in_one_year_only():
    spec = wind.SPECS["cerra"]
    base = _main_rows()

    assembled = wind.assemble_rows(
        base=base, wind=_product_values(base=base, spec=spec), spec=spec, features=_cerra_features()
    )

    frame = assembled.frame
    one_year = frame.filter(pl.col("time").dt.month().is_in([7, 8])).height / frame.height
    assert {"fold", "era_code"} <= set(frame.columns)
    assert assembled.main_rows == base.height
    assert assembled.uncovered_fitted == 0.0
    assert 0.0 < one_year < 1.0
    assert assembled.one_year_only_fitted == pytest.approx(one_year)
    assert 0.0 <= assembled.uncovered_main_folds <= 1.0


def test_assemble_rows_reports_a_positive_avoidable_share_when_a_month_lacks_training_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    spec = wind.SPECS["cerra"]
    base = _main_rows()
    monkeypatch.setattr(
        wind, "with_covering_folds", lambda *, frame: (wind.with_eras(frame=frame), {0: 0})
    )

    assembled = wind.assemble_rows(
        base=base, wind=_product_values(base=base, spec=spec), spec=spec, features=_cerra_features()
    )

    assert assembled.uncovered_fitted == pytest.approx(assembled.uncovered_main_folds)
    assert assembled.uncovered_fitted > 0.0


@pytest.mark.parametrize("column", ["speed_10m_cerra", "speed_hub_cerra", "power_mw"])
def test_assemble_rows_raises_on_a_missing_value_in_any_arm_column(column: str):
    spec = wind.SPECS["cerra"]
    base = _main_rows()
    values = _product_values(base=base, spec=spec)
    if column == "power_mw":
        base = base.with_columns(
            power_mw=pl.when(pl.col("time").dt.day() == 3).then(None).otherwise(1.0)
        )
    else:
        values = values.with_columns(
            pl.when(pl.col("time").dt.day() == 3).then(None).otherwise(pl.col(column)).alias(column)
        )

    with pytest.raises(ValueError, match=column):
        wind.assemble_rows(base=base, wind=values, spec=spec, features=_cerra_features())


def _losses(*, arms: tuple[str, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "arm": arm,
                "setting": setting,
                "seed": seed,
                "site": site,
                "time": datetime(2025, month, 1, tzinfo=UTC),
                "month": month,
                "fold": month % 5,
                "absolute_error_capped_fraction_of_capacity": 0.05
                + 0.01 * arm_index
                + 0.001 * month
                + (0.02 if setting == "sensitivity" else 0.0),
            }
            for setting in ("pooled", "sensitivity")
            for arm_index, arm in enumerate(arms)
            for seed in range(3)
            for site in ("W1", "W2")
            for month in range(1, 13)
        ]
    )


def _built(*, extra_heights_m: tuple[int, ...], omitted: tuple[str, ...]) -> Any:
    spec = wind.SPECS["cerra"]
    features = wind.arm_features(spec=spec, with_direction=True, extra_heights_m=extra_heights_m)
    frame = _hours(days=2).with_columns(hour_of_day=pl.col("time").dt.hour(), fold=0, power_mw=1.0)
    assembled = wind.AssembledRows(
        frame=frame,
        main_rows=frame.height,
        fold_offsets={0: 0},
        uncovered_main_folds=0.179,
        uncovered_fitted=0.0,
        one_year_only_fitted=0.123,
    )
    return wind.Built(
        assembled=assembled,
        features=features,
        with_direction=True,
        distance_range_km=(1.0, 2.0),
        extra_heights_m=extra_heights_m,
        omitted_arms=omitted,
    )


def test_the_report_labels_planned_and_exploratory_contrasts_at_both_settings():
    spec = wind.SPECS["cerra"]
    built = _built(extra_heights_m=(150,), omitted=("cerra_75m_wind",))
    losses = _losses(arms=tuple(built.features))
    sites = pl.DataFrame({"latitude": [52.0, 52.2], "longitude": [-1.0, -1.1]})

    report = wind._report(
        spec=spec,
        built=built,
        losses=losses,
        sites=sites,
        job_list=wind.jobs(features=built.features),
    )

    assert report.count("| all | cerra_wind − era5_wind |") == 1
    assert report.count("| sensitivity | cerra_wind − era5_wind |") == 1
    assert "| all | cerra_wind − icon_d2_wind |" in report
    assert "| all (exploratory) | cerra_150m_wind − cerra_wind |" in report
    assert "| sensitivity (exploratory) | cerra_150m_wind − cerra_wind |" in report
    assert "| all (exploratory) | cerra_wind − ukv_wind |" in report
    assert "cerra_75m_wind − cerra_wind" not in report
    assert "omitted because their direction file is missing: `cerra_75m_wind`" in report
    assert "12.3%" in report
    assert "17.9%" in report


def test_the_report_prints_absolute_skill_at_both_settings():
    spec = wind.SPECS["cerra"]
    built = _built(extra_heights_m=(), omitted=())
    arms = tuple(built.features)
    losses = _losses(arms=arms)
    sites = pl.DataFrame({"latitude": [52.0, 52.2], "longitude": [-1.0, -1.1]})

    report = wind._report(
        spec=spec,
        built=built,
        losses=losses,
        sites=sites,
        job_list=wind.jobs(features=built.features),
    )

    head, _, second = report.partition("#### Absolute error at the second hyperparameter setting")
    assert second
    # era5 is arm 0: mean error 0.05 + 0.001 * 6.5 = 5.650 pooled and 7.650 at the second setting.
    assert "| era5_wind | 5.650 |" in head
    assert "| era5_wind | 7.650 |" in second.split("#### Planned contrasts")[0]
