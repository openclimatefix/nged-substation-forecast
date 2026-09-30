import sys
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path

import polars as pl
import pytest

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

from dot_interval_vs_ens import (  # noqa: E402
    SOURCES,
    SourceType,
    chart_rows,
    comparisons,
    compute,
    contrast_rows,
    draw,
    finding_title,
    load_arms,
    repo_data_dir,
    write_once,
)
from nwp_forecast_comparison import DomainType  # noqa: E402

METRIC = "absolute_error_capped_fraction_of_capacity"
SEEDS = (0, 1, 2)
SITES = ("A", "B")


def _arm(
    arm: str,
    *,
    error: float,
    months: int = 12,
    setting: str = "primary",
    sites: Sequence[str] = SITES,
) -> pl.DataFrame:
    """One arm's rows: `months` months of 4 hourly rows per site and seed, with a constant error."""
    rows = [
        {"arm": arm, "site": site, "time": month * 100 + hour, "seed": seed, "month": month}
        for site in sites
        for month in range(months)
        for hour in range(4)
        for seed in SEEDS
    ]
    return pl.DataFrame(rows).with_columns(setting=pl.lit(setting), **{METRIC: pl.lit(error)})


def _write(
    *frames: pl.DataFrame, data_dir: Path, source: SourceType, domain: str, day: int
) -> None:
    spec = SOURCES[source]
    path = data_dir / spec.folder / spec.pattern.format(domain=domain, day=day)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.concat(frames)
    if path.exists():
        frame = pl.concat([pl.read_parquet(path), frame])
    frame.write_parquet(path)


def test_a_product_is_subtracted_from_the_ens_mean_of_its_own_folder(tmp_path: Path) -> None:
    # A decoy ENS-mean copy sits in another folder: a wrong reference changes the answer.
    _write(
        _arm("ukv_day0", error=0.10), data_dir=tmp_path, source="leads_day10", domain="solar", day=0
    )
    _write(
        _arm("ens_mean_day0", error=0.08),
        data_dir=tmp_path,
        source="leads_day10",
        domain="solar",
        day=0,
    )
    _write(
        _arm("ens_mean_day0", error=0.50),
        data_dir=tmp_path,
        source="leads_day10b",
        domain="solar",
        day=0,
    )

    plan = [c for c in comparisons(domain="solar") if c.treatment == "ukv_day0"]
    wanted = {(c.treatment_source, c.treatment) for c in plan} | {
        (c.reference_source, c.reference) for c in plan
    }

    rows = contrast_rows(
        arms=load_arms(data_dir=tmp_path, domain="solar", wanted=wanted), plan=plan
    )

    assert rows["value"].to_list() == pytest.approx([2.0])
    assert rows["reference_source"].to_list() == ["leads_day10"]


def _full_solar_fixture(data_dir: Path, *, short_months: Mapping[str, int] | None = None) -> None:
    """Write every arm the solar plan reads; each product gets a distinct constant error.

    Args:
        data_dir: Where the fake `data/studies` goes.
        short_months: Arms to write with fewer months than the default.
    """
    short_months = short_months or {}
    written: set[tuple[str, str]] = set()
    for index, comparison in enumerate(comparisons(domain="solar")):
        for arm, source, error in (
            (comparison.treatment, comparison.treatment_source, 0.10 + 0.001 * index),
            (comparison.reference, comparison.reference_source, 0.05),
        ):
            if (source, arm) in written:
                continue
            written.add((source, arm))
            months = short_months.get(arm, 7 if source.startswith("wn3") else 12)
            _write(
                _arm(arm, error=error, months=months),
                data_dir=data_dir,
                source=source,
                domain="solar",
                day=comparison.day,
            )


def test_a_folder_without_an_ens_mean_reads_the_ens_mean_of_the_leads_day10_folder(
    tmp_path: Path,
) -> None:
    _full_solar_fixture(tmp_path)
    # Decoy: the folder of ICON-EU day 3 holds an ENS mean at day 3 that must not be used.
    _write(
        _arm("ens_mean_day3", error=0.99),
        data_dir=tmp_path,
        source="leads_day10b",
        domain="solar",
        day=3,
    )

    rows = compute(data_dir=tmp_path, domain="solar")

    icon_eu_day3 = rows.filter((pl.col("label") == "ICON-EU") & (pl.col("day") == 3))
    assert icon_eu_day3["reference_source"].to_list() == ["leads_day10"]
    assert icon_eu_day3["value"].item() < 10.0


D10: SourceType = "leads_day10"
D10B: SourceType = "leads_day10b"

EXPECTED_LEADERBOARD_ENS_SOURCE = {
    0: D10,
    1: D10,
    2: D10B,
    3: D10,
    5: D10,
    7: D10B,
    10: D10,
    14: D10,
}
"""The folder of the ENS-mean arm the leaderboards draw at each lead day, read off the saved
folders: `ens_mean_day2` and `ens_mean_day7` are GPU refits in `_leads_day10b`, and every other
day's GPU refit is in `_leads_day10`."""

EXPECTED_SOLAR_LABELS_BY_DAY = {
    0: {
        "UKV",
        "ICON-D2",
        "ICON-EU",
        "ICON global",
        "GFS (Open-Meteo)",
        "GFS (native)",
        "IFS 0.25°",
        "IFS HRES (9 km, Open-Meteo)",
        "ARPEGE Europe",
        "AROME France",
        "DMI HARMONIE-AROME",
        "KNMI HARMONIE-AROME",
        "GEFS mean",
        "ENS control member",
        "AIFS Single",
        "AIFS ENS mean",
        "WeatherNext 3 mean (7 months)",
    },
    1: {
        "UKV",
        "ICON-D2",
        "ICON-EU",
        "ICON global",
        "GFS (Open-Meteo)",
        "GFS (native)",
        "IFS 0.25°",
        "IFS HRES (9 km, Open-Meteo)",
        "ARPEGE Europe",
        "AROME France",
        "DMI HARMONIE-AROME",
        "KNMI HARMONIE-AROME",
        "GEFS mean",
        "ENS control member",
        "AIFS Single",
        "AIFS ENS mean",
        "WeatherNext 3 mean (7 months)",
    },
    2: {
        "ICON-EU",
        "ICON global",
        "GFS (Open-Meteo)",
        "GFS (native)",
        "IFS 0.25°",
        "IFS HRES (9 km, Open-Meteo)",
        "ARPEGE Europe",
        "GEFS mean",
        "ENS control member",
        "AIFS Single",
        "AIFS ENS mean",
        "WeatherNext 3 mean (7 months)",
    },
    3: {
        "ICON-EU",
        "ICON global",
        "GFS (Open-Meteo)",
        "GFS (native)",
        "IFS 0.25°",
        "IFS HRES (9 km, Open-Meteo)",
        "ARPEGE Europe",
        "GEFS mean",
        "ENS control member",
        "AIFS Single",
        "AIFS ENS mean",
        "WeatherNext 3 mean (7 months)",
    },
    4: {"AIFS Single", "AIFS ENS mean", "WeatherNext 3 mean (7 months)"},
    5: {
        "ICON global",
        "GFS (Open-Meteo)",
        "GFS (native)",
        "IFS 0.25°",
        "IFS HRES (9 km, Open-Meteo)",
        "GEFS mean",
        "ENS control member",
    },
    7: {
        "GFS (Open-Meteo)",
        "GFS (native)",
        "IFS 0.25°",
        "IFS HRES (9 km, Open-Meteo)",
        "GEFS mean",
        "ENS control member",
        "AIFS Single",
        "AIFS ENS mean",
        "WeatherNext 3 mean (7 months)",
    },
    10: {
        "GFS (native)",
        "GEFS mean",
        "ENS control member",
        "AIFS Single",
        "AIFS ENS mean",
        "WeatherNext 3 mean (7 months)",
    },
    14: {
        "GFS (native)",
        "GEFS mean",
        "ENS control member",
        "AIFS Single",
        "AIFS ENS mean",
        "WeatherNext 3 mean (7 months)",
    },
}
"""The products each lead day's panel holds for solar, read off the leaderboards' saved arms."""


def test_each_panel_holds_exactly_the_products_the_leaderboards_carry_at_that_day() -> None:
    plan = comparisons(domain="solar")

    for day, labels in EXPECTED_SOLAR_LABELS_BY_DAY.items():
        assert {c.label for c in plan if c.day == day} == labels
    assert {c.day for c in plan} == set(EXPECTED_SOLAR_LABELS_BY_DAY)
    assert len(plan) == sum(len(labels) for labels in EXPECTED_SOLAR_LABELS_BY_DAY.values())


def test_wind_has_the_solar_rows_without_arpege_and_arome() -> None:
    solar = {(c.label, c.day) for c in comparisons(domain="solar")}
    wind = {(c.label, c.day) for c in comparisons(domain="wind")}

    assert solar - wind == {
        (label, day)
        for label in ("ARPEGE Europe", "AROME France")
        for day in (0, 1, 2, 3)
        if (label, day) in solar
    }
    assert wind <= solar


def test_every_leaderboard_product_is_subtracted_from_the_leaderboards_ens_mean_arm() -> None:
    leaderboard_sources = {D10, D10B, "leads_day10c", "leads_day10d"}
    for domain in ("solar", "wind"):
        for c in comparisons(domain=domain):
            if c.treatment_source in leaderboard_sources:
                assert c.reference == f"ens_mean_day{c.day}"
                assert c.reference_source == EXPECTED_LEADERBOARD_ENS_SOURCE[c.day]


def test_every_product_fitted_in_the_leads_folders_has_a_source_for_each_of_its_days() -> None:
    by_source = {(c.label, c.day): c.treatment_source for c in comparisons(domain="solar")}

    assert by_source[("ICON-EU", 2)] == D10B
    assert by_source[("ICON-EU", 3)] == D10B
    assert by_source[("ICON-EU", 1)] == D10
    assert by_source[("ENS control member", 0)] == D10
    assert by_source[("ENS control member", 1)] == D10
    assert by_source[("ENS control member", 5)] == D10B
    assert by_source[("GEFS mean", 7)] == D10B
    assert by_source[("GEFS mean", 5)] == D10
    assert by_source[("GFS (native)", 14)] == "leads_day10c"
    assert by_source[("IFS HRES (9 km, Open-Meteo)", 7)] == "leads_day10d"


def test_aifs_and_wn3_read_their_own_ens_mean_from_their_own_file() -> None:
    folders = {
        "AIFS Single": "aifs_single",
        "AIFS ENS mean": "aifs_ens",
        "WeatherNext 3 mean (7 months)": "wn3",
    }
    for domain in ("solar", "wind"):
        for c in comparisons(domain=domain):
            if c.label in folders:
                kind = "extra" if c.day in (0, 3, 4, 10) else "blends"
                assert c.treatment_source == f"{folders[c.label]}_{kind}"
                assert c.reference_source == c.treatment_source


def test_wind_weathernext_3_uses_the_mean_vector_reference_and_solar_the_plain_mean() -> None:
    wind = [c for c in comparisons(domain="wind") if c.treatment.startswith("wn3")]
    solar = [c for c in comparisons(domain="solar") if c.treatment.startswith("wn3")]
    days = (0, 1, 2, 3, 4, 7, 10, 14)

    assert {c.reference for c in wind} == {f"ens_meanvec_day{d}" for d in days}
    assert {c.reference for c in solar} == {f"ens_mean_day{d}" for d in days}
    assert {c.label for c in wind + solar} == {"WeatherNext 3 mean (7 months)"}
    assert all(
        c.reference.startswith("ens_mean_day")
        for c in comparisons(domain="wind")
        if not c.treatment.startswith("wn3")
    )


def test_the_sensitivity_setting_is_left_out(tmp_path: Path) -> None:
    _write(
        _arm("ukv_day0", error=0.10),
        _arm("ukv_day0", error=0.90, setting="sensitivity"),
        data_dir=tmp_path,
        source="leads_day10",
        domain="solar",
        day=0,
    )

    arms = load_arms(data_dir=tmp_path, domain="solar", wanted={("leads_day10", "ukv_day0")})

    assert arms[("leads_day10", "ukv_day0")][METRIC].unique().to_list() == [0.10]


def test_a_repeated_key_raises(tmp_path: Path) -> None:
    arm = _arm("ukv_day0", error=0.10)
    _write(arm, arm, data_dir=tmp_path, source="leads_day10", domain="solar", day=0)

    with pytest.raises(ValueError, match="more than once"):
        load_arms(data_dir=tmp_path, domain="solar", wanted={("leads_day10", "ukv_day0")})


def test_a_missing_arm_raises(tmp_path: Path) -> None:
    _write(
        _arm("ens_mean_day0", error=0.08),
        data_dir=tmp_path,
        source="leads_day10",
        domain="solar",
        day=0,
    )

    with pytest.raises(ValueError, match="ukv_day0"):
        load_arms(data_dir=tmp_path, domain="solar", wanted={("leads_day10", "ukv_day0")})


def test_a_site_label_that_is_not_anonymised_raises(tmp_path: Path) -> None:
    _write(
        _arm("ukv_day0", error=0.1, sites=("Real Name",)),
        data_dir=tmp_path,
        source="leads_day10",
        domain="solar",
        day=0,
    )

    with pytest.raises(ValueError, match="not anonymised"):
        load_arms(data_dir=tmp_path, domain="solar", wanted={("leads_day10", "ukv_day0")})


def test_a_row_with_fewer_than_six_months_has_a_dot_and_no_interval(tmp_path: Path) -> None:
    _full_solar_fixture(tmp_path, short_months={"ukv_day0": 5})

    rows = compute(data_dir=tmp_path, domain="solar")
    short = rows.filter((pl.col("label") == "UKV") & (pl.col("day") == 0))
    shaped = chart_rows(rows=rows, day=0, with_conditions=True).filter(pl.col("label") == "UKV")

    assert short["n_months"].to_list() == [5]
    assert short["has_interval"].to_list() == [False]
    assert shaped["lower_95"].null_count() == 1
    assert shaped["upper_95"].null_count() == 1
    assert shaped["difference"].null_count() == 0
    assert shaped["condition"].item().startswith("Fewer than 6 months")


def test_six_months_is_enough_for_an_interval_and_the_hollow_mark_is_in_the_chart(
    tmp_path: Path,
) -> None:
    _full_solar_fixture(tmp_path, short_months={"ukv_day0": 6, "icon_d2_day0": 5})

    rows = compute(data_dir=tmp_path, domain="solar")

    assert rows.filter((pl.col("label") == "UKV") & (pl.col("day") == 0))[
        "has_interval"
    ].to_list() == [True]
    assert rows.filter((pl.col("label") == "ICON-D2") & (pl.col("day") == 0))[
        "has_interval"
    ].to_list() == [False]
    spec = str(draw(rows=rows, domain="solar", number=19).to_dict())
    assert "'filled': False" in spec
    assert "Fewer than 6 months: no interval" in spec


def test_chart_rows_are_sorted_best_first_and_hold_one_day(tmp_path: Path) -> None:
    _full_solar_fixture(tmp_path)
    rows = compute(data_dir=tmp_path, domain="solar")

    shaped = chart_rows(rows=rows, day=0, with_conditions=False)

    assert shaped["difference"].to_list() == sorted(shaped["difference"].to_list())
    assert shaped.height == rows.filter(pl.col("day") == 0).height


def test_the_figure_draws_with_its_number_and_a_title_counted_from_the_rows(tmp_path: Path) -> None:
    _full_solar_fixture(tmp_path)
    rows = compute(data_dir=tmp_path, domain="solar")

    chart = draw(rows=rows, domain="solar", number=19)

    assert "Figure 19:" in str(chart.to_dict())


def _rows_for_title(*intervals: tuple[float, float, float, bool]) -> pl.DataFrame:
    return pl.DataFrame(intervals, schema=["value", "lower", "upper", "has_interval"], orient="row")


def test_the_title_counts_better_worse_spanning_and_short_rows() -> None:
    rows = _rows_for_title(
        (-1.0, -1.5, -0.5, True),
        (1.0, 0.5, 1.5, True),
        (0.1, -0.4, 0.6, True),
        (3.0, 2.0, 4.0, False),
    )

    assert finding_title(rows=rows, domain="solar") == (
        "For solar power, 1 of 4 product-and-lead rows have a higher error than the ENS mean, "
        "1 have a lower error, and 1 cannot be told apart; 1 have too few months for an interval"
    )


def test_the_figure_draws_no_accessibility_text_on_its_marks(tmp_path: Path) -> None:
    _full_solar_fixture(tmp_path)
    rows = compute(data_dir=tmp_path, domain="solar")

    spec = str(draw(rows=rows, domain="solar", number=19).to_dict())

    assert "'aria': False" in spec


def test_write_once_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    write_once(path=path, write="first")

    with pytest.raises(FileExistsError):
        write_once(path=path, write="second")
    assert path.read_text() == "first"


_PAGE_DATA = repo_data_dir() / "studies"


@cache
def _page_rows(domain: DomainType) -> pl.DataFrame:
    """Bootstrap every dot of one technology once, for all the data-gated checks."""
    return compute(data_dir=_PAGE_DATA, domain=domain)


@pytest.mark.skipif(
    not (_PAGE_DATA / SOURCES["leads_day10"].folder).exists(),
    reason="the private study data is not in this checkout",
)
@pytest.mark.parametrize(
    ("domain", "label", "day", "value", "lower", "upper"),
    [
        ("solar", "GEFS mean", 0, 1.233, 0.940, 1.521),
        ("wind", "UKV", 0, -0.446, -0.673, -0.218),
        ("solar", "IFS 0.25°", 0, -0.332, -0.524, -0.139),
        ("solar", "GFS (native)", 10, 0.410, 0.082, 0.727),
        ("solar", "GEFS mean", 5, 1.118, 0.704, 1.569),
        ("solar", "GEFS mean", 14, -0.002, -0.284, 0.256),
        ("wind", "GEFS mean", 7, 0.290, -0.496, 1.119),
        ("solar", "IFS 0.25°", 5, 0.088, -0.377, 0.575),
        ("wind", "IFS 0.25°", 5, 0.507, 0.081, 0.937),
        ("solar", "ENS control member", 2, 0.496, 0.303, 0.698),
        ("wind", "ENS control member", 5, 1.487, 1.028, 1.891),
        ("solar", "ENS control member", 14, -0.110, -0.329, 0.108),
        ("solar", "WeatherNext 3 mean (7 months)", 1, -0.243, -0.440, -0.069),
        ("solar", "WeatherNext 3 mean (7 months)", 14, -0.334, -0.996, 0.194),
        ("wind", "WeatherNext 3 mean (7 months)", 2, -0.801, -1.206, -0.321),
        ("wind", "WeatherNext 3 mean (7 months)", 7, -1.050, -2.870, 0.917),
        ("solar", "IFS HRES (9 km, Open-Meteo)", 3, 1.091, 0.775, 1.416),
        ("solar", "WeatherNext 3 mean (7 months)", 3, -1.478, -2.196, -0.831),
        ("wind", "WeatherNext 3 mean (7 months)", 10, 1.652, 0.014, 3.502),
    ],
)
def test_the_dots_reproduce_the_contrasts_the_page_states(
    domain: DomainType, label: str, day: int, value: float, lower: float, upper: float
) -> None:
    rows = _page_rows(domain)

    row = rows.filter((pl.col("label") == label) & (pl.col("day") == day)).row(0, named=True)
    assert (row["value"], row["lower"], row["upper"]) == pytest.approx(
        (value, lower, upper), abs=0.0006
    )
