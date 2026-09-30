import re
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
    Comparison,
    SourceType,
    _check_same_keys,
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
D4S: SourceType = "day4_shared"

EXPECTED_LEADERBOARD_ENS_SOURCE = {
    0: D10,
    1: D10,
    2: D10B,
    3: D10,
    4: D4S,
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
    4: {
        "ICON-EU",
        "ICON global",
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
    5: {
        "AIFS Single",
        "AIFS ENS mean",
        "WeatherNext 3 mean (7 months)",
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
    for label in (
        "ENS control member",
        "GEFS mean",
        "GFS (native)",
        "GFS (Open-Meteo)",
        "IFS HRES (9 km, Open-Meteo)",
        "ICON-EU",
        "ICON global",
        "IFS 0.25°",
    ):
        assert by_source[(label, 4)] == D4S


def test_aifs_and_wn3_read_their_own_ens_mean_from_their_own_file() -> None:
    folders = {
        "AIFS Single": "aifs_single",
        "AIFS ENS mean": "aifs_ens",
        "WeatherNext 3 mean (7 months)": "wn3",
    }
    for domain in ("solar", "wind"):
        for c in comparisons(domain=domain):
            if c.label in folders:
                kind = "extra" if c.day in (0, 3, 4, 10) else "day5" if c.day == 5 else "blends"
                assert c.treatment_source == f"{folders[c.label]}_{kind}"
                assert c.reference_source == c.treatment_source


def test_wind_weathernext_3_uses_the_mean_vector_reference_and_solar_the_plain_mean() -> None:
    wind = [c for c in comparisons(domain="wind") if c.treatment.startswith("wn3")]
    solar = [c for c in comparisons(domain="solar") if c.treatment.startswith("wn3")]
    days = (0, 1, 2, 3, 4, 5, 7, 10, 14)

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


IFS = "IFS HRES (9 km, Open-Meteo)"


def test_a_row_with_fewer_than_six_months_has_a_dot_and_no_interval(tmp_path: Path) -> None:
    # IFS HRES 9 km is the one product allowed to lack keys the ENS mean holds.
    _full_solar_fixture(tmp_path, short_months={"ifs_single_day0": 5})

    rows = compute(data_dir=tmp_path, domain="solar")
    short = rows.filter((pl.col("label") == IFS) & (pl.col("day") == 0))
    shaped = chart_rows(rows=rows, day=0, with_conditions=True).filter(pl.col("label") == IFS)

    assert short["n_months"].to_list() == [5]
    assert short["has_interval"].to_list() == [False]
    assert shaped["lower_95"].null_count() == 1
    assert shaped["upper_95"].null_count() == 1
    assert shaped["difference"].null_count() == 0
    assert shaped["condition"].item().startswith("Fewer than 6 months")


def test_six_months_is_enough_for_an_interval_and_the_hollow_mark_is_in_the_chart(
    tmp_path: Path,
) -> None:
    _full_solar_fixture(tmp_path, short_months={"ifs_single_day0": 6, "ifs_single_day1": 5})

    rows = compute(data_dir=tmp_path, domain="solar")

    assert rows.filter((pl.col("label") == IFS) & (pl.col("day") == 0))[
        "has_interval"
    ].to_list() == [True]
    assert rows.filter((pl.col("label") == IFS) & (pl.col("day") == 1))[
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


def _arm_cells(path: Path) -> set[tuple[str, int]]:
    """Return the (product prefix, lead day) of every product arm a losses file holds."""
    not_products = {"ens_mean", "persistence", "diurnal_persistence", "smart_persistence"}
    cells = set()
    for arm in pl.read_parquet(path, columns=["arm"])["arm"].unique().to_list():
        match = re.fullmatch(r"(.+)_day(\d+)", arm)
        if match and match[1] not in not_products and not match[1].startswith("blend_"):
            cells.add((match[1], int(match[2])))
    return cells


@pytest.mark.skipif(
    not (_PAGE_DATA / SOURCES["leads_day10"].folder).exists(),
    reason="the private study data is not in this checkout",
)
@pytest.mark.parametrize("domain", ["solar", "wind"])
def test_the_plan_holds_every_product_arm_the_leads_folders_hold_and_no_other(
    domain: DomainType,
) -> None:
    plan = comparisons(domain=domain)
    prefix_of = {c.label: c.treatment.rsplit("_day", 1)[0] for c in plan}
    for source in ("leads_day10", "leads_day10b", "leads_day10c", "leads_day10d", "day4_shared"):
        folder = _PAGE_DATA / SOURCES[source].folder
        held = _arm_cells(folder / f"{domain}_losses.parquet")
        planned = {(prefix_of[c.label], c.day) for c in plan if c.treatment_source == source}
        assert planned == held, f"{source}: plan and saved arms differ"
    published = _arm_cells(_PAGE_DATA / "nwp_forecast_comparison" / f"{domain}_losses.parquet")
    assert published <= {(prefix_of[c.label], c.day) for c in plan}


def _keyed(arm: str, *, months: int = 12, skip_month: int | None = None) -> pl.DataFrame:
    frame = _arm(arm, error=0.1, months=months)
    return frame if skip_month is None else frame.filter(pl.col("month") != skip_month)


def _comparison(prefix: str, *, day: int = 0, domain: DomainType = "solar") -> Comparison:
    return Comparison(
        domain=domain,
        day=day,
        label=prefix,
        treatment=f"{prefix}_day{day}",
        treatment_source="leads_day10",
        reference=f"ens_mean_day{day}",
        reference_source="leads_day10",
        reference_label="ENS mean",
    )


def test_a_product_with_keys_the_reference_lacks_raises() -> None:
    comparison = _comparison("ukv")

    with pytest.raises(ValueError, match="only in ukv_day0"):
        _check_same_keys(
            comparison=comparison,
            treatment=_keyed("ukv_day0"),
            reference=_keyed("ens_mean_day0", skip_month=3),
        )


def test_a_product_lacking_keys_the_reference_holds_raises_unless_it_is_ifs_single() -> None:
    with pytest.raises(ValueError, match="only in ens_mean_day0"):
        _check_same_keys(
            comparison=_comparison("ukv"),
            treatment=_keyed("ukv_day0", skip_month=3),
            reference=_keyed("ens_mean_day0"),
        )
    _check_same_keys(
        comparison=_comparison("ifs_single"),
        treatment=_keyed("ifs_single_day0", skip_month=3),
        reference=_keyed("ens_mean_day0"),
    )


def _lacks_a_month(prefix: str, *, day: int, domain: DomainType) -> None:
    _check_same_keys(
        comparison=_comparison(prefix, day=day, domain=domain),
        treatment=_keyed(f"{prefix}_day{day}", skip_month=3),
        reference=_keyed(f"ens_mean_day{day}"),
    )


def test_icon_global_may_lack_reference_keys_at_day_4_for_solar_only() -> None:
    _lacks_a_month("icon_global", day=4, domain="solar")
    with pytest.raises(ValueError, match="only in ens_mean_day4"):
        _lacks_a_month("icon_global", day=4, domain="wind")
    with pytest.raises(ValueError, match="only in ens_mean_day5"):
        _lacks_a_month("icon_global", day=5, domain="solar")


def test_ifs_single_may_lack_reference_keys_at_any_day_in_both_technologies() -> None:
    for domain in ("solar", "wind"):
        _lacks_a_month("ifs_single", day=3, domain=domain)
        _lacks_a_month("ifs_single", day=7, domain=domain)


def test_ifs_single_keys_the_treatment_lacks_still_raise_the_other_way() -> None:
    with pytest.raises(ValueError, match="only in ifs_single_day0"):
        _check_same_keys(
            comparison=_comparison("ifs_single"),
            treatment=_keyed("ifs_single_day0"),
            reference=_keyed("ens_mean_day0", skip_month=3),
        )


def test_each_row_records_how_many_rows_each_arm_holds(tmp_path: Path) -> None:
    _full_solar_fixture(tmp_path)

    rows = compute(data_dir=tmp_path, domain="solar")

    row = rows.filter((pl.col("label") == "UKV") & (pl.col("day") == 0)).row(0, named=True)
    assert row["treatment_rows"] == row["reference_rows"] == 2 * 12 * 4 * 3


@pytest.mark.skipif(
    not (_PAGE_DATA / SOURCES["day4_shared"].folder).exists(),
    reason="the private study data is not in this checkout",
)
@pytest.mark.parametrize(
    ("domain", "label", "day", "value", "lower", "upper"),
    [
        ("solar", "IFS 0.25°", 4, 0.23, -0.02, 0.47),
        ("wind", "IFS 0.25°", 4, 0.42, 0.00, 0.88),
        ("solar", "WeatherNext 3 mean (7 months)", 5, -0.66, -1.19, -0.14),
        ("wind", "WeatherNext 3 mean (7 months)", 5, -2.13, -3.78, -0.86),
    ],
)
def test_the_day_4_and_day_5_dots_reproduce_the_folder_reports(
    domain: DomainType, label: str, day: int, value: float, lower: float, upper: float
) -> None:
    rows = _page_rows(domain)

    row = rows.filter((pl.col("label") == label) & (pl.col("day") == day)).row(0, named=True)
    assert (row["value"], row["lower"], row["upper"]) == pytest.approx(
        (value, lower, upper), abs=0.0061
    )


@pytest.mark.skipif(
    not (_PAGE_DATA / SOURCES["wn3_day5"].folder).exists(),
    reason="the private study data is not in this checkout",
)
def test_wind_weathernext_3_at_day_5_against_the_plain_ens_mean_reproduces_the_report() -> None:
    comparison = Comparison(
        domain="wind",
        day=5,
        label="WeatherNext 3 mean (7 months)",
        treatment="wn3_mean_day5",
        treatment_source="wn3_day5",
        reference="ens_mean_day5",
        reference_source="wn3_day5",
        reference_label="ENS mean",
    )
    arms = load_arms(
        data_dir=_PAGE_DATA,
        domain="wind",
        wanted={("wn3_day5", "wn3_mean_day5"), ("wn3_day5", "ens_mean_day5")},
    )

    row = contrast_rows(arms=arms, plan=[comparison]).row(0, named=True)

    assert (row["value"], row["lower"], row["upper"]) == pytest.approx(
        (-1.39, -2.71, -0.37), abs=0.0061
    )


@pytest.mark.skipif(
    not (_PAGE_DATA / SOURCES["day4_shared"].folder).exists(),
    reason="the private study data is not in this checkout",
)
def test_the_arms_that_drop_their_own_null_days_hold_fewer_rows_than_their_reference() -> None:
    solar = _page_rows("solar")

    def held(label: str, day: int) -> tuple[int, int]:
        row = solar.filter((pl.col("label") == label) & (pl.col("day") == day)).row(0, named=True)
        return row["treatment_rows"], row["reference_rows"]

    assert held("ICON global", 4)[0] < held("ICON global", 4)[1]
    assert held("IFS HRES (9 km, Open-Meteo)", 4)[0] < held("IFS HRES (9 km, Open-Meteo)", 4)[1]
    assert held("ICON-EU", 4)[0] == held("ICON-EU", 4)[1]
