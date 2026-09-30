import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl
import pytest

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

from dot_interval_vs_ens import (  # noqa: E402
    PRODUCTS,
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
            months = short_months.get(arm, 7 if source == "wn3" else 12)
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


def test_every_dot_is_subtracted_from_an_ens_mean_arm_of_the_same_day() -> None:
    for domain in ("solar", "wind"):
        for comparison in comparisons(domain=domain):
            if comparison.treatment.startswith("wn3") and domain == "wind":
                assert comparison.reference == f"ens_meanvec_day{comparison.day}"
            else:
                assert comparison.reference == f"ens_mean_day{comparison.day}"
            assert comparison.treatment.endswith(f"_day{comparison.day}")


def test_wind_weathernext_3_uses_the_mean_vector_reference_and_solar_the_plain_mean() -> None:
    wind = [c for c in comparisons(domain="wind") if c.treatment.startswith("wn3")]
    solar = [c for c in comparisons(domain="solar") if c.treatment.startswith("wn3")]

    assert {c.reference for c in wind} == {f"ens_meanvec_day{d}" for d in (0, 3, 4, 10)}
    assert {c.reference for c in solar} == {f"ens_mean_day{d}" for d in (0, 3, 4, 10)}
    assert {c.label for c in wind + solar} == {"WeatherNext 3 mean (7 months)"}


def test_the_plan_holds_the_products_the_archive_has_at_days_4_and_10() -> None:
    plan = comparisons(domain="solar")
    at_day_4 = {c.label for c in plan if c.day == 4}
    at_day_10 = {c.label for c in plan if c.day == 10}

    assert at_day_4 == {"AIFS Single", "AIFS ENS mean", "WeatherNext 3 mean (7 months)"}
    assert at_day_10 == {
        "GEFS mean",
        "ENS control member",
        "GFS (native)",
        "AIFS Single",
        "AIFS ENS mean",
        "WeatherNext 3 mean (7 months)",
    }


def test_arpege_and_arome_have_no_wind_row() -> None:
    labels = {c.label for c in comparisons(domain="wind")}

    assert "ARPEGE Europe" not in labels
    assert "AROME France" not in labels
    assert len(PRODUCTS) > 0


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
    short = rows.filter(pl.col("label") == "UKV")
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

    assert rows.filter(pl.col("label") == "UKV")["has_interval"].to_list() == [True]
    assert rows.filter(pl.col("label") == "ICON-D2")["has_interval"].to_list() == [False]
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
        ("solar", "IFS HRES (9 km, Open-Meteo)", 3, 1.091, 0.775, 1.416),
        ("solar", "WeatherNext 3 mean (7 months)", 3, -1.478, -2.196, -0.831),
        ("wind", "WeatherNext 3 mean (7 months)", 10, 1.652, 0.014, 3.502),
    ],
)
def test_the_dots_reproduce_the_contrasts_the_page_states(
    domain: DomainType, label: str, day: int, value: float, lower: float, upper: float
) -> None:
    rows = compute(data_dir=_PAGE_DATA, domain=domain)

    row = rows.filter((pl.col("label") == label) & (pl.col("day") == day)).row(0, named=True)
    assert (row["value"], row["lower"], row["upper"]) == pytest.approx(
        (value, lower, upper), abs=0.0006
    )
