from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from studies.midas import (
    null_night_spikes,
    read_hourly_weather,
    read_radiation,
    read_station_metadata,
    select_nearest_stations,
)

# Invented coordinates on the equator, in the Gulf of Guinea, so no fixture places a real station.
FIXTURE_LONGITUDE = 0.0
KM_PER_DEGREE_OF_LATITUDE = 111.195


def _stamps(*, first: datetime, count: int) -> list[datetime]:
    return [first + timedelta(hours=hour) for hour in range(count)]


def _write_radiation(*, path: Path, rows: pl.DataFrame) -> Path:
    frame = rows.with_columns(
        pl.col("time").dt.replace_time_zone("UTC"),
        ob_hour_count=pl.lit(1, dtype=pl.Int64),
        glbl_irad_amt_q=pl.lit(6000, dtype=pl.Int64),
    )
    frame.write_parquet(path)
    return path


def test_radiation_is_converted_from_kilojoules_to_watts_and_clipped_at_zero(tmp_path: Path):
    path = _write_radiation(
        path=tmp_path / "radiation.parquet",
        rows=pl.DataFrame(
            {
                "src_id": ["00002", "00001", "00001", "00001"],
                "time": [
                    datetime(2024, 6, 1, 12),
                    datetime(2024, 6, 1, 12),
                    datetime(2024, 6, 1, 11),
                    datetime(2024, 6, 1, 10),
                ],
                "glbl_irad_amt": [1800.0, 360.0, -41.0, 0.0],
            }
        ),
    )

    radiation = read_radiation(path=path)

    assert radiation["src_id"].to_list() == ["00001", "00001", "00001", "00002"]
    assert radiation["time"].to_list() == [
        datetime(2024, 6, 1, 10, tzinfo=UTC),
        datetime(2024, 6, 1, 11, tzinfo=UTC),
        datetime(2024, 6, 1, 12, tzinfo=UTC),
        datetime(2024, 6, 1, 12, tzinfo=UTC),
    ]
    assert radiation["ghi_w_m2"].to_list() == [0.0, 0.0, 100.0, 500.0]
    assert radiation["time"].dtype == pl.Datetime("us", "UTC")


def test_radiation_refuses_a_duplicate_hour(tmp_path: Path):
    path = _write_radiation(
        path=tmp_path / "radiation.parquet",
        rows=pl.DataFrame(
            {
                "src_id": ["00001", "00001"],
                "time": [datetime(2024, 6, 1, 12)] * 2,
                "glbl_irad_amt": [1.0, 2.0],
            }
        ),
    )

    with pytest.raises(ValueError, match="twice"):
        read_radiation(path=path)


def test_radiation_refuses_a_row_that_is_not_one_hour(tmp_path: Path):
    frame = pl.DataFrame(
        {
            "src_id": ["00001"],
            "time": [datetime(2024, 6, 1, 12, tzinfo=UTC)],
            "ob_hour_count": [3],
            "glbl_irad_amt": [1.0],
        }
    )
    frame.write_parquet(tmp_path / "radiation.parquet")

    with pytest.raises(ValueError, match="exactly one hour"):
        read_radiation(path=tmp_path / "radiation.parquet")


def _write_weather(*, path: Path, rows: pl.DataFrame) -> Path:
    rows.with_columns(pl.col("time").dt.replace_time_zone("UTC")).write_parquet(path)
    return path


def test_relative_humidity_is_clipped_and_temperature_is_untouched(tmp_path: Path):
    path = _write_weather(
        path=tmp_path / "weather.parquet",
        rows=pl.DataFrame(
            {
                "src_id": ["00001"] * 3,
                "time": _stamps(first=datetime(2024, 6, 1), count=3),
                "air_temperature": [12.5, -3.0, 40.0],
                "rltv_hum": [107.5, 100.0, 55.5],
                "wind_speed_m_s": [1.0, 2.0, 3.0],
            }
        ),
    )

    weather = read_hourly_weather(path=path, columns=["air_temperature", "rltv_hum"])

    assert weather.columns == ["src_id", "time", "air_temperature", "rltv_hum"]
    assert weather["rltv_hum"].to_list() == [100.0, 100.0, 55.5]
    assert weather["air_temperature"].to_list() == [12.5, -3.0, 40.0]


def test_a_quality_control_flag_column_is_refused(tmp_path: Path):
    path = _write_weather(
        path=tmp_path / "weather.parquet",
        rows=pl.DataFrame(
            {
                "src_id": ["00001"],
                "time": [datetime(2024, 6, 1)],
                "air_temperature": [1.0],
                "air_temperature_q": [1],
            }
        ),
    )

    with pytest.raises(ValueError, match="quality-control flags"):
        read_hourly_weather(path=path, columns=["air_temperature", "air_temperature_q"])


def test_weather_refuses_a_duplicate_instant(tmp_path: Path):
    path = _write_weather(
        path=tmp_path / "weather.parquet",
        rows=pl.DataFrame(
            {
                "src_id": ["00001"] * 2,
                "time": [datetime(2024, 6, 1)] * 2,
                "air_temperature": [1.0, 2.0],
            }
        ),
    )

    with pytest.raises(ValueError, match="twice"):
        read_hourly_weather(path=path, columns=["air_temperature"])


METADATA_CSV = """Conventions,G,BADC-CSV,1
title,G,invented header text
data
src_id,station_name,station_file_name,station_latitude,station_longitude,station_elevation,extra
00001,ALPHA,alpha,1.5,-2.5,10.0,x,surplus
00002,BETA,beta,3.0,4.0,20.5,y
end data
"""


def test_the_metadata_reader_keeps_ids_and_coordinates_and_drops_the_name(tmp_path: Path):
    path = tmp_path / "metadata.csv"
    path.write_text(METADATA_CSV)

    stations = read_station_metadata(path=path)

    assert stations.columns == ["src_id", "latitude", "longitude", "elevation_m"]
    assert stations.rows() == [("00001", 1.5, -2.5, 10.0), ("00002", 3.0, 4.0, 20.5)]


def test_the_metadata_reader_refuses_a_file_with_no_data_line(tmp_path: Path):
    path = tmp_path / "metadata.csv"
    path.write_text("title,G,nothing here\n")

    with pytest.raises(ValueError, match="no line reading 'data'"):
        read_station_metadata(path=path)


def test_the_metadata_reader_refuses_a_file_missing_a_column(tmp_path: Path):
    path = tmp_path / "metadata.csv"
    path.write_text("data\nsrc_id,station_latitude\n00001,1.0\nend data\n")

    with pytest.raises(ValueError, match="lacks the columns"):
        read_station_metadata(path=path)


def _night_fixture() -> tuple[pl.DataFrame, pl.DataFrame]:
    # At 40 degrees north, 30 degrees west, on 1 June the solar zenith angle at 06:00, 07:00,
    # 21:00 and 22:00 UTC is 96, 86, 87 and 97 degrees, so the hours ending at 07:00 and 22:00 hold
    # a sunrise and a sunset, and the hours ending at 03:00 and 04:00 are wholly at night.
    hours = (3, 4, 7, 12, 22)
    radiation = pl.DataFrame(
        {
            "src_id": ["00001"] * len(hours),
            "time": [datetime(2024, 6, 1, hour, tzinfo=UTC) for hour in hours],
            "ghi_w_m2": [400.0, 4.0, 90.0, 600.0, 30.0],
        }
    )
    stations = pl.DataFrame({"src_id": ["00001"], "latitude": [40.0], "longitude": [-30.0]})
    return radiation, stations


def test_a_night_spike_is_nulled_and_sunrise_sunset_daytime_and_small_night_values_survive():
    radiation, stations = _night_fixture()

    cleaned = null_night_spikes(radiation=radiation, stations=stations)

    assert cleaned["time"].to_list() == radiation["time"].to_list()
    assert cleaned["ghi_w_m2"].to_list() == [None, 4.0, 90.0, 600.0, 30.0]


def test_a_station_without_coordinates_is_refused():
    radiation, stations = _night_fixture()

    with pytest.raises(ValueError, match="no coordinates"):
        null_night_spikes(radiation=radiation, stations=stations.with_columns(src_id=pl.lit("9")))


def _sites(*rows: tuple[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": [site for site, _ in rows],
            "latitude": [latitude for _, latitude in rows],
            "longitude": [FIXTURE_LONGITUDE] * len(rows),
        }
    )


def _stations(*rows: tuple[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "src_id": [station for station, _ in rows],
            "latitude": [latitude for _, latitude in rows],
            "longitude": [FIXTURE_LONGITUDE] * len(rows),
        }
    )


HOURS = _stamps(first=datetime(2024, 6, 1, tzinfo=UTC), count=10)


def _observed(*, missing: dict[str, int] | None = None, stations: list[str]) -> pl.DataFrame:
    """Every station observed at every hour, less the first `missing[station]` hours."""
    missing = missing or {}
    return pl.DataFrame(
        [
            {"src_id": station, "time": hour}
            for station in stations
            for hour in HOURS[missing.get(station, 0) :]
        ],
        schema={"src_id": pl.String, "time": pl.Datetime("us", "UTC")},
    )


def _required(*sites: str) -> pl.DataFrame:
    return pl.DataFrame(
        [{"site": site, "time": hour} for site in sites for hour in HOURS],
        schema={"site": pl.String, "time": pl.Datetime("us", "UTC")},
    )


def test_the_nearest_complete_station_is_chosen_with_its_distance():
    stations = _stations(("far", 5.0), ("near", 1.0), ("mid", 2.0))

    chosen = select_nearest_stations(
        sites=_sites(("A", 0.0)),
        stations=stations,
        observed=_observed(stations=["far", "near", "mid"]),
        required=_required("A"),
        k=1,
        min_coverage=1.0,
    )

    assert chosen.select("site", "rank", "src_id", "coverage", "skipped_nearer").rows() == [
        ("A", 1, "near", 1.0, 0)
    ]
    assert chosen["distance_km"].to_list() == pytest.approx([KM_PER_DEGREE_OF_LATITUDE], rel=1e-3)


def test_a_nearer_station_with_a_gap_is_skipped_and_counted():
    stations = _stations(("near", 1.0), ("mid", 2.0), ("far", 5.0))

    chosen = select_nearest_stations(
        sites=_sites(("A", 0.0)),
        stations=stations,
        observed=_observed(stations=["near", "mid", "far"], missing={"near": 1}),
        required=_required("A"),
        k=1,
        min_coverage=1.0,
    )

    assert chosen.select("src_id", "skipped_nearer").rows() == [("mid", 1)]


def test_a_station_covering_exactly_the_minimum_is_eligible_and_one_hour_less_is_not():
    stations = _stations(("near", 1.0), ("far", 5.0))
    observed = _observed(stations=["near", "far"], missing={"near": 1})

    at_the_minimum = select_nearest_stations(
        sites=_sites(("A", 0.0)),
        stations=stations,
        observed=observed,
        required=_required("A"),
        k=1,
        min_coverage=0.9,
    )
    above_the_minimum = select_nearest_stations(
        sites=_sites(("A", 0.0)),
        stations=stations,
        observed=observed,
        required=_required("A"),
        k=1,
        min_coverage=0.91,
    )

    assert at_the_minimum["src_id"].to_list() == ["near"]
    assert at_the_minimum["coverage"].to_list() == pytest.approx([0.9])
    assert above_the_minimum["src_id"].to_list() == ["far"]


def test_a_tie_in_distance_goes_to_the_lower_station_id_however_the_rows_are_ordered():
    stations = _stations(("00009", 1.0), ("00003", -1.0), ("00005", 3.0))

    chosen = select_nearest_stations(
        sites=_sites(("A", 0.0)),
        stations=stations,
        observed=_observed(stations=["00009", "00003", "00005"]),
        required=_required("A"),
        k=2,
        min_coverage=1.0,
    )

    assert chosen["src_id"].to_list() == ["00003", "00009"]
    assert chosen["rank"].to_list() == [1, 2]


def test_each_site_takes_its_own_nearest_stations_and_its_own_required_hours():
    stations = _stations(("south", -3.0), ("north", 3.0))
    required = pl.concat([_required("S").head(10), _required("N").head(10)])

    chosen = select_nearest_stations(
        sites=_sites(("S", -2.0), ("N", 2.0)),
        stations=stations,
        observed=_observed(stations=["south", "north"]),
        required=required,
        k=1,
        min_coverage=1.0,
    )

    assert chosen.select("site", "src_id").rows() == [("N", "north"), ("S", "south")]


def test_k_stations_come_back_ranked_with_the_skips_before_each_counted():
    stations = _stations(("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0))

    chosen = select_nearest_stations(
        sites=_sites(("A", 0.0)),
        stations=stations,
        observed=_observed(stations=["a", "b", "c", "d"], missing={"a": 2, "c": 2}),
        required=_required("A"),
        k=2,
        min_coverage=1.0,
    )

    assert chosen.select("rank", "src_id", "skipped_nearer").rows() == [(1, "b", 1), (2, "d", 2)]


def test_too_few_eligible_stations_raises():
    stations = _stations(("near", 1.0))

    with pytest.raises(ValueError, match="fewer than k=2"):
        select_nearest_stations(
            sites=_sites(("A", 0.0)),
            stations=stations,
            observed=_observed(stations=["near"]),
            required=_required("A"),
            k=2,
            min_coverage=1.0,
        )


@pytest.mark.parametrize("min_coverage", [0.0, 1.01])
def test_a_coverage_outside_zero_to_one_raises(min_coverage: float):
    with pytest.raises(ValueError, match="min_coverage"):
        select_nearest_stations(
            sites=_sites(("A", 0.0)),
            stations=_stations(("near", 1.0)),
            observed=_observed(stations=["near"]),
            required=_required("A"),
            k=1,
            min_coverage=min_coverage,
        )


def test_a_k_below_one_raises():
    with pytest.raises(ValueError, match="k must be at least 1"):
        select_nearest_stations(
            sites=_sites(("A", 0.0)),
            stations=_stations(("near", 1.0)),
            observed=_observed(stations=["near"]),
            required=_required("A"),
            k=0,
            min_coverage=1.0,
        )


def test_a_site_with_no_required_hours_raises():
    with pytest.raises(ValueError, match="no required hours"):
        select_nearest_stations(
            sites=_sites(("A", 0.0)),
            stations=_stations(("near", 1.0)),
            observed=_observed(stations=["near"]),
            required=_required("B"),
            k=1,
            min_coverage=1.0,
        )
