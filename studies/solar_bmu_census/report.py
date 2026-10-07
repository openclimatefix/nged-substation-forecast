"""Print every number the solar-BMU census page quotes into `report.md`.

The BMU register, B1610, IGCPU, TEC, and REPD are public, so `report.md` names the BMUs it lists.
Run after `fetch_sources.py`, `classify.py`, `collate.py`, and
`recall_check.py`: `uv run python studies/solar_bmu_census/report.py`.
"""

from datetime import UTC, datetime
from decimal import Decimal
from itertools import pairwise
from typing import Final

import numpy as np
import polars as pl
from classify import (
    COMMISSIONING_SKIP,
    DAYLIGHT_COS_ZENITH,
    MIN_POSITIVE_HALF_HOURS,
    REFERENCE_LATITUDE,
    REFERENCE_LONGITUDE,
    SOLAR_CORRELATION_THRESHOLD,
    analysis_series,
    drop_daytime_zeros,
)
from collate import CAPACITY_COLUMNS
from fetch_sources import OUTPUT_DIR, STUDY_DIR, Window, fetch_bmu_reference, recorded_run
from recall_check import recall_table
from studies.solar import zenith

CORRELATION_BANDS: Final[tuple[float, ...]] = (0.0, 0.3, 0.6, 0.8, 1.0)
"""Edges of the correlation bands, `[lower, upper)`. A correlation of 1.0 is its own band."""
SEASONS: Final[dict[str, tuple[int, ...]]] = {
    "April to September": (4, 5, 6, 7, 8, 9),
    "November to February": (11, 12, 1, 2),
}
MEL_AGREEMENT_MW: Final[float] = 0.5
AGGREGATE_LEAD_PARTIES: Final[int] = 5
NIGHT_ZENITH_DEGREES: Final[float] = 93.0
"""Above this zenith angle the sun is well below the horizon, so a solar unit has no output."""
STORAGE_SHARE: Final[float] = 0.05
"""Output below minus this share of capacity is large enough to be a battery charging."""
GROUPS: Final[tuple[tuple[str, str | None], ...]] = (
    ("All solar BMUs, hybrids included", None),
    ("Hybrid BMUs only", "hybrid"),
    ("Pure PV BMUs only", "pure PV"),
    ("Technology unknown", "unknown"),
)


def _md(frame: pl.DataFrame) -> str:
    """Render a frame as a markdown table, naming the count column `BMUs`."""
    frame = frame.rename({"len": "BMUs"}) if "len" in frame.columns else frame
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    lines = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.iter_rows()]
    return "\n".join([header, rule, *lines])


def _as_float(value: object) -> float:
    """Return a Polars aggregate as a float, treating a missing value as zero.

    Raises:
        TypeError: If the value is not a number.
    """
    if value is None:
        return 0.0
    if isinstance(value, int | float | Decimal):
        return float(value)
    raise TypeError(f"Expected a number, got {type(value).__name__}")


def band_label(*, correlation: float) -> str:
    """Name the correlation band a value falls in."""
    for lower, upper in pairwise(CORRELATION_BANDS):
        if lower <= correlation < upper:
            return f"{lower:.1f} to {upper:.1f}"
    return "1.0" if correlation >= CORRELATION_BANDS[-1] else "below 0.0"


def capacity_table(*, table: pl.DataFrame) -> pl.DataFrame:
    """Sum each capacity column, for all solar BMUs and for each technology group.

    A TEC project or a REPD row that several BMUs match is summed once, because its figure is for
    the site and not for each BMU. Columns are never added to each other.

    Args:
        table: The census table's single-site rows.

    Returns:
        One row per group and capacity column: the number of BMUs, the number of those with a
        value, and the sum in MW.
    """
    rows = []
    for label, technology in GROUPS:
        group = table if technology is None else table.filter(pl.col("technology") == technology)
        for column in CAPACITY_COLUMNS:
            if column == "tec_mw":
                values = group.filter(pl.col(column).is_not_null()).unique("tec_project_id")
            elif column == "repd_installed_capacity_mw":
                values = group.filter(pl.col(column).is_not_null()).unique("repd_ref_id")
            else:
                values = group.filter(pl.col(column).is_not_null())
            rows.append(
                {
                    "group": label,
                    "bmus": group.height,
                    "capacity column": column,
                    "bmus with a value": group.filter(pl.col(column).is_not_null()).height,
                    "sum (MW)": round(float(values[column].sum()), 1) if values.height else 0.0,
                }
            )
    return pl.DataFrame(rows)


def hour_centres(*, solar_ids: list[str], window_label: str) -> pl.DataFrame:
    """Return the output-weighted mean UTC hour of day, by season, over the given BMUs."""
    frames = [
        pl.read_parquet(OUTPUT_DIR / f"{bmu_id}_{window_label}.parquet") for bmu_id in solar_ids
    ]
    output = pl.concat(frames).filter(pl.col("output_mwh") > 0)
    midpoint = pl.col("half_hour_end_time").dt.offset_by("-15m")
    output = output.with_columns(
        month=midpoint.dt.month(), hour=midpoint.dt.hour() + midpoint.dt.minute() / 60
    )
    rows = []
    for season, months in SEASONS.items():
        subset = output.filter(pl.col("month").is_in(months))
        rows.append(
            {
                "season": season,
                "output-weighted mean UTC hour": round(
                    float(
                        np.average(
                            subset["hour"].to_numpy(), weights=subset["output_mwh"].to_numpy()
                        )
                    ),
                    2,
                ),
            }
        )
    return pl.DataFrame(rows)


def data_checks(*, window_label: str, expected_half_hours: int) -> pl.DataFrame:
    """Check every downloaded B1610 file, and return one row per check.

    The checks are the `data-validation` skill's: duplicate half-hours, nulls and NaNs, the span of
    the timestamps, and the number of rows against the half-hours the window holds.
    """
    file_count = len(list(OUTPUT_DIR.glob(f"*_{window_label}.parquet")))
    lazy = pl.scan_parquet(OUTPUT_DIR / f"*_{window_label}.parquet", include_file_paths="file")
    per_file = (
        lazy.group_by("file")
        .agg(
            rows=pl.len(),
            duplicates=pl.len() - pl.col("half_hour_end_time").n_unique(),
            nulls=pl.col("output_mwh").is_null().sum(),
            nans=pl.col("output_mwh").is_nan().sum(),
            first=pl.col("half_hour_end_time").min(),
            last=pl.col("half_hour_end_time").max(),
        )
        .collect()
    )
    rows = [
        ("BMU files", file_count),
        ("BMU files with no rows", file_count - per_file.height),
        ("Total half-hourly rows", int(per_file["rows"].sum())),
        ("Duplicate half-hours (all files)", int(per_file["duplicates"].sum())),
        ("Null outputs", int(per_file["nulls"].sum())),
        ("NaN outputs", int(per_file["nans"].sum())),
        (
            "Files with more rows than the window holds",
            per_file.filter(pl.col("rows") > expected_half_hours).height,
        ),
        ("Earliest half-hour end (UTC)", str(per_file["first"].min())),
        ("Latest half-hour end (UTC)", str(per_file["last"].max())),
    ]
    return pl.DataFrame(rows, schema=["check", "value"], orient="row", strict=False)


def gap_table(*, classes: pl.DataFrame) -> pl.DataFrame:
    """Return the correlations that bound the gap between solar and not-solar BMUs.

    Args:
        classes: The `classes.parquet` rows of every BMU.

    Returns:
        For the single-site BMUs, the lowest correlation among BMUs classed solar and the highest
        among those classed not solar, with and without the daytime zeros removed; and for each
        scope, the number of BMUs classed solar by behaviour with the daytime zeros removed and
        kept.
    """

    def kept(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.filter(
            (pl.col("raw_correlation") > SOLAR_CORRELATION_THRESHOLD)
            & (pl.col("positive_half_hours") >= MIN_POSITIVE_HALF_HOURS)
        )

    single = classes.filter(pl.col("scope") == "single-site")
    solar = single.filter(pl.col("behaviour") == "solar")
    other = single.filter(pl.col("behaviour") == "not_solar")
    rows = [
        ("Single-site, solar: lowest correlation", f"{_as_float(solar['correlation'].min()):.2f}"),
        (
            "Single-site, not solar: highest correlation",
            f"{_as_float(other['correlation'].max()):.2f}",
        ),
        (
            "Single-site, solar: lowest correlation, daytime zeros kept",
            f"{_as_float(solar['raw_correlation'].min()):.2f}",
        ),
        (
            "Single-site, not solar: highest correlation, daytime zeros kept",
            f"{_as_float(other['raw_correlation'].max()):.2f}",
        ),
    ]
    for scope in ("single-site", "aggregate"):
        group = classes.filter(pl.col("scope") == scope)
        rows.append(
            (
                f"{scope} BMUs solar by behaviour, daytime zeros removed / kept",
                f"{group.filter(pl.col('behaviour') == 'solar').height} / {kept(group).height}",
            )
        )
    return pl.DataFrame(rows, schema=["quantity", "value"], orient="row", strict=False)


def mel_agreement_table(*, single: pl.DataFrame) -> pl.DataFrame:
    """Compare each census BMU's Generation Capacity with its largest Maximum Export Limit.

    Args:
        single: The census table's single-site rows.

    Returns:
        The number of BMUs whose two figures agree within 0.5 MW, the largest gap in megawatts,
        and the sum of the absolute gaps as a share of the Generation Capacity sum.
    """
    gap = (pl.col("largest_mel_mw") - pl.col("generation_capacity_mw")).abs()
    frame = single.with_columns(gap=gap)
    total = _as_float(frame["generation_capacity_mw"].sum())
    rows = [
        ("BMUs within 0.5 MW", str(frame.filter(pl.col("gap") <= MEL_AGREEMENT_MW).height)),
        ("Largest gap (MW)", f"{_as_float(frame['gap'].max()):.1f}"),
        (
            "Sum of gaps, % of Generation Capacity",
            f"{_as_float(frame['gap'].sum()) / total * 100:.1f}",
        ),
    ]
    return pl.DataFrame(rows, schema=["quantity", "value"], orient="row", strict=False)


def aggregate_lead_party_table(*, aggregates: pl.DataFrame) -> pl.DataFrame:
    """Return the five lead parties with the most aggregate Generation Capacity."""
    return (
        aggregates.group_by("lead_party")
        .agg(
            BMUs=pl.len(),
            generation_capacity_mw=pl.col("generation_capacity_mw").sum().round(1),
        )
        .sort("generation_capacity_mw", descending=True)
        .head(AGGREGATE_LEAD_PARTIES)
    )


def storage_pattern_table(*, single: pl.DataFrame, window_label: str) -> pl.DataFrame:
    """Return the mean output of each census site's storage BMU around midday and in the evening.

    Args:
        single: The census table's single-site rows, with `storage_bmu_ids`.
        window_label: The window's label in the file names.

    Returns:
        One row for each storage BMU: its identifier and its mean output in megawatts at 13:00 and
        at 18:00 UTC over the window.
    """
    rows = []
    for ids in single["storage_bmu_ids"].to_list():
        for bmu_id in [bmu for bmu in (ids or "").split(";") if bmu]:
            output = pl.read_parquet(OUTPUT_DIR / f"{bmu_id}_{window_label}.parquet")
            hourly = (
                output.with_columns(
                    hour=pl.col("half_hour_end_time").dt.offset_by("-15m").dt.hour(),
                    megawatts=pl.col("output_mwh") * 2,
                )
                .group_by("hour")
                .agg(pl.col("megawatts").mean())
            )
            rows.append(
                {
                    "elexon_bmu_id": bmu_id,
                    "mean_mw_at_13_utc": round(_hour_mean(hourly, 13), 1),
                    "mean_mw_at_18_utc": round(_hour_mean(hourly, 18), 1),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "elexon_bmu_id": pl.String,
            "mean_mw_at_13_utc": pl.Float64,
            "mean_mw_at_18_utc": pl.Float64,
        },
    )


def _hour_mean(hourly: pl.DataFrame, hour: int) -> float:
    """Return the mean output at one hour of day from a table of hourly means."""
    return _as_float(hourly.filter(pl.col("hour") == hour)["megawatts"].item())


def cleaning_table(*, solar_ids: list[str], window: Window) -> pl.DataFrame:
    """Return how much of each solar BMU's output the two cleaning rules remove, in total.

    Args:
        solar_ids: The single-site BMUs that follow the sun.
        window: The study window.

    Returns:
        The share of all their half-hours before the commissioning month, in the commissioning
        month, and removed as daytime zeros, over all of the BMUs together.
    """
    total = commissioning = analysed = dropped = 0
    for bmu_id in solar_ids:
        output = pl.read_parquet(OUTPUT_DIR / f"{bmu_id}_{window.label}.parquet")
        series = analysis_series(output=output, window_start=window.start)
        cleaned = drop_daytime_zeros(series=series)
        total += output.height
        analysed += series.height
        commissioning += output.height - series.height
        dropped += series.height - cleaned.height
    rows = [
        ("Half-hours in the files of the BMUs that follow the sun", total),
        (
            f"Left out as the first {COMMISSIONING_SKIP.days} days after first output, or earlier",
            commissioning,
        ),
        (
            f"Removed as zero output while cos(zenith) > {DAYLIGHT_COS_ZENITH}",
            dropped,
        ),
        ("Half-hours judged", analysed - dropped),
    ]
    return pl.DataFrame(rows, schema=["quantity", "half-hours"], orient="row", strict=False)


def storage_signature_table(*, single: pl.DataFrame, window_label: str) -> pl.DataFrame:
    """List the census BMUs whose output goes below minus 5% of Generation Capacity.

    A battery in the same BMU would show large negative output when it charges. A small negative
    reading at night is the unit's own import.

    Args:
        single: The census table's single-site rows, with `generation_capacity_mw`.
        window_label: The window's label in the file names.

    Returns:
        One row for each BMU with at least one half-hour of import below minus 5% of capacity, or of
        export above 5% of capacity while the sun is more than 3 degrees below the horizon: its
        identifier, name, the two counts, the lowest output in megawatts, and the lowest output as
        a share of capacity.
    """
    rows = []
    for row in single.iter_rows(named=True):
        output = pl.read_parquet(OUTPUT_DIR / f"{row['elexon_bmu_id']}_{window_label}.parquet")
        megawatts = output["output_mwh"] * 2
        capacity = row["generation_capacity_mw"]
        below = int((megawatts < -STORAGE_SHARE * capacity).sum())
        night = (
            zenith(
                stamps=output["half_hour_end_time"].dt.offset_by("-15m"),
                latitude=REFERENCE_LATITUDE,
                longitude=REFERENCE_LONGITUDE,
            )
            > NIGHT_ZENITH_DEGREES
        )
        exports_at_night = int(((megawatts.to_numpy() > STORAGE_SHARE * capacity) & night).sum())
        if below or exports_at_night:
            lowest = _as_float(megawatts.min())
            rows.append(
                {
                    "elexon_bmu_id": row["elexon_bmu_id"],
                    "site_name": row["display_name"],
                    "half_hours_below": below,
                    "night_export_half_hours": exports_at_night,
                    "lowest_mw": round(lowest, 1),
                    "lowest_share_of_capacity": round(lowest / capacity, 2),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "elexon_bmu_id": pl.String,
            "site_name": pl.String,
            "half_hours_below": pl.Int64,
            "night_export_half_hours": pl.Int64,
            "lowest_mw": pl.Float64,
            "lowest_share_of_capacity": pl.Float64,
        },
    )


def summary_table(*, census: pl.DataFrame, window_label: str) -> pl.DataFrame:
    """Return the figures the page's prose quotes that no other table holds.

    Args:
        census: The census table, single-site and aggregate rows.
        window_label: The window's label in the file names.

    Returns:
        The correlation range of the single-site BMUs that follow the sun, the correlation range
        of the aggregates, the largest output as a share of Generation Capacity, the gap between
        the Generation Capacity and Maximum Export Limit sums, and counts of positions by region.
    """
    single = census.filter(pl.col("scope") == "single-site")
    followers = single.filter(pl.col("basis").str.contains("behaviour"))
    aggregates = census.filter(pl.col("scope") == "aggregate").filter(
        pl.col("basis").str.contains("behaviour")
    )
    peak = 0.0
    for row in single.iter_rows(named=True):
        output = pl.read_parquet(OUTPUT_DIR / f"{row['elexon_bmu_id']}_{window_label}.parquet")
        peak = max(peak, _as_float(output["output_mwh"].max()) * 2 / row["generation_capacity_mw"])
    generation = float(single["generation_capacity_mw"].sum())
    mel = float(single["largest_mel_mw"].sum())
    located = single.filter(pl.col("latitude").is_not_null())
    rows = [
        (
            "Single-site BMUs that follow the sun: lowest correlation",
            f"{followers['correlation'].min():.2f}",
        ),
        (
            "Single-site BMUs that follow the sun: highest correlation",
            f"{followers['correlation'].max():.2f}",
        ),
        (
            "Aggregate BMUs that follow the sun: lowest correlation",
            f"{aggregates['correlation'].min():.2f}",
        ),
        (
            "Aggregate BMUs that follow the sun: highest correlation",
            f"{aggregates['correlation'].max():.2f}",
        ),
        ("Largest output of a census BMU, share of its Generation Capacity", f"{peak:.2f}"),
        (
            "Generation Capacity sum minus largest-MEL sum, % of Generation Capacity",
            f"{(generation - mel) / generation * 100:.1f}",
        ),
        ("Single-site BMUs with a position", str(located.height)),
        (
            "Positions north of 55.5 degrees north",
            str(located.filter(pl.col("latitude") > 55.5).height),
        ),
        (
            "Positions south of 53 degrees north",
            str(located.filter(pl.col("latitude") < 53.0).height),
        ),
        (
            "Positions east of the Greenwich meridian",
            str(located.filter(pl.col("longitude") > 0.0).height),
        ),
    ]
    return pl.DataFrame(rows, schema=["quantity", "value"], orient="row", strict=False)


def main() -> None:
    """Write `report.md`."""
    today = datetime.now(UTC)
    _, window = recorded_run()
    classes = pl.read_parquet(STUDY_DIR / "classes.parquet")
    census = pl.read_parquet(STUDY_DIR / "solar_bmus.parquet")
    single = census.filter(pl.col("scope") == "single-site")
    aggregates = census.filter(pl.col("scope") == "aggregate")
    scoped = classes.with_columns(
        band=pl.col("correlation")
        .map_elements(lambda c: band_label(correlation=c), return_dtype=pl.String)
        .fill_null("undefined")
    )
    bands = scoped.group_by("scope", "band").len().sort("scope", "band")
    solar_ids = single.filter(pl.col("basis").str.contains("behaviour"))["elexon_bmu_id"].to_list()
    recall, provenance = recall_table(census_ids=set(census["elexon_bmu_id"]))

    names = {str(r["elexonBmUnit"]): str(r["bmUnitName"]) for r in fetch_bmu_reference()}
    listing = (
        single.select(
            "elexon_bmu_id",
            "site_name",
            "lead_party",
            "connection_type",
            "technology",
            "technology_evidence",
            "basis",
            "correlation",
            *CAPACITY_COLUMNS,
        )
        .with_columns(pl.col("correlation").round(2))
        .sort("elexon_bmu_id")
    )
    to_inspect = (
        scoped.filter(
            (pl.col("scope") == "single-site")
            & (
                pl.col("correlation").is_between(0.3, 0.8)
                | (pl.col("igcpu_solar") & (pl.col("behaviour") != "solar"))
            )
        )
        .with_columns(
            name=pl.col("elexon_bmu_id").replace_strict(
                names, default=None, return_dtype=pl.String
            ),
            correlation=pl.col("correlation").round(2),
        )
        .select("elexon_bmu_id", "name", "behaviour", "correlation", "positive_half_hours")
        .sort("elexon_bmu_id")
    )

    expected = window.expected_half_hours
    intro = (
        f"Generated {today:%Y-%m-%d %H:%M} UTC. Window {window.start:%Y-%m-%d} to "
        f"{window.end:%Y-%m-%d} (UTC, half-open), {expected} half-hours."
    )
    band_note = (
        f"The threshold is {SOLAR_CORRELATION_THRESHOLD}. Bands are `[lower, upper)`, and "
        "`undefined` is a BMU with no output or a constant output."
    )
    aggregate_note = (
        f"{aggregates.height} BMUs; sum of Generation Capacity "
        f"{aggregates['generation_capacity_mw'].sum():.1f} MW."
    )
    by_type = (
        single.group_by("connection_type", "technology").len().sort("connection_type", "technology")
    )
    sections = [
        "# Solar-BMU census report",
        intro,
        "## BMUs classified\n\n" + _md(scoped.group_by("scope").len().sort("scope")),
        "## Correlation with the cosine of the solar zenith, by band\n\n"
        + band_note
        + "\n\n"
        + _md(bands),
        "## What put a BMU in the census\n\n"
        + _md(census.group_by("scope", "basis").len().sort("scope", "basis")),
        "## Single-site solar BMUs by connection type and technology\n\n" + _md(by_type),
        "## Technology evidence\n\n"
        + _md(single.group_by("technology", "technology_evidence").len().sort("technology")),
        "## The single-site census BMUs\n\n" + _md(listing),
        "## Single-site BMUs in the gap band, or typed Solar and not following the sun\n\n"
        + _md(to_inspect),
        "## Capacity figures (single-site BMUs; columns are never added together)\n\n"
        + _md(capacity_table(table=single)),
        "## Aggregate BMUs (supplier, virtual, and other identifiers), reported apart\n\n"
        + aggregate_note,
        "## Output-weighted UTC hour of day, single-site BMUs that follow the sun\n\n"
        + _md(hour_centres(solar_ids=solar_ids, window_label=window.label)),
        f"## TEC recall check (mapping: {provenance})\n\n"
        + _md(
            recall.group_by("Project Status", "technology", "outcome").len().sort("Project Status")
        ),
        "## Classes by scope\n\n"
        + _md(scoped.group_by("scope", "behaviour").len().sort("scope", "behaviour")),
        "## The gap between solar and not solar, single-site BMUs\n\n"
        + _md(gap_table(classes=classes)),
        "## What the cleaning rules remove, BMUs that follow the sun\n\n"
        + _md(cleaning_table(solar_ids=solar_ids, window=window)),
        f"## Census BMUs below -{STORAGE_SHARE * 100:.0f}% of Generation Capacity\n\n"
        + _md(storage_signature_table(single=single, window_label=window.label)),
        "## Generation Capacity against the largest Maximum Export Limit\n\n"
        + _md(mel_agreement_table(single=single)),
        "## Aggregate BMUs: the five lead parties with the most Generation Capacity\n\n"
        + _md(aggregate_lead_party_table(aggregates=aggregates)),
        "## Storage BMUs at the census sites: mean output by hour of day\n\n"
        + _md(storage_pattern_table(single=single, window_label=window.label)),
        "## Figures the page quotes\n\n"
        + _md(summary_table(census=census, window_label=window.label)),
        "## Data checks on the downloaded B1610 files\n\n"
        + _md(data_checks(window_label=window.label, expected_half_hours=expected)),
        "## Single-site solar BMUs: output coverage\n\n"
        + _md(
            single.join(
                classes.select("elexon_bmu_id", "half_hours", "positive_half_hours"),
                on="elexon_bmu_id",
            )
            .select(
                bmus=pl.len(),
                min_rows_share=(pl.col("half_hours") / expected).min().round(3),
                median_rows_share=(pl.col("half_hours") / expected).median().round(3),
                bmus_with_no_output_after_commissioning_month=(
                    pl.col("positive_half_hours") < MIN_POSITIVE_HALF_HOURS
                ).sum(),
            )
            .rename({"bmus": "BMUs"})
        ),
    ]
    (STUDY_DIR / "report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
