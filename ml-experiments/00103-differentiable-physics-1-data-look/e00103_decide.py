"""Compute experiment 103's decisions, tables, and handed-on files.

Writes to the private folder: `tables/*.parquet` (inputs of the figures), `sites.parquet`,
`rows.parquet`, and `decisions.json`; prints every number the write-up quotes. The two metered
generator series are printed as G-a and G-b, never by D-label or identifier.

    POLARS_MAX_THREADS=4 uv run python ml-experiments/00103-*/e00103_decide.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
from e00103_common import (
    GENERATOR_SERIES,
    MVA_PRIMARIES,
    MW_LONG_HISTORY,
    MW_PRIMARIES,
    PERIOD_END,
    PRIMARIES,
    WORK_DIR,
    load_series,
)
from e00103_stats import (
    availability_by_month,
    clip_slopes,
    daily_table,
    dark_site_years,
    early_calibration,
    first_clean_month,
    heating_slope_drift,
    heating_slopes,
    in_lockdown_window,
    leading_cut,
    level_correlations,
    load_panel,
    lockdown_departures,
    monthly_levels,
    scarce_census,
    season_year,
    timestamp_correlation,
)

TABLE_DIR: Final = WORK_DIR / "tables"
ORIGINS: Final[tuple[datetime, ...]] = tuple(
    datetime(y, 7, 1, tzinfo=UTC) for y in (2022, 2023, 2024, 2025)
)
"""The first instant of each tuning year; months of history are counted up to these."""
NEGATIVE_DARK_SHARE_LIMIT: Final[float] = 0.01
"""A series is flagged when more than this share of its dark clean half-hours are below zero."""
MIN_DARK_SITE_YEARS: Final[float] = 0.5


def label(*, site: str) -> str:
    """The name printed or plotted for a series: G-a or G-b for the metered generators."""
    return GENERATOR_SERIES.get(site, site)


def months_between(*, start: datetime, end: datetime) -> float:
    """Whole-and-fractional months from `start` to `end`, zero when `start` is later."""
    return max(0.0, (end - start).total_seconds() / (30.4375 * 86400))


def main() -> None:  # noqa: PLR0915  (runs each decision in turn and prints its numbers)
    """Run every decision in turn and write the files."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    series = load_series()
    sites = dict(series.select("time_series_id", "site").iter_rows())
    ids = {site: sid for sid, site in sites.items()}
    extended = load_panel(extended=True)
    panel = extended.filter(pl.col("time") <= PERIOD_END)

    first_clean = first_clean_month(panel=extended)
    levels_ext = monthly_levels(panel=extended)
    calibration = early_calibration(levels=levels_ext, first_clean=first_clean)
    primary_ids = [ids[s] for s in PRIMARIES]
    cuts = leading_cut(calibration=calibration, first_clean=first_clean).with_columns(
        first_usable=pl.when(pl.col("time_series_id").is_in(primary_ids))
        .then(pl.col("first_usable"))
        .otherwise(pl.col("first_clean"))
    )
    first_usable = dict(cuts.select("time_series_id", "first_usable").iter_rows())
    calibration.write_parquet(TABLE_DIR / "early_calibration.parquet")
    cuts.write_parquet(TABLE_DIR / "cuts.parquet")
    print("\n== early-calibration cuts (Primaries) ==")
    for site in PRIMARIES:
        row = cuts.filter(pl.col("time_series_id") == ids[site]).row(0, named=True)
        print(
            f"{site}: first clean {row['first_clean']:%Y-%m-%d}, cut {row['cut_months']} months,"
            f" first usable {row['first_usable']:%Y-%m-%d}"
        )

    levels = monthly_levels(panel=panel)
    levels.write_parquet(TABLE_DIR / "monthly_levels.parquet")
    availability_by_month(panel=panel).write_parquet(TABLE_DIR / "availability.parquet")

    # Lockdown window.
    departures = lockdown_departures(
        levels=levels,
        series_ids=[ids[s] for s in ("D5", "D10", "D11", "D14")],
        first_usable=first_usable,
    )
    departures.write_parquet(TABLE_DIR / "lockdown_departures.parquet")
    print(
        "\n== lockdown: fixed window 2020-03-23 to 2021-07-19 (experiment 6's COVID indicator) =="
    )
    print("weekday 09:00-16:59 departure from a no-lockdown fit, points of effective capacity:")
    print(departures.pivot(on="time_series_id", index="month", values="departure").sort("month"))

    # Mark rows.
    panel = panel.join(
        cuts.select("time_series_id", "first_usable"), on="time_series_id"
    ).with_columns(
        early_cut=pl.col("time") < pl.col("first_usable"),
        lockdown=in_lockdown_window(local_date=pl.col("local_date")),
        season=season_year(date=pl.col("local_date")),
    )
    after_cut = panel.filter(~pl.col("early_cut"))

    # Dark site-years per window.
    usable_dark = after_cut.filter(pl.col("clean"), pl.col("dark"), ~pl.col("switching_mask"))
    windows = {
        "all history, lockdown kept": usable_dark,
        "lockdown removed": usable_dark.filter(~pl.col("lockdown")),
        "latest two season-years": usable_dark.filter(pl.col("season") >= 2024),
    }
    dark_rows = []
    for name, frame in windows.items():
        table = (
            frame.group_by("time_series_id")
            .agg(
                dark_half_hours=pl.len(),
                evening=(
                    (pl.col("local_hour_start") >= 16) & (pl.col("local_hour_start") < 22)
                ).sum(),
            )
            .with_columns(
                window=pl.lit(name),
                site_years=pl.col("dark_half_hours") / (365.25 * 48),
                evening_share=pl.col("evening") / pl.col("dark_half_hours"),
            )
        )
        dark_rows.append(table)
    dark = pl.concat(dark_rows).with_columns(
        site=pl.col("time_series_id").replace_strict(sites, return_dtype=pl.String)
    )
    dark.write_parquet(TABLE_DIR / "dark_site_years.parquet")
    by_season = dark_site_years(panel=after_cut.filter(~pl.col("switching_mask")))
    by_season.write_parquet(TABLE_DIR / "dark_by_season.parquet")
    print("\n== dark site-years per MVA Primary and window ==")
    mva = (
        dark.filter(pl.col("site").is_in(MVA_PRIMARIES))
        .pivot(on="window", index="site", values="site_years")
        .sort("site")
    )
    print(mva)
    print(
        "totals:",
        dark.filter(pl.col("site").is_in(MVA_PRIMARIES))
        .group_by("window")
        .agg(
            pl.col("site_years").sum(),
            (pl.col("evening").sum() / pl.col("dark_half_hours").sum()).alias("evening_share"),
        ),
    )

    # Generator screen on rows after cuts and masks.
    screen = generator_screen(
        after_cut=after_cut.filter(~pl.col("switching_mask")), sites=sites, series=series
    )
    screen.write_parquet(TABLE_DIR / "generator_screen.parquet")
    print("\n== generator screen (after cuts and masks) ==")
    print(screen.with_columns(site=pl.col("site").replace(GENERATOR_SERIES)).drop("time_series_id"))

    # Clip, heating, correlations, timestamps, census.
    clip = clip_slopes(panel=after_cut.filter(~pl.col("switching_mask")))
    clip.write_parquet(TABLE_DIR / "clip.parquet")
    daily = daily_table(panel=after_cut.filter(~pl.col("switching_mask")))
    daily.write_parquet(TABLE_DIR / "daily.parquet")
    heating = heating_slopes(daily=daily)
    heating.write_parquet(TABLE_DIR / "heating.parquet")
    drift = pl.DataFrame(
        [
            heating_slope_drift(daily=daily, series_id=ids[s], first_usable=first_usable[ids[s]])
            for s in MW_LONG_HISTORY
        ]
    )
    drift.write_parquet(TABLE_DIR / "heating_drift.parquet")
    print("\n== clip slopes (MW Primaries; points per 100 W/m2) ==")
    print(
        clip.with_columns(
            site=pl.col("time_series_id").replace_strict(sites, return_dtype=pl.String)
        )
        .filter(pl.col("site").is_in(MW_PRIMARIES))
        .sort("site", "season")
    )
    print("\n== heating-slope drift (last minus first winter, slopes in points per degC) ==")
    print(
        drift.with_columns(site=pl.col("series_id").replace_strict(sites, return_dtype=pl.String))
    )
    correlations = {}
    for band in ("night", "evening"):
        corr = level_correlations(
            levels=levels, series_ids=[ids[s] for s in MW_LONG_HISTORY if s != "D9"], band=band
        )
        corr.write_parquet(TABLE_DIR / f"level_correlation_{band}.parquet")
        matrix = corr.drop("series", "months").to_numpy()
        off = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        eigenvalues = np.linalg.eigvalsh(matrix)
        correlations[band] = {
            "mean_pairwise": float(off.mean()),
            "first_component_share": float(eigenvalues[-1] / eigenvalues.sum()),
            "months": int(corr["months"][0]),
        }
    print("\n== correlation of monthly levels (D5, D10, D11, D14, D16; D9 left out, too short) ==")
    print(correlations)
    timestamp_correlation(panel=extended).write_parquet(TABLE_DIR / "timestamp_all_months.parquet")
    timestamps = timestamp_correlation(
        panel=extended.filter(pl.col("time").dt.month().is_between(4, 9))
    )
    timestamps.write_parquet(TABLE_DIR / "timestamp.parquet")
    print("\n== timestamp: mean correlation over MW Primaries by shift ==")
    print(
        timestamps.group_by("after_fix", "shift_minutes")
        .agg(pl.col("correlation").mean(), pl.col("n").sum())
        .sort("after_fix", "shift_minutes")
    )
    census = scarce_census(panel=panel.filter(pl.col("time_series_id").is_in(primary_ids)))
    census.write_parquet(TABLE_DIR / "census.parquet")
    print("\n== scarce-data census: mean per MW long-history site ==")
    print(
        census.filter(pl.col("time_series_id").is_in([ids[s] for s in MW_LONG_HISTORY]))
        .group_by("window")
        .agg(
            pl.col("cold_days").mean(),
            pl.col("bank_holidays").mean(),
            pl.col("christmas_days").mean(),
            pl.col("time_series_id").n_unique().alias("sites"),
        )
        .sort("window")
    )
    mask_census = (
        panel.filter(pl.col("time_series_id").is_in(primary_ids), pl.col("switching_mask"))
        .group_by("time_series_id", "season")
        .agg(masked=pl.len())
        .sort("time_series_id", "season")
        .with_columns(site=pl.col("time_series_id").replace_strict(sites, return_dtype=pl.String))
    )
    mask_census.write_parquet(TABLE_DIR / "mask_census.parquet")
    print("\n== switching mask half-hours per season-year (Primaries) ==")
    print(mask_census.pivot(on="season", index="site", values="masked").sort("site"))

    write_handed_on(
        panel=panel,
        series=series,
        sites=sites,
        ids=ids,
        cuts=cuts,
        dark=dark,
        screen=screen,
        clip=clip,
        drift=drift,
        correlations=correlations,
    )


def generator_screen(*, after_cut: pl.DataFrame, sites: dict, series: pl.DataFrame) -> pl.DataFrame:
    """Per series: negative dark share, winter night slope, and amplitude ratios (after cuts)."""
    clean = after_cut.filter(pl.col("clean"))
    dark = (
        clean.filter(pl.col("dark"))
        .group_by("time_series_id")
        .agg(dark_rows=pl.len(), negative_dark_rows=(pl.col("power") < 0).sum())
        .with_columns(negative_dark_share=pl.col("negative_dark_rows") / pl.col("dark_rows"))
    )
    daily = daily_table(panel=after_cut)
    winter = daily.filter(
        pl.col("local_date").dt.month().is_in([11, 12, 1, 2]),
        pl.col("night_n") >= 8,
        pl.col("night_mean").is_not_null(),
        pl.col("day_n") >= 44,
    )
    rows = []
    for series_id, part in winter.group_by("time_series_id"):
        x = part["temperature_c"].to_numpy().astype(float)
        y = part["night_mean"].to_numpy().astype(float)
        rows.append(
            {
                "time_series_id": series_id[0],
                "winter_night_slope": float(np.polyfit(x, y, 1)[0]) if part.height > 20 else None,
            }
        )
    slopes = pl.DataFrame(
        rows, schema={"time_series_id": pl.Int32, "winter_night_slope": pl.Float64}
    )
    amplitude = (
        daily.filter(pl.col("day_n") >= 44)
        .group_by("time_series_id")
        .agg(
            range_over_capacity=pl.col("day_range").median() / 100.0,
            range_over_mean=(pl.col("day_range") / pl.col("day_mean").abs()).median(),
        )
    )
    return (
        series.select("time_series_id", "site", "units", "group")
        .join(dark, on="time_series_id", how="left")
        .join(slopes, on="time_series_id", how="left")
        .join(amplitude, on="time_series_id", how="left")
        .with_columns(
            flagged=(pl.col("units") == "MW")
            & (pl.col("negative_dark_share") > NEGATIVE_DARK_SHARE_LIMIT)
        )
        .sort("site")
    )


def write_handed_on(
    *,
    panel: pl.DataFrame,
    series: pl.DataFrame,
    sites: dict[int, str],
    ids: dict[str, int],
    cuts: pl.DataFrame,
    dark: pl.DataFrame,
    screen: pl.DataFrame,
    clip: pl.DataFrame,
    drift: pl.DataFrame,
    correlations: dict[str, dict[str, float]],
) -> None:
    """Write `sites.parquet`, `rows.parquet`, and `decisions.json`."""
    import holidays

    england = holidays.UnitedKingdom(subdiv="ENG", years=range(2019, 2027))
    bank_days = pl.Series(sorted(england.keys())).cast(pl.Date)
    flagged = set(screen.filter(pl.col("flagged"))["site"].to_list())
    first_usable = dict(cuts.select("time_series_id", "first_usable").iter_rows())
    dark_lockdown_removed = {
        r["site"]: r["site_years"]
        for r in dark.filter(pl.col("window") == "lockdown removed").iter_rows(named=True)
    }
    dark_two_years = {
        r["site"]: r["site_years"]
        for r in dark.filter(pl.col("window") == "latest two season-years").iter_rows(named=True)
    }
    rows = []
    for site in series["site"].to_list():
        sid = ids[site]
        start = first_usable[sid]
        history = {f"months_before_{o:%Y-%m}": months_between(start=start, end=o) for o in ORIGINS}
        if site in GENERATOR_SERIES or site in ("D1", "D6", "D13", "D17"):
            role, reason = "excluded", "not a Primary substation"
        elif site == "D15":
            role, reason = "generator_case", "carries an embedded generator; out of the demand pool"
        elif site in flagged:
            role, reason = "excluded", "flagged by the generator screen"
        elif site in MVA_PRIMARIES:
            role = "mva_dark" if dark_two_years.get(site, 0) >= MIN_DARK_SITE_YEARS else "excluded"
            reason = (
                "under 0.5 dark site-years in the latest two season-years"
                if role == "excluded"
                else ""
            )
        elif history["months_before_2025-07"] >= 12 and site != "D18":
            role, reason = "mw_long", ""
        else:
            role, reason = (
                "mw_short",
                "under 12 months before 2025-07-01 or short history by definition",
            )
        rows.append(
            {
                "site": GENERATOR_SERIES.get(site, site),
                "time_series_id": sid if site not in GENERATOR_SERIES else None,
                "units": series.filter(pl.col("site") == site)["units"][0],
                "role": role,
                "first_usable": start,
                "cut_months": cuts.filter(pl.col("time_series_id") == sid)["cut_months"][0],
                "note": reason,
                **history,
                "dark_site_years_no_lockdown": dark_lockdown_removed.get(site),
                "dark_site_years_two_years": dark_two_years.get(site),
            }
        )
    sites_table = pl.DataFrame(rows)
    sites_table.write_parquet(WORK_DIR / "sites.parquet")
    print("\n== site table ==")
    print(sites_table.drop("time_series_id"))
    role_of = {r["site"]: r["role"] for r in rows}
    pool = panel.filter(pl.col("time_series_id").is_in([ids[s] for s in PRIMARIES]))
    day_type = (
        pl.when(pl.col("local_date").is_in(bank_days.implode()))
        .then(pl.lit("bank_holiday"))
        .when(pl.col("weekday") == 6)
        .then(pl.lit("saturday"))
        .when(pl.col("weekday") == 7)
        .then(pl.lit("sunday"))
        .otherwise(pl.lit("weekday"))
    )
    christmas = (
        (pl.col("local_date").dt.month() == 12) & (pl.col("local_date").dt.day() >= 24)
    ) | ((pl.col("local_date").dt.month() == 1) & (pl.col("local_date").dt.day() == 1))
    pool = pool.with_columns(
        day_type=day_type,
        christmas_run=christmas,
        role=pl.col("site").replace_strict(role_of, return_dtype=pl.String),
    ).with_columns(
        in_pool=pl.col("role").is_in(["mw_long", "mw_short", "mva_dark"])
        & ~pl.col("early_cut")
        & pl.col("clean"),
        scarce_kind=pl.when(pl.col("temperature_c").mean().over("time_series_id", "local_date") < 0)
        .then(pl.lit("cold_day"))
        .when(pl.col("day_type") == "bank_holiday")
        .then(pl.lit("bank_holiday"))
        .when(pl.col("christmas_run"))
        .then(pl.lit("christmas_run")),
    )
    rows_table = (
        pool.filter(pl.col("time") >= pl.col("first_usable"))
        .select(
            "time_series_id",
            "site",
            "time",
            "power",
            "present",
            "in_pool",
            "early_cut",
            "lockdown",
            "switching_mask",
            "dark",
            "scarce_kind",
            "season",
            "day_type",
            "role",
        )
        .rename({"season": "season_year"})
    )
    rows_table.write_parquet(WORK_DIR / "rows.parquet")
    mw_rows = rows_table.filter(
        pl.col("in_pool"), pl.col("role").is_in(["mw_long", "mw_short"])
    ).height
    mva_rows = rows_table.filter(
        pl.col("in_pool"), pl.col("role") == "mva_dark", pl.col("dark")
    ).height
    print(
        f"\nrows.parquet: {rows_table.height} rows; dp_mw_v1 rows {mw_rows};"
        f" dp_mva_dark_v1 rows {mva_rows}"
    )
    decisions = {
        "lockdown_window": "fixed, 2020-03-23 to 2021-07-19 (experiment 6's COVID indicator)",
        "correlations_of_monthly_levels": correlations,
        "roles": role_of,
        "heating_slope_drift_sites_excluding_zero": [
            r["series_id"]
            for r in drift.iter_rows(named=True)
            if r.get("low") is not None and (r["low"] > 0 or r["high"] < 0)
        ],
        "row_counts": {"dp_mw_v1": mw_rows, "dp_mva_dark_v1": mva_rows},
    }
    (WORK_DIR / "decisions.json").write_text(json.dumps(decisions, indent=2, default=str))


if __name__ == "__main__":
    main()
