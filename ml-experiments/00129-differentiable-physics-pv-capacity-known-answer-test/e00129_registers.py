"""Stage E of experiment 129: the hindsight capacity beside the registers experiment 45 processed.

Both registers are lower bounds, so the check is whether the 80% range reaches the register total
rather than whether the estimate matches it.

- **Feed-in Tariff (FiT) register**, solar installations commissioned by 2021-03, at most 1 MW
  (the sites below the metering cut-off) with a sensitivity at most 5 MW. A postcode district's
  capacity is shared to a Primary by the share of the district's delivery points inside the
  Primary's supply polygon (`fit_district_polygon_shares.parquet`); these shares overlap, so a
  district counted for two Primaries is counted in both. FiT reports installed capacity (DC) and
  declared net capacity (the AC figure).
- **DESNZ domestic solar by constituency**, shared by the share of each constituency inside the
  polygon. Its capacity basis (DC or AC) is not stated, so DC is assumed.

The Embedded Capacity Register and the Renewable Energy Planning Database are not among the files
experiment 45 processed, so they are not compared here.

    uv run --no-sync python ml-experiments/00129-*/e00129_registers.py
"""

from typing import Final

import e00129_common as common
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUTSIDE_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00045-pvdis-4-weekly-capacity-asymmetric-l1" / "outside"
)
SERIES_TABLE: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00103-differentiable-physics-1-data-look" / "series_table.parquet"
)
SIZE_CUTS_KW: Final[tuple[float, ...]] = (1000.0, 5000.0)
"""The register cut-offs: 1 MW is the plan's main line, 5 MW the sensitivity line."""
ARMS: Final[tuple[str, ...]] = ("arm2", "arm4@0.1")


def fit_by_primary(*, ids: dict[int, str]) -> pl.DataFrame:
    """Return the FiT capacity per Primary, in MW, for each size cut-off."""
    installations = pl.read_parquet(OUTSIDE_DIR / "fit_solar_pv_installations.parquet")
    shares = pl.read_parquet(OUTSIDE_DIR / "fit_district_polygon_shares.parquet")
    frames = []
    for cut in SIZE_CUTS_KW:
        by_district = (
            installations.filter(pl.col("installed_capacity_kw") <= cut)
            .group_by("postcode")
            .agg(
                dc_kw=pl.col("installed_capacity_kw").sum(),
                ac_kw=pl.col("declared_net_capacity_kw").sum(),
            )
        )
        frames.append(
            shares.join(by_district, left_on="district", right_on="postcode")
            .group_by("time_series_id")
            .agg(
                fit_dc_mw=(pl.col("share") * pl.col("dc_kw")).sum() / 1000.0,
                fit_ac_mw=(pl.col("share") * pl.col("ac_kw")).sum() / 1000.0,
            )
            .with_columns(cut_mw=pl.lit(cut / 1000.0))
        )
    return (
        pl.concat(frames)
        .with_columns(site=pl.col("time_series_id").replace_strict(ids, default=None))
        .drop_nulls("site")
    )


def desnz_by_primary(*, ids: dict[int, str]) -> pl.DataFrame:
    """Return the DESNZ domestic solar capacity per Primary, in MW."""
    return (
        pl.read_parquet(OUTSIDE_DIR / "constituency_primary_membership.parquet")
        .join(
            pl.read_parquet(OUTSIDE_DIR / "desnz_constituency_domestic_solar.parquet"),
            left_on="PCON24CD",
            right_on="constituency_code",
        )
        .group_by("time_series_id")
        .agg(
            desnz_domestic_mw=(
                pl.col("share_of_feature_inside_polygon") * pl.col("domestic_installed_capacity_mw")
            ).sum()
        )
        .with_columns(site=pl.col("time_series_id").replace_strict(ids, default=None))
        .drop_nulls("site")
    )


def main() -> None:
    """Write `work/registers.parquet` and print the comparison."""
    series = pl.read_parquet(SERIES_TABLE)
    ids = dict(series.select("time_series_id", "site").iter_rows())
    sites = list(common.DECISION_SITES)
    estimates = pl.concat(
        [
            pl.read_parquet(common.WORK_DIR / f"real_{arm}_to_july.parquet").filter(
                pl.col("season_year") == common.SEASON_YEARS[-1]
            )
            for arm in ARMS
        ]
    ).select("site", "arm", "estimate_mw", "range_low_mw", "range_high_mw", "pinned_mw")
    table = (
        estimates.join(fit_by_primary(ids=ids), on="site")
        .join(desnz_by_primary(ids=ids).drop("time_series_id"), on="site")
        .filter(pl.col("site").is_in(sites))
        .with_columns(range_reaches_fit_dc=pl.col("range_high_mw") >= pl.col("fit_dc_mw"))
        .sort("arm", "cut_mw", "site")
    )
    table.write_parquet(common.WORK_DIR / "registers.parquet")
    with pl.Config(tbl_rows=40, tbl_cols=14, float_precision=2):
        print(table.drop("time_series_id"))


if __name__ == "__main__":
    main()
