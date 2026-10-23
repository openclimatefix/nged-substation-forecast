"""Build the paired node-and-generator arms of experiment 271 in MW.

The node forecast is experiment 48's best guess, held fixed. The three generators that are not
metered farms keep experiment 48's best-guess generator forecast. Only the six farms' forecasts
differ between arms.

A stored farm forecast is the 00 UTC run, 12 to 36 hours before its valid time. A node forecast is
made at 09 UTC. For a valid time at or after 12 UTC on the day after the node origin, the farm run
is the 00 UTC run of that later day, which the node forecast cannot have seen. The `causal` pairing
therefore drops those node rows from every arm. The `valid_time` pairing keeps them and gives each
node row the farm forecast stored for its valid time.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import e00271_inputs as inputs
import polars as pl
from studies.node_curtailment import NodeTable, capacity_forecast, scale_to_megawatts

PairingType = Literal["causal", "valid_time"]

CAPACITY_BASELINE: Final[str] = "nged_baseline_unconfirmed"
FLOOR_HEURISTIC: Final[str] = "floor_heuristic"
FLOOR_CLIMATOLOGY: Final[str] = "floor_climatology"
PLANT: Final[str] = "plant_two_number"
XGBOOST: Final[str] = "xgboost"
H5: Final[str] = "h5"
H5_ZERO: Final[str] = "h5_zero_history"
H5_MEAN5: Final[str] = "h5_mean5"
PERFECT: Final[str] = "perfect"
DIAGNOSTIC: Final[str] = "best_guess_nodes_actual_generators"
FORECAST_ARMS: Final[tuple[str, ...]] = (
    FLOOR_HEURISTIC,
    FLOOR_CLIMATOLOGY,
    PLANT,
    XGBOOST,
    H5,
    H5_ZERO,
)
ARM_LABELS: Final[Mapping[str, str]] = {
    CAPACITY_BASELINE: "NGED baseline (unconfirmed)",
    FLOOR_HEURISTIC: "Farm floor: manual heuristic",
    FLOOR_CLIMATOLOGY: "Farm floor: climatology",
    PLANT: "Plant, two-number",
    XGBOOST: "XGBoost",
    H5: "H5 (plant + neural head)",
    H5_ZERO: "H5, zero history",
    PERFECT: "Perfect forecast",
    DIAGNOSTIC: "Best-guess nodes, actual generation",
}
"""Display names; `CAPACITY_BASELINE` is the maintainer's understanding of NGED's practice."""


@dataclass(frozen=True)
class Arms:
    """The frames `node_curtailment` scores, and counts for the page."""

    node_frames: dict[str, pl.DataFrame]
    generator_frames: dict[str, pl.DataFrame]
    n_node_keys_before: int
    n_node_keys_after: int
    farm_frames_mw: dict[str, pl.DataFrame]
    """Each forecast arm's joined farm rows in MW (daylight and night), for the tail table."""


def _node_and_other_generators(
    *, table: NodeTable, capacity_mw: Mapping[int, float]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return experiment 48's best-guess node forecast and non-farm generator forecast, in MW."""
    raw = inputs.read_arm_frames(names=("weekly", "gen_monthly"))
    window = (pl.col("valid_time") >= inputs.SCORED_START) & (
        pl.col("valid_time") < inputs.SCORED_END
    )
    node_ids = set(table.nodes["time_series_id"].to_list())
    farm_ids = set(inputs.farm_sites()["time_series_id"].to_list())
    generator_ids = set(table.generators["generator_id"].to_list()) - farm_ids
    nodes = scale_to_megawatts(
        frame=raw["weekly"].filter(pl.col("time_series_id").is_in(node_ids), window),
        capacity_mw=capacity_mw,
    )
    others = scale_to_megawatts(
        frame=raw["gen_monthly"].filter(pl.col("time_series_id").is_in(generator_ids), window),
        capacity_mw=capacity_mw,
    )
    return nodes, others


def _daylight_by_valid_time(*, valid_times: pl.Series, sites: pl.DataFrame) -> pl.DataFrame:
    """Return `farm`, `valid_time`, `daylight` for every farm and distinct valid time."""
    unique = valid_times.unique().sort()
    return pl.concat(
        pl.DataFrame(
            {
                "farm": site["site"],
                "valid_time": unique,
                "daylight": inputs.is_daylight(
                    valid_times=unique, latitude=site["latitude"], longitude=site["longitude"]
                ),
            }
        )
        for site in sites.iter_rows(named=True)
    )


def _farm_rows(
    *,
    forecast: pl.DataFrame,
    keys: pl.DataFrame,
    daylight: pl.DataFrame,
    farm_power: pl.DataFrame,
    sites: pl.DataFrame,
    capacity_by_series: Mapping[int, float],
) -> pl.DataFrame:
    """Join one arm's farm forecasts onto the node keys, in MW, with 0 MW at night.

    Daylight half-hours take the stored forecast and its `actual`; a daylight half-hour the arm does
    not hold stays absent, so the generator is not instructed. Other half-hours take a forecast of
    0 MW and the cleaned metered output, which is null where the cleaning flagged it.
    """
    farms = sites.select("site", series="time_series_id")
    capacity = pl.DataFrame(
        {"series": list(capacity_by_series), "capacity_mw": list(capacity_by_series.values())},
        schema={"series": pl.Int64, "capacity_mw": pl.Float64},
    )
    grid = (
        keys.join(farms, how="cross")
        .join(daylight.rename({"farm": "site"}), on=["site", "valid_time"], how="left")
        .join(capacity, on="series", how="left")
    )
    night = (
        grid.filter(~pl.col("daylight"))
        .join(
            farm_power.rename({"time_series_id": "series"}),
            on=["series", "valid_time"],
            how="left",
        )
        .with_columns(
            pl.col("output_mw").alias("actual"),
            *(pl.lit(0.0).alias(q) for q in inputs.QUANTILES),
            night=pl.lit(True),
        )
    )
    day = (
        grid.filter(pl.col("daylight"))
        .join(
            forecast.drop("lead_hours").rename({"farm": "site"}),
            on=["site", "valid_time"],
            how="inner",
        )
        .with_columns(
            (pl.col("actual") * pl.col("capacity_mw")).alias("actual"),
            *(
                (pl.col(q) * pl.col("capacity_mw")).clip(lower_bound=0.0).alias(q)
                for q in inputs.QUANTILES
            ),
            night=pl.lit(False),
        )
    )
    columns = [
        "site",
        "series",
        "power_fcst_init_time",
        "valid_time",
        "seed",
        "capacity_mw",
        "actual",
        "night",
    ]
    return pl.concat(
        [night.select(*columns, *inputs.QUANTILES), day.select(*columns, *inputs.QUANTILES)]
    ).rename({"series": "time_series_id"})


def _clip_to_capacity(*, frame: pl.DataFrame, capacities: pl.DataFrame) -> pl.DataFrame:
    """Cap every quantile at the generator's table capacity, leaving `actual` alone."""
    quantiles = [q for q in frame.columns if q.startswith("q") and q[1:].isdigit()]
    joined = frame.join(capacities, on="time_series_id", how="left", maintain_order="left")
    return joined.with_columns(
        *(pl.min_horizontal(q, "capacity_limit_mw").alias(q) for q in quantiles)
    ).drop("capacity_limit_mw")


def build_arms(
    *,
    table: NodeTable,
    capacity_mw: Mapping[int, float],
    pairing: PairingType,
    forecasts: Mapping[str, pl.DataFrame],
    reference_arm: str,
) -> Arms:
    """Return every arm's node and generator forecast frames in MW.

    Args:
        table: The scoring table (two bulk supply points).
        capacity_mw: Effective capacity in MW by series, as experiment 48 normalised by.
        pairing: `causal` drops the node rows whose daylight farm forecast comes from a later run
            than the node origin; `valid_time` keeps every row.
        forecasts: The farm forecast arms to score, each on the same farm rows, in fractions of
            capacity.
        reference_arm: The arm in `forecasts` whose rows the baseline, the perfect arm and the
            diagnostic arm copy.

    Returns:
        The arms, and the count of node `(origin, valid time)` keys before and after the pairing.
    """
    sites = inputs.farm_sites()
    nodes, others = _node_and_other_generators(table=table, capacity_mw=capacity_mw)
    lead = pl.col("valid_time") - pl.col("power_fcst_init_time")
    nodes = nodes.filter(lead >= pl.duration(hours=12), lead < pl.duration(hours=36))
    others = others.filter(lead >= pl.duration(hours=12), lead < pl.duration(hours=36))
    keys = nodes.select("power_fcst_init_time", "valid_time", "seed").unique()
    daylight = _daylight_by_valid_time(valid_times=keys["valid_time"], sites=sites)
    n_before = keys.height
    if pairing == "causal":
        any_day = daylight.group_by("valid_time").agg(any_day=pl.col("daylight").any())
        late = (
            keys.join(any_day, on="valid_time", how="left")
            .with_columns(
                run_time=pl.col("valid_time")
                .dt.offset_by(f"-{int(inputs.RUN_LEAD_MIN_HOURS)}h")
                .dt.truncate("1d")
            )
            .filter(pl.col("any_day") & (pl.col("run_time") > pl.col("power_fcst_init_time")))
            .select("power_fcst_init_time", "valid_time", "seed")
        )
        keys = keys.join(late, on=["power_fcst_init_time", "valid_time", "seed"], how="anti")
        nodes = nodes.join(late, on=["power_fcst_init_time", "valid_time", "seed"], how="anti")
        others = others.join(late, on=["power_fcst_init_time", "valid_time", "seed"], how="anti")
    farm_power = inputs.read_farm_power_mw()
    capacities = table.generators.select(
        pl.col("generator_id").cast(pl.Int64).alias("time_series_id"),
        pl.col("capacity_mw").alias("capacity_limit_mw"),
    )
    capacity_by_series = dict(
        zip(
            table.generators["generator_id"].cast(pl.Int64).to_list(),
            table.generators["capacity_mw"].to_list(),
            strict=True,
        )
    )
    farm_frames = {
        arm: _farm_rows(
            forecast=forecasts[arm],
            keys=keys,
            daylight=daylight,
            farm_power=farm_power,
            sites=sites,
            capacity_by_series=capacity_by_series,
        )
        for arm in forecasts
    }
    generator_columns = [
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        "seed",
        "actual",
        *inputs.QUANTILES,
    ]
    others_full = others.select(
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        "seed",
        "actual",
        *(
            pl.col(q) if q in others.columns else pl.lit(None, dtype=pl.Float64).alias(q)
            for q in inputs.QUANTILES
        ),
    ).with_columns(pl.col("seed").cast(pl.Int64))

    def generators(arm: str) -> pl.DataFrame:
        farms = (
            farm_frames[arm].select(generator_columns).with_columns(pl.col("seed").cast(pl.Int64))
        )
        both = pl.concat([farms, others_full.select(generator_columns)], how="vertical_relaxed")
        return _clip_to_capacity(frame=both, capacities=capacities)

    generator_frames = {arm: generators(arm) for arm in forecasts}
    node_frame = nodes.drop("capacity_missing", strict=False).with_columns(
        pl.col("seed").cast(pl.Int64)
    )
    node_frames = dict.fromkeys(forecasts, node_frame)

    quantile_columns = list(inputs.QUANTILES)

    def perfect(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(*(pl.col("actual").alias(q) for q in quantile_columns))

    reference_generators = generator_frames[reference_arm]
    generator_frames[PERFECT] = _clip_to_capacity(
        frame=perfect(reference_generators), capacities=capacities
    )
    node_frames[PERFECT] = perfect(
        node_frame.with_columns(
            *(
                pl.lit(None, dtype=pl.Float64).alias(q)
                for q in quantile_columns
                if q not in node_frame.columns
            )
        )
    )
    node_frames[DIAGNOSTIC] = node_frame
    generator_frames[DIAGNOSTIC] = generator_frames[PERFECT]
    node_frames[CAPACITY_BASELINE] = node_frame
    generator_frames[CAPACITY_BASELINE] = capacity_forecast(
        generator_frame=reference_generators, table=table
    )
    return Arms(
        node_frames=node_frames,
        generator_frames=generator_frames,
        n_node_keys_before=n_before,
        n_node_keys_after=keys.height,
        farm_frames_mw=farm_frames,
    )
