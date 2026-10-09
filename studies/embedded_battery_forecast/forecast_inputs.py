"""Build the input frames of the embedded-battery forecast, one wide frame per battery and issue.

For each battery and each issue time (`DA-early`, `DA-late`, `ID-1h`), `build_issue_frame` returns
one row per half-hour of the window with the target (the battery's output), the three price sources
and their derived columns, the persistence value, the probabilistic climatology, the battery's own
Physical Notification, and the neighbours' statistics. `arm_frame` then picks the fixed tuple of
`FEATURE_COLUMNS` that one arm of the forecast sees, so every arm has the same number of columns.

Run: `uv run python studies/embedded_battery_forecast/forecast_inputs.py`. Writes the frames and
`inputs_report.md` under `data/studies/per_study/embedded_battery_forecast/inputs/`.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final, Literal

import numpy as np
import polars as pl
from census import BMU_LIST_PATH, PN_PATH
from contracts.common import DELIVERY_QUANTILES
from studies.battery_dispatch import rank_rule_schedule
from studies.battery_forecast import (
    BANK_HOLIDAYS_ENGLAND_AND_WALES,
    IssueType,
    asof_at_issue_time,
    climatology_quantiles,
    day_shuffle_map,
    neighbour_ids,
    neighbour_statistics,
    persistence_source_times,
    with_issue_time,
)
from studies.battery_market import (
    B1610_DIR,
    B1610_SUFFIX,
    FOLD_MONTHS,
    HALF_HOURS_PER_DAY,
    WINDOW_START,
    battery_output,
    market_frame,
    p99_output_mw,
)
from studies.nged_battery_a import battery_a_raw, window_filter
from studies.sources import EMBEDDED_BATTERY_FORECAST_INPUTS_DIR as INPUTS_DIR
from studies.sources import MARKET_DOWNLOADS_DIR

WINDOW_DAYS: Final[int] = 365
WINDOW_HALF_HOURS: Final[int] = WINDOW_DAYS * HALF_HOURS_PER_DAY
MIN_SHARE: Final[float] = 0.95
"""A battery is in the testbed only if B1610 and the Physical Notification file each hold at least
this share of the window's half-hours."""
SCORING_START: Final[datetime] = datetime(2025, 10, 1, tzinfo=UTC)
"""No arm is scored before this date, by which at least 28 days of history exist."""
SHUFFLE_SEED: Final[int] = 20251001
"""The seed of the day shuffle that makes the `shuffled` price and the shuffled neighbour slots."""
RANK_RULE_DURATION_HALF_HOURS: Final[int] = 4
"""The duration of the unscaled rank-rule schedule in `rank_rule_value`. The fitted duration and
scale replace it when the forecast is fitted."""
NGED_BATTERY_A: Final[str] = "NGED battery A"
"""The only name under which the private series appears."""

PriceSourceType = Literal["actual", "naive", "model", "shuffled"]
"""Where a price column comes from: the N2EX price, the price seven days earlier, a model's
forecast (added by the price-model script), or the price of another day of the same month and day
type."""

OwnSlotType = Literal["filler", "own_fpn"]
NeighbourSlotType = Literal[
    "filler",
    "different_party",
    "different_party_shuffled",
    "same_party",
    "all_testbed",
    "all_testbed_shuffled",
    "without_largest_party",
    "without_largest_party_shuffled",
]
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "tod",
    "day_of_week",
    "price",
    "price_rank",
    "price_day_mean",
    "price_day_range",
    "rank_rule_value",
    "persistence_mw",
    "climatology_median_mw",
    "own_fpn_slot",
    "neighbour_mean_slot",
    "neighbour_previous_slot",
    "neighbour_share_slot",
)
"""The one fixed column tuple every arm of the XGBoost quantile model is given."""


@dataclass(frozen=True)
class ArmSpec:
    """What one arm of the forecast sees.

    Attributes:
        price_source: Which price fills the price columns.
        own_slot: `own_fpn` puts the battery's own Physical Notification in its slot; `filler`
            puts the same slot's value from seven days earlier there.
        neighbour_slot: Which neighbour statistics fill the three neighbour slots, or `filler`.
    """

    price_source: PriceSourceType
    own_slot: OwnSlotType
    neighbour_slot: NeighbourSlotType


DAY_AHEAD_ARMS: Final[dict[str, ArmSpec]] = {
    f"price_{source}": ArmSpec(price_source=source, own_slot="filler", neighbour_slot="filler")
    for source in ("actual", "naive", "model", "shuffled")
}
"""The arms at `DA-early` and `DA-late`: one per price source, with every Physical Notification
slot at its filler."""

GATE_CLOSURE_ARMS: Final[dict[str, ArmSpec]] = {
    "no_neighbour": ArmSpec(price_source="actual", own_slot="filler", neighbour_slot="filler"),
    "own_fpn": ArmSpec(price_source="actual", own_slot="own_fpn", neighbour_slot="filler"),
    "neighbour_fpn": ArmSpec(
        price_source="actual", own_slot="filler", neighbour_slot="different_party"
    ),
    "neighbour_fpn_shuffled": ArmSpec(
        price_source="actual", own_slot="filler", neighbour_slot="different_party_shuffled"
    ),
    "neighbour_fpn_same_party": ArmSpec(
        price_source="actual", own_slot="filler", neighbour_slot="same_party"
    ),
    "fleet_fpn": ArmSpec(price_source="actual", own_slot="filler", neighbour_slot="all_testbed"),
    "fleet_fpn_shuffled": ArmSpec(
        price_source="actual", own_slot="filler", neighbour_slot="all_testbed_shuffled"
    ),
    "neighbour_fpn_without_largest_party": ArmSpec(
        price_source="actual", own_slot="filler", neighbour_slot="without_largest_party"
    ),
    "neighbour_fpn_without_largest_party_shuffled": ArmSpec(
        price_source="actual", own_slot="filler", neighbour_slot="without_largest_party_shuffled"
    ),
}
"""The arms at `ID-1h`. `fleet_fpn` and its shuffled twin are rung B2's arms for NGED battery A.
The two `without_largest_party` arms leave the testbed's largest lead party out of the neighbour
set."""

TESTBED_GATE_CLOSURE_ARMS: Final[tuple[str, ...]] = (
    "no_neighbour",
    "own_fpn",
    "neighbour_fpn",
    "neighbour_fpn_shuffled",
)
"""The arms of rung A5 and its own-FPN denominator, which fix a testbed battery's scored
half-hours at gate closure. The same-party arm is exploratory and applies only to a battery with
another unit of its lead party, so it never decides which half-hours are scored."""

NGED_BATTERY_A_GATE_CLOSURE_ARMS: Final[tuple[str, ...]] = (
    "no_neighbour",
    "fleet_fpn",
    "fleet_fpn_shuffled",
)
"""The arms of rung B2, the only ones NGED battery A can run at gate closure."""


@dataclass(frozen=True)
class Battery:
    """One battery's series and the facts that decide its neighbours.

    Attributes:
        battery_id: The BMU identifier, or `NGED battery A`.
        output: Columns `time` (period start, UTC) and `output_mw`.
        p99_mw: The 99th percentile of the absolute output, in the unit of `output_mw`.
        lead_party: The BMU's lead party, or None for NGED battery A.
        has_fpn: Whether the battery submits Physical Notifications.
    """

    battery_id: str
    output: pl.DataFrame
    p99_mw: float
    lead_party: str | None
    has_fpn: bool


def load_nged_battery_a() -> Battery:
    """Load NGED battery A as a fraction of its own 99th-percentile absolute output.

    Returns:
        A `Battery` whose `output_mw` is that fraction (so its scale is 1), with no Physical
        Notification and no lead party. Nothing about the site is printed or kept.
    """
    raw = battery_a_raw().filter(window_filter())
    scale = float(np.quantile(raw["power"].abs().to_numpy(), 0.99))
    output = raw.select("time", output_mw=pl.col("power") / scale).with_columns(
        pl.col("time").dt.cast_time_unit("us")
    )
    return Battery(
        battery_id=NGED_BATTERY_A, output=output, p99_mw=1.0, lead_party=None, has_fpn=False
    )


def scored_arms(*, battery: Battery, issue: IssueType, has_model_price: bool) -> dict[str, ArmSpec]:
    """Return the arms whose inputs decide which half-hours of a setting are scored.

    Args:
        battery: The target battery.
        issue: The issue type.
        has_model_price: Whether the frame carries the price model's `price_model` column. Only
            `DA-early` runs the `model` price, so the flag changes nothing at the other issue times.

    Returns:
        The day-ahead arms at `DA-early` and `DA-late` (without `price_model` until the price
        model has run), and the gate-closure arms that the battery can run at `ID-1h`.
    """
    if issue != "ID-1h":
        return {
            name: arm
            for name, arm in DAY_AHEAD_ARMS.items()
            if arm.price_source != "model" or (issue == "DA-early" and has_model_price)
        }
    names = TESTBED_GATE_CLOSURE_ARMS if battery.has_fpn else NGED_BATTERY_A_GATE_CLOSURE_ARMS
    return {name: GATE_CLOSURE_ARMS[name] for name in names}


def half_hour_grid() -> pl.DataFrame:
    """Return the window's half-hour starts, UTC, as a column `time`."""
    return pl.DataFrame(
        {
            "time": pl.datetime_range(
                WINDOW_START,
                WINDOW_START + timedelta(days=WINDOW_DAYS) - timedelta(minutes=30),
                interval="30m",
                time_unit="us",
                eager=True,
            )
        }
    )


def pn_share(*, pn: pl.DataFrame) -> pl.DataFrame:
    """Return each BMU's share of the window's half-hours that the Physical Notification covers."""
    end = WINDOW_START + timedelta(days=WINDOW_DAYS)
    return (
        pn.filter((pl.col("time") >= WINDOW_START) & (pl.col("time") < end))
        .group_by("bmu_id")
        .agg(pn_share=pl.col("fpn_mw").drop_nulls().len() / WINDOW_HALF_HOURS)
        .rename({"bmu_id": "elexon_bmu_id"})
    )


def load_physical_notifications() -> pl.DataFrame:
    """Return the final Physical Notification of every BMU as columns `bmu_id`, `time`, `fpn_mw`."""
    return pl.read_parquet(PN_PATH).select(
        "bmu_id",
        pl.col("time").dt.cast_time_unit("us"),
        fpn_mw=pl.col("mean_level_mw").cast(pl.Float64),
    )


def testbed(*, pn: pl.DataFrame) -> pl.DataFrame:
    """Return the embedded battery BMUs with enough public data.

    Args:
        pn: The frame `load_physical_notifications` returns.

    Returns:
        One row per testbed BMU: `elexon_bmu_id`, `bmu_name`, `lead_party_name`, `gsp_group_name`,
        `generation_capacity_mw`, `b1610_share`, and `pn_share`. A BMU qualifies when its type is E
        and both shares reach `MIN_SHARE`.
    """
    listed = pl.read_csv(BMU_LIST_PATH).filter(pl.col("bmu_type") == "E")
    b1610 = pl.DataFrame(
        {
            "elexon_bmu_id": listed["elexon_bmu_id"],
            "b1610_share": [
                _b1610_rows(bmu_id=bmu_id) / WINDOW_HALF_HOURS
                for bmu_id in listed["elexon_bmu_id"].to_list()
            ],
        }
    )
    return (
        listed.select(
            "elexon_bmu_id",
            "bmu_name",
            "lead_party_name",
            "gsp_group_name",
            "generation_capacity_mw",
        )
        .join(b1610, on="elexon_bmu_id")
        .join(pn_share(pn=pn), on="elexon_bmu_id", how="left")
        .with_columns(pn_share=pl.col("pn_share").fill_null(0.0))
        .filter((pl.col("b1610_share") >= MIN_SHARE) & (pl.col("pn_share") >= MIN_SHARE))
        .sort("elexon_bmu_id")
    )


def _b1610_rows(*, bmu_id: str) -> int:
    """Return the number of B1610 rows a BMU has, or 0 if it has no file."""
    path = B1610_DIR / f"{bmu_id}{B1610_SUFFIX}"
    return pl.read_parquet(path).height if path.exists() else 0


def load_testbed_batteries(*, members: pl.DataFrame) -> dict[str, Battery]:
    """Load the output of every testbed BMU.

    Args:
        members: The frame `testbed` returns.

    Returns:
        A dictionary from BMU identifier to `Battery`.
    """
    batteries = {}
    for row in members.iter_rows(named=True):
        output = battery_output(bmu_id=row["elexon_bmu_id"]).select("time", "output_mw")
        batteries[row["elexon_bmu_id"]] = Battery(
            battery_id=row["elexon_bmu_id"],
            output=output,
            p99_mw=p99_output_mw(frame=output),
            lead_party=row["lead_party_name"],
            has_fpn=True,
        )
    return batteries


def price_sources(*, grid: pl.DataFrame) -> pl.DataFrame:
    """Return the day-ahead price and its `naive` and `shuffled` versions on the half-hour grid.

    Args:
        grid: The frame `half_hour_grid` returns.

    Returns:
        Columns `time`, `price_actual` (the N2EX price), `price_naive` (the price seven days
        earlier), and `price_shuffled` (the price of the day `day_shuffle_map` draws for this day,
        at the same half-hour of day), all in GBP per MWh.
    """
    actual = market_frame().select("time", price_actual=pl.col("day_ahead_gbp_per_mwh"))
    shuffled = (
        shuffled_time(grid=grid)
        .join(
            actual.rename({"time": "source_time", "price_actual": "price_shuffled"}),
            on="source_time",
            how="left",
        )
        .select("time", "price_shuffled")
    )
    naive = actual.select(
        time=pl.col("time") + pl.duration(days=7), price_naive=pl.col("price_actual")
    )
    return (
        grid.join(actual, on="time", how="left")
        .join(naive, on="time", how="left")
        .join(shuffled, on="time", how="left")
    )


def price_columns(*, frame: pl.DataFrame, source: str) -> pl.DataFrame:
    """Add a price source's within-day rank, day mean, day range, and rank-rule value.

    Args:
        frame: A half-hour frame with `time` and `price_<source>`.
        source: The price source name.

    Returns:
        The frame with `price_rank_<source>` (the half-hour's rank among its UTC day's 48, ties
        averaged, scaled to the open interval 0 to 1), `price_day_mean_<source>`,
        `price_day_range_<source>`, and `rank_rule_<source>` (the unscaled rank-rule schedule of
        `RANK_RULE_DURATION_HALF_HOURS`; a day with a missing price is zero).
    """
    price = f"price_{source}"
    with_day = frame.with_columns(_day=pl.col("time").dt.date())
    ranked = with_day.with_columns(
        **{
            f"price_rank_{source}": (pl.col(price).rank("average").over("_day") - 0.5)
            / HALF_HOURS_PER_DAY,
            f"price_day_mean_{source}": pl.col(price).mean().over("_day"),
            f"price_day_range_{source}": pl.col(price).max().over("_day")
            - pl.col(price).min().over("_day"),
        }
    ).drop("_day")
    schedule = rank_rule_schedule(
        prices=ranked[price].to_numpy(), duration_half_hours=RANK_RULE_DURATION_HALF_HOURS
    )
    return ranked.with_columns(**{f"rank_rule_{source}": pl.Series(schedule, dtype=pl.Float64)})


def wind_forecast_at_issue(*, issue: IssueType, grid: pl.DataFrame) -> pl.DataFrame:
    """Return the NESO wind forecast that was published by each half-hour's issue time.

    Args:
        issue: The issue type.
        grid: The frame `half_hour_grid` returns.

    Returns:
        Columns `time` and `wind_forecast_mw`: the newest forecast of that hour published at or
        before the issue time. The forecast is hourly, so a half-hour takes its hour's value.
    """
    vintages = pl.read_parquet(MARKET_DOWNLOADS_DIR / "elexon_windfor" / "elexon_windfor.parquet")
    targets = with_issue_time(frame=grid, issue=issue).with_columns(
        hour=pl.col("time").dt.truncate("1h")
    )
    hourly = targets.select(time=pl.col("hour"), issue_time=pl.col("issue_time")).unique()
    attached = asof_at_issue_time(
        targets=hourly,
        vintages=vintages.select("time", "publish_time", wind_forecast_mw=pl.col("forecast_mw")),
        value_columns=["wind_forecast_mw"],
    ).select("time", "issue_time", "wind_forecast_mw")
    return targets.join(
        attached, left_on=["hour", "issue_time"], right_on=["time", "issue_time"], how="left"
    ).select("time", "wind_forecast_mw")


def shuffled_time(*, grid: pl.DataFrame) -> pl.DataFrame:
    """Return, for each half-hour, the time of the same half-hour of day on its drawn day."""
    dates = grid["time"].dt.date().unique().to_list()
    draw = day_shuffle_map(
        days=dates, non_working_dates=BANK_HOLIDAYS_ENGLAND_AND_WALES, seed=SHUFFLE_SEED
    )
    return (
        grid.with_columns(date=pl.col("time").dt.date())
        .join(draw, on="date")
        .select(
            "time",
            source_time=pl.col("time")
            - pl.duration(days=(pl.col("date") - pl.col("source_date")).dt.total_days()),
        )
    )


def slot_columns(*, stats: pl.DataFrame, prefix: str, shuffle: pl.DataFrame) -> pl.DataFrame:
    """Return a neighbour set's three slots at `t`, at the shuffled time, and at `t` less a week."""
    slots = {
        "mean": "neighbour_mean_fraction",
        "previous": "neighbour_mean_fraction_previous",
        "share": "neighbour_discharging_share",
    }
    at_time = stats.select("time", **{f"{prefix}__{k}": v for k, v in slots.items()})
    week_ago = stats.select(
        time=pl.col("time") + pl.duration(days=7),
        **{f"{prefix}__{k}_filler": v for k, v in slots.items()},
    )
    drawn = shuffle.join(
        stats.rename({"time": "source_time"}),
        on="source_time",
        how="left",
    ).select(
        "time",
        **{f"{prefix}_shuffled__{k}": v for k, v in slots.items()},
    )
    return at_time.join(week_ago, on="time", how="full", coalesce=True).join(
        drawn, on="time", how="full", coalesce=True
    )


def build_issue_frame(
    *,
    battery: Battery,
    issue: IssueType,
    grid: pl.DataFrame,
    prices: pl.DataFrame,
    pn: pl.DataFrame,
    batteries: dict[str, Battery],
    shuffle: pl.DataFrame,
) -> pl.DataFrame:
    """Return one battery's wide input frame at one issue time.

    Args:
        battery: The target battery.
        issue: The issue type.
        grid: The frame `half_hour_grid` returns.
        prices: The frame `price_sources` returns, with the `model` price joined if it exists.
        pn: The frame `load_physical_notifications` returns.
        batteries: Every testbed battery, which supplies the neighbours.
        shuffle: The frame `shuffled_time` returns.

    Returns:
        One row per half-hour of the window. Columns: `time`, `issue_time`, `tod`, `day_of_week`,
        `month`, `fold`, `output_mw`, the persistence value `persistence_mw`, the climatology
        quantiles `q<level>` and `climatology_n`, `climatology_median_mw`, `climatology_filler_mw`,
        the price columns of every source in `prices`, `own_fpn_mw`, `own_fpn_filler_mw`, and the
        neighbour columns `<set>__mean`, `<set>__previous`, `<set>__share` with their `_filler`
        versions for each neighbour set that the battery has (`all_testbed`, plus `different_party`
        and `same_party` for a battery with a lead party).
    """
    frame = with_issue_time(frame=grid, issue=issue).with_columns(
        tod=pl.col("time").dt.hour().cast(pl.Int64) * 2 + pl.col("time").dt.minute() // 30,
        day_of_week=pl.col("time").dt.weekday().cast(pl.Int64),
        month=pl.col("time").dt.strftime("%Y-%m"),
        fold=pl.col("time").dt.month().replace_strict(FOLD_MONTHS, return_dtype=pl.Int64),
    )
    frame = frame.join(battery.output, on="time", how="left")
    sources = [c.removeprefix("price_") for c in prices.columns if c.startswith("price_")]
    frame = frame.join(prices, on="time", how="left")
    for source in sources:
        frame = price_columns(frame=frame, source=source)
    # Persistence: the output at the source half-hour that the issue time allows.
    source_times = persistence_source_times(target_times=frame["time"].to_list(), issue=issue)
    persistence = (
        frame.select("time")
        .with_columns(source_time=pl.Series(source_times, dtype=frame.schema["time"]))
        .join(
            battery.output.rename({"time": "source_time", "output_mw": "persistence_mw"}),
            on="source_time",
            how="left",
        )
        .select("time", "persistence_mw")
    )
    frame = frame.join(persistence, on="time", how="left")
    clim = climatology_quantiles(
        history=battery.output,
        targets=frame.select("time", "issue_time"),
        levels=DELIVERY_QUANTILES,
        non_working_dates=BANK_HOLIDAYS_ENGLAND_AND_WALES,
    ).drop("issue_time")
    frame = frame.join(clim, on="time", how="left").with_columns(
        climatology_median_mw=pl.col("q0.5")
    )
    frame = frame.join(
        frame.select(
            time=pl.col("time") + pl.duration(days=7),
            climatology_filler_mw=pl.col("q0.5"),
        ),
        on="time",
        how="left",
    )
    return _with_fpn_columns(
        frame=frame, battery=battery, pn=pn, batteries=batteries, shuffle=shuffle
    )


def _with_fpn_columns(
    *,
    frame: pl.DataFrame,
    battery: Battery,
    pn: pl.DataFrame,
    batteries: dict[str, Battery],
    shuffle: pl.DataFrame,
) -> pl.DataFrame:
    """Add the battery's own Physical Notification, if it has one, and its neighbour statistics."""
    if battery.has_fpn:
        own = pn.filter(pl.col("bmu_id") == battery.battery_id).select(
            "time", own_fpn_mw=pl.col("fpn_mw")
        )
        frame = frame.join(own, on="time", how="left").join(
            own.select(
                time=pl.col("time") + pl.duration(days=7),
                own_fpn_filler_mw=pl.col("own_fpn_mw"),
            ),
            on="time",
            how="left",
        )
    p99 = {b.battery_id: b.p99_mw for b in batteries.values()}
    sets = {"all_testbed": sorted(b for b in batteries if b != battery.battery_id)}
    if battery.lead_party is not None:
        lead_party = {b.battery_id: str(b.lead_party) for b in batteries.values()}
        sets["different_party"] = neighbour_ids(target=battery.battery_id, lead_party=lead_party)
        sets["same_party"] = neighbour_ids(
            target=battery.battery_id, lead_party=lead_party, rule="same_party"
        )
    for name, ids in sets.items():
        stats = neighbour_statistics(fpn=pn, neighbours=ids, p99_mw=p99)
        frame = frame.join(
            slot_columns(stats=stats, prefix=name, shuffle=shuffle), on="time", how="left"
        )
    return frame


def arm_expressions(*, columns: Iterable[str], arm: ArmSpec) -> dict[str, pl.Expr]:
    """Return the expression that builds each of one arm's `FEATURE_COLUMNS`.

    Args:
        columns: The columns of a frame `build_issue_frame` returns.
        arm: What the arm sees. A slot with no source (NGED battery A has no Physical Notification
            and no lead party) holds the climatology median from seven days earlier.

    Returns:
        A dictionary from each name in `FEATURE_COLUMNS`, in order, to its expression.
    """
    available = set(columns)
    source = arm.price_source
    own = (
        pl.col("own_fpn_mw")
        if arm.own_slot == "own_fpn"
        else pl.col(
            "own_fpn_filler_mw" if "own_fpn_filler_mw" in available else "climatology_filler_mw"
        )
    )
    slots = ("mean", "previous", "share")
    if arm.neighbour_slot == "filler":
        with_party = "different_party__mean_filler" in available
        neighbour = {
            slot: pl.col(
                f"different_party__{slot}_filler" if with_party else "climatology_filler_mw"
            )
            for slot in slots
        }
    else:
        neighbour = {slot: pl.col(f"{arm.neighbour_slot}__{slot}") for slot in slots}
    return {
        "tod": pl.col("tod"),
        "day_of_week": pl.col("day_of_week"),
        "price": pl.col(f"price_{source}"),
        "price_rank": pl.col(f"price_rank_{source}"),
        "price_day_mean": pl.col(f"price_day_mean_{source}"),
        "price_day_range": pl.col(f"price_day_range_{source}"),
        "rank_rule_value": pl.col(f"rank_rule_{source}"),
        "persistence_mw": pl.col("persistence_mw"),
        "climatology_median_mw": pl.col("climatology_median_mw"),
        "own_fpn_slot": own,
        "neighbour_mean_slot": neighbour["mean"],
        "neighbour_previous_slot": neighbour["previous"],
        "neighbour_share_slot": neighbour["share"],
    }


def arm_frame(*, base: pl.DataFrame, arm: ArmSpec) -> pl.DataFrame:
    """Pick one arm's feature columns from a wide frame.

    Args:
        base: A frame `build_issue_frame` returns.
        arm: What the arm sees.

    Returns:
        `time`, `fold`, `month`, `output_mw`, and the `FEATURE_COLUMNS`, in that order.
    """
    expressions = arm_expressions(columns=base.columns, arm=arm)
    assert tuple(expressions) == FEATURE_COLUMNS
    return base.select("time", "fold", "month", "output_mw", **expressions)


def is_scored(*, base: pl.DataFrame, arms: dict[str, ArmSpec]) -> pl.Series:
    """Return which half-hours every arm of a setting can be scored on.

    A half-hour is scored when it is on or after `SCORING_START`, the output exists, and every
    arm's feature columns are non-null. The decision reads the target and the inputs that the arms
    share (every arm's columns, because the arms are scored together), never one arm's input alone.

    Args:
        base: A frame `build_issue_frame` returns.
        arms: The arms of the setting.

    Returns:
        A Boolean series aligned with `base`.
    """
    keep = (pl.col("time") >= SCORING_START) & pl.col("output_mw").is_not_null()
    mask = base.select(keep.alias("keep"))["keep"]
    for arm in arms.values():
        if arm.price_source == "model" and "price_model" not in base.columns:
            continue
        columns = arm_frame(base=base, arm=arm).select(FEATURE_COLUMNS)
        complete = columns.select(pl.all_horizontal(pl.all().is_not_null())).to_series()
        mask = mask & complete
    return mask


def _arm_column_lines(*, base: pl.DataFrame, arms: dict[str, ArmSpec]) -> list[str]:
    """Return report lines giving each arm's feature columns and where each comes from."""
    lines = []
    for name, arm in arms.items():
        expressions = arm_expressions(columns=base.columns, arm=arm)
        mapping = ", ".join(
            f"{column} <- {', '.join(expression.meta.root_names())}"
            for column, expression in expressions.items()
        )
        lines.append(f"- `{name}` ({len(expressions)} columns): {mapping}.")
    return lines


def main() -> None:
    """Build and save every battery's frames, and write `inputs_report.md`."""
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pn = load_physical_notifications()
    members = testbed(pn=pn)
    batteries = load_testbed_batteries(members=members)
    battery_a = load_nged_battery_a()
    grid = half_hour_grid()
    prices = price_sources(grid=grid)
    shuffle = shuffled_time(grid=grid)
    nged_groups = {"East Midlands", "Midlands", "South Wales", "South Western"}
    lines = [
        "# Inputs of the embedded-battery forecast",
        "",
        "## The testbed",
        "",
        (
            f"- Testbed batteries: {members.height} (the plan expects 35). "
            f"Lead parties: {members['lead_party_name'].n_unique()}. "
            f"In NGED's four grid supply point groups: "
            f"{int(members['gsp_group_name'].is_in(list(nged_groups)).sum())}."
        ),
        (
            f"- Each needs at least {MIN_SHARE:.0%} of the window's {WINDOW_HALF_HOURS} half-hours "
            "in both B1610 and the Physical Notification file."
        ),
        "",
        "| Lead party | Testbed batteries |",
        "|---|---|",
        *[
            f"| {row[0]} | {row[1]} |"
            for row in members.group_by("lead_party_name")
            .len()
            .sort("len", "lead_party_name", descending=[True, False])
            .iter_rows()
        ],
        "",
        "## Each target's neighbours",
        "",
        "| Target | Lead party | Different-party neighbours | Same-party neighbours |",
        "|---|---|---|---|",
    ]
    parties = {b.battery_id: str(b.lead_party) for b in batteries.values()}
    for battery_id, battery in batteries.items():
        different = neighbour_ids(target=battery_id, lead_party=parties)
        same = neighbour_ids(target=battery_id, lead_party=parties, rule="same_party")
        lines.append(
            f"| {battery_id} | {battery.lead_party} | {len(different)}: {', '.join(different)} "
            f"| {len(same)}: {', '.join(same) or 'none'} |"
        )
    lines += ["", "## Scored half-hours by battery and issue time", ""]
    lines += ["| Battery | DA-early | DA-late | ID-1h |", "|---|---|---|---|"]
    column_lines: dict[str, list[str]] = {}
    for battery in [*batteries.values(), battery_a]:
        counts = []
        for issue in ("DA-early", "DA-late", "ID-1h"):
            base = build_issue_frame(
                battery=battery,
                issue=issue,
                grid=grid,
                prices=prices,
                pn=pn,
                batteries=batteries,
                shuffle=shuffle,
            )
            arms = scored_arms(battery=battery, issue=issue, has_model_price=False)
            scored = is_scored(base=base, arms=arms)
            counts.append(int(scored.sum()))
            assert base["time"].is_unique().all()
            file_name = (
                "nged_battery_a" if battery.battery_id == NGED_BATTERY_A else battery.battery_id
            )
            base.write_parquet(INPUTS_DIR / f"{issue}__{file_name}.parquet")
            column_lines.setdefault(
                f"{issue} for {'NGED battery A' if not battery.has_fpn else 'a testbed battery'}",
                _arm_column_lines(base=base, arms=arms),
            )
        lines.append(f"| {battery.battery_id} | {counts[0]} | {counts[1]} | {counts[2]} |")
    lines += ["", "## Each arm's feature columns", ""]
    for heading, arm_lines in column_lines.items():
        lines += [f"### {heading}", "", *arm_lines, ""]
    lines += _checks(grid=grid, prices=prices, shuffle=shuffle)
    (INPUTS_DIR / "inputs_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


def _checks(*, grid: pl.DataFrame, prices: pl.DataFrame, shuffle: pl.DataFrame) -> list[str]:
    """Return report lines for the checks on the grid, the prices, and the day shuffle."""
    drawn = shuffle.with_columns(
        same_day=pl.col("source_time").dt.date() == pl.col("time").dt.date(),
        same_month=pl.col("source_time").dt.month() == pl.col("time").dt.month(),
        same_half_hour=(pl.col("source_time").dt.hour() == pl.col("time").dt.hour())
        & (pl.col("source_time").dt.minute() == pl.col("time").dt.minute()),
    )
    return [
        "## Checks",
        "",
        (
            f"- Half-hour grid: {grid.height} rows (expected {WINDOW_HALF_HOURS}), "
            f"first {grid['time'].min()}, last {grid['time'].max()}."
        ),
        "- Price nulls: "
        + ", ".join(f"{c}: {prices[c].null_count()}" for c in prices.columns if c != "time")
        + ". The naive price is null for the first week by construction.",
        (
            f"- Shuffle: {int(drawn['same_day'].sum())} half-hours drew their own day; "
            f"{int((~drawn['same_month']).sum())} drew another month; "
            f"{int((~drawn['same_half_hour']).sum())} drew another half-hour of day."
        ),
        "",
    ]


if __name__ == "__main__":
    main()
