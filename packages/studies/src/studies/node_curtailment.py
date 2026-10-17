"""The cost of curtailing generators when a bulk or grid supply point would export over its limit.

The question asked of every half-hour is: will the node (a bulk supply point or a grid supply
point) export more than its firm limit, and if so which generators beneath it must be capped, and
what does that cost in pounds? A generator is capped only when the forecast says the node's export
would exceed the node's limit. Two tables hold every fact about the electricity network (a node
table and a generator table), so scoring 2,500 series needs longer tables and no new code.

Units. Flows and limits are in megawatts (MW) and are power. Energy is in megawatt-hours (MWh), and
a half-hour of power `p` MW is `0.5 * p` MWh. Prices are pounds per MWh.

The rule, per forecast row of a node (`r` is the breach ratio, `tau = 1 - 1/r`):

1. Planned node curtailment `v_node = max(0, q_tau(export) - L) * 0.5` MWh, where `L` is the node's
   limit for the season of the valid time and `q_tau(export)` is the tau quantile of the net export.
   Net export is `export_sign * flow`, so for `export_sign` of -1 the quantile of the export is
   minus the `1 - tau` quantile of the flow.
2. The generators under the node whose forecast is given (the curtailable ones) share `v_node` pro
   rata by capacity, each share clamped at the generator's own forecast energy `f_g * 0.5` MWh and
   the excess redistributed to the unclamped generators. Any remainder is unallocated.
3. A generator with a share `v_g > 0` is capped at `cap_g = max(0, f_g - v_g / 0.5)` MW. It loses
   `c_g = max(0, a_g - cap_g) * 0.5` MWh against its metered output `a_g`. A generator with no
   share is not capped.
4. The node's residual breach is `max(0, actual export - sum_g(c_g) / 0.5 - L) * 0.5` MWh.

The headline is the energy proxy for the payment rule of DCUSA Schedule 2D: `price * (sum_g(c_g) +
r * breach)` pounds, with the breach priced at `r` times the price. The DCUSA form is reported
beside it: it charges each generator's instructed reduction of its Maximum Export Capacity,
`capacity_g - cap_g` MW, rather than the energy lost, and pays only for Full Export Curtailment
Hours above the Curtailment Limit. See
<https://openclimatefix.github.io/nged-substation-forecast/roadmap/cost-savings-metrics/> for the
reasoning.

Everything runs as Polars expressions over `(node, valid_time)` with no loop over nodes. The one
loop is the progressive filling in `allocate_pro_rata_by_capacity`, which runs once per round of
clamping over the whole frame, and a round is needed at most once per generator under one node.
"""

import datetime as dt
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import patito as pt
import polars as pl

from studies.tail_calibration import align_arms, level_column

SeasonType = Literal["Winter", "Intermediate Cool", "Intermediate Warm", "Summer"]
"""NGED's four seasons, each a set of whole calendar months."""

NodeKindType = Literal["BSP", "GSP"]
"""A bulk supply point or a grid supply point."""

DirectionType = Literal["export", "import"]
"""Which way the flow crosses a node's limit."""

SEASONS: Final[Mapping[int, SeasonType]] = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Intermediate Cool",
    4: "Intermediate Cool",
    11: "Intermediate Cool",
    5: "Intermediate Warm",
    9: "Intermediate Warm",
    10: "Intermediate Warm",
    6: "Summer",
    7: "Summer",
    8: "Summer",
}
"""NGED's seasons by calendar month number, taken from the valid time in UTC."""

SEASON_NAMES: Final[tuple[SeasonType, ...]] = (
    "Winter",
    "Intermediate Cool",
    "Intermediate Warm",
    "Summer",
)
"""The four season names, in calendar order from winter."""

HALF_HOUR_H: Final[float] = 0.5
"""Hours in one half-hourly row, which turns MW into MWh."""

HOURS_PER_YEAR: Final[float] = 8766.0
"""Hours in a Julian year of 365.25 days."""

PRICE_ILLUSTRATIVE_BALANCING_GBP_PER_MWH: Final[float] = 100.0
"""An illustrative balancing-style constraint price, the lowest of the four prices swept."""

PRICE_LOW_GBP_PER_MWH: Final[float] = 380.0
"""The low end of the GBP 380 to 11,100 per MWh range reconstructed from April 2025 contracts."""

PRICE_HEADLINE_GBP_PER_MWH: Final[float] = 2000.0
"""The headline price, near the geometric middle of the reconstructed range."""

PRICE_HIGH_GBP_PER_MWH: Final[float] = 11100.0
"""The high end of the reconstructed range."""

PRICES_GBP_PER_MWH: Final[tuple[float, ...]] = (
    PRICE_ILLUSTRATIVE_BALANCING_GBP_PER_MWH,
    PRICE_LOW_GBP_PER_MWH,
    PRICE_HEADLINE_GBP_PER_MWH,
    PRICE_HIGH_GBP_PER_MWH,
)
"""The price sweep, lowest first."""

SCALED_LIMIT_FRACTIONS: Final[tuple[float, ...]] = (0.9, 0.75, 0.5)
"""The fractions `f` of a node's maximum observed export that define a scaled limit, 0.9 default."""

DEFAULT_SCALED_LIMIT_FRACTION: Final[float] = 0.9
"""The default fraction. A scaled limit of 0.9 is 0.9 times the maximum, never a percentile."""

KEY_COLUMNS: Final[tuple[str, ...]] = ("node_id", "power_fcst_init_time", "valid_time", "seed")
"""The columns that identify one forecast row of one node."""

FLOW_FRAME_COLUMNS: Final[tuple[str, ...]] = ("time_series_id", "valid_time", "flow_mw")
"""The columns of a frame of metered flows for `scaled_limits` and `event_census`."""

REASON_UNKNOWN_SIGN: Final[str] = "unknown_sign"
"""Why a node whose `export_sign` is null is excluded."""

REASON_INCOMPLETE_LIMITS: Final[str] = "incomplete_limits"
"""Why a node without an export limit in each of the four seasons is excluded."""

REASON_NO_FORECAST_ROWS: Final[str] = "no_forecast_rows"
"""Why a node whose time series has no forecast row in the lead window is excluded."""


class NodeRow(pt.Model):
    """One node: a bulk or grid supply point, and the sign that turns its metered flow into export.

    `export_sign` is +1 where a positive metered flow is export, -1 where a positive metered flow is
    import, and null where the sign is not known, which excludes the node.
    """

    node_id: str = pt.Field(unique=True)
    time_series_id: int = pt.Field(unique=True)
    node_type: str
    parent_node_id: str | None
    export_sign: int | None
    sign_source: str


class SeasonLimitRow(pt.Model):
    """The limits of one node in one season, in MW."""

    node_id: str
    season: str
    export_limit_mw: float = pt.Field(gt=0)
    import_limit_mw: float | None = pt.Field(gt=0)
    limit_source: str


class GeneratorRow(pt.Model):
    """One generator beneath a node.

    `generator_id` is the generator's `time_series_id`, which is also the `time_series_id` of its
    rows in the generator forecast frames. A generator with `is_forecast` false has no forecast and
    is never capped, because capping it from its actual output would leak the outcome.
    """

    generator_id: int = pt.Field(unique=True)
    node_id: str
    capacity_mw: float = pt.Field(gt=0)
    connection_date: dt.date | None
    is_forecast: bool


@dataclass(frozen=True)
class NodeTable:
    """The two tables, validated on construction, that hold every fact about the network.

    Attributes:
        nodes: One row per node, with the columns of `NodeRow`.
        limits: One row per `(node_id, season)`, with the columns of `SeasonLimitRow`.
        generators: One row per generator, with the columns of `GeneratorRow`.
    """

    nodes: pl.DataFrame
    limits: pl.DataFrame
    generators: pl.DataFrame

    def __post_init__(self) -> None:
        """Validate the three frames and the references between them.

        Raises:
            ValueError: If a node type, sign, season, or reference is not allowed, or a
                `(node_id, season)` pair repeats.
        """
        NodeRow.validate(self.nodes)
        SeasonLimitRow.validate(self.limits)
        GeneratorRow.validate(self.generators)
        node_ids = set(self.nodes["node_id"])
        checks = {
            "node_type must be BSP or GSP": ~self.nodes["node_type"].is_in(["BSP", "GSP"]),
            "export_sign must be -1, 1, or null": self.nodes["export_sign"].is_in([-1, 1]).not_()
            & self.nodes["export_sign"].is_not_null(),
            "season must be one of NGED's four": ~self.limits["season"].is_in(SEASON_NAMES),
            "parent_node_id must name a node": ~self.nodes["parent_node_id"].is_in(node_ids)
            & self.nodes["parent_node_id"].is_not_null(),
            "limits must name a node": ~self.limits["node_id"].is_in(node_ids),
            "generators must name a node": ~self.generators["node_id"].is_in(node_ids),
        }
        for message, bad in checks.items():
            if bad.any():
                raise ValueError(message)
        if self.limits.select(pl.struct("node_id", "season").is_duplicated().any()).item():
            msg = "a (node_id, season) pair repeats in the limits"
            raise ValueError(msg)


@dataclass(frozen=True)
class NodeCurtailmentAssumptions:
    """The choices behind a score.

    Attributes:
        prices_gbp_per_mwh: The prices swept.
        headline_price_gbp_per_mwh: The price the per-node-and-season table is priced at.
        breach_ratio: `r`. The residual breach is priced at `r` times the price, and the forecast
            quantile is `tau = 1 - 1/r`, which must be a whole percent (20 gives 0.95).
        curtailment_limit_hours: The Curtailment Limit, in hours, deducted from each generator's
            Full Export Curtailment Hours over the scored window before the DCUSA-form payment.
            Rankings use 0.
        lead_min_hours: The shortest forecast lead scored, inclusive.
        lead_max_hours: The longest forecast lead scored, exclusive.
        use_generators: True to cap generators; false for the node-only variant, in which the
            planned node curtailment is assumed delivered exactly.
    """

    prices_gbp_per_mwh: tuple[float, ...] = PRICES_GBP_PER_MWH
    headline_price_gbp_per_mwh: float = PRICE_HEADLINE_GBP_PER_MWH
    breach_ratio: int = 20
    curtailment_limit_hours: float = 0.0
    lead_min_hours: float = 12.0
    lead_max_hours: float = 36.0
    use_generators: bool = True

    @property
    def tau(self) -> float:
        """The forecast quantile level, `1 - 1/breach_ratio`."""
        return 1.0 - 1.0 / self.breach_ratio


@dataclass(frozen=True)
class NodeCurtailmentResult:
    """What `node_curtailment` returns.

    Attributes:
        per_arm: One row per `(arm, price_gbp_per_mwh)`: `arm`, `price_gbp_per_mwh`,
            `is_headline_price`, `n_node_half_hours`, `needed_mwh`, `planned_mwh`,
            `curtailed_mwh`, `breach_mwh`, `energy_proxy_gbp`, `dcusa_form_gbp`, and
            `energy_proxy_gbp_per_mw_year`, the energy proxy per MW of curtailable generator
            capacity per year (null with no curtailable capacity). Energies are means over seeds
            of the total over the scored window. In node-only mode `dcusa_form_gbp` is null.
        per_node_season: One row per `(arm, node_id, season)`: the same energies, and
            `energy_proxy_gbp` at the headline price.
        event_census: One row per `(node_id, season, direction, limit_kind, limit_mw)` with
            `n_half_hours` and `n_exceedances`, counted on the actual flow.
        month_totals: One row per `(arm, seed, month)` holding the additive sums a month
            resampling needs: `n_node_half_hours`, `curtailed_mwh`, `breach_mwh`, `needed_mwh`,
            `energy_proxy_mwh_equivalent`, and `dcusa_gross_mwh_equivalent`. Multiply the last two
            by a price to get pounds.
        excluded: One row per excluded node, with `node_id` and `reason`.
        n_excluded_unknown_sign: How many nodes were excluded for a null `export_sign`.
        n_rows_dropped_missing_generator: Node rows dropped because a curtailable generator
            beneath the node had no forecast row.
        n_rows_dropped_not_in_every_arm: Node rows dropped because another arm lacked them.
    """

    per_arm: pl.DataFrame
    per_node_season: pl.DataFrame
    event_census: pl.DataFrame
    month_totals: pl.DataFrame
    excluded: pl.DataFrame
    n_excluded_unknown_sign: int
    n_rows_dropped_missing_generator: int
    n_rows_dropped_not_in_every_arm: int


def season_expression(*, time_column: str) -> pl.Expr:
    """Return the expression giving NGED's season name for a datetime column.

    Args:
        time_column: The datetime column, whose calendar month in UTC picks the season.

    Returns:
        A string expression with one of `SEASON_NAMES`.
    """
    return pl.col(time_column).dt.month().replace_strict(dict(SEASONS), return_dtype=pl.String)


def allocate_pro_rata_by_capacity(
    *,
    frame: pl.DataFrame,
    group_by: Sequence[str],
    capacity_column: str,
    forecast_mwh_column: str,
    need_mwh_column: str,
) -> pl.DataFrame:
    """Share each group's need among its rows by capacity, never above a row's forecast energy.

    This is progressive filling. In each round every unclamped row gets the group's remaining need
    in proportion to capacity; the rows whose share exceeds their forecast energy are clamped at
    that energy; the next round shares what remains among the rest. A clamped row stays clamped,
    because removing it only raises the others' shares. The loop runs once per round over the whole
    frame, so its length is bounded by the largest group and not by the number of groups.

    Args:
        frame: One row per generator per group.
        group_by: The columns that identify a group, such as a node at one valid time.
        capacity_column: The generator capacity in MW, above zero.
        forecast_mwh_column: The most the row can be asked to give, in MWh, at least zero.
        need_mwh_column: The group's need in MWh, constant within a group.

    Returns:
        `frame` with an added `allocated_mwh` column. The group's allocations sum to its need, or to
        the sum of the forecast energies if that is smaller, and the remainder is unallocated.
    """
    keys = list(group_by)
    work = frame.with_columns(
        _allocated=pl.lit(0.0),
        _active=pl.lit(value=True),
        _remaining=pl.col(need_mwh_column),
    )
    capacity = pl.col(capacity_column)
    forecast = pl.col(forecast_mwh_column)
    while True:
        work = work.with_columns(
            _share=pl.when(pl.col("_active"))
            .then(
                pl.col("_remaining")
                * capacity
                / capacity.filter(pl.col("_active")).sum().over(keys)
            )
            .otherwise(0.0)
        )
        clamped = pl.col("_active") & (forecast <= pl.col("_share"))
        if not work.select(clamped.any()).item():
            return work.with_columns(allocated_mwh=pl.col("_allocated") + pl.col("_share")).drop(
                "_allocated", "_active", "_remaining", "_share"
            )
        work = work.with_columns(
            _allocated=pl.when(clamped).then(forecast).otherwise(pl.col("_allocated")),
            _active=pl.col("_active") & ~clamped,
        ).with_columns(_remaining=pl.col(need_mwh_column) - pl.col("_allocated").sum().over(keys))


def scaled_limits(
    *, table: NodeTable, flows: pl.DataFrame, cutoff: dt.datetime, fraction: float
) -> pl.DataFrame:
    """Return each node's scaled limit: `fraction` times its maximum export before `cutoff`.

    The rule needs no external limit, so it applies to every node. "p90" means 0.9 times the
    maximum, never the 90th percentile of the node's flow, because a percentile of a node's own flow
    is exceeded on 10% of its half-hours by construction. A node whose flow is positive for import
    has its export in the negative tail, and its import limit is the same rule on the other tail.
    Data on or after `cutoff` never changes a limit.

    Args:
        table: The node table, for each node's time series and `export_sign`.
        flows: Metered flows with the columns of `FLOW_FRAME_COLUMNS`, in MW as metered.
        cutoff: A timezone-aware datetime. Only flows with `valid_time` strictly before it count.
        fraction: `f`, above 0 and at most 1.

    Returns:
        One row per node with a known sign and history before `cutoff`: `node_id`,
        `export_limit_mw` and `import_limit_mw`. A limit is null where the node's maximum export
        (or import) before `cutoff` is not above zero.

    Raises:
        ValueError: If `fraction` is not above 0 and at most 1.
    """
    if not 0.0 < fraction <= 1.0:
        msg = f"fraction must be above 0 and at most 1: {fraction}"
        raise ValueError(msg)
    history = (
        flows.filter(pl.col("valid_time") < cutoff)
        .join(
            table.nodes.filter(pl.col("export_sign").is_not_null()).select(
                "node_id", "time_series_id", "export_sign"
            ),
            on="time_series_id",
            how="inner",
        )
        .with_columns(export_mw=pl.col("export_sign") * pl.col("flow_mw"))
        .group_by("node_id")
        .agg(
            max_export_mw=pl.col("export_mw").max(),
            max_import_mw=(-pl.col("export_mw")).max(),
        )
        .sort("node_id")
    )

    def limit(column: str) -> pl.Expr:
        return pl.when(pl.col(column) > 0).then(fraction * pl.col(column))

    return history.select(
        "node_id",
        export_limit_mw=limit("max_export_mw"),
        import_limit_mw=limit("max_import_mw"),
    )


def table_with_scaled_limits(
    *, table: NodeTable, scaled: pl.DataFrame, fraction: float
) -> NodeTable:
    """Return a node table whose limits, in all four seasons, are the scaled limits.

    Args:
        table: The node table whose nodes and generators are kept.
        scaled: The output of `scaled_limits`.
        fraction: The fraction that made `scaled`, recorded in `limit_source`.

    Returns:
        A table in which a node with a null scaled export limit has no limits, so
        `node_curtailment` excludes it as `REASON_INCOMPLETE_LIMITS`.
    """
    limits = (
        scaled.filter(pl.col("export_limit_mw").is_not_null())
        .join(pl.DataFrame({"season": list(SEASON_NAMES)}), how="cross")
        .select(
            "node_id",
            "season",
            "export_limit_mw",
            "import_limit_mw",
            limit_source=pl.lit(f"scaled f={fraction}"),
        )
    )
    return NodeTable(nodes=table.nodes, limits=limits, generators=table.generators)


def scale_to_megawatts(*, frame: pl.DataFrame, capacity_mw: Mapping[int, float]) -> pl.DataFrame:
    """Convert a generator forecast frame from fractions of capacity to MW.

    The stored sets of a generator are fractions of the series' effective capacity, an estimate, so
    its error carries into every cap.

    Args:
        frame: A forecast frame with `time_series_id`, `actual`, and `q<percent>` columns.
        capacity_mw: Each series' effective capacity in MW.

    Returns:
        `frame` with `actual` and every `q<percent>` column multiplied by the series' capacity.
        A series missing from `capacity_mw` gets null values.
    """
    capacities = pl.DataFrame(
        {"time_series_id": list(capacity_mw), "_capacity_mw": list(capacity_mw.values())},
        schema={"time_series_id": frame.schema["time_series_id"], "_capacity_mw": pl.Float64},
    )
    scaled_columns = [c for c in frame.columns if c == "actual" or _is_quantile_column(c)]
    return (
        frame.join(capacities, on="time_series_id", how="left", maintain_order="left")
        .with_columns(*(pl.col(c) * pl.col("_capacity_mw") for c in scaled_columns))
        .drop("_capacity_mw")
    )


def _is_quantile_column(name: str) -> bool:
    return len(name) == 3 and name[0] == "q" and name[1:].isdigit()


def _usable_nodes(
    *, table: NodeTable, forecast_series: Collection[int]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the nodes into those that can be scored and those excluded, with a reason each."""
    complete = (
        table.limits.group_by("node_id")
        .agg(n_seasons=pl.col("season").n_unique())
        .filter(pl.col("n_seasons") == len(SEASON_NAMES))["node_id"]
    )
    reason = (
        pl.when(pl.col("export_sign").is_null())
        .then(pl.lit(REASON_UNKNOWN_SIGN))
        .when(~pl.col("node_id").is_in(complete))
        .then(pl.lit(REASON_INCOMPLETE_LIMITS))
        .when(~pl.col("time_series_id").is_in(list(forecast_series)))
        .then(pl.lit(REASON_NO_FORECAST_ROWS))
    )
    tagged = table.nodes.with_columns(reason=reason)
    excluded = tagged.filter(pl.col("reason").is_not_null()).select("node_id", "reason")
    usable = tagged.filter(pl.col("reason").is_null()).select(
        "node_id", "time_series_id", "export_sign"
    )
    return usable, excluded.sort("node_id")


def _node_rows(
    *,
    frame: pl.DataFrame,
    arm: str,
    usable: pl.DataFrame,
    limits: pl.DataFrame,
    assumptions: NodeCurtailmentAssumptions,
) -> pl.DataFrame:
    """Compute, per node row, the limit, the actual export, the need and the planned curtailment."""
    low = level_column(level=round(1.0 - assumptions.tau, 2))
    high = level_column(level=round(assumptions.tau, 2))
    for column in (low, high):
        if column not in frame.columns:
            msg = f"arm {arm!r} needs the column {column} for tau {assumptions.tau}"
            raise ValueError(msg)
    lead = pl.col("valid_time") - pl.col("power_fcst_init_time")
    return (
        frame.join(usable, on="time_series_id", how="inner")
        .filter(
            lead >= pl.duration(hours=assumptions.lead_min_hours),
            lead < pl.duration(hours=assumptions.lead_max_hours),
        )
        .with_columns(season=season_expression(time_column="valid_time"))
        .join(limits, on=["node_id", "season"], how="inner")
        .with_columns(
            arm=pl.lit(arm),
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
            export_mw=pl.col("export_sign") * pl.col("actual"),
            q_export_mw=pl.when(pl.col("export_sign") == 1)
            .then(pl.col(high))
            .otherwise(-pl.col(low)),
        )
        .with_columns(
            planned_mwh=(pl.col("q_export_mw") - pl.col("limit_mw")).clip(lower_bound=0.0)
            * HALF_HOUR_H,
            needed_mwh=(pl.col("export_mw") - pl.col("limit_mw")).clip(lower_bound=0.0)
            * HALF_HOUR_H,
        )
        .filter(pl.col("planned_mwh").is_finite() & pl.col("export_mw").is_finite())
        .select(
            "arm",
            *KEY_COLUMNS,
            "season",
            "month",
            "limit_mw",
            "export_mw",
            "planned_mwh",
            "needed_mwh",
        )
    )


def _generator_pairs(
    *,
    rows: pl.DataFrame,
    generator_frame: pl.DataFrame,
    generators: pl.DataFrame,
) -> pl.DataFrame:
    """Pair every node row with each curtailable generator connected at its valid time.

    A pair whose generator has no forecast row has null `forecast_mw`, which the caller uses to
    drop the node row.
    """
    keys = [c for c in KEY_COLUMNS if c != "node_id"]
    curtailable = generators.filter(pl.col("is_forecast")).select(
        "generator_id", "node_id", "capacity_mw", "connection_date"
    )
    forecasts = generator_frame.select(
        *keys,
        generator_id="time_series_id",
        forecast_mw="q50",
        output_mw="actual",
    ).filter(pl.col("forecast_mw").is_finite() & pl.col("output_mw").is_finite())
    return (
        rows.select("arm", *KEY_COLUMNS, "planned_mwh")
        .join(curtailable, on="node_id", how="inner")
        .filter(
            pl.col("connection_date").is_null()
            | (pl.col("connection_date") <= pl.col("valid_time").dt.date())
        )
        .drop("connection_date")
        .join(forecasts, on=["generator_id", *keys], how="left")
    )


def _cap_generators(*, pairs: pl.DataFrame) -> pl.DataFrame:
    """Allocate each node row's planned curtailment to its generators and cap them."""
    group = ["arm", *KEY_COLUMNS]
    allocated = allocate_pro_rata_by_capacity(
        frame=pairs.with_columns(
            forecast_mwh=pl.col("forecast_mw").clip(lower_bound=0.0) * HALF_HOUR_H
        ),
        group_by=group,
        capacity_column="capacity_mw",
        forecast_mwh_column="forecast_mwh",
        need_mwh_column="planned_mwh",
    )
    instructed = pl.col("allocated_mwh") > 0
    cap_mw = (pl.col("forecast_mw") - pl.col("allocated_mwh") / HALF_HOUR_H).clip(lower_bound=0.0)
    return allocated.with_columns(
        cap_mw=cap_mw,
        lost_mwh=pl.when(instructed)
        .then((pl.col("output_mw") - cap_mw).clip(lower_bound=0.0) * HALF_HOUR_H)
        .otherwise(0.0),
        instruction_mw=pl.when(instructed)
        .then(
            (pl.col("capacity_mw") - cap_mw).clip(
                lower_bound=0.0, upper_bound=pl.col("capacity_mw")
            )
        )
        .otherwise(0.0),
    )


def _per_seed_means(
    *, frame: pl.DataFrame, by: Sequence[str], values: Sequence[str]
) -> pl.DataFrame:
    """Total `values` within each `by` group and seed, then average the totals over seeds."""
    return (
        frame.group_by([*by, "seed"])
        .agg(*(pl.col(v).sum() for v in values), n_node_half_hours=pl.len())
        .group_by(list(by))
        .agg(*(pl.col(v).mean() for v in values), pl.col("n_node_half_hours").mean())
    )


def _dcusa_payment_mwh_equivalent(*, pairs: pl.DataFrame, limit_hours: float) -> pl.DataFrame:
    """Return per arm and seed the DCUSA payment in MWh-equivalent: hours over the limit x MW."""
    return (
        pairs.group_by("arm", "seed", "generator_id", "capacity_mw")
        .agg(fech_hours=(HALF_HOUR_H * pl.col("instruction_mw") / pl.col("capacity_mw")).sum())
        .with_columns(
            payment=(pl.col("fech_hours") - limit_hours).clip(lower_bound=0.0)
            * pl.col("capacity_mw")
        )
        .group_by("arm", "seed")
        .agg(payment_mwh_equivalent=pl.col("payment").sum())
    )


def _export_census(
    *,
    table: NodeTable,
    exports: pl.DataFrame,
    scaled: Mapping[float, pl.DataFrame] | None,
) -> pl.DataFrame:
    """Count the half-hours of `exports` beyond the real limit and each scaled limit.

    `exports` holds `node_id`, `valid_time`, `export_mw`, and one row per node and half-hour.
    """
    real = table.limits.select("node_id", "season", "export_limit_mw", "import_limit_mw")
    kinds = [
        ("real", direction, real.select("node_id", "season", limit_mw=column).drop_nulls())
        for direction, column in (("export", "export_limit_mw"), ("import", "import_limit_mw"))
    ]
    seasons = pl.DataFrame({"season": list(SEASON_NAMES)})
    for fraction, frame in (scaled or {}).items():
        for direction, column in (("export", "export_limit_mw"), ("import", "import_limit_mw")):
            limits = (
                frame.select("node_id", limit_mw=column)
                .drop_nulls()
                .join(seasons, how="cross")
                .select("node_id", "season", "limit_mw")
            )
            kinds.append((f"scaled f={fraction}", direction, limits))
    thresholds = pl.concat(
        [
            limits.with_columns(limit_kind=pl.lit(kind), direction=pl.lit(direction))
            for kind, direction, limits in kinds
        ]
    )
    return (
        exports.with_columns(season=season_expression(time_column="valid_time"))
        .join(thresholds, on=["node_id", "season"], how="inner")
        .with_columns(
            flow_mw=pl.when(pl.col("direction") == "export")
            .then(pl.col("export_mw"))
            .otherwise(-pl.col("export_mw"))
        )
        .group_by("node_id", "season", "direction", "limit_kind", "limit_mw")
        .agg(
            n_half_hours=pl.len(),
            n_exceedances=(pl.col("flow_mw") > pl.col("limit_mw")).sum(),
        )
        .sort("node_id", "season", "direction", "limit_kind")
    )


def event_census(
    *,
    table: NodeTable,
    flows: pl.DataFrame,
    scaled: Mapping[float, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Count, per node and season, the half-hours whose flow exceeds each limit.

    Args:
        table: The node table. A node with a null `export_sign` is left out.
        flows: Metered flows with the columns of `FLOW_FRAME_COLUMNS`, in MW as metered, one row per
            series and half-hour.
        scaled: Each fraction mapped to the output of `scaled_limits`, for the scaled-limit rows.

    Returns:
        One row per `(node_id, season, direction, limit_kind, limit_mw)` with `n_half_hours` and
        `n_exceedances`. `direction` is "export" or "import" and `limit_kind` is "real" or
        "scaled f=0.9" (with the fraction used). An exceedance is a flow strictly above the limit in
        that direction.
    """
    exports = (
        flows.join(
            table.nodes.filter(pl.col("export_sign").is_not_null()).select(
                "node_id", "time_series_id", "export_sign"
            ),
            on="time_series_id",
            how="inner",
        )
        .with_columns(export_mw=pl.col("export_sign") * pl.col("flow_mw"))
        .select("node_id", "valid_time", "export_mw")
    )
    return _export_census(table=table, exports=exports, scaled=scaled)


def node_curtailment(
    *,
    table: NodeTable,
    node_forecasts: Mapping[str, pl.DataFrame],
    generator_forecasts: Mapping[str, pl.DataFrame],
    assumptions: NodeCurtailmentAssumptions,
    scaled: Mapping[float, pl.DataFrame] | None = None,
) -> NodeCurtailmentResult:
    """Score each arm's node and generator forecasts by what curtailment would have cost.

    Args:
        table: The node and generator tables. Nodes with a null sign, or without a limit in every
            season, or without forecast rows are excluded and listed in the result.
        node_forecasts: Each arm's forecast of the nodes' flows. Rows hold `time_series_id`,
            `power_fcst_init_time`, `valid_time`, `seed`, `actual` (the metered flow in MW), and a
            `q<percent>` column at `1 - tau` and at `tau` (`q05` and `q95` for a breach ratio of
            20). A perfect forecast sets every quantile to `actual`.
        generator_forecasts: The same arm names mapped to the generators' forecasts, in MW, with the
            same keys, `actual` (metered output in MW) and `q50` (the median forecast in MW). Use
            `scale_to_megawatts` to convert fractions of capacity. Ignored in node-only mode.
        assumptions: Prices, breach ratio, Curtailment Limit, lead window, and mode.
        scaled: Each fraction mapped to the output of `scaled_limits`, for the scaled-limit rows of
            the event census.

    Returns:
        A `NodeCurtailmentResult`. Every arm is scored on exactly the node rows that every arm holds
        and for which every curtailable generator has a forecast.

    Raises:
        ValueError: If the arms of the two mappings differ, an arm lacks a quantile column that tau
            needs, or `breach_ratio` does not give a whole-percent quantile.
    """
    if assumptions.use_generators and set(node_forecasts) != set(generator_forecasts):
        msg = "node_forecasts and generator_forecasts must name the same arms"
        raise ValueError(msg)
    if abs(assumptions.tau * 100 - round(assumptions.tau * 100)) > 1e-9:
        msg = f"breach_ratio {assumptions.breach_ratio} does not give a whole-percent quantile"
        raise ValueError(msg)
    arms = sorted(node_forecasts)
    aligned_nodes = align_arms(arms=node_forecasts)
    aligned_generators = align_arms(arms=generator_forecasts) if assumptions.use_generators else {}
    forecast_series = {
        int(s) for frame in aligned_nodes.values() for s in frame["time_series_id"].unique()
    }
    usable, excluded = _usable_nodes(table=table, forecast_series=forecast_series)
    limits = table.limits.select("node_id", "season", limit_mw="export_limit_mw")

    rows_by_arm = [
        _node_rows(
            frame=aligned_nodes[arm], arm=arm, usable=usable, limits=limits, assumptions=assumptions
        )
        for arm in arms
    ]
    rows = pl.concat(rows_by_arm)
    n_dropped_generator = 0
    if assumptions.use_generators:
        pairs = pl.concat(
            _generator_pairs(
                rows=arm_rows,
                generator_frame=aligned_generators[arm],
                generators=table.generators,
            )
            for arm, arm_rows in zip(arms, rows_by_arm, strict=True)
        )
        group = ["arm", *KEY_COLUMNS]
        incomplete = pairs.filter(pl.col("forecast_mw").is_null()).select(group).unique()
        n_dropped_generator = incomplete.select(KEY_COLUMNS).unique().height
        rows = rows.join(incomplete, on=group, how="anti")
        pairs = _cap_generators(pairs=pairs.join(incomplete, on=group, how="anti"))
        curtailed = pairs.group_by(group).agg(
            curtailed_mwh=pl.col("lost_mwh").sum(),
            dcusa_gross_mwh=(HALF_HOUR_H * pl.col("instruction_mw")).sum(),
        )
        rows = rows.join(curtailed, on=group, how="left").with_columns(
            pl.col("curtailed_mwh", "dcusa_gross_mwh").fill_null(0.0)
        )
    else:
        pairs = None
        rows = rows.with_columns(curtailed_mwh=pl.col("planned_mwh"), dcusa_gross_mwh=pl.lit(None))

    # Every arm is scored on the node rows that every arm holds.
    common = rows.group_by(KEY_COLUMNS).agg(n_arms=pl.len()).filter(pl.col("n_arms") == len(arms))
    n_dropped_arms = rows.select(KEY_COLUMNS).unique().height - common.height
    rows = rows.join(common.select(KEY_COLUMNS), on=list(KEY_COLUMNS), how="inner").with_columns(
        breach_mwh=(
            pl.col("export_mw") - pl.col("curtailed_mwh") / HALF_HOUR_H - pl.col("limit_mw")
        ).clip(lower_bound=0.0)
        * HALF_HOUR_H
    )
    if pairs is not None:
        pairs = pairs.join(common.select(KEY_COLUMNS), on=list(KEY_COLUMNS), how="inner")

    energies = ["needed_mwh", "planned_mwh", "curtailed_mwh", "breach_mwh"]
    return NodeCurtailmentResult(
        per_arm=_per_arm(rows=rows, pairs=pairs, table=table, assumptions=assumptions),
        per_node_season=_per_node_season(rows=rows, energies=energies, assumptions=assumptions),
        event_census=_export_census(
            table=table,
            exports=rows.unique(subset=["node_id", "valid_time"]).select(
                "node_id", "valid_time", "export_mw"
            ),
            scaled=scaled,
        ),
        month_totals=_month_totals(rows=rows, assumptions=assumptions),
        excluded=excluded,
        n_excluded_unknown_sign=excluded.filter(pl.col("reason") == REASON_UNKNOWN_SIGN).height,
        n_rows_dropped_missing_generator=n_dropped_generator,
        n_rows_dropped_not_in_every_arm=n_dropped_arms,
    )


def _per_arm(
    *,
    rows: pl.DataFrame,
    pairs: pl.DataFrame | None,
    table: NodeTable,
    assumptions: NodeCurtailmentAssumptions,
) -> pl.DataFrame:
    """Total the energies per arm, then price them at each price in the sweep."""
    ratio = assumptions.breach_ratio
    energies = ["needed_mwh", "planned_mwh", "curtailed_mwh", "breach_mwh"]
    totals = _per_seed_means(frame=rows, by=["arm"], values=energies)
    if pairs is None:
        payment = totals.select("arm", payment_mwh_equivalent=pl.lit(None, dtype=pl.Float64))
        capacity_mw = 0.0
    else:
        payment = (
            _dcusa_payment_mwh_equivalent(
                pairs=pairs, limit_hours=assumptions.curtailment_limit_hours
            )
            .group_by("arm")
            .agg(pl.col("payment_mwh_equivalent").mean())
        )
        capacity_mw = float(
            table.generators.filter(
                pl.col("is_forecast") & pl.col("node_id").is_in(rows["node_id"].unique())
            )["capacity_mw"].sum()
        )
    years = rows["valid_time"].n_unique() * HALF_HOUR_H / HOURS_PER_YEAR
    prices = pl.DataFrame({"price_gbp_per_mwh": list(assumptions.prices_gbp_per_mwh)})
    return (
        totals.join(payment, on="arm", how="left")
        .join(prices, how="cross")
        .with_columns(
            is_headline_price=pl.col("price_gbp_per_mwh") == assumptions.headline_price_gbp_per_mwh,
            energy_proxy_gbp=pl.col("price_gbp_per_mwh")
            * (pl.col("curtailed_mwh") + ratio * pl.col("breach_mwh")),
            dcusa_form_gbp=pl.col("price_gbp_per_mwh")
            * (pl.col("payment_mwh_equivalent") + ratio * pl.col("breach_mwh")),
        )
        .with_columns(
            energy_proxy_gbp_per_mw_year=(
                pl.col("energy_proxy_gbp") / (capacity_mw * years) if capacity_mw > 0 else None
            )
        )
        .select(
            "arm",
            "price_gbp_per_mwh",
            "is_headline_price",
            "n_node_half_hours",
            *energies,
            "energy_proxy_gbp",
            "dcusa_form_gbp",
            "energy_proxy_gbp_per_mw_year",
        )
        .sort("arm", "price_gbp_per_mwh")
    )


def _per_node_season(
    *, rows: pl.DataFrame, energies: Sequence[str], assumptions: NodeCurtailmentAssumptions
) -> pl.DataFrame:
    """Total the energies per arm, node and season, priced at the headline price."""
    return (
        _per_seed_means(frame=rows, by=["arm", "node_id", "season"], values=energies)
        .with_columns(
            energy_proxy_gbp=assumptions.headline_price_gbp_per_mwh
            * (pl.col("curtailed_mwh") + assumptions.breach_ratio * pl.col("breach_mwh"))
        )
        .sort("arm", "node_id", "season")
    )


def _month_totals(*, rows: pl.DataFrame, assumptions: NodeCurtailmentAssumptions) -> pl.DataFrame:
    """Sum the additive quantities per arm, seed and month, for a month resampling."""
    ratio = assumptions.breach_ratio
    return (
        rows.group_by("arm", "seed", "month")
        .agg(
            n_node_half_hours=pl.len(),
            curtailed_mwh=pl.col("curtailed_mwh").sum(),
            breach_mwh=pl.col("breach_mwh").sum(),
            needed_mwh=pl.col("needed_mwh").sum(),
            dcusa_gross_mwh=pl.col("dcusa_gross_mwh").sum(),
        )
        .with_columns(
            energy_proxy_mwh_equivalent=pl.col("curtailed_mwh") + ratio * pl.col("breach_mwh"),
            dcusa_gross_mwh_equivalent=pl.col("dcusa_gross_mwh") + ratio * pl.col("breach_mwh"),
        )
        .drop("dcusa_gross_mwh")
        .sort("arm", "seed", "month")
    )


def month_total_arrays(
    *, month_totals: pl.DataFrame, arm: str, column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Shape one arm's month totals for `studies.bootstrap.bootstrap_month_totals`.

    Args:
        month_totals: `NodeCurtailmentResult.month_totals`.
        arm: The arm to take.
        column: The additive column to resample, such as `energy_proxy_mwh_equivalent`.

    Returns:
        `month_sums` of shape (n_seeds, n_months) and `month_counts` of shape (n_months,), with
        months in sorted order.

    Raises:
        ValueError: If the seeds do not all hold the same months and row counts, which a month
            resampling that pairs seeds needs.
    """
    arm_rows = month_totals.filter(pl.col("arm") == arm).sort("seed", "month")
    seeds = arm_rows["seed"].unique(maintain_order=True)
    sums = np.stack(
        [arm_rows.filter(pl.col("seed") == s)[column].to_numpy() for s in seeds]
    ).astype(np.float64)
    counts = np.stack(
        [arm_rows.filter(pl.col("seed") == s)["n_node_half_hours"].to_numpy() for s in seeds]
    ).astype(np.float64)
    if not (counts == counts[0]).all():
        msg = "every seed must hold the same months and row counts"
        raise ValueError(msg)
    return sums, counts[0]


def node_table_pilot(
    *,
    metadata: pl.DataFrame,
    substations: pl.DataFrame,
    transformer_export_limits_mw: Mapping[str, Sequence[float]],
    signs: Mapping[str, tuple[int | None, str]],
    capacity_mw: Mapping[int, float],
    forecast_ids: Collection[int],
    transformer_import_limits_mw: Mapping[str, Sequence[float]] | None = None,
) -> NodeTable:
    """Build the pilot node table and generator table from NGED's private files.

    The nodes are the metadata rows whose `substation_type` is BSP or GSP. A bulk supply point's
    parent is the grid supply point that `substations` names for it, and a grid supply point has no
    parent. The generators are the EHV-customer metadata rows whose substation sits beneath a pilot
    bulk supply point in `substations`. The limit of a node is the firm limit, the smaller of its
    per-transformer limits, in all four seasons. The inputs are private, so nothing here publishes
    them.

    Args:
        metadata: `data/NGED/metadata.parquet`, with `time_series_id`, `time_series_name`,
            `substation_number`, and `substation_type`.
        substations: `Substations.csv` of the substation hierarchy, with `substation_number`,
            `substation_name`, `bsp_substation_number`, `gsp_substation_number`, and `gsp_name`.
        transformer_export_limits_mw: Each node's published per-transformer export limits in MW,
            keyed by the node's `substation_name` in `substations`.
        signs: Each node's `export_sign` (+1, -1, or null) and `sign_source`, keyed the same way.
            A node absent from `signs` gets a null sign and the source "not given".
        capacity_mw: Each generator's capacity in MW, keyed by `time_series_id`.
        forecast_ids: The `time_series_id`s that have a forecast, which become `is_forecast`.
        transformer_import_limits_mw: Per-transformer import limits, if known; null otherwise.

    Returns:
        The validated `NodeTable`. Connection dates are null.

    Raises:
        ValueError: If a node has no export limit given, or a generator has no capacity given.
    """
    names = substations.select(
        "substation_number", "substation_name", "bsp_substation_number", "gsp_substation_number"
    )
    node_meta = (
        metadata.filter(pl.col("substation_type").is_in(["BSP", "GSP"]))
        .select(
            "time_series_id",
            "substation_number",
            node_type=pl.col("substation_type").cast(pl.String),
        )
        .join(
            names.select(
                "substation_number", node_id="substation_name", gsp="gsp_substation_number"
            ),
            on="substation_number",
            how="inner",
        )
    )
    parents = names.select(gsp="substation_number", parent_name="substation_name")
    node_meta = node_meta.join(parents, on="gsp", how="left").with_columns(
        parent_node_id=pl.when(pl.col("node_type") == "BSP").then(pl.col("parent_name"))
    )
    node_ids = node_meta["node_id"].to_list()
    missing = [n for n in node_ids if n not in transformer_export_limits_mw]
    if missing:
        msg = f"no export limits given for {len(missing)} node(s): {missing}"
        raise ValueError(msg)
    nodes = node_meta.select(
        "node_id",
        pl.col("time_series_id").cast(pl.Int64),
        "node_type",
        "parent_node_id",
        export_sign=pl.col("node_id").replace_strict(
            {n: signs.get(n, (None, ""))[0] for n in node_ids}, return_dtype=pl.Int64
        ),
        sign_source=pl.col("node_id").replace_strict(
            {n: signs.get(n, (None, "not given"))[1] for n in node_ids}, return_dtype=pl.String
        ),
    )
    import_limits = transformer_import_limits_mw or {}
    limits = pl.DataFrame(
        [
            {
                "node_id": node,
                "season": season,
                "export_limit_mw": min(transformer_export_limits_mw[node]),
                "import_limit_mw": min(import_limits[node]) if node in import_limits else None,
                "limit_source": "published firm limit: smaller of the per-transformer limits",
            }
            for node in node_ids
            for season in SEASON_NAMES
        ],
        schema={
            "node_id": pl.String,
            "season": pl.String,
            "export_limit_mw": pl.Float64,
            "import_limit_mw": pl.Float64,
            "limit_source": pl.String,
        },
    )
    generator_meta = (
        metadata.filter(pl.col("substation_type") == "EHV Customer")
        .select("time_series_id", "substation_number")
        .join(
            names.select("substation_number", bsp="bsp_substation_number"),
            on="substation_number",
            how="inner",
        )
        .join(
            names.select(bsp="substation_number", node_id="substation_name"),
            on="bsp",
            how="inner",
        )
        .filter(pl.col("node_id").is_in(node_ids))
    )
    no_capacity = [i for i in generator_meta["time_series_id"] if i not in capacity_mw]
    if no_capacity:
        msg = f"no capacity given for {len(no_capacity)} generator(s)"
        raise ValueError(msg)
    generators = generator_meta.select(
        generator_id=pl.col("time_series_id").cast(pl.Int64),
        node_id=pl.col("node_id"),
        capacity_mw=pl.col("time_series_id").replace_strict(
            dict(capacity_mw), return_dtype=pl.Float64, default=None
        ),
        connection_date=pl.lit(None, dtype=pl.Date),
        is_forecast=pl.col("time_series_id").is_in(list(forecast_ids)),
    ).sort("generator_id")
    return NodeTable(nodes=nodes.sort("node_id"), limits=limits, generators=generators)
