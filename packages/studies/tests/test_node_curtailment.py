"""Tests of the node curtailment metric. Every fixture is synthetic: no NGED series appears here.

Each test works its answer by hand in the docstring. Power is in MW and energy in MWh, and a
half-hour of `p` MW is `0.5 * p` MWh. Every fixture has a node with a 45.0 MW limit in all four
seasons unless a test says otherwise, so a 70 MW export is a 25.0 MW excess, which is 12.5 MWh in
the half-hour. The breach ratio is the default 20, so tau is 0.95 and the quantile columns are q05
and q95.
"""

import datetime as dt
from collections.abc import Sequence

import polars as pl
import pytest
from studies.bootstrap import bootstrap_month_totals
from studies.node_curtailment import (
    DEFAULT_SCALED_LIMIT_FRACTION,
    PRICES_GBP_PER_MWH,
    SEASON_NAMES,
    SEASONS,
    NodeCurtailmentAssumptions,
    NodeCurtailmentResult,
    NodeTable,
    allocate_pro_rata_by_capacity,
    event_census,
    month_total_arrays,
    node_curtailment,
    node_table_pilot,
    scale_to_megawatts,
    scaled_limits,
    table_with_scaled_limits,
)

UTC = dt.UTC
NODE_SERIES = 1
GENERATOR_SERIES = 101
SECOND_GENERATOR_SERIES = 102
HEADLINE = 2000.0


def _valid(*, month: int = 1, step: int = 0) -> dt.datetime:
    return dt.datetime(2026, month, 15, tzinfo=UTC) + dt.timedelta(minutes=30 * step)


def _nodes(*rows: tuple[str, int, int | None]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "node_id": node,
                "time_series_id": series,
                "node_type": "BSP",
                "parent_node_id": None,
                "export_sign": sign,
                "sign_source": "synthetic",
            }
            for node, series, sign in rows
        ],
        schema={
            "node_id": pl.String,
            "time_series_id": pl.Int64,
            "node_type": pl.String,
            "parent_node_id": pl.String,
            "export_sign": pl.Int64,
            "sign_source": pl.String,
        },
    )


def _limits(
    *, nodes: dict[str, float | Sequence[float]], import_mw: float | None = None
) -> pl.DataFrame:
    rows = []
    for node, value in nodes.items():
        per_season = [value] * 4 if isinstance(value, float | int) else list(value)
        rows.extend(
            {
                "node_id": node,
                "season": season,
                "export_limit_mw": float(limit),
                "import_limit_mw": import_mw,
                "limit_source": "synthetic",
            }
            for season, limit in zip(SEASON_NAMES, per_season, strict=True)
        )
    return pl.DataFrame(
        rows,
        schema={
            "node_id": pl.String,
            "season": pl.String,
            "export_limit_mw": pl.Float64,
            "import_limit_mw": pl.Float64,
            "limit_source": pl.String,
        },
    )


def _generators(*rows: tuple[int, str, float, bool]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "generator_id": gid,
                "node_id": node,
                "capacity_mw": capacity,
                "connection_date": None,
                "is_forecast": is_forecast,
            }
            for gid, node, capacity, is_forecast in rows
        ],
        schema={
            "generator_id": pl.Int64,
            "node_id": pl.String,
            "capacity_mw": pl.Float64,
            "connection_date": pl.Date,
            "is_forecast": pl.Boolean,
        },
    )


def _table(
    *,
    limit_mw: float = 45.0,
    sign: int | None = 1,
    generators: Sequence[tuple[int, float]] = (),
    import_mw: float | None = None,
) -> NodeTable:
    return NodeTable(
        nodes=_nodes(("A", NODE_SERIES, sign)),
        limits=_limits(nodes={"A": limit_mw}, import_mw=import_mw),
        generators=_generators(*((gid, "A", cap, True) for gid, cap in generators)),
    )


def _forecast(
    *,
    series: int,
    valid_times: Sequence[dt.datetime],
    actual: Sequence[float],
    q05: Sequence[float] | None = None,
    q50: Sequence[float] | None = None,
    q95: Sequence[float] | None = None,
    lead_h: int = 24,
    seed: int = 0,
) -> pl.DataFrame:
    """A forecast frame; any quantile left out defaults to the actual value (a perfect forecast)."""
    return pl.DataFrame(
        {
            "time_series_id": [series] * len(valid_times),
            "power_fcst_init_time": [v - dt.timedelta(hours=lead_h) for v in valid_times],
            "valid_time": list(valid_times),
            "seed": [seed] * len(valid_times),
            "actual": list(actual),
            "q05": list(q05 if q05 is not None else actual),
            "q50": list(q50 if q50 is not None else actual),
            "q95": list(q95 if q95 is not None else actual),
        },
        schema_overrides={"time_series_id": pl.Int64},
    )


def _one_row(
    *, flow: float, q05: float | None = None, q95: float | None = None, month: int = 1
) -> pl.DataFrame:
    return _forecast(
        series=NODE_SERIES,
        valid_times=[_valid(month=month)],
        actual=[flow],
        q05=_opt(q05),
        q95=_opt(q95),
    )


def _opt(value: float | None) -> list[float] | None:
    return None if value is None else [value]


def _generator_row(
    *, series: int = GENERATOR_SERIES, output: float, forecast: float, month: int = 1
) -> pl.DataFrame:
    return _forecast(
        series=series, valid_times=[_valid(month=month)], actual=[output], q50=[forecast]
    )


def _assumptions(**overrides: object) -> NodeCurtailmentAssumptions:
    return NodeCurtailmentAssumptions(**overrides)  # ty: ignore[invalid-argument-type]


def _headline(result_per_arm: pl.DataFrame, *, arm: str = "arm", column: str) -> float:
    row = result_per_arm.filter((pl.col("arm") == arm) & (pl.col("price_gbp_per_mwh") == HEADLINE))
    return row[column].item()


def _score(
    *,
    table: NodeTable,
    node: pl.DataFrame,
    generators: pl.DataFrame | None = None,
    assumptions: NodeCurtailmentAssumptions | None = None,
) -> NodeCurtailmentResult:
    return node_curtailment(
        table=table,
        node_forecasts={"arm": node},
        generator_forecasts={"arm": generators} if generators is not None else {},
        assumptions=assumptions or _assumptions(use_generators=generators is not None),
    )


# --- seasons and limits -------------------------------------------------------------------------


def test_seasons_are_whole_months_in_nged_groups() -> None:
    """Winter is Dec-Feb, Intermediate Cool is Mar/Apr/Nov, Warm is May/Sep/Oct, Summer Jun-Aug.

    That is 3 + 3 + 3 + 3 = 12 months, each in exactly one season.
    """
    by_season = {name: sorted(m for m, s in SEASONS.items() if s == name) for name in SEASON_NAMES}
    assert by_season["Winter"] == [1, 2, 12]
    assert by_season["Intermediate Cool"] == [3, 4, 11]
    assert by_season["Intermediate Warm"] == [5, 9, 10]
    assert by_season["Summer"] == [6, 7, 8]
    assert sorted(SEASONS) == list(range(1, 13))


def test_limits_are_picked_by_the_season_of_the_valid_time() -> None:
    """Limits Winter 30, Cool 35, Warm 40, Summer 45 MW; a 100 MW export in each season.

    January is Winter, April is Cool, September is Warm, July is Summer. With a perfect node
    forecast and no generators, the planned curtailment is the whole excess: (100 - L) * 0.5 MWh,
    so 35.0, 32.5, 30.0 and 27.5 MWh.
    """
    table = NodeTable(
        nodes=_nodes(("A", NODE_SERIES, 1)),
        limits=_limits(nodes={"A": [30.0, 35.0, 40.0, 45.0]}),
        generators=_generators(),
    )
    months = {1: "Winter", 4: "Intermediate Cool", 9: "Intermediate Warm", 7: "Summer"}
    node = pl.concat([_one_row(flow=100.0, month=m) for m in months])
    result = _score(table=table, node=node)
    needed = dict(result.per_node_season.select("season", "needed_mwh").iter_rows())
    assert needed == {
        "Winter": pytest.approx(35.0),
        "Intermediate Cool": pytest.approx(32.5),
        "Intermediate Warm": pytest.approx(30.0),
        "Summer": pytest.approx(27.5),
    }


def test_negative_sign_flow_over_the_limit_is_a_twelve_and_a_half_mwh_excess() -> None:
    """Sign -1, flow -70 MW against 45.0 MW: export 70, excess 25.0 MW, 12.5 MWh in the half-hour.

    A perfect forecast (q05 = q95 = -70) plans 12.5 MWh of node curtailment, delivered exactly, so
    the breach is max(0, 70 - 12.5 / 0.5 - 45) = 0 MW. Energy proxy at 2,000 GBP per MWh: 2,000 *
    (12.5 + 20 * 0) = 25,000 GBP.
    """
    result = _score(table=_table(sign=-1), node=_one_row(flow=-70.0))
    assert _headline(result.per_arm, column="needed_mwh") == pytest.approx(12.5)
    assert _headline(result.per_arm, column="planned_mwh") == pytest.approx(12.5)
    assert _headline(result.per_arm, column="curtailed_mwh") == pytest.approx(12.5)
    assert _headline(result.per_arm, column="breach_mwh") == pytest.approx(0.0)
    assert _headline(result.per_arm, column="energy_proxy_gbp") == pytest.approx(25_000.0)


def test_negative_sign_uses_the_low_flow_quantile_as_the_export_quantile() -> None:
    """Sign -1 with q05 = -80 and q95 = -10 MW: the export's 95th percentile is -q05 = 80 MW.

    The planned curtailment is (80 - 45) * 0.5 = 17.5 MWh, not the 0 that the flow's q95 (-10) would
    give.
    """
    result = _score(table=_table(sign=-1), node=_one_row(flow=-70.0, q05=-80.0, q95=-10.0))
    assert _headline(result.per_arm, column="planned_mwh") == pytest.approx(17.5)


def test_a_null_sign_node_is_excluded_and_counted() -> None:
    """Two nodes, one with a null sign: only the other is scored and one exclusion is counted."""
    table = NodeTable(
        nodes=_nodes(("A", 1, 1), ("B", 2, None)),
        limits=_limits(nodes={"A": 45.0, "B": 45.0}),
        generators=_generators(),
    )
    node = pl.concat(
        [
            _forecast(series=1, valid_times=[_valid()], actual=[70.0]),
            _forecast(series=2, valid_times=[_valid()], actual=[70.0]),
        ]
    )
    result = _score(table=table, node=node)
    assert result.n_excluded_unknown_sign == 1
    assert result.excluded.to_dicts() == [{"node_id": "B", "reason": "unknown_sign"}]
    assert result.per_node_season["node_id"].unique().to_list() == ["A"]


def test_a_node_without_a_limit_in_every_season_or_forecast_rows_is_excluded() -> None:
    """A node with 3 of 4 seasons' limits, and a node with no forecast rows, each get a reason."""
    limits = _limits(nodes={"A": 45.0, "B": 45.0, "C": 45.0}).filter(
        ~((pl.col("node_id") == "B") & (pl.col("season") == "Summer"))
    )
    table = NodeTable(
        nodes=_nodes(("A", 1, 1), ("B", 2, 1), ("C", 3, 1)), limits=limits, generators=_generators()
    )
    node = pl.concat([_forecast(series=s, valid_times=[_valid()], actual=[70.0]) for s in (1, 2)])
    result = _score(table=table, node=node)
    assert dict(result.excluded.iter_rows()) == {
        "B": "incomplete_limits",
        "C": "no_forecast_rows",
    }


# --- the rule: perfect, loose, and over-tight caps ----------------------------------------------


def test_perfect_forecasts_cost_exactly_the_needed_curtailment_when_the_generator_covers_it() -> (
    None
):
    """Export 70 MW, limit 45 MW, one 30 MW generator running at its forecast of 30 MW.

    Planned node curtailment: (70 - 45) * 0.5 = 12.5 MWh. Forecast energy 30 * 0.5 = 15 MWh covers
    it, so the generator's share is 12.5 MWh. Cap = 30 - 12.5 / 0.5 = 5 MW. Energy lost =
    (30 - 5) * 0.5 = 12.5 MWh. Breach = max(0, 70 - 12.5 / 0.5 - 45) = 0 MW. Energy proxy at 2,000
    GBP per MWh = 2,000 * 12.5 = 25,000 GBP.
    """
    result = _score(
        table=_table(generators=[(GENERATOR_SERIES, 30.0)]),
        node=_one_row(flow=70.0),
        generators=_generator_row(output=30.0, forecast=30.0),
    )
    assert _headline(result.per_arm, column="curtailed_mwh") == pytest.approx(12.5)
    assert _headline(result.per_arm, column="breach_mwh") == pytest.approx(0.0)
    assert _headline(result.per_arm, column="energy_proxy_gbp") == pytest.approx(25_000.0)


def test_uncovered_excess_becomes_breach_when_the_generator_cannot_cover_it() -> None:
    """Export 70 MW, limit 45 MW, a 30 MW generator that produces and is forecast at only 10 MW.

    Planned: 12.5 MWh. Forecast energy is 10 * 0.5 = 5 MWh, so the share is clamped at 5 MWh.
    Cap = 10 - 5 / 0.5 = 0 MW; energy lost = (10 - 0) * 0.5 = 5 MWh. Breach = 70 - 5 / 0.5 - 45 =
    15 MW = 7.5 MWh. Energy proxy at 2,000 GBP per MWh: 2,000 * (5 + 20 * 7.5) = 310,000 GBP.
    """
    result = _score(
        table=_table(generators=[(GENERATOR_SERIES, 30.0)]),
        node=_one_row(flow=70.0),
        generators=_generator_row(output=10.0, forecast=10.0),
    )
    assert _headline(result.per_arm, column="curtailed_mwh") == pytest.approx(5.0)
    assert _headline(result.per_arm, column="breach_mwh") == pytest.approx(7.5)
    assert _headline(result.per_arm, column="energy_proxy_gbp") == pytest.approx(310_000.0)


def test_a_forecast_too_high_leaves_a_loose_cap_and_the_breach_is_charged_exactly() -> None:
    """Forecast 40 MW, output 30 MW, capacity 50 MW; node 70 MW over 45 MW (perfect node forecast).

    Planned 12.5 MWh; forecast energy 20 MWh covers it. Cap = 40 - 25 = 15 MW. Energy lost =
    (30 - 15) * 0.5 = 7.5 MWh. Breach = 70 - 7.5 / 0.5 - 45 = 10 MW = 5 MWh. Energy proxy: 2,000 *
    (7.5 + 20 * 5) = 215,000 GBP.
    """
    result = _score(
        table=_table(generators=[(GENERATOR_SERIES, 50.0)]),
        node=_one_row(flow=70.0),
        generators=_generator_row(output=30.0, forecast=40.0),
    )
    assert _headline(result.per_arm, column="curtailed_mwh") == pytest.approx(7.5)
    assert _headline(result.per_arm, column="breach_mwh") == pytest.approx(5.0)
    assert _headline(result.per_arm, column="energy_proxy_gbp") == pytest.approx(215_000.0)


def test_a_forecast_too_low_over_cuts_by_an_exact_amount() -> None:
    """Forecast 26 MW, output 30 MW, capacity 50 MW; node 70 MW over 45 MW (perfect node forecast).

    Planned 12.5 MWh; forecast energy 13 MWh covers it. Cap = 26 - 25 = 1 MW. Energy lost =
    (30 - 1) * 0.5 = 14.5 MWh, which over-cuts the needed 12.5 MWh by 2.0 MWh. Breach = 70 - 29 - 45
    < 0, so 0. Energy proxy: 2,000 * 14.5 = 29,000 GBP.
    """
    result = _score(
        table=_table(generators=[(GENERATOR_SERIES, 50.0)]),
        node=_one_row(flow=70.0),
        generators=_generator_row(output=30.0, forecast=26.0),
    )
    assert _headline(result.per_arm, column="curtailed_mwh") == pytest.approx(14.5)
    assert _headline(result.per_arm, column="needed_mwh") == pytest.approx(12.5)
    assert _headline(result.per_arm, column="breach_mwh") == pytest.approx(0.0)
    assert _headline(result.per_arm, column="energy_proxy_gbp") == pytest.approx(29_000.0)


def test_a_generator_with_no_share_is_not_capped_even_when_the_node_is_under_its_limit() -> None:
    """Export 40 MW against 45 MW: planned 0, so no generator is capped and nothing is lost.

    A cap of f - 0 = 25 MW on an output of 30 MW would otherwise lose 2.5 MWh for no reason.
    """
    result = _score(
        table=_table(generators=[(GENERATOR_SERIES, 50.0)]),
        node=_one_row(flow=40.0),
        generators=_generator_row(output=30.0, forecast=25.0),
    )
    assert _headline(result.per_arm, column="curtailed_mwh") == pytest.approx(0.0)
    assert _headline(result.per_arm, column="energy_proxy_gbp") == pytest.approx(0.0)


# --- allocation ---------------------------------------------------------------------------------


def _allocate(
    *,
    groups: Sequence[str],
    capacity: Sequence[float],
    forecast: Sequence[float],
    need: Sequence[float],
) -> list[float]:
    frame = pl.DataFrame(
        {"group": groups, "capacity": capacity, "forecast": forecast, "need": need}
    )
    return allocate_pro_rata_by_capacity(
        frame=frame,
        group_by=["group"],
        capacity_column="capacity",
        forecast_mwh_column="forecast",
        need_mwh_column="need",
    )["allocated_mwh"].to_list()


def test_allocation_is_pro_rata_by_capacity() -> None:
    """Capacities 10 and 30 MW share a need of 8 MWh 25% / 75%: 2 and 6 MWh.

    Both forecast energies (5 and 15 MWh) exceed their shares, so nothing clamps.
    """
    assert _allocate(
        groups=["g", "g"], capacity=[10.0, 30.0], forecast=[5.0, 15.0], need=[8.0, 8.0]
    ) == pytest.approx([2.0, 6.0])


def test_allocation_clamps_at_the_forecast_and_redistributes() -> None:
    """Need 8 MWh, capacities 10 and 30, forecast energies 1 and 15 MWh.

    Round 1: shares 2 and 6; the first exceeds its 1 MWh forecast, so it is clamped at 1. Round 2:
    the remaining 7 MWh goes to the second generator alone, which has 15 MWh of forecast: 7.
    """
    assert _allocate(
        groups=["g", "g"], capacity=[10.0, 30.0], forecast=[1.0, 15.0], need=[8.0, 8.0]
    ) == pytest.approx([1.0, 7.0])


def test_allocation_leaves_a_remainder_when_every_generator_is_at_its_forecast() -> None:
    """Need 8 MWh, forecast energies 1 and 3 MWh: allocations 1 and 3, 4 MWh left unallocated."""
    allocated = _allocate(
        groups=["g", "g"], capacity=[10.0, 30.0], forecast=[1.0, 3.0], need=[8.0, 8.0]
    )
    assert allocated == pytest.approx([1.0, 3.0])
    assert 8.0 - sum(allocated) == pytest.approx(4.0)


def test_allocation_repeats_clamping_over_several_rounds() -> None:
    """Capacities 1, 2, 2 MW, need 9 MWh, forecast energies 1, 3.9, 20 MWh.

    Round 1: shares 1.8, 3.6, 3.6; only the first is above its forecast (1), so it clamps at 1.
    Round 2: 8 MWh over capacities 2 and 2 gives 4 and 4; the second exceeds its 3.9, so it clamps.
    Round 3: the remaining 9 - 1 - 3.9 = 4.1 MWh goes to the third generator.
    """
    assert _allocate(
        groups=["g"] * 3, capacity=[1.0, 2.0, 2.0], forecast=[1.0, 3.9, 20.0], need=[9.0] * 3
    ) == pytest.approx([1.0, 3.9, 4.1])


def test_allocation_keeps_groups_independent() -> None:
    """Group a (need 8, forecasts 1 and 15) clamps; group b (need 4, forecasts 9 and 9) does not.

    Group a: [1, 7]. Group b with equal capacities: [2, 2].
    """
    assert _allocate(
        groups=["a", "a", "b", "b"],
        capacity=[10.0, 30.0, 5.0, 5.0],
        forecast=[1.0, 15.0, 9.0, 9.0],
        need=[8.0, 8.0, 4.0, 4.0],
    ) == pytest.approx([1.0, 7.0, 2.0, 2.0])


def test_two_unequal_generators_share_the_curtailment_through_the_whole_chain() -> None:
    """Capacities 10 and 30 MW, both running at forecast (10 and 30 MW); node 70 MW over 45 MW.

    Planned 12.5 MWh; forecast energies 5 and 15 MWh. Shares 3.125 and 9.375 MWh. The first is under
    its 5 MWh forecast energy and the second under 15, so there is no clamp. Caps: 10 - 6.25 = 3.75
    MW and 30 - 18.75 = 11.25 MW. Energy lost: (10 - 3.75) * 0.5 = 3.125 and (30 - 11.25) * 0.5 =
    9.375 MWh, 12.5 MWh in all, with no breach.
    """
    result = _score(
        table=_table(generators=[(GENERATOR_SERIES, 10.0), (SECOND_GENERATOR_SERIES, 30.0)]),
        node=_one_row(flow=70.0),
        generators=pl.concat(
            [
                _generator_row(series=GENERATOR_SERIES, output=10.0, forecast=10.0),
                _generator_row(series=SECOND_GENERATOR_SERIES, output=30.0, forecast=30.0),
            ]
        ),
    )
    assert _headline(result.per_arm, column="curtailed_mwh") == pytest.approx(12.5)
    assert _headline(result.per_arm, column="breach_mwh") == pytest.approx(0.0)


# --- the DCUSA form -----------------------------------------------------------------------------


def _dcusa_case(*, output: float, forecast: float, limit_hours: float) -> pl.DataFrame:
    return _score(
        table=_table(generators=[(GENERATOR_SERIES, 30.0)]),
        node=_one_row(flow=70.0),
        generators=_generator_row(output=output, forecast=forecast),
        assumptions=_assumptions(curtailment_limit_hours=limit_hours),
    ).per_arm


def test_dcusa_form_charges_capacity_minus_cap_for_the_instructed_half_hour() -> None:
    """Capacity 30 MW, forecast and output 30 MW, node 70 MW over 45 MW: cap 5 MW (as above).

    The instruction value is 30 - 5 = 25 MW for 0.5 h, so Full Export Curtailment Hours are
    0.5 * 25 / 30 = 0.41667 h. At a Curtailment Limit of 0, the payment is 0.41667 h * 30 MW *
    2,000 GBP per MWh = 25,000 GBP, and there is no breach.
    """
    per_arm = _dcusa_case(output=30.0, forecast=30.0, limit_hours=0.0)
    assert _headline(per_arm, column="dcusa_form_gbp") == pytest.approx(25_000.0)


def test_dcusa_form_pays_only_the_hours_above_the_curtailment_limit() -> None:
    """The same case, 0.41667 h of Full Export Curtailment Hours.

    Limit 0.25 h: (0.41667 - 0.25) * 30 * 2,000 = 10,000 GBP. Limit 0.5 h, above the hours used:
    nothing is paid, so 0 GBP.
    """
    assert _headline(
        _dcusa_case(output=30.0, forecast=30.0, limit_hours=0.25), column="dcusa_form_gbp"
    ) == pytest.approx(10_000.0)
    assert _headline(
        _dcusa_case(output=30.0, forecast=30.0, limit_hours=0.5), column="dcusa_form_gbp"
    ) == pytest.approx(0.0)


def test_dcusa_form_charges_the_headroom_of_a_generator_running_below_capacity() -> None:
    """Capacity 30 MW but output and forecast 10 MW; node 70 MW over 45 MW.

    As in the energy-proxy case, the share is clamped at 5 MWh, the cap is 0 MW, and the breach is
    7.5 MWh. The instruction value is 30 - 0 = 30 MW, so the hours are 0.5 * 30 / 30 = 0.5 h and the
    payment at limit 0 is 0.5 * 30 * 2,000 = 30,000 GBP. Adding the breach, 2,000 * 20 * 7.5 =
    300,000, gives 330,000 GBP, against the energy proxy's 310,000 GBP: the forms differ.
    """
    per_arm = _dcusa_case(output=10.0, forecast=10.0, limit_hours=0.0)
    assert _headline(per_arm, column="dcusa_form_gbp") == pytest.approx(330_000.0)
    assert _headline(per_arm, column="energy_proxy_gbp") == pytest.approx(310_000.0)


# --- import side, node-only mode, prices --------------------------------------------------------


def test_an_import_exceedance_is_counted_and_not_priced() -> None:
    """Sign +1, flow -60 MW is an import of 60 MW against a 50 MW import limit: one exceedance.

    The export side sees -60 MW against 45 MW, no exceedance. Nothing is priced: 0 GBP.
    """
    result = _score(table=_table(import_mw=50.0), node=_one_row(flow=-60.0))
    census = result.event_census.filter(pl.col("season") == "Winter")
    imports = census.filter(pl.col("direction") == "import")
    exports = census.filter(pl.col("direction") == "export")
    assert imports["n_exceedances"].to_list() == [1]
    assert exports["n_exceedances"].to_list() == [0]
    assert _headline(result.per_arm, column="energy_proxy_gbp") == 0.0


def test_node_only_mode_equals_full_mode_with_one_perfect_generator_that_can_cover_it() -> None:
    """Node 70 MW over 45 MW; one generator, forecast and output 30 MW, at least 25 MW needed.

    Node-only: curtailed = planned = 12.5 MWh, breach 0, 25,000 GBP. Full mode (as in the first
    generator test): the same 12.5 MWh, 0 breach, 25,000 GBP.
    """
    node = _one_row(flow=70.0)
    node_only = _score(table=_table(), node=node)
    full = _score(
        table=_table(generators=[(GENERATOR_SERIES, 30.0)]),
        node=node,
        generators=_generator_row(output=30.0, forecast=30.0),
    )
    for column in ("curtailed_mwh", "breach_mwh", "energy_proxy_gbp"):
        assert _headline(node_only.per_arm, column=column) == pytest.approx(
            _headline(full.per_arm, column=column)
        )
    assert _headline(node_only.per_arm, column="dcusa_form_gbp") is None


def test_gbp_scales_linearly_over_the_four_prices_with_the_headline_marked() -> None:
    """The energy proxy for 12.5 MWh curtailed and no breach is 12.5 * price.

    Prices 100, 380, 2,000, 11,100 GBP per MWh give 1,250, 4,750, 25,000 and 138,750 GBP; only the
    2,000 GBP row is the headline.
    """
    per_arm = _score(table=_table(), node=_one_row(flow=70.0)).per_arm
    assert per_arm["price_gbp_per_mwh"].to_list() == list(PRICES_GBP_PER_MWH)
    assert PRICES_GBP_PER_MWH == (100.0, 380.0, 2000.0, 11100.0)
    assert per_arm["energy_proxy_gbp"].to_list() == pytest.approx(
        [1250.0, 4750.0, 25000.0, 138750.0]
    )
    assert per_arm["is_headline_price"].to_list() == [False, False, True, False]


def test_only_forecast_generators_are_curtailed() -> None:
    """A generator with `is_forecast` false is never capped, so the whole excess is breach.

    Node 70 MW over 45 MW, one unforecast generator: curtailed 0, breach (70 - 45) * 0.5 = 12.5 MWh.
    """
    table = NodeTable(
        nodes=_nodes(("A", NODE_SERIES, 1)),
        limits=_limits(nodes={"A": 45.0}),
        generators=_generators((GENERATOR_SERIES, "A", 30.0, False)),
    )
    result = _score(
        table=table,
        node=_one_row(flow=70.0),
        generators=_generator_row(output=30.0, forecast=30.0),
    )
    assert _headline(result.per_arm, column="curtailed_mwh") == pytest.approx(0.0)
    assert _headline(result.per_arm, column="breach_mwh") == pytest.approx(12.5)


def test_a_generator_not_yet_connected_is_not_curtailable() -> None:
    """A generator connected on 2026-02-01 takes no share on 2026-01-15: the breach is 12.5 MWh."""
    generators = _generators((GENERATOR_SERIES, "A", 30.0, True)).with_columns(
        connection_date=pl.lit(dt.date(2026, 2, 1))
    )
    table = NodeTable(
        nodes=_nodes(("A", NODE_SERIES, 1)),
        limits=_limits(nodes={"A": 45.0}),
        generators=generators,
    )
    result = _score(
        table=table, node=_one_row(flow=70.0), generators=_generator_row(output=30.0, forecast=30.0)
    )
    assert _headline(result.per_arm, column="curtailed_mwh") == pytest.approx(0.0)
    assert _headline(result.per_arm, column="breach_mwh") == pytest.approx(12.5)


# --- rows, arms, and many nodes -----------------------------------------------------------------


def test_the_lead_window_is_12_inclusive_to_36_exclusive() -> None:
    """Leads 6, 12, 24 and 36 hours with a 70 MW export over 45 MW: 12.5 MWh needed per row scored.

    Only the 12 h and 24 h rows are inside the window [12, 36), so 2 * 12.5 = 25.0 MWh.
    """
    node = pl.concat(
        [
            _forecast(series=NODE_SERIES, valid_times=[_valid(step=i)], actual=[70.0], lead_h=lead)
            for i, lead in enumerate((6, 12, 24, 36))
        ]
    )
    result = _score(table=_table(), node=node)
    assert _headline(result.per_arm, column="needed_mwh") == pytest.approx(25.0)
    assert _headline(result.per_arm, column="n_node_half_hours") == 2


def test_a_row_missing_from_one_arm_is_dropped_from_every_arm() -> None:
    """Arm x holds 2 rows and arm y holds 1 of them: both are scored on the 1 shared row.

    Each row needs 12.5 MWh, so each arm's needed energy is 12.5 MWh and one row is dropped.
    """
    both = _forecast(
        series=NODE_SERIES, valid_times=[_valid(step=0), _valid(step=1)], actual=[70.0, 70.0]
    )
    result = node_curtailment(
        table=_table(),
        node_forecasts={"x": both, "y": both.head(1)},
        generator_forecasts={},
        assumptions=_assumptions(use_generators=False),
    )
    assert result.per_arm.filter(pl.col("price_gbp_per_mwh") == HEADLINE)[
        "needed_mwh"
    ].to_list() == pytest.approx([12.5, 12.5])
    assert result.n_rows_dropped_not_in_every_arm == 0  # align_arms already intersected the keys


def test_a_node_row_whose_generator_has_no_forecast_row_is_dropped_and_counted() -> None:
    """Two half-hours; the generator has a forecast row for only the first.

    The second half-hour would otherwise be scored as if the generator were absent, so it is
    dropped: one node row scored (needed 12.5 MWh), one dropped.
    """
    node = _forecast(
        series=NODE_SERIES, valid_times=[_valid(step=0), _valid(step=1)], actual=[70.0, 70.0]
    )
    result = _score(
        table=_table(generators=[(GENERATOR_SERIES, 30.0)]),
        node=node,
        generators=_generator_row(output=30.0, forecast=30.0),
    )
    assert _headline(result.per_arm, column="needed_mwh") == pytest.approx(12.5)
    assert result.n_rows_dropped_missing_generator == 1


def test_many_nodes_are_scored_together_with_each_nodes_own_limit() -> None:
    """50 nodes with limits 40, 41, ..., 89 MW, each exporting 70 MW (perfect forecast, node only).

    Node i (limit 40 + i) needs max(0, 30 - i) * 0.5 MWh; the sum over i = 0..29 of (30 - i) * 0.5
    = 0.5 * 465 = 232.5 MWh. Node 10 (limit 50) alone needs 20 * 0.5 = 10.0 MWh.
    """
    ids = [f"N{i:02d}" for i in range(50)]
    table = NodeTable(
        nodes=_nodes(*((n, i + 1, 1) for i, n in enumerate(ids))),
        limits=_limits(nodes={n: 40.0 + i for i, n in enumerate(ids)}),
        generators=_generators(),
    )
    node = pl.concat(
        [_forecast(series=i + 1, valid_times=[_valid()], actual=[70.0]) for i in range(50)]
    )
    result = _score(table=table, node=node)
    assert _headline(result.per_arm, column="needed_mwh") == pytest.approx(232.5)
    node_10 = result.per_node_season.filter(
        (pl.col("node_id") == "N10") & (pl.col("season") == "Winter")
    )
    assert node_10["needed_mwh"].item() == pytest.approx(10.0)


def test_the_per_arm_gbp_per_mw_year_divides_by_curtailable_capacity_and_years() -> None:
    """One half-hour is 0.5 / 8,766 years. With 30 MW of capacity: GBP / (30 * 0.5 / 8,766).

    The energy proxy is 25,000 GBP, so 25,000 / (30 * 0.5 / 8,766) = 14,610,000 GBP per MW-year.
    """
    result = _score(
        table=_table(generators=[(GENERATOR_SERIES, 30.0)]),
        node=_one_row(flow=70.0),
        generators=_generator_row(output=30.0, forecast=30.0),
    )
    assert _headline(result.per_arm, column="energy_proxy_gbp_per_mw_year") == pytest.approx(
        25_000.0 / (30.0 * 0.5 / 8766.0)
    )


def test_month_totals_sum_to_the_per_arm_totals_and_feed_the_month_bootstrap() -> None:
    """Rows in January and April each need 12.5 MWh: month sums [12.5, 12.5], counts [1, 1].

    The array shapes are (n_seeds, n_months) = (1, 2) and (2,), and `bootstrap_month_totals`
    accepts them. The energy-proxy MWh-equivalent per month is 12.5 (no breach).
    """
    node = pl.concat([_one_row(flow=70.0, month=1), _one_row(flow=70.0, month=4)])
    result = _score(table=_table(), node=node)
    sums, counts = month_total_arrays(
        month_totals=result.month_totals, arm="arm", column="energy_proxy_mwh_equivalent"
    )
    assert sums.tolist() == [[12.5, 12.5]]
    assert counts.tolist() == [1.0, 1.0]
    lower, upper = bootstrap_month_totals(month_sums=sums, month_counts=counts)
    assert lower == pytest.approx(12.5)
    assert upper == pytest.approx(12.5)


# --- scaled limits ------------------------------------------------------------------------------

CUTOFF = dt.datetime(2025, 7, 2, tzinfo=UTC)


def _flows(*, series: int, values: Sequence[float], start: dt.datetime) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_series_id": pl.Series([series] * len(values), dtype=pl.Int64),
            "valid_time": [start + dt.timedelta(minutes=30 * i) for i in range(len(values))],
            "flow_mw": values,
        }
    )


def _history(*, before: Sequence[float], after: Sequence[float], series: int = 1) -> pl.DataFrame:
    return pl.concat(
        [
            _flows(series=series, values=before, start=CUTOFF - dt.timedelta(days=30)),
            _flows(series=series, values=after, start=CUTOFF),
        ]
    )


def test_the_scaled_limit_is_f_times_the_maximum_export_before_the_cutoff() -> None:
    """Exports 10, 40, 20 MW before the cutoff: the maximum is 40, so f=0.9 gives 36.0 MW.

    f = 0.75 gives 30.0 and f = 0.5 gives 20.0 MW. It is a multiple of the maximum, not the 90th
    percentile of [10, 40, 20] (which would be 36 by interpolation by coincidence, but 10% of the
    half-hours would exceed it by construction).
    """
    table = _table()
    history = _history(before=[10.0, 40.0, 20.0], after=[])
    assert DEFAULT_SCALED_LIMIT_FRACTION == 0.9
    for fraction, expected in ((0.9, 36.0), (0.75, 30.0), (0.5, 20.0)):
        result = scaled_limits(table=table, flows=history, cutoff=CUTOFF, fraction=fraction)
        assert result["export_limit_mw"].to_list() == pytest.approx([expected])


def test_data_on_or_after_the_cutoff_never_changes_the_scaled_limit() -> None:
    """Flows of 1,000 MW at the cutoff instant and after leave the limit at 0.9 * 40 = 36.0 MW.

    A row exactly at the cutoff is excluded, so only the earlier maximum of 40 MW counts.
    """
    table = _table()
    quiet = scaled_limits(
        table=table, flows=_history(before=[40.0], after=[5.0]), cutoff=CUTOFF, fraction=0.9
    )
    loud = scaled_limits(
        table=table,
        flows=_history(before=[40.0], after=[1000.0, 1000.0]),
        cutoff=CUTOFF,
        fraction=0.9,
    )
    assert quiet.equals(loud)
    assert loud["export_limit_mw"].to_list() == pytest.approx([36.0])


def test_the_scaled_limit_works_for_an_import_positive_node() -> None:
    """Sign -1 node with flows 5, -40, -20 MW: exports are -5, 40, 20, so the maximum is 40 MW.

    The limit is 0.9 * 40 = 36.0 MW. The maximum import is 5 MW, so the import limit is 4.5 MW.
    """
    table = _table(sign=-1)
    result = scaled_limits(
        table=table,
        flows=_history(before=[5.0, -40.0, -20.0], after=[]),
        cutoff=CUTOFF,
        fraction=0.9,
    )
    assert result["export_limit_mw"].to_list() == pytest.approx([36.0])
    assert result["import_limit_mw"].to_list() == pytest.approx([4.5])


def test_a_node_that_never_exported_has_a_null_scaled_limit_and_is_then_excluded() -> None:
    """Exports -5, -9, -2 MW (sign +1): a limit of 0.9 * -2 is not a limit, so it is null.

    `table_with_scaled_limits` then leaves the node with no limits and `node_curtailment` excludes
    it as `incomplete_limits`.
    """
    table = _table()
    scaled = scaled_limits(
        table=table,
        flows=_history(before=[-5.0, -9.0, -2.0], after=[]),
        cutoff=CUTOFF,
        fraction=0.9,
    )
    assert scaled["export_limit_mw"].to_list() == [None]
    result = _score(
        table=table_with_scaled_limits(table=table, scaled=scaled, fraction=0.9),
        node=_one_row(flow=70.0),
    )
    assert result.excluded.to_dicts() == [{"node_id": "A", "reason": "incomplete_limits"}]


def test_a_null_sign_node_has_no_scaled_limit() -> None:
    """A node with a null sign cannot say which tail is export, so it is left out."""
    table = _table(sign=None)
    result = scaled_limits(
        table=table, flows=_history(before=[10.0], after=[]), cutoff=CUTOFF, fraction=0.9
    )
    assert result.height == 0


def test_scaled_limits_replace_the_real_limit_in_all_four_seasons() -> None:
    """Scaled limit 36.0 MW from a 40 MW maximum; a 70 MW export needs (70 - 36) * 0.5 = 17 MWh."""
    table = _table()
    scaled = scaled_limits(
        table=table, flows=_history(before=[40.0], after=[]), cutoff=CUTOFF, fraction=0.9
    )
    scaled_table = table_with_scaled_limits(table=table, scaled=scaled, fraction=0.9)
    assert scaled_table.limits["limit_source"].unique().to_list() == ["scaled f=0.9"]
    assert scaled_table.limits.height == 4
    result = _score(table=scaled_table, node=_one_row(flow=70.0))
    assert _headline(result.per_arm, column="needed_mwh") == pytest.approx(17.0)


def test_scaled_limits_reject_a_fraction_outside_zero_to_one() -> None:
    with pytest.raises(ValueError, match="fraction"):
        scaled_limits(
            table=_table(), flows=_history(before=[1.0], after=[]), cutoff=CUTOFF, fraction=1.5
        )


# --- the event census ---------------------------------------------------------------------------


def test_the_census_counts_exceedances_at_the_real_limit_and_each_scaled_limit() -> None:
    """Winter exports 10, 40, 44, 46 MW against a real 45 MW limit: 1 exceedance (46).

    Scaled from a 40 MW maximum, f=0.9 gives 36 MW: 3 exceedances (40, 44, 46); f=0.5 gives 20 MW:
    3 as well. Strictly above: a value equal to a limit is not an exceedance. Four half-hours.
    """
    table = _table()
    flows = _flows(series=NODE_SERIES, values=[10.0, 40.0, 44.0, 46.0], start=_valid())
    scaled = {
        0.9: pl.DataFrame({"node_id": ["A"], "export_limit_mw": [36.0], "import_limit_mw": [None]}),
        0.5: pl.DataFrame({"node_id": ["A"], "export_limit_mw": [20.0], "import_limit_mw": [None]}),
    }
    census = event_census(table=table, flows=flows, scaled=scaled).filter(
        (pl.col("season") == "Winter") & (pl.col("direction") == "export")
    )
    counts = dict(census.select("limit_kind", "n_exceedances").iter_rows())
    assert counts == {"real": 1, "scaled f=0.9": 3, "scaled f=0.5": 3}
    assert census["n_half_hours"].unique().to_list() == [4]


# --- validation and the scaling helper ----------------------------------------------------------


def test_the_table_rejects_a_bad_sign_a_repeated_season_and_an_unknown_node() -> None:
    nodes = _nodes(("A", 1, 1))
    with pytest.raises(ValueError, match="export_sign"):
        NodeTable(
            nodes=_nodes(("A", 1, 0)), limits=_limits(nodes={"A": 45.0}), generators=_generators()
        )
    with pytest.raises(ValueError, match="repeats"):
        NodeTable(
            nodes=nodes,
            limits=pl.concat([_limits(nodes={"A": 45.0}), _limits(nodes={"A": 45.0})]),
            generators=_generators(),
        )
    with pytest.raises(ValueError, match="generators must name a node"):
        NodeTable(
            nodes=nodes,
            limits=_limits(nodes={"A": 45.0}),
            generators=_generators((9, "Z", 10.0, True)),
        )


def test_the_table_rejects_a_nonpositive_capacity() -> None:
    with pytest.raises(Exception, match="capacity_mw"):
        NodeTable(
            nodes=_nodes(("A", 1, 1)),
            limits=_limits(nodes={"A": 45.0}),
            generators=_generators((9, "A", 0.0, True)),
        )


def test_scale_to_megawatts_multiplies_actual_and_quantiles_by_each_capacity() -> None:
    """A fraction of 0.5 of a 40 MW series is 20 MW; of a 10 MW series 5 MW."""
    frame = pl.concat(
        [
            _forecast(series=1, valid_times=[_valid()], actual=[0.5], q50=[0.25]),
            _forecast(series=2, valid_times=[_valid()], actual=[0.5], q50=[0.25]),
        ]
    )
    scaled = scale_to_megawatts(frame=frame, capacity_mw={1: 40.0, 2: 10.0})
    assert scaled["actual"].to_list() == pytest.approx([20.0, 5.0])
    assert scaled["q50"].to_list() == pytest.approx([10.0, 2.5])


# --- the pilot builder --------------------------------------------------------------------------


def test_the_pilot_builder_takes_the_smaller_transformer_limit_and_places_generators() -> None:
    """A BSP with transformers at 30.0 and 31.5 MW gets the firm limit 30.0 MW in all four seasons.

    Its parent is the grid supply point named in the hierarchy; the grid supply point has none and
    a null sign by default. The 2 EHV-customer series under the BSP become generators, only the one
    in `forecast_ids` with `is_forecast` true.
    """
    metadata = pl.DataFrame(
        {
            "time_series_id": [1, 2, 11, 12, 13],
            "substation_number": [100, 200, 301, 302, 999],
            "substation_type": ["BSP", "GSP", "EHV Customer", "EHV Customer", "EHV Customer"],
        }
    )
    substations = pl.DataFrame(
        {
            "substation_number": [100, 200, 301, 302],
            "substation_name": ["NODE ONE", "NODE TWO", "SITE X", "SITE Y"],
            "bsp_substation_number": [100, None, 100, 100],
            "gsp_substation_number": [200, 200, 200, 200],
        }
    )
    table = node_table_pilot(
        metadata=metadata,
        substations=substations,
        transformer_export_limits_mw={"NODE ONE": [31.5, 30.0], "NODE TWO": [260.0]},
        signs={"NODE ONE": (-1, "from our own data")},
        capacity_mw={11: 5.0, 12: 7.0},
        forecast_ids={11},
    )
    one = table.nodes.filter(pl.col("node_id") == "NODE ONE").row(0, named=True)
    two = table.nodes.filter(pl.col("node_id") == "NODE TWO").row(0, named=True)
    assert (one["node_type"], one["parent_node_id"], one["export_sign"]) == ("BSP", "NODE TWO", -1)
    assert (two["node_type"], two["parent_node_id"], two["export_sign"]) == ("GSP", None, None)
    assert (
        table.limits.filter(pl.col("node_id") == "NODE ONE")["export_limit_mw"].to_list()
        == [30.0] * 4
    )
    assert table.generators.sort("generator_id").select(
        "generator_id", "node_id", "capacity_mw", "is_forecast"
    ).rows() == [
        (11, "NODE ONE", 5.0, True),
        (12, "NODE ONE", 7.0, False),
    ]


def test_the_pilot_builder_names_a_node_without_limits() -> None:
    metadata = pl.DataFrame(
        {"time_series_id": [1], "substation_number": [100], "substation_type": ["BSP"]}
    )
    substations = pl.DataFrame(
        {
            "substation_number": [100],
            "substation_name": ["NODE ONE"],
            "bsp_substation_number": [100],
            "gsp_substation_number": [100],
        }
    )
    with pytest.raises(ValueError, match="no export limits"):
        node_table_pilot(
            metadata=metadata,
            substations=substations,
            transformer_export_limits_mw={},
            signs={},
            capacity_mw={},
            forecast_ids=set(),
        )


def test_totals_are_averaged_over_fitting_seeds() -> None:
    """Seed 0 exports 70 MW (needs 12.5 MWh) and seed 1 exports 80 MW (needs (80-45)*0.5=17.5).

    Each seed's total is taken first, then averaged: (12.5 + 17.5) / 2 = 15.0 MWh, over one
    half-hour per seed.
    """
    node = pl.concat(
        [
            _forecast(series=NODE_SERIES, valid_times=[_valid()], actual=[70.0], seed=0),
            _forecast(series=NODE_SERIES, valid_times=[_valid()], actual=[80.0], seed=1),
        ]
    )
    result = _score(table=_table(), node=node)
    assert _headline(result.per_arm, column="needed_mwh") == pytest.approx(15.0)
    assert _headline(result.per_arm, column="n_node_half_hours") == pytest.approx(1.0)


def test_month_totals_are_price_free_and_match_the_priced_figures() -> None:
    """The 10 MW generator case: curtailed 5 MWh, breach 7.5 MWh, instruction 30 MW for 0.5 h.

    Energy-proxy MWh-equivalent: 5 + 20 * 7.5 = 155, so 155 * 2,000 = 310,000 GBP. DCUSA gross
    MWh-equivalent: 0.5 * 30 + 20 * 7.5 = 165, so 165 * 2,000 = 330,000 GBP (the case above).
    """
    result = _score(
        table=_table(generators=[(GENERATOR_SERIES, 30.0)]),
        node=_one_row(flow=70.0),
        generators=_generator_row(output=10.0, forecast=10.0),
    )
    totals = result.month_totals.row(0, named=True)
    assert totals["energy_proxy_mwh_equivalent"] == pytest.approx(155.0)
    assert totals["dcusa_gross_mwh_equivalent"] == pytest.approx(165.0)
    assert totals["n_node_half_hours"] == 1
    assert totals["month"] == "2026-01"
