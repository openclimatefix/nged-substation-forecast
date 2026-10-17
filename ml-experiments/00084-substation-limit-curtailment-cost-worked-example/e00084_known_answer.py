"""A known-answer example: one made-up half-hour through the whole chain, beside a hand answer.

Every number here is invented; no NGED series is used. The hand answers are worked in the comments
and written as literals, never computed with the code under test, so a disagreement shows a fault
in one or the other.

The made-up bulk supply point has a firm limit of 45.0 MW and a negative sign (a positive reading is
import). Its two generators have capacities of 10 and 20 MW. In the half-hour, the node's forecast
5% flow quantile is -70 MW, so the 0.95 quantile of export is 70 MW, 25 MW over the limit. The
actual flow is -65 MW (an export of 65 MW).

    Planned node curtailment = (70 - 45) x 0.5 h                     = 12.5 MWh
    Needed (actual)          = (65 - 45) x 0.5 h                     = 10.0 MWh
    Pro rata by capacity     = 12.5 x 10/30 = 4.1667 MWh for A, 12.5 x 20/30 = 8.3333 MWh for B
    A forecasts 8 MW, so can give at most 8 x 0.5 = 4.0 MWh: A is clamped at 4.0 MWh, and
    B takes the rest, 12.5 - 4.0                                     = 8.5 MWh
    Caps: A = max(0, 8 - 4.0/0.5) = 0 MW; B = max(0, 18 - 8.5/0.5) = 1 MW
    Actual output: A 8 MW, B 12 MW. Energy lost: A = (8 - 0) x 0.5 = 4.0 MWh;
    B = (12 - 1) x 0.5 = 5.5 MWh. Curtailed total                    = 9.5 MWh
    Residual breach = max(0, 65 - 9.5/0.5 - 45) x 0.5 = (65 - 19 - 45) x 0.5 = 0.5 MWh
    Energy proxy at GBP 2,000/MWh = 2,000 x (9.5 + 20 x 0.5)        = GBP 39,000
    DCUSA form: instruction values 10 - 0 = 10 MW (A) and 20 - 1 = 19 MW (B).
    Full Export Curtailment Hours: A 0.5 x 10/10 = 0.5 h, B 0.5 x 19/20 = 0.475 h; with a
    Curtailment Limit of 0 the payment is 0.5 x 10 + 0.475 x 20 = 14.5 MWh-equivalent, plus the
    breach 20 x 0.5 = 10 MWh-equivalent: 2,000 x 24.5               = GBP 49,000
"""

import datetime as dt
import json
from typing import Final

import e00084_inputs as inputs
import polars as pl
from studies.node_curtailment import (
    NodeCurtailmentAssumptions,
    NodeTable,
    allocate_pro_rata_by_capacity,
    node_curtailment,
)

UTC = dt.UTC
RUN = dt.datetime(2026, 1, 14, 0, tzinfo=UTC)
VALID = dt.datetime(2026, 1, 15, 0, tzinfo=UTC)
"""A lead of 24 hours, inside the 12 to 36 hour window; January is Winter."""

HAND_ANSWER: Final[dict[str, float]] = {
    "Planned node curtailment (MWh)": 12.5,
    "Needed (MWh)": 10.0,
    "Generator A share, clamped at its forecast (MWh)": 4.0,
    "Generator B share, after the clamp (MWh)": 8.5,
    "Curtailed (MWh)": 9.5,
    "Residual breach (MWh)": 0.5,
    "Energy proxy (GBP)": 39000.0,
    "DCUSA form (GBP)": 49000.0,
}
"""The hand-worked answer from the module docstring, as literals."""


def _frame(*, series: int, q: dict[str, float], actual: float) -> pl.DataFrame:
    row = {
        "time_series_id": series,
        "power_fcst_init_time": RUN,
        "valid_time": VALID,
        "seed": 0,
        "actual": actual,
        **q,
    }
    return pl.DataFrame([row])


def run() -> list[dict[str, object]]:
    """Run the made-up half-hour through `node_curtailment` and compare with the hand answer."""
    table = NodeTable(
        nodes=pl.DataFrame(
            {
                "node_id": ["X"],
                "time_series_id": [1],
                "node_type": ["BSP"],
                "parent_node_id": [None],
                "export_sign": [-1],
                "sign_source": ["made up"],
            },
            schema_overrides={"parent_node_id": pl.String},
        ),
        limits=pl.DataFrame(
            {
                "node_id": ["X"] * 4,
                "season": ["Winter", "Intermediate Cool", "Intermediate Warm", "Summer"],
                "export_limit_mw": [45.0] * 4,
                "import_limit_mw": [None] * 4,
                "limit_source": ["made up"] * 4,
            },
            schema_overrides={"import_limit_mw": pl.Float64},
        ),
        generators=pl.DataFrame(
            {
                "generator_id": [101, 102],
                "node_id": ["X", "X"],
                "capacity_mw": [10.0, 20.0],
                "connection_date": [None, None],
                "is_forecast": [True, True],
            },
            schema_overrides={"connection_date": pl.Date},
        ),
    )
    node = _frame(series=1, q={"q05": -70.0, "q95": -40.0}, actual=-65.0)
    generators = pl.concat(
        [
            _frame(series=101, q={"q50": 8.0}, actual=8.0),
            _frame(series=102, q={"q50": 18.0}, actual=12.0),
        ]
    )
    result = node_curtailment(
        table=table,
        node_forecasts={"made_up": node},
        generator_forecasts={"made_up": generators},
        assumptions=NodeCurtailmentAssumptions(),
    )
    row = result.per_arm.filter(pl.col("is_headline_price")).row(0, named=True)
    shares = allocate_pro_rata_by_capacity(
        frame=pl.DataFrame(
            {
                "group": [0, 0],
                "capacity_mw": [10.0, 20.0],
                "forecast_mwh": [4.0, 9.0],
                "need_mwh": [12.5, 12.5],
            }
        ),
        group_by=["group"],
        capacity_column="capacity_mw",
        forecast_mwh_column="forecast_mwh",
        need_mwh_column="need_mwh",
    )["allocated_mwh"].to_list()
    computed = {
        "Planned node curtailment (MWh)": row["planned_mwh"],
        "Needed (MWh)": row["needed_mwh"],
        "Generator A share, clamped at its forecast (MWh)": shares[0],
        "Generator B share, after the clamp (MWh)": shares[1],
        "Curtailed (MWh)": row["curtailed_mwh"],
        "Residual breach (MWh)": row["breach_mwh"],
        "Energy proxy (GBP)": row["energy_proxy_gbp"],
        "DCUSA form (GBP)": row["dcusa_form_gbp"],
    }
    return [
        {
            "quantity": name,
            "hand": HAND_ANSWER[name],
            "code": float(computed[name]),
            "agrees": abs(float(computed[name]) - HAND_ANSWER[name]) < 1e-6,
        }
        for name in HAND_ANSWER
    ]


def main() -> None:
    """Print the comparison, save it for the figure, and fail if the code disagrees."""
    rows = run()
    for row in rows:
        print(row)
    inputs.DATA_DIR.joinpath("results").mkdir(parents=True, exist_ok=True)
    (inputs.DATA_DIR / "results" / "known_answer.json").write_text(json.dumps(rows))
    if not all(r["agrees"] for r in rows):
        msg = "the code disagrees with the hand-worked answer"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
