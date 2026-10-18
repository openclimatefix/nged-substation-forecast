"""The two descriptive checks of experiment 00088: false alarms and the real events.

- `false`: 500 random (series, origin) pairs far from any logged closure, where no transfer
  exists. Arms 2 to 4 run with the causal XL MW on the unmodified power and are scored against
  the same power, so any rise over the ceiling is the harm of a false causal alarm.
- `real`: the conserved two-sided transfers of experiment 41 in the validation year whose two
  sides are each at most half missing. For each side the normal-running-arrangement (NRA) estimate
  during the transfer is its metered power plus the other side's labelled step: the donor adds
  the receivers' steps, and each receiver adds the donor's (negative) step times its share. Outside
  the transfer the target is the metered power. Arms 1, 2, and 4 with causal XL MW, and arm 3 with
  the oracle labels' MW, are scored on forecast origins inside the transfer. The truth is itself
  an estimate.

    uv run python ml-experiments/00088-*/e00088_controls.py false real
"""

import argparse
import json
from typing import Final

import e00088_cases as cases
import e00088_causal as causal
import e00088_common as common
import e00088_fit as fitting
import e00088_grid as grid
import e00088_harness as harness
import e00088_labels as labelling
import numpy as np
import polars as pl

N_FALSE: Final[int] = 500
SEED: Final[int] = 8801
CLEAR_STAMPS: Final[int] = 17 * grid.BLOCK
"""A false-alarm pair has no logged closure from 7 days before the lags to 10 days after."""
FALSE_VARIANTS: Final[tuple[tuple[str, str, str | None], ...]] = (
    ("ceiling", "plain", None),
    ("lagmw_xl_caus", "lagmw_xl", "caus"),
    ("nra_oracle_caus", "nra_oracle", "caus"),
    ("nra_xl_caus", "nra_xl", "caus"),
    ("lagmw_xl_cmax", "lagmw_xl", "cmax"),
    ("nra_xl_cmax", "nra_xl", "cmax"),
)
REAL_VARIANTS: Final[tuple[tuple[str, str, str | None], ...]] = (
    ("plain", "plain", None),
    ("lagmw_xl_caus", "lagmw_xl", "caus"),
    ("nra_xl_caus", "nra_xl", "caus"),
    ("nra_xl_cmax", "nra_xl", "cmax"),
    ("nra_oracle_orac", "nra_oracle", "orac"),
)
MAX_MISSING: Final[float] = 0.5


def false_alarms(*, state: dict) -> pl.DataFrame:
    """Score arms 2 to 4 with causal XL MW on origins with no transfer."""
    series, arrays, labels = fitting.context()
    rows_of = {int(r["time_series_id"]): r for r in series.iter_rows(named=True)}
    rng = np.random.default_rng(SEED)
    origins = common.origins_of_subset(subset="gap")
    sites = [rows_of[s]["site"] for s in grid.SERIES_IDS]
    out: list[dict] = []
    chosen = 0
    attempts = 0
    while chosen < N_FALSE and attempts < 50 * N_FALSE:
        attempts += 1
        origin = origins[int(rng.integers(0, len(origins)))]
        site = sites[int(rng.integers(0, len(sites)))]
        row = next(r for r in rows_of.values() if r["site"] == site)
        newest = cases.newest_stamp(origin=origin)
        span = slice(harness.OFFSET + newest - CLEAR_STAMPS, harness.OFFSET + newest + CLEAR_STAMPS)
        closed = (
            labels[site]["oracle"].closed[span]
            | labels[site]["oracle"].in_event[span]
            | labels[site]["xl"].in_event[span]
        )
        data = np.isfinite(arrays[site].power[span]).mean()
        if closed.any() or data < 0.9:
            continue
        chosen += 1
        cut = newest + 1
        intervals = causal.causal_intervals(
            state=state,
            residual=state["panel"]["residual"],
            offset=state["offset_panel"]["residual"],
            cut=cut,
        )
        signed = {
            "inj": np.zeros_like(arrays[site].power),
            "caus": cases.causal_array(
                intervals=intervals,
                series_id=int(row["time_series_id"]),
                capacity=row["effective_capacity_mw"],
                size=arrays[site].power.size,
            ),
            "cmax": cases.causal_array(
                intervals=intervals,
                series_id=int(row["time_series_id"]),
                capacity=row["effective_capacity_mw"],
                size=arrays[site].power.size,
                merge="max",
            ),
        }
        index = common.grid_index(times=origin) - common.NEWEST_AGE
        out.extend(
            {
                "site": site,
                "origin": origin,
                "caus_mw_newest": float(signed["caus"][index]),
                "n_intervals": sum(i.series == row["time_series_id"] for i in intervals),
                **cell,
            }
            for cell in cases.forecast_cell(
                site=site,
                origin=origin,
                power=arrays[site].power,
                signed=signed,
                variants=FALSE_VARIANTS,
            )
        )
    return pl.DataFrame(out, infer_schema_length=None)


def real_events() -> list[dict]:
    """Return the conserved two-sided transfers of the validation year with both sides present."""
    labelled = pl.read_parquet(labelling.LABELS_PATH).filter(
        pl.col("kind") == "real_operation",
        pl.col("visibility") == "conserved",
        pl.col("max_side_missing_share") <= MAX_MISSING,
        pl.col("onset_utc") >= common.FIRST_RUN,
        pl.col("end_utc") < common.LAST_RUN,
        pl.col("donor").is_not_null(),
    )
    return list(labelled.iter_rows(named=True))


def nra_truth(*, event: dict, side: int, rows_of: dict, arrays: dict) -> np.ndarray | None:
    """Return a side's target for one real transfer: the NRA estimate inside it, else metered.

    Args:
        event: A row of the labelled set.
        side: The side's time series id.
        rows_of: The Primary series rows by id.
        arrays: The series' arrays by site.

    Returns:
        The target over capacity on the dense grid, or None when the side is not a Primary.
    """
    row = rows_of.get(side)
    if row is None:
        return None
    moved = float(event["moved_mw_p50"])
    receivers = json.loads(event["receivers"] or "[]")
    if side == int(event["donor"]):
        step = moved
    else:
        share = next((float(r["share"]) for r in receivers if int(r["series"]) == side), None)
        if share is None:
            return None
        step = -moved * share
    truth = arrays[row["site"]].power.copy()
    first = max(int(np.ceil((event["onset_utc"] - common.GRID_START) / common.STEP)), 0)
    last = int(np.floor((event["end_utc"] - common.GRID_START) / common.STEP)) + 1
    truth[first:last] += np.float32(step / row["effective_capacity_mw"])
    return truth


def real_check(*, state: dict) -> pl.DataFrame:
    """Score arms 1, 2, 3, and 4 on origins inside the real transfers."""
    series, arrays, labels = fitting.context()
    rows_of = {int(r["time_series_id"]): r for r in series.iter_rows(named=True)}
    out: list[dict] = []
    origins = common.origins_of_subset(subset="gap")
    for event in real_events():
        first = grid.stamp_of(when=event["onset_utc"])
        last = grid.stamp_of(when=event["end_utc"]) + 1
        sides = {
            int(event["donor"]),
            *(int(r["series"]) for r in json.loads(event["receivers"] or "[]")),
        }
        for origin in origins:
            newest = cases.newest_stamp(origin=origin)
            if not (first <= newest < last and newest - first <= 7 * grid.BLOCK):
                continue
            intervals = causal.causal_intervals(
                state=state,
                residual=state["panel"]["residual"],
                offset=state["offset_panel"]["residual"],
                cut=newest + 1,
            )
            for side in sorted(sides):
                truth = nra_truth(event=event, side=side, rows_of=rows_of, arrays=arrays)
                if truth is None:
                    continue
                row = rows_of[side]
                site = row["site"]
                signed = {
                    "caus": cases.causal_array(
                        intervals=intervals,
                        series_id=side,
                        capacity=row["effective_capacity_mw"],
                        size=arrays[site].power.size,
                    ),
                    "cmax": cases.causal_array(
                        intervals=intervals,
                        series_id=side,
                        capacity=row["effective_capacity_mw"],
                        size=arrays[site].power.size,
                        merge="max",
                    ),
                    "orac": labels[site]["oracle"].signed_mw,
                    "inj": np.zeros_like(arrays[site].power),
                }
                out.extend(
                    {
                        "card_id": event["card_id"],
                        "site": site,
                        "side": "donor" if side == int(event["donor"]) else "receiver",
                        "origin": origin,
                        "hours": (newest - first) / 2,
                        "moved_mw": float(event["moved_mw_p50"]),
                        **cell,
                    }
                    for cell in cases.forecast_cell(
                        site=site,
                        origin=origin,
                        power=arrays[site].power,
                        signed=signed,
                        variants=REAL_VARIANTS,
                        truth=truth,
                    )
                )
    return pl.DataFrame(out, infer_schema_length=None)


def main() -> None:
    """Run the requested checks and write their cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checks", nargs="+", choices=("false", "real"))
    args = parser.parse_args()
    state = harness.make_state()
    cases.OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "false" in args.checks:
        table = false_alarms(state=state)
        common.atomic_write(frame=table, target=cases.OUT_DIR / "controls_false.parquet")
        print(f"false-alarm cells: {table.height:,}")
    if "real" in args.checks:
        table = real_check(state=state)
        common.atomic_write(frame=table, target=cases.OUT_DIR / "controls_real.parquet")
        print(f"real-event cells: {table.height:,}")
        print(f"events scored: {table['card_id'].n_unique() if table.height else 0}")


if __name__ == "__main__":
    main()
