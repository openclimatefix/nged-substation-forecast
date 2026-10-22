# ruff: noqa: E501
"""Which logged load shifts each alarm source finds, and where each mask sits against the true one.

Two tables, both from the alarm interval tables alone (no forecast is refitted):

- `found35`: experiment 195's count of the 35 labelled real load shifts lasting over 1 day that an
  alarm source finds (an alarm on one of the load shift's series with its step within 2 hours of
  the logged onset and its estimated end within the larger of 6 hours and 10% of the duration of
  the logged end). A learned source uses the fold that holds out the load shift's Primary pair.
- `per_transfer`: for each of experiment 163's logged load shifts at the six decision substations
  and each source, when the first matching alarm is known, the onset-stamp error, how many hours
  the mask covers inside and outside the logged interval, and how many logged hours it never
  covers. The mask covers readings from the alarm's step stamp to its estimated end.

    OMP_NUM_THREADS=1 uv run --no-sync python e00226_decompose.py
"""

from __future__ import annotations

from typing import Final

import e00226_alarms as al
import e00226_events as ev
import e00226_fed as fed
import polars as pl
from e00226_common import WORK_DIR

ONSET_TOLERANCE: Final[int] = 4
"""Half-hours between an alarm's step and the logged onset for the alarm to find the load shift."""
END_MINIMUM: Final[int] = 12
END_FRACTION: Final[float] = 0.10
SOURCES: Final[tuple[str, ...]] = (
    "PAIR_NEW",
    "PAIR_GATED",
    "RONC_195",
    "LEARNED_F",
    "LEARNED_R",
    "U1",
    "U2",
    "U3",
    "U4",
)
WINDOW: Final[int] = 144
"""Half-hours either side of a logged interval inside which an alarm counts as a mask on it."""


def source_tables() -> dict[str, dict[int, pl.DataFrame]]:
    """Each source's interval tables by fold (the pair alarm has one table for every fold)."""
    pair = ev.alarm_tables()
    out: dict[str, dict[int, pl.DataFrame]] = {
        "PAIR_NEW": dict.fromkeys(range(al.N_FOLDS), pair["NEW"]),
        "PAIR_GATED": dict.fromkeys(range(al.N_FOLDS), pair["GATED"]),
    }
    for arm in ("RONC_195", "LEARNED_F", "LEARNED_R", "U1", "U2", "U3", "U4"):
        out[arm] = al.learned_tables(arm=arm)
    return out


def end_tolerance(*, onset: int, end: int) -> float:
    """The larger of 6 hours and 10% of the duration, in half-hours."""
    return max(float(END_MINIMUM), END_FRACTION * (end - onset))


def union_length(*, spans: list[tuple[int, int]]) -> int:
    """Total length of the union of half-open intervals."""
    total, last_end = 0, None
    for lo, hi in sorted(spans):
        if last_end is None or lo > last_end:
            total += hi - lo
            last_end = hi
        elif hi > last_end:
            total += hi - last_end
            last_end = hi
    return total


def overlap(*, spans: list[tuple[int, int]], lo: int, hi: int) -> int:
    """Length of the union of `spans` that lies inside [lo, hi)."""
    return union_length(
        spans=[(max(a, lo), min(b, hi)) for a, b in spans if min(b, hi) > max(a, lo)]
    )


def found35(*, tables: dict[str, dict[int, pl.DataFrame]]) -> pl.DataFrame:
    """One row per (source, labelled real load shift over 1 day): found, onset found, onset error."""
    folds = al.pair_folds()
    results = al.REAL_RESULTS
    events = (
        pl.read_parquet(results)
        .filter((pl.col("setting") == "XL") & (pl.col("hours") > 24))
        .select("card_id", "onset", "end", "hours", pair=pl.col("series"))
    )
    rows = []
    for e in events.iter_rows(named=True):
        series = [int(s) for s in e["pair"].split(",")]
        for source, by_fold in tables.items():
            table = by_fold[folds[e["pair"]]]
            onset_found, found, errors = False, False, []
            for r in table.filter(pl.col("series").is_in(series)).iter_rows(named=True):
                gap = abs(r["onset"] - e["onset"])
                if gap <= 48:
                    errors.append(gap)
                if gap <= ONSET_TOLERANCE:
                    onset_found = True
                    if abs(r["end"] - e["end"]) <= end_tolerance(onset=e["onset"], end=e["end"]):
                        found = True
            rows.append(
                {
                    "source": source,
                    "card_id": e["card_id"],
                    "hours": e["hours"],
                    "onset_found": onset_found,
                    "found": found,
                    "onset_error_h": min(errors) / 2 if errors else None,
                }
            )
    return pl.DataFrame(rows)


def per_transfer(*, tables: dict[str, dict[int, pl.DataFrame]]) -> pl.DataFrame:
    """One row per (source, logged load shift, decision substation)."""
    folds = al.card_folds()
    rows = []
    for site, series in fed.SERIES_OF_SITE.items():
        logged = ev.logged_transfers().filter(pl.col("series").list.contains(series))
        for r in logged.iter_rows(named=True):
            lo, hi = r["onset"], r["end"]
            for source, by_fold in tables.items():
                table = by_fold[folds.get(r["card"], 0)].filter(pl.col("series") == series)
                near = table.filter(
                    (pl.col("end") > lo - WINDOW) & (pl.col("onset") < hi + WINDOW)
                ).sort("onset")
                match = near.filter(
                    (pl.col("onset") >= lo - ev.MATCH_BEFORE) & (pl.col("onset") < hi)
                )
                spans = [(a["onset"], a["end"]) for a in near.iter_rows(named=True)]
                masked = union_length(spans=spans)
                inside = overlap(spans=spans, lo=lo, hi=hi)
                first = match.row(0, named=True) if match.height else None
                rows.append(
                    {
                        "source": source,
                        "site": site,
                        "card": r["card"],
                        "true_hours": (hi - lo) / 2,
                        "n_alarms_near": near.height,
                        "matched": first is not None,
                        "onset_error_h": None if first is None else (first["onset"] - lo) / 2,
                        "known_after_onset_h": None
                        if first is None
                        else (first["onset_known"] - lo) / 2,
                        "known_minus_end_h": None
                        if first is None
                        else (first["onset_known"] - hi) / 2,
                        "mask_end_error_h": None if first is None else (first["end"] - hi) / 2,
                        "masked_hours": masked / 2,
                        "masked_inside_h": inside / 2,
                        "masked_outside_h": (masked - inside) / 2,
                        "true_hours_never_masked": ((hi - lo) - inside) / 2,
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def main() -> None:
    """Write and print both tables."""
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    tables = source_tables()
    f35 = found35(tables=tables)
    f35.write_parquet(WORK_DIR / "found35.parquet")
    print("== found of 35 logged load shifts over 1 day (195's rule) ==")
    print(
        f35.group_by("source", maintain_order=True).agg(
            found=pl.col("found").sum(),
            onset_found=pl.col("onset_found").sum(),
            n=pl.len(),
            median_onset_error_h=pl.col("onset_error_h").filter(pl.col("found")).median(),
        )
    )
    pt = per_transfer(tables=tables)
    pt.write_parquet(WORK_DIR / "decompose.parquet")
    print("\n== per logged load shift and source (hours) ==")
    print(
        pt.filter(pl.col("source").is_in(["PAIR_NEW", "LEARNED_R", "U1"])).sort(
            "site", "card", "source"
        )
    )


if __name__ == "__main__":
    main()
