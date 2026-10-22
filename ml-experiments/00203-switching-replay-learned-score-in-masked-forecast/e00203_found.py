"""Which logged transfers each alarm arm finds, and how long after the step it is known.

A transfer at a decision substation is found by an arm when an alarm of the arm on the substation's
series has its step from 24 hours before the logged onset to the logged end (experiment 156's
rule, `e00203_events.first_known`). Learned arms use the fold that holds out the transfer's pair.

    OMP_NUM_THREADS=1 uv run --no-sync python e00203_found.py
"""

from __future__ import annotations

import e00203_alarms as al
import e00203_events as ev
import e00203_fed as fed
import polars as pl


def main() -> None:
    """Print one row per (decision substation, logged transfer) with each arm's known-minus-end."""
    folds = al.card_folds()
    tables = {arm: al.learned_tables(arm=arm) for arm in al.ARMS}
    pair = ev.alarm_tables()
    rows = []
    for site, series in fed.SERIES_OF_SITE.items():
        logged = ev.logged_transfers().filter(pl.col("series").list.contains(series))
        for r in logged.iter_rows(named=True):
            row = {
                "site": site,
                "card": r["card"][:6],
                "hours": round((r["end"] - r["onset"]) / 2),
            }
            for name, table in (("PAIR_NEW", pair["NEW"]), ("RONC_147", pair["RONC"])):
                k = ev.first_known(table=table, series=series, onset=r["onset"], end=r["end"])
                row[name] = None if k is None else k - r["end"]
            for arm, by_fold in tables.items():
                fold = folds.get(r["card"], -1)
                k = (
                    None
                    if fold < 0
                    else ev.first_known(
                        table=by_fold[fold], series=series, onset=r["onset"], end=r["end"]
                    )
                )
                row[arm] = None if k is None else k - r["end"]
            rows.append(row)
    out = pl.DataFrame(rows, infer_schema_length=None).sort("site", "card")
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    print("Known stamp minus logged end, in half-hours (negative: known before the transfer ended)")
    print(out)
    print(
        "found (not null):",
        {
            c: int(out[c].is_not_null().sum())
            for c in out.columns
            if c not in ("site", "card", "hours")
        },
    )
    print(
        "known before the end:",
        {c: int((out[c] < 0).sum()) for c in out.columns if c not in ("site", "card", "hours")},
    )


if __name__ == "__main__":
    main()
