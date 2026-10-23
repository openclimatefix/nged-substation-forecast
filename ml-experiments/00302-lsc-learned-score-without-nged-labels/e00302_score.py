# ruff: noqa: E501
"""Thresholds and alarms of L0 to L3 on the real record, and matches of the 35 logged load shifts.

Reads the real-record scores written by `e00302_fit.py` (L1, L2) and experiment 227's stored scores
(L0). Writes `alarms_<arm>_f<k>.parquet` (the flagged candidates), `shift_hits.parquet` (one row per
arm and load shift), `alarm_counts.parquet` (merged alarms per arm, fold, and series), and
`thresholds.json`. The 35 logged load shifts are scored last and choose nothing.

    OMP_NUM_THREADS=1 uv run --no-sync python e00302_score.py
"""

from __future__ import annotations

import json
from typing import Final

import e00302_rule as rule
import polars as pl
from e00302_common import E195_DIR, E227_DIR, WORK_DIR, pair_folds, real_transfers

FOLDS: Final[tuple[int, ...]] = (0, 1, 2)
FIXED_RATES: Final[tuple[int, ...]] = (5, 2)
"""Alarms per Primary-year for the arms that read no label at all."""
KEY: Final[list[str]] = ["series", "stamp", "source"]
ARMS_LABELLED: Final[tuple[str, ...]] = ("L0", "L1")


def real_scores(*, arm: str, fold: int, cands: pl.DataFrame) -> pl.DataFrame:
    """Scored real candidates of one arm and fold, with the opposite-step counts."""
    if arm == "L0":
        table = pl.read_parquet(E227_DIR / f"scores_f{fold}.parquet").filter(
            pl.col("arm") == "R re-ranker"
        )
    else:
        table = pl.read_parquet(WORK_DIR / f"real_scores_{arm}_f{fold}.parquet")
    extra = cands.select(*KEY, "n_opposite", "opp_ratio")
    drop = [c for c in ("n_opposite", "opp_ratio") if c in table.columns]
    return table.drop(drop).join(extra, on=KEY, how="left", validate="1:1")


def shifts_table() -> pl.DataFrame:
    """The 35 logged load shifts lasting over 1 day, with fold, stratum, and moved megawatts."""
    folds = pair_folds()
    t = real_transfers().filter(pl.col("hours") > 24)
    return t.with_columns(
        fold=pl.col("pair").replace_strict(folds, return_dtype=pl.Int64),
        long_deep=(pl.col("hours") >= 48) & (pl.col("share") >= 0.10),
        long=pl.col("hours") >= 48,
    )


def main() -> None:
    """Compute thresholds, alarms, and matches for every arm."""
    cands = pl.read_parquet(E195_DIR / "real_candidates.parquet")
    primaries = rule.primary_table()
    total_years = float(primaries["years"].sum())
    shifts = shifts_table()
    thresholds: dict[str, dict] = {}
    hits_rows, count_rows = [], []
    for fold in FOLDS:
        tables = {a: real_scores(arm=a, fold=fold, cands=cands) for a in ("L0", "L1", "L2")}
        series_ids = sorted({int(s) for s in tables["L0"]["series"].to_list()})
        thr: dict[str, dict[int, float]] = {}
        t0 = rule.budget_threshold(table=tables["L0"], held_out_odd=True)
        thr["L0"] = dict.fromkeys(series_ids, t0)
        thr["L1"] = dict.fromkeys(
            series_ids, rule.budget_threshold(table=tables["L1"], held_out_odd=False)
        )
        flag0 = rule.flagged_by(table=tables["L0"], thresholds=thr["L0"])
        n0 = rule.merged_count(
            series=tables["L0"]["series"].to_numpy()[flag0],
            stamp=tables["L0"]["stamp"].to_numpy()[flag0],
        )
        l2 = tables["L2"]
        thr["L2"] = dict.fromkeys(
            series_ids,
            rule.threshold_for_count(
                score=l2["score"].to_numpy(),
                series=l2["series"].to_numpy(),
                stamp=l2["stamp"].to_numpy(),
                target=n0,
            ),
        )
        thr["L3"] = rule.per_primary_thresholds(table=l2, total=n0, primaries=primaries)
        tables["L3"] = l2
        for r in FIXED_RATES:
            name = f"L3_{r}"
            thr[name] = rule.per_primary_thresholds(
                table=l2, total=r * total_years, primaries=primaries
            )
            tables[name] = l2
        thresholds[str(fold)] = {
            "L0_alarm_count": n0,
            **{a: {str(s): v for s, v in d.items()} for a, d in thr.items()},
        }
        events = shifts.filter(pl.col("fold") == fold)
        for arm, d in thr.items():
            table = tables[arm]
            flagged = rule.flagged_by(table=table, thresholds=d)
            table.filter(pl.Series(flagged)).write_parquet(
                WORK_DIR / f"alarms_{arm}_f{fold}.parquet"
            )
            confirmed = flagged & (table["n_opposite"].fill_null(0).to_numpy() >= 1)
            for variant, fl in (("all", flagged), ("confirmed", confirmed)):
                hits = rule.real_hits(table=table, flagged=fl, events=events).with_columns(
                    arm=pl.lit(arm), fold=pl.lit(fold), variant=pl.lit(variant)
                )
                hits_rows.append(hits)
                sub = table.filter(pl.Series(fl))
                for s in series_ids:
                    m = sub["series"].to_numpy() == s
                    count_rows.append(
                        {
                            "arm": arm,
                            "fold": fold,
                            "variant": variant,
                            "series": s,
                            "alarms": rule.merged_count(
                                series=sub["series"].to_numpy()[m], stamp=sub["stamp"].to_numpy()[m]
                            ),
                        }
                    )
            print(
                f"fold {fold} {arm}: flagged {int(flagged.sum())}, merged alarms "
                f"{rule.merged_count(series=table['series'].to_numpy()[flagged], stamp=table['stamp'].to_numpy()[flagged])}, "
                f"found {int(hits_rows[-2]['found'].sum())} of {events.height}",
                flush=True,
            )
    all_hits = pl.concat(hits_rows).join(
        shifts.select("card_id", "hours", "share", "moved_mw_p50", "pair", "long_deep", "long"),
        on="card_id",
    )
    all_hits.write_parquet(WORK_DIR / "shift_hits.parquet")
    pl.DataFrame(count_rows).join(
        primaries.select("series", "site", "years"), on="series", how="left"
    ).write_parquet(WORK_DIR / "alarm_counts.parquet")
    (WORK_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=1, default=float))
    summary = (
        all_hits.filter(pl.col("variant") == "all")
        .group_by("arm")
        .agg(found=pl.col("found").sum(), onset_found=pl.col("onset_found").sum())
        .sort("arm")
    )
    print(summary)


if __name__ == "__main__":
    main()
