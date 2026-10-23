"""Recall on injected load shifts of held-out Primary pairs, at each arm's real-record thresholds.

The swapped eval worlds inject in even months (the months L1 to L3 did not train on). A threshold
is a score on the real record, so it applies to injected candidates unchanged.

    OMP_NUM_THREADS=1 uv run --no-sync python e00302_inj.py
"""

from __future__ import annotations

import json

import e00302_rule as rule
import polars as pl
from e00302_common import WORK_DIR
from e00302_data import KINDS

SCORE_OF: dict[str, str] = {"L1": "L1", "L2": "L2", "L3": "L2", "L3_5": "L2", "L3_2": "L2"}


def main() -> None:
    """Score every arm on the evaluation worlds of every fold."""
    thresholds = json.loads((WORK_DIR / "thresholds.json").read_text())
    rows = []
    for fold in range(3):
        for arm, model in SCORE_OF.items():
            table = pl.read_parquet(WORK_DIR / f"inj_scores_{model}_f{fold}.parquet")
            thr = {int(s): v for s, v in thresholds[str(fold)][arm].items()}
            flagged = rule.flagged_by(table=table, thresholds=thr)
            for kind in ("eval_held", "eval_fitted"):
                events = pl.concat(
                    [
                        pl.read_parquet(
                            WORK_DIR / "worlds" / f"f{fold}_{kind}_{i}_events.parquet"
                        ).with_columns(world=pl.lit(i))
                        for i in range(KINDS[kind][0])
                    ]
                )
                for i in range(KINDS[kind][0]):
                    mask = ((table["world"] == i) & (table["kind"] == kind)).to_numpy()
                    hits = rule.injected_hits(
                        table=table.filter(pl.Series(mask)),
                        flagged=flagged[mask],
                        events=events.filter(pl.col("world") == i),
                    )
                    rows.append(
                        hits.join(
                            events.filter(pl.col("world") == i).select(
                                "event_id", "hours", "depth_z", "n_receivers"
                            ),
                            on="event_id",
                        ).with_columns(
                            arm=pl.lit(arm), kind=pl.lit(kind), fold=pl.lit(fold), world=pl.lit(i)
                        )
                    )
            print(f"fold {fold} {arm} injected done", flush=True)
    out = pl.concat(rows)
    out.write_parquet(WORK_DIR / "injected_hits.parquet")
    print(
        out.group_by("arm", "kind")
        .agg(recall=pl.col("found").mean(), onset=pl.col("onset_found").mean(), n=pl.len())
        .sort("kind", "arm")
    )


if __name__ == "__main__":
    main()
