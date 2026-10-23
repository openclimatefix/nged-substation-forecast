"""Part (b) on the real record: the uses of an alarm for the carried sources at one substation.

Rows, K0 baseline, spread, and alarms are those of `e00227_run.py` (read from its stored
`scores_<site>` file, so the rows are identical). For each carried source and fold the arms are
`<source>|SOFT_2`, `|SOFT_4`, `|SOFT_16`, `|HALF`, `|SUB`, and `|PAIRSUM`; the MASK arm is the
stored part (a) arm of the same source. Writes `scores_b_<site>.parquet` and
`pairsum_counts_<site>.parquet` under the experiment's private folder.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00227_run_b.py D5
"""

from __future__ import annotations

import sys
import time
from typing import Final

import e00227_alarms as al
import e00227_events as ev
import e00227_fed as fed
import e00227_nra as nra
import e00227_uses as uses
import numpy as np
import polars as pl
from e00227_common import WORK_DIR
from e00227_g_arms import quantile_score

CARRIED: Final[tuple[str, ...]] = ("LR_b20", "LR_b80", "PG_g70")
"""The sources the part (a) carry-forward rule fixed before any part (b) score was read."""
EXTRA: Final[tuple[str, ...]] = ("RO_b80",)
"""RO-NC at 80 alarm-budget, run after the rest as the source with the most ordinary-day harm."""


def main(site: str, *, extra: bool = False) -> None:
    """Score every use of every carried source at one substation."""
    started = time.time()
    data = fed.get_site(site)
    n = len(data.stream)
    stored = pl.read_parquet(WORK_DIR / f"scores_{site}.parquet").filter(pl.col("arm") == "K0")
    rows = stored["row"].to_numpy().astype(np.int64)
    cutoffs = data.cutoffs(delay_hours=fed.DELAY_HOURS)[data.run_of_row]
    spread = fed.k0_spread(site)
    pair_gated = ev.alarm_tables()["GATED"]
    k0 = fed.correction_for(data=data, stream=data.stream, prior=np.zeros(n), multiplier=1.0)
    corrections: dict[tuple[str, int], np.ndarray] = {}
    counts = []
    for source in EXTRA if extra else CARRIED:
        for fold, table in al.tables(source=source).items():
            if al.is_foldless(source) and fold > 0:
                continue
            f = -1 if al.is_foldless(source) else fold
            spans = nra.detected_spans(data=data, site=site, intervals=table)
            kw = {"data": data, "stream": data.stream, "spans": spans, "cutoffs": cutoffs}
            masked = uses.mask(rows=rows, **kw)
            for factor in uses.SOFT_FACTORS:
                corrections[(f"{source}|SOFT_{int(factor)}", f)] = uses.soft(
                    rows=rows, factor=factor, **kw
                )
            corrections[(f"{source}|HALF", f)] = uses.half(k0=k0, masked=masked)
            corrections[(f"{source}|SUB", f)] = uses.sub(rows=rows, **kw)
            sub_spans, mask_spans, count = uses.pairsum_split(
                data=data, site=site, source_table=table, pair_gated=pair_gated
            )
            corrections[(f"{source}|PAIRSUM", f)] = uses.mixed_correction(
                data=data,
                stream=data.stream,
                sub_spans=sub_spans,
                mask_spans=mask_spans,
                cutoffs=cutoffs,
                rows=rows,
            )
            counts.append({"site": site, "source": source, "fold": f, **count})
            print(f"{site} {source} fold {f} done at {time.time() - started:.0f} s", flush=True)
    scores = []
    for (arm, fold), correction in corrections.items():
        q = fed.quantiles_at(data=data, correction=correction, spread=spread, rows=rows)
        scores.append(
            pl.DataFrame(
                {
                    "arm": arm,
                    "fold": fold,
                    "row": rows,
                    "qs": quantile_score(actual=data.actual[rows], quantiles=q).astype(np.float32),
                    "q50": q[:, 5].astype(np.float32),
                    "mean_forecast": (data.member_mean[rows] + correction[rows]).astype(np.float32),
                }
            )
        )
    pl.concat(scores).write_parquet(
        WORK_DIR / f"scores_b{'_extra' if extra else ''}_{site}.parquet"
    )
    pl.DataFrame(counts).write_parquet(
        WORK_DIR / f"pairsum_counts{'_extra' if extra else ''}_{site}.parquet"
    )
    print(f"{site}: done in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1], extra=len(sys.argv) > 2 and sys.argv[2] == "extra")
