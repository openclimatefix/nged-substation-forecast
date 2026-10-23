"""Where the net annual loss of the LR_b20 mask comes from: the alarms that cost the most.

For each series, scores K0 and the LR_b20 arm of every fold on every row. For each LR_b20 span it
sums the score difference of that span's own fold (weighted 1/|folds|) over the rows whose target
lies inside the span (from the step stamp to the end of the span's last known stamp plus 72 hours).
It lists the spans with the largest total, with the length of the span and whether a labelled load
shift or a logged closure overlaps it.

    uv run --no-sync python e00258_net_loss.py
"""

from __future__ import annotations

import e00258_harness as h
import e00258_score as sc
import numpy as np
import polars as pl


def main() -> None:
    """Write `net_loss_by_alarm.parquet` and print the largest contributors."""
    band = h.band_grid()
    events = h.labelled_events()
    rows = []
    for series in h.SERIES:
        data = h.load_series(series=series)
        stored = np.load(h.HARNESS_DIR / f"series_{series}.npz")
        folds = [int(f) for f in stored["folds"]]
        offsets = stored["offsets"]
        _, _, targets = h.origin_grid()
        clipped = np.minimum(targets, h.N_GRID - 1)
        k0 = sc.row_metrics(
            data=data, correction=stored["K0"].astype(float), offsets=offsets, band=band
        )["qs"]
        diff_of_fold = {}
        for fold in folds:
            arm = sc.row_metrics(
                data=data,
                correction=stored[f"LR_b20_f{fold}"].astype(float),
                offsets=offsets,
                band=band,
            )["qs"]
            diff_of_fold[fold] = (arm - k0) / len(folds)
        own = sc.closure_blocks(series=series, neighbours=False)
        spans = pl.read_parquet(h.HARNESS_DIR / f"spans_{series}.parquet").filter(
            pl.col("arm").str.starts_with("LR_b20")
        )
        mine = events.filter(pl.col("series").list.contains(series))
        for r in spans.iter_rows(named=True):
            diff = diff_of_fold[int(r["arm"].rsplit("_f", 1)[1])]
            lo, hi = r["onset"], r["known"] + sc.AFTER_HALF_HOURS
            inside = (targets >= lo) & (targets < hi) & np.isfinite(diff)
            labelled = any(e["onset"] < hi and e["end"] > lo for e in mine.iter_rows(named=True))
            rows.append(
                {
                    "series": series,
                    "arm": r["arm"],
                    "onset": r["onset"],
                    "span_days": (r["known"] - r["onset"]) / 48.0,
                    "excess_score_sum": float(np.nansum(diff[inside])),
                    "rows": int(inside.sum()),
                    "overlaps_logged_closure": bool(own[max(lo, 0) : min(hi, h.N_GRID)].any()),
                    "overlaps_labelled_load_shift": labelled,
                    "is_d8_series": series == sc.D8_SERIES,
                }
            )
        del clipped
    table = pl.DataFrame(rows)
    table.write_parquet(h.HARNESS_DIR / "net_loss_by_alarm.parquet")
    total = table["excess_score_sum"].sum()
    top = table.sort("excess_score_sum", descending=True).head(10)
    print(top)
    print("total excess over all spans:", total)
    print(
        table.group_by("overlaps_logged_closure", "overlaps_labelled_load_shift").agg(
            spans=pl.len(), excess=pl.col("excess_score_sum").sum()
        )
    )


if __name__ == "__main__":
    main()
