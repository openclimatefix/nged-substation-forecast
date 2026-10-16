"""Tier B contrasts on the real stack: month-resampled intervals, three seeds pooled."""

import polars as pl
from e00038_analyse import contrast_row
from e00038_base import WORK_DIR

CONTRASTS = (
    ("d1_mask7 vs plain7", "d1_mask7", "plain7"),
    ("d1_mask14 vs plain14", "d1_mask14", "plain14"),
    ("oracle_conf14 vs plain14", "oracle_conf14", "plain14"),
    ("oracle_logged14 vs plain14", "oracle_logged14", "plain14"),
    ("random14 vs plain14", "random14", "plain14"),
    ("plain7 vs plain14", "plain7", "plain14"),
    ("post hoc: d1fx_mask7 vs plain7", "d1fx_mask7", "plain7"),
    ("post hoc: d1fx_mask14 vs plain14", "d1fx_mask14", "plain14"),
    ("post hoc: randomfx14 vs plain14", "randomfx14", "plain14"),
    ("d1_persist14 vs plain14 (all rows)", "d1_persist14", "plain14"),
    ("post hoc: d1fx_persist14 vs plain14 (all rows)", "d1fx_persist14", "plain14"),
)


def main() -> None:
    """Write `tier_b_contrasts.parquet` and print the numbers the write-up quotes."""
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(220)
    rows = pl.read_parquet(WORK_DIR / "tier_b_rows.parquet")
    out = []
    for label, arm, ref in CONTRASTS:
        views = (
            {"all": rows}
            if "persist" in arm
            else {
                "out_of_event": rows.filter("out_of_event"),
                "in_confirmed": rows.filter("in_confirmed"),
                "all": rows,
            }
        )
        out.extend(
            contrast_row(rows=subset, metric=metric, label=label, arm=arm, ref=ref, view=view)
            for view, subset in views.items()
            for metric in ("tw", "qs")
        )
    table = pl.DataFrame(out)
    table.write_parquet(WORK_DIR / "tier_b_contrasts.parquet")
    show = [
        "view",
        "metric",
        "contrast",
        "n_rows",
        "reference_mean",
        "arm_mean",
        "change_pct",
        "lo",
        "hi",
        "without_largest_pct",
        "largest_month",
    ]
    print(table.select(show))


if __name__ == "__main__":
    main()
