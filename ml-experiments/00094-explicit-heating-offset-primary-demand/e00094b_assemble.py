"""Assemble block B's arm `amh` the way experiment 00064 assembles `am`.

Reads `work/b_arms_raw/amh_seed<s>_<site>.parquet` (see `e00094b_fit.py`), adds the 14-day
correction from the arm's own medians, applies the spread factor 1.15, joins the observed power,
and writes `work/b_arms/amh.parquet` (scored form, with `actual`). The reference `am` is
experiment 00064's stored scored form, read from its `work/arms/am.parquet`.

    uv run python ml-experiments/00094-*/e00094b_assemble.py
"""

import e00094b_common as common
import e00094b_fit as fitting
import polars as pl


def main() -> None:
    """Write the scored form of `amh`."""
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    arrays = common.load_series_arrays(series=series)
    actual = common.actual_table(series=series, arrays=arrays)
    pieces = []
    for seed in common.SEEDS:
        for site in common.SITES:
            path = fitting.RAW_DIR / f"amh_seed{seed}_{site}.parquet"
            pieces.append(
                pl.read_parquet(path)
                .filter(pl.col("variant") == "base")
                .select(*common.KEYS, "seed", *common.LEVEL_COLUMNS)
                .with_columns(pl.col("seed").cast(pl.Int8))
            )
    raw = pl.concat(pieces)
    corrected = common.with_corrections(base=raw, series=series, arrays=arrays)
    scored = common.stack(frame=corrected, correct=True, spread=True)
    scored = scored.filter(pl.col("valid_time") < common.HOLDOUT).join(
        actual, on=list(common.KEYS), how="left", validate="m:1"
    )
    target = fitting.B_WORK_DIR / "b_arms" / "amh.parquet"
    common.write_atomically(frame=scored, target=target)
    print(f"amh: {scored.height:,} rows", flush=True)


if __name__ == "__main__":
    main()
