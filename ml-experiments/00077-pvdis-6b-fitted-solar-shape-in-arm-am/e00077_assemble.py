"""Assemble the arms of experiment 00077 from the per-series fit files and copy the references.

For the solar shape named by `E77_ARM`, reads
`work/arm/<shape>/arms_raw/<arm>_seed<s>_<site>.parquet` (see `e00077_fit.py`), adds the 14-day
correction to the arm's own forecasts, applies the 1.15 spread factor, joins the observed power, and
writes under `work/arms/`:

- `am_<shape>.parquet` (and `am_blend_twin.parquet` for the blend's placebo twin): the scored form
  (correction plus spread), with `actual`;
- `<name>_c.parquet`: the correction without the spread factor;
- the scored form as a private prediction set of experiment 77, row set `fold_a_every5d_v1`.

The 14-day correction is the mean over the 14 days before the origin of the actual minus the arm's
own median, so it uses only power observed before the forecast origin.

`--references` copies the stored scored arms of the earlier experiments, restricted to the five
series and to valid times before the holdout, into `work/arms/` without a further correction: `am`,
`am_twin`, `d0`, `d1l`, `climatology`, and `manual_heuristic` (experiment 00064), `am0` (experiment
00076), and `am_sy`, `am_sy_twin`, and `d0_m0` (experiment 00078).

    E77_ARM=blend uv run python ml-experiments/00077-*/e00077_assemble.py
    uv run python ml-experiments/00077-*/e00077_assemble.py --references
"""

import argparse
from typing import Final

import e00077_common as common
import polars as pl

from studies import experiment_store

SHAPE_ARM_NAMES: Final[dict[str, str]] = {"am": f"am_{common.SHAPE_ARM}"}
if common.SHAPE_ARM == "blend":
    SHAPE_ARM_NAMES["am_twin"] = "am_blend_twin"
"""Each fit arm of this process's shape mapped to the name it is stored and scored under."""
SHARED_ARMS_DIR: Final = common.BASE_WORK_DIR / "arms"
REFERENCE_ARMS: Final[dict[str, str]] = {
    "am": "00064-pvdis-5-net-demand-as-demand-minus-forecast-solar",
    "am_twin": "00064-pvdis-5-net-demand-as-demand-minus-forecast-solar",
    "d0": "00064-pvdis-5-net-demand-as-demand-minus-forecast-solar",
    "d1l": "00064-pvdis-5-net-demand-as-demand-minus-forecast-solar",
    "climatology": "00064-pvdis-5-net-demand-as-demand-minus-forecast-solar",
    "manual_heuristic": "00064-pvdis-5-net-demand-as-demand-minus-forecast-solar",
    "am0": "00076-pvdis-5-leak-check-beyond-10-days",
    "am_sy": "00078-pvdis-5-current-capacity-vintages",
    "am_sy_twin": "00078-pvdis-5-current-capacity-vintages",
    "d0_m0": "00078-pvdis-5-current-capacity-vintages",
}
"""Each reference arm mapped to the folder, under `ML_EXPERIMENTS_DATA_DIR`, of its experiment."""


def raw_base(*, fit_arm: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return the variant `base` of one fit arm for every seed and site.

    Args:
        fit_arm: The arm in the file name.
        series: `series_table()` filtered to the five sites.

    Returns:
        Columns `KEYS`, `seed` (Int8), and the nine quantiles.

    Raises:
        FileNotFoundError: If a seed and site file is missing.
    """
    pieces = []
    for seed in common.SEEDS:
        for site in common.SITES:
            path = common.RAW_ARMS_DIR / f"{fit_arm}_seed{seed}_{site}.parquet"
            if not path.exists():
                msg = f"{path} is missing"
                raise FileNotFoundError(msg)
            pieces.append(
                pl.read_parquet(path)
                .filter(pl.col("variant") == "base")
                .select(*common.KEYS, "seed", *common.LEVEL_COLUMNS)
                .with_columns(pl.col("seed").cast(pl.Int8))
            )
    frame = pl.concat(pieces)
    assert frame["time_series_id"].n_unique() == series.height
    return frame


def copy_references(*, series: pl.DataFrame) -> None:
    """Copy each reference arm, restricted to the five series and valid times before the holdout."""
    SHARED_ARMS_DIR.mkdir(parents=True, exist_ok=True)
    for name, folder in REFERENCE_ARMS.items():
        target = SHARED_ARMS_DIR / f"{name}.parquet"
        if target.exists():
            continue
        source = common.ML_EXPERIMENTS_DATA_DIR / folder / "work" / "arms" / f"{name}.parquet"
        frame = (
            pl.read_parquet(source)
            .filter(
                pl.col("time_series_id").is_in(series["time_series_id"].implode())
                & (pl.col("valid_time") < common.HOLDOUT)
            )
            .select(*common.KEYS, "seed", *common.LEVEL_COLUMNS, "actual")
            .with_columns(pl.col("seed").cast(pl.Int8))
        )
        common.write_atomically(frame=frame, target=target)
        print(f"{name}: copied {frame.height:,} rows from {folder}", flush=True)


def main() -> None:
    """Write this shape's scored and correction-only forms and prediction sets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", action="store_true")
    args = parser.parse_args()
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    if args.references:
        copy_references(series=series)
        return
    SHARED_ARMS_DIR.mkdir(parents=True, exist_ok=True)
    arrays = common.load_series_arrays(series=series)
    actual = common.actual_table(series=series, arrays=arrays)
    for fit_arm, name in SHAPE_ARM_NAMES.items():
        target = SHARED_ARMS_DIR / f"{name}.parquet"
        if target.exists():
            continue
        raw = raw_base(fit_arm=fit_arm, series=series)
        corrected = common.with_corrections(base=raw, series=series, arrays=arrays)
        forms = {
            f"{name}_c": common.stack(frame=corrected, correct=True, spread=False),
            name: common.stack(frame=corrected, correct=True, spread=True),
        }
        # The holdout (valid times from 2026-07-01) is never stored or scored.
        forms = {k: v.filter(pl.col("valid_time") < common.HOLDOUT) for k, v in forms.items()}
        experiment_store.write_predictions(
            experiment_id=common.EXPERIMENT_ID,
            name=name,
            frame=forms[name],
            kind="quantiles",
            row_set_id=common.ROW_SET_ID,
        )
        for form, table in forms.items():
            joined = table.join(actual, on=list(common.KEYS), how="left", validate="m:1")
            common.write_atomically(frame=joined, target=SHARED_ARMS_DIR / f"{form}.parquet")
        print(
            f"{name}: {forms[name].height:,} rows, scored form and correction-only form", flush=True
        )


if __name__ == "__main__":
    main()
