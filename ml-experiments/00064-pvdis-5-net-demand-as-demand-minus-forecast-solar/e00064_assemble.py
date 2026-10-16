"""Assemble the arms of experiment 00064 from the per-series fit files and the reference arms.

Reads `work/arms_raw/<file arm>_seed<s>_<site>.parquet` (see `e00064_fit.py`), adds the 14-day
correction to every arm's own forecasts, applies the 1.15 spread factor, joins the observed power,
and writes:

- `work/arms/<arm>.parquet`: the scored form (correction plus spread), with `actual`.
- `work/arms/<arm>_c.parquet`: the correction without the spread factor.
- the scored form as a private prediction set of experiment 64, row set `fold_a_every5d_v1`.

The correction is the mean over the 14 days before the origin of the actual minus the arm's own
median, so for `b` it acts on the net residuals of the quantiles after the solar shape has been
subtracted. The arms are `sf`, `sf_member`, `sf_twin`, `sf_temp`, `a`, `a_twin`, `am`, `am_twin`,
`b`, `b_member`, and `b_twin`.

The reference arms are copied, restricted to the five series, into `work/arms/` without a
correction: `d0`, `d1l`, and `d1l_ens_shift` (experiment 00046's scored arms; the last is its
fixed-coefficient member shift with the 2023/24 capacity) and `climatology` and `manual_heuristic`
(experiment 00033's `_cs` arms, the correction and spread already applied).

    uv run python ml-experiments/00064-*/e00064_assemble.py
"""

from typing import Final

import e00064_common as common
import polars as pl

from studies import experiment_store

ARMS: Final[dict[str, tuple[str, str]]] = {
    "sf": ("d1l_shift", "sf"),
    "sf_member": ("d1l_shift", "sf_member"),
    "sf_twin": ("d1l_shift", "sf_twin"),
    "sf_temp": ("d1l_shift", "sf_temp"),
    "a": ("a", "base"),
    "a_twin": ("a_twin", "base"),
    "am": ("am", "base"),
    "am_twin": ("am_twin", "base"),
    "b": ("b", "base"),
    "b_member": ("b", "member"),
    "b_twin": ("b", "twin"),
}
"""Each new arm's name mapped to the fit file's arm and the variant it reads."""
REFERENCE_ARMS: Final[dict[str, tuple[str, str]]] = {
    "d0": ("e46", "d0"),
    "d1l": ("e46", "d1l"),
    "d1l_ens_shift": ("e46", "d1l_ens_shift"),
    "climatology": ("e33", "climatology_cs"),
    "manual_heuristic": ("e33", "manual_heuristic_cs"),
}
"""Each reference arm's name mapped to its source experiment and stored file."""


def raw_variant(*, fit_arm: str, variant: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return one variant of one fit arm for every seed and site.

    Args:
        fit_arm: The arm in the file name.
        variant: The variant name in the fit files.
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
                .filter(pl.col("variant") == variant)
                .select(*common.KEYS, "seed", *common.LEVEL_COLUMNS)
                .with_columns(pl.col("seed").cast(pl.Int8))
            )
    frame = pl.concat(pieces)
    assert frame["time_series_id"].n_unique() == series.height
    return frame


def copy_reference(*, name: str, series: pl.DataFrame) -> None:
    """Copy a reference arm, restricted to the five series and valid times before the holdout."""
    source, stem = REFERENCE_ARMS[name]
    folder = common.E46_ARMS_DIR if source == "e46" else common.E33_ARMS_DIR
    target = common.ARMS_DIR / f"{name}.parquet"
    if target.exists():
        return
    frame = (
        pl.read_parquet(folder / f"{stem}.parquet")
        .filter(
            pl.col("time_series_id").is_in(series["time_series_id"].implode())
            & (pl.col("valid_time") < common.HOLDOUT)
        )
        .select(*common.KEYS, "seed", *common.LEVEL_COLUMNS, "actual")
        .with_columns(pl.col("seed").cast(pl.Int8))
    )
    common.write_atomically(frame=frame, target=target)
    print(f"{name}: copied {frame.height:,} rows from {source}", flush=True)


def main() -> None:
    """Write every arm's scored and correction-only forms and its prediction set."""
    common.ARMS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    arrays = common.load_series_arrays(series=series)
    actual = common.actual_table(series=series, arrays=arrays)
    for name in REFERENCE_ARMS:
        copy_reference(name=name, series=series)
    for name, (fit_arm, variant) in ARMS.items():
        target = common.ARMS_DIR / f"{name}.parquet"
        if target.exists():
            continue
        raw = raw_variant(fit_arm=fit_arm, variant=variant, series=series)
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
            common.write_atomically(frame=joined, target=common.ARMS_DIR / f"{form}.parquet")
        print(
            f"{name}: {forms[name].height:,} rows, scored form and correction-only form", flush=True
        )


if __name__ == "__main__":
    main()
