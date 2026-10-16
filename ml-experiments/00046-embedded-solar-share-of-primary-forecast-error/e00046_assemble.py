"""Assemble the arms of experiment 00046 from the per-site fit files.

Reads `work/arms_raw/<arm>_seed<s>_<site>.parquet` (see `e00046_fit.py`), adds the 14-day correction
to every arm's own forecasts, applies the 1.15 spread factor, joins the observed power, and writes:

- `work/arms/<arm>.parquet`: the scored form (correction plus spread), with `actual`.
- `work/arms/<arm>_c.parquet`: the correction without the spread factor.
- the scored form as a private prediction set of experiment 46, row set `fold_a_every5d_v1`.

The arms are `d0`, `d1l`, the two planned shifted arms, the two headline-capacity sensitivity arms,
and the exploratory ensemble-mean-only shift.

    uv run python ml-experiments/00046-*/e00046_assemble.py
"""

from typing import Final

import e00046_common as common
import polars as pl

from studies import experiment_store

ARMS: Final[dict[str, tuple[str, str]]] = {
    "d0": ("d0", "base"),
    "d1l": ("d1l", "base"),
    "d1l_ens_shift": ("d1l", "ens_shift"),
    "d1l_cams_shift": ("d1l", "cams_shift"),
    "d1l_ens_shift_headline": ("d1l", "ens_shift_headline"),
    "d1l_cams_shift_headline": ("d1l", "cams_shift_headline"),
    "d1l_ens_mean_shift": ("d1l", "ens_mean_shift"),
}
"""Each arm's name mapped to the fit arm and the variant it reads."""


def raw_variant(*, fit_arm: str, variant: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return one variant of one fit arm for every seed and site.

    Args:
        fit_arm: `d0` or `d1l`.
        variant: The variant name in the fit files.
        series: `series_table()` filtered to the six sites.

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


def main() -> None:
    """Write every arm's scored and correction-only forms and its prediction set."""
    common.ARMS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    arrays = common.load_series_arrays(series=series)
    actual = common.actual_table(series=series, arrays=arrays)
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
        for form, table in forms.items():
            joined = table.join(actual, on=list(common.KEYS), how="left", validate="m:1")
            common.write_atomically(frame=joined, target=common.ARMS_DIR / f"{form}.parquet")
        experiment_store.write_predictions(
            experiment_id=common.EXPERIMENT_ID,
            name=name,
            frame=forms[name],
            kind="quantiles",
            row_set_id=common.ROW_SET_ID,
        )
        print(
            f"{name}: {forms[name].height:,} rows, scored form and correction-only form", flush=True
        )


if __name__ == "__main__":
    main()
