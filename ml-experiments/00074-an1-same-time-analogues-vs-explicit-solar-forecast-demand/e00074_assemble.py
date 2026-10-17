"""Assemble the arms of experiment 00074 from the fit files and the reference arms.

Reads `work/arms_raw/<site>_s<seed>_<refit>.parquet` (see `e00074_fit.py`), checks that every
validation case of every series and seed appears exactly once per arm, applies the 1.15 spread
factor (no 14-day correction is applied), joins the observed power, and writes:

- `work/arms/<arm>.parquet`: the scored form with `actual`.
- the scored form as a private prediction set of experiment 74, row set `fold_a_every5d_v1`.

The context arms of experiment 00064 (`d0`, `d1l`, `am`, `climatology`, and `manual_heuristic`) are
copied, restricted to the five series, into `work/arms/` as `ref_<name>.parquet`. They are frozen
models, and `d0`, `d1l`, and `am` carry the 14-day correction, so they enter no contrast.

    uv run python ml-experiments/00074-*/e00074_assemble.py
"""

from typing import Final

import e00074_common as common
import polars as pl

from studies import experiment_store

E64_ARMS_DIR: Final = common.E64_WORK_DIR / "arms"
REFERENCE_ARMS: Final[tuple[str, ...]] = ("d0", "d1l", "am", "climatology", "manual_heuristic")


def read_raw() -> pl.DataFrame:
    """Return every fit file as one table.

    Returns:
        The concatenated fit files: `arm`, `seed`, `KEYS`, and the nine quantiles.

    Raises:
        FileNotFoundError: If no fit file exists.
    """
    files = sorted(common.RAW_ARMS_DIR.glob("*.parquet"))
    if not files:
        msg = f"no fit files in {common.RAW_ARMS_DIR}"
        raise FileNotFoundError(msg)
    return pl.concat([pl.read_parquet(path) for path in files])


def check_coverage(*, raw: pl.DataFrame, expected_cases: int) -> None:
    """Raise unless every arm, seed, and series holds each case exactly once.

    Args:
        raw: `read_raw()`.
        expected_cases: The number of distinct (series, origin, valid time) keys of the
            validation runs.

    Raises:
        ValueError: If a key is duplicated or an arm-seed pair misses cases.
    """
    counts = raw.group_by("arm", "seed").agg(
        rows=pl.len(), distinct=pl.struct(list(common.KEYS)).n_unique()
    )
    bad = counts.filter((pl.col("rows") != pl.col("distinct")) | (pl.col("rows") != expected_cases))
    if not bad.is_empty():
        msg = f"arms with duplicated or missing cases:\n{bad}"
        raise ValueError(msg)


def set_name(*, arm: str, seeds: list[int]) -> str:
    """Return the prediction set's name: the arm, with `-s012` once seeds 1 and 2 are added.

    A prediction set is immutable, so the set of seed 0 alone keeps the plain name and the set
    of all seeds is stored beside it.
    """
    suffix = "" if seeds == [0] else "-s" + "".join(str(s) for s in seeds)
    return arm.lower() + suffix


def main() -> None:
    """Write every arm's scored form and prediction set, and the context arms."""
    common.ARMS_DIR.mkdir(parents=True, exist_ok=True)
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    arrays = common.load_series_arrays(series=series)
    actual = common.actual_table(series=series, arrays=arrays)
    actual = actual.filter(pl.col("valid_time") < common.HOLDOUT)
    raw = read_raw().filter(pl.col("valid_time") < common.HOLDOUT)
    check_coverage(raw=raw, expected_cases=actual.height)
    for name in raw["arm"].unique().sort().to_list():
        frame = common.spread_quantiles(frame=raw.filter(pl.col("arm") == name))
        experiment_store.write_predictions(
            experiment_id=common.EXPERIMENT_ID,
            name=set_name(arm=name, seeds=sorted(frame["seed"].unique())),
            frame=frame,
            kind="quantiles",
            row_set_id=common.ROW_SET_ID,
        )
        joined = frame.join(actual, on=list(common.KEYS), how="left", validate="m:1")
        common.write_atomically(frame=joined, target=common.ARMS_DIR / f"{name}.parquet")
        print(f"{name}: {frame.height:,} rows over seeds {sorted(frame['seed'].unique())}")
    for name in REFERENCE_ARMS:
        frame = (
            pl.read_parquet(E64_ARMS_DIR / f"{name}.parquet")
            .filter(
                pl.col("time_series_id").is_in(series["time_series_id"].implode())
                & (pl.col("valid_time") < common.HOLDOUT)
            )
            .select(*common.KEYS, "seed", *common.LEVEL_COLUMNS, "actual")
            .with_columns(pl.col("seed").cast(pl.Int8))
        )
        common.write_atomically(frame=frame, target=common.ARMS_DIR / f"ref_{name}.parquet")
        print(f"ref_{name}: copied {frame.height:,} rows from experiment 64")


if __name__ == "__main__":
    main()
