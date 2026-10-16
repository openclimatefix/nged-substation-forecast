"""Block-bootstrap the arm vector of every real directional Primary-year.

Each draw resamples whole 7-day blocks of local dates, with replacement, until the draw has as many
blocks as the season-year, and refits the six arms and the daily fit. The 200 draws of a
Primary-year give both its noise (each vector minus the mean vector) and an interval for its
headline capacity.

    POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        uv run --with holidays python \
        ml-experiments/00040-pv-capacity-error-bars/e00040_bootstrap.py
"""

import e00040_arms as arms
import e00040_common as common
import e00040_inputs as inputs
import e00040_model as model
import numpy as np
import polars as pl

DAY_STRIDE = 100_000
"""Offset between the date codes of two copies of a block, so every resampled day is unique."""


def block_indices(*, days: np.ndarray) -> list[np.ndarray]:
    """Split row indices into blocks of `common.BLOCK_DAYS` consecutive local dates."""
    block = (days - days.min()) // common.BLOCK_DAYS
    order = np.argsort(block, kind="stable")
    cuts = np.nonzero(np.diff(block[order]))[0] + 1
    return np.split(order, cuts)


def resample(
    *, window: model.Window, blocks: list[np.ndarray], rng: np.random.Generator
) -> model.Window:
    """Return a window of whole blocks drawn with replacement, every copied day with a new code."""
    picks = rng.integers(0, len(blocks), size=len(blocks))
    index = np.concatenate([blocks[b] for b in picks])
    copy = np.concatenate([np.full(len(blocks[b]), k) for k, b in enumerate(picks)])
    drawn = window.take(mask=index)
    return model.Window(
        **{
            **{name: getattr(drawn, name) for name in drawn.__dataclass_fields__},
            "days": drawn.days + copy * DAY_STRIDE,
        }
    )


def main() -> None:
    """Write `bootstrap.parquet`."""
    prepared = inputs.load_prepared()
    primaries = inputs.primary_rows(prepared=prepared)
    out: list[dict[str, object]] = []
    for primary in common.DIRECTIONAL_PRIMARIES:
        for year, rows in inputs.season_year_rows(rows=primaries.filter(pl.col("site") == primary)):
            window = model.window_from(rows=rows)
            blocks = block_indices(days=window.days)
            rng = np.random.default_rng([common.BOOTSTRAP_SEED, int(primary[1:]), year])
            for draw in range(common.BOOTSTRAP_DRAWS):
                drawn = resample(window=window, blocks=blocks, rng=rng)
                vector = arms.arm_capacities(y=drawn.power, window=drawn)
                out.append(
                    {
                        "site": primary,
                        "year": year,
                        "draw": draw,
                        **dict(zip(common.ALL_FITS, vector.tolist(), strict=True)),
                    }
                )
        print(primary, "done", flush=True)
    pl.DataFrame(out).write_parquet(common.BOOTSTRAP_PATH)


if __name__ == "__main__":
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    main()
