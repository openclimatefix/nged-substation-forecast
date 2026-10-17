"""Fit the naive and unfolding models to the real series, with a week-block bootstrap.

The 8 MVA-only Primaries are the target. The 8 MW Primaries are fitted too, with their sign hidden
and no injection, to give the false reverse flow and the natural reverse flow. Each window's
bootstrap resamples whole 7-day blocks of days with replacement and refits.

    POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 uv run --with holidays python \
        ml-experiments/00079-pvdis-7-mw-at-mva-only-primaries/e00079_real.py
"""

from multiprocessing import get_context

import e00079_common as common
import e00079_fit as fit
import e00079_inputs as inputs
import numpy as np
import polars as pl

_WINDOWS: dict[str, list[inputs.YearWindow]] = {}


def _fit_row(*, w: inputs.YearWindow, index: np.ndarray) -> dict[str, float | int]:
    """Fit both models on the rows `index` of a window and return the summary."""
    design = fit.Design.build(cells=w.cells[index], nuisance=w.nuisance[index])
    a = np.abs(w.power[index])
    u = w.shape[index]
    c_naive, m_naive = fit.naive_fit(design=design, a=a, u=u)
    lo, hi = common.MIDDAY_HALF_HOURS
    hh = w.half_hour[index]
    scale = float(np.median(a[(hh >= lo) & (hh < hi)]))
    unfolded = fit.unfold_fit(design=design, a=a, u=u, scale=scale)
    loss_naive = fit.abs_loss(a=a, m=m_naive)
    return {
        "c_naive": c_naive,
        "c_unfold": unfolded.capacity,
        "negative_share": float((unfolded.mean < 0.0).mean()),
        "loss_naive": loss_naive,
        "loss_unfold": unfolded.loss,
        "loss_reduction": 1.0 - unfolded.loss / loss_naive if loss_naive > 0 else 0.0,
        "n_distinct": unfolded.n_starts_distinct,
        "loss_gap": unfolded.loss_gap,
    }


def _job(primary: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return the point fits and the bootstrap fits of one Primary."""
    point: list[dict[str, object]] = []
    boot: list[dict[str, object]] = []
    for w in _WINDOWS[primary]:
        everything = np.arange(len(w.power))
        scale = float(np.median(np.abs(w.power)))
        meta = {"site": primary, "year": w.year, "rows": len(w.power), "scale": scale}
        point.append({**meta, **_fit_row(w=w, index=everything)})
        blocks = (w.days - w.days.min()) // common.BLOCK_DAYS
        labels = np.unique(blocks)
        rows_of = {int(b): np.flatnonzero(blocks == b) for b in labels}
        rng = np.random.default_rng([common.BOOTSTRAP_SEED, int(w.year), int(primary[1:])])
        for draw in range(common.WEEK_BOOTSTRAP_DRAWS):
            picks = rng.choice(labels, size=len(labels), replace=True)
            index = np.concatenate([rows_of[int(b)] for b in picks])
            boot.append({**meta, "draw": draw, **_fit_row(w=w, index=index)})
    return point, boot


def main() -> None:
    """Fit every window of the 16 Primaries and write `real.parquet` and `real_boot.parquet`."""
    global _WINDOWS  # noqa: PLW0603
    prepared = inputs.load_prepared()
    sites = (*common.MVA_PRIMARIES, *common.MW_PRIMARIES)
    _WINDOWS = {s: inputs.year_windows(prepared=prepared, primary=s) for s in sites}
    with get_context("fork").Pool(4) as pool:
        results = pool.map(_job, sites)
    unit = {s: ("MVA" if s in common.MVA_PRIMARIES else "MW") for s in sites}
    point = pl.DataFrame([r for p, _ in results for r in p]).with_columns(
        units=pl.col("site").replace_strict(unit)
    )
    boot = pl.DataFrame([r for _, b in results for r in b]).with_columns(
        units=pl.col("site").replace_strict(unit)
    )
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    point.write_parquet(common.REAL_PATH)
    boot.write_parquet(common.REAL_BOOT_PATH)
    print(
        point.group_by("site")
        .agg(
            pl.col("c_naive").median(),
            pl.col("c_unfold").median(),
            pl.col("negative_share").median(),
        )
        .sort("site")
    )


if __name__ == "__main__":
    main()
