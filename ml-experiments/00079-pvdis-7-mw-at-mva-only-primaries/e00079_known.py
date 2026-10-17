"""Known-answer fits: hide the sign of the 8 MW Primaries' power, with and without injected solar.

For each MW Primary window, each farm, and each injection level the script subtracts the scaled
farm output from the signed power, fits the signed model (the truth), the naive model, and the
unfolding model to the magnitude, and writes one row per case with the capacities and the sign
counts. A no-injection row per window and one reactive-floor row per `REACTIVE_SHARES` entry at
level 1.0 are added.

    POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 uv run --with holidays python \
        ml-experiments/00079-pvdis-7-mw-at-mva-only-primaries/e00079_known.py
"""

from multiprocessing import get_context

import e00079_common as common
import e00079_fit as fit
import e00079_inputs as inputs
import e00079_selftest as selftest
import numpy as np
import polars as pl

_PREPARED: pl.DataFrame | None = None
_FARMS: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def case_row(
    *,
    window: inputs.YearWindow,
    mask: np.ndarray,
    y: np.ndarray,
    reading: np.ndarray,
    scale: float,
    base_capacity: float,
    meta: dict[str, object],
) -> dict[str, object]:
    """Fit the three models to one case and return its capacities and sign counts."""
    design = fit.Design.build(cells=window.cells[mask], nuisance=window.nuisance[mask])
    u = window.shape[mask]
    c_signed, _ = fit.signed_fit(design=design, y=y, u=u)
    c_naive, m_naive = fit.naive_fit(design=design, a=reading, u=u)
    unfolded = fit.unfold_fit(design=design, a=reading, u=u, scale=scale)
    floor = common.SIGN_FLOOR_SHARE * scale
    keep = np.abs(y) >= floor
    truth_negative = y[keep] < 0.0
    row: dict[str, object] = {
        **meta,
        "rows": int(mask.sum()),
        "scale": scale,
        "c_base_signed": base_capacity,
        "c_signed": c_signed,
        "c_naive": c_naive,
        "c_unfold": unfolded.capacity,
        "loss_unfold": unfolded.loss,
        "loss_naive": fit.abs_loss(a=reading, m=m_naive),
        "loss_gap": unfolded.loss_gap,
        "n_distinct": unfolded.n_starts_distinct,
        "truth_negative": int(truth_negative.sum()),
        "truth_positive": int((~truth_negative).sum()),
        "fitted_negative_share": float((unfolded.mean < 0.0).mean()),
    }
    for name, mean in (("unfold", unfolded.mean), ("naive", m_naive)):
        fitted_negative = mean[keep] < 0.0
        row[f"tp_{name}"] = int((fitted_negative & truth_negative).sum())
        row[f"tn_{name}"] = int((~fitted_negative & ~truth_negative).sum())
    return row


def _job(primary: str) -> list[dict[str, object]]:
    """Return all known-answer rows of one MW Primary."""
    assert _PREPARED is not None
    out: list[dict[str, object]] = []
    for w in inputs.year_windows(prepared=_PREPARED, primary=primary):
        for farm, series in _FARMS.items():
            mask, injected = inputs.aligned_injection(window=w, farm=series)
            scale = w.midday_scale(mask=mask)
            base_design = fit.Design.build(cells=w.cells[mask], nuisance=w.nuisance[mask])
            base_capacity, _ = fit.signed_fit(design=base_design, y=w.power[mask], u=w.shape[mask])
            cases: list[tuple[float, float]] = [(level, 0.0) for level in (0.0, *common.LEVELS)]
            cases += [(1.0, q) for q in common.REACTIVE_SHARES]
            for level, q_share in cases:
                if level == 0.0 and farm != next(iter(_FARMS)):
                    continue
                y = w.power[mask] - level * scale * injected
                reading = np.sqrt(y**2 + (q_share * scale) ** 2) if q_share else np.abs(y)
                out.append(
                    case_row(
                        window=w, mask=mask, y=y, reading=reading, scale=scale,
                        base_capacity=base_capacity,
                        meta={
                            "site": primary, "year": w.year,
                            "farm": farm if level else "-", "level": level,
                            "q_share": q_share,
                            "injected_peak_mw": level * scale,
                        },
                    )
                )  # fmt: skip
    return out


def main() -> None:
    """Run the self-tests, then every known-answer case, and write `known.parquet`."""
    global _PREPARED, _FARMS  # noqa: PLW0603
    selftest.run()
    _PREPARED = inputs.load_prepared()
    _FARMS = inputs.farm_series(prepared=_PREPARED)
    with get_context("fork").Pool(4) as pool:
        rows = [r for chunk in pool.map(_job, common.MW_PRIMARIES) for r in chunk]
    table = pl.DataFrame(rows)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(common.KNOWN_PATH)
    print(
        table.group_by("level", "q_share").agg(pl.len(), pl.col("c_unfold").median()).sort("level")
    )


if __name__ == "__main__":
    main()
