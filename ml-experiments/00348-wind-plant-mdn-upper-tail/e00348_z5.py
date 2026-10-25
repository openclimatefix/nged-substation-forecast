# ruff: noqa: E501
"""Step 2 of experiment 00348: arm z5, the solar recipe on the frozen Stage 3 wind plant.

Works on stored forecasts only (tier 1), with no neural network:

- `z5a`, plant with offsets: per farm, the plant's 11 member-mixture quantiles plus
  `Q_tau(actual - plant quantile_tau)` per bin of the plant's mean (6 quantile bins) and ensemble
  standard deviation (3 terciles), fitted on the kept hindcast half-hours.
- `z5b`, plant with a residual band: the plant's median plus `Q_tau(actual - plant median)` per
  bin of the plant's mean (6 quantile bins) and lead band (under 24 h, 24 to 72 h, 72 to 240 h,
  over 240 h).
- `z5`, the blend: per quantile and lead band, `w * z5b + (1 - w) * default` with `w` in steps of
  0.1 chosen by pinball loss on the kept scored half-hours of the months before the month being
  predicted (0.5 for the first scored month). Quantiles are clipped to [0, 1.0] and sorted.

    PYTHONPATH=<348>:<347>:<230>:<303>:<310>:<326>:<336 folders> \
    uv run --no-sync python ml-experiments/00348-*/e00348_z5.py
"""

from typing import Final

import e00303_common as c303  # isort: skip  # ty: ignore[unresolved-import]
import e00347_s4_data as data  # isort: skip  # ty: ignore[unresolved-import]
import e00348_common as common
import numpy as np
import polars as pl

MEAN_BINS: Final[int] = 6
SD_BINS: Final[int] = 3
LEAD_EDGES: Final[tuple[float, ...]] = (24.0, 72.0, 240.0)
GRID: Final[np.ndarray] = np.linspace(0.0, 1.0, 11)
MIN_BIN_ROWS: Final[int] = 200


def lead_band(*, hours: np.ndarray) -> np.ndarray:
    """Return the lead band index 0 to 3 of each lead in hours."""
    return np.searchsorted(np.array(LEAD_EDGES), hours, side="right")


def bin_ids(*, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the bin of each value given interior bin edges."""
    return np.searchsorted(edges, values, side="right")


def pinball_level(*, q: np.ndarray, actual: np.ndarray, tau: float) -> np.ndarray:
    """Return the pinball loss of one quantile forecast per row."""
    error = actual - q
    return np.maximum(tau * error, (tau - 1.0) * error)


def fit_and_apply(
    *,
    hind: pl.DataFrame,
    scored: pl.DataFrame,
    sum_h: np.ndarray,
    sum_s: np.ndarray,
    mix_h: np.ndarray,
    mix_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the offset forecast (a) and the band forecast (b) for the scored rows: (rows, 11)."""
    levels = c303.LEVELS
    train_all = (~hind["masked"]).to_numpy()
    actual_h = hind["actual_norm"].to_numpy().astype(np.float64)
    lead_h = lead_band(hours=hind["lead_hours"].to_numpy().astype(np.float64))
    lead_s = lead_band(hours=scored["lead_hours"].to_numpy().astype(np.float64))
    out_a = np.zeros((len(scored), 11))
    out_b = np.zeros((len(scored), 11))
    for farm in common.FARMS:
        rows_h = train_all & (hind["site"] == farm).to_numpy()
        rows_s = (scored["site"] == farm).to_numpy()
        mean_edges = np.quantile(sum_h[rows_h, 1], np.linspace(0, 1, MEAN_BINS + 1)[1:-1])
        sd_edges = np.quantile(sum_h[rows_h, 2], np.linspace(0, 1, SD_BINS + 1)[1:-1])
        mean_h = bin_ids(values=sum_h[:, 1], edges=mean_edges)
        mean_s = bin_ids(values=sum_s[:, 1], edges=mean_edges)
        sd_h = bin_ids(values=sum_h[:, 2], edges=sd_edges)
        sd_s = bin_ids(values=sum_s[:, 2], edges=sd_edges)
        bin_a_h, bin_a_s = mean_h * SD_BINS + sd_h, mean_s * SD_BINS + sd_s
        bin_b_h, bin_b_s = mean_h * 4 + lead_h, mean_s * 4 + lead_s
        for k, tau in enumerate(levels):
            diff_a = actual_h - mix_h[:, k]
            diff_b = actual_h - sum_h[:, 0]
            fallback_a = np.quantile(diff_a[rows_h], tau)
            fallback_b = np.quantile(diff_b[rows_h], tau)
            for ids_h, ids_s, diff, fallback, out, base_s in (
                (bin_a_h, bin_a_s, diff_a, fallback_a, out_a, mix_s[:, k]),
                (bin_b_h, bin_b_s, diff_b, fallback_b, out_b, sum_s[:, 0]),
            ):
                offset = np.full(MEAN_BINS * 4 + 1, fallback)
                for b in np.unique(ids_h[rows_h]):
                    sel = rows_h & (ids_h == b)
                    if sel.sum() >= MIN_BIN_ROWS:
                        offset[b] = np.quantile(diff[sel], tau)
                out[rows_s, k] = base_s[rows_s] + offset[ids_s[rows_s]]
    return out_a, out_b


def blend(
    *, plant: np.ndarray, default: np.ndarray, scored: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return the blend and the weights used: (rows, 11) and (months, 4, 11)."""
    levels = c303.LEVELS
    actual = scored["actual_norm"].to_numpy().astype(np.float64)
    kept = (~scored["masked"]).to_numpy()
    bands = lead_band(hours=scored["lead_hours"].to_numpy().astype(np.float64))
    month = scored["month"].to_numpy()
    months = np.unique(month)
    out = np.zeros_like(plant)
    weights_used = np.full((len(months), 4, 11), 0.5)
    for m_i, m in enumerate(months):
        earlier = kept & (month < m)
        now = month == m
        for band in range(4):
            sel_e = earlier & (bands == band)
            sel_n = now & (bands == band)
            for k, tau in enumerate(levels):
                w = 0.5
                if sel_e.sum() > 0:
                    losses = [
                        pinball_level(
                            q=g * plant[sel_e, k] + (1 - g) * default[sel_e, k],
                            actual=actual[sel_e],
                            tau=tau,
                        ).mean()
                        for g in GRID
                    ]
                    w = float(GRID[int(np.argmin(losses))])
                weights_used[m_i, band, k] = w
                out[sel_n, k] = w * plant[sel_n, k] + (1 - w) * default[sel_n, k]
    return np.sort(np.clip(out, 0.0, 1.0), axis=1), weights_used


def replicate_z5() -> None:
    """Store z5's replicate: fit before 2024-12-01 hindcast runs, score the daily runs of 2025-01 to 2025-06.

    Run after the main scores were read (the plan said z5 could not be replicated; the default's
    replicate forecasts turned out to exist), so the replicate is post hoc.
    """
    import e00347_s4_data as d347  # isort: skip  # ty: ignore[unresolved-import]

    hind = data.hindcast_frame()
    in_rep = (hind["power_fcst_init_time"] >= data.REPLICATE_START).to_numpy()
    rep_cases = pl.read_parquet(data.E336_FINAL / "replicate_cases.parquet")
    replicate = hind.filter(pl.Series(in_rep)).with_columns(month=rep_cases["month"])
    fit = hind.filter(pl.Series(~in_rep & (hind["valid_time"] < data.REPLICATE_START).to_numpy()))
    keep_fit = ~in_rep & (hind["valid_time"] < data.REPLICATE_START).to_numpy()
    sum_h = d347.summaries(arm_plant="y2", set_name="hindcast")
    mix_h = d347.mixtures(arm_plant="y2", set_name="hindcast")
    _, band = fit_and_apply(
        hind=fit,
        scored=replicate,
        sum_h=sum_h[keep_fit],
        sum_s=sum_h[in_rep],
        mix_h=mix_h[keep_fit],
        mix_s=mix_h[in_rep],
    )
    default = np.load(data.E339_FINAL / "default_rep.npy")[0].astype(np.float64)
    band = np.sort(np.clip(band, 0.0, 1.0), axis=1)
    z5, _ = blend(plant=band, default=default, scored=replicate)
    common.PARTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(common.PARTS_DIR / "z5_rep_h0.npy", z5.astype(np.float32))


def main() -> None:
    """Build z5a, z5b, and z5 and store them (one seed, shape (rows, 11))."""
    hind, scored = data.hindcast_frame(), data.scored_frame()
    sum_h, sum_s = (data.summaries(arm_plant="y2", set_name=s) for s in ("hindcast", "scored"))
    mix_h, mix_s = (data.mixtures(arm_plant="y2", set_name=s) for s in ("hindcast", "scored"))
    default = np.load(data.E339_FINAL / "default.npy")[0].astype(np.float64)
    a, b = fit_and_apply(
        hind=hind, scored=scored, sum_h=sum_h, sum_s=sum_s, mix_h=mix_h, mix_s=mix_s
    )
    a, b = np.sort(np.clip(a, 0.0, 1.0), axis=1), np.sort(np.clip(b, 0.0, 1.0), axis=1)
    z5, weights = blend(plant=b, default=default, scored=scored)
    replicate_z5()
    common.PARTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, array in (("z5a", a), ("z5b", b), ("z5", z5)):
        np.save(common.PARTS_DIR / f"{name}_h0.npy", array.astype(np.float32))
    np.save(common.DATA_DIR / "z5_blend_weights.npy", weights)
    print("z5 stored; mean blend weight on the plant side by quantile level:")
    print(np.round(weights.mean(axis=(0, 1)), 2).tolist())


if __name__ == "__main__":
    main()
