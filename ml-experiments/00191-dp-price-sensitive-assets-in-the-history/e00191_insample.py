# ruff: noqa: E501
"""Experiment 191: in-sample gain of the price columns, and the fitted price response in MW.

For each monthly fit, the mean absolute error on the fit's own training rows of the model without
price columns (a0) and with them (joint, at the true price). The price response is the fitted
output when every indicator is on, in MW (points times the site's effective capacity over 100).

    uv run --no-sync python ml-experiments/00191-*/e00191_insample.py
"""

import pickle

import e00191_common as common
import e00191_engine as engine
import numpy as np
from studies.dp_gam import TrainingWindow, predict_gam, prepare_fit


def main() -> None:
    """Print the in-sample table for D8, D14 and D5."""
    cache = common.load_cache()
    for site in ("D8", "D14", "D5"):
        work = engine.masked_cache(cache=cache, site=site)
        grid = engine.build_price_grid(cache=work, site=site, ungated=False)
        fits = pickle.loads((common.FITS_DIR / "gated" / f"{site}.pkl").read_bytes())
        capacity = work.sites[site].table.effective_capacity_mw
        print(f"\n== {site} (effective capacity {capacity:.2f} MW)")
        print("cut        rows  a0 MAE  joint MAE  gain%   response MW (all on)  theta (points)")
        gains = []
        for cut_fits in fits:
            _, index = prepare_fit(
                cache=work, site=site, spec=engine.gam_spec(), cut=cut_fits.cut,
                window=TrainingWindow(months=common.WINDOW_MONTHS), season_years=common.SEASON_YEARS,
            )  # fmt: skip
            y = work.sites[site].y[index]
            a0 = np.abs(y - predict_gam(cache=work, fit=cut_fits.a0, index=index)).mean()
            joint_pred = predict_gam(
                cache=work, fit=cut_fits.joint.fit, index=index
            ) - engine.price_output(
                cut_fits=cut_fits, mode="true", cache=work, grid=grid, rows=index, origin=None
            )
            joint = np.abs(y - joint_pred).mean()
            gains.append((a0 - joint) / a0 * 100.0)
            theta = cut_fits.joint.theta_price
            print(
                f"{cut_fits.cut:%Y-%m-%d} {len(index):6d} {a0:7.3f} {joint:9.3f} "
                f"{gains[-1]:6.2f}  {theta.sum() * capacity / 100.0:8.2f}   {np.round(theta, 2)}"
            )
        print(f"mean in-sample gain {np.mean(gains):.2f}%, last cut {gains[-1]:.2f}%")


if __name__ == "__main__":
    main()
