"""Experiment 00341, step 1: build the half-hour table, describe the strong set, estimate the noise.

Writes `work/prep.json` (per farm: saturation and cut-out speeds, strong-set shares by margin and
year, the mismatch rate, the speed-error and power-noise estimates, the visibility margin) and
`work/strong_by_month.parquet` for the figures.

    uv run --no-sync python ml-experiments/00341-*/e00341_prep.py
"""

import json
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00341_common as common  # noqa: E402
import e00341_lib as lib  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

MARGINS = (0.0, 1.0, 2.0, 3.0)


def estimate_noise(*, farm: common.Farm) -> dict:
    """Return the speed-error and power-noise estimates from the calibration window.

    The log of observed over expected power has variance `s^2 sigma_v^2 + sigma_p^2`, with `s` the
    local log slope of the fitted power curve. Bin the half-hours by `s`, take each bin's variance
    of the (robustly winsorised) log ratio, and regress the variances on `s^2`.
    """
    grid, curve = common.power_curve_lookup(speed=farm.speed, exp=farm.exp, cut_out=farm.cut_out)
    sel = np.flatnonzero(
        (farm.exp >= common.WEIGHTED_LOW)
        & (farm.exp <= 1.05)
        & (farm.day >= common.CAL_START_DAY)
        & (farm.day <= common.CAL_END_DAY)
    )
    s = lib.log_slope(speed=farm.speed[sel], curve_speed=grid, curve_exp=curve)
    y = np.clip(
        np.log((np.maximum(farm.obs[sel], 0.0) + common.EPS_RATIO) / (farm.exp[sel] + common.EPS_RATIO)),
        *common.LOG_RATIO_CLIP,
    )
    edges = np.quantile(s, np.linspace(0, 1, 9))
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, 7)
    s2 = np.array([np.mean(s[idx == i] ** 2) for i in range(8)])
    var = np.array([(1.4826 * np.median(np.abs(y[idx == i] - np.median(y[idx == i])))) ** 2 for i in range(8)])
    design = np.column_stack([np.ones(8), s2])
    coef, *_ = np.linalg.lstsq(design, var, rcond=None)
    sigma_p2 = max(float(coef[0]), 1e-4)
    sigma_v2 = max(float(coef[1]), 1e-4)
    return {
        "sigma_p": sigma_p2**0.5,
        "sigma_v": sigma_v2**0.5,
        "bins_slope_squared": s2.tolist(),
        "bins_variance": var.tolist(),
    }


def main() -> None:
    """Write the preparation tables."""
    common.build_half_hours()
    farms = common.load_farms()
    out: dict = {}
    monthly: list[pl.DataFrame] = []
    for site, farm in farms.items():
        noise = estimate_noise(farm=farm)
        info: dict = {
            "v_sat": farm.v_sat,
            "cut_out": farm.cut_out,
            "scale": farm.scale,
            "n_turbines": farm.n_turbines,
            "noise": noise,
            "margins": {},
        }
        n_all = len(farm.day)
        for margin in MARGINS:
            prep = lib.prepare(
                farm=farm, margin=margin, sigma_v=noise["sigma_v"], sigma_p=noise["sigma_p"]
            )
            days = np.unique(farm.day[prep.strong])
            obs_strong = farm.obs[prep.strong]
            year = 2019 + (farm.day[prep.strong] + 243) // 365
            info["margins"][str(margin)] = {
                "share_of_half_hours": len(prep.strong) / n_all,
                "n_half_hours": int(len(prep.strong)),
                "days_with_any": int(len(days)),
                "share_of_days_with_any": len(days) / common.N_DAYS,
                "mismatch_rate_under_50pct": float((obs_strong < 0.5).mean()),
                "p999_observed": float(np.quantile(obs_strong, 0.999)),
                "p99_observed": float(np.quantile(obs_strong, 0.99)),
                "max_observed": float(obs_strong.max()),
                "visibility_margin_p999": float(1.0 - np.quantile(obs_strong, 0.999)),
                "ref_p999_calibration": prep.ref_p999,
                "n_by_year": {
                    str(y): int((year == y).sum()) for y in np.unique(year)
                },
            }
            if margin == 2.0:
                month = (farm.day[prep.strong] // 30.4375).astype(int)
                frame = pl.DataFrame({"site": site, "month_index": month}).group_by(
                    "site", "month_index"
                ).len()
                monthly.append(frame)
        out[site] = info
    (common.WORK_DIR / "prep.json").write_text(json.dumps(out, indent=1))
    pl.concat(monthly).write_parquet(common.WORK_DIR / "strong_by_month.parquet")
    for site, info in out.items():
        m = info["margins"]["2.0"]
        print(
            site,
            f"v_sat={info['v_sat']:.1f} cut_out={info['cut_out']:.1f}",
            f"sigma_v={info['noise']['sigma_v']:.3f} sigma_p={info['noise']['sigma_p']:.3f}",
            f"strong share={m['share_of_half_hours']:.4f} days_any={m['share_of_days_with_any']:.3f}",
            f"mismatch={m['mismatch_rate_under_50pct']:.3f} p999={m['p999_observed']:.3f}",
        )
        for k, v in info["margins"].items():
            print("  margin", k, f"{v['share_of_half_hours']:.4f}", f"days {v['share_of_days_with_any']:.3f}")


if __name__ == "__main__":
    main()
