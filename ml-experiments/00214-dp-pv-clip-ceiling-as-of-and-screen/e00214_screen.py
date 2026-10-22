"""Experiment 214, stage 1: screen the Primaries for experiment 180's export-limit signature.

No fit. On clear May to August half-hours before 2025-07-01 (clear-sky index of CAMS irradiance
0.9 or more), for each summer of 2022, 2023, and 2024, prints and stores the implied points of solar
per unit of the fixed fleet shape by decile of the shape, as experiment 180's stage 0 did. The
signature index is the mean of the top two deciles over the mean of deciles 4 to 6; a Primary
shows the signature when the index is at most 0.85 in all three summers.

    uv run --no-sync python ml-experiments/00214-*/e00214_screen.py
"""

from typing import Final

import e00214_common as common
import numpy as np
import polars as pl
from studies.baselines import haurwitz_w_m2

SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
SUMMERS: Final[tuple[int, ...]] = (2022, 2023, 2024)
CLEAR_INDEX: Final[float] = 0.9
SIGNATURE_MAX_INDEX: Final[float] = 0.85
SUN_UP: Final[float] = 0.01
MIN_ROWS: Final[int] = 300
"""A summer with fewer clear half-hours than this is not screened (its index is `NaN`)."""


def main() -> None:
    """Write `work/stage1.json` and print the tables."""
    cache = common.load_cache()
    local_date = cache.calendar.local_date
    month = local_date.astype("datetime64[M]").astype(np.int64) % 12 + 1
    year = local_date.astype("datetime64[Y]").astype(np.int64) + 1970
    stored: dict = {}
    for site in SITES:
        f = cache.sites[site]
        clear = haurwitz_w_m2(apparent_zenith_deg=90.0 - f.elevation_deg)
        csi = np.where(clear > 20, f.ghi_w_m2 / np.where(clear > 20, clear, 1), np.nan)
        night = np.isfinite(f.y) & (f.elevation_deg < -5) & (year <= 2024)
        for summer in SUMMERS:
            ok = np.isfinite(f.y) & np.isfinite(f.pv_unit_shape) & (csi >= CLEAR_INDEX)
            rows = ok & np.isin(month, [5, 6, 7, 8]) & (f.pv_unit_shape > SUN_UP) & (year == summer)
            night_year = night & (year == summer)
            if rows.sum() < MIN_ROWS or not night_year.any():
                stored[f"{site}|{summer}"] = {"index": float("nan"), "rows": int(rows.sum())}
                print(f"{site} {summer}: rows {int(rows.sum()):5d} too few to screen")
                continue
            night_mean = float(np.mean(f.y[night_year]))
            frame = pl.DataFrame({"y": f.y[rows], "u": f.pv_unit_shape[rows]})
            edges = np.quantile(frame["u"].to_numpy(), np.linspace(0, 1, 11)[1:-1])
            deciles = (
                frame.with_columns(d=pl.Series(np.searchsorted(edges, frame["u"].to_numpy())))
                .group_by("d")
                .agg(u=pl.col("u").mean(), y=pl.col("y").mean())
                .sort("d")
            )
            slope = ((night_mean - deciles["y"]) / deciles["u"]).to_numpy()
            top = float(np.mean(slope[-2:]))
            middle = float(np.mean(slope[3:6]))
            index = top / middle if middle > 0 else float("nan")
            stored[f"{site}|{summer}"] = {
                "slope": [float(s) for s in slope],
                "index": index,
                "rows": frame.height,
                "night_mean_points": night_mean,
            }
            print(
                f"{site} {summer}: rows {frame.height:5d} index {index:6.2f} "
                f"slopes {[round(float(s), 1) for s in slope]}"
            )
    verdict = {}
    for site in SITES:
        indices = [stored[f"{site}|{s}"]["index"] for s in SUMMERS]
        verdict[site] = bool(all(np.isfinite(i) and i <= SIGNATURE_MAX_INDEX for i in indices))
    stored["signature"] = verdict
    print("signature (index <= 0.85 in all three summers):", verdict)
    common.write_json_atomic(path=common.WORK_DIR / "stage1.json", content=stored)


if __name__ == "__main__":
    main()
