"""Test why B1's cold-winter-night harm stays above the pre-registered +0.03 points.

For the realised cold-winter-night group, report by 24-hour lead band the share of half-hours that
the forecast rule (ensemble-mean temperature below 2 C) also flags, and B1's loss difference on
the flagged and the unflagged half-hours.

    PYTHONPATH=<folders of experiments 184, 202, 317, 319, 320> \
        uv run --no-sync python ml-experiments/00320-*/e00320_diagnostics.py
"""

import e00320_arms as arms
import e00320_common as common
import numpy as np
import polars as pl


def main() -> None:
    """Write `work/report/diagnostics.json`."""
    rows = pl.read_parquet(common.REPORT_DIR / "scored_rows.parquet")
    keys = ["site", "power_fcst_init_time", "valid_time"]
    chain = arms.load_chain_all_leads().select(*keys, "ens_mean_c").unique(subset=keys)
    rows = rows.join(chain, on=keys, how="left", validate="1:1")
    group = rows["cold_winter_night"].to_numpy()
    flagged = np.nan_to_num(rows["ens_mean_c"].to_numpy(), nan=99.0) < common.COLD_FORECAST_C
    lead = rows["lead_hours"].to_numpy()
    d_b1 = (rows["loss_b1"] - rows["loss_default"]).to_numpy()
    d_r = (rows["loss_r319"] - rows["loss_default"]).to_numpy()
    out: dict = {"by_band": {}}
    for lo in range(0, 240, 24):
        band = (lead > lo) & (lead <= lo + 24)
        sel = group & band
        out["by_band"][f"{lo}-{lo + 24} h"] = {
            "group_rows": int(sel.sum()),
            "share_flagged_by_forecast": float(flagged[sel].mean()) if sel.sum() else None,
        }
    long = lead > common.STORED_HORIZON_HOURS
    for name, sel in (
        ("group_flagged", group & flagged & long),
        ("group_unflagged", group & ~flagged & long),
        ("flagged_outside_group", ~group & flagged & long),
    ):
        out[name] = {
            "rows": int(sel.sum()),
            "b1_minus_default": float(d_b1[sel].mean()),
            "r319_minus_default": float(d_r[sel].mean()),
        }
    common.write_json_atomic(path=common.REPORT_DIR / "diagnostics.json", content=out)
    print(out)


if __name__ == "__main__":
    main()
