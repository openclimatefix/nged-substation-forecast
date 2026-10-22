# ruff: noqa: E501
"""Print the spread of the member responses on cold nights, the by-substation table, and the tails.

uv run --no-sync python ml-experiments/00192-*/e00192_extra.py
"""

import json

import e00192_arms as arms
import e00192_common as common
import e00192_data as data
import numpy as np
import polars as pl


def main() -> None:
    """Write `extra.json` and print the numbers the write-up quotes."""
    report = json.loads((common.WORK_DIR / "report" / "report.json").read_text())
    cache = common.load_cache()
    out: dict = {}
    for period in ("scored", "replicate"):
        frame, deltas = data.load_period(period=period, cache=cache)
        cold = (frame["cold_night"] & pl.Series(data.usable_rows(frame=frame))).to_numpy()
        daylight = (frame["kind"] == "daylight").to_numpy() & data.usable_rows(frame=frame)
        spread = {}
        for name in ("delta_t", "delta_m", "delta_st"):
            d = deltas[name].astype(np.float64)
            spread[name] = {
                "cold_std_of_members": float(np.nanmean(np.nanstd(d[cold], axis=1))),
                "cold_mean_response": float(np.nanmean(d[cold])),
                "daylight_std_of_members": float(np.nanmean(np.nanstd(d[daylight], axis=1))),
            }
        quantiles = frame.select(list(data.QUANTILES)).to_numpy()
        half_width = float(np.mean((quantiles[cold, 9] - quantiles[cold, 1]) / 2.0))
        out[period] = {"spread": spread, "arm_a_cold_half_width_q05_q95": half_width}
        print(period, json.dumps(out[period], indent=1))
        subsets = report[period]["subsets"]
        for cut in ("cold nights", *(f"cold nights, {s}" for s in common.DECISION_SITES)):
            e = subsets[cut]
            print(
                f"  {cut}: n={e['rows']} A qs {e['a']['qs']:.3f} mae {e['a']['abs_median']:.3f}"
                f" in90 {e['a']['inside90']:.3f} width {e['a']['width']:.3f}"
                f" below05 {e['a']['below05']:.3f} below95 {e['a']['below95']:.3f}"
                f" | T {e['t']['change_pct']['point']:+.2f}% [{e['t']['change_pct']['lower']:+.2f},{e['t']['change_pct']['upper']:+.2f}]"
                f" ST {e['st']['change_pct']['point']:+.2f}% mae {e['t']['abs_median']:.3f} width {e['t']['width']:.3f}"
                f" in90 {e['t']['inside90']:.3f} MR in90 {e['mr']['inside90']:.3f} width {e['mr']['width']:.3f}"
                f" pin01 A {e['a']['pin01']:.3f} T {e['t']['pin01']:.3f} pin99 A {e['a']['pin99']:.3f} T {e['t']['pin99']:.3f}"
            )
        mean_gap = {}
        for site in common.DECISION_SITES:
            idx = np.flatnonzero(cold & (frame["site"] == site).to_numpy())
            mean_gap[site] = float(np.nanmean(deltas["delta_t"][idx].astype(np.float64)))
        out[period]["mean_response_t_by_site"] = mean_gap
    arms.ARM_DIR.mkdir(parents=True, exist_ok=True)
    common.write_json_atomic(path=common.WORK_DIR / "report" / "extra.json", content=out)


if __name__ == "__main__":
    main()
