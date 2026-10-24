# ruff: noqa: E501
"""Stage 0: tail rates of Primary demand research default 3 by 24-hour lead band, group, and Primary."""

import e00335_common as common
import numpy as np
import polars as pl


def rates(*, actual: np.ndarray, q: np.ndarray, mask: np.ndarray) -> dict:
    """Return the share below q01, q05, above q95, q99, and the 5% to 95% band coverage."""
    a, qq = actual[mask], q[mask]
    if len(a) == 0:
        return {"n": 0}
    return {
        "n": len(a),
        "below_q01": float((a < qq[:, 0]).mean()),
        "below_q05": float((a < qq[:, 1]).mean()),
        "above_q95": float((a > qq[:, 9]).mean()),
        "above_q99": float((a > qq[:, 10]).mean()),
        "in_5_95": float(((a >= qq[:, 1]) & (a <= qq[:, 9])).mean()),
    }


def main() -> None:
    """Write the tail-rate tables and print the pooled band table."""
    rows = pl.read_parquet(common.WORK_DIR / "default3_scored.parquet")
    actual = rows["actual"].to_numpy().astype(np.float64)
    q = rows.select([f"d3_{n}" for n in common.QUANTILE_NAMES]).to_numpy().astype(np.float64)
    band = common.band_index(lead_hours=rows["lead_hours"].to_numpy())
    group = common.group_index(frame=rows)
    site = rows["site"].to_numpy()
    realised = {
        "cold winter nights": rows["cold_winter_night"].fill_null(False).to_numpy(),
        "sunny daylight": rows["sunny_daylight"].fill_null(False).to_numpy(),
    }
    out: dict = {"by_band": {}, "by_band_group": {}, "by_band_site": {}, "by_band_realised": {}}
    for b, label in enumerate(common.BAND_LABELS):
        m = band == b
        out["by_band"][label] = rates(actual=actual, q=q, mask=m)
        out["by_band_group"][label] = {
            g: rates(actual=actual, q=q, mask=m & (group == i))
            for i, g in enumerate(common.GROUP_NAMES)
        }
        out["by_band_site"][label] = {
            s: rates(actual=actual, q=q, mask=m & (site == s)) for s in sorted(set(site))
        }
        out["by_band_realised"][label] = {
            g: rates(actual=actual, q=q, mask=m & v) for g, v in realised.items()
        }
    common.write_json_atomic(path=common.WORK_DIR / "stage0.json", content=out)
    print("band n below_q01 below_q05 above_q95 above_q99 in_5_95")
    for label, r in out["by_band"].items():
        print(
            label,
            r["n"],
            *[
                f"{r[k]:.4f}"
                for k in ("below_q01", "below_q05", "above_q95", "above_q99", "in_5_95")
            ],
        )
    miss = [
        max(abs(r["below_q01"] - 0.01), abs(r["above_q99"] - 0.01)) for r in out["by_band"].values()
    ]
    print("max tail miss per band", [round(m, 4) for m in miss])
    print("cold winter nights (realised) by band, above_q95 / above_q99:")
    for label, r in out["by_band_realised"].items():
        c = r["cold winter nights"]
        print(label, c.get("n"), c.get("above_q95"), c.get("above_q99"))


if __name__ == "__main__":
    main()
