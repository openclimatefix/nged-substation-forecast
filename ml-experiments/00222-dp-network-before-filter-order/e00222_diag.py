"""Print and store the pre-registered network's residual check on the scored year, per site.

For each site: the mean absolute residual of the filter's stream, and of day-ahead forecast rows,
without and with the network's correction (scored year, outside switching events), the stream row
counts, and the epoch settings. Writes `work/diag.json`.

    uv run --no-sync python ml-experiments/00222-*/e00222_diag.py
"""

import json

import e00222_common as common
import e00222_fit as fit
import e00222_inputs as inputs
import e00222_stream as stream_module
import numpy as np
import polars as pl


def main() -> None:
    """Write the per-site residual check."""
    cache = common.load_cache()
    out: dict[str, dict[str, float]] = {}
    for site in common.SCORED_SITES:
        site_stream = stream_module.load_site_stream(site=site, cache=cache)
        delta = np.load(fit.NN_DIR / f"{site}.npz")
        mask = (
            (cache.end_times >= inputs.SCORED_START)
            & delta["has_delta"]
            & ~cache.sites[site].switching_mask
        )
        s, d = site_stream.stream[mask], delta["delta"][mask]
        rows = pl.read_parquet(fit.NN_DIR.parent / "arms" / f"pre_s1.00_{site}.parquet").filter(
            (pl.col("period") == "scored")
            & ~pl.col("switching_mask")
            & pl.col("lead_hours").is_between(15.5, 39)
        )
        y, raw, dl = (rows[c].to_numpy() for c in ("actual", "raw", "delta"))
        out[site] = {
            "stream_mae_without": float(np.abs(s).mean()),
            "stream_mae_with": float(np.abs(s - d).mean()),
            "d1_mae_without": float(np.abs(y - raw).mean()),
            "d1_mae_with": float(np.abs(y - raw - dl).mean()),
        }
        print(site, {k: round(v, 2) for k, v in out[site].items()})
    (common.WORK_DIR / "diag.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
