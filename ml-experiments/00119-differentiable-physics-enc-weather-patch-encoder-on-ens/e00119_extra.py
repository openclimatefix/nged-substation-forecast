"""Extra tables for experiment 119's page: each arm against K0, by month and hour, and GAM-0.

Writes `extra_<direction>.json` from the stored final fits. Every interval is a 99% paired
interval resampling whole calendar months.

    uv run python ml-experiments/00119-*/e00119_extra.py
"""

import e00119_common as common
import e00119_cv as cv
import e00119_gamk as gamk
import e00119_score as score
import e00119_stats as stats
import numpy as np
import polars as pl

HOUR_BANDS = {
    "night 0-6": (0, 6),
    "morning 6-11": (6, 11),
    "midday 11-16": (11, 16),
    "evening 16-24": (16, 24),
}


def gam0_scores(*, scored: score.Scored) -> np.ndarray:
    """Return GAM-0's quantile score on the scored rows (NaN where GAM-0 has no row)."""
    pieces = []
    for index, site in enumerate(common.SCORED_SITES):
        for period in ("tuning", "replicate", "scored"):
            path = gamk.WORK_111 / "dp_preds" / f"gam0_{period}_{site}.parquet"
            frame = pl.read_parquet(path).select("run", "valid_time", *common.QUANTILE_COLUMNS)
            pieces.append(frame.with_columns(site_index=pl.lit(index, dtype=pl.Int8)))
    gam0 = pl.concat(pieces)
    rows = scored.rows.select("site_index", "run", "valid_time", "actual").with_row_index("i")
    joined = rows.join(gam0, on=["site_index", "run", "valid_time"], how="left").sort("i")
    quantiles = joined.select(common.QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    out = stats.quantile_score(
        actual=joined["actual"].to_numpy().astype(np.float64), quantiles=quantiles
    )
    out[np.isnan(quantiles).any(axis=1)] = np.nan
    return out


def main() -> None:
    """Write the extra tables of both directions."""
    for direction in cv.DIRECTIONS.values():
        scored = score.load(direction=direction)
        masks = score.scopes(rows=scored.rows)
        rows = scored.rows
        decision = masks["0.5-168 h"]
        result: dict = {"arm_vs_k0": {}, "by_month": {}, "by_hour": {}}
        for arm in score.ARMS:
            result["arm_vs_k0"][arm] = {
                scope: score.contrast(
                    scored=scored, treatment=arm, reference="K0", mask=masks[scope]
                )
                for scope in ("0.5-168 h", "D+1 (15.5-39 h)", "winter (Nov-Mar)")
            }
        month = stats.month_of_run(run_date=rows["run_date"])
        for m in np.unique(month[decision]):
            at = decision & (month == m)
            result["by_month"][f"{m // 12}-{m % 12 + 1:02d}"] = {
                arm: float((scored.score[arm][at] - scored.score["K0"][at]).mean())
                for arm in ("ENC-c", "ENC-c0", "ENC-cpt")
            }
        hour = rows["valid_time"].dt.hour().to_numpy()
        for name, (lo, hi) in HOUR_BANDS.items():
            at = decision & (hour >= lo) & (hour < hi)
            result["by_hour"][name] = {
                arm: float((scored.score[arm][at] - scored.score["K0"][at]).mean())
                for arm in ("ENC-c", "ENC-c0", "ENC-cpt")
            }
        gam0 = gam0_scores(scored=scored)
        shared = decision & np.isfinite(gam0)
        result["gam0"] = {
            "share_of_rows": float(shared.sum() / decision.sum()),
            "gam0": float(gam0[shared].mean()),
            "k0": float(scored.score["K0"][shared].mean()),
        }
        result["gam0_by_lead"] = {
            band: {
                "gam0": float(np.nanmean(gam0[masks[band] & np.isfinite(gam0)])),
                "k0": float(scored.score["K0"][masks[band] & np.isfinite(gam0)].mean()),
            }
            for band in score.LEAD_BANDS
        }
        common.write_json_atomic(
            path=common.WORK_DIR / f"extra_{direction.name}.json", content=result
        )
        print(direction.name, "done", flush=True)


if __name__ == "__main__":
    main()
