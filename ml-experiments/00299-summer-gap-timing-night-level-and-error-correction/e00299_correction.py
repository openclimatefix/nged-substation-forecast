# ruff: noqa: E501
"""Part 2 of experiment 299: does the error-correction layer cause the spring excess?

Stored pairs only. Experiment 111's GAM-K and GAM0 (the convex model with and without the Kalman
filter), and experiment 201's neural-network heads (arms 1, 5, and 6) against arm D, all scored
on experiment 266's rows.

    OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 systemd-run --user --scope -p MemoryMax=8G \
        -p OOMPolicy=continue uv run --no-sync python ml-experiments/00299-*/e00299_correction.py
"""

from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

import e00299_common as common  # isort: skip
import numpy as np
import polars as pl

KEYS: Final[list[str]] = ["site", "power_fcst_init_time", "valid_time"]
HEAD_ARMS: Final[tuple[str, ...]] = ("arm1", "arm5", "arm6")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
FOCUS_SITES: Final[tuple[str, ...]] = ("D5", "D9")


def scored(*, frame: pl.DataFrame, name: str) -> pl.DataFrame:
    """Return `frame`'s keys with the per-row quantile score named `name`."""
    qs = common.qs_per_row(
        actual=frame["actual"].to_numpy(), q=frame.select(common.QUANTILES).to_numpy()
    )
    return frame.select(KEYS).with_columns(pl.Series(name, qs))


def load_older_gamk() -> pl.DataFrame:
    """Return per-row scores of experiment 111's GAM-K and GAM0 at the six substations."""
    out = []
    for arm in ("gamk", "gam0"):
        files = [
            pl.read_parquet(common.E111_PREDS / f"{arm}_scored_{site}.parquet").with_columns(
                site=pl.lit(site)
            )
            for site in common.SITES
        ]
        out.append(scored(frame=pl.concat(files), name=f"qs_{arm}"))
    return out[0].join(out[1], on=KEYS, how="inner", validate="1:1")


def load_heads() -> pl.DataFrame:
    """Return the per-row score of each head arm, averaged over its seeds."""
    joined: pl.DataFrame | None = None
    for arm in HEAD_ARMS:
        seeds = [
            scored(
                frame=pl.read_parquet(
                    ML_EXPERIMENTS_DATA_DIR
                    / "00201-dp-gam-residual-mlp" / "work" / "preds" / f"{arm}_seed{s}.parquet"
                ),
                name=f"qs_head_{arm}",
            )
            for s in SEEDS
        ]  # fmt: skip
        mean = pl.concat(seeds).group_by(KEYS).agg(pl.col(f"qs_head_{arm}").mean())
        joined = mean if joined is None else joined.join(mean, on=KEYS, how="inner")
    assert joined is not None
    return joined


def ratio_interval(
    *, num: np.ndarray, den: np.ndarray, mask: np.ndarray, blocks: np.ndarray
) -> dict:
    """Return sum(num)/sum(den) over `mask` with a 95% interval resampling `blocks`."""
    keep = mask & np.isfinite(num) & np.isfinite(den)
    labels, codes = np.unique(blocks[keep], return_inverse=True)
    n = len(labels)
    a = np.bincount(codes, weights=num[keep], minlength=n)
    b = np.bincount(codes, weights=den[keep], minlength=n)
    draws = np.random.default_rng(0).integers(0, n, size=(common.N_RESAMPLES, n))
    res = a[draws].sum(axis=1) / b[draws].sum(axis=1)
    lo, hi = np.percentile(res, [2.5, 97.5])
    return {"est": float(a.sum() / b.sum()), "lo": float(lo), "hi": float(hi), "blocks": int(n)}


def main() -> None:
    """Write `work/correction.json` and print the headline numbers."""
    m = common.read_master()
    m = m.join(load_older_gamk(), on=KEYS, how="left").join(load_heads(), on=KEYS, how="left")
    for name in ("qs_gamk", "qs_gam0", *(f"qs_head_{a}" for a in HEAD_ARMS)):
        assert m[name].null_count() == 0, f"{name} lost rows in the key join"
    out: dict = {
        "rows": m.height,
        "gamk_rows": int(m["qs_gamk"].is_not_null().sum()),
        "head_rows": int(m["qs_head_arm1"].is_not_null().sum()),
    }
    print(out)
    site = m["site"].to_numpy()
    day = m["day"].to_numpy()
    month = m["init_month"].to_numpy()
    blocks = m["init_week"].to_numpy()
    col = {
        k: m[k].to_numpy()
        for k in ("qs_d", "qs_bg2", "qs_gamk", "qs_gam0", *(f"qs_head_{a}" for a in HEAD_ARMS))
    }
    windows = {"AMJ": np.isin(month, common.AMJ), "JAS": np.isin(month, common.JAS)}
    contrasts = {
        "gamk_minus_bg2": col["qs_gamk"] - col["qs_bg2"],
        "gam0_minus_bg2": col["qs_gam0"] - col["qs_bg2"],
        "gamk_minus_gam0": col["qs_gamk"] - col["qs_gam0"],
        "armD_minus_bg2": col["qs_d"] - col["qs_bg2"],
        **{f"{a}_minus_armD": col[f"qs_head_{a}"] - col["qs_d"] for a in HEAD_ARMS},
    }
    out["contrasts"] = {}
    for s in (*common.SITES, "all"):
        sm = np.ones(len(site), dtype=bool) if s == "all" else site == s
        for w, wm in windows.items():
            for dlabel, dm in (("daylight", day), ("night", ~day)):
                key = f"{s}|{w}|{dlabel}"
                out["contrasts"][key] = {
                    name: common.block_mean_interval(values=v, mask=sm & wm & dm, blocks=blocks)
                    for name, v in contrasts.items()
                }
    # Share of the spring excess that removing each layer removes, at D5 and D9, daylight.
    out["reduction"] = {}
    for s in (*FOCUS_SITES, "D5+D9"):
        sm = np.isin(site, FOCUS_SITES) if s == "D5+D9" else site == s
        amj, jas = windows["AMJ"], windows["JAS"]
        d: dict = {}
        # Share of the layer-included excess that removing the layer removes; negative: the layer helps.
        d["filter_removal_share_AMJ"] = ratio_interval(
            num=contrasts["gamk_minus_gam0"],
            den=contrasts["gamk_minus_bg2"],
            mask=sm & amj & day,
            blocks=blocks,
        )
        d["filter_removal_change_JAS"] = common.block_mean_interval(
            values=-contrasts["gamk_minus_gam0"], mask=sm & jas & day, blocks=blocks
        )
        for a in HEAD_ARMS:
            d[f"{a}_removal_share_AMJ"] = ratio_interval(
                num=contrasts[f"{a}_minus_armD"],
                den=contrasts[f"{a}_minus_armD"] + contrasts["armD_minus_bg2"],
                mask=sm & amj & day,
                blocks=blocks,
            )
            d[f"{a}_removal_change_JAS"] = common.block_mean_interval(
                values=-contrasts[f"{a}_minus_armD"], mask=sm & jas & day, blocks=blocks
            )
        out["reduction"][s] = d
    # Filter benefit by season: GAM-K minus GAM0 by init month, all rows.
    out["filter_by_init_month"] = (
        m.filter(pl.col("qs_gamk").is_not_null())
        .group_by("init_month")
        .agg((pl.col("qs_gamk") - pl.col("qs_gam0")).mean().alias("gamk_minus_gam0"), pl.len())
        .sort("init_month")
        .to_dicts()
    )
    common.write_json_atomic(path=common.WORK_DIR / "correction.json", content=out)
    for s, d in out["reduction"].items():
        print(s)
        for k, v in d.items():
            print("  ", k, {a: round(b, 3) for a, b in v.items() if isinstance(b, float)})
    for r in out["filter_by_init_month"]:
        print(r)


if __name__ == "__main__":
    main()
