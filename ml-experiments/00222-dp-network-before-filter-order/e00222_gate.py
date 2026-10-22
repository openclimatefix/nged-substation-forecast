"""The known-answer gate of experiment 222.

**G1, the network learns a known systematic error.** In the rows the network trains and stops on
(never the held-out fold), 3% of the convex forecast is added to the residual on days whose mean
driver temperature is above the site's 95th percentile (the convex model then looks as if it
under-forecasts hot days by 3%). A network that holds out one calendar-month fold is trained on
the injected residuals and on the clean residuals with the same seed, and the recovery is the mean
difference of the two networks' corrections on the held-out fold's hot days, divided by the mean
injected amount. The gate asks for at least 50%.

**G2, the filter still follows a level step.** A step of +3 points is added to the observed power,
and so to the residual stream, for 8 weeks (2025-04-07 to 2025-06-02). The network's inputs do not
change (they are the convex forecast and the weather), so its correction is unchanged. The filter
runs on the clean and on the stepped stream, with the network's correction subtracted (PRE) and
without it (R0). Recovery is the mean rise in the filter's level plus anomaly at day-ahead leads
(15.5 to 39 hours) on rows valid in the block, from runs with origins from the third day of the
block, divided by 3. The gate asks for PRE's recovery to be within 5 percentage points of R0's.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00222-*/e00222_gate.py
"""

import json
from typing import Final

import e00222_build as build
import e00222_common as common
import e00222_fit as fit
import e00222_inputs as inputs
import e00222_stream as stream_module
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

INJECTED_SHARE: Final[float] = 0.03
STEP_POINTS: Final[float] = 3.0
STEP_BLOCK: Final[tuple[str, str]] = ("2025-04-07", "2025-06-02")
HELD_OUT_FOLD: Final[int] = 0
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
D1_FIRST_LEAD: Final[float] = 15.5
D1_LAST_LEAD: Final[float] = 39.0


def hot_day_mask(*, pooled: fit.Pooled) -> np.ndarray:
    """Return the stream rows on days above their site's 95th-percentile mean temperature."""
    local = pl.Series(pooled.times).dt.replace_time_zone("UTC").dt.convert_time_zone(inputs.LONDON)
    frame = pl.DataFrame({"site": pooled.site, "day": local.dt.date(), "temp": pooled.temp})
    day_mean = frame.select(pl.col("temp").mean().over("site", "day"))["temp"].to_numpy()
    hot = np.zeros(len(day_mean), dtype=bool)
    for index in range(len(inputs.SITES)):
        rows = pooled.site == index
        threshold = np.percentile(day_mean[rows & pooled.pre_scored], 95)
        hot[rows] = day_mean[rows] > threshold
    return hot


def gate_network_learns(*, pooled: fit.Pooled) -> dict[str, object]:
    """Run G1 and return the recoveries by seed and their mean."""
    hot = hot_day_mask(pooled=pooled)
    train, val = fit.split_masks(pooled=pooled, held_out=HELD_OUT_FOLD)
    inject = INJECTED_SHARE * np.abs(pooled.g)
    injected_y = np.where((train | val) & hot, pooled.y + inject, pooled.y)
    held = pooled.pre_scored & ~pooled.switching & (pooled.fold == HELD_OUT_FOLD) & hot
    recoveries = []
    for seed in SEEDS:
        clean = fit.train_networks(pooled=pooled, seed=seed, folds=(HELD_OUT_FOLD,))[0]
        marked = fit.train_networks(pooled=pooled, seed=seed, y=injected_y, folds=(HELD_OUT_FOLD,))[
            0
        ]
        rise = marked.predict(x=pooled.x[held]) - clean.predict(x=pooled.x[held])
        recoveries.append(float(rise.mean() / inject[held].mean()))
        print(f"G1 seed {seed}: recovery {recoveries[-1]:+.1%}", flush=True)
    return {
        "recovery_by_seed": recoveries,
        "recovery_mean": float(np.mean(recoveries)),
        "held_out_hot_rows": int(held.sum()),
        "training_hot_rows_injected": int(((train | val) & hot).sum()),
        "mean_injected_points": float(inject[held].mean()),
    }


def gate_filter_follows() -> dict[str, object]:
    """Run G2 and return the level-step recovery of R0's filter and PRE's filter."""
    start, stop = (np.datetime64(day, "us") for day in STEP_BLOCK)
    rises: dict[str, list[np.ndarray]] = {"R0": [], "PRE": []}
    for site in common.SCORED_SITES:
        base = build.read_a2(site=site)
        site_stream = stream_module.load_site_stream(site=site)
        times = site_stream.cache.end_times
        step = np.where((times >= start) & (times < stop), STEP_POINTS, 0.0)
        delta = np.load(fit.NN_DIR / f"{site}.npz")
        corrected = np.where(
            delta["has_delta"], site_stream.stream - delta["delta"], site_stream.stream
        )
        rows = base.filter(
            pl.col("lead_hours").is_between(D1_FIRST_LEAD, D1_LAST_LEAD)  # ty: ignore[unsupported-operator]
            & (
                pl.col("power_fcst_init_time").dt.replace_time_zone(None)
                >= start + np.timedelta64(3, "D")
            )
            & (pl.col("valid_time").dt.replace_time_zone(None) < stop)
            & (pl.col("valid_time").dt.replace_time_zone(None) >= start)
        )
        for name, stream in (("R0", site_stream.stream), ("PRE", corrected)):
            clean = stream_module.run_filter(cache=site_stream.cache, stream=stream)
            stepped = stream_module.run_filter(cache=site_stream.cache, stream=stream + step)
            level_c, anomaly_c = stream_module.corrections_for_rows(
                cache=site_stream.cache, trace=clean, frame=rows
            )
            level_s, anomaly_s = stream_module.corrections_for_rows(
                cache=site_stream.cache, trace=stepped, frame=rows
            )
            rises[name].append((level_s + anomaly_s) - (level_c + anomaly_c))
    recovery = {name: float(np.concatenate(r).mean() / STEP_POINTS) for name, r in rises.items()}
    return {
        "recovery": recovery,
        "pre_minus_r0": recovery["PRE"] - recovery["R0"],
        "rows": int(sum(len(r) for r in rises["R0"])),
    }


def main() -> None:
    """Run both gates and write `work/gate.json`."""
    torch.set_num_threads(4)
    pooled = fit.load_pooled()
    g1, g2 = gate_network_learns(pooled=pooled), gate_filter_follows()
    g1["passes"] = bool(g1["recovery_mean"] >= 0.5)  # ty: ignore[unsupported-operator]
    g2["passes"] = bool(g2["pre_minus_r0"] >= -0.05)  # ty: ignore[unsupported-operator]
    result = {"G1": g1, "G2": g2}
    (common.WORK_DIR / "gate.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
