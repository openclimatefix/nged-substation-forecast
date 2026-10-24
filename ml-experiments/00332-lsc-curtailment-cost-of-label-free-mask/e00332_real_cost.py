# ruff: noqa: E501
"""Curtailment cost of the label-free mask at the bulk supply point D13 on the real record (Part 2).

The node flow is the bulk supply point's own metered flow. Each demand arm's node forecast is the sum
of the 7 child Primaries' arm locations plus quantile offsets fitted on the odd-month residual of
that sum against the metered flow. The metered farms A, C, and E and the other generators under the
node take experiment 243's H5 forecast and experiment 48's generator forecast, paired by valid time
and shared by every arm.

    PYTHONPATH=../00268-lsc-curtailment-metric-v2-member-wise-ens:../00309-lsc-two-forecast-gap-choosing-without-truth \
        uv run --no-sync python e00332_real_cost.py
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace
from typing import Final

import e00268_arms as arms_module  # ty: ignore[unresolved-import]
import e00268_inputs as inputs  # ty: ignore[unresolved-import]
import e00309_harness as h
import numpy as np
import polars as pl
from studies.node_curtailment import (
    NodeCurtailmentAssumptions,
    capacity_forecast,
    month_total_arrays,
    node_curtailment,
    scaled_limits,
    table_with_scaled_limits,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final = ML_EXPERIMENTS_DATA_DIR / "00332-lsc-curtailment-cost-of-label-free-mask"
CHILDREN: Final[tuple[int, ...]] = (7, 8, 9, 10, 11, 24, 25)
ARMS: Final[tuple[str, ...]] = ("C", "D", "A", "B", "E", "AF")
SKEGNESS: Final = inputs.SKEGNESS
LEAD_LO: Final = 24
LEAD_HI: Final = 72
EVEN_WINDOW_MONTHS: Final = {8, 10, 12, 2, 4, 6}
WINDOW_START: Final = dt.datetime(2025, 7, 2, tzinfo=dt.UTC)
WINDOW_END: Final = dt.datetime(2026, 7, 1, tzinfo=dt.UTC)
GRID_START: Final = np.datetime64("2022-01-01T00:00")
N_DRAWS: Final = 20_000
FRACTIONS: Final = (0.5, 0.75, 0.9)
CONTRASTS: Final = (("A", "C"), ("B", "C"), ("D", "C"), ("A", "B"), ("E", "C"), ("AF", "C"))


def load_locations() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return per-arm summed locations (origins by 48 leads, MW) and the group masks."""
    parts = [np.load(WORK / "locations" / f"series_{s}.npz") for s in CHILDREN]
    loc = {a: sum(p[f"loc_{a}"].astype(np.float64) for p in parts) for a in ARMS}
    masks = {
        "ordinary": np.logical_and.reduce([p["ordinary"] for p in parts]),
        "after48": np.logical_or.reduce([p["after48"] for p in parts]),
        "after7d": np.logical_or.reduce([p["after7d"] for p in parts]),
        "masked": np.logical_or.reduce([p["masked_target"] for p in parts]),
    }
    return loc, masks


def node_actual() -> np.ndarray:
    """The bulk supply point's metered flow on the 327 grid, MW, NaN where absent."""
    node_series = 2
    flows = inputs.read_flows(
        table=type("T", (), {"nodes": pl.DataFrame({"time_series_id": [node_series]})})()
    )
    grid = GRID_START + np.arange(h.N_GRID) * np.timedelta64(30, "m")
    out = np.full(h.N_GRID, np.nan)
    stamps = flows["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[m]")
    idx = ((stamps - GRID_START) // np.timedelta64(30, "m")).astype(np.int64)
    ok = (idx >= 0) & (idx < h.N_GRID)
    out[idx[ok]] = flows["flow_mw"].to_numpy()[ok]
    del grid
    return out


def block_bootstrap(*, sums: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """Total over resampled whole months."""
    n = len(sums)
    draws = rng.integers(0, n, size=(N_DRAWS, n))
    boot = sums[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"est": float(sums.sum()), "lo": float(lo), "hi": float(hi), "se": float(boot.std())}


def main() -> None:  # noqa: PLR0912, PLR0915
    """Build the frames, score them, and print the contrasts."""
    rng = np.random.default_rng(332)
    loc, masks = load_locations()
    actual_grid = node_actual()
    origins, _, targets = h.origin_grid()
    clipped = np.minimum(targets, h.N_GRID - 1)[:, LEAD_LO:LEAD_HI]
    valid_idx = targets[:, LEAD_LO:LEAD_HI]
    actual = actual_grid[clipped]
    actual[valid_idx >= h.N_GRID] = np.nan
    band = h.band_grid()[clipped]
    month = h.utc_month_grid()[clipped]
    lead_band = h.lead_band_of_lead()[LEAD_LO:LEAD_HI][None, :].repeat(len(origins), axis=0)
    odd = np.isin(month, list(h.TUNING_MONTHS))
    target_time = GRID_START + valid_idx.astype("int64") * np.timedelta64(30, "m")
    in_window = (target_time >= np.datetime64("2025-07-02T00:00")) & (
        target_time < np.datetime64("2026-07-01T00:00")
    )
    even_ok = ~odd & np.isin(month, list(EVEN_WINDOW_MONTHS)) & in_window & (valid_idx < h.N_GRID)
    scored = even_ok & np.isfinite(actual) & ~masks["masked"]
    print("even-month window rows with a node reading:", int((even_ok & np.isfinite(actual)).sum()))
    print(
        "rows excluded because a child's span or label masks the target:",
        int((even_ok & np.isfinite(actual) & masks["masked"]).sum()),
    )
    print("scored rows:", int(scored.sum()))
    for lag in (-1, 0, 1):
        s = sum(
            np.load(
                WORK.parent
                / "00327-lsc-label-free-detect-mask-train-pipeline"
                / "power"
                / f"power_{c}.npy"
            )
            for c in CHILDREN
        )
        a = actual_grid
        x = np.roll(s, lag)
        ok = np.isfinite(x) & np.isfinite(a)
        print(
            "lag",
            lag,
            "corr children sum vs node flow",
            round(float(np.corrcoef(x[ok], a[ok])[0, 1]), 4),
        )
    # offsets per arm: odd-month residual quantiles by (hour band, lead band)
    quant: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm in ARMS:
        resid = actual - loc[arm]
        use = odd & np.isfinite(resid) & (valid_idx < h.N_GRID)
        off05 = np.zeros((len(np.unique(band)) + 1, 3))
        off95 = np.zeros_like(off05)
        for b in range(off05.shape[0]):
            for lb in range(3):
                cell = use & (band == b) & (lead_band == lb)
                if cell.sum() >= 50:
                    off05[b, lb], off95[b, lb] = np.quantile(resid[cell], [0.05, 0.95])
        quant[arm] = (off05, off95)
    capacity_mw = inputs.read_capacity_mw()
    ids = set(
        inputs.read_arm_frames(names=("gen_monthly",))["gen_monthly"]["time_series_id"]
        .unique()
        .to_list()
    )
    full = inputs.build_pilot_table(capacity_mw=capacity_mw, forecast_ids=ids)
    scoring = inputs.scoring_table(table=full)
    keep = [SKEGNESS]
    scoring = type(scoring)(
        nodes=scoring.nodes.filter(pl.col("node_id").is_in(keep)),
        limits=scoring.limits.filter(pl.col("node_id").is_in(keep)),
        generators=scoring.generators.filter(pl.col("node_id").is_in(keep)),
    )
    node_ts = int(scoring.nodes["time_series_id"][0])
    flows = inputs.read_flows(table=scoring)
    # keys of the scored rows
    o_i, l_i = np.nonzero(scored)
    init_time = GRID_START + origins[o_i].astype("int64") * np.timedelta64(30, "m")
    valid_time = target_time[o_i, l_i]
    keys = pl.DataFrame(
        {
            "power_fcst_init_time": init_time.astype("datetime64[us]"),
            "valid_time": valid_time.astype("datetime64[us]"),
        }
    ).with_columns(
        pl.col("power_fcst_init_time").dt.replace_time_zone("UTC"),
        pl.col("valid_time").dt.replace_time_zone("UTC"),
        seed=pl.lit(0, dtype=pl.Int64),
    )
    row_keys = keys.with_columns(
        ordinary=pl.Series(masks["ordinary"][o_i, l_i]),
        after48=pl.Series(masks["after48"][o_i, l_i]),
        after7d=pl.Series(masks["after7d"][o_i, l_i]),
    )
    node_frames: dict[str, pl.DataFrame] = {}
    for arm in ARMS:
        o5, o95 = quant[arm]
        base = loc[arm][o_i, l_i]
        q05 = base + o5[band[o_i, l_i], lead_band[o_i, l_i]]
        q95 = base + o95[band[o_i, l_i], lead_band[o_i, l_i]]
        node_frames[arm] = row_keys.with_columns(
            time_series_id=pl.lit(node_ts, dtype=pl.Int64),
            actual=pl.Series(actual[o_i, l_i]),
            q05=pl.Series(q05),
            q95=pl.Series(q95),
        )
    node_frames["perfect"] = node_frames["C"].with_columns(
        q05=pl.col("actual"), q95=pl.col("actual")
    )
    # generators: farms and the other generators under the node, paired by valid time
    sites = inputs.farm_sites()
    unique_keys = keys.select("power_fcst_init_time", "valid_time", "seed")
    daylight = arms_module._daylight_by_valid_time(
        valid_times=unique_keys["valid_time"], sites=sites
    )
    farm_power = inputs.read_farm_power_mw()
    capacity_by_series = dict(
        zip(
            scoring.generators["generator_id"].cast(pl.Int64).to_list(),
            scoring.generators["capacity_mw"].to_list(),
            strict=True,
        )
    )
    forecasts = inputs.read_farm_forecasts()
    farm_rows = arms_module._farm_rows(
        forecast=forecasts[arms_module.H5],
        keys=unique_keys,
        daylight=daylight,
        farm_power=farm_power,
        sites=sites,
        capacity_by_series=capacity_by_series,
    )
    _, others = arms_module._node_and_other_generators(table=scoring, capacity_mw=capacity_mw)
    generator_ids = set(scoring.generators["generator_id"].to_list())
    columns = [
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        "seed",
        "actual",
        *inputs.QUANTILES,
    ]
    others_paired = (
        others.filter(pl.col("time_series_id").is_in(generator_ids))
        .unique(subset=["time_series_id", "valid_time"], keep="first")
        .drop("power_fcst_init_time", "seed", strict=False)
        .join(unique_keys, on="valid_time", how="inner")
        .with_columns(pl.col("seed").cast(pl.Int64))
    )
    others_full = others_paired.select(
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        "seed",
        "actual",
        *(
            pl.col(q) if q in others_paired.columns else pl.lit(None, dtype=pl.Float64).alias(q)
            for q in inputs.QUANTILES
        ),
    )
    farms = farm_rows.select(columns).with_columns(pl.col("seed").cast(pl.Int64))
    capacities = scoring.generators.select(
        pl.col("generator_id").cast(pl.Int64).alias("time_series_id"),
        pl.col("capacity_mw").alias("capacity_limit_mw"),
    )
    gens = arms_module._clip_to_capacity(
        frame=pl.concat(
            [
                farms.filter(pl.col("time_series_id").is_in(generator_ids)),
                others_full.select(columns),
            ],
            how="vertical_relaxed",
        ),
        capacities=capacities,
    )
    gen_frames = dict.fromkeys(ARMS, gens)
    gen_frames["perfect"] = arms_module._clip_to_capacity(
        frame=gens.with_columns(*(pl.col("actual").alias(q) for q in inputs.QUANTILES)),
        capacities=capacities,
    )
    print(
        "generators under the node:",
        sorted(generator_ids),
        "with forecast rows:",
        gens["time_series_id"].unique().sort().to_list(),
    )
    groups = {
        "G0_all": None,
        "G1_ordinary": "ordinary",
        "G4_after48": "after48",
        "G5_after7d": "after7d",
    }
    out: dict[str, dict] = {}
    for fraction in FRACTIONS:
        scaled = scaled_limits(table=scoring, flows=flows, cutoff=WINDOW_START, fraction=fraction)
        table = table_with_scaled_limits(table=scoring, scaled=scaled, fraction=fraction)
        for gname, col in groups.items():
            nf = {a: (f if col is None else f.filter(pl.col(col))) for a, f in node_frames.items()}
            key_cols = ["power_fcst_init_time", "valid_time", "seed"]
            if col is None:
                gf = gen_frames
            else:
                sel = nf["C"].select(key_cols)
                gf = {a: f.join(sel, on=key_cols, how="semi") for a, f in gen_frames.items()}
            # the NGED baseline arm: every generator assumed at capacity, node as arm C
            nf["baseline"] = nf["C"]
            gf["baseline"] = capacity_forecast(generator_frame=gf["C"], table=table)
            result = node_curtailment(
                table=table,
                node_forecasts=nf,
                generator_forecasts=gf,
                assumptions=replace(
                    NodeCurtailmentAssumptions(use_generators=True),
                    lead_min_hours=12.0,
                    lead_max_hours=36.0,
                ),
            )
            if fraction == 0.5:
                print(
                    gname,
                    "rows per arm:",
                    {a: f.height for a, f in nf.items()},
                    "month totals:",
                    result.month_totals.filter(pl.col("arm") == "C")
                    .select("month", "n_node_half_hours", "energy_proxy_mwh_equivalent")
                    .sort("month")
                    .to_dicts(),
                )
            arrays = {}
            for arm, frame_arm in nf.items():
                if frame_arm.is_empty():
                    break
                sums, counts = month_total_arrays(
                    month_totals=result.month_totals, arm=arm, column="energy_proxy_mwh_equivalent"
                )
                arrays[arm] = sums[0] * 2000.0
            if len(arrays) < len(nf):
                print("group", gname, "has no rows")
                continue
            perfect = arrays["perfect"]
            n = float(counts.sum())
            rec = {"n_half_hours": n, "arms": {}, "contrasts": {}}
            for arm, a in arrays.items():
                rec["arms"][arm] = {"cost": float(a.sum()), "avoidable": float((a - perfect).sum())}
            for x, y in CONTRASTS:
                if x in arrays and y in arrays and len(arrays[x]) == len(arrays[y]):
                    r = block_bootstrap(sums=arrays[x] - arrays[y], rng=rng)
                    r["mde"] = 2.8 * r["se"]
                    r["base"] = float(arrays[y].sum())
                    rec["contrasts"][f"{x}-{y}"] = r
            out[f"f{fraction}|{gname}"] = rec
            if fraction == 0.5:
                print(
                    f"\nf=0.5 {gname}: node half-hours {n:.0f}, event census rows above limit: {result.event_census['n_exceedances'].sum()}"
                )
                for arm, v in rec["arms"].items():
                    print(
                        f"   {arm:9s} cost GBP {v['cost']:12,.0f} avoidable {v['avoidable']:12,.0f}"
                    )
                for k, r in rec["contrasts"].items():
                    print(
                        f"   {k:6s} {r['est']:+12,.0f} [{r['lo']:+12,.0f}, {r['hi']:+12,.0f}] ({100 * r['est'] / r['base']:+.1f}% of {k.split('-')[1]}), smallest resolvable {r['mde']:,.0f}"
                    )
    (WORK / "real_cost.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
