# ruff: noqa: E501, PLR0912, PLR0915
"""Score the zero-shot Chronos-2 arms (Z0, Z1) against Primary demand research default 3 at 0.5 to 240 hours.

Rows, benchmarks, and metric are those of experiment 322: six Primaries, scored year, rows with a
reading outside the hand-filtered events, the quantile score over 11 levels in points of effective
capacity. The default is experiment 251's arm 1 below 168 hours and experiment 322's d1 above.
Intervals resample the 12 months of the forecast run's initialisation time (20,000 draws).

    uv run --no-sync python ml-experiments/00333-*/e00333_score.py
"""

import json

import e00317_score as s317  # ty: ignore[unresolved-import]
import e00319_score as s319  # ty: ignore[unresolved-import]
import e00320_common as c320  # ty: ignore[unresolved-import]
import e00322_common as c322  # ty: ignore[unresolved-import]
import e00333_common as common
import numpy as np
import polars as pl

QUANTILES = common.QUANTILES
KEYS = ["site", "power_fcst_init_time", "valid_time"]
GATE_ALL, GATE_LONG, GATE_TOLERANCE = 2.727, 3.077, 0.01
LEARNED = ("z0", "z1")
REFS = ("default", "manual_heuristic", "climatology")
CONTRASTS = (
    ("z0", "default"), ("z1", "default"), ("z1", "z0"),
    ("z0", "manual_heuristic"), ("z1", "manual_heuristic"), ("z0", "climatology"), ("z1", "climatology"),
)  # fmt: skip


def assemble() -> pl.DataFrame:
    """Return the scored rows with the stored references and the Chronos-2 forecasts."""
    chain = s317.load_chain()
    manual = s317.read_reference(name="ref_manual_heuristic", tag="mh")
    clim = s317.read_reference(name="ref_climatology", tag="cl")
    rows = chain.join(manual, on=s317.common.KEYS, how="inner", validate="1:1")
    rows = rows.join(clim, on=s317.common.KEYS, how="inner", validate="1:1")
    rows = rows.filter(pl.col("lead_hours") <= c320.HORIZON_HOURS)
    stored = (
        pl.read_parquet(c320.c319.E251_ARMS / "rows.parquet").select(KEYS).with_row_index("i251")
    )
    rows = rows.join(stored, on=KEYS, how="left", validate="1:1")
    long_arms = pl.read_parquet(
        c322.ARMS_DIR / "scored_long_lead.parquet",
        columns=[*KEYS, *[f"d1_{q}" for q in QUANTILES]],
    )
    rows = rows.join(long_arms, on=KEYS, how="left", validate="1:1")
    z = pl.concat(
        [
            pl.read_parquet(common.FORECAST_DIR / f"z0_{s}.parquet").join(
                pl.read_parquet(common.FORECAST_DIR / f"z1_{s}.parquet"),
                on=KEYS,
                how="inner",
                validate="1:1",
            )
            for s in common.SCORED_SITES
        ]
    )
    rows = rows.join(z, on=KEYS, how="left", validate="1:1")
    rows = s317.flags(rows=rows)
    long = rows["lead_hours"].to_numpy() > c320.STORED_HORIZON_HOURS
    keep = (
        (long | rows["i251"].is_not_null().to_numpy())
        & rows["z0_q50"].is_not_null().to_numpy()
        & rows["z1_q50"].is_not_null().to_numpy()
    )
    return rows.filter(pl.Series(keep))


def main() -> None:
    """Score every arm and contrast, write the report, and print the gate and the band table."""
    rows = assemble()
    lead = rows["lead_hours"].to_numpy()
    long = lead > c320.STORED_HORIZON_HOURS
    idx = rows["i251"].fill_null(0).to_numpy()
    short_q = np.load(c320.c319.E251_ARMS / "arm1.npy")[idx].astype(np.float64)
    d1 = rows.select([f"d1_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    quantiles = {
        "default": np.where(long[:, None], d1, short_q),
        "manual_heuristic": rows.select([f"mh_{q}" for q in QUANTILES])
        .to_numpy()
        .astype(np.float64),
        "climatology": rows.select([f"cl_{q}" for q in QUANTILES]).to_numpy().astype(np.float64),
    }
    for arm in LEARNED:
        quantiles[arm] = (
            rows.select([f"{arm}_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
        )
    for arm in (*LEARNED, "default"):
        if not np.isfinite(quantiles[arm]).all():
            msg = f"arm {arm} has a non-finite quantile"
            raise ValueError(msg)
    actual = rows["actual"].to_numpy().astype(np.float64)
    site = rows["site"].to_numpy()
    threshold = np.array([np.quantile(actual[site == s], 0.9) for s in c320.SCORED_SITES])
    thr = threshold[[c320.SCORED_SITES.index(s) for s in site]]
    pin = {k: s317.pin(actual=actual, q=v) for k, v in quantiles.items()}
    loss = {k: 2.0 * v.mean(axis=1) for k, v in pin.items()}
    tw = {
        k: 2.0 * np.maximum(
            c320.LEVELS * (np.maximum(actual, thr)[:, None] - np.maximum(v, thr[:, None])),
            (c320.LEVELS - 1.0) * (np.maximum(actual, thr)[:, None] - np.maximum(v, thr[:, None])),
        ).mean(axis=1)
        for k, v in quantiles.items()
    }  # fmt: skip
    month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    band = np.clip(
        np.searchsorted(s317.BAND_EDGES, lead, side="left") - 1, 0, len(s317.BAND_LABELS) - 1
    )
    cold = rows["cold_winter_night"].to_numpy()
    sunny = rows["sunny_daylight"].to_numpy()
    every = np.ones(rows.height, dtype=bool)
    groups = {
        "all rows": every,
        "cold winter nights": cold,
        "sunny daylight": sunny,
        "ordinary rows": ~cold & ~sunny,
    }
    windows = {
        "0-240 h": every,
        "0-168 h": ~long,
        "168-240 h": long,
        "24-120 h": (lead > 24) & (lead <= 120),
    }
    arms = (*LEARNED, *REFS)
    rep: dict = {"n_rows": rows.height}
    rep["gate"] = {
        "default_0_240": float(loss["default"].mean()), "expected_0_240": GATE_ALL,
        "default_168_240": float(loss["default"][long].mean()), "expected_168_240": GATE_LONG,
    }  # fmt: skip
    rep["gate"]["pass"] = bool(
        abs(rep["gate"]["default_0_240"] - GATE_ALL) <= GATE_TOLERANCE
        and abs(rep["gate"]["default_168_240"] - GATE_LONG) <= GATE_TOLERANCE
    )
    rep["by_band"] = {}
    for i, label in enumerate(s317.BAND_LABELS):
        mask = band == i
        entry: dict = {"n": int(mask.sum())}
        for arm in arms:
            entry[arm] = s319.mean_ci(values=loss[arm], mask=mask, blocks=month)
        for a, b in CONTRASTS:
            entry[f"{a}_vs_{b}"] = s319.contrast(
                treat=loss[a], ref=loss[b], mask=mask, blocks=month
            )
        rep["by_band"][label] = entry
    no_d10 = site != "D10"
    rep["by_group"] = {}
    for variant, vmask in (("with D10", every), ("without D10", no_d10)):
        rep["by_group"][variant] = {}
        for gname, gmask in groups.items():
            rep["by_group"][variant][gname] = {}
            for wname, wmask in windows.items():
                mask = gmask & wmask & vmask
                if mask.sum() < 50:
                    continue
                e: dict = {"n": int(mask.sum())}
                for arm in arms:
                    e[arm] = s319.mean_ci(values=loss[arm], mask=mask, blocks=month)
                for a, b in CONTRASTS:
                    e[f"{a}_vs_{b}"] = s319.contrast(
                        treat=loss[a], ref=loss[b], mask=mask, blocks=month
                    )
                for a in LEARNED:
                    e[f"{a}_twcrps_vs_default"] = s319.contrast(
                        treat=tw[a], ref=tw["default"], mask=mask, blocks=month
                    )
                rep["by_group"][variant][gname][wname] = e  # fmt: skip
    rep["tails"] = {
        wname: {
            arm: {
                "cover_q01": float((actual < quantiles[arm][:, 0])[wmask].mean()),
                "cover_q05": float((actual < quantiles[arm][:, 1])[wmask].mean()),
                "cover_q95": float((actual < quantiles[arm][:, 9])[wmask].mean()),
                "cover_q99": float((actual < quantiles[arm][:, 10])[wmask].mean()),
                "pin_q01": float(pin[arm][wmask, 0].mean()), "pin_q05": float(pin[arm][wmask, 1].mean()),
                "pin_q95": float(pin[arm][wmask, 9].mean()), "pin_q99": float(pin[arm][wmask, 10].mean()),
                "twcrps": float(tw[arm][wmask].mean()),
            }
            for arm in arms
        }
        for wname, wmask in windows.items()
    }  # fmt: skip
    rep["leave_one_out"] = {
        s: {f"{a}_vs_default": s319.contrast(treat=loss[a], ref=loss["default"], mask=site != s, blocks=month) for a in LEARNED}
        for s in c320.SCORED_SITES
    }  # fmt: skip
    rep["by_site"] = {
        s: {arm: float(loss[arm][site == s].mean()) for arm in arms} for s in c320.SCORED_SITES
    }
    mean_cols = {arm: rows[f"{arm}_mean"].to_numpy().astype(np.float64) for arm in LEARNED}
    rep["median_minus_mean_by_site"] = {
        arm: {s: float((quantiles[arm][site == s, 5] - mean_cols[arm][site == s]).mean()) for s in c320.SCORED_SITES}
        for arm in LEARNED
    }  # fmt: skip
    rep["mae_median"] = {
        wname: {arm: float(np.abs(actual - quantiles[arm][:, 5])[wmask].mean()) for arm in arms}
        for wname, wmask in windows.items()
    }  # fmt: skip
    rep["mae_mean"] = {
        wname: {arm: float(np.abs(actual - mean_cols[arm])[wmask].mean()) for arm in LEARNED}
        for wname, wmask in windows.items()
    }  # fmt: skip
    gap0 = loss["z0"] - loss["default"]
    gap1 = loss["z1"] - loss["default"]
    for wname in ("24-120 h", "0-240 h"):
        m = windows[wname]
        g0, g1 = float(gap0[m].mean()), float(gap1[m].mean())
        rep[f"gap_closed_by_z1_{wname}"] = {
            "z0_gap": g0,
            "z1_gap": g1,
            "share_closed": (g0 - g1) / g0 if g0 else None,
        }
    common.write_json_atomic(path=common.REPORT_DIR / "report.json", content=rep)
    rows.select(KEYS, "lead_hours", "actual", "cold_winter_night", "sunny_daylight").with_columns(
        *[pl.Series(f"loss_{arm}", loss[arm].astype(np.float32)) for arm in arms]
    ).write_parquet(common.REPORT_DIR / "scored_rows.parquet")  # fmt: skip
    print(json.dumps(rep["gate"], default=float), flush=True)
    for label, e in rep["by_band"].items():
        print(label, e["n"], *[f"{a} {e[a]['est']:.3f}" for a in arms], flush=True)
    for wname in ("0-240 h", "24-120 h"):
        for variant in ("with D10", "without D10"):
            e = rep["by_group"][variant]["all rows"][wname]
            print(wname, variant, *[f"{k} {e[k]['diff']:+.3f} [{e[k]['diff_lo_95']:+.3f},{e[k]['diff_hi_95']:+.3f}]" for k in ("z0_vs_default", "z1_vs_default", "z1_vs_z0")], flush=True)  # fmt: skip
    print(rep["gap_closed_by_z1_24-120 h"], rep["gap_closed_by_z1_0-240 h"], flush=True)


if __name__ == "__main__":
    main()
