# ruff: noqa: E501, PLR0912, PLR0915
"""Score the default and the two arms to 240 hours against the manual heuristic and climatology.

Rows, metric, and benchmark rows are experiment 317's: six Primaries, scored year, rows with a
reading outside the hand-filtered events. Below 168 hours the default is experiment 251's A+H+S
and both arms equal it; above 168 hours the default is experiment 317's convex chain and the arms
are `e00319_arms.py`'s x1 and x2. Intervals resample the 12 months of the forecast run's
initialisation time (20,000 draws) and are read at 95% and at the 99.5% level that two
pre-registered arms set (1 - 0.01 / 2).

    PYTHONPATH=ml-experiments/00317-primary-demand-default-1-at-7-to-10-days \
        uv run --no-sync python ml-experiments/00319-*/e00319_score.py
"""

import json

import e00317_score as s317  # ty: ignore[unresolved-import]
import e00319_common as common
import numpy as np
import polars as pl

N_RESAMPLES = 20_000
ARMS = ("default", "x1", "x2", "xr")
BAND_EDGES = s317.BAND_EDGES
BAND_LABELS = s317.BAND_LABELS
TAIL_COLUMNS = (("q01", 0), ("q99", 10))
RNG = np.random.default_rng(common.EXPERIMENT_ID)


def resample(*, blocks: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return block codes, block count, and the bootstrap draws of block indices for a mask."""
    labels, codes = np.unique(blocks[mask], return_inverse=True)
    n = len(labels)
    draws = RNG.integers(0, n, size=(N_RESAMPLES, n))
    return codes, np.array([n]), draws


def contrast(*, treat: np.ndarray, ref: np.ndarray, mask: np.ndarray, blocks: np.ndarray) -> dict:
    """Return the mean difference (treat minus ref) and the gain with 95% and 99.5% intervals."""
    if mask.sum() == 0:
        return {"n": 0}
    codes, count, draws = resample(blocks=blocks, mask=mask)
    n = int(count[0])
    st = np.bincount(codes, weights=treat[mask], minlength=n)
    sr = np.bincount(codes, weights=ref[mask], minlength=n)
    nn = np.bincount(codes, minlength=n).astype(np.float64)
    c = nn[draws].sum(axis=1)
    mt, mr = st[draws].sum(axis=1) / c, sr[draws].sum(axis=1) / c
    diff, gain = mt - mr, 1.0 - mt / mr
    pct = {"95": (2.5, 97.5), "995": (0.25, 99.75)}
    out = {
        "n": int(mask.sum()),
        "treat": float(st.sum() / nn.sum()),
        "ref": float(sr.sum() / nn.sum()),
        "diff": float((st.sum() - sr.sum()) / nn.sum()),
        "gain": float(1.0 - st.sum() / sr.sum()),
        "diff_hi_one_sided_95": float(np.percentile(diff, 95.0)),
    }
    for tag, (lo, hi) in pct.items():
        out[f"diff_lo_{tag}"], out[f"diff_hi_{tag}"] = (
            float(v) for v in np.percentile(diff, [lo, hi])
        )
        out[f"gain_lo_{tag}"], out[f"gain_hi_{tag}"] = (
            float(v) for v in np.percentile(gain, [lo, hi])
        )
    return out


def mean_ci(*, values: np.ndarray, mask: np.ndarray, blocks: np.ndarray) -> dict:
    """Return a mean over a mask with a 95% block-resampled interval."""
    if mask.sum() == 0:
        return {"n": 0}
    codes, count, draws = resample(blocks=blocks, mask=mask)
    n = int(count[0])
    sums = np.bincount(codes, weights=values[mask], minlength=n)
    nn = np.bincount(codes, minlength=n).astype(np.float64)
    res = sums[draws].sum(axis=1) / nn[draws].sum(axis=1)
    lo, hi = np.percentile(res, [2.5, 97.5])
    return {
        "est": float(sums.sum() / nn.sum()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(mask.sum()),
    }


def load_chain_period(*, period: str) -> pl.DataFrame:
    """Return a period's chain rows at the six Primaries with a reading outside the hand-filtered events."""
    pieces = [
        pl.read_parquet(common.CHAIN_DIR / f"{site}.parquet")
        .filter(pl.col("period") == period)
        .with_columns(site=pl.lit(site))
        for site in common.SCORED_SITES
    ]
    return pl.concat(pieces).filter(~pl.col("switching_mask") & pl.col("actual").is_finite())


def replicate_report() -> dict:
    """Return the second period's contrasts at 168 to 240 hours (pooled, sunny daylight, by season)."""
    chain = load_chain_period(period="replicate")
    arms = pl.read_parquet(common.ARMS_DIR / "replicate_long_lead.parquet")
    keys = ["site", "power_fcst_init_time", "valid_time"]
    rows = chain.join(
        arms.select(
            *keys,
            *[f"{a}_{q}" for a in ("x1", "x2", "xr") for q in common.QUANTILES],
            "heuristic_median",
        ),
        on=keys,
        how="inner",
        validate="1:1",
    ).filter(pl.col("lead_hours") > common.STORED_HORIZON_HOURS)
    sunny = pl.read_parquet(common.E178_ROWS, columns=["site", "valid_time", "sunny"]).unique(
        subset=["site", "valid_time"]
    )
    daylight = pl.read_parquet(
        common.E220_ARMS / "rows.parquet", columns=["site", "valid_time", "daylight"]
    ).unique(subset=["site", "valid_time"])
    rows = rows.join(sunny, on=["site", "valid_time"], how="left", validate="m:1").join(
        daylight, on=["site", "valid_time"], how="left", validate="m:1"
    )
    actual = rows["actual"].to_numpy().astype(np.float64)
    q = {"default": rows.select(list(common.QUANTILES)).to_numpy().astype(np.float64)}
    for a in ("x1", "x2", "xr"):
        q[a] = rows.select([f"{a}_{k}" for k in common.QUANTILES]).to_numpy().astype(np.float64)
    loss = {k: 2.0 * s317.pin(actual=actual, q=v).mean(axis=1) for k, v in q.items()}
    month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    vmonth = rows["valid_time"].dt.convert_time_zone("Europe/London").dt.month().to_numpy()
    winter = np.isin(vmonth, list(common.WINTER_MONTHS))
    sunny_day = (rows["sunny"] & rows["daylight"]).fill_null(False).to_numpy()
    masks = {
        "all rows": np.ones(rows.height, dtype=bool),
        "sunny daylight": sunny_day,
        "winter (Nov-Mar)": winter,
        "April to October": ~winter,
    }
    out: dict = {"n_rows": rows.height, "months": sorted(set(month))}
    for g, mask in masks.items():
        out[g] = {
            f"{a}_vs_default": contrast(treat=loss[a], ref=loss["default"], mask=mask, blocks=month)
            for a in ("x1", "x2", "xr")
        }
        out[g]["default"] = float(loss["default"][mask].mean())
    return out


def main() -> None:
    """Assemble the rows, run the gates, score every contrast, and write the report."""
    chain = s317.load_chain()
    manual = s317.read_reference(name="ref_manual_heuristic", tag="mh")
    clim = s317.read_reference(name="ref_climatology", tag="cl")
    rows = chain.join(manual, on=s317.common.KEYS, how="inner", validate="1:1")
    rows = rows.join(clim, on=s317.common.KEYS, how="inner", validate="1:1")
    rows = rows.filter(pl.col("lead_hours") <= common.HORIZON_HOURS)
    rows_after_reference_joins = rows.height
    stored = (
        pl.read_parquet(common.E251_ARMS / "rows.parquet")
        .select(["site", "power_fcst_init_time", "valid_time"])
        .with_row_index("i251")
    )
    keys = ["site", "power_fcst_init_time", "valid_time"]
    rows = rows.join(stored, on=keys, how="left", validate="1:1")
    long_arms = pl.read_parquet(
        common.ARMS_DIR / "scored_long_lead.parquet",
        columns=[
            *keys,
            "heuristic_median",
            *[f"{a}_{q}" for a in ("x1", "x2", "xr") for q in common.QUANTILES],
        ],
    )
    rows = rows.join(long_arms, on=keys, how="left", validate="1:1")
    rows = s317.flags(rows=rows)
    lead = rows["lead_hours"].to_numpy()
    long = lead > common.STORED_HORIZON_HOURS
    in251 = rows["i251"].is_not_null().to_numpy()
    keep = long | in251
    rows_dropped_short = int((~keep).sum())
    rows = rows.filter(pl.Series(keep))
    lead = rows["lead_hours"].to_numpy()
    long = lead > common.STORED_HORIZON_HOURS
    n = rows.height
    arms251 = {k: np.load(common.E251_ARMS / f"{k}.npy") for k in ("arm1", "bg2")}
    idx = rows["i251"].fill_null(0).to_numpy()
    chain_q = rows.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    default_q = np.where(long[:, None], chain_q, arms251["arm1"][idx].astype(np.float64))
    x1_q = np.where(
        long[:, None],
        rows.select([f"x1_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64),
        default_q,
    )
    x2_q = np.where(
        long[:, None],
        rows.select([f"x2_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64),
        default_q,
    )
    xr_q = np.where(
        long[:, None],
        rows.select([f"xr_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64),
        default_q,
    )
    mh_q = rows.select([f"mh_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)
    cl_q = rows.select([f"cl_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)
    bg2_q = np.where(long[:, None], np.nan, arms251["bg2"][idx].astype(np.float64))
    actual = rows["actual"].to_numpy().astype(np.float64)
    site = rows["site"].to_numpy()
    # Threshold for the twCRPS stand-in: each site's 90th percentile of the scored readings.
    threshold = np.array([np.quantile(actual[site == s], 0.9) for s in common.SCORED_SITES])
    thr = threshold[[common.SCORED_SITES.index(s) for s in site]]
    quantiles = {
        "default": default_q,
        "x1": x1_q,
        "x2": x2_q,
        "xr": xr_q,
        "manual_heuristic": mh_q,
        "climatology": cl_q,
        "BG2": bg2_q,
    }
    pin = {k: s317.pin(actual=actual, q=v) for k, v in quantiles.items()}
    loss = {k: 2.0 * v.mean(axis=1) for k, v in pin.items()}
    tw = {
        k: 2.0
        * np.maximum(
            common.LEVELS * (np.maximum(actual, thr)[:, None] - np.maximum(v, thr[:, None])),
            (common.LEVELS - 1.0)
            * (np.maximum(actual, thr)[:, None] - np.maximum(v, thr[:, None])),
        ).mean(axis=1)
        for k, v in quantiles.items()
    }
    month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    valid_month = rows["valid_time"].dt.convert_time_zone("Europe/London").dt.month().to_numpy()
    winter = np.isin(valid_month, list(common.WINTER_MONTHS))
    band = np.clip(np.searchsorted(BAND_EDGES, lead, side="left") - 1, 0, len(BAND_LABELS) - 1)
    rep: dict = {
        "n_rows": n,
        "rows_dropped_short_not_in_251": rows_dropped_short,
        "n_long": int(long.sum()),
    }
    # Gates: the default below 168 h reproduces experiment 251 (6-Primary score 2.579) and 317's chain.
    short = ~long
    rep["gate"] = {
        "default_below_168_six_primaries": float(loss["default"][short].mean()),
        "x1_equals_default_below_168": float(np.abs(x1_q[short] - default_q[short]).max()),
        "x2_equals_default_below_168": float(np.abs(x2_q[short] - default_q[short]).max()),
        "chain_168_240_score": float(loss["default"][long].mean()),
        "x1_long_rows_missing": int(rows.filter(pl.Series(long))["x1_q50"].is_null().sum()),
        "x2_long_rows_missing": int(rows.filter(pl.Series(long))["x2_q50"].is_null().sum()),
        "xr_long_rows_missing": int(rows.filter(pl.Series(long))["xr_q50"].is_null().sum()),
        "rows_after_reference_joins": rows_after_reference_joins,
    }
    heuristic_gate = json.loads((common.ARMS_DIR / "gates_arms.json").read_text())
    rep["gate"]["heuristic_rebuild_vs_experiment_87"] = heuristic_gate
    groups = {
        "all rows": np.ones(n, dtype=bool),
        "cold winter nights": rows["cold_winter_night"].to_numpy(),
        "sunny daylight": rows["sunny_daylight"].to_numpy(),
        "winter (Nov-Mar)": winter,
        "April to October": ~winter,
    }
    windows = {
        "0-240 h": np.ones(n, dtype=bool),
        "0-168 h": ~long,
        "168-240 h": long,
        "168-216 h": long & (lead <= 216.0),
        "216-240 h": lead > 216.0,
    }
    rep["group_rows"] = {k: int(v.sum()) for k, v in groups.items()}
    contrasts = [
        ("x1", "default"),
        ("x2", "default"),
        ("x2", "manual_heuristic"),
        ("x1", "manual_heuristic"),
        ("default", "manual_heuristic"),
        ("default", "climatology"),
        ("x2", "climatology"),
        ("x1", "climatology"),
        ("xr", "default"),
        ("x1", "xr"),
    ]
    # By 24-hour band.
    rep["by_band"] = {}
    for i, label in enumerate(BAND_LABELS):
        mask = band == i
        entry: dict = {"n": int(mask.sum())}
        for arm in (*ARMS, "manual_heuristic", "climatology", "BG2"):
            if np.isfinite(loss[arm][mask]).all():
                entry[arm] = mean_ci(values=loss[arm], mask=mask, blocks=month)
        for a, b in contrasts:
            entry[f"{a}_vs_{b}"] = contrast(treat=loss[a], ref=loss[b], mask=mask, blocks=month)
        rep["by_band"][label] = entry
    # By group and window.
    rep["by_group"] = {}
    for gname, gmask in groups.items():
        rep["by_group"][gname] = {}
        for wname, wmask in windows.items():
            mask = gmask & wmask
            if mask.sum() < 50:
                continue
            e: dict = {"n": int(mask.sum())}
            for arm in (*ARMS, "manual_heuristic", "climatology"):
                e[arm] = mean_ci(values=loss[arm], mask=mask, blocks=month)
            for a, b in contrasts:
                e[f"{a}_vs_{b}"] = contrast(treat=loss[a], ref=loss[b], mask=mask, blocks=month)
            for a in ("x1", "x2"):
                e[f"{a}_twcrps_vs_default"] = contrast(
                    treat=tw[a], ref=tw["default"], mask=mask, blocks=month
                )
                for t, pos in TAIL_COLUMNS:
                    e[f"{a}_{t}_pinball_diff"] = float(
                        pin[a][mask, pos].mean() - pin["default"][mask, pos].mean()
                    )
                    e[f"{a}_{t}_pinball_contrast"] = contrast(
                        treat=pin[a][:, pos], ref=pin["default"][:, pos], mask=mask, blocks=month
                    )
            rep["by_group"][gname][wname] = e
    # Tails and coverage by 24-hour band.
    rep["tails_by_band"] = {}
    for i, label in enumerate(BAND_LABELS):
        mask = band == i
        rep["tails_by_band"][label] = {
            arm: {
                "q01_below": float((actual < quantiles[arm][:, 0])[mask].mean()),
                "q99_below": float((actual < quantiles[arm][:, 10])[mask].mean()),
                "in90": float(
                    ((actual >= quantiles[arm][:, 1]) & (actual <= quantiles[arm][:, 9]))[
                        mask
                    ].mean()
                ),
                "q01_pinball": float(pin[arm][mask, 0].mean()),
                "q99_pinball": float(pin[arm][mask, 10].mean()),
                "mae_median": float(np.abs(actual - quantiles[arm][:, 5])[mask].mean()),
                "median_minus_mean_quantile": float(
                    (quantiles[arm][mask, 5] - quantiles[arm][mask].mean(axis=1)).mean()
                ),
            }
            for arm in (*ARMS, "manual_heuristic", "climatology")
        }
    # Leave one Primary out, 168-240 h, each arm against the default and the heuristic.
    rep["leave_one_out_168_240"] = {
        s: {
            f"{a}_vs_{b}": contrast(
                treat=loss[a], ref=loss[b], mask=long & (site != s), blocks=month
            )
            for a, b in (("x1", "default"), ("x2", "default"), ("x2", "manual_heuristic"))
        }
        for s in common.SCORED_SITES
    }
    rep["by_site_168_240"] = {
        s: {
            f"{a}_vs_default": contrast(
                treat=loss[a], ref=loss["default"], mask=long & (site == s), blocks=month
            )
            for a in ("x1", "x2")
        }
        for s in common.SCORED_SITES
    }
    # x1 sunny daylight at 168-240 against the heuristic, and the share of x2's gain in April to October.
    sunny = groups["sunny daylight"] & long
    rep["sunny_168_240_vs_heuristic"] = {
        a: contrast(treat=loss[a], ref=loss["manual_heuristic"], mask=sunny, blocks=month)
        for a in ARMS
    }
    gain_total = (loss["default"][long] - loss["x2"][long]).sum()
    gain_ao = (loss["default"][long & ~winter] - loss["x2"][long & ~winter]).sum()
    rep["x2_gain_share_april_to_october"] = (
        float(gain_ao / gain_total) if gain_total != 0 else float("nan")
    )
    # Checks: median minus mean forecast per site, and the mean-of-median error beside the contrasts.
    rep["median_minus_mean_by_site"] = {
        arm: {
            s: float((quantiles[arm][site == s, 5] - quantiles[arm][site == s].mean(axis=1)).mean())
            for s in common.SCORED_SITES
        }
        for arm in ARMS
    }
    rep["median_abs_error_168_240"] = {
        arm: float(np.abs(actual - quantiles[arm][:, 5])[long].mean())
        for arm in (*ARMS, "manual_heuristic", "climatology")
    }
    # The replicate year: factors and weights fitted on the tuning months, applied to the replicate months.
    rep["replicate"] = replicate_report()
    # Blend weights by band and season (month of fit) for the figure.
    common.write_json_atomic(path=common.REPORT_DIR / "report.json", content=rep)
    rows.select(
        "site",
        "power_fcst_init_time",
        "valid_time",
        "lead_hours",
        "actual",
        "cold_winter_night",
        "sunny_daylight",
    ).with_columns(
        *[pl.Series(f"{arm}_q50", quantiles[arm][:, 5].astype(np.float32)) for arm in ARMS],
        *[
            pl.Series(f"loss_{arm}", loss[arm].astype(np.float32))
            for arm in (*ARMS, "manual_heuristic", "climatology")
        ],
    ).write_parquet(common.REPORT_DIR / "scored_rows.parquet")
    print(json.dumps(rep["gate"], default=float), flush=True)
    for label, e in rep["by_band"].items():
        print(
            label,
            e["n"],
            *[
                f"{a} {e[a]['est']:.3f}"
                for a in (*ARMS, "manual_heuristic", "climatology")
                if a in e
            ],
            "x1-d",
            f"{e['x1_vs_default']['diff']:.4f}",
            "x2-d",
            f"{e['x2_vs_default']['diff']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
