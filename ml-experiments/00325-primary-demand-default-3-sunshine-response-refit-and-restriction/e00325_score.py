# ruff: noqa: E501, PLR0912, PLR0915
"""Score the default and the refitted sunshine-response arms to 240 hours against the manual heuristic.

Rows, metric, and benchmark rows are experiment 322's: six Primaries, scored year, rows with a
reading outside the hand-filtered events. The default is Primary demand research default 3 as
experiment 322 stored it: experiment 251's A+H+S below 168 hours and the daylight-only blend `d1`
above. A variant's quantiles below 168 hours are its `r2` plus experiment 220's median
correction `delta` (experiment 251's `arm1` minus `a_s`); above 168 hours they are its `d1`.
Intervals resample the 12 months of the forecast run's initialisation time (20,000 draws), and the
April-to-June subsets resample ISO weeks.

    PYTHONPATH=<folders of experiments 184, 202, 220, 251, 317, 319, 320, 322, 323> \
        uv run --no-sync python ml-experiments/00323-*/e00323_score.py
"""

import json

import e00317_score as s317  # ty: ignore[unresolved-import]
import e00319_score as s319  # ty: ignore[unresolved-import]
import e00320_common as c320  # ty: ignore[unresolved-import]
import e00322_score as s322  # ty: ignore[unresolved-import]
import e00323_common as common323  # ty: ignore[unresolved-import]
import e00325_common as common
import numpy as np
import polars as pl

QUANTILES = c320.QUANTILES
VARIANTS = ("f1_as_run", "f1r", "f1s", "f1sel")
ARMS = ("default", *VARIANTS)
common323_arms = common323.ARMS_DIR
REFERENCES = ("manual_heuristic", "climatology")
KEYS = ["site", "power_fcst_init_time", "valid_time"]
CONTRASTS = (
    ("f1_as_run", "default"),
    ("f1r", "default"),
    ("f1s", "default"),
    ("f1sel", "default"),
    ("f1r", "f1_as_run"),
    ("f1s", "f1r"),
    ("f1sel", "f1s"),
    ("f1s", "manual_heuristic"),
    ("default", "manual_heuristic"),
    ("default", "climatology"),
)
APRIL_TO_JUNE = (4, 5, 6)


SELECTION_MARGIN = 0.02
SELECTION_MONTHS = 12


def read_delta(*, variant: str) -> pl.DataFrame:
    """Return the three-seed mean of the network's median shift for every scored-year row."""
    frames = [
        pl.read_parquet(common.NET_DIR / variant / "preds" / f"arm2_seed{seed}.parquet").select(
            "site", "run", "valid_time", "delta"
        )
        for seed in range(3)
    ]
    return (
        pl.concat(frames)
        .group_by(["site", "run", "valid_time"])
        .agg(pl.col("delta").mean())
        .with_columns(pl.col("run").cast(pl.Int32))
    )


def sunny_proxy_history() -> pl.DataFrame:
    """Return the replicate months' rows with the f1s-minus-control r2 loss and a sunny flag.

    The sunny flag is experiment 178's: the unit PV shape on the analysis proxy at or above the
    Primary's tercile edge (the smallest unit shape among its sunny scored-year half-hours).
    """
    import e00184_common as c184  # ty: ignore[unresolved-import]

    cache = c184.load_cache()
    edges = (
        pl.read_parquet(s317.common.E178_ROWS, columns=["site", "unit", "sunny"])
        .filter(pl.col("sunny"))
        .group_by("site")
        .agg(pl.col("unit").min().alias("edge"))
    )
    edge = dict(zip(edges["site"].to_list(), edges["edge"].to_list(), strict=True))
    pieces = []
    for variant in ("v0r", "f1s"):
        short = pl.read_parquet(common.ARMS_DIR / f"short_{variant}.parquet").filter(
            pl.col("period") == "replicate"
        )
        pieces.append(short)
    control, treated = pieces
    keys = ["site", "run", "valid_time"]
    chain = pl.concat(
        [
            pl.read_parquet(
                common.CHAIN_DIR / "v0r" / f"{site}.parquet",
                columns=["run", "valid_time", "actual", "switching_mask"],
            ).with_columns(site=pl.lit(site))
            for site in common.SITES
        ]
    ).with_columns(pl.col("run").cast(pl.Int32))
    joined = (
        control.with_columns(pl.col("run").cast(pl.Int32))
        .join(
            treated.with_columns(pl.col("run").cast(pl.Int32)).select(
                *keys, *[pl.col(f"r2_{q}").alias(f"t_{q}") for q in QUANTILES]
            ),
            on=keys,
            how="inner",
            validate="1:1",
        )
        .join(chain, on=keys, how="inner", validate="1:1")
        .filter(~pl.col("switching_mask") & pl.col("actual").is_finite())
    )
    actual = joined["actual"].to_numpy().astype(np.float64)
    control_q = joined.select([f"r2_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    treated_q = joined.select([f"t_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    difference = 2.0 * (
        s317.pin(actual=actual, q=np.sort(treated_q, axis=1)).mean(axis=1)
        - s317.pin(actual=actual, q=np.sort(control_q, axis=1)).mean(axis=1)
    )
    valid = joined["valid_time"].dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
    sunny = np.zeros(joined.height, dtype=bool)
    site_of = joined["site"].to_numpy()
    for name in common.SITES:
        rows_site = site_of == name
        position = np.searchsorted(cache.end_times, valid[rows_site])
        unit = cache.sites[name].pv_unit_shape[position]
        sunny[rows_site] = np.nan_to_num(unit, nan=0.0) >= edge[name]
    return joined.select("site", "power_fcst_init_time", "valid_time").with_columns(
        difference=pl.Series(difference), sunny=pl.Series(sunny)
    )


def select_per_primary(
    *, rows: pl.DataFrame, loss_default: np.ndarray, loss_f1s: np.ndarray
) -> dict:
    """Return the as-of switch of F1sel: per Primary and scored month, whether F1s is used.

    The response is used at a Primary in a month if, over the half-hours with a known outcome
    before the month starts and within the previous 12 months, F1s beat the default by at least
    0.02 points on sunny daylight. The history is the replicate months before the scored year
    (F1s minus the rebuilt control, both without the network's median shift) plus the scored
    months already completed (F1s minus the default).
    """
    history = sunny_proxy_history()
    scored = rows.select(
        "site", "power_fcst_init_time", "valid_time", "sunny_daylight"
    ).with_columns(
        difference=pl.Series((loss_f1s - loss_default).astype(np.float64)),
        sunny=pl.col("sunny_daylight"),
    )
    pieces = pl.concat(
        [
            history.select("site", "power_fcst_init_time", "valid_time", "difference", "sunny"),
            scored.select("site", "power_fcst_init_time", "valid_time", "difference", "sunny"),
        ]
    )
    months = sorted(set(rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_list()))
    out: dict = {"on": {}, "difference": {}, "flips": {}, "margin": SELECTION_MARGIN}
    for site in common.SITES:
        part = pieces.filter((pl.col("site") == site) & pl.col("sunny"))
        out["on"][site], out["difference"][site] = {}, {}
        previous = None
        flips = 0
        for month in months:
            start = pl.lit(f"{month}-01").str.to_date().cast(pl.Datetime("us", "UTC"))
            window_start = start.dt.offset_by(f"-{SELECTION_MONTHS}mo")
            window = part.filter(
                (pl.col("valid_time") < start) & (pl.col("power_fcst_init_time") >= window_start)
            )
            mean_difference = float(window["difference"].mean()) if window.height else float("nan")
            use = bool(window.height and mean_difference <= -SELECTION_MARGIN)
            out["on"][site][month] = use
            out["difference"][site][month] = mean_difference
            if previous is not None and use != previous:
                flips += 1
            previous = use
        out["flips"][site] = flips
    return out


def main() -> None:
    """Assemble the rows, run the gates, score every contrast, and write the report."""
    chain = s317.load_chain()
    manual = s317.read_reference(name="ref_manual_heuristic", tag="mh")
    clim = s317.read_reference(name="ref_climatology", tag="cl")
    rows = chain.join(manual, on=s317.common.KEYS, how="inner", validate="1:1")
    rows = rows.join(clim, on=s317.common.KEYS, how="inner", validate="1:1")
    rows = rows.filter(pl.col("lead_hours") <= c320.HORIZON_HOURS)
    e251 = pl.read_parquet(c320.c319.E251_ARMS / "rows.parquet").select(KEYS).with_row_index("i251")
    rows = rows.join(e251, on=KEYS, how="left", validate="1:1")
    stored = pl.read_parquet(
        common.E322_ARMS / "scored_long_lead.parquet",
        columns=[*KEYS, "kind", *[f"d1_{q}" for q in QUANTILES]],
    ).rename({"kind": "kind_long"})
    rows = rows.join(stored, on=KEYS, how="left", validate="1:1")
    for name, arms_dir in (
        ("v0r", common.ARMS_DIR),
        ("f1", common.ARMS_DIR),
        ("f1s", common.ARMS_DIR),
        ("e323f1", common323_arms),
    ):
        file_name = "f1" if name == "e323f1" else name
        short = pl.read_parquet(arms_dir / f"short_{file_name}.parquet").filter(
            pl.col("period") == "scored"
        )
        long = pl.read_parquet(arms_dir / f"long_{file_name}.parquet").filter(
            pl.col("period") == "scored"
        )
        tag = f"_{name}"
        rows = rows.join(
            short.select(*KEYS, *[pl.col(f"r2_{q}").alias(f"r2{tag}_{q}") for q in QUANTILES]),
            on=KEYS,
            how="left",
            validate="1:1",
        ).join(
            long.select(*KEYS, *[pl.col(f"d1_{q}").alias(f"d1{tag}_{q}") for q in QUANTILES]),
            on=KEYS,
            how="left",
            validate="1:1",
        )
    deltas = {name: read_delta(variant=name) for name in common.VARIANTS}
    for name, delta in deltas.items():
        rows = rows.join(
            delta.rename({c: f"net_{name}_{c}" for c in ("delta",)}),
            on=["site", "run", "valid_time"],
            how="left",
            validate="m:1",
        )
    rows = s317.flags(rows=rows)
    lead = rows["lead_hours"].to_numpy()
    long = lead > c320.STORED_HORIZON_HOURS
    keep = long | rows["i251"].is_not_null().to_numpy()
    rows = rows.filter(pl.Series(keep))
    lead = rows["lead_hours"].to_numpy()
    long = lead > c320.STORED_HORIZON_HOURS
    n = rows.height
    arms251 = {
        k: np.load(c320.c319.E251_ARMS / f"{k}.npy").astype(np.float64)
        for k in ("arm1", "a_s", "bg2")
    }
    idx = rows["i251"].fill_null(0).to_numpy()
    full_delta = arms251["arm1"][idx] - arms251["a_s"][idx]
    short_default = arms251["arm1"][idx]
    default = np.where(
        long[:, None],
        rows.select([f"d1_{q}" for q in QUANTILES]).to_numpy().astype(np.float64),
        short_default,
    )
    quantiles = {"default": default}
    missing_delta = {}
    for name in ("v0r", "f1", "f1s"):
        short_r2 = rows.select([f"r2_{name}_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
        long_q = rows.select([f"d1_{name}_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
        net = rows[f"net_{name}_delta"].to_numpy().astype(np.float64)
        missing_delta[name] = int((np.isnan(net) & ~long).sum())
        net = np.where(np.isnan(net), 0.0, net)
        quantiles[name] = np.sort(np.where(long[:, None], long_q, short_r2 + net[:, None]), axis=1)
    e323_short = rows.select([f"r2_e323f1_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    e323_long = rows.select([f"d1_e323f1_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    quantiles["f1_as_run"] = np.sort(
        np.where(long[:, None], e323_long, e323_short + full_delta), axis=1
    )
    quantiles["f1r"] = quantiles.pop("f1")
    gate = {
        "v0r_equals_default_max_abs": float(np.abs(quantiles["v0r"] - default).max()),
        "v0r_equals_default_below_168": float(np.abs(quantiles["v0r"] - default)[~long].max()),
        "v0r_equals_default_above_168": float(np.abs(quantiles["v0r"] - default)[long].max()),
        "default_0_240": float(
            (
                2.0
                * s317.pin(actual=rows["actual"].to_numpy().astype(np.float64), q=default).mean(
                    axis=1
                )
            ).mean()
        ),
        "expected_0_240": 2.727,
        "n_rows": n,
        "n_long": int(long.sum()),
        "rows_below_168_without_network_delta": missing_delta,
    }
    print("gate", json.dumps(gate), flush=True)
    actual = rows["actual"].to_numpy().astype(np.float64)
    site = rows["site"].to_numpy()
    month_of_origin = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    selection = select_per_primary(
        rows=rows,
        loss_default=2.0 * s317.pin(actual=actual, q=quantiles["default"]).mean(axis=1),
        loss_f1s=2.0 * s317.pin(actual=actual, q=quantiles["f1s"]).mean(axis=1),
    )
    on = np.array([selection["on"][s_][m_] for s_, m_ in zip(site, month_of_origin, strict=True)])
    quantiles["f1sel"] = np.where(on[:, None], quantiles["f1s"], default)
    quantiles.pop("v0r")
    quantiles["manual_heuristic"] = (
        rows.select([f"mh_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    )
    quantiles["climatology"] = (
        rows.select([f"cl_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    )
    quantiles["BG2"] = np.where(long[:, None], np.nan, arms251["bg2"][idx])
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
        if k != "BG2"
    }  # fmt: skip
    month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    week = rows["power_fcst_init_time"].dt.strftime("%G-%V").to_numpy()
    valid_month = rows["valid_time"].dt.convert_time_zone("Europe/London").dt.month().to_numpy()
    winter = np.isin(valid_month, list(c320.WINTER_MONTHS))
    band = np.clip(
        np.searchsorted(s317.BAND_EDGES, lead, side="left") - 1, 0, len(s317.BAND_LABELS) - 1
    )
    cold = rows["cold_winter_night"].to_numpy()
    day = rows["daylight"].fill_null(False).to_numpy()
    sunny = rows["sunny_daylight"].to_numpy()
    e7 = s322.economy_7_night(valid_time=rows["valid_time"])
    init_month = rows["power_fcst_init_time"].dt.month().to_numpy()
    spring = np.isin(init_month, APRIL_TO_JUNE)
    coastal = np.isin(site, ["D5", "D9"])
    groups = {
        "all rows": np.ones(n, dtype=bool),
        "cold winter nights": cold,
        "sunny daylight": sunny,
        "all daylight": day,
        "all night": ~day,
        "Economy 7 nights": e7 & ~day,
        "winter (Nov-Mar)": winter,
        "April to October": ~winter,
        "April to June daylight": day & spring,
        "April to June daylight, D5 and D9": day & spring & coastal,
        "April to June daylight, D5": day & spring & (site == "D5"),
        "April to June daylight, D9": day & spring & (site == "D9"),
        "D10 and D16 (placebo)": np.isin(site, ["D10", "D16"]),
    }
    windows = {"0-240 h": np.ones(n, dtype=bool), "0-168 h": ~long, "168-240 h": long}
    rep: dict = {
        "gate": gate,
        "n_rows": n,
        "group_rows": {k: int(v.sum()) for k, v in groups.items()},
    }
    rep["by_band"] = {}
    for i, label in enumerate(s317.BAND_LABELS):
        mask = band == i
        entry: dict = {"n": int(mask.sum())}
        for arm in (*ARMS, *REFERENCES, "BG2"):
            if np.isfinite(loss[arm][mask]).all():
                entry[arm] = s319.mean_ci(values=loss[arm], mask=mask, blocks=month)
        for a, b in CONTRASTS:
            entry[f"{a}_vs_{b}"] = s319.contrast(
                treat=loss[a], ref=loss[b], mask=mask, blocks=month
            )
        rep["by_band"][label] = entry
    rep["by_group"] = {}
    for gname, gmask in groups.items():
        rep["by_group"][gname] = {}
        blocks = week if "April to June" in gname else month
        for wname, wmask in windows.items():
            mask = gmask & wmask
            if mask.sum() < 50:
                continue
            e: dict = {"n": int(mask.sum())}
            for arm in (*ARMS, *REFERENCES):
                e[arm] = s319.mean_ci(values=loss[arm], mask=mask, blocks=blocks)
            for a, b in CONTRASTS:
                e[f"{a}_vs_{b}"] = s319.contrast(
                    treat=loss[a], ref=loss[b], mask=mask, blocks=blocks
                )
            for a in VARIANTS:
                e[f"{a}_twcrps_vs_default"] = s319.contrast(
                    treat=tw[a], ref=tw["default"], mask=mask, blocks=blocks
                )
                for t, pos in (("q01", 0), ("q99", 10)):
                    e[f"{a}_{t}_pinball_diff"] = float(
                        pin[a][mask, pos].mean() - pin["default"][mask, pos].mean()
                    )
            rep["by_group"][gname][wname] = e
    rep["tails_by_band"] = {
        label: {
            arm: {
                "q01_below": float((actual < quantiles[arm][:, 0])[band == i].mean()),
                "q99_below": float((actual < quantiles[arm][:, 10])[band == i].mean()),
                "q01_pinball": float(pin[arm][band == i, 0].mean()),
                "q99_pinball": float(pin[arm][band == i, 10].mean()),
            }
            for arm in (*ARMS, *REFERENCES)
        }
        for i, label in enumerate(s317.BAND_LABELS)
    }
    rep["by_site"], rep["leave_one_out"] = {}, {}
    for wname in ("0-240 h", "0-168 h", "168-240 h"):
        wmask = windows[wname]
        rep["by_site"][wname] = {
            s: {f"{a}_vs_default": s319.contrast(treat=loss[a], ref=loss["default"], mask=wmask & (site == s), blocks=month) for a in VARIANTS}
            for s in c320.SCORED_SITES
        }  # fmt: skip
        rep["leave_one_out"][wname] = {
            s: {f"{a}_vs_default": s319.contrast(treat=loss[a], ref=loss["default"], mask=wmask & (site != s), blocks=month) for a in VARIANTS}
            for s in c320.SCORED_SITES
        }  # fmt: skip
    rep["by_site_april_to_june_daylight_0_168"] = {
        s: {f"{a}_vs_default": s319.contrast(treat=loss[a], ref=loss["default"], mask=day & spring & ~long & (site == s), blocks=week) for a in VARIANTS}
        for s in c320.SCORED_SITES
    }  # fmt: skip
    rep["median_minus_mean_by_site"] = {
        arm: {s: float((quantiles[arm][site == s, 5] - quantiles[arm][site == s].mean(axis=1)).mean()) for s in c320.SCORED_SITES}
        for arm in ARMS
    }  # fmt: skip
    rep["mean_abs_median_error_by_window"] = {
        wname: {arm: float(np.abs(actual - quantiles[arm][:, 5])[wmask].mean()) for arm in (*ARMS, *REFERENCES)}
        for wname, wmask in windows.items()
    }  # fmt: skip
    rep["signed_median_error_by_window"] = {
        wname: {arm: float((actual - quantiles[arm][:, 5])[wmask].mean()) for arm in (*ARMS, *REFERENCES)}
        for wname, wmask in windows.items()
    }  # fmt: skip
    rep["selection"] = selection
    rep["replicate"] = replicate_report()
    common.write_json_atomic(path=common.REPORT_DIR / "report.json", content=rep)
    rows.select(
        "site", "power_fcst_init_time", "valid_time", "lead_hours", "actual",
        "cold_winter_night", "sunny_daylight", "daylight",
    ).with_columns(
        *[pl.Series(f"loss_{arm}", loss[arm].astype(np.float32)) for arm in (*ARMS, *REFERENCES)],
    ).write_parquet(common.REPORT_DIR / "scored_rows.parquet")  # fmt: skip
    for label, e in rep["by_band"].items():
        print(
            label,
            e["n"],
            *[f"{a} {e[a]['est']:.3f}" for a in (*ARMS, *REFERENCES) if a in e],
            flush=True,
        )


def replicate_report() -> dict:
    """Return the replicate year's contrasts of each variant against `v0r`, which has no response."""
    chain = s319.load_chain_period(period="replicate")
    pieces = {}
    for name in ("v0r", "f1", "f1s"):
        short = pl.read_parquet(common.ARMS_DIR / f"short_{name}.parquet").filter(
            pl.col("period") == "replicate"
        )
        long = pl.read_parquet(common.ARMS_DIR / f"long_{name}.parquet").filter(
            pl.col("period") == "replicate"
        )
        pieces[name] = (short, long)
    rows = chain.select(*KEYS, "actual", "lead_hours")
    for name, (short, long) in pieces.items():
        rows = rows.join(
            short.select(*KEYS, *[pl.col(f"r2_{q}").alias(f"r2_{name}_{q}") for q in QUANTILES]), on=KEYS, how="left", validate="1:1"
        ).join(
            long.select(*KEYS, "kind", *[pl.col(f"d1_{q}").alias(f"d1_{name}_{q}") for q in QUANTILES]).rename({"kind": f"kind_{name}"}),
            on=KEYS, how="left", validate="1:1",
        )  # fmt: skip
    rows = rows.filter(pl.col("lead_hours") <= c320.HORIZON_HOURS)
    long = (rows["lead_hours"] > 168.0).to_numpy()
    actual = rows["actual"].to_numpy().astype(np.float64)
    loss = {}
    for name in ("v0r", "f1", "f1s"):
        short_q = rows.select([f"r2_{name}_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
        long_q = rows.select([f"d1_{name}_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
        q = np.sort(np.where(long[:, None], long_q, short_q), axis=1)
        ok = np.isfinite(q).all(axis=1)
        loss[name] = np.where(
            ok, 2.0 * s317.pin(actual=actual, q=np.where(ok[:, None], q, 0.0)).mean(axis=1), np.nan
        )
    init_month = rows["power_fcst_init_time"].dt.month().to_numpy()
    week = rows["power_fcst_init_time"].dt.strftime("%G-%V").to_numpy()
    month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    site = rows["site"].to_numpy()
    valid_utc_hour = rows["valid_time"].dt.hour().to_numpy()
    day = np.isfinite(loss["v0r"]) & (rows["lead_hours"].to_numpy() > 0)
    sun_up = (valid_utc_hour >= 6) & (
        valid_utc_hour <= 17
    )  # a UTC proxy for daylight, replicate rows carry no kind below 168 hours
    ok = np.isfinite(loss["v0r"]) & np.isfinite(loss["f1"]) & np.isfinite(loss["f1s"])
    out: dict = {
        "n_rows": int(ok.sum()),
        "note": "variant minus v0r (no response, rebuilt), A+S below 168 hours, no median correction",
    }
    subsets = {
        "all rows": ok,
        "April and June daylight (UTC 06-18), D5 and D9, 0-168 h": ok
        & np.isin(init_month, (4, 6))
        & sun_up
        & np.isin(site, ["D5", "D9"])
        & ~long,
        "April daylight, D5 and D9, 0-168 h": ok
        & (init_month == 4)
        & sun_up
        & np.isin(site, ["D5", "D9"])
        & ~long,
        "June daylight, D5 and D9, 0-168 h": ok
        & (init_month == 6)
        & sun_up
        & np.isin(site, ["D5", "D9"])
        & ~long,
        "D10 and D16, 0-240 h": ok & np.isin(site, ["D10", "D16"]),
    }
    del day
    for name, mask in subsets.items():
        blocks = week if "April" in name or "June" in name else month
        out[name] = {
            a: s319.contrast(treat=loss[a], ref=loss["v0r"], mask=mask, blocks=blocks)
            for a in ("f1", "f1s")
        }
    return out


if __name__ == "__main__":
    main()
