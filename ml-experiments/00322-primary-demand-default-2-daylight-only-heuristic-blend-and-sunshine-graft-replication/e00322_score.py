# ruff: noqa: E501, PLR0912, PLR0915
"""Score the default and the daylight-only blend arms to 240 hours against the manual heuristic.

Rows, metric, and benchmark rows are experiment 320's: six Primaries, scored year, rows with a
reading outside the hand-filtered events. The default is Primary demand research default 2:
experiment 251's A+H+S below 168 hours and experiment 319's x1 above. Intervals resample the 12
months of the forecast run's initialisation time (20,000 draws) and are read at the 95% and 99.5%
levels.

    PYTHONPATH=<folders of experiments 184, 202, 317, 319, 320, 322> \
        uv run --no-sync python ml-experiments/00322-*/e00322_score.py
"""

import json

import e00317_score as s317  # ty: ignore[unresolved-import]
import e00319_score as s319  # ty: ignore[unresolved-import]
import e00320_common as c320  # ty: ignore[unresolved-import]
import e00322_common as common
import numpy as np
import polars as pl

QUANTILES = c320.QUANTILES
LONG_ARMS = ("r319", "d1", "d2", "d3", "d3b")
ARMS = ("default", *LONG_ARMS)
REFERENCES = ("manual_heuristic", "climatology")
BAND_EDGES = s317.BAND_EDGES
BAND_LABELS = s317.BAND_LABELS
TAIL_COLUMNS = (("q01", 0), ("q99", 10))
CONTRASTS = (
    ("r319", "default"),
    ("d1", "default"),
    ("d2", "default"),
    ("d3", "default"),
    ("d3", "d1"),
    ("d3b", "default"),
    ("d3b", "d1"),
    ("d3b", "manual_heuristic"),
    ("d2", "d1"),
    ("d1", "r319"),
    ("d1", "manual_heuristic"),
    ("d3", "manual_heuristic"),
    ("r319", "manual_heuristic"),
    ("default", "manual_heuristic"),
    ("default", "climatology"),
)
SHARE_THRESHOLDS = (0.3, 0.5, 0.7)


def economy_7_night(*, valid_time: pl.Series) -> np.ndarray:
    """Return True for half-hours from 01:00 to 07:00 local time (the cheap-rate window of Economy 7 tariffs)."""
    local = valid_time.dt.convert_time_zone("Europe/London")
    minute = (local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute().cast(pl.Int32)).to_numpy()
    return (minute >= 60) & (minute < 420)


def add_d3b(*, frame: pl.DataFrame, kind_column: str) -> pl.DataFrame:
    """Add the post hoc arm d3b: the experiment 319 blend everywhere except forecast cold winter nights.

    A forecast cold winter night is a winter (November to March) half-hour after dark, local 18:00
    to 07:00, where at least half of the 51 members have a 24-hour mean temperature below 3 C. On
    those half-hours the chain alone is kept (experiment 320's B1 with the member share in place of
    the ensemble-mean rule).
    """
    local_hour = frame["valid_time"].dt.convert_time_zone("Europe/London").dt.hour().to_numpy()
    month = frame["valid_time"].dt.month().to_numpy()
    flag = (
        (np.nan_to_num(frame["p_share"].to_numpy(), nan=0.0) >= common.FLAG_SHARE)
        & (frame[kind_column] != "daylight").fill_null(False).to_numpy()
        & np.isin(month, list(c320.WINTER_MONTHS))
        & ((local_hour >= 18) | (local_hour < 7))
    )
    return frame.with_columns(
        pl.Series(f"d3b_{q}", np.where(flag, frame[f"x1_{q}"], frame[f"r319_{q}"]))
        for q in QUANTILES
    ).with_columns(share_rule=pl.Series(flag))


def recall_table(*, rows: pl.DataFrame, long: np.ndarray) -> dict:
    """Return, by 24-hour band past 168 hours, the recall and false-flag rate of the mean rule and the share rule."""
    lead = rows["lead_hours"].to_numpy()
    group = rows["cold_winter_night"].to_numpy()
    local_hour = rows["valid_time"].dt.convert_time_zone("Europe/London").dt.hour().to_numpy()
    month = rows["valid_time"].dt.month().to_numpy()
    winter_night = (
        np.isin(month, list(s317.WINTER_MONTHS))
        & ((local_hour >= 18) | (local_hour < 7))
        & ~rows["daylight"].fill_null(True).to_numpy()
    )
    mean_flag = np.nan_to_num(rows["ens_mean_c"].to_numpy(), nan=99.0) < c320.COLD_FORECAST_C
    share = np.nan_to_num(rows["p_share"].to_numpy(), nan=0.0)
    month_block = rows["power_fcst_init_time"].dt.strftime("%Y-%m-%d").to_numpy()
    out: dict = {}
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    for lo in (168, 192, 216):
        band = long & (lead > lo) & (lead <= lo + 24)
        in_group = band & group
        others = band & winter_night & ~group
        entry: dict = {
            "group_rows": int(in_group.sum()),
            "winter_night_rows": int((band & winter_night).sum()),
            "base_rate": float(in_group.sum() / max((band & winter_night).sum(), 1)),
        }
        rules = {"mean_below_2c": mean_flag} | {
            f"share_at_least_{t}": share >= t for t in SHARE_THRESHOLDS
        }
        for name, flag in rules.items():
            recall = float(flag[in_group].mean()) if in_group.sum() else float("nan")
            false_flag = float(flag[others].mean()) if others.sum() else float("nan")
            # Block bootstrap by origin day for the recall.
            labels, codes = np.unique(month_block[in_group], return_inverse=True)
            hit = np.bincount(codes, weights=flag[in_group].astype(float), minlength=len(labels))
            cnt = np.bincount(codes, minlength=len(labels)).astype(float)
            draws = rng.integers(0, len(labels), size=(5_000, len(labels)))
            res = hit[draws].sum(axis=1) / cnt[draws].sum(axis=1)
            entry[name] = {
                "recall": recall,
                "recall_lo": float(np.percentile(res, 2.5)),
                "recall_hi": float(np.percentile(res, 97.5)),
                "false_flag_rate": false_flag,
                "false_flag_over_base_rate": false_flag / entry["base_rate"]
                if entry["base_rate"] > 0
                else float("nan"),
            }
        out[f"{lo}-{lo + 24} h"] = entry
    pooled = long & (lead > 168)
    in_group = pooled & group
    others = pooled & winter_night & ~group
    out["168-240 h"] = {
        "group_rows": int(in_group.sum()),
        "base_rate": float(in_group.sum() / max((pooled & winter_night).sum(), 1)),
        **{
            name: {
                "recall": float(flag[in_group].mean()),
                "false_flag_rate": float(flag[others].mean()),
            }
            for name, flag in (
                ("mean_below_2c", mean_flag),
                *((f"share_at_least_{t}", share >= t) for t in SHARE_THRESHOLDS),
            )
        },
    }
    return out


def replicate_report() -> dict:
    """Return the replicate year's contrasts at 168 to 240 hours for each arm."""
    chain = s319.load_chain_period(period="replicate")
    stored = add_d3b(
        frame=pl.read_parquet(common.ARMS_DIR / "replicate_long_lead.parquet"), kind_column="kind"
    )
    keys = ["site", "power_fcst_init_time", "valid_time"]
    rows = chain.select(*keys, "actual").join(stored, on=keys, how="inner", validate="1:1")
    rows = rows.filter(pl.col("lead_hours") > c320.STORED_HORIZON_HOURS)
    actual = rows["actual"].to_numpy().astype(np.float64)
    q = {"default": rows.select([f"x1_{k}" for k in QUANTILES]).to_numpy().astype(np.float64)}
    for a in LONG_ARMS:
        q[a] = rows.select([f"{a}_{k}" for k in QUANTILES]).to_numpy().astype(np.float64)
    loss = {k: 2.0 * s317.pin(actual=actual, q=v).mean(axis=1) for k, v in q.items()}
    month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    vmonth = rows["valid_time"].dt.convert_time_zone("Europe/London").dt.month().to_numpy()
    winter = np.isin(vmonth, list(c320.WINTER_MONTHS))
    day = (rows["kind"] == "daylight").to_numpy()
    masks = {
        "all rows": np.ones(rows.height, dtype=bool),
        "daylight": day,
        "night": ~day,
        "winter (Nov-Mar)": winter,
        "April to October": ~winter,
    }
    out: dict = {"n_rows": rows.height}
    for g, mask in masks.items():
        out[g] = {
            f"{a}_vs_default": s319.contrast(
                treat=loss[a], ref=loss["default"], mask=mask, blocks=month
            )
            for a in LONG_ARMS
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
    rows = rows.filter(pl.col("lead_hours") <= c320.HORIZON_HOURS)
    stored = (
        pl.read_parquet(c320.c319.E251_ARMS / "rows.parquet")
        .select(["site", "power_fcst_init_time", "valid_time"])
        .with_row_index("i251")
    )
    keys = ["site", "power_fcst_init_time", "valid_time"]
    rows = rows.join(stored, on=keys, how="left", validate="1:1")
    long_arms = add_d3b(
        frame=pl.read_parquet(
            common.ARMS_DIR / "scored_long_lead.parquet",
            columns=[
                *keys,
                "p_share",
                "kind",
                *[f"{a}_{q}" for a in ("x1", "r319", "d1", "d2", "d3") for q in QUANTILES],
            ],
        ),
        kind_column="kind",
    ).rename({"kind": "kind_long"})
    rows = rows.join(long_arms, on=keys, how="left", validate="1:1")
    rows = s317.flags(rows=rows)
    lead = rows["lead_hours"].to_numpy()
    long = lead > c320.STORED_HORIZON_HOURS
    keep = long | rows["i251"].is_not_null().to_numpy()
    rows = rows.filter(pl.Series(keep))
    lead = rows["lead_hours"].to_numpy()
    long = lead > c320.STORED_HORIZON_HOURS
    n = rows.height
    arms251 = {k: np.load(c320.c319.E251_ARMS / f"{k}.npy") for k in ("arm1", "bg2")}
    idx = rows["i251"].fill_null(0).to_numpy()
    short_q = arms251["arm1"][idx].astype(np.float64)

    def long_or_default(*, name: str) -> np.ndarray:
        return rows.select([f"{name}_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)

    quantiles = {"default": np.where(long[:, None], long_or_default(name="x1"), short_q)}
    for a in LONG_ARMS:
        quantiles[a] = np.where(long[:, None], long_or_default(name=a), short_q)
    quantiles["manual_heuristic"] = (
        rows.select([f"mh_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    )
    quantiles["climatology"] = (
        rows.select([f"cl_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    )
    quantiles["BG2"] = np.where(long[:, None], np.nan, arms251["bg2"][idx].astype(np.float64))
    actual = rows["actual"].to_numpy().astype(np.float64)
    site = rows["site"].to_numpy()
    threshold = np.array([np.quantile(actual[site == s], 0.9) for s in c320.SCORED_SITES])
    thr = threshold[[c320.SCORED_SITES.index(s) for s in site]]
    pin = {k: s317.pin(actual=actual, q=v) for k, v in quantiles.items()}
    loss = {k: 2.0 * v.mean(axis=1) for k, v in pin.items()}
    tw = {
        k: 2.0
        * np.maximum(
            c320.LEVELS * (np.maximum(actual, thr)[:, None] - np.maximum(v, thr[:, None])),
            (c320.LEVELS - 1.0) * (np.maximum(actual, thr)[:, None] - np.maximum(v, thr[:, None])),
        ).mean(axis=1)
        for k, v in quantiles.items()
        if k != "BG2"
    }
    month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    valid_month = rows["valid_time"].dt.convert_time_zone("Europe/London").dt.month().to_numpy()
    winter = np.isin(valid_month, list(c320.WINTER_MONTHS))
    band = np.clip(np.searchsorted(BAND_EDGES, lead, side="left") - 1, 0, len(BAND_LABELS) - 1)
    cold_scored = rows["cold_winter_night"].to_numpy()
    day = rows["daylight"].fill_null(False).to_numpy()
    e7 = economy_7_night(valid_time=rows["valid_time"])
    short_default = quantiles["default"][~long]
    rep: dict = {"n_rows": n, "n_long": int(long.sum())}
    rep["gate"] = {
        "default_0_240": float(loss["default"].mean()),
        "default_168_240": float(loss["default"][long].mean()),
        "expected_0_240": 2.749,
        "expected_168_240": 3.152,
        "default_below_168_equals_251": float(np.abs(short_default - short_q[~long]).max()),
        "d_arms_equal_default_below_168": float(
            max(np.abs(quantiles[a][~long] - short_default).max() for a in ("d1", "d2", "d3"))
        ),
        "cold_winter_night_rows_168_240": int((cold_scored & long).sum()),
        "cold_winter_night_rows_kind_daylight": int(
            (
                cold_scored & long & (rows["kind_long"] == "daylight").fill_null(False).to_numpy()
            ).sum()
        ),
        "d1_equals_default_on_cold_nights": float(
            np.abs(
                quantiles["d1"][cold_scored & long] - quantiles["default"][cold_scored & long]
            ).max()
        ),
    }
    groups = {
        "all rows": np.ones(n, dtype=bool),
        "cold winter nights": cold_scored,
        "sunny daylight": rows["sunny_daylight"].to_numpy(),
        "all daylight": day,
        "all night": ~day,
        "Economy 7 nights": e7 & ~day,
        "winter (Nov-Mar)": winter,
        "April to October": ~winter,
    }
    windows = {"0-240 h": np.ones(n, dtype=bool), "0-168 h": ~long, "168-240 h": long}
    rep["group_rows"] = {k: int(v.sum()) for k, v in groups.items()}
    rep["by_band"] = {}
    for i, label in enumerate(BAND_LABELS):
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
        for wname, wmask in windows.items():
            mask = gmask & wmask
            if mask.sum() < 50:
                continue
            e: dict = {"n": int(mask.sum())}
            for arm in (*ARMS, *REFERENCES):
                e[arm] = s319.mean_ci(values=loss[arm], mask=mask, blocks=month)
            for a, b in CONTRASTS:
                e[f"{a}_vs_{b}"] = s319.contrast(
                    treat=loss[a], ref=loss[b], mask=mask, blocks=month
                )
            for a in LONG_ARMS:
                e[f"{a}_twcrps_vs_default"] = s319.contrast(
                    treat=tw[a], ref=tw["default"], mask=mask, blocks=month
                )
                for t, pos in TAIL_COLUMNS:
                    e[f"{a}_{t}_pinball_diff"] = float(
                        pin[a][mask, pos].mean() - pin["default"][mask, pos].mean()
                    )
            rep["by_group"][gname][wname] = e
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
            }
            for arm in (*ARMS, *REFERENCES)
        }
    rep["by_site"], rep["leave_one_out"] = {}, {}
    for wname in ("0-240 h", "168-240 h"):
        wmask = windows[wname]
        rep["by_site"][wname] = {
            s: {
                f"{a}_vs_default": s319.contrast(
                    treat=loss[a], ref=loss["default"], mask=wmask & (site == s), blocks=month
                )
                for a in LONG_ARMS
            }
            for s in c320.SCORED_SITES
        }
        rep["leave_one_out"][wname] = {
            s: {
                f"{a}_vs_default": s319.contrast(
                    treat=loss[a], ref=loss["default"], mask=wmask & (site != s), blocks=month
                )
                for a in LONG_ARMS
            }
            for s in c320.SCORED_SITES
        }
    # Gain by UTC half-hour of the day at 168 to 240 hours.
    hh = (
        rows["valid_time"].dt.hour().cast(pl.Int32) * 2
        + rows["valid_time"].dt.minute().cast(pl.Int32) // 30
    ).to_numpy()
    rep["gain_by_half_hour"] = {
        a: [float((loss[a] - loss["default"])[long & (hh == h)].mean()) for h in range(48)]
        for a in ("r319", "d1", "d3")
    }
    rep["gain_by_half_hour_winter"] = {
        a: [float((loss[a] - loss["default"])[long & winter & (hh == h)].mean()) for h in range(48)]
        for a in ("r319", "d1", "d3")
    }
    rep["recall"] = recall_table(rows=rows, long=long)
    rep["median_minus_mean_by_site"] = {
        arm: {
            s: float((quantiles[arm][site == s, 5] - quantiles[arm][site == s].mean(axis=1)).mean())
            for s in c320.SCORED_SITES
        }
        for arm in ARMS
    }
    rep["mean_abs_median_error_by_window"] = {
        wname: {
            arm: float(np.abs(actual - quantiles[arm][:, 5])[wmask].mean())
            for arm in (*ARMS, *REFERENCES)
        }
        for wname, wmask in windows.items()
    }
    rep["replicate"] = replicate_report()
    common.write_json_atomic(path=common.REPORT_DIR / "report.json", content=rep)
    rows.select(
        "site", "power_fcst_init_time", "valid_time", "lead_hours", "actual",
        "cold_winter_night", "sunny_daylight", "p_share", "ens_mean_c",
    ).with_columns(
        *[pl.Series(f"loss_{arm}", loss[arm].astype(np.float32)) for arm in (*ARMS, *REFERENCES)],
    ).write_parquet(common.REPORT_DIR / "scored_rows.parquet")  # fmt: skip
    print(json.dumps(rep["gate"], default=float), flush=True)
    for label, e in rep["by_band"].items():
        print(
            label, e["n"],
            *[f"{a} {e[a]['est']:.3f}" for a in (*ARMS, *REFERENCES) if a in e],
            flush=True,
        )  # fmt: skip


if __name__ == "__main__":
    main()
