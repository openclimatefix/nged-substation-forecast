# ruff: noqa: E501, PLR0912, PLR0915
"""Score the default and the arms to 240 hours against the manual heuristic and climatology.

Rows, metric, and benchmark rows are experiment 319's: six Primaries, scored year, rows with a
reading outside the hand-filtered events. The default is Primary demand research default 2:
experiment 251's A+H+S below 168 hours and experiment 319's x1 above. Intervals resample the 12
months of the forecast run's initialisation time (20,000 draws) and are read at the 95% and 99.5%
levels.

    PYTHONPATH=<folders of experiments 184, 202, 317, 319, 320> \
        uv run --no-sync python ml-experiments/00320-*/e00320_score.py
"""

import json

import e00317_score as s317  # ty: ignore[unresolved-import]
import e00319_score as s319  # ty: ignore[unresolved-import]
import e00320_arms as arms
import e00320_common as common
import numpy as np
import polars as pl

LONG_ARMS = ("b1", "b2", "r319")
ARMS = ("default", "b1", "b2", "r319", "n", "n_b1", "n_b2")
REFERENCES = ("manual_heuristic", "climatology")
BAND_EDGES = s317.BAND_EDGES
BAND_LABELS = s317.BAND_LABELS
TAIL_COLUMNS = (("q01", 0), ("q99", 10))
CONTRASTS = (
    ("b1", "default"),
    ("b2", "default"),
    ("r319", "default"),
    ("n", "default"),
    ("n_b1", "default"),
    ("n_b2", "default"),
    ("b2", "b1"),
    ("b1", "r319"),
    ("n_b1", "b1"),
    ("n_b2", "b2"),
    ("b1", "manual_heuristic"),
    ("b2", "manual_heuristic"),
    ("r319", "manual_heuristic"),
    ("default", "manual_heuristic"),
    ("default", "climatology"),
)


def width_multipliers(*, rows: pl.DataFrame, log: dict) -> dict[str, dict[str, float]]:
    """Return the month's multipliers as {month: {site: {block: m}}} keyed by the row's origin month."""
    return {m: v["width"] for m, v in log.items() if m != "replicate"}


def apply_width_by_month(*, q: np.ndarray, rows: pl.DataFrame, log: dict) -> np.ndarray:
    """Apply the as-of width multiplier of each row's origin month, per site and night or day."""
    out = q.copy()
    month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    site = rows["site"].to_numpy()
    night = arms.is_night_minute(valid_time=rows["valid_time"])
    for m in np.unique(month):
        sel = np.flatnonzero(month == m)
        out[sel] = arms.apply_width(
            base=q[sel], site=site[sel], night=night[sel], mult=log[m]["width"]
        )
    return out


def replicate_report(*, log: dict) -> dict:
    """Return the replicate year's contrasts at 168 to 240 hours for each arm."""
    chain = s319.load_chain_period(period="replicate")
    stored = pl.read_parquet(common.ARMS_DIR / "replicate_long_lead.parquet")
    keys = ["site", "power_fcst_init_time", "valid_time"]
    rows = chain.select(*keys, "actual").join(stored, on=keys, how="inner", validate="1:1")
    rows = rows.filter(pl.col("lead_hours") > common.STORED_HORIZON_HOURS)
    actual = rows["actual"].to_numpy().astype(np.float64)
    q = {
        "default": rows.select([f"x1_{k}" for k in common.QUANTILES]).to_numpy().astype(np.float64)
    }
    for a in LONG_ARMS:
        q[a] = rows.select([f"{a}_{k}" for k in common.QUANTILES]).to_numpy().astype(np.float64)
    night = rows["night"].to_numpy()
    q["n"] = arms.apply_width(
        base=q["default"], site=rows["site"].to_numpy(), night=night, mult=log["replicate"]["width"]
    )
    loss = {k: 2.0 * s317.pin(actual=actual, q=v).mean(axis=1) for k, v in q.items()}
    month = rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    vmonth = rows["valid_time"].dt.convert_time_zone("Europe/London").dt.month().to_numpy()
    winter = np.isin(vmonth, list(common.WINTER_MONTHS))
    masks = {
        "all rows": np.ones(rows.height, dtype=bool),
        "winter (Nov-Mar)": winter,
        "April to October": ~winter,
        "forecast cold winter nights": rows["cold_rule"].to_numpy(),
    }
    out: dict = {"n_rows": rows.height}
    for g, mask in masks.items():
        out[g] = {
            f"{a}_vs_default": s319.contrast(
                treat=loss[a], ref=loss["default"], mask=mask, blocks=month
            )
            for a in ("b1", "b2", "r319", "n")
        }
        out[g]["default"] = float(loss["default"][mask].mean())
    return out


def lead_weight_table(*, rows: pl.DataFrame, chain_q: np.ndarray, heuristic: np.ndarray) -> dict:
    """Return the best chain weight per 24-hour lead band and season on the scored year (exploratory)."""
    actual = rows["actual"].to_numpy().astype(np.float64)
    lead = rows["lead_hours"].to_numpy()
    vmonth = rows["valid_time"].dt.convert_time_zone("Europe/London").dt.month().to_numpy()
    winter = np.isin(vmonth, list(common.WINTER_MONTHS))
    out: dict = {}
    for i, label in enumerate(BAND_LABELS):
        band = (
            np.clip(np.searchsorted(BAND_EDGES, lead, side="left") - 1, 0, len(BAND_LABELS) - 1)
            == i
        )
        for name, season in (("Nov-Mar", winter), ("Apr-Oct", ~winter)):
            chosen = np.flatnonzero(band & season & np.isfinite(heuristic))
            shift = heuristic[chosen] - chain_q[chosen, 5]
            scores = [
                float(
                    2.0
                    * s317.pin(
                        actual=actual[chosen],
                        q=chain_q[chosen] + (1.0 - w) * shift[:, None],
                    ).mean()
                )
                for w in arms.a319.W_GRID
            ]
            out[f"{label} {name}"] = {
                "n": len(chosen),
                "best_w": float(arms.a319.W_GRID[int(np.argmin(scores))]),
                "score_at_best": float(min(scores)),
                "score_chain_only": float(scores[-1]),
                "score_heuristic_only": float(scores[0]),
            }
    return out


def main() -> None:
    """Assemble the rows, run the gates, score every contrast, and write the report."""
    log = json.loads((common.ARMS_DIR / "fit_log.json").read_text())
    chain = s317.load_chain()
    manual = s317.read_reference(name="ref_manual_heuristic", tag="mh")
    clim = s317.read_reference(name="ref_climatology", tag="cl")
    rows = chain.join(manual, on=s317.common.KEYS, how="inner", validate="1:1")
    rows = rows.join(clim, on=s317.common.KEYS, how="inner", validate="1:1")
    rows = rows.filter(pl.col("lead_hours") <= common.HORIZON_HOURS)
    stored = (
        pl.read_parquet(common.c319.E251_ARMS / "rows.parquet")
        .select(["site", "power_fcst_init_time", "valid_time"])
        .with_row_index("i251")
    )
    keys = ["site", "power_fcst_init_time", "valid_time"]
    rows = rows.join(stored, on=keys, how="left", validate="1:1")
    long_arms = pl.read_parquet(
        common.ARMS_DIR / "scored_long_lead.parquet",
        columns=[
            *keys,
            "cold_rule",
            *[f"{a}_{q}" for a in ("x1", *LONG_ARMS) for q in common.QUANTILES],
        ],
    )
    rows = rows.join(long_arms, on=keys, how="left", validate="1:1")
    rows = s317.flags(rows=rows)
    lead = rows["lead_hours"].to_numpy()
    long = lead > common.STORED_HORIZON_HOURS
    keep = long | rows["i251"].is_not_null().to_numpy()
    rows = rows.filter(pl.Series(keep))
    lead = rows["lead_hours"].to_numpy()
    long = lead > common.STORED_HORIZON_HOURS
    n = rows.height
    arms251 = {k: np.load(common.c319.E251_ARMS / f"{k}.npy") for k in ("arm1", "bg2")}
    idx = rows["i251"].fill_null(0).to_numpy()
    chain_q = rows.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    short_q = arms251["arm1"][idx].astype(np.float64)

    def long_or_default(*, name: str) -> np.ndarray:
        return rows.select([f"{name}_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)

    default_q = np.where(long[:, None], long_or_default(name="x1"), short_q)
    quantiles = {"default": default_q}
    for a in LONG_ARMS:
        quantiles[a] = np.where(long[:, None], long_or_default(name=a), short_q)
    quantiles["n"] = apply_width_by_month(q=default_q, rows=rows, log=log)
    quantiles["n_b1"] = apply_width_by_month(q=quantiles["b1"], rows=rows, log=log)
    quantiles["n_b2"] = apply_width_by_month(q=quantiles["b2"], rows=rows, log=log)
    quantiles["manual_heuristic"] = (
        rows.select([f"mh_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)
    )
    quantiles["climatology"] = (
        rows.select([f"cl_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)
    )
    quantiles["BG2"] = np.where(long[:, None], np.nan, arms251["bg2"][idx].astype(np.float64))
    actual = rows["actual"].to_numpy().astype(np.float64)
    site = rows["site"].to_numpy()
    threshold = np.array([np.quantile(actual[site == s], 0.9) for s in common.SCORED_SITES])
    thr = threshold[[common.SCORED_SITES.index(s) for s in site]]
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
    cold_rule = rows["cold_rule"].fill_null(False).to_numpy()
    cold_scored = rows["cold_winter_night"].to_numpy()
    rep: dict = {"n_rows": n, "n_long": int(long.sum())}
    rep["gate"] = {
        "default_0_240": float(loss["default"].mean()),
        "default_168_240": float(loss["default"][long].mean()),
        "expected_0_240": 2.749,
        "expected_168_240": 3.152,
        "default_below_168_equals_251": float(np.abs(default_q[~long] - short_q[~long]).max()),
        "b1_equals_default_below_168": float(
            np.abs(quantiles["b1"][~long] - default_q[~long]).max()
        ),
        "b1_equals_default_on_rule_nights_nonheuristic": None,
        "r319_harm_cold_winter_nights_168_240": float(
            loss["r319"][cold_scored & long].mean() - loss["default"][cold_scored & long].mean()
        ),
        "rule_rows_168_240": int((cold_rule & long).sum()),
        "rule_rows_overlap_scored_cold_group": int((cold_rule & cold_scored & long).sum()),
        "scored_cold_group_rows_168_240": int((cold_scored & long).sum()),
    }
    groups = {
        "all rows": np.ones(n, dtype=bool),
        "cold winter nights": cold_scored,
        "sunny daylight": rows["sunny_daylight"].to_numpy(),
        "winter (Nov-Mar)": winter,
        "April to October": ~winter,
    }
    windows = {
        "0-240 h": np.ones(n, dtype=bool),
        "0-168 h": ~long,
        "168-240 h": long,
    }
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
            for a in ARMS[1:]:
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
    # By Primary and leave one Primary out, at both windows.
    rep["by_site"] = {}
    rep["leave_one_out"] = {}
    for wname in ("0-240 h", "168-240 h"):
        wmask = windows[wname]
        rep["by_site"][wname] = {
            s: {
                f"{a}_vs_default": s319.contrast(
                    treat=loss[a], ref=loss["default"], mask=wmask & (site == s), blocks=month
                )
                for a in ARMS[1:]
            }
            for s in common.SCORED_SITES
        }
        rep["leave_one_out"][wname] = {
            s: {
                f"{a}_vs_default": s319.contrast(
                    treat=loss[a], ref=loss["default"], mask=wmask & (site != s), blocks=month
                )
                for a in ARMS[1:]
            }
            for s in common.SCORED_SITES
        }
    # Night and day split of N at each Primary (the D10 band-head comparison).
    night = arms.is_night_minute(valid_time=rows["valid_time"])
    rep["n_by_site_night"] = {
        s: {
            blk: s319.contrast(
                treat=loss["n"],
                ref=loss["default"],
                mask=(site == s) & (night == flag),
                blocks=month,
            )
            for blk, flag in (("night", True), ("day", False))
        }
        for s in common.SCORED_SITES
    }
    rep["n_pooled_other_five"] = s319.contrast(
        treat=loss["n"], ref=loss["default"], mask=site != "D10", blocks=month
    )
    rep["n_pooled_other_five_diff_sum"] = float(
        sum(
            rep["by_site"]["0-240 h"][s]["n_vs_default"]["diff"]
            for s in common.SCORED_SITES
            if s != "D10"
        )
        / 5
    )
    rep["median_minus_mean_by_site"] = {
        arm: {
            s: float((quantiles[arm][site == s, 5] - quantiles[arm][site == s].mean(axis=1)).mean())
            for s in common.SCORED_SITES
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
    rep["exploratory_weight_by_lead"] = lead_weight_table(
        rows=rows,
        chain_q=chain_q,
        heuristic=rows["mh_q50"].to_numpy().astype(np.float64),
    )
    rep["replicate"] = replicate_report(log=log)
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
        *[pl.Series(f"loss_{arm}", loss[arm].astype(np.float32)) for arm in (*ARMS, *REFERENCES)],
        *[pl.Series(f"{arm}_q50", quantiles[arm][:, 5].astype(np.float32)) for arm in ARMS],
    ).write_parquet(common.REPORT_DIR / "scored_rows.parquet")
    print(json.dumps(rep["gate"], default=float), flush=True)
    for label, e in rep["by_band"].items():
        print(
            label,
            e["n"],
            *[f"{a} {e[a]['est']:.3f}" for a in (*ARMS, *REFERENCES) if a in e],
            flush=True,
        )


if __name__ == "__main__":
    main()
