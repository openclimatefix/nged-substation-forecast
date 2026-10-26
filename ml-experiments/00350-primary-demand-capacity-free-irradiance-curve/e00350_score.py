# ruff: noqa: E501, PLR0912, PLR0915
"""Score arms A, B, and C of experiment 350 and print every number the page quotes.

Rows: the scored year (runs from 2025-07-02 to 2026-06-30, origin 09:00 UTC), six Primaries, rows
with a reading outside the hand-filtered events, and a stored forecast of both references. The
quantile score is twice the mean pinball loss over the 11 quantiles, in points of effective
capacity. Intervals resample the 12 months of the forecast run's initialisation time, paired across
arms, 20,000 draws, and are read at the 95% level and at the 97.5% level that the Bonferroni
correction over the two pre-registered arms (B and C) sets.

    PYTHONPATH=ml-experiments/00317-primary-demand-default-1-at-7-to-10-days:ml-experiments/00350-primary-demand-capacity-free-irradiance-curve:<folders of experiments 319, 320, 322> \
        uv run --no-sync python ml-experiments/00350-*/e00350_score.py
"""

import itertools
import json
import pickle
from typing import Final

import e00136_common as c136  # ty: ignore[unresolved-import]
import e00317_score as s317  # ty: ignore[unresolved-import]
import e00350_common as common
import e00350_curve as curve
import e00350_rows as rows_module
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_row_difference

N_RESAMPLES: Final[int] = 20_000
LEAD_RANGES: Final[dict[str, float]] = {"0.5-240 h": 0.0, "24-240 h": 24.0}
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (("B", "A"), ("C", "A"), ("C", "B"))
ARMS_SCORED: Final[tuple[str, ...]] = ("A", "B", "C")
LEVEL_WEIGHTS: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (common.LEVELS[1:] + common.LEVELS[:-1]) / 2.0, [1.0]])
)
"""Midpoint-rule weights over the 11 levels on [0, 1]; twice the weighted pinball loss approximates the CRPS."""
TOLERANCE: Final[dict[str, float]] = {
    "all rows": 0.02,
    "cold winter nights": 0.05,
    "sunny daylight": 0.05,
}


def pin(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the pinball loss at each of the 11 levels, shape (rows, 11)."""
    return s317.pin(actual=actual, q=q)


def draws_for(*, blocks: np.ndarray, mask: np.ndarray, rng: np.random.Generator):  # noqa: ANN201
    """Return block codes and resample indices for the rows in `mask`."""
    labels, codes = np.unique(blocks[mask], return_inverse=True)
    return codes, len(labels), rng.integers(0, len(labels), size=(N_RESAMPLES, len(labels)))


def contrast(
    *, treat: np.ndarray, ref: np.ndarray, mask: np.ndarray, blocks: np.ndarray, seed: int
) -> dict:
    """Return the mean difference (treat minus ref) with 95% and 97.5% intervals and a one-sided 95% bound."""
    if mask.sum() == 0:
        return {"n": 0}
    codes, n, draws = draws_for(blocks=blocks, mask=mask, rng=np.random.default_rng(seed))
    st = np.bincount(codes, weights=treat[mask], minlength=n)
    sr = np.bincount(codes, weights=ref[mask], minlength=n)
    nn = np.bincount(codes, minlength=n).astype(np.float64)
    count = nn[draws].sum(axis=1)
    diff = st[draws].sum(axis=1) / count - sr[draws].sum(axis=1) / count
    lo95, hi95 = np.percentile(diff, [2.5, 97.5])
    lo975, hi975 = np.percentile(diff, [1.25, 98.75])
    return {
        "n": int(mask.sum()),
        "treat": float(st.sum() / nn.sum()),
        "ref": float(sr.sum() / nn.sum()),
        "diff": float((st.sum() - sr.sum()) / nn.sum()),
        "lo95": float(lo95),
        "hi95": float(hi95),
        "lo975": float(lo975),
        "hi975": float(hi975),
        "hi_one_sided_95": float(np.percentile(diff, 95.0)),
    }


def mean_interval(*, values: np.ndarray, mask: np.ndarray, blocks: np.ndarray, seed: int) -> dict:
    """Return a mean over a mask with a month-resampled 95% interval."""
    if mask.sum() == 0:
        return {"n": 0}
    codes, n, draws = draws_for(blocks=blocks, mask=mask, rng=np.random.default_rng(seed))
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


def metrics_of(
    *, actual: np.ndarray, q: np.ndarray, threshold: np.ndarray
) -> dict[str, np.ndarray]:
    """Return per-row metrics: quantile score, CRPS approximation, twCRPS, MAE of the median, tails."""
    pinball = pin(actual=actual, q=q)
    clipped_actual = np.maximum(actual, threshold)[:, None]
    clipped_q = np.maximum(q, threshold[:, None])
    error = clipped_actual - clipped_q
    twpin = np.maximum(common.LEVELS * error, (common.LEVELS - 1.0) * error)
    return {
        "qs": 2.0 * pinball.mean(axis=1),
        "crps": 2.0 * (pinball * LEVEL_WEIGHTS).sum(axis=1),
        "twcrps": 2.0 * twpin.mean(axis=1),
        "mae_median": np.abs(actual - q[:, 5]),
        "pin_q01": pinball[:, 0],
        "pin_q99": pinball[:, 10],
        "cover_q01": (actual <= q[:, 0]).astype(np.float64),
        "cover_q99": (actual <= q[:, 10]).astype(np.float64),
        "cover_90": ((actual >= q[:, 1]) & (actual <= q[:, 9])).astype(np.float64),
    }


class Report:
    """Collects the report's lines and a JSON mirror of every number."""

    def __init__(self) -> None:
        """Start an empty report."""
        self.lines: list[str] = []
        self.data: dict = {}

    def add(self, *, text: str = "") -> None:
        """Append one line."""
        self.lines.append(text)
        print(text, flush=True)


def build_rows(*, period: str) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return the joined rows of the three arms and the references, and each arm's quantile array."""
    cache = c136.load_cache()
    arms = {arm: rows_module.load_arm(arm=arm, period=period) for arm in common.ARMS}
    base = arms["A"]
    for arm in ("B", "C"):
        same = arms[arm].select(rows_module.ROW_KEYS).equals(
            base.select(rows_module.ROW_KEYS)
        ) and np.array_equal(arms[arm]["actual"].to_numpy(), base["actual"].to_numpy())
        if not same:
            msg = f"arm {arm} has other rows than arm A"
            raise ValueError(msg)
    quantiles = {
        arm: arms[arm].select(list(common.QUANTILES)).to_numpy().astype(np.float64)
        for arm in common.ARMS
    }
    frame = rows_module.add_sun_flags(frame=base, cache=cache)
    return frame, quantiles


def main() -> None:
    """Join the arms and references, run the gate, score every contrast, and write the report."""
    report = Report()
    out = report.add
    frame, quantiles = build_rows(period="scored")
    manual = s317.read_reference(name="ref_manual_heuristic", tag="mh")
    clim = s317.read_reference(name="ref_climatology", tag="cl")
    keys_ts = s317.common.KEYS
    frame = frame.with_row_index("row")
    joined = frame.join(manual, on=keys_ts, how="left", validate="1:1").join(
        clim, on=keys_ts, how="left", validate="1:1"
    )
    have = joined["mh_q50"].is_not_null() & joined["cl_q50"].is_not_null()
    out(
        text=f"# Experiment 350: scored-year report\n\nRows in the arms: {frame.height:,}; rows with both references: {int(have.sum()):,}.\n"
    )
    keep = have.to_numpy()
    joined = joined.filter(pl.Series(keep))
    quantiles = {k: v[keep] for k, v in quantiles.items()}
    quantiles["manual_heuristic"] = (
        joined.select([f"mh_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)
    )
    quantiles["climatology"] = (
        joined.select([f"cl_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)
    )
    flagged = s317.flags(rows=joined.drop("sunny_daylight", "daylight"))
    # `s317.flags` joins experiment 178's `sunny`; only its cold-winter-night flag is used here.
    cold = flagged["cold_winter_night"].to_numpy()
    known = flagged["known_flags"].to_numpy()
    actual = joined["actual"].to_numpy().astype(np.float64)
    lead = joined["lead_hours"].to_numpy()
    site = joined["site"].to_numpy()
    month = joined["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    threshold = np.array([np.quantile(actual[site == s], 0.9) for s in common.SCORED_SITES])
    thr = threshold[[common.SCORED_SITES.index(s) for s in site]]
    allq = quantiles
    per_row = {k: metrics_of(actual=actual, q=v, threshold=thr) for k, v in allq.items()}
    day = joined["daylight"].to_numpy()
    sunny = joined["sunny_daylight"].to_numpy()
    groups = {
        "all rows": np.ones(len(actual), dtype=bool),
        "sunny daylight": sunny,
        "all daylight": day,
        "all night": ~day,
        "cold winter nights": cold,
    }
    out(
        text=f"Rows with known cold-night flag: {int(known.sum()):,} of {len(known):,}; sunny daylight rows: {int(sunny.sum()):,}; daylight rows: {int(day.sum()):,}; cold winter night rows: {int(cold.sum()):,}.\n"
    )
    out(
        text="Sunny threshold (ensemble-mean shortwave, W/m2, 75th percentile of the Primary's daylight half-hours): "
        + ", ".join(
            f"{s} {joined.filter(pl.col('site') == s)['sunny_threshold_w_m2'][0]:.0f}"
            for s in common.SCORED_SITES
        )
        + "\n"
    )
    report_data: dict = {
        "n_rows": len(actual),
        "group_rows": {k: int(v.sum()) for k, v in groups.items()},
    }

    # --- Reproduction gate -------------------------------------------------------------------
    out(text="## Reproduction gate (P4)\n")
    stored = (
        pl.concat(
            [
                pl.read_parquet(s317.common.CHAIN_DIR / f"{s}.parquet")
                .filter(pl.col("period") == "scored")
                .with_columns(site=pl.lit(s))
                for s in common.SCORED_SITES
            ]
        )
        .filter(~pl.col("switching_mask") & pl.col("actual").is_finite())
        .sort(rows_module.ROW_KEYS)
    )
    stored_aligned = joined.select(rows_module.ROW_KEYS).join(
        stored, on=rows_module.ROW_KEYS, how="left", validate="1:1"
    )
    if stored_aligned["q50"].null_count():
        msg = "experiment 317's stored chain lacks rows that arm A has"
        raise ValueError(msg)
    qs_stored = metrics_of(
        actual=actual,
        q=stored_aligned.select(list(common.QUANTILES)).to_numpy().astype(np.float64),
        threshold=thr,
    )["qs"]
    gate = {
        "qs_arm_a": float(per_row["A"]["qs"].mean()),
        "qs_stored_317_chain": float(qs_stored.mean()),
        "qs_difference": float(per_row["A"]["qs"].mean() - qs_stored.mean()),
        "max_abs_quantile_diff": float(
            np.abs(allq["A"] - stored_aligned.select(list(common.QUANTILES)).to_numpy()).max()
        ),
        "rows": len(actual),
    }
    stored_fits_diff = max(
        float(
            json.loads((common.CHAIN_DIR / "A" / f"gates_{s}.json").read_text())[
                "theta_max_abs_diff_to_stored_fits"
            ]
        )
        for s in common.SCORED_SITES
    )
    gate["max_abs_theta_diff_to_stored_fits_of_136"] = stored_fits_diff
    gate["passes_within_0.005"] = abs(gate["qs_difference"]) < 0.005
    out(text="```json\n" + json.dumps(gate, indent=2) + "\n```\n")
    report_data["gate"] = gate

    # --- Default 3 reference rows --------------------------------------------------------------
    out(text="## Stored Primary demand research default 3, as a reference row\n")
    d3 = default3_quantiles(joined=joined)
    in_d3 = d3["available"]
    out(
        text=f"Rows with a stored default 3 forecast: {int(in_d3.sum()):,} of {len(actual):,} (below 168 h from experiment 251's A+H+S; above 168 h from experiment 322's D1).\n"
    )
    d3_metrics = metrics_of(
        actual=actual, q=np.where(in_d3[:, None], d3["q"], np.nan), threshold=thr
    )
    report_data["default3"] = {}
    for gname, gmask in groups.items():
        mask = gmask & in_d3
        if mask.sum() < 50:
            continue
        entry = {
            "n": int(mask.sum()),
            "default3": mean_interval(values=d3_metrics["qs"], mask=mask, blocks=month, seed=1),
            "A": mean_interval(values=per_row["A"]["qs"], mask=mask, blocks=month, seed=1),
            "A_minus_default3": contrast(
                treat=per_row["A"]["qs"], ref=d3_metrics["qs"], mask=mask, blocks=month, seed=2
            ),
            "manual_heuristic": mean_interval(
                values=per_row["manual_heuristic"]["qs"], mask=mask, blocks=month, seed=1
            ),
        }
        report_data["default3"][gname] = entry
        out(
            text=f"- {gname} (n={entry['n']:,}): default 3 {entry['default3']['est']:.3f}, arm A {entry['A']['est']:.3f}, arm A minus default 3 {entry['A_minus_default3']['diff']:+.3f} [{entry['A_minus_default3']['lo95']:+.3f}, {entry['A_minus_default3']['hi95']:+.3f}], manual heuristic {entry['manual_heuristic']['est']:.3f}"
        )
    out()

    # --- Absolute scores ---------------------------------------------------------------------
    out(text="## Absolute quantile score (points of effective capacity; smaller is better)\n")
    out(text="| Group | Lead range | n | A | B | C | manual heuristic | climatology |")
    out(text="|---|---|---|---|---|---|---|---|")
    report_data["absolute"] = {}
    for gname, gmask in groups.items():
        for wname, lo in LEAD_RANGES.items():
            mask = gmask & (lead >= lo)
            cells = {
                a: mean_interval(values=per_row[a]["qs"], mask=mask, blocks=month, seed=1)
                for a in (*ARMS_SCORED, "manual_heuristic", "climatology")
            }
            report_data["absolute"][f"{gname}|{wname}"] = cells
            out(
                text=f"| {gname} | {wname} | {int(mask.sum()):,} | "
                + " | ".join(
                    f"{cells[a]['est']:.3f} [{cells[a]['lo']:.3f}, {cells[a]['hi']:.3f}]"
                    for a in (*ARMS_SCORED, "manual_heuristic", "climatology")
                )
                + " |"
            )
    out()

    # --- Contrasts ---------------------------------------------------------------------------
    out(
        text="## Contrasts in the quantile score (first minus second; negative favours the first)\n"
    )
    out(text="Columns: difference [95% interval] [97.5% interval] (one-sided 95% upper bound).\n")
    out(
        text="| Group | Lead range | Contrast | Difference | 95% interval | 97.5% interval | One-sided 95% upper bound |"
    )
    out(text="|---|---|---|---|---|---|---|")
    report_data["contrasts"] = {}
    for gname, gmask in groups.items():
        for wname, lo in LEAD_RANGES.items():
            mask = gmask & (lead >= lo)
            for a, b in CONTRASTS:
                c = contrast(
                    treat=per_row[a]["qs"], ref=per_row[b]["qs"], mask=mask, blocks=month, seed=3
                )
                report_data["contrasts"][f"{gname}|{wname}|{a}-{b}"] = c
                out(
                    text=f"| {gname} | {wname} | {a} - {b} | {c['diff']:+.3f} | [{c['lo95']:+.3f}, {c['hi95']:+.3f}] | [{c['lo975']:+.3f}, {c['hi975']:+.3f}] | {c['hi_one_sided_95']:+.3f} |"
                )
    out()
    head = [("sunny daylight", w, a) for w in LEAD_RANGES for a in ("B", "C")]
    # Cross-check one headline interval with the shared month-block bootstrap of `studies.bootstrap`.
    mask = groups["sunny daylight"] & (lead >= 24.0)
    shared = bootstrap_row_difference(
        values=(per_row["B"]["qs"] - per_row["A"]["qs"])[mask], months=month[mask]
    )
    out(
        text=f"Cross-check of B - A, sunny daylight, 24-240 h with `studies.bootstrap.bootstrap_row_difference` (2,000 draws): {shared['difference']:+.3f} [{shared['lower_95']:+.3f}, {shared['upper_95']:+.3f}] over {shared['n_months']} months.\n"
    )
    report_data["bootstrap_cross_check"] = dict(shared)

    out(text="## Decision rule on the headline row (sunny daylight half-hours)\n")
    report_data["decision"] = {}
    for _, w, a in head:
        c = report_data["contrasts"][f"sunny daylight|{w}|{a}-A"]
        simplicity = c["hi975"] < 0.05
        gain = c["hi975"] < 0.0
        no_worse_than_005 = c["diff"] <= 0.05
        p1_in_band = -0.10 <= c["diff"] <= 0.05
        report_data["decision"][f"{a}|{w}"] = {
            "diff": c["diff"],
            "hi975": c["hi975"],
            "simplicity_case": bool(simplicity),
            "gain_case": bool(gain),
            "point_estimate_within_p1_band": bool(p1_in_band),
            "point_estimate_not_worse_than_0.05": bool(no_worse_than_005),
        }
        out(
            text=f"- arm {a}, {w}: difference {c['diff']:+.3f}, 97.5% upper bound {c['hi975']:+.3f}; simplicity case (upper bound < +0.05) {simplicity}; gain case (interval wholly below 0) {gain}; P1 band (-0.10 to +0.05) {p1_in_band}"
        )
    c = report_data["contrasts"]
    for w in LEAD_RANGES:
        cb, cc = c[f"sunny daylight|{w}|B-A"], c[f"sunny daylight|{w}|C-A"]
        ccb = c[f"sunny daylight|{w}|C-B"]
        out(
            text=f"- P2 at {w}: C minus B on sunny daylight {ccb['diff']:+.3f} [{ccb['lo95']:+.3f}, {ccb['hi95']:+.3f}] (within 0.03: {abs(ccb['diff']) < 0.03})"
        )
        _ = (cb, cc)
    out()

    # --- No-harm groups ----------------------------------------------------------------------
    out(
        text="## No-harm groups (one-sided 95% upper bound of the harm against the research-default tolerance)\n"
    )
    out(
        text="| Arm | Group | Lead range | Difference | One-sided 95% upper bound | Tolerance | Passes |"
    )
    out(text="|---|---|---|---|---|---|---|")
    report_data["no_harm"] = {}
    for a in ("B", "C"):
        for gname in ("all rows", "cold winter nights", "sunny daylight"):
            for wname, lo in LEAD_RANGES.items():
                mask = groups[gname] & (lead >= lo)
                cc = report_data["contrasts"][f"{gname}|{wname}|{a}-A"]
                tol = TOLERANCE[gname]
                passes = cc["hi_one_sided_95"] <= tol
                report_data["no_harm"][f"{a}|{gname}|{wname}"] = {
                    "tolerance": tol,
                    "passes": bool(passes),
                    **cc,
                }
                out(
                    text=f"| {a} | {gname} | {wname} | {cc['diff']:+.3f} | {cc['hi_one_sided_95']:+.3f} | {tol} | {passes} |"
                )
        for wname, lo in LEAD_RANGES.items():
            mask = lead >= lo
            for tail in ("pin_q01", "pin_q99"):
                cc = contrast(
                    treat=per_row[a][tail], ref=per_row["A"][tail], mask=mask, blocks=month, seed=4
                )
                passes = cc["hi_one_sided_95"] <= 0.05
                report_data["no_harm"][f"{a}|{tail}|{wname}"] = {
                    "tolerance": 0.05,
                    "passes": bool(passes),
                    **cc,
                }
                out(
                    text=f"| {a} | {tail} pinball, all rows | {wname} | {cc['diff']:+.3f} | {cc['hi_one_sided_95']:+.3f} | 0.05 | {passes} |"
                )
            cc = contrast(
                treat=per_row[a]["twcrps"],
                ref=per_row["A"]["twcrps"],
                mask=mask,
                blocks=month,
                seed=5,
            )
            relative = cc["hi_one_sided_95"] / cc["ref"]
            passes = relative <= 0.02
            report_data["no_harm"][f"{a}|twcrps|{wname}"] = {
                "tolerance_relative": 0.02,
                "passes": bool(passes),
                **cc,
            }
            out(
                text=f"| {a} | twCRPS (relative), all rows | {wname} | {cc['diff'] / cc['ref']:+.3%} | {relative:+.3%} | 2% | {passes} |"
            )
    out()

    # --- Tails, coverage, CRPS, MAE ------------------------------------------------------------
    out(text="## Tails, coverage, CRPS and error of the median, by arm\n")
    out(
        text="| Group | Lead range | Arm | Quantile score | CRPS approx. | twCRPS | MAE of median | q01 pinball | q99 pinball | q01 coverage | q99 coverage | 5-95% coverage |"
    )
    out(text="|---|---|---|---|---|---|---|---|---|---|---|---|")
    report_data["tails"] = {}
    for gname in ("all rows", "sunny daylight", "cold winter nights"):
        for wname, lo in LEAD_RANGES.items():
            mask = groups[gname] & (lead >= lo)
            for a in (*ARMS_SCORED, "manual_heuristic", "climatology"):
                m = per_row[a]
                vals = {k: float(v[mask].mean()) for k, v in m.items()}
                report_data["tails"][f"{gname}|{wname}|{a}"] = vals
                out(
                    text=f"| {gname} | {wname} | {a} | {vals['qs']:.3f} | {vals['crps']:.3f} | {vals['twcrps']:.3f} | {vals['mae_median']:.3f} | {vals['pin_q01']:.4f} | {vals['pin_q99']:.4f} | {vals['cover_q01']:.4f} | {vals['cover_q99']:.4f} | {vals['cover_90']:.4f} |"
                )
    out()
    out(
        text="Contrasts of the CRPS approximation and of the median's absolute error against arm A (negative favours the arm):\n"
    )
    out(text="| Arm | Metric | Group | Lead range | Difference | 95% interval |")
    out(text="|---|---|---|---|---|---|")
    for a in ("B", "C"):
        for metric in ("crps", "mae_median"):
            for gname in ("all rows", "sunny daylight", "cold winter nights"):
                for wname, lo in LEAD_RANGES.items():
                    mask = groups[gname] & (lead >= lo)
                    cc = contrast(
                        treat=per_row[a][metric],
                        ref=per_row["A"][metric],
                        mask=mask,
                        blocks=month,
                        seed=6,
                    )
                    report_data["contrasts"][f"{metric}|{gname}|{wname}|{a}-A"] = cc
                    out(
                        text=f"| {a} | {metric} | {gname} | {wname} | {cc['diff']:+.3f} | [{cc['lo95']:+.3f}, {cc['hi95']:+.3f}] |"
                    )
    out()
    # Median minus mean forecast per site (audit "Checks").
    out(
        text="## Checks: median minus mean forecast, per Primary, scored year (points; mean over rows)\n"
    )
    out(text="| Primary | A | B | C |")
    out(text="|---|---|---|---|")
    report_data["median_minus_mean"] = {}
    means = {
        a: rows_module.load_arm(arm=a, period="scored").sort(rows_module.ROW_KEYS)
        for a in common.ARMS
    }
    for s in common.SCORED_SITES:
        cells = []
        for a in common.ARMS:
            part = means[a].filter(pl.col("site") == s)
            cells.append(float(np.mean((part["q50"] - part["mean"]).to_numpy())))
        report_data["median_minus_mean"][s] = dict(zip(common.ARMS, cells, strict=True))
        out(text=f"| {s} | " + " | ".join(f"{v:+.4f}" for v in cells) + " |")
    out()
    out(
        text="Error of the mean forecast (mean of actual minus mean, absolute; points), sunny daylight, 24-240 h:\n"
    )
    mean_arrays = {
        a: means[a]
        .join(joined.select(rows_module.ROW_KEYS), on=rows_module.ROW_KEYS, how="semi")
        .sort(rows_module.ROW_KEYS)["mean"]
        .to_numpy()
        .astype(np.float64)
        for a in common.ARMS
    }
    mask = groups["sunny daylight"] & (lead >= 24.0)
    for a in common.ARMS:
        e = np.abs(actual - mean_arrays[a])
        out(
            text=f"- arm {a}: mean absolute error of the mean forecast {e[mask].mean():.3f}, signed (actual minus mean) {float((actual - mean_arrays[a])[mask].mean()):+.3f}"
        )
    for a in ("B", "C"):
        e_a = np.abs(actual - mean_arrays["A"])
        e_x = np.abs(actual - mean_arrays[a])
        cc = contrast(treat=e_x, ref=e_a, mask=mask, blocks=month, seed=7)
        report_data["contrasts"][f"mean_forecast_mae|sunny daylight|24-240 h|{a}-A"] = cc
        out(
            text=f"- {a} - A, mean-forecast absolute error, sunny daylight, 24-240 h: {cc['diff']:+.3f} [{cc['lo95']:+.3f}, {cc['hi95']:+.3f}]"
        )
    out()

    # --- Leave one Primary out and per Primary -------------------------------------------------
    out(
        text="## Leave-one-Primary-out and per-Primary headline contrasts (sunny daylight, difference to arm A)\n"
    )
    out(
        text="| Left out / Primary | Lead range | B - A (leave out) | C - A (leave out) | B - A (alone) | C - A (alone) |"
    )
    out(text="|---|---|---|---|---|---|")
    report_data["leave_one_out"] = {}
    for s in common.SCORED_SITES:
        for wname, lo in LEAD_RANGES.items():
            base_mask = groups["sunny daylight"] & (lead >= lo)
            cells = {}
            for a in ("B", "C"):
                cells[f"{a}_without"] = contrast(
                    treat=per_row[a]["qs"],
                    ref=per_row["A"]["qs"],
                    mask=base_mask & (site != s),
                    blocks=month,
                    seed=8,
                )
                cells[f"{a}_alone"] = contrast(
                    treat=per_row[a]["qs"],
                    ref=per_row["A"]["qs"],
                    mask=base_mask & (site == s),
                    blocks=month,
                    seed=9,
                )
            report_data["leave_one_out"][f"{s}|{wname}"] = cells
            out(
                text=f"| {s} | {wname} | "
                + " | ".join(
                    f"{cells[k]['diff']:+.3f} [{cells[k]['lo95']:+.3f}, {cells[k]['hi95']:+.3f}]"
                    for k in ("B_without", "C_without", "B_alone", "C_alone")
                )
                + " |"
            )
    out()

    # --- Replicate period ----------------------------------------------------------------------
    out(
        text="## Replicate period (even months of 2024-07 to 2025-06; sunny daylight; the sign check of the research-default rule)\n"
    )
    rep_frame, rep_q = build_rows(period="replicate")
    rep_actual = rep_frame["actual"].to_numpy().astype(np.float64)
    rep_lead = rep_frame["lead_hours"].to_numpy()
    rep_month = rep_frame["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    rep_loss = {a: 2.0 * pin(actual=rep_actual, q=rep_q[a]).mean(axis=1) for a in common.ARMS}
    rep_sunny = rep_frame["sunny_daylight"].to_numpy()
    report_data["replicate"] = {}
    for wname, lo in LEAD_RANGES.items():
        mask = rep_sunny & (rep_lead >= lo)
        for a in ("B", "C"):
            cc = contrast(
                treat=rep_loss[a], ref=rep_loss["A"], mask=mask, blocks=rep_month, seed=10
            )
            report_data["replicate"][f"{a}|{wname}"] = cc
            out(
                text=f"- {a} - A, {wname}, n={cc['n']:,}: {cc['diff']:+.3f} [{cc['lo95']:+.3f}, {cc['hi95']:+.3f}]; arm A {cc['ref']:.3f}"
            )
    out()

    # --- Fitted curves -----------------------------------------------------------------------
    out(
        text="## Fitted curves at the last cut (2026-06-01): points of effective capacity subtracted from demand\n"
    )
    grid = np.array([100.0, 200.0, 400.0, 600.0, 800.0, 1000.0])
    out(text="| Primary | Arm | " + " | ".join(f"{g:.0f} W/m2" for g in grid) + " |")
    out(text="|---|---|" + "---|" * len(grid))
    report_data["curves"] = {}
    for s in common.SCORED_SITES:
        for a in ("B", "C"):
            fits = pickle.loads((common.FITS_DIR / a / f"{s}.pkl").read_bytes())
            theta = fits[-1].theta[-curve.N_RAMPS :]
            values = curve.ramps(ghi_w_m2=grid) @ theta
            report_data["curves"][f"{s}|{a}"] = values.tolist()
            out(text=f"| {s} | {a} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    shape = np.mean(
        [
            pickle.loads((common.FITS_DIR / "B" / f"{s}.pkl").read_bytes())[-1].theta[
                -curve.N_RAMPS :
            ]
            / pickle.loads((common.FITS_DIR / "B" / f"{s}.pkl").read_bytes())[-1]
            .theta[-curve.N_RAMPS :]
            .sum()
            for s in common.SCORED_SITES
        ],
        axis=0,
    )
    out(
        text=f"\nArm B shared shape at the last cut (share of the 1000 W/m2 value reached at {', '.join(f'{g:.0f}' for g in grid)} W/m2): "
        + ", ".join(f"{v:.2f}" for v in curve.ramps(ghi_w_m2=grid) @ shape)
    )
    b_losses = json.loads((common.FITS_DIR / "b_losses.json").read_text())
    rises = max(max(b - a for a, b in itertools.pairwise(v)) for v in b_losses.values())
    out(
        text=f"Largest rise of the arm B penalised loss between alternating rounds, over all cuts: {rises:.3g} (0 or negative means the alternation never raised the loss)."
    )
    report_data["b_loss_largest_rise"] = rises

    common.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (common.REPORT_DIR / "report.md").write_text("\n".join(report.lines) + "\n")
    common.write_json_atomic(path=common.REPORT_DIR / "report.json", content=report_data)
    out(text=f"\nReport: {common.REPORT_DIR / 'report.md'}")


def default3_quantiles(*, joined: pl.DataFrame) -> dict:
    """Return the stored default 3 quantiles on the joined rows, and which rows have one."""
    import e00320_common as c320  # ty: ignore[unresolved-import]
    import e00322_common as c322  # ty: ignore[unresolved-import]

    keys = rows_module.ROW_KEYS
    stored251 = (
        pl.read_parquet(c320.c319.E251_ARMS / "rows.parquet").select(*keys).with_row_index("i251")
    )
    arm1 = np.load(c320.c319.E251_ARMS / "arm1.npy").astype(np.float64)
    long = pl.read_parquet(
        c322.ARMS_DIR / "scored_long_lead.parquet",
        columns=[*keys, *[f"d1_{q}" for q in common.QUANTILES]],
    )
    out = (
        joined.select(*keys, "lead_hours")
        .join(stored251, on=keys, how="left", validate="1:1")
        .join(long, on=keys, how="left", validate="1:1")
    )
    is_long = (out["lead_hours"] > c320.STORED_HORIZON_HOURS).to_numpy()
    has_short = out["i251"].is_not_null().to_numpy()
    has_long = out["d1_q50"].is_not_null().to_numpy()
    short_q = arm1[out["i251"].fill_null(0).to_numpy()]
    long_q = out.select([f"d1_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)
    q = np.where(is_long[:, None], long_q, short_q)
    available = np.where(is_long, has_long, has_short)
    return {"q": q, "available": available}


if __name__ == "__main__":
    main()
