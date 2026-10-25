# ruff: noqa: E501
"""Stage 4, step 3 of experiment 00347: score every arm and reference row, and print the report.

Builds the reference rows that need no head (the plants with experiment 00259's spread factor and
offset table, the plants driven by ERA5), reads the stored rows of experiments 00339, 00326, 00324,
and 00303, scores everything on the 53 scored Tuesday runs (kept rows headline, all half-hours
beside, days 1 to 2 and days 3 to 10 separately), and prints `stage4_report.txt` with every number
the write-up quotes. Intervals resample whole calendar months of the run date (50,000 resamples);
farms are not resampled. Per-seed scores are averaged within each row before resampling.

    PYTHONPATH=<347>:<230>:<303>:<310>:<326>:<336 folders> \
    uv run --no-sync python ml-experiments/00347-*/e00347_s4_score.py
"""

import json
import os
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import e00303_common as c303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_ens_final as ens_final  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_score as score303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00326_score as score326  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00347_common as common  # noqa: E402
import e00347_s4_data as data  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

LEVEL: Final[float] = 1 - 0.01 / 4
"""Condition 1's interval level for four arms."""
ADOPTION_ORDER: Final[tuple[str, ...]] = ("y4", "y3", "y2", "y1")
HEAD_ARMS: Final[tuple[str, ...]] = ("y1", "y2", "y3", "y4")
MIN_GAIN: Final[float] = 0.03
TAIL_TOLERANCE: Final[float] = 0.05
ALL_TOLERANCE: Final[float] = 0.05
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
REPLICATE_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
SUBSETS: Final[tuple[str, ...]] = ("kept", "true_1_10", "all", "days_1_2", "days_3_10", "idle210")
SCOPES: Final[tuple[str, ...]] = ("kept", "true_1_10")
"""The two scopes of the headline: every lead in the row set (0.5 hours to 15 days), and true
calendar days 1 to 10 (lead hours 24 to 240), both on the kept rows."""
MEAN_LEVELS: Final[slice] = slice(1, 10)
PLANT_OF_ARM: Final[dict[str, str]] = {"y1": "y1", "y2": "y2", "y3": "y2", "y4": "y4"}
LABELS: Final[dict[str, str]] = {
    "y1": "y1 Stage 2 plant + MDN",
    "y2": "y2 Stage 3 plant + MDN",
    "y3": "y3 Stage 3 plant + MDN (mean, sd only)",
    "y4": "y4 pass-2 plant + MDN",
    "base": "base-only head, no farm columns (unequal first control)",
    "basefarm": "base-only head (Stage 3 plant median + lead + farm)",
    "s2_table": "Stage 2 plant + offset table",
    "s3_table": "Stage 3 plant + offset table",
    "s3b_table": "pass-2 plant + offset table",
    "s2_spread": "Stage 2 plant + spread factor",
    "s3_spread": "Stage 3 plant + spread factor",
    "s3b_spread": "pass-2 plant + spread factor",
    "s2_era5": "Stage 2 plant with ERA5",
    "s3_era5": "Stage 3 plant with ERA5",
    "s3b_era5": "pass-2 plant with ERA5",
    "default": "default (research default 1: XGBoost + table)",
    "default_notable": "XGBoost alone (no table)",
    "c": "XGBoost + adapter column + table (326 c)",
    "W4": "physics core + adapter + MDN (324 W4)",
    "P2": "physics core + adapter (303 P2)",
    "P3": "physics core + adapter + table (303 P3)",
    "W2": "pure XGBoost, own history (313)",
    "W3": "weather-only MDN (313)",
    "W0": "climatology by farm, month, hour (324 W0)",
}
FAMILIES: Final[tuple[str, ...]] = (
    "default",
    "default_notable",
    "c",
    "P2",
    "P3",
    "W2",
    "W3",
    "s2_table",
    "s3_table",
    "s3b_table",
)
COMPARISONS: Final[tuple[str, ...]] = ("default", "default_notable", "c", "W4", "W0")
TABLE_PLANTS: Final[dict[str, str]] = {"s2": "y1", "s3": "y2", "s3b": "y4"}
"""Reference-row prefix -> the stored plant summaries it uses."""


class Intervals(score303.Resampler):
    """The shared month resampler, read at 95% and at condition 1's level for four arms."""

    def read(self, *, values: np.ndarray, keep: np.ndarray) -> dict:
        """Return the mean, the 95% and 99.75% intervals, and the one-sided 95% upper bound."""
        picked, totals = self.draws(values=values, keep=keep)
        means = picked / np.maximum(totals, 1)
        out: dict = {"mean": float(values[:, keep].mean())}
        for level in (0.95, LEVEL):
            low, high = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
            out[f"{level:.4f}"] = [float(low), float(high)]
        out["upper_95_one_sided"] = float(np.quantile(means, 0.95))
        return out


def parts(*, name: str, seeds: tuple[int, ...]) -> np.ndarray:
    """Return the stored head quantiles of `name` for `seeds`: (seeds, rows, 11)."""
    return np.stack([np.load(data.PARTS_DIR / f"{name}_h{s}.npy") for s in seeds])


def stitched_leave_out(*, arm: str, cases: pl.DataFrame) -> np.ndarray:
    """Return the leave-one-farm-out head's scored quantiles for all rows: (1, rows, 11)."""
    out = np.full((len(cases), 11), np.nan, dtype=np.float32)
    sites = cases["site"].to_numpy()
    for farm in data.FARMS:
        out[sites == farm] = np.load(data.PARTS_DIR / f"{arm}_lofo_{farm}_h0.npy")
    return out[None]


def plant_table_rows(*, hindcast: pl.DataFrame, scored: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return experiment 00259's recipe on each plant: spread factor, then the offset table."""
    sites = c303.site_table()
    hind_cases = ens_final.hindcast_cases(sites=sites)
    usable = hind_cases["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    hind_usable = hind_cases.filter(pl.Series(usable))
    scored_clip = ens_final.scored_cases(sites=sites)
    assert scored_clip.select(*c303.KEY_COLUMNS).equals(scored.select(*c303.KEY_COLUMNS))
    all_cases = pl.concat(
        [
            hind_usable.select(*c303.KEY_COLUMNS, "lead_hours", "actual_norm"),
            scored_clip.select(*c303.KEY_COLUMNS, "lead_hours", "actual_norm"),
        ],
        how="vertical_relaxed",
    )
    out = {}
    for prefix, plant_name in TABLE_PLANTS.items():
        mix_h = data.mixtures(arm_plant=plant_name, set_name="hindcast")
        mix_s = data.mixtures(arm_plant=plant_name, set_name="scored")
        assert len(mix_h) == hind_usable.height, (len(mix_h), hind_usable.height)
        adjusted = c303.apply_spread(
            cases=all_cases, mixture=np.concatenate([mix_h, mix_s]).astype(np.float64)
        )
        adj_h, adj_s = adjusted[: hind_usable.height], adjusted[hind_usable.height :]
        table, keys = ens_final.offsets_table(hindcast=hind_usable, quantiles=adj_h)
        out[f"{prefix}_spread"] = adj_s.astype(np.float32)[None]
        out[f"{prefix}_table"] = ens_final.apply_offsets(
            cases=scored_clip, quantiles=adj_s, table=table, keys=keys
        )[None]
        out[f"{prefix}_era5"] = data.era5_quantiles(arm_plant=plant_name)[None]
    return out


def load_arms(*, cases: pl.DataFrame, hindcast: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return every scored arm and reference row, each (seeds, rows, 11)."""
    arms: dict[str, np.ndarray] = {}
    for arm in (*HEAD_ARMS, "base"):
        arms[arm] = parts(name=arm, seeds=SEEDS)
        arms[f"{arm}_placebo"] = parts(name=f"{arm}_placebo", seeds=SEEDS)
    arms["basefarm"] = parts(name="basefarm", seeds=SEEDS)
    for arm in HEAD_ARMS:
        arms[f"{arm}_placebo2"] = parts(name=f"{arm}_placebo2", seeds=SEEDS)
    for arm in ("y2", "y4"):
        arms[f"{arm}_lofo"] = stitched_leave_out(arm=arm, cases=cases)
    arms.update(plant_table_rows(hindcast=hindcast, scored=cases))
    arms["default"] = np.load(data.E339_FINAL / "default.npy")
    arms["default_notable"] = np.load(data.E339_FINAL / "default_notable.npy")
    arms["c"] = np.load(data.E326_FINAL / "c.npy")
    arms["W4"] = np.load(data.E324_FINAL / "W4.npy")
    arms["W0"] = np.load(data.E324_FINAL / "W0.npy")
    for name in ("P2", "P3", "W2", "W3"):
        arms[name] = np.load(data.E339_FINAL / f"{name}.npy")
    return arms


def squared_errors(*, quantiles: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-seed, per-row squared errors (points squared) of the mean and median forecasts."""
    scale = c303.POINTS_PER_FRACTION
    mean_sq, median_sq = [], []
    for q in quantiles.astype(np.float64):
        mean_forecast = q[:, MEAN_LEVELS] @ c303.NINE_MASSES
        mean_sq.append(((actual - mean_forecast) * scale) ** 2)
        median_sq.append(((actual - q[:, c303.MEDIAN_INDEX]) * scale) ** 2)
    return {"mean_sq_error": np.stack(mean_sq), "median_sq_error": np.stack(median_sq)}


def fmt(entry: dict) -> str:
    """Return 'mean (95% low to high; 99.75% low to high)' for an interval entry."""
    low, high = entry["0.9500"]
    wide_low, wide_high = entry[f"{LEVEL:.4f}"]
    return (
        f"{entry['mean']:+.3f} (95% {low:+.3f} to {high:+.3f}; "
        f"99.75% {wide_low:+.3f} to {wide_high:+.3f})"
    )


def main() -> None:  # noqa: PLR0912, PLR0915
    """Score everything and write the report and the score file."""
    cases = data.scored_frame().with_columns(lead_day=pl.col("lead_day").clip(0, 10))
    hindcast = data.hindcast_frame()
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    arms = load_arms(cases=cases, hindcast=hindcast)
    flags = {
        "all": np.ones(cases.height, dtype=bool),
        "kept": ~cases["masked"].to_numpy(),
        "idle210": cases["masked"].to_numpy(),
        "days_1_2": cases["lead_day"].is_between(1, 2).to_numpy(),
        "days_3_10": cases["lead_day"].is_between(3, 10).to_numpy(),
        "true_1_10": (~cases["masked"] & cases["lead_hours"].is_between(24, 240)).to_numpy(),
    }
    sites = cases["site"].to_numpy()
    resampler = Intervals(month=cases["month"].to_numpy(), seeds=1)
    lines: list[str] = ["STAGE 4: physics plant + MDN head on the 51 ensemble members"]
    report: dict = {"scores": {}}

    metrics: dict[str, dict[str, np.ndarray]] = {}
    for name, q in arms.items():
        raw = {
            **score303.row_metrics(quantiles=q, actual=actual),
            **score326.extra_metrics(quantiles=q, actual=actual),
        }
        raw.update(squared_errors(quantiles=q, actual=actual))
        metrics[name] = {k: v.mean(axis=0, keepdims=True) for k, v in raw.items()}
        report["seed_scores_kept"] = report.get("seed_scores_kept", {})
        report["seed_scores_kept"][name] = [
            float(s) for s in raw["qs"][:, flags["kept"]].mean(axis=1)
        ]
    qs = {name: m["qs"] for name, m in metrics.items()}

    lines.append(
        f"Scored cases {cases.height} ({int(flags['kept'].sum())} kept, {int(flags['idle210'].sum())} "
        "in experiment 210's idle band), 53 Tuesday runs, 2025-07 to 2026-06; farms W1 to W3."
    )

    lines.append("-- absolute quantile scores (points of capacity; mean over seeds) --")
    header = "arm".ljust(58) + "".join(s.rjust(11) for s in SUBSETS) + "   seed sd (kept)"
    lines.append(header)
    for name in metrics:
        if name not in LABELS and not name.endswith(("_placebo", "_placebo2", "_lofo")):
            continue
        label = LABELS.get(name, name)
        row = "".join(f"{qs[name][0, flags[s]].mean():11.3f}" for s in SUBSETS)
        seeds = report["seed_scores_kept"][name]
        sd = f"{np.std(seeds, ddof=1):.3f}" if len(seeds) > 1 else "n/a"
        lines.append(f"{label[:57].ljust(58)}{row}   {sd}")
        report["scores"][name] = {s: float(qs[name][0, flags[s]].mean()) for s in SUBSETS}

    # Benchmarks that cover only some rows.
    benchmarks = {}
    for bench in ("CLIM", "PERS"):
        ref = score326.benchmark_frame(name=bench, cases=cases)
        if ref is None:
            continue
        present = np.isfinite(ref[0]).all(axis=1)
        row_score = (
            c303.quantile_score(quantiles=np.nan_to_num(ref[0]), actual=actual)
            * c303.POINTS_PER_FRACTION
        )
        benchmarks[bench] = (present, row_score[None])
        keep = flags["kept"] & present
        lines.append(
            f"{bench} (stored by experiment 230): {int(keep.sum())} kept rows, score "
            f"{row_score[keep].mean():.3f}; default on the same rows "
            f"{qs['default'][0, keep].mean():.3f}; y2 {qs['y2'][0, keep].mean():.3f}; "
            f"y4 {qs['y4'][0, keep].mean():.3f}"
        )
        report["scores"][bench] = {"rows": int(keep.sum()), "score": float(row_score[keep].mean())}

    lines.append("-- by farm, kept rows (points) --")
    for name in ("default", "y1", "y2", "y3", "y4", "base", "basefarm", "s3_table", "s3b_table"):
        row = "  ".join(
            f"{f} {qs[name][0, flags['kept'] & (sites == f)].mean():.3f}" for f in data.FARMS
        )
        lines.append(f"{LABELS[name][:57].ljust(58)}{row}")
    lines.append(
        "-- by calendar lead day (0 to 15, from the run's date to the valid time's date), kept rows (points) --"
    )
    days = data.scored_frame()["lead_day"].to_numpy().astype(int)
    rows_per_day = [int((flags["kept"] & (days == d)).sum()) for d in range(16)]
    report["kept_rows_per_lead_day"] = rows_per_day
    lines.append("kept rows per day: " + " ".join(f"{d}:{n}" for d, n in enumerate(rows_per_day)))
    lines.append("arm".ljust(58) + "".join(f"{d:7d}" for d in range(16)))
    lead_names = ("default", "default_notable", "c", "W4", "y1", "y2", "y3", "y4", "s3_table")
    for name in lead_names:
        row = "".join(
            (
                f"{qs[name][0, flags['kept'] & (days == d)].mean():7.2f}"
                if rows_per_day[d]
                else "    n/a"
            )
            for d in range(16)
        )
        lines.append(f"{LABELS[name][:57].ljust(58)}{row}")
    report["by_lead_day"] = {
        name: [
            float(qs[name][0, flags["kept"] & (days == d)].mean()) if rows_per_day[d] else None
            for d in range(16)
        ]
        for name in lead_names
    }

    # Contrasts against every comparison row.
    lines.append(
        "-- contrasts: new arm minus comparison row, quantile score in points (negative favours the "
        "new arm); month-resampled intervals --"
    )
    contrasts: dict = {}
    contrast_arms = (*HEAD_ARMS, "s2_table", "s3_table", "s3b_table")
    for arm in contrast_arms:
        for ref in COMPARISONS:
            for subset in ("kept", "true_1_10", "all", "days_1_2", "days_3_10"):
                entry = resampler.read(values=qs[arm] - qs[ref], keep=flags[subset])
                contrasts[f"{arm} - {ref} | {subset}"] = entry
                lines.append(f"{arm} minus {ref}, {subset}: {fmt(entry)}")
        for bench, (present, row_score) in benchmarks.items():
            for scope in SCOPES:
                keep = flags[scope] & present
                if not keep.any():
                    lines.append(f"{arm} minus {bench} ({scope}): no rows carry a {bench} forecast")
                    continue
                entry = resampler.read(values=qs[arm] - row_score, keep=keep)
                contrasts[f"{arm} - {bench} | {scope}"] = entry
                lines.append(
                    f"{arm} minus {bench} ({scope}, rows with a {bench} forecast, {int(keep.sum())}): {fmt(entry)}"
                )
    report["contrasts"] = contrasts

    lines.append("-- contrasts between our arms and controls, kept rows --")
    pairs = {
        "y2 - y1": ("y2", "y1"),
        "y3 - y2": ("y3", "y2"),
        "y4 - y2": ("y4", "y2"),
        "y2 - base (no farm columns)": ("y2", "base"),
        "y2 - basefarm": ("y2", "basefarm"),
        "y3 - basefarm": ("y3", "basefarm"),
        "basefarm - default": ("basefarm", "default"),
        "basefarm - base (no farm columns)": ("basefarm", "base"),
        "y2 placebo - y2": ("y2_placebo", "y2"),
        "y4 placebo - y4": ("y4_placebo", "y4"),
        "y2 spread placebo - y2": ("y2_placebo2", "y2"),
        "y4 spread placebo - y4": ("y4_placebo2", "y4"),
        "y1 spread placebo - y1": ("y1_placebo2", "y1"),
        "y3 spread placebo - y3": ("y3_placebo2", "y3"),
        "y2 spread placebo - basefarm": ("y2_placebo2", "basefarm"),
        "y2 - s3 table": ("y2", "s3_table"),
        "y4 - s3b table": ("y4", "s3b_table"),
        "s3 table - s2 table": ("s3_table", "s2_table"),
        "s3b table - s3 table": ("s3b_table", "s3_table"),
        "y2 leave-one-farm-out - default": ("y2_lofo", "default"),
        "y4 leave-one-farm-out - default": ("y4_lofo", "default"),
        "y2 leave-one-farm-out - y2": ("y2_lofo", "y2"),
        "y4 leave-one-farm-out - y4": ("y4_lofo", "y4"),
        "y2 - W4": ("y2", "W4"),
    }
    pair_results = {}
    for label, (a, b) in pairs.items():
        for subset in ("kept", "true_1_10", "all"):
            entry = resampler.read(values=qs[a] - qs[b], keep=flags[subset])
            pair_results[f"{label} | {subset}"] = entry
            lines.append(f"{label}, {subset}: {fmt(entry)}")
    report["pairs"] = pair_results

    # Tails.
    tail_names = (
        "default",
        "default_notable",
        "c",
        "W4",
        "y1",
        "y2",
        "y3",
        "y4",
        "base",
        "basefarm",
        "y2_placebo",
        "y2_placebo2",
        "y2_lofo",
        "s3_table",
        "s3b_table",
    )
    tails_by_scope: dict[str, dict[str, list[float]]] = {}
    for scope in SCOPES:
        lines.append(
            f"-- tail table, scope {scope}: share of observations below q01 / q05 / above q95 / q99 (nominal 1%, 5%, 5%, 1%), pinball losses, 5-95 coverage (nominal 90%) --"
        )
        lines.append(
            "arm".ljust(58)
            + "".join(
                h.rjust(9)
                for h in ("<q01", "<q05", ">q95", ">q99", "p01", "p05", "p95", "p99", "cover")
            )
        )
        tails = {}
        for name in tail_names:
            m = metrics[name]
            k = flags[scope]
            values = [
                m[key][0, k].mean() for key in ("below_q01", "below_q05", "above_q95", "above_q99")
            ]
            values = [100 * v for v in values] + [
                m[key][0, k].mean() for key in ("p01", "p05", "p95", "p99", "covered")
            ]
            values[-1] *= 100
            tails[name] = [float(v) for v in values]
            lines.append(
                f"{LABELS.get(name, name)[:57].ljust(58)}" + "".join(f"{v:9.3f}" for v in values)
            )
        tails_by_scope[scope] = tails
        lines.append("(shares and coverage are in percent)")
    tails = tails_by_scope["kept"]
    report["tails"] = tails
    report["tails_true_1_10"] = tails_by_scope["true_1_10"]

    # Checks: mean and median forecast.
    lines.append(
        "-- checks: median forecast minus mean forecast per farm, kept rows (points of capacity) --"
    )
    for name in ("default", "y1", "y2", "y3", "y4", "basefarm"):
        row = "  ".join(
            f"{f} {metrics[name]['median_minus_mean'][0, flags['kept'] & (sites == f)].mean():+.2f}"
            for f in data.FARMS
        )
        lines.append(f"{LABELS[name][:57].ljust(58)}{row}")
    lines.append(
        "-- checks: error of the mean and median forecast beside each quantile-score contrast (arm minus default). "
        "The mean forecast is the 9-level mean; absolute error is minimised by the median, so squared error is the matching yardstick for a mean forecast. --"
    )
    trigger3 = {}
    trigger3_abs_mean_only = {}
    check_stats = {
        "quantile score": "qs",
        "mean-forecast squared error": "mean_sq_error",
        "median-forecast squared error": "median_sq_error",
        "mean-forecast absolute error": "mean_abs_error",
        "median-forecast absolute error": "median_abs_error",
    }
    for scope in SCOPES:
        for arm in (*HEAD_ARMS, "basefarm", "s3_table", "s3b_table"):
            entries = {
                label: resampler.read(
                    values=metrics[arm][key] - metrics["default"][key], keep=flags[scope]
                )
                for label, key in check_stats.items()
            }
            lines.append(
                f"{scope} {arm} minus default: "
                + "; ".join(f"{label} {fmt(e)}" for label, e in entries.items())
            )
            sign = np.sign(entries["quantile score"]["mean"])
            fires = bool(
                sign != np.sign(entries["mean-forecast squared error"]["mean"])
                or sign != np.sign(entries["median-forecast absolute error"]["mean"])
            )
            trigger3[f"{arm} | {scope}"] = fires
            trigger3_abs_mean_only[f"{arm} | {scope}"] = bool(
                sign != np.sign(entries["mean-forecast absolute error"]["mean"])
            )
            report.setdefault("checks", {})[f"{arm} | {scope}"] = entries
    for scope in SCOPES:
        for arm in ("default", "y2", "y4"):
            k = flags[scope]
            rmse = float(np.sqrt(metrics[arm]["mean_sq_error"][0, k].mean()))
            lines.append(f"{scope} {arm}: RMSE of the mean forecast {rmse:.2f} points")
    lines.append(
        "trigger 3 (sign differs between the quantile score and the mean-forecast squared error or the "
        f"median-forecast absolute error): {trigger3}"
    )
    lines.append(
        "  for reference, sign differs between the quantile score and the mean-forecast absolute error "
        f"(not a valid check for a mean forecast): {trigger3_abs_mean_only}"
    )
    report["trigger3"] = trigger3
    report["trigger3_abs_mean_only"] = trigger3_abs_mean_only

    # Leave-one-farm-out for contrasts.
    lines.append("-- leave-one-farm-out for contrasts: arm minus default, kept rows --")
    lofo_flip = {}
    for arm in HEAD_ARMS:
        row_signs = []
        for farm in data.FARMS:
            site = sites == farm
            without = resampler.read(values=qs[arm] - qs["default"], keep=flags["kept"] & ~site)
            only = resampler.read(values=qs[arm] - qs["default"], keep=flags["kept"] & site)
            row_signs.append(np.sign(without["mean"]))
            lines.append(
                f"{arm} minus default without {farm}: {fmt(without)}; only {farm}: {fmt(only)}"
            )
        full = np.sign(contrasts[f"{arm} - default | kept"]["mean"])
        lofo_flip[arm] = bool(any(s != full for s in row_signs))
    lines.append(f"trigger 4 (leaving a farm out flips the sign): {lofo_flip}")
    report["trigger4"] = lofo_flip

    # Standard architecture and families.
    lines.append(
        "-- standard architecture: best plant + MDN arm against the best single family (kept rows) --"
    )
    family_scores = {f: float(qs[f][0, flags["kept"]].mean()) for f in FAMILIES}
    best_family = min(family_scores, key=family_scores.__getitem__)
    best_arm = min(HEAD_ARMS, key=lambda a: float(qs[a][0, flags["kept"]].mean()))
    for family, score in sorted(family_scores.items(), key=lambda kv: kv[1]):
        lines.append(f"family {LABELS[family]}: {score:.3f}")
    gap = resampler.read(values=qs[best_arm] - qs[best_family], keep=flags["kept"])
    gap_all = resampler.read(values=qs[best_arm] - qs[best_family], keep=flags["all"])
    relative = qs[best_arm][0, flags["kept"]].mean() / qs[best_family][0, flags["kept"]].mean() - 1
    lines.append(
        f"standard architecture (physics plant + MDN head): {best_arm} {qs[best_arm][0, flags['kept']].mean():.3f}; "
        f"best single family {LABELS[best_family]} {family_scores[best_family]:.3f}; gap kept {fmt(gap)}, "
        f"all half-hours {fmt(gap_all)}, relative {100 * relative:+.2f}%"
    )
    report["standard_architecture"] = {
        "arm": best_arm,
        "best_family": best_family,
        "family_scores": family_scores,
        "gap_kept": gap,
        "gap_all": gap_all,
        "relative_kept": float(relative),
    }
    gap_true = resampler.read(values=qs[best_arm] - qs[best_family], keep=flags["true_1_10"])
    relative_true = (
        qs[best_arm][0, flags["true_1_10"]].mean() / qs[best_family][0, flags["true_1_10"]].mean()
        - 1
    )
    lines.append(
        f"standard architecture on true days 1 to 10 (kept): {best_arm} {qs[best_arm][0, flags['true_1_10']].mean():.3f}; "
        f"{LABELS[best_family]} {qs[best_family][0, flags['true_1_10']].mean():.3f}; gap {fmt(gap_true)}, relative {100 * relative_true:+.2f}%"
    )
    report["standard_architecture"]["gap_true_1_10"] = gap_true
    report["standard_architecture"]["relative_true_1_10"] = float(relative_true)
    lines.append("standard architecture against XGBoost families alone:")
    for family in ("default", "default_notable", "c"):
        for scope in SCOPES:
            g = resampler.read(values=qs[best_arm] - qs[family], keep=flags[scope])
            lines.append(f"  {scope} {best_arm} minus {family}: {fmt(g)}")

    # Replicate.
    lines.append(
        "-- replicate: heads trained on 2024-04 to 2024-11 (stopping on December 2024), scored on the daily runs 2025-01 to 2025-06 --"
    )
    rep_cases = pl.read_parquet(data.E336_FINAL / "replicate_cases.parquet").with_columns(
        lead_day=pl.col("lead_day").clip(0, 10)
    )
    rep_actual = rep_cases["actual_norm"].to_numpy().astype(np.float64)
    rep_flags = {
        "all": np.ones(rep_cases.height, dtype=bool),
        "kept": ~rep_cases["masked"].to_numpy(),
        "days_1_2": rep_cases["lead_day"].is_between(1, 2).to_numpy(),
        "days_3_10": rep_cases["lead_day"].is_between(3, 10).to_numpy(),
        "true_1_10": (
            ~rep_cases["masked"] & rep_cases["lead_hours"].is_between(24, 240)
        ).to_numpy(),
    }
    rep_resampler = Intervals(month=rep_cases["month"].to_numpy(), seeds=1)
    default_rep = np.load(data.E339_FINAL / "default_rep.npy")
    rep_default_qs = (
        np.mean(
            [
                c303.quantile_score(quantiles=q.astype(np.float64), actual=rep_actual)
                for q in default_rep
            ],
            axis=0,
        )[None]
        * c303.POINTS_PER_FRACTION
    )
    replicate = {}
    for arm in (*HEAD_ARMS, "base", "basefarm"):
        q_rep = np.stack([np.load(data.PARTS_DIR / f"{arm}_rep_h{s}.npy") for s in REPLICATE_SEEDS])
        rep_qs = (
            np.mean(
                [
                    c303.quantile_score(quantiles=q.astype(np.float64), actual=rep_actual)
                    for q in q_rep
                ],
                axis=0,
            )[None]
            * c303.POINTS_PER_FRACTION
        )
        entry: dict = {
            s: rep_resampler.read(values=rep_qs - rep_default_qs, keep=rep_flags[s])
            for s in rep_flags
        }
        months = np.unique(rep_cases["month"].to_numpy())
        for scope in SCOPES:
            monthly = np.array(
                [
                    (rep_qs - rep_default_qs)[
                        0, rep_flags[scope] & (rep_cases["month"].to_numpy() == m)
                    ].mean()
                    for m in months
                ]
            )
            test = score326.t_test(monthly=monthly)
            suffix = "" if scope == "kept" else "_true_1_10"
            entry[f"t_test{suffix}"] = test
            entry[f"monthly{suffix}"] = monthly.tolist()
            lines.append(
                f"{arm} replicate, {scope}: score {rep_qs[0, rep_flags[scope]].mean():.3f} (default {rep_default_qs[0, rep_flags[scope]].mean():.3f}); "
                f"minus default {fmt(entry[scope])}; monthly differences "
                f"{[round(float(v), 3) for v in monthly]}; t {test['t']:+.2f}, p {test['p_two_sided']:.3f} over {test['months']} months"
            )
        entry["score_kept"] = float(rep_qs[0, rep_flags["kept"]].mean())
        replicate[arm] = entry
        lines.append(f"{arm} replicate, all half-hours: minus default {fmt(entry['all'])}")
        if arm == "base":
            lines.append(
                "  (the line above is the base-only head without farm columns, the unequal first control)"
            )
    report["replicate"] = replicate

    # Pre-registered predictions.
    lines.append("-- pre-registered predictions --")
    y2_y1 = pair_results["y2 - y1 | kept"]["mean"]
    y2_def = contrasts["y2 - default | kept"]["mean"]
    y3_y2 = pair_results["y3 - y2 | kept"]["mean"]
    y4_y2 = pair_results["y4 - y2 | kept"]["mean"]
    y2_base = pair_results["y2 - basefarm | kept"]["mean"]
    y2_base_unequal = pair_results["y2 - base (no farm columns) | kept"]["mean"]
    above_q99 = {a: tails[a][3] for a in HEAD_ARMS}
    predictions = {
        "y2 beats y1 by under 0.05 (y2 - y1 > -0.05)": y2_y1 > -0.05,
        "y2 within 0.15 of the default (y2 - default <= 0.15) and does not beat it by 0.03 (> -0.03)": (
            y2_def <= 0.15
        )
        and (y2_def > -0.03),
        "y3 within 0.03 of y2 (|y3 - y2| < 0.03)": abs(y3_y2) < 0.03,
        "head fed only the plant median (plus the lead terms and the farm) does as well as the full head (y2 - basefarm >= -0.03)": y2_base
        >= -0.03,
        "share above q99 stays above 1% for every arm": all(v > 1.0 for v in above_q99.values()),
        "y4 differs from y2 by under 0.05 (|y4 - y2| < 0.05)": abs(y4_y2) < 0.05,
    }
    for text, held in predictions.items():
        lines.append(f"{'HOLDS' if held else 'FAILS'}: {text}")
    lines.append(
        f"values: y2-y1 {y2_y1:+.3f}; y2-default {y2_def:+.3f}; y3-y2 {y3_y2:+.3f}; y2-basefarm {y2_base:+.3f} (y2 minus the unequal base head without farm columns: {y2_base_unequal:+.3f}); "
        f"y4-y2 {y4_y2:+.3f}; share above q99 {', '.join(f'{a} {v:.2f}%' for a, v in above_q99.items())}"
    )
    report["predictions"] = {k: bool(v) for k, v in predictions.items()}

    # Adoption conditions.
    gate = json.loads((data.E336_FINAL / "reproduction.json").read_text())
    lines.append(
        f"reproduction gate: the default re-fit scores {gate['kept_score_refit']:.3f} against the stored "
        f"experiment 259/303 P1 {gate['kept_score_stored_p1']:.3f} (difference {gate['gate_difference']:.3f}); "
        f"gate passed: {gate['gate_passed']}"
    )
    seed_sd = {a: float(np.std(report["seed_scores_kept"][a], ddof=1)) for a in HEAD_ARMS}
    lines.append(f"five-seed spread of the kept score (sd): {seed_sd}")
    adoption: dict[str, dict] = {}
    for scope in SCOPES:
        lines.append(
            f"-- adoption conditions on scope {scope} "
            f"({'every lead in the row set, 0.5 hours to 15 days' if scope == 'kept' else 'true calendar days 1 to 10, lead hours 24 to 240'}) --"
        )
        suffix = "" if scope == "kept" else "_true_1_10"
        for arm in ADOPTION_ORDER:
            main_entry = contrasts[f"{arm} - default | {scope}"]
            low, high = main_entry[f"{LEVEL:.4f}"]
            gain = -main_entry["mean"]
            tail_bounds = {
                t: resampler.read(
                    values=metrics[arm][t] - metrics["default"][t], keep=flags[scope]
                )["upper_95_one_sided"]
                for t in ("p01", "p99")
            }
            rep = replicate[arm]
            rep_test = rep[f"t_test{suffix}"]
            contradicted = rep_test["t"] > 0 and rep_test["p_two_sided"] < 0.05
            cond3 = bool(rep[scope]["mean"] < 0 and not contradicted)
            placebo_all = resampler.read(values=qs[f"{arm}_placebo"] - qs[arm], keep=flags[scope])[
                "mean"
            ]
            placebo = resampler.read(values=qs[f"{arm}_placebo2"] - qs[arm], keep=flags[scope])[
                "mean"
            ]
            entry = {
                "gain": gain,
                "interval_99_75": [low, high],
                "condition_1": bool(high < 0 and gain >= MIN_GAIN),
                "tail_upper_bounds": tail_bounds,
                "condition_2": all(v <= TAIL_TOLERANCE for v in tail_bounds.values()),
                "all_half_hours_difference": contrasts[f"{arm} - default | all"]["mean"],
                "all_half_hours_within_tolerance": bool(
                    contrasts[f"{arm} - default | all"]["mean"] <= ALL_TOLERANCE
                ),
                "condition_3": cond3,
                "placebo_all_summaries_minus_arm": placebo_all,
                "placebo_spread_only_minus_arm": placebo,
                "placebo_loses_gain": bool(placebo > 0.5 * max(gain, 0.0)) if gain > 0 else False,
                "seed_sd_below_half_gain": bool(gain > 0 and seed_sd[arm] < 0.5 * gain),
                "gate": bool(gate["gate_passed"]),
            }
            entry["condition_4_numeric"] = bool(
                entry["gate"] and entry["placebo_loses_gain"] and entry["seed_sd_below_half_gain"]
            )
            entry["all_five_numeric"] = bool(
                entry["condition_1"]
                and entry["condition_2"]
                and entry["condition_3"]
                and entry["condition_4_numeric"]
            )
            adoption[f"{arm} | {scope}"] = entry
            lines.append(
                f"{arm}: gain {gain:+.3f} (99.75% interval of arm minus default {low:+.3f} to {high:+.3f}): condition 1 {entry['condition_1']}; "
                f"p01 / p99 harm upper bounds {tail_bounds['p01']:+.3f} / {tail_bounds['p99']:+.3f}: condition 2 {entry['condition_2']}; "
                f"replicate {rep[scope]['mean']:+.3f}, t {rep_test['t']:+.2f}, p {rep_test['p_two_sided']:.3f}: condition 3 {cond3}; "
                f"spread-only placebo minus arm {placebo:+.3f} (all-summaries placebo {placebo_all:+.3f}), seed sd {seed_sd[arm]:.3f}: condition 4 (numeric part) {entry['condition_4_numeric']}; "
                f"all half-hours difference {entry['all_half_hours_difference']:+.3f} (tolerance {ALL_TOLERANCE})"
            )
        passing = [a for a in ADOPTION_ORDER if adoption[f"{a} | {scope}"]["all_five_numeric"]]
        report[f"adopted_numeric_{scope}"] = passing[0] if passing else None
        lines.append(
            f"arm passing every numeric condition ({scope}): {report[f'adopted_numeric_{scope}']}"
        )
    report["adoption"] = adoption

    # Trigger 5: nested arms.
    lines.append("-- audit triggers --")
    nested = {}
    for label, (nested_arm, containing) in {
        "y3 (mean, sd) inside y2": ("y3", "y2"),
        "base (median and farm) inside y2": ("basefarm", "y2"),
        "base (median and farm) inside y3": ("basefarm", "y3"),
    }.items():
        entry = resampler.read(values=qs[nested_arm] - qs[containing], keep=flags["kept"])
        nested[label] = {"entry": entry, "fires": bool(entry["0.9500"][1] < 0)}
        lines.append(
            f"trigger 5, {label}: nested minus containing {fmt(entry)}; fires {nested[label]['fires']}"
        )
    placebo_entries = {}
    for arm in HEAD_ARMS:
        entry = resampler.read(values=qs[f"{arm}_placebo"] - qs[arm], keep=flags["kept"])
        placebo_entries[arm] = entry
        lines.append(f"shuffled-summary placebo minus {arm}: {fmt(entry)}")
    report["trigger5"] = {k: v["fires"] for k, v in nested.items()}
    # Trigger 6: interval width against experiment 339's x1 on the same rows.
    sister = json.loads((data.E339_FINAL.parent / "score.json").read_text())["contrasts"]["scored"][
        "x1"
    ]["kept | qs"]["0.9500"]
    sister_width = sister[1] - sister[0]
    widths = {}
    for arm in HEAD_ARMS:
        low, high = contrasts[f"{arm} - default | kept"]["0.9500"]
        widths[arm] = (high - low) / sister_width
    lines.append(
        f"trigger 6: 95% interval width of (arm minus default, kept) over experiment 339's x1 minus default ({sister_width:.3f}): "
        + ", ".join(f"{a} {w:.2f}" for a, w in widths.items())
    )
    report["trigger6_ratios"] = widths
    report["trigger6_fires"] = any(w > 2 or w < 0.5 for w in widths.values())

    lines.append("-- audit triggers: which fire --")
    earlier = (common.DATA_DIR / "stage123b_report.txt").read_text()
    injection_failures = [
        line
        for line in earlier.splitlines()
        if "pre-registered criterion" in line and "NOT met" in line
    ]
    prediction_failures = [text for text, held in predictions.items() if not held]
    stage_failures = [
        line
        for line in earlier.splitlines()
        if line.startswith("PREDICTION CHECK") and "FAILS" in line
    ]
    fired = {
        "1 (a control or known-answer test fails)": bool(injection_failures),
        "2 (a result contradicts its own prediction)": bool(prediction_failures or stage_failures),
        "3 (quantile score and mean-forecast squared error or median error disagree in sign)": any(
            trigger3[f"{a} | {scope}"] for a in HEAD_ARMS for scope in SCOPES
        ),
        "4 (leaving a farm out flips a contrast)": any(lofo_flip.values()),
        "5 (a nested arm beats its container)": any(v["fires"] for v in nested.values()),
        "6 (headline interval twice or half as wide as a sister experiment's)": report[
            "trigger6_fires"
        ],
    }
    for name, did in fired.items():
        lines.append(f"trigger {name}: {'FIRES' if did else 'does not fire'}")
    lines.append(
        f"  trigger 1 evidence: {len(injection_failures)} Stage 3b injection-test rows missed the pre-registered criterion (W1 and W2 rows listed in stage123b_report.txt)"
    )
    lines.append(
        f"  trigger 2 evidence: Stage 4 predictions failed: {prediction_failures}; Stage 3 prediction failures: {stage_failures}"
    )
    lines.append(
        "trigger 7 (a later experiment's design depends on this result and it is unreplicated): not known to this session; the manager decides. The replicate period supports the sign of the headline contrast."
    )
    report["triggers_fired"] = fired
    lines.append(
        "Curtailment-cost saving against the manual heuristic and threshold-weighted CRPS: not computed here (this experiment's scoring is the quantile score and the tail table; the predictions with q01 to q99, actual, lead_hours, and time_series_id are stored for the node metric)."
    )
    for arm in HEAD_ARMS:
        frame = cases.select(
            "time_series_id", "power_fcst_init_time", "valid_time", "lead_hours", "site"
        ).with_columns(
            actual=pl.Series(actual),
            **{
                c303.LEVEL_COLUMNS[k]: pl.Series(arms[arm][:, :, k].mean(axis=0)) for k in range(11)
            },
        )
        frame.write_parquet(common.DATA_DIR / "s4" / f"predictions_{arm}.parquet")
    lines.append(
        "predictions (five-seed mean of the sorted quantiles) stored privately as s4/predictions_<arm>.parquet; experiment_store.write_predictions not called"
    )

    lines.append("-- serving --")
    lines.append(
        "Each arm needs, per farm: the frozen plant's parameters (a JSON of about 15 numbers: cut-in, rated, "
        "cut-out and their widths, scale, blur, hub height, and for y2 to y4 three direction dips), a pass of "
        "each of the 51 ensemble members' 100 m speed, 10 m speed, 2 m temperature and 100 m direction through the plant, "
        "the summary of the 51 power values, and one small neural network (two hidden layers of 64 units, three-component "
        "Gaussian mixture) that reads the summary, the lead, and the farm. No XGBoost model, no offset table, "
        "no neural adapter, and no ERA5 at forecast time."
    )

    bin_rows = []
    for name in ("default", "y2", "y4", "s3_table"):
        median = arms[name][:, :, 5].mean(axis=0)
        kept = flags["kept"]
        edges = np.quantile(median[kept], np.linspace(0, 1, 21))
        index = np.clip(np.digitize(median, edges[1:-1]), 0, 19)
        for b in range(20):
            sel = kept & (index == b)
            bin_rows.append(
                {
                    "arm": name,
                    "bin": b,
                    "forecast_median": float(median[sel].mean()),
                    "actual_mean": float(actual[sel].mean()),
                    "q05_mean": float(arms[name][:, sel, 1].mean()),
                    "q95_mean": float(arms[name][:, sel, 9].mean()),
                    "share_in_band": float(
                        (
                            (actual[sel] >= arms[name][:, sel, 1].mean(axis=0))
                            & (actual[sel] <= arms[name][:, sel, 9].mean(axis=0))
                        ).mean()
                    ),
                }
            )
    pl.DataFrame(bin_rows).write_parquet(common.DATA_DIR / "s4" / "median_bins.parquet")

    out_path = common.DATA_DIR / "stage4_report.txt"
    out_path.write_text("\n".join(lines) + "\n")
    (common.DATA_DIR / "s4" / "score.json").write_text(json.dumps(report, indent=1, default=float))
    save_rows(cases=cases, qs=qs, flags=flags)
    print("\n".join(lines))


def save_rows(*, cases: pl.DataFrame, qs: dict[str, np.ndarray], flags: dict) -> None:
    """Store per-row seed-mean scores for the charts."""
    frame = pl.DataFrame(
        {
            "site": cases["site"],
            "month": cases["month"],
            "lead_hours": cases["lead_hours"],
            "lead_day": cases["lead_day"],
            "masked": cases["masked"],
            **{f"qs__{name}": pl.Series(v[0]) for name, v in qs.items()},
        }
    )
    frame.write_parquet(Path(common.DATA_DIR) / "s4" / "rows.parquet")


if __name__ == "__main__":
    main()
