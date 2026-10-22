# ruff: noqa: E501, PLR0912, PLR0915
"""Print every number the page quotes and write `report.json` beside the private score tables.

Reads the `meta_`, `after_`, `scores_`, `tails_`, and `spans_` files of `e00226_run.py` and
experiment 87's stored best guess 2 predictions.

    OMP_NUM_THREADS=1 uv run --no-sync python e00226_report.py
"""

from __future__ import annotations

import json
from typing import Final

import numpy as np
import polars as pl
from e00226_alarms import ARMS
from e00226_common import WORK_DIR
from studies.sources import ML_EXPERIMENTS_DATA_DIR
from studies.tail_calibration import tail_calibration_table

SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16", "D9")
LEARNED: Final[tuple[str, ...]] = tuple(ARMS)
FOLDLESS: Final[tuple[str, ...]] = ("K0", "PAIR_NEW", "PAIR_GATED")
E87: Final = ML_EXPERIMENTS_DATA_DIR / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
QCOLS: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
KEYS: Final[list[str]] = ["site", "power_fcst_init_time", "valid_time"]
WINTER: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
N_DRAWS: Final[int] = 20_000
CONTROLS: Final[dict[str, float]] = {"K0": 13.686, "PAIR_NEW": 6.321, "BG2": 10.091}


def boot(*, num: np.ndarray, groups: np.ndarray, seed: int = 203) -> tuple[float, float, float]:
    """Mean of `num` with a 95% interval from resampling whole groups (ratio of sums)."""
    labels, inverse = np.unique(groups, return_inverse=True)
    n = np.bincount(inverse, weights=num, minlength=len(labels))
    d = np.bincount(inverse, minlength=len(labels)).astype(float)
    picks = np.random.default_rng(seed).integers(0, len(labels), size=(N_DRAWS, len(labels)))
    draws = n[picks].sum(axis=1) / d[picks].sum(axis=1)
    return (
        float(n.sum() / d.sum()),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def fmt(t: tuple[float, float, float]) -> str:
    """Format an estimate with its interval."""
    return f"{t[0]:+.3f} ({t[1]:+.3f} to {t[2]:+.3f})"


def bg2() -> pl.DataFrame:
    """Best guess 2's per-row score and tail quantiles in points, keyed like the stack's rows."""
    f = pl.read_parquet(E87 / "predictions" / "p_s6_bg2" / "predictions.parquet").filter(
        pl.col("site").is_in(SITES)
    )
    f = f.select(
        *KEYS,
        (pl.col("actual").cast(pl.Float64) * 100).alias("actual"),
        *[(pl.col(c).cast(pl.Float64) * 100).alias(c) for c in QCOLS],
    )
    from e00226_g_arms import quantile_score

    score = quantile_score(actual=f["actual"].to_numpy(), quantiles=f.select(QCOLS).to_numpy())
    return f.with_columns(qs=pl.Series(score), mean_forecast=pl.col("q50")).filter(
        pl.col("qs").is_finite()
    )


def load() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """The meta, event-row, score, tail, and span tables of every site."""

    def cat(name: str) -> pl.DataFrame:
        frames = [pl.read_parquet(WORK_DIR / f"{name}_{s}.parquet") for s in SITES]
        return pl.concat(
            [
                f if "site" in f.columns else f.with_columns(site=pl.lit(s))
                for f, s in zip(frames, SITES, strict=True)
            ]
        )

    return cat("meta"), cat("after"), cat("scores"), cat("tails"), cat("spans")


def main() -> None:
    """Compute and print the headline contrasts, the controls, and the checks."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    meta, after, scores, tails, spans = load()
    meta = meta.with_columns(
        mo=pl.col("valid_time").dt.month(), month=pl.col("valid_time").dt.strftime("%Y-%m")
    )
    report: dict = {}

    # ---- ordinary days: one column per arm; learned arms averaged over the three folds ----
    ordinary = meta.filter(pl.col("ordinary")).select("site", "row", "month", "mo", "actual")
    sc = scores.join(meta.select("site", "row", "ordinary"), on=["site", "row"], how="left")
    sc = sc.filter(pl.col("ordinary"))
    wide = (
        sc.group_by("site", "row", "arm")
        .agg(
            qs=pl.col("qs").mean(),
            q50=pl.col("q50").mean(),
            mean_forecast=pl.col("mean_forecast").mean(),
        )
        .pivot(on="arm", index=["site", "row"], values=["qs", "q50", "mean_forecast"])
    )
    wide = ordinary.join(wide, on=["site", "row"])
    print(
        "ordinary rows:",
        wide.height,
        "by site",
        wide.group_by("site").len().sort("site").to_dicts(),
    )

    def month_contrast(frame: pl.DataFrame, a: str, b: str = "K0") -> tuple[float, float, float]:
        diff = (frame[f"qs_{a}"] - frame[f"qs_{b}"]).to_numpy().astype(float)
        return boot(num=diff, groups=frame["month"].to_numpy())

    print("\n== G2: ordinary-day rows, arm minus K0 (points of effective capacity) ==")
    for arm in ("PAIR_NEW", "PAIR_GATED", *LEARNED):
        all_rows = month_contrast(wide, arm)
        winter = month_contrast(wide.filter(pl.col("mo").is_in(list(WINTER))), arm)
        other = month_contrast(wide.filter(~pl.col("mo").is_in(list(WINTER))), arm)
        mean_err = (
            (
                (wide["actual"] - wide[f"mean_forecast_{arm}"]).abs()
                - (wide["actual"] - wide["mean_forecast_K0"]).abs()
            )
            .to_numpy()
            .astype(float)
        )
        mae = boot(num=mean_err, groups=wide["month"].to_numpy())
        print(
            f"{arm:12s} pooled {fmt(all_rows)}  winter {fmt(winter)}  other {fmt(other)}  mean-forecast |error| {fmt(mae)}"
        )
        report[f"G2|{arm}"] = {
            "pooled": all_rows,
            "winter": winter,
            "other": other,
            "mean_abs_error": mae,
        }
    print("K0 absolute:", round(float(wide["qs_K0"].mean()), 3))  # ty: ignore[invalid-argument-type]
    for arm in LEARNED:
        for fold in range(3):
            f = (
                sc.filter((pl.col("arm") == arm) & (pl.col("fold") == fold))
                .join(
                    sc.filter(pl.col("arm") == "K0").select("site", "row", k0=pl.col("qs")),
                    on=["site", "row"],
                )
                .join(ordinary.select("site", "row", "month"), on=["site", "row"])
            )
            t = boot(num=(f["qs"] - f["k0"]).to_numpy().astype(float), groups=f["month"].to_numpy())
            print(f"  {arm:12s} fold {fold}: {fmt(t)}")
            report[f"G2|{arm}|fold{fold}"] = t
    print("\n-- leave one substation out (pooled, arm minus K0) --")
    for arm in ("PAIR_GATED", "LEARNED_R", "LEARNED_F", "RONC_195"):
        out = {}
        for s in SITES:
            out[s] = month_contrast(wide.filter(pl.col("site") != s), arm)
        print(arm, {s: round(v[0], 3) for s, v in out.items()})
        report[f"G2_loso|{arm}"] = out
    print("\n-- median minus mean forecast, per site (ordinary rows, points) --")
    for arm in ("K0", "LEARNED_R"):
        d = (
            wide.group_by("site")
            .agg((pl.col(f"q50_{arm}") - pl.col(f"mean_forecast_{arm}")).mean().alias(arm))
            .sort("site")
        )
        print(d.to_dicts())

    # ---- masked half-hours ----
    print(
        "\n== alarms and masked half-hours (all folds summed per arm; per fold in report.json) =="
    )
    sp = (
        spans.group_by("arm", "fold")
        .agg(
            alarms=pl.len(),
            half_hours=pl.col("n_half_hours").sum(),
            outside_logged=pl.col("n_outside_logged").sum(),
        )
        .with_columns(share_outside=pl.col("outside_logged") / pl.col("half_hours"))
        .sort("arm", "fold")
    )
    print(sp)
    report["spans"] = sp.to_dicts()

    # ---- real transfers: AFTER rows ----
    ev = after.join(meta, on=["site", "row"], how="inner")
    long = []
    for arm in (*FOLDLESS, *LEARNED):
        s = scores.filter(pl.col("arm") == arm)
        if arm in FOLDLESS:
            j = ev.join(s.drop("fold"), on=["site", "row"], how="left")
        else:
            j = ev.join(
                s, left_on=["site", "row", "fold"], right_on=["site", "row", "fold"], how="left"
            )
        long.append(
            j.select(
                "site",
                "card",
                "row",
                "fold",
                "power_fcst_init_time",
                "valid_time",
                "run_period",
                "actual",
                pl.lit(arm).alias("arm"),
                "qs",
                "q50",
                "mean_forecast",
            )
        )
    long = pl.concat(long)
    evw = long.pivot(
        on="arm",
        index=["site", "card", "row", "power_fcst_init_time", "valid_time", "run_period", "actual"],
        values=["qs", "mean_forecast"],
    )
    ext = bg2().select(*KEYS, bg2_qs=pl.col("qs"))
    evw = evw.with_columns(
        pl.col("power_fcst_init_time").dt.replace_time_zone("UTC"),
        pl.col("valid_time").dt.replace_time_zone("UTC"),
    ).join(ext, on=KEYS, how="left")
    evw = evw.with_columns(
        shared=pl.col("bg2_qs").is_not_null(),
        in163=pl.col("run_period").is_in(["scored", "replicate"]),
        mo=pl.col("valid_time").dt.month(),
    )
    scopes = {
        "shared": evw.filter(pl.col("in163") & pl.col("shared")),
        "stack_only": evw.filter(pl.col("in163")),
        "all_runs": evw,
    }
    arms_all = [*FOLDLESS, *LEARNED]
    print("\n== AFTER rows: mean score by scope (points of effective capacity) ==")
    for name, f in scopes.items():
        means = {a: round(float(f[f"qs_{a}"].mean()), 3) for a in arms_all}  # ty: ignore[invalid-argument-type]
        if name == "shared":
            means["BG2"] = round(float(f["bg2_qs"].mean()), 3)  # ty: ignore[invalid-argument-type]
        print(f"{name:10s} rows {f.height:5d} cards {f['card'].n_unique():2d} {means}")
        report[f"means|{name}"] = {"rows": f.height, "cards": f["card"].n_unique(), **means}
    print("\nPositive controls (shared scope; printed to 3 decimals):")
    shared = scopes["shared"]
    got = {
        "K0": float(shared["qs_K0"].mean()),  # ty: ignore[invalid-argument-type]
        "PAIR_NEW": float(shared["qs_PAIR_NEW"].mean()),  # ty: ignore[invalid-argument-type]
        "BG2": float(shared["bg2_qs"].mean()),  # ty: ignore[invalid-argument-type]
    }
    for k, v in CONTROLS.items():
        print(
            f"  {k}: {got[k]:.3f} (163 printed {v:.3f})  {'OK' if abs(got[k] - v) < 0.002 else 'MISMATCH'}"
        )
    report["controls"] = {k: [got[k], v] for k, v in CONTROLS.items()}

    print("\n== Gain contrasts on AFTER rows (arm minus PAIR_NEW; resampling whole transfers) ==")
    for name, f in scopes.items():
        for arm in LEARNED:
            for ref in ("PAIR_NEW", "K0"):
                d = (f[f"qs_{arm}"] - f[f"qs_{ref}"]).to_numpy().astype(float)
                t = boot(num=d, groups=f["card"].to_numpy())
                print(f"{name:10s} {arm:12s} minus {ref:9s} {fmt(t)}")
                report[f"G1|{name}|{arm}|{ref}"] = t
        for arm in ("PAIR_NEW",):
            d = (f[f"qs_{arm}"] - f["qs_K0"]).to_numpy().astype(float)
            print(
                f"{name:10s} {arm:12s} minus K0        {fmt(boot(num=d, groups=f['card'].to_numpy()))}"
            )
        w = f.filter(pl.col("mo").is_in(list(WINTER)))
        print(
            f"{name:10s} winter rows {w.height} cards {w['card'].n_unique()}  LEARNED_R minus PAIR_NEW {fmt(boot(num=(w['qs_LEARNED_R'] - w['qs_PAIR_NEW']).to_numpy().astype(float), groups=w['card'].to_numpy())) if w.height else 'n/a'}"
        )
        d = (
            (
                (f["actual"] - f["mean_forecast_LEARNED_R"]).abs()
                - (f["actual"] - f["mean_forecast_PAIR_NEW"]).abs()
            )
            .to_numpy()
            .astype(float)
        )
        print(
            f"{name:10s} mean-forecast |error| LEARNED_R minus PAIR_NEW {fmt(boot(num=d, groups=f['card'].to_numpy()))}"
        )
    print("\n-- leave one substation out, stack_only, LEARNED_R minus PAIR_NEW --")
    f = scopes["stack_only"]
    loso = {}
    for s in SITES:
        g = f.filter(pl.col("site") != s)
        if g.height:
            loso[s] = boot(
                num=(g["qs_LEARNED_R"] - g["qs_PAIR_NEW"]).to_numpy().astype(float),
                groups=g["card"].to_numpy(),
            )
    print({s: round(v[0], 3) for s, v in loso.items()})
    report["G1_loso"] = loso

    print("\n== per transfer (stack_only AFTER): mean score ==")
    per = (
        scopes["stack_only"]
        .group_by("site", "card", "run_period")
        .agg(
            rows=pl.len(),
            **{a: pl.col(f"qs_{a}").mean().round(2) for a in arms_all},
            BG2=pl.col("bg2_qs").mean().round(2),
        )
        .sort("run_period", "site", "card")
    )
    print(per)
    per.write_parquet(WORK_DIR / "per_transfer.parquet")

    print("\n== tail calibration on stack_only AFTER rows (coverage and pinball, points) ==")
    tl = tails.join(
        after.select("site", "row", "fold", "card"), on=["site", "row"], how="inner", suffix="_ev"
    )
    tl = tl.filter((pl.col("fold") == -1) | (pl.col("fold") == pl.col("fold_ev")))
    tl = tl.join(
        meta.select("site", "row", "power_fcst_init_time", "actual", "run_period"),
        on=["site", "row"],
    )
    tl = tl.filter(pl.col("run_period").is_in(["scored", "replicate"]))
    for arm in ("K0", "PAIR_NEW", "LEARNED_R", "LEARNED_F", "RONC_195"):
        t = tail_calibration_table(frame=tl.filter(pl.col("arm") == arm))
        print(arm)
        print(t.select("level", "n_rows", "coverage", "gap", "pinball"))
        report[f"tails|{arm}"] = t.to_dicts()

    (WORK_DIR / "report.json").write_text(json.dumps(report, default=str, indent=1))


if __name__ == "__main__":
    main()
