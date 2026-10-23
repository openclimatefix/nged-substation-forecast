"""Score experiment 305's refits, attribute BG2's twin by input family, and write report.json.

Reads the predictions and boosters that `e00305_fit.py` wrote and experiment 266's master frame, and
prints every number the write-up quotes into `work/report.json`.
"""

import json
from pathlib import Path
from typing import Final

import e00305_fit as fit
import numpy as np
import polars as pl
import xgboost as xgb

DATA_DIR: Final[Path] = fit.DATA_DIR
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
QUANTILES: Final[tuple[str, ...]] = tuple(f"q{round(v * 100):02d}" for v in LEVELS)
FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    "solar": fit.SOLAR,
    "temperature": fit.TEMPERATURE,
    "calendar": (
        "hour_local",
        "day_of_week_local",
        "covid_indicator",
        "days_to_christmas",
        "lead_hours",
    ),
    "lags": fit.LAGS,
    "other_weather": ("longwave", "pressure_surface", "precipitation", "wind_speed_10m"),
}
TARGETS: Final[tuple[str, ...]] = ("D5", "D9")
PLACEBOS: Final[tuple[str, ...]] = ("D10", "D16")
N_RESAMPLES: Final[int] = 20_000
SHAP_ROWS: Final[int] = 3_000


def qs_per_row(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 quantiles, per row."""
    error = actual.astype(np.float64)[:, None] - q.astype(np.float64)
    pin = np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error)
    return 2.0 * pin.mean(axis=1)


def ratio_interval(*, num: np.ndarray, den: np.ndarray, blocks: np.ndarray, seed: int = 0) -> dict:
    """Return sum(num) / sum(den) with a 95% interval that resamples the block labels."""
    labels, codes = np.unique(blocks, return_inverse=True)
    n = len(labels)
    nums = np.bincount(codes, weights=num, minlength=n)
    dens = np.bincount(codes, weights=den, minlength=n)
    draws = np.random.default_rng(seed).integers(0, n, size=(N_RESAMPLES, n))
    res = nums[draws].sum(axis=1) / dens[draws].sum(axis=1)
    lo, hi = np.percentile(res, [2.5, 97.5])
    return {
        "est": float(nums.sum() / dens.sum()),
        "lo": float(lo),
        "hi": float(hi),
        "blocks": int(n),
    }


def load() -> pl.DataFrame:
    """Return the April-to-June master rows of the five sites with every variant's quantiles."""
    master = pl.read_parquet(fit.MASTER).filter(
        pl.col("site").is_in(fit.SITES)
        & pl.col("init_month").is_in(["2026-04", "2026-05", "2026-06"])
    )
    keys = ["site", "power_fcst_init_time", "valid_time"]
    for variant in fit.VARIANTS:
        files = sorted((DATA_DIR / "work" / "preds" / variant).glob("*.parquet"))
        if variant == fit.ADDITIVE:
            files = [p for p in files if not p.name.startswith(("D14", "D16"))]
        table = pl.concat([pl.read_parquet(p) for p in files])
        table = table.rename({q: f"{variant}_{q}" for q in QUANTILES})
        master = master.join(table, on=keys, how="left", validate="1:1")
    return master.with_columns(
        week=pl.col("power_fcst_init_time").dt.iso_year() * 100
        + pl.col("power_fcst_init_time").dt.week()
    )


def variant_qs(*, rows: pl.DataFrame, variant: str) -> np.ndarray:
    """Return a variant's quantile score on each row."""
    return qs_per_row(
        actual=rows["actual"].to_numpy(),
        q=rows.select([f"{variant}_{q}" for q in QUANTILES]).to_numpy(),
    )


def selected(*, rows: pl.DataFrame, site: str, share: float = 0.25) -> pl.DataFrame:
    """Return the share of a site's daylight rows where arm D trails stored BG2 by the most."""
    day = rows.filter((pl.col("site") == site) & pl.col("daylight")).with_columns(
        gap=pl.col("qs_d") - pl.col("qs_bg2")
    )
    cut = day["gap"].quantile(1.0 - share)
    return day.filter(pl.col("gap") >= cut)


def share_of_advantage(*, rows: pl.DataFrame, constrained: str) -> dict:
    """Return S = (QS_constrained - QS_full) / (QS_D - QS_full) with a week-block interval."""
    full = variant_qs(rows=rows, variant="full")
    cons = variant_qs(rows=rows, variant=constrained)
    out = ratio_interval(
        num=cons - full, den=rows["qs_d"].to_numpy() - full, blocks=rows["week"].to_numpy()
    )
    out["qs_d"] = float(np.mean(rows["qs_d"].to_numpy()))
    out["qs_full"] = float(full.mean())
    out["qs_constrained"] = float(cons.mean())
    out["qs_bg2_stored"] = float(np.mean(rows["qs_bg2"].to_numpy()))
    out["rows"] = rows.height
    return out


def attribution(*, rows: pl.DataFrame) -> dict:
    """Return Shapley, shuffling, and Shapley-interaction summaries on the selected rows."""
    inputs = pl.concat(
        [pl.read_parquet(p) for p in sorted((DATA_DIR / "work" / "inputs").glob("*.parquet"))]
    )
    keys = ["site", "power_fcst_init_time", "valid_time"]
    joined = rows.join(inputs, on=keys, how="inner", validate="1:1")
    columns = fit.fit.input_columns(a7=False)
    rng = np.random.default_rng(0)
    families = list(FAMILIES)
    shap_sum = dict.fromkeys(families, 0.0)
    inter_sum = {(a, b): 0.0 for i, a in enumerate(families) for b in families[i + 1 :]}
    perm = dict.fromkeys(families, 0.0)
    perm_pair = dict.fromkeys(inter_sum, 0.0)
    corr_rows: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    total = 0
    for (site, month), group in joined.group_by(
        ["site", joined["init_month"]], maintain_order=True
    ):
        booster = xgb.Booster()
        booster.load_model(str(DATA_DIR / "work" / "models" / "full" / f"{site}_{month}.ubj"))
        x = group.select(columns).to_numpy().astype(np.float32)
        n = x.shape[0]
        sub = rng.choice(n, size=min(n, SHAP_ROWS), replace=False)
        contribs = booster.predict(xgb.DMatrix(x[sub]), pred_contribs=True)[:, 5, :-1]
        fam_shap = {
            f: contribs[:, [columns.index(c) for c in FAMILIES[f]]].sum(axis=1) for f in families
        }
        for f in families:
            shap_sum[f] += float(np.abs(fam_shap[f]).sum())
        inter = booster.predict(xgb.DMatrix(x[sub]), pred_interactions=True)[:, 5, :-1, :-1]
        for a, b in inter_sum:
            ia = [columns.index(c) for c in FAMILIES[a]]
            ib = [columns.index(c) for c in FAMILIES[b]]
            inter_sum[(a, b)] += float(np.abs(2.0 * inter[:, ia][:, :, ib].sum(axis=(1, 2))).sum())
        corr_rows.append(((group["med_bg2"] - group["med_d"]).to_numpy()[sub], fam_shap))
        total += len(sub)
        actual = group["actual"].to_numpy()
        base_score = qs_per_row(
            actual=actual, q=np.sort(booster.predict(xgb.DMatrix(x)), axis=1) * fit.SCALE
        ).sum()

        def permuted(
            names: tuple[str, ...],
            x: np.ndarray = x,
            actual: np.ndarray = actual,
            booster: xgb.Booster = booster,
            base_score: float = base_score,
        ) -> float:
            z = x.copy()
            order = rng.permutation(x.shape[0])
            for c in names:
                z[:, columns.index(c)] = x[order, columns.index(c)]
            return float(
                qs_per_row(
                    actual=actual, q=np.sort(booster.predict(xgb.DMatrix(z)), axis=1) * fit.SCALE
                ).sum()
                - base_score
            )

        single = {f: np.mean([permuted(FAMILIES[f]) for _ in range(3)]) for f in families}
        for f in families:
            perm[f] += single[f]
        for a, b in inter_sum:
            joint = np.mean([permuted((*FAMILIES[a], *FAMILIES[b])) for _ in range(3)])
            perm_pair[(a, b)] += joint - single[a] - single[b]
    count = joined.height
    delta = np.concatenate([d for d, _ in corr_rows])
    correlation = {
        f: float(np.corrcoef(delta, np.concatenate([s[f] for _, s in corr_rows]))[0, 1])
        for f in families
    }
    return {
        "rows": count,
        "shap_rows": total,
        "mean_abs_shap_median_output": {f: v / total for f, v in shap_sum.items()},
        "mean_abs_shap_interaction_median_output": {
            f"{a}x{b}": v / total for (a, b), v in inter_sum.items()
        },
        "permutation_qs_increase_per_row": {f: v / count for f, v in perm.items()},
        "permutation_interaction_qs_per_row": {
            f"{a}x{b}": v / count for (a, b), v in perm_pair.items()
        },
        "corr_bg2_minus_d_with_family_shap": correlation,
    }


def main() -> None:
    """Write `work/report.json`."""
    rows = load()
    report: dict = {"gate": {}, "S": {}, "attribution": {}}
    for site in (*TARGETS, *PLACEBOS, "D14"):
        amj = rows.filter((pl.col("site") == site) & pl.col("daylight"))
        report["gate"][site] = {
            "mae_full_median": float(np.mean((amj["actual"] - amj["full_q50"]).abs().to_numpy())),
            "mae_bg2_stored": float(np.mean((amj["actual"] - amj["med_bg2"]).abs().to_numpy())),
            "mae_d": float(np.mean((amj["actual"] - amj["med_d"]).abs().to_numpy())),
            "qs_full_all_daylight": float(variant_qs(rows=amj, variant="full").mean()),
            "qs_bg2_all_daylight": float(np.mean(amj["qs_bg2"].to_numpy())),
            "qs_d_all_daylight": float(np.mean(amj["qs_d"].to_numpy())),
        }
    pooled = pl.concat([selected(rows=rows, site=s) for s in TARGETS])
    groups = {"D5+D9": pooled, **{s: selected(rows=rows, site=s) for s in TARGETS}}
    groups.update({f"{s} (placebo)": selected(rows=rows, site=s) for s in PLACEBOS})
    # Sensitivity: the selection uses stored BG2's realised loss, so also score every daylight row.
    groups["D5+D9 (all daylight rows)"] = rows.filter(
        pl.col("site").is_in(TARGETS) & pl.col("daylight")
    )
    for name, frame in groups.items():
        names = [
            v for v in list(fit.VARIANTS)[1:] if v != fit.ADDITIVE or not name.startswith("D16")
        ]
        report["S"][name] = {v: share_of_advantage(rows=frame, constrained=v) for v in names}
        report["S"][name]["median_abs_error"] = {
            "d": float(np.mean((frame["actual"] - frame["med_d"]).abs().to_numpy())),
            "bg2_stored": float(np.mean((frame["actual"] - frame["med_bg2"]).abs().to_numpy())),
            **{
                v: float(np.mean((frame["actual"] - frame[f"{v}_q50"]).abs().to_numpy()))
                for v in fit.VARIANTS
                if v != fit.ADDITIVE or not name.startswith("D16")
            },
        }
    report["attribution"] = attribution(rows=pooled)
    out = DATA_DIR / "work" / "report.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
