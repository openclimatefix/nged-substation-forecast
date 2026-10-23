# ruff: noqa: E501
"""Score arm D with ensemble, identity-calibrated ensemble, and realised sunshine against BG2.

Reads experiment 266's master frame (experiment 251's 561,207 rows with arm D's and BG2's
per-row scores), joins `work/swap/` (the base engine's mean under each sunshine input), shifts
every quantile of stored arm D by the change in the base engine's mean (the method of experiment
251's clip arm), and writes `work/report.json` and `work/rows.parquet`.

    uv run --no-sync python ml-experiments/00289-*/e00289_analyse.py
"""

import json
from typing import Any, Final

import e00289_common as common
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

E266: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00266-summer-gap-diagnostics-solar-term" / "work" / "master.parquet"
)
E251_ARMS: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2"
    / "work" / "arms" / "scored"
)  # fmt: skip
SITES_ALL: Final = ("D5", "D9", "D10", "D11", "D14", "D16")
DECISION: Final = ("D5", "D9", "D10", "D16")
AMJ: Final = ("2026-04", "2026-05", "2026-06")
JAS: Final = ("2025-07", "2025-08", "2025-09")
LEVELS: Final = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
N_RESAMPLES: Final = 20_000
LEAD_BANDS: Final = {"0-24 h": (0.0, 24.0), "24-72 h": (24.0, 72.0), "72-168 h": (72.0, 168.01)}


def qs(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Twice the mean pinball loss over the 11 quantiles, per row."""
    err = actual[:, None] - q
    return 2.0 * np.maximum(LEVELS * err, (LEVELS - 1.0) * err).mean(axis=1)


def site_week_mean(
    *,
    values: np.ndarray,
    mask: np.ndarray,
    site: np.ndarray,
    week: np.ndarray,
    sites: tuple[str, ...],
    level: float = 95.0,
    return_draws: bool = False,
) -> dict:
    """Mean of `values` over `mask`, averaged over `sites`, with a week-block interval.

    Each draw resamples whole ISO weeks of initialisation time (the same weeks at every
    substation) and averages the per-substation means, so no substation decides alone.
    """
    keep = mask & np.isfinite(values) & np.isin(site, sites)
    weeks = np.unique(week[keep])
    index = {w: i for i, w in enumerate(weeks)}
    codes = np.array([index[w] for w in week[keep]])
    sums = np.zeros((len(sites), len(weeks)))
    counts = np.zeros((len(sites), len(weeks)))
    for k, s in enumerate(sites):
        pick = site[keep] == s
        sums[k] = np.bincount(codes[pick], weights=values[keep][pick], minlength=len(weeks))
        counts[k] = np.bincount(codes[pick], minlength=len(weeks))
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(weeks), size=(N_RESAMPLES, len(weeks)))
    num = sums[:, draws].sum(axis=2)
    den = counts[:, draws].sum(axis=2)
    per_site = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    res = np.nanmean(per_site, axis=0)
    est_sites = np.divide(
        sums.sum(axis=1),
        counts.sum(axis=1),
        out=np.full(len(sites), np.nan),
        where=counts.sum(axis=1) > 0,
    )
    lo, hi = np.percentile(res, [(100 - level) / 2, 100 - (100 - level) / 2])
    out = {
        "est": float(np.nanmean(est_sites)),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(keep.sum()),
        "weeks": len(weeks),
    }
    if return_draws:
        out["draws"] = res
    return out


def load() -> pl.DataFrame:
    """Return the master frame with the swap means and the shifted arm D quantiles' scores."""
    master = pl.read_parquet(E266).with_columns(
        week=pl.col("power_fcst_init_time").dt.truncate("1w").dt.strftime("%Y-%m-%d")
    )
    swap = pl.concat(
        [
            pl.read_parquet(
                common.WORK_DIR / "swap" / f"{s}.parquet",
                columns=["site", "run", "valid_time", "mean_ens", "mean_ens_id", "mean_cams"],
            )
            for s in SITES_ALL
        ]
    )
    frame = master.join(swap, on=["site", "run", "valid_time"], how="left", validate="1:1")
    assert frame["mean_cams"].null_count() == 0, "swap rows missing"
    d = np.load(E251_ARMS / "d.npy").astype(np.float64)
    assert d.shape[0] == frame.height, "arm D and master rows differ"
    actual = frame["actual"].to_numpy().astype(np.float64)
    out = {}
    for name, column in (("ens", None), ("id", "mean_ens_id"), ("cams", "mean_cams")):
        shift = (
            np.zeros(frame.height)
            if column is None
            else (frame[column] - frame["mean_ens"]).to_numpy()
        )
        out[f"qs_d_{name}"] = qs(actual=actual, q=d + shift[:, None])
        out[f"med_d_{name}"] = d[:, 5] + shift
        out[f"shift_{name}"] = shift
    assert np.abs(out["qs_d_ens"] - frame["qs_d"].to_numpy()).max() < 1e-4, (
        "arm D rows are misaligned"
    )
    return frame.with_columns([pl.Series(k, v) for k, v in out.items()])


def contrasts(
    *,
    frame: pl.DataFrame,
    months: tuple[str, ...],
    sites: tuple[str, ...],
    lead: tuple[float, float],
    daylight: bool,
    level: float = 95.0,
) -> dict:
    """Return the gap and the sunshine contrasts on one subset."""
    month = frame["month"].is_in(list(months)).to_numpy()
    lead_ok = ((frame["lead_hours"] > lead[0]) & (frame["lead_hours"] <= lead[1])).to_numpy()
    mask = (
        month
        & lead_ok
        & (frame["daylight"].to_numpy() if daylight else np.ones(frame.height, bool))
    )
    site = frame["site"].to_numpy()
    week = frame["week"].to_numpy()
    actual = frame["actual"].to_numpy()
    bg2_mae = np.abs(actual - frame["med_bg2"].to_numpy())
    mae = {n: np.abs(actual - frame[f"med_d_{n}"].to_numpy()) for n in ("ens", "id", "cams")}
    qsd = {n: frame[f"qs_d_{n}"].to_numpy() for n in ("ens", "id", "cams")}
    qbg2 = frame["qs_bg2"].to_numpy()
    args: dict[str, Any] = {"mask": mask, "site": site, "week": week, "sites": sites}
    mean = lambda v, lv=95.0: site_week_mean(values=v, level=lv, **args)  # noqa: E731
    return {
        "score_bg2": mean(qbg2),
        "score_d_ens": mean(qsd["ens"]),
        "score_d_id": mean(qsd["id"]),
        "score_d_cams": mean(qsd["cams"]),
        "mae_bg2": mean(bg2_mae),
        "mae_d_ens": mean(mae["ens"]),
        "mae_d_id": mean(mae["id"]),
        "mae_d_cams": mean(mae["cams"]),
        "gap_qs": mean(qsd["ens"] - qbg2),
        "gap_mae": mean(mae["ens"] - bg2_mae),
        "gap_qs_cams": mean(qsd["cams"] - qbg2),
        "gap_mae_cams": mean(mae["cams"] - bg2_mae),
        "C1_qs_ens_minus_cams": mean(qsd["ens"] - qsd["cams"], level),
        "C2_mae_ens_minus_cams": mean(mae["ens"] - mae["cams"], level),
        "calibration_qs_ens_minus_id": mean(qsd["ens"] - qsd["id"]),
        "input_qs_id_minus_cams": mean(qsd["id"] - qsd["cams"]),
    }


def seasonal_excess(*, frame: pl.DataFrame) -> dict:
    """Post hoc: how much of the April to June gap beyond July to September's the swap removes."""
    out = {}
    for key, column in (("qs", "qs_d"), ("mae", None)):
        draws = {}
        for label, months in (("amj", AMJ), ("jas", JAS)):
            mask = (
                frame["month"].is_in(list(months))
                & frame["daylight"]
                & (frame["lead_hours"] <= 36.0)
            ).to_numpy()
            actual = frame["actual"].to_numpy()
            if column:
                gap_ens = frame["qs_d_ens"].to_numpy() - frame["qs_bg2"].to_numpy()
                gap_cams = frame["qs_d_cams"].to_numpy() - frame["qs_bg2"].to_numpy()
            else:
                bg2 = np.abs(actual - frame["med_bg2"].to_numpy())
                gap_ens = np.abs(actual - frame["med_d_ens"].to_numpy()) - bg2
                gap_cams = np.abs(actual - frame["med_d_cams"].to_numpy()) - bg2
            args: dict[str, Any] = {
                "mask": mask,
                "site": frame["site"].to_numpy(),
                "week": frame["week"].to_numpy(),
                "sites": DECISION,
                "return_draws": True,
            }
            draws[label] = (
                site_week_mean(values=gap_ens, **args),
                site_week_mean(values=gap_cams, **args),
            )
        # Excess = AMJ gap minus JAS gap; the swap's removal = excess before minus excess after.
        before = draws["amj"][0]["draws"] - draws["jas"][0]["draws"]
        after = draws["amj"][1]["draws"] - draws["jas"][1]["draws"]
        removed = before - after
        amj_pair, jas_pair = draws["amj"], draws["jas"]

        def point(i: int, a: tuple = amj_pair, j: tuple = jas_pair) -> float:
            return a[i]["est"] - j[i]["est"]

        lo, hi = np.percentile(removed, [2.5, 97.5])
        blo, bhi = np.percentile(before, [2.5, 97.5])
        alo, ahi = np.percentile(after, [2.5, 97.5])
        out[key] = {
            "excess_before": {"est": point(0), "lo": float(blo), "hi": float(bhi)},
            "excess_after_cams": {"est": point(1), "lo": float(alo), "hi": float(ahi)},
            "removed": {"est": point(0) - point(1), "lo": float(lo), "hi": float(hi)},
            "share_removed": (point(0) - point(1)) / point(0),
        }
    return out


def tails(*, frame: pl.DataFrame) -> dict:
    """Coverage and pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles, April to June."""
    mask = (
        frame["month"].is_in(list(AMJ))
        & frame["daylight"]
        & (frame["lead_hours"] <= 36.0)
        & frame["site"].is_in(list(DECISION))
    ).to_numpy()
    d = np.load(E251_ARMS / "d.npy").astype(np.float64)
    bg2 = np.load(E251_ARMS / "bg2.npy").astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    shift = frame["shift_cams"].to_numpy()
    arms = {"bg2": bg2, "d_ens": d, "d_cams": d + shift[:, None]}
    result = {}
    for name, q in arms.items():
        row = {}
        for level, col in ((0.01, 0), (0.05, 1), (0.95, 9), (0.99, 10)):
            err = actual[mask] - q[mask, col]
            row[str(level)] = {
                "coverage": float((err <= 0).mean()),
                "pinball": float(np.maximum(level * err, (level - 1.0) * err).mean()),
            }
        result[name] = row
    return result


def main() -> None:
    """Write the report."""
    frame = load()
    frame.drop("week").write_parquet(common.WORK_DIR / "rows.parquet")
    report: dict = {}
    # Gate: arm D minus BG2, April to June, all six substations, all leads, all half-hours (experiment 266: +0.319).
    report["gate_d_minus_bg2_amj_all"] = contrasts(
        frame=frame, months=AMJ, sites=SITES_ALL, lead=(0.0, 168.01), daylight=False
    )["gap_qs"]
    report["gate_mean_ens_equals_e136"] = (
        "see swap files (max abs difference 0.0 at all six substations)"
    )
    report["decision"] = contrasts(
        frame=frame, months=AMJ, sites=DECISION, lead=(0.0, 36.0), daylight=True, level=97.5
    )
    report["decision_95"] = contrasts(
        frame=frame, months=AMJ, sites=DECISION, lead=(0.0, 36.0), daylight=True
    )
    report["negative_control_jas"] = contrasts(
        frame=frame, months=JAS, sites=DECISION, lead=(0.0, 36.0), daylight=True, level=97.5
    )
    report["by_lead_band"] = {
        k: contrasts(frame=frame, months=AMJ, sites=DECISION, lead=v, daylight=True)
        for k, v in LEAD_BANDS.items()
    }
    report["by_site"] = {
        s: contrasts(frame=frame, months=AMJ, sites=(s,), lead=(0.0, 36.0), daylight=True)
        for s in SITES_ALL
    }
    report["by_site_all_leads"] = {
        s: contrasts(frame=frame, months=AMJ, sites=(s,), lead=(0.0, 168.01), daylight=True)
        for s in SITES_ALL
    }
    report["amj_night"] = contrasts(
        frame=frame, months=AMJ, sites=DECISION, lead=(0.0, 36.0), daylight=False
    )
    # Mean minus median per site for arm D (checks).
    report["checks_median_minus_mean_d"] = {
        s: float(
            (
                frame.filter(pl.col("site") == s)["med_d_ens"]
                - frame.filter(pl.col("site") == s)["mean_d"]
            ).mean()  # ty: ignore[invalid-argument-type]
        )
        for s in SITES_ALL
    }
    report["mean_shift_cams_amj_daylight"] = {
        s: float(
            frame.filter(
                (pl.col("site") == s) & pl.col("month").is_in(list(AMJ)) & pl.col("daylight")
            )["shift_cams"].mean()  # ty: ignore[invalid-argument-type]
        )
        for s in SITES_ALL
    }
    report["post_hoc_seasonal"] = seasonal_excess(frame=frame)
    report["tails_amj_decision_0_36"] = tails(frame=frame)
    common.write_json_atomic(path=common.WORK_DIR / "report.json", content=report)

    def show(name: str, c: dict) -> None:
        print(name)
        for k, v in c.items():
            print(
                f"  {k:30s} {v['est']:+.3f} ({v['lo']:+.3f} to {v['hi']:+.3f}) n={v['n']} weeks={v['weeks']}"
            )

    print("gate", report["gate_d_minus_bg2_amj_all"])
    print(json.dumps(report["post_hoc_seasonal"], indent=1))
    print(json.dumps(report["tails_amj_decision_0_36"], indent=1))
    print(report["checks_median_minus_mean_d"], report["mean_shift_cams_amj_daylight"])
    show("DECISION 97.5%", report["decision"])
    show("NEGATIVE CONTROL JAS 97.5%", report["negative_control_jas"])
    for k, v in report["by_lead_band"].items():
        show(
            f"lead {k}",
            {
                x: v[x]
                for x in ("gap_qs", "C1_qs_ens_minus_cams", "gap_mae", "C2_mae_ens_minus_cams")
            },
        )
    for k, v in report["by_site"].items():
        show(
            f"site {k} 0-36h",
            {
                x: v[x]
                for x in ("gap_qs", "C1_qs_ens_minus_cams", "gap_mae", "C2_mae_ens_minus_cams")
            },
        )


if __name__ == "__main__":
    main()
