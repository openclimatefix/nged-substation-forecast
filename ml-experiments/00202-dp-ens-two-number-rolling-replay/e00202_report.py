# ruff: noqa: E501
"""Score the rolling replay against arm A, arm M, and experiment 193's fixed-factor Q2.

Reads the arms written by `e00202_replay.py`, scores them with experiment 193's machinery (copied
into `e00202_score.py`), and writes `work/report/report.json` and prints every number the page
quotes.

    uv run --no-sync python ml-experiments/00202-*/e00202_report.py
"""

import json
from typing import Final

import e00202_common as common
import e00202_data as data
import e00202_score as score
import numpy as np
import polars as pl

ARM_NAMES: Final[tuple[str, ...]] = ("a", "m", "q2", "r2", "r2f64", "r2o", "rr")
REPORT_DIR = common.WORK_DIR / "report"
HOUR_GROUPS: Final[dict[str, tuple[int, int]]] = {
    "valid hour 06 to 09 UTC": (6, 9),
    "valid hour 09 to 12 UTC": (9, 12),
    "valid hour 12 to 15 UTC": (12, 15),
    "valid hour 15 to 18 UTC": (15, 18),
    "valid hour 18 to 21 UTC": (18, 21),
}


def load_arms() -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return the scored rows and every arm's quantiles."""
    frame = pl.read_parquet(common.WORK_DIR / "arms" / "scored_frame.parquet")
    arms = {
        name: np.load(score.E184_ARM_DIR / f"scored_{name}.npy").astype(np.float64)
        for name in ("a", "m")
    }
    arms["q2"] = np.load(common.E193_ARM_DIR / "scored_q2.npy").astype(np.float64)
    for name in ("r2", "r2f64", "r2o", "rr"):
        arms[name] = np.load(common.WORK_DIR / "arms" / f"scored_{name}.npy").astype(np.float64)
    return frame, arms


def contrast(
    *, month: np.ndarray, metrics: dict, mask: np.ndarray, arm: str, reference: str = "a"
) -> dict:
    """Return an arm's percentage change from a reference arm over a mask, with its interval."""
    return score.percent_change(
        month=month[mask],
        arm=metrics[arm]["qs"][mask],
        reference=metrics[reference]["qs"][mask],
        level=score.LEVEL,
        seed=202,
    )


def tails(*, frame: pl.DataFrame, arms: dict, metrics: dict, mask: np.ndarray) -> dict:
    """Return tail coverage and pinball loss over a mask for arms A, Q2, and R2."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    out = {}
    for arm in ("a", "q2", "r2"):
        q, rm = arms[arm], metrics[arm]
        out[arm] = {
            "cover01": float((actual <= q[:, 0])[mask].mean()),
            "cover05": float(rm["below05"][mask].mean()),
            "cover95": float(rm["below95"][mask].mean()),
            "cover99": float((actual <= q[:, 10])[mask].mean()),
            "pin01": float(rm["pin01"][mask].mean()),
            "pin05": float(rm["pin05"][mask].mean()),
            "pin95": float(rm["pin95"][mask].mean()),
            "pin99": float(rm["pin99"][mask].mean()),
        }
    return out


def main() -> None:
    """Write `report.json`."""
    frame, arms = load_arms()
    labelled = frame  # already labelled by e00202_arms.labelled
    metrics = {name: score.row_metrics(frame=labelled, quantiles=q) for name, q in arms.items()}
    month = labelled["month"].to_numpy()
    masks = score.subsets(frame=labelled)
    day = masks["daylight"]
    hours = labelled["valid_time"].dt.hour().to_numpy()
    first_half = np.isin(month, sorted(set(month))[:6])
    extra = {
        "daylight, first six months": day & first_half,
        "daylight, last months": day & ~first_half,
        **{
            f"daylight, {k}": day & (hours >= lo) & (hours < hi)
            for k, (lo, hi) in HOUR_GROUPS.items()
        },
    }
    masks = {**masks, **extra}
    report: dict = {"n_rows": labelled.height, "months": sorted(set(month.tolist())), "subsets": {}}
    for name, mask in masks.items():
        if mask.sum() == 0:
            continue
        entry: dict = {"rows": int(mask.sum())}
        for arm in ARM_NAMES:
            m = {k: float(v[mask].mean()) for k, v in metrics[arm].items()}
            if arm != "a":
                m["change_pct"] = contrast(month=month, metrics=metrics, mask=mask, arm=arm)
            if arm in ("r2", "r2f64", "r2o", "rr"):
                m["change_vs_q2_pct"] = contrast(
                    month=month, metrics=metrics, mask=mask, arm=arm, reference="q2"
                )
            if arm in ("q2", "r2", "r2o", "rr"):
                m["share_of_m"] = score.retained_share(
                    month=month[mask],
                    arm=metrics[arm]["qs"][mask],
                    member=metrics["m"]["qs"][mask],
                    reference=metrics["a"]["qs"][mask],
                    seed=202,
                )
            entry[arm] = m
        report["subsets"][name] = entry
    # Per-month daylight change of R2 (shows the early months with less history).
    report["monthly_daylight_change_pct"] = {
        m: 100.0
        * float((metrics["r2"]["qs"] - metrics["a"]["qs"])[day & (month == m)].sum())
        / float(metrics["a"]["qs"][day & (month == m)].sum())
        for m in report["months"]
        if (day & (month == m)).sum() > 0
    }
    # Gates and tails.
    r2, r2f64, rr, q2 = (report["subsets"]["daylight"][a] for a in ("r2", "r2f64", "rr", "q2"))
    gates = json.loads((common.WORK_DIR / "gates_pre.json").read_text())
    gates["g2_float32_vs_float64_qs_difference"] = r2["qs"] - r2f64["qs"]
    gates["g4_placebo_share_of_r2_gain"] = rr["change_pct"]["point"] / r2["change_pct"]["point"]
    gates["g1_q2_daylight_change_pct"] = q2["change_pct"]["point"]
    report["gates"] = gates
    report["tails"] = tails(frame=labelled, arms=arms, metrics=metrics, mask=day)
    lead = labelled["lead_hours"].to_numpy()
    sel = (lead >= 12.0) & (lead < 36.0) & data.usable_rows(frame=labelled)
    mean = labelled["mean"].to_numpy().astype(np.float64)
    site = labelled["site"].to_numpy()
    report["median_minus_mean_12_36h"] = {
        arm: {
            s: float((arms[arm][:, 5] - mean)[sel & (site == s)].mean())
            for s in common.SCORED_SITES
        }
        for arm in ("a", "q2", "r2")
    }
    # Leave-one-site-out for the decisive contrast.
    report["leave_one_out_r2_daylight_pct"] = {
        s: contrast(month=month, metrics=metrics, mask=day & (site != s), arm="r2")
        for s in common.DECISION_SITES
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.json").write_text(json.dumps(report, indent=1, default=float))
    for name in ("daylight", "night", "twilight", "all rows"):
        e = report["subsets"][name]
        print(name, e["rows"], "A qs", round(e["a"]["qs"], 4))
        for arm in ARM_NAMES[1:]:
            c = e[arm]["change_pct"]
            print(
                f"   {arm}: {c['point']:+.2f}% [{c['lower']:+.2f}, {c['upper']:+.2f}]  qs {e[arm]['qs']:.4f}"
            )
    print(json.dumps(report["gates"], indent=1))
    print(json.dumps(report["monthly_daylight_change_pct"], indent=1))


if __name__ == "__main__":
    main()
