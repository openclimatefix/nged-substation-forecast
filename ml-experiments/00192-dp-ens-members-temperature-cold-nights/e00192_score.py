"""Score every arm on daylight, night, and the cuts, with month-resampled intervals.

The score is the quantile score: twice the mean pinball loss over the 11 quantiles, in points of
effective capacity. A contrast is the arm's total minus arm A's total over a subset, as a
percentage of arm A's total, with the months of the period resampled as whole blocks (paired).

    uv run --no-sync python ml-experiments/00192-*/e00192_score.py
"""

import json
from typing import Final

import e00192_common as common
import e00192_data as data
import e00192_fit as fit
import numpy as np
import polars as pl
from studies.dp_ens_weather import lead_band_of_hours

N_DRAWS: Final[int] = 20_000
HEADLINE_LEVEL: Final[float] = 97.5
"""Two planned contrasts, so each is read at 100 - 5 / 2 per cent."""
CSI_EDGES: Final[tuple[float, ...]] = (0.4, 0.7)
CSI_LABELS: Final[tuple[str, ...]] = (
    "overcast (under 0.4)",
    "partly cloudy (0.4 to 0.7)",
    "clear (0.7 and over)",
)
LEAD_CUTS: Final[dict[str, tuple[float, float]]] = {
    "under 24 h": (0.0, 24.0),
    "24 to 72 h": (24.0, 72.0),
    "72 to 168 h": (72.0, 168.0),
    "12 to 36 h": (12.0, 36.0),
}
DECISION: Final[tuple[str, ...]] = common.DECISION_SITES
SCORE_DIR = common.WORK_DIR / "report"


def load(*, period: str) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return a period's labelled rows and every arm's quantiles."""
    frame = pl.read_parquet(fit.ARM_DIR / f"{period}_frame.parquet")
    arms = {
        arm: np.load(fit.ARM_DIR / f"{period}_{arm}.npy").astype(np.float64) for arm in fit.ARMS
    }
    return frame, arms


def row_metrics(*, frame: pl.DataFrame, quantiles: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-row score columns for one arm."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    error = actual[:, None] - quantiles
    pinball = np.maximum(data.LEVELS * error, (data.LEVELS - 1.0) * error)
    q = {
        name: quantiles[:, data.QUANTILES.index(name)]
        for name in ("q01", "q05", "q50", "q95", "q99")
    }
    return {
        "qs": 2.0 * pinball.mean(axis=1),
        "pin01": pinball[:, 0],
        "pin05": pinball[:, 1],
        "pin95": pinball[:, 9],
        "pin99": pinball[:, 10],
        "below05": (actual <= q["q05"]).astype(np.float64),
        "below95": (actual <= q["q95"]).astype(np.float64),
        "inside90": ((actual >= q["q05"]) & (actual <= q["q95"])).astype(np.float64),
        "width": q["q95"] - q["q05"],
        "abs_median": np.abs(actual - q["q50"]),
    }


def month_blocks(
    *, month: np.ndarray, values: np.ndarray, base: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the per-month sums of `values`, of `base`, and the per-month row counts."""
    labels, inverse = np.unique(month, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(labels))
    base_sums = np.bincount(inverse, weights=base, minlength=len(labels))
    counts = np.bincount(inverse, minlength=len(labels)).astype(np.float64)
    return sums, base_sums, counts


def percent_change(
    *, month: np.ndarray, arm: np.ndarray, reference: np.ndarray, level: float, seed: int
) -> dict:
    """Return the arm's percentage change from the reference with a paired month-block interval."""
    sums, base_sums, _ = month_blocks(month=month, values=arm - reference, base=reference)
    point = 100.0 * sums.sum() / base_sums.sum()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(N_DRAWS, len(sums)))
    draws = 100.0 * sums[picks].sum(axis=1) / base_sums[picks].sum(axis=1)
    tail = (100.0 - level) / 2.0
    low, high = np.percentile(draws, [tail, 100.0 - tail])
    return {"point": float(point), "lower": float(low), "upper": float(high)}


def label_columns(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the lead band, clear-sky-index band, and sunniest-third flag."""
    usable = pl.Series(data.usable_rows(frame=frame))
    csi = frame["csi_raw"].to_numpy()
    band = np.array(CSI_LABELS, dtype=object)[np.digitize(np.nan_to_num(csi), CSI_EDGES)]
    csi_band = pl.Series(
        [b if np.isfinite(c) else None for b, c in zip(band, csi, strict=True)], dtype=pl.String
    )
    edge = {}
    for (site,), group in frame.filter(usable & (pl.col("unit") > data.SUN_UP_UNIT_SHAPE)).group_by(
        "site"
    ):
        edge[site] = float(np.quantile(group["unit"].to_numpy(), 2 / 3))
    sunny = pl.col("unit") >= pl.col("site").replace_strict(edge, return_dtype=pl.Float64)
    return frame.with_columns(
        usable=usable,
        csi_band=csi_band,
        sunny=sunny & (pl.col("unit") > data.SUN_UP_UNIT_SHAPE),
        lead_band=pl.Series(lead_band_of_hours(lead_hours=frame["lead_hours"].to_numpy())),
    )


def subsets(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the named boolean row masks of every scored subset."""
    usable = frame["usable"].to_numpy()
    five = frame["site"].is_in(list(DECISION)).to_numpy() & usable
    kind = frame["kind"].to_numpy()
    lead = frame["lead_hours"].to_numpy()
    out: dict[str, np.ndarray] = {}
    for kind_name in ("daylight", "night", "twilight"):
        out[kind_name] = five & (kind == kind_name)
    out["all rows"] = five
    day = out["daylight"]
    for label, (low, high) in LEAD_CUTS.items():
        out[f"daylight, {label}"] = day & (lead >= low) & (lead < high)
    for label in CSI_LABELS:
        out[f"daylight, {label}"] = day & (frame["csi_band"].to_numpy() == label)
        out[f"daylight, {label}, under 72 h"] = out[f"daylight, {label}"] & (lead < 72.0)
    out["daylight, elevation 17 degrees and over"] = day & (frame["elevation"].to_numpy() >= 17.0)
    out["sunniest third (experiment 178)"] = five & frame["sunny"].to_numpy()
    out["sunniest third, under 72 h"] = out["sunniest third (experiment 178)"] & (lead < 72.0)
    out["daylight, D5"] = usable & day & (frame["site"].to_numpy() == "D5")
    out["daylight, four other substations"] = day & (frame["site"].to_numpy() != "D5")
    out["daylight, D9"] = usable & day & (frame["site"].to_numpy() == "D9")
    for site in DECISION:
        out[f"daylight, {site}"] = usable & day & (frame["site"].to_numpy() == site)
    return out


def score_period(*, period: str) -> dict:
    """Return a period's tables: per subset and arm, scores, contrasts, and calibration."""
    frame, arms = load(period=period)
    frame = label_columns(frame=frame)
    metrics = {arm: row_metrics(frame=frame, quantiles=q) for arm, q in arms.items()}
    month = frame["month"].to_numpy()
    out: dict = {"n_rows": frame.height, "subsets": {}}
    for name, mask in subsets(frame=frame).items():
        if mask.sum() == 0:
            continue
        entry: dict = {"rows": int(mask.sum())}
        for arm in fit.ARMS:
            m = {k: v[mask] for k, v in metrics[arm].items()}
            row: dict = {k: float(v.mean()) for k, v in m.items()}
            if arm != "a":
                level = HEADLINE_LEVEL if arm in ("m", "s") else 95.0
                row["change_pct"] = percent_change(
                    month=month[mask],
                    arm=m["qs"],
                    reference=metrics["a"]["qs"][mask],
                    level=level,
                    seed=184,
                )
            entry[arm] = row
        out["subsets"][name] = entry
    return out


def main() -> None:
    """Score every period and write `report.json`."""
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    report = {period: score_period(period=period) for period in ("tuning", "replicate", "scored")}
    (SCORE_DIR / "report.json").write_text(json.dumps(report, indent=1, default=float))
    for period in ("replicate", "scored"):
        print(period)
        for name in ("daylight", "night", "twilight", "all rows"):
            entry = report[period]["subsets"][name]
            line = [f"{name}: n={entry['rows']}, A qs={entry['a']['qs']:.4f}"]
            for arm in ("m", "mc", "f", "s", "sp", "mr", "mp"):
                c = entry[arm]["change_pct"]
                line.append(f"{arm} {c['point']:+.2f}% [{c['lower']:+.2f},{c['upper']:+.2f}]")
            print("  ", "; ".join(line))


if __name__ == "__main__":
    main()
