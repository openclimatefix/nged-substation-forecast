"""Score the arms of experiment 00085 and apply the pre-registered rank-agreement rule.

Prints, and writes to `work/results/`:

- `absolute`: each arm's mean quantile score (points of capacity) with a 95% interval, its mean
  absolute error of the median, the 0.90 central-interval coverage, and the pinball losses at the
  0.05 and 0.95 quantiles, on the cases every arm shares.
- `contrasts`: the reference contrasts R1 (full monthly minus full weekly) and R2 (zero start minus
  estimated start at full weekly), and the schedule contrasts K1 and K2, each with a 95% interval
  from 20,000 resamples of whole origin months, paired on series, run, and valid time.
- `noise`: the noise floor N and the tolerance T, from the repeat fits.
- `agreement`: for each schedule and contrast, whether it agrees with the reference under the rule.
- `cost`: the mean fit seconds of one refit of each configuration, from the fit logs, and the cost
  of a year of 16 series under each schedule.

    uv run python ml-experiments/00085-*/e00085_stats.py
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00085_common as common
import numpy as np
import polars as pl

from studies import bootstrap

N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 95.0
NOISE_MINIMUM: Final[float] = 0.01
TOLERANCE_MINIMUM: Final[float] = 0.05
REFITS_PER_YEAR: Final[dict[str, int]] = {"weekly": 52, "monthly": 12, "quarterly": 4}
N_SERIES: Final[int] = 16
LOG_DIR: Final[Path] = common.WORK_DIR.parent / "logs"


@dataclass(frozen=True)
class Contrast:
    """A paired difference of two arms: treatment minus reference.

    Attributes:
        label: The contrast's label.
        schedule: The schedule it describes, or `reference` for R1 and R2.
        kind: `K1` (cadence: monthly minus weekly) or `K2` (zero start minus estimated start).
        treatment: The treatment arm.
        reference: The reference arm.
        compare_to: The reference contrast the schedule's must agree with, or `None`.
        scored: Whether the contrast enters the pass rule.
    """

    label: str
    schedule: str
    kind: str
    treatment: str
    reference: str
    compare_to: str | None = None
    scored: bool = True


CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast("R1", "reference", "K1", "m", "w", scored=False),
    Contrast("R2", "reference", "K2", "w0", "w", scored=False),
    Contrast("K1 P-roll", "P-roll monthly", "K1", "pm", "pw", "R1"),
    Contrast("K1 whole-history 150 rounds", "weekly 150 rounds", "K1", "fm", "fw", "R1"),
    Contrast(
        "K1 P-roll quarterly (descriptive)", "P-roll quarterly", "K1", "pq", "pw", scored=False
    ),
    Contrast("K2 P-roll monthly", "P-roll monthly", "K2", "pm0", "pm", "R2"),
    Contrast("K2 P-roll quarterly", "P-roll quarterly", "K2", "pq0", "pq", "R2"),
    Contrast("K2 P-roll weekly", "P-roll weekly", "K2", "pw0", "pw", "R2"),
    Contrast("K2 full monthly", "full monthly", "K2", "m0", "m", "R2"),
    Contrast("K2 weekly 150 rounds", "weekly 150 rounds", "K2", "fw0", "fw", "R2"),
    Contrast("K2 P-roll monthly repeat", "noise", "K2", "pm0_rep", "pm_rep", scored=False),
)


def month_bounds(*, values: np.ndarray, months: np.ndarray) -> tuple[float, float]:
    """Return the 95% interval of the mean of `values`, resampling whole months.

    The draws are those of `studies.bootstrap`'s resampler: for each resample one seed draw, then
    one draw of the months with replacement. The resampled mean is the sum of the drawn months'
    sums over the sum of their counts.
    """
    _unique, inverse = np.unique(months, return_inverse=True)
    sums = np.bincount(inverse, weights=values.astype(np.float64))
    counts = np.bincount(inverse)
    generator = np.random.default_rng(bootstrap.BOOTSTRAP_SEED)
    resampled = np.empty(N_RESAMPLES)
    for draw in range(N_RESAMPLES):
        generator.integers(0, 1)
        drawn = generator.integers(0, len(sums), size=len(sums))
        resampled[draw] = sums[drawn].sum() / counts[drawn].sum()
    tail = (100.0 - LEVEL) / 2.0
    return float(np.percentile(resampled, tail)), float(np.percentile(resampled, 100.0 - tail))


def case_table(*, name: str) -> pl.DataFrame:
    """Return one arm's per-case metrics, keyed by series, run, and valid time."""
    frame = (
        pl.read_parquet(common.ARMS_DIR / f"{name}.parquet")
        .filter(pl.col("actual").is_finite() & (pl.col("valid_time") < common.HOLDOUT))
        .sort(*common.KEYS, "seed")
    )
    quantiles = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1)
    actual = frame["actual"].to_numpy().astype(np.float64)
    score = common.pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0
    miss = actual[:, None] - quantiles
    pinball = np.maximum(common.LEVELS * miss, (common.LEVELS - 1.0) * miss) * 100.0
    return frame.select(*common.KEYS, "site").with_columns(
        qs=pl.Series(score),
        mae=pl.Series(np.abs(quantiles[:, 4] - actual) * 100.0),
        pin05=pl.Series(pinball[:, 0]),
        pin95=pl.Series(pinball[:, 8]),
        cov90=pl.Series(((actual >= quantiles[:, 0]) & (actual <= quantiles[:, 8])).astype(float)),
        month=frame["power_fcst_init_time"].dt.strftime("%Y-%m"),
    )


def parse_cost() -> pl.DataFrame:
    """Return the mean fit seconds of one refit per configuration, from the fit logs."""
    rows = []
    for path in sorted(LOG_DIR.glob("*.log")):
        seconds = [float(m.group(1)) for m in re.finditer(r" fit (\d+) s", path.read_text())]
        if seconds:
            rows.append({"config": path.stem, "fits": len(seconds), "mean_fit_s": np.mean(seconds)})
    return pl.DataFrame(rows)


def absolute_rows(*, arms: dict[str, pl.DataFrame], shared: pl.DataFrame) -> list[dict[str, Any]]:
    """Return each arm's mean metrics on the shared cases, with a 95% interval on the score."""
    absolute = []
    for name, table in arms.items():
        part = table.join(shared, on=list(common.KEYS), how="inner")
        low, high = month_bounds(values=part["qs"].to_numpy(), months=part["month"].to_numpy())
        absolute.append(
            {
                "arm": name,
                "qs": part["qs"].mean(),
                "qs_low": low,
                "qs_high": high,
                "mae": part["mae"].mean(),
                "cov90": part["cov90"].mean(),
                "pin05": part["pin05"].mean(),
                "pin95": part["pin95"].mean(),
            }
        )
    return absolute


def contrast_rows(*, arms: dict[str, pl.DataFrame]) -> list[dict[str, Any]]:
    """Return each contrast whose two arms exist: estimate and 95% interval, paired on case."""
    rows = []
    for spec in CONTRASTS:
        if spec.treatment not in arms or spec.reference not in arms:
            continue
        paired = arms[spec.treatment].join(
            arms[spec.reference], on=list(common.KEYS), how="inner", suffix="_ref"
        )
        difference = (paired["qs"] - paired["qs_ref"]).to_numpy()
        low, high = month_bounds(values=difference, months=paired["month"].to_numpy())
        rows.append(
            {
                "contrast": spec.label,
                "schedule": spec.schedule,
                "kind": spec.kind,
                "cases": paired.height,
                "estimate": float(difference.mean()),
                "low": low,
                "high": high,
            }
        )
    return rows


def main() -> None:
    """Compute and print every table."""
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arms = {p.stem: case_table(name=p.stem) for p in sorted(common.ARMS_DIR.glob("*.parquet"))}
    print("arms:", ", ".join(arms))
    shared: pl.DataFrame | None = None
    for table in arms.values():
        keys = table.select(*common.KEYS)
        shared = keys if shared is None else shared.join(keys, on=list(common.KEYS), how="inner")
    if shared is None:
        msg = "no arm tables found; run e00085_assemble.py first"
        raise FileNotFoundError(msg)
    shared = shared.unique()
    print(f"cases every arm shares: {shared.height:,}")
    absolute = absolute_rows(arms=arms, shared=shared)
    absolute_frame = pl.DataFrame(absolute)
    absolute_frame.write_parquet(common.RESULTS_DIR / "absolute.parquet")
    with pl.Config(tbl_rows=40, tbl_cols=10, float_precision=3):
        print(absolute_frame)
    rows = contrast_rows(arms=arms)
    contrasts = pl.DataFrame(rows)
    contrasts.write_parquet(common.RESULTS_DIR / "contrasts.parquet")
    with pl.Config(tbl_rows=40, tbl_cols=10, float_precision=3, fmt_str_lengths=60):
        print(contrasts)
    by = {row["contrast"]: row for row in rows}
    arm_qs = {row["arm"]: row["qs"] for row in absolute}
    noise = max(
        NOISE_MINIMUM,
        abs(arm_qs.get("pm_rep", arm_qs["pm"]) - arm_qs["pm"]),
        abs(arm_qs.get("pm0_rep", arm_qs.get("pm0", 0.0)) - arm_qs.get("pm0", 0.0)),
        abs(by["K2 P-roll monthly repeat"]["estimate"] - by["K2 P-roll monthly"]["estimate"])
        if "K2 P-roll monthly repeat" in by and "K2 P-roll monthly" in by
        else 0.0,
    )
    tolerance = max(TOLERANCE_MINIMUM, noise)
    print(f"noise floor N = {noise:.4f}; tolerance T = {tolerance:.4f}")
    verdicts = []
    for spec in CONTRASTS:
        if not spec.scored or spec.compare_to is None or spec.label not in by:
            continue
        ref, got = by[spec.compare_to], by[spec.label]
        if ref["low"] * ref["high"] > 0 and abs(ref["estimate"]) > noise:
            agrees = (
                np.sign(got["estimate"]) == np.sign(ref["estimate"])
                and got["low"] * got["high"] > 0
                and abs(got["estimate"] - ref["estimate"]) <= tolerance
            )
            rule = "i"
        else:
            agrees = abs(got["estimate"]) <= tolerance
            rule = "ii"
        verdicts.append(
            {
                "schedule": spec.schedule,
                "contrast": spec.label,
                "reference": round(ref["estimate"], 3),
                "schedule_estimate": round(got["estimate"], 3),
                "rule": rule,
                "agrees": bool(agrees),
            }
        )
    agreement = pl.DataFrame(verdicts)
    agreement.write_parquet(common.RESULTS_DIR / "agreement.parquet")
    with pl.Config(tbl_rows=40, tbl_cols=10, fmt_str_lengths=60):
        print(agreement)
        print(
            agreement.group_by("schedule").agg(
                scored=pl.len(), agreed=pl.col("agrees").sum(), passes=pl.col("agrees").all()
            )
        )
    cost = parse_cost()
    with pl.Config(tbl_rows=40, float_precision=1):
        print(cost)
    cost.write_parquet(common.RESULTS_DIR / "cost.parquet")


if __name__ == "__main__":
    main()
