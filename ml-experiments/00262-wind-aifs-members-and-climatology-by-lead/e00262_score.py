"""Score experiment 00262's arms and read the planned contrasts.

Reads the arms from `work/arms/` (`<arm>` without and `<arm>+T` with the offset table) and
experiment 00259's stored A0r (R0, spread factor and table fitted on its hindcast) and experiment
230's climatology as reference rows. Scores October 2025 to June 2026 only: the rows of July to
September 2025 train the weights, the spread factors, and the tables. Scores are the quantile score
of experiment 33 (nine levels, points of capacity), the pinball loss at the named tail levels, the
tail score T (the mean of the 0.01 and 0.99 pinball losses), coverage of the 5% to 95% and 1% to
99% bands, and the share of rows below the median, by lead-day band and row set. Contrasts resample
whole months (of the ENS run) and the three seeds; the farms are not resampled.

Output: `work/score.json`.

    uv run --no-sync python ml-experiments/00262-*/e00262_score.py
"""

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00262_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ARM_DIR: Final = common.WORK_DIR / "arms"
RESAMPLES: Final[int] = 50_000
BONFERRONI_LEVEL: Final[float] = 1 - 0.05 / 3
LEVEL_95: Final[float] = 0.95
FIRST_SCORED_MONTH: Final[str] = "2025-10"
AIFS_V2: Final = datetime(2026, 5, 12, tzinfo=UTC)
"""The first AIFS ensemble run of version 2."""
CONTRAST_METRICS: Final[tuple[str, ...]] = ("qs", "tail_t", "pinball_q99")
ARMS: Final[tuple[str, ...]] = (
    "M0",
    "M1q",
    "M1p",
    "M2",
    "M3",
    "M4",
    "M5",
    "M5b",
    "M0+Thindcast",
    "M3prev",
)
TRAINING_MONTHS: Final[tuple[str, ...]] = ("2025-07", "2025-08", "2025-09")
SCORE_KEYS: Final[tuple[str, ...]] = (*common.KEY_COLUMNS, "seed")


def read_arm(*, name: str, cases: pl.DataFrame) -> pl.DataFrame | None:
    """Return one arm's quantiles joined to `cases`, sorted by seed and key, or None if absent."""
    path = ARM_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    frame = pl.read_parquet(path)
    return cases.join(frame, on=list(common.KEY_COLUMNS), how="inner", validate="1:m").sort(
        "seed", *common.KEY_COLUMNS
    )


def case_scores(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return per-row scores of one arm (points of capacity for the scores)."""
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual_norm"].to_numpy().astype(np.float64)
    scale = common.POINTS_PER_FRACTION
    out = {
        "qs": common.quantile_score(quantiles=quantiles, actual=actual) * scale,
        "covered": common.covered(quantiles=quantiles, actual=actual).astype(np.float64),
        "covered_1_99": ((actual >= quantiles[:, 0]) & (actual <= quantiles[:, 10])).astype(
            np.float64
        ),
        "below_median": (actual < quantiles[:, common.MEDIAN_INDEX]).astype(np.float64),
    }
    for name in common.NAMED_LEVELS:
        out[f"pinball_{name}"] = (
            common.pinball_at(quantiles=quantiles, actual=actual, column=name) * scale
        )
    out["tail_t"] = (out["pinball_q01"] + out["pinball_q99"]) / 2.0
    return out


def bootstrap(
    *, difference: np.ndarray, month: np.ndarray, seed: np.ndarray, levels: tuple[float, ...]
) -> list[float]:
    """Return the mean difference and its interval at each of `levels`, resampling months, seeds."""
    months, seeds = np.unique(month), np.unique(seed)
    sums = np.zeros((months.size, seeds.size))
    counts = np.zeros_like(sums)
    for i, m in enumerate(months):
        for j, s in enumerate(seeds):
            keep = (month == m) & (seed == s)
            sums[i, j], counts[i, j] = difference[keep].sum(), keep.sum()
    rng = np.random.default_rng(0)
    draw_m = rng.integers(0, months.size, size=(RESAMPLES, months.size))
    draw_s = rng.integers(0, seeds.size, size=(RESAMPLES, seeds.size))
    picked = (draw_m[:, :, None], draw_s[:, None, :])
    means = sums[picked].sum(axis=(1, 2)) / counts[picked].sum(axis=(1, 2))
    out = [float(sums.sum() / counts.sum())]
    for level in levels:
        tail = (1 - level) / 2
        out += [float(x) for x in np.quantile(means, [tail, 1 - tail])]
    return out


def main() -> None:  # noqa: PLR0912, PLR0915
    """Score every arm present and write the report."""
    sites = common.site_table()
    cases = (
        common.cases_table(sites=sites)
        .filter(pl.col("actual_norm").is_finite())
        .with_columns(
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
            aifs_v2=pl.col("power_fcst_init_time") >= AIFS_V2,
        )
    )
    arms: dict[str, tuple[pl.DataFrame, dict[str, np.ndarray]]] = {}
    names = [n for base in ARMS for n in (base, f"{base}+T")]
    for name in names:
        frame = read_arm(name=name, cases=cases)
        if frame is not None:
            arms[name] = (frame, case_scores(frame=frame))
    row_sets: dict[str, Callable[[pl.DataFrame], np.ndarray]] = {
        "all rows": lambda f: np.ones(f.height, dtype=bool),
        "kept rows": lambda f: ~f["masked"].to_numpy(),
    }
    band_masks: dict[str, Callable[..., np.ndarray]] = {
        name: lambda f, lo=lo, hi=hi: (
            (f["lead_day"].to_numpy() >= lo) & (f["lead_day"].to_numpy() <= hi)
        )
        for name, lo, hi in common.LEAD_DAY_BANDS
    }
    band_masks["hours 12-36"] = lambda f: (
        (f["lead_hours"].to_numpy() >= 12.0) & (f["lead_hours"].to_numpy() <= 36.0)
    )
    scored = {n: f["month"].to_numpy() >= FIRST_SCORED_MONTH for n, (f, _) in arms.items()}
    report: dict = {"scores": {}, "contrasts": {}, "checks": {}, "by_version": {}}
    for name, (frame, scores) in arms.items():
        report["scores"][name] = {}
        for rs_name, rs in row_sets.items():
            for band, bm in band_masks.items():
                keep = rs(frame) & bm(frame) & scored[name]
                report["scores"][name][f"{rs_name} | {band}"] = {
                    metric: float(values[keep].mean()) if keep.any() else None
                    for metric, values in scores.items()
                } | {"rows": int(keep.sum())}
        report["by_version"][name] = {
            label: float(
                scores["qs"][
                    ~frame["masked"].to_numpy()
                    & scored[name]
                    & (frame["aifs_v2"].to_numpy() == is_v2)
                ].mean()
            )
            for label, is_v2 in (("before 2026-05-12", False), ("from 2026-05-12", True))
        }
    for label_prefix in ("M1",):
        # The better of M1q and M1p on the training months (in-sample for their weights).
        chosen = None
        if "M1q+T" in arms and "M1p+T" in arms:
            training = {
                m: float(
                    arms[m][1]["qs"][
                        ~arms[m][0]["masked"].to_numpy()
                        & arms[m][0]["month"].is_in(TRAINING_MONTHS).to_numpy()
                    ].mean()
                )
                for m in ("M1q+T", "M1p+T")
            }
            chosen = min(training, key=training.__getitem__)
            report["checks"]["M1 choice on training months"] = {
                "scores": training,
                "chosen": chosen,
            }
        report["checks"][label_prefix] = chosen
    contrasts: list[tuple[str, str | None, str, bool]] = [
        ("C1", report["checks"].get("M1"), "M0+T", True),
        ("C2", "M3+T", "M0+T", True),
        ("C3", "M5+T", "M4+T", True),
        ("D1", "M5+T", "M5b+T", False),
        ("D2", "M4+T", "M0+T", False),
        ("D3", "M1p+T", "M1q+T", False),
        ("D4", "M2+T", "M0+T", False),
        ("D5", "M0+T", "M0", False),
        ("D6", "M0+T", "M0+Thindcast", False),
        ("D7", "M3", "M0", False),
        ("D8", "M5", "M4", False),
        ("D10", "M1q", "M0", False),
    ]
    for label, first, second, decisive in contrasts:
        if first is None or first not in arms or second not in arms:
            continue
        (frame_a, sa), (frame_b, sb) = arms[first], arms[second]
        if not frame_a.select(SCORE_KEYS).equals(frame_b.select(SCORE_KEYS)):
            msg = f"{first} and {second} do not cover the same rows"
            raise ValueError(msg)
        report["contrasts"][label] = {"first": first, "second": second}
        for rs_name, rs in row_sets.items():
            for band, bm in band_masks.items():
                keep = rs(frame_a) & bm(frame_a) & scored[first]
                levels = (LEVEL_95, BONFERRONI_LEVEL) if decisive else (LEVEL_95,)
                for metric in CONTRAST_METRICS:
                    report["contrasts"][label][f"{rs_name} | {band} | {metric}"] = bootstrap(
                        difference=(sa[metric] - sb[metric])[keep],
                        month=frame_a["month"].to_numpy()[keep],
                        seed=frame_a["seed"].to_numpy()[keep],
                        levels=levels,
                    )
    if "M3prev+T" in arms and "M3+T" in arms:
        # Previous-run check: both arms on the rows that have a previous AIFS run.
        frame_p, _ = arms["M3prev+T"]
        both = frame_p.select(SCORE_KEYS)
        frame_m = arms["M3+T"][0].join(both, on=list(SCORE_KEYS), how="semi")
        if not frame_m.select(SCORE_KEYS).equals(both):
            msg = "M3+T and M3prev+T rows are not in the same order"
            raise ValueError(msg)
        sp, sm = arms["M3prev+T"][1], case_scores(frame=frame_m)
        keep = ~frame_p["masked"].to_numpy() & (frame_p["month"].to_numpy() >= FIRST_SCORED_MONTH)
        report["contrasts"]["D9"] = {
            "first": "M3prev+T",
            "second": "M3+T (same rows)",
            "kept rows | all leads | qs": bootstrap(
                difference=(sp["qs"] - sm["qs"])[keep],
                month=frame_p["month"].to_numpy()[keep],
                seed=frame_p["seed"].to_numpy()[keep],
                levels=(LEVEL_95,),
            ),
        }
    weights_path = ARM_DIR / "weights.json"
    if weights_path.exists():
        report["checks"]["weights_file"] = str(weights_path.name)
    (common.WORK_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for name in arms:
        print(
            name,
            {
                k: round(v["qs"], 3)
                for k, v in report["scores"][name].items()
                if k in ("kept rows | all leads", "all rows | all leads")
            },
        )
    for label, content in report["contrasts"].items():
        print(
            label,
            content["first"],
            "-",
            content["second"],
            "kept | all leads | qs:",
            [round(x, 3) for x in content.get("kept rows | all leads | qs", [])],
        )
    print(report["checks"])


if __name__ == "__main__":
    main()
