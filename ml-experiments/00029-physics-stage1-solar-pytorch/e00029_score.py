"""Score experiment 00029: the planned contrasts, absolute CRPS by lead band, and coverage.

Reads the prediction sets written by `e00029_arms.py`, experiment 00020's `base` set, and
experiment 00002's climatology rows, and writes `WORK_DIR/results/` tables and `report.md`. Every
number on the write-up page is printed here.

    uv run --no-sync python ml-experiments/00029-*/e00029_score.py
"""

import json
from typing import Final, cast

import numpy as np
import polars as pl
from e00029_common import (
    KEY_COLUMNS,
    LEAD_BANDS,
    LEVEL_COLUMNS,
    ROW_SET_ID,
    SCORES_DIR,
    SEEDS,
    WORK_DIR,
    covered,
    crps_estimate,
    site_table,
)
from e00029_fit import ARMS, PARAMS_PATH, POINT_DIR

from studies import bootstrap, experiment_store

RESULTS_DIR: Final = WORK_DIR / "results"
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 4)
N_RESAMPLES: Final[int] = 20_000
GAIN_THRESHOLD: Final[float] = -0.05
PLANNED: Final[tuple[tuple[str, str, str], ...]] = (
    ("C1", "torch", "base"),
    ("C2", "torch_hybrid", "base"),
    ("C3", "torch", "scipy"),
    ("C4", "scipy", "base"),
)
"""Each contrast is the arm named first minus the arm named second."""
PUBLISHED_00012_W_MAE: Final[float] = 7.641
"""Experiment 00012's arm `w`: MAE of the 51-member average for the 6 solar farms, points."""


def arm_rows(*, experiment_id: int, name: str) -> pl.DataFrame:
    """Return one prediction set's per-case CRPS (points) and coverage, solar farms only."""
    solar = site_table()["time_series_id"].to_list()
    frame = experiment_store.read_predictions(
        experiment_id=experiment_id, name=name, row_set_id=ROW_SET_ID
    ).filter(pl.col("time_series_id").is_in(solar))
    quantiles = frame.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual_norm"].to_numpy().astype(np.float64)
    return frame.select(*KEY_COLUMNS, "seed", "lead_hours", "band", "trivial").with_columns(
        crps=pl.Series(100.0 * crps_estimate(quantiles=quantiles, actual=actual)),
        covered=pl.Series(covered(quantiles=quantiles, actual=actual)).cast(pl.Float64),
    )


def climatology_rows(*, base: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 00002's month-by-half-hour climatology on `base`'s cases."""
    stored = pl.concat(
        [
            pl.read_parquet(SCORES_DIR / f"rows_seed{seed}.parquet").filter(
                pl.col("arm") == "climatology_q"
            )
            for seed in SEEDS
        ]
    ).select(*KEY_COLUMNS, "seed", "crps", covered=pl.col("cover_90") / 100.0)
    return base.select(*KEY_COLUMNS, "seed", "lead_hours", "band", "trivial").join(
        stored, on=[*KEY_COLUMNS, "seed"], how="left", validate="1:1"
    )


def losses(*, rows: dict[str, pl.DataFrame], arms: list[str]) -> pl.DataFrame:
    """Return the paired loss frame of the arms."""
    return experiment_store.losses_for_bootstrap(
        arm_losses={name: rows[name].select(*KEY_COLUMNS, "seed", "crps") for name in arms}
    )


def verdict(*, c1: dict, c2: dict) -> str:
    """Apply the decision rule to the two headline contrasts."""
    classes = []
    for row in (c1, c2):
        if row["upper"] < 0.0:
            classes.append("gain" if row["difference"] <= GAIN_THRESHOLD else "small gain")
        elif row["lower"] > GAIN_THRESHOLD:
            classes.append("refuted")
        else:
            classes.append("inconclusive")
    if "gain" in classes:
        return "supported"
    if "small gain" in classes:
        return "small gain"
    if classes.count("refuted") == 2:
        return "refuted"
    return "inconclusive"


def physics_point_mae() -> pl.DataFrame:
    """Return the member-average physics power's MAE (points) against actual power, by arm."""
    cases = arm_rows(experiment_id=20, name="base").filter(pl.col("seed") == SEEDS[0])
    actual = (
        experiment_store.read_predictions(experiment_id=20, name="base", row_set_id=ROW_SET_ID)
        .filter(pl.col("seed") == SEEDS[0])
        .select(*KEY_COLUMNS, "actual_norm")
    )
    point = pl.read_parquet(POINT_DIR / "physics_point.parquet").join(
        actual, on=list(KEY_COLUMNS), how="inner", validate="1:1"
    )
    assert point.height == cases.height
    return pl.DataFrame(
        [
            {
                "arm": arm,
                "mae": 100.0 * float(cast(float, (point[arm] - point["actual_norm"]).abs().mean())),
            }
            for arm in ("scipy", "torch")
        ]
    )


def main() -> None:
    """Score every arm and print the report."""
    rows = {arm: arm_rows(experiment_id=29, name=arm.replace("_", "-")) for arm in ARMS}
    rows["base"] = arm_rows(experiment_id=20, name="base")
    rows["climatology"] = climatology_rows(base=rows["base"])
    contrasts = []
    absolute = [
        {"arm": arm}
        | bootstrap.bootstrap_absolute(losses=losses(rows=rows, arms=[arm]), arm=arm, metric="crps")
        for arm in rows
    ]
    for contrast, arm, reference in PLANNED:
        frame = losses(rows=rows, arms=[arm, reference])
        differences, _ = bootstrap.paired_differences(
            losses=frame, treatment=arm, reference=reference, metric="crps"
        )
        lower, upper = bootstrap.bootstrap_difference_at_level(
            losses=frame, treatment=arm, reference=reference, metric="crps",
            level=PLANNED_LEVEL, n_resamples=N_RESAMPLES,
        )  # fmt: skip
        contrasts.append(
            {"contrast": contrast, "arm": arm, "reference": reference,
             "difference": float(differences.mean()), "lower": lower, "upper": upper}
        )  # fmt: skip
    absolute_frame, contrast_frame = pl.DataFrame(absolute), pl.DataFrame(contrasts)
    by_band = pl.concat(
        [
            frame.group_by("band")
            .agg(pl.col("crps").mean(), pl.col("covered").mean())
            .with_columns(arm=pl.lit(arm))
            for arm, frame in rows.items()
        ]
    )
    heads = {r["contrast"]: r for r in contrast_frame.iter_rows(named=True)}
    final = verdict(c1=heads["C1"], c2=heads["C2"])
    point = physics_point_mae()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    absolute_frame.write_parquet(RESULTS_DIR / "absolute.parquet")
    contrast_frame.write_parquet(RESULTS_DIR / "contrasts.parquet")
    by_band.write_parquet(RESULTS_DIR / "by_band.parquet")
    lines = [
        f"# Experiment 00029 report (verdict: {final})",
        "",
        "## Mean CRPS (points, 95% interval)",
    ]
    lines += [
        f"- {r['arm']}: {r['value']:.3f} ({r['lower_95']:.3f} to {r['upper_95']:.3f}), "
        f"seed spread {r['seed_spread']:.3f}"
        for r in absolute_frame.sort("value").iter_rows(named=True)
    ]
    lines += ["", f"## Planned contrasts ({PLANNED_LEVEL:.2f}% intervals)"]
    lines += [
        f"- {r['contrast']} {r['arm']} - {r['reference']}: {r['difference']:+.3f} "
        f"({r['lower']:+.3f} to {r['upper']:+.3f})"
        for r in contrast_frame.iter_rows(named=True)
    ]
    lines += ["", "## Physics point forecast, member-average MAE (points)"]
    lines += [
        f"- {r['arm']}: {r['mae']:.2f} (experiment 00012 arm w: {PUBLISHED_00012_W_MAE})"
        for r in point.iter_rows(named=True)
    ]
    lines += ["", "## CRPS by lead band (points)"]
    for arm in rows:
        part = by_band.filter(pl.col("arm") == arm)
        cells = [f"{part.filter(pl.col('band') == b)['crps'].item():.2f}" for b, _, _ in LEAD_BANDS]
        lines.append(
            f"- {arm}: "
            + ", ".join(f"{b} {c}" for (b, _, _), c in zip(LEAD_BANDS, cells, strict=True))
        )
    lines += ["", "## Coverage of the 5% to 95% band (%, all non-trivial and trivial cases)"]
    lines += [
        f"- {arm}: {100 * float(np.mean(frame['covered'].to_numpy())):.1f}"
        for arm, frame in rows.items()
    ]
    if PARAMS_PATH.exists():
        params = json.loads(PARAMS_PATH.read_text())
        lines += ["", "## Fitted parameters (torch: posterior mean and 90% interval; scipy)"]
        for site, entry in sorted(params.items()):
            t, s = entry["torch"], entry["scipy"]
            lines.append(
                f"- {site}: torch tilt {t['tilt']:.1f} ({t['tilt_lo']:.1f} to {t['tilt_hi']:.1f}), "
                f"azimuth {t['azimuth']:.1f} ({t['azimuth_lo']:.1f} to {t['azimuth_hi']:.1f}), "
                f"DC {t['dc']:.2f}, AC {t['ac']:.2f}, DC:AC {t['ratio']:.2f}; "
                f"scipy tilt {s['tilt']:.1f}, azimuth {s['azimuth']:.1f}, DC {s['dc']:.2f}, "
                f"AC {s['ac']:.2f}"
            )
    report = "\n".join(lines) + "\n"
    (RESULTS_DIR / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
