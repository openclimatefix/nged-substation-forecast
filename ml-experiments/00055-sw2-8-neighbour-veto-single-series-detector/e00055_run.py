r"""Match thresholds on odd months, then score injected and labelled transfers.

Stages, in order. Each writes files to the private output folder.

- `harness-check`: D0 and D1 at experiment 43's thresholds must reproduce its alarm counts and the
  outcome of each of its 600 injected transfers. Stop if they do not.
- `tune`: for D0M (no veto) and for D2 at each veto strength, the matched thresholds and curves.
- `ceiling`: D0M against D0 on experiment 43's headline-band transfers that start in odd months.
- `score`: the outcome of every injected transfer for every setting at each veto strength.
- `real`: the real alarms, their tiers, and the labelled real transfers, at each veto strength.

    uv run python ml-experiments/00055-*/e00055_run.py harness-check tune ceiling score real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import polars as pl
from e00055_common import EXPERIMENT_43_DIR, WORK_DIR, near_log_masks, stamp_months
from e00055_detect import (
    EXPLAIN_PAD,
    NO_VETO,
    OLD_THRESHOLDS,
    SEED_43,
    SEED_NEW,
    TARGET_COUNTS,
    VETO_ZS,
    detect_all,
    load,
    new_events,
    odd_unexplained,
    original_events,
    quiet_masks,
    real_alarm_rows,
    real_events,
    score_events,
    score_real,
    seeded_batches,
    tune_d2,
)
from e00055_detectors import prepare
from e00055_inject import HEADLINE_BAND, events_frame

HARNESS_ALARM_COUNTS: Final[dict[str, int]] = {"D0_8": 432, "D1_8": 444, "D0_48": 297, "D1_48": 110}
HARNESS_FOUND_D0_48: Final[int] = 47
SAMPLE_NAMES: Final[dict[str, str]] = {
    "D0_8": "alarms_D0_8_6.0",
    "D1_8": "alarms_D1_8_3.75",
    "D0_48": "alarms_D0_48_4.0",
    "D1_48": "alarms_D1_48_2.5",
}


def vz_tag(veto_z: float) -> str:
    """File tag of a veto strength."""
    return "none" if veto_z == NO_VETO else f"{veto_z:g}"


def all_thresholds(*, matched: dict[str, dict[str, float]], veto_z: float) -> dict[str, float]:
    """Thresholds of every setting: experiment 43's for D0 and D1, matched for D0M and D2."""
    return {
        **OLD_THRESHOLDS,
        "D0M_8": matched["none"]["8"],
        "D0M_48": matched["none"]["48"],
        "D2_8": matched[vz_tag(veto_z)]["8"],
        "D2_48": matched[vz_tag(veto_z)]["48"],
    }


def read_matched(*, out: Path) -> dict[str, dict[str, float]]:
    """The matched thresholds `tune` wrote, by veto-strength tag and window."""
    return json.loads((out / "thresholds.json").read_text())


def harness_check(*, out: Path) -> None:
    """Reproduce experiment 43's alarm counts and event outcomes."""
    panel, scales = load()
    thresholds = {**OLD_THRESHOLDS, "D0M_8": 6.0, "D0M_48": 4.0, "D2_8": 6.0, "D2_48": 4.0}
    results = detect_all(
        residual=panel["residual"],
        expected=panel["expected"],
        scales=scales,
        thresholds=thresholds,
        veto_z=2.0,
    )
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    for setting, expected in HARNESS_ALARM_COUNTS.items():
        count = results[setting][0].height
        print(f"{setting}: {count} alarms (experiment 43: {expected})")
        if count != expected:
            raise SystemExit(f"STOP: {setting} alarm count differs from experiment 43")
    for window, target in TARGET_COUNTS.items():
        count = odd_unexplained(table=results[f"D0_{window}"][0], logged=logged)
        print(f"D0_{window}: {count} odd-month unexplained alarms (target {target})")
        if count != target:
            raise SystemExit("STOP: odd-month unexplained count differs from experiment 43")
    events = original_events(power=panel["power"])
    scored = score_events(
        panel=panel,
        scales=scales,
        thresholds=thresholds,
        veto_z=2.0,
        groups=seeded_batches(events=events, seed=SEED_43),
    )
    stored = (
        pl.read_parquet(EXPERIMENT_43_DIR / "event_results.parquet")
        .filter(pl.col("variant") == "plain")
        .with_columns(
            setting=pl.col("setting").replace_strict({v: k for k, v in SAMPLE_NAMES.items()})
        )
    )
    joined = scored.join(stored, on=["event_id", "setting"], suffix="_43", validate="1:1")
    different = joined.filter(
        (pl.col("found") != pl.col("found_43"))
        | (pl.col("onset_found") != pl.col("onset_found_43"))
    )
    table = events_frame(events=events)
    headline = table.filter(
        pl.col("size_band") == HEADLINE_BAND[0], pl.col("duration_band") == HEADLINE_BAND[1]
    )["event_id"]
    found = joined.filter(
        pl.col("setting") == "D0_48", pl.col("event_id").is_in(headline.implode())
    )["found"].sum()
    print(f"{joined.height} event rows, {different.height} differ; D0 24 h found {found} of 160")
    if different.height or found != HARNESS_FOUND_D0_48:
        raise SystemExit("STOP: the harness does not reproduce experiment 43")
    print("harness reproduced")


def tune(*, out: Path) -> None:
    """Match D0M's and D2's thresholds on odd months, for each veto strength."""
    panel, _ = load()
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    matched: dict[str, dict[str, float]] = {}
    curves: list[dict] = []
    for veto_z in (NO_VETO, *VETO_ZS):
        matched[vz_tag(veto_z)] = {}
        for window in (8, 48):
            inputs = prepare(residuals=panel["residual"], window=window)
            threshold, curve = tune_d2(inputs=inputs, veto_z=veto_z, logged=logged)
            matched[vz_tag(veto_z)][str(window)] = threshold
            curves += [
                {"veto_z": vz_tag(veto_z), "window": window, "threshold": t, "odd_unexplained": c}
                for t, c in curve
            ]
        print(vz_tag(veto_z), matched[vz_tag(veto_z)], flush=True)
    (out / "thresholds.json").write_text(json.dumps(matched))
    pl.DataFrame(curves).write_parquet(out / "curves.parquet")


def event_sets(*, out: Path, power: dict) -> tuple[list, list, list, pl.DataFrame]:
    """Draw the original, extra, and holiday sets; write the combined event table."""
    original = original_events(power=power)
    extra, holiday = new_events(power=power)
    full = quiet_masks()
    pieces = []
    for name, events in (("original", original), ("extra", extra), ("holiday", holiday)):
        table = events_frame(events=events).with_columns(set=pl.lit(name))
        clash = [
            any(full[s][max(e.onset - 96, 0) : e.end + 96].any() for s in e.series) for e in events
        ]
        pieces.append(table.with_columns(alarm_quiet=pl.Series([not c for c in clash])))
    table = pl.concat(pieces)
    table.write_parquet(out / "events.parquet")
    return original, extra, holiday, table


def ceiling(*, out: Path) -> None:
    """D0M against D0 on experiment 43's odd-month headline transfers."""
    panel, scales = load()
    matched = read_matched(out=out)
    original = original_events(power=panel["power"])
    scored = score_events(
        panel=panel,
        scales=scales,
        thresholds=all_thresholds(matched=matched, veto_z=2.0),
        veto_z=2.0,
        groups=seeded_batches(events=original, seed=SEED_43),
    )
    table = events_frame(events=original)
    months = stamp_months()
    table = table.with_columns(month=pl.Series(months[table["onset"].to_numpy()])).filter(
        pl.col("size_band") == HEADLINE_BAND[0],
        pl.col("duration_band") == HEADLINE_BAND[1],
        pl.col("month") % 2 == 1,
    )
    part = scored.join(table.select("event_id"), on="event_id")
    for setting in ("D0_48", "D0M_48", "D0_8", "D0M_8"):
        rows = part.filter(pl.col("setting") == setting)
        print(f"{setting}: found {rows['found'].sum()} of {rows.height}")
    part.write_parquet(out / "ceiling.parquet")


def score(*, out: Path) -> None:
    """Score every event of every set at each veto strength."""
    panel, scales = load()
    matched = read_matched(out=out)
    original, extra, holiday, _ = event_sets(out=out, power=panel["power"])
    groups = seeded_batches(events=original, seed=SEED_43) + seeded_batches(
        events=extra + holiday, seed=SEED_NEW
    )
    for veto_z in VETO_ZS:
        scored = score_events(
            panel=panel,
            scales=scales,
            thresholds=all_thresholds(matched=matched, veto_z=veto_z),
            veto_z=veto_z,
            groups=groups,
        )
        scored.write_parquet(out / f"event_results_{vz_tag(veto_z)}.parquet")
        print(f"veto {vz_tag(veto_z)}: {scored.height} event rows", flush=True)


def real(*, out: Path) -> None:
    """Write the real alarms and the labelled real transfers' outcomes for each veto strength."""
    panel, scales = load()
    matched = read_matched(out=out)
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    events = real_events(power=panel["power"])
    for veto_z in VETO_ZS:
        results = detect_all(
            residual=panel["residual"],
            expected=panel["expected"],
            scales=scales,
            thresholds=all_thresholds(matched=matched, veto_z=veto_z),
            veto_z=veto_z,
        )
        real_alarm_rows(results=results, logged=logged).write_parquet(
            out / f"alarms_{vz_tag(veto_z)}.parquet"
        )
        score_real(results=results, real=events).write_parquet(
            out / f"real_results_{vz_tag(veto_z)}.parquet"
        )
        print(f"veto {vz_tag(veto_z)}: real alarms and events written", flush=True)


def main() -> None:
    """Run the requested stages."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stages", nargs="+", choices=("harness-check", "tune", "ceiling", "score", "real")
    )
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = {
        "harness-check": harness_check,
        "tune": tune,
        "ceiling": ceiling,
        "score": score,
        "real": real,
    }
    for stage in args.stages:
        stages[stage](out=args.out)


if __name__ == "__main__":
    main()
