"""Part 2: print every number of the onset-location and mask results.

Reads the tables of `e00188_p2_events.py` (alarms and events, with located onsets) and
`e00188_p2_score.py` (per-row scores of each mask arm). Every number quoted in `result.md` is
printed here and saved to `report_p2.json`.

    uv run --no-sync python e00188_p2_report.py | tee printed_p2.txt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00188_common import WORK_DIR

P2: Final = WORK_DIR / "p2"
SCORES: Final = WORK_DIR / "p2_scores"
N_DRAWS: Final[int] = 20_000
ARMS: Final[tuple[str, ...]] = (
    "K0",
    "CLEAN",
    "MASK_ALARM",
    "MASK_STEP",
    "MASK_LOCATED",
    "MASK_TRUE",
)
REPORT: dict[str, object] = {}
TAUS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}


def read_all(*, pattern: str, folder: Path) -> pl.DataFrame:
    """Concatenate the non-empty parquet files matching a pattern."""
    frames = [pl.read_parquet(p) for p in sorted(folder.glob(pattern))]
    return pl.concat([f for f in frames if f.width and f.height], how="diagonal_relaxed")


def boot_ratio(
    *, extra_a: np.ndarray, extra_b: np.ndarray, seed: int = 188
) -> tuple[float, float, float]:
    """1 - mean(extra_b) / mean(extra_a) with a 95% interval from resampling events."""
    n = len(extra_a)
    picks = np.random.default_rng(seed).integers(0, n, size=(N_DRAWS, n))
    draws = 1.0 - extra_b[picks].mean(axis=1) / extra_a[picks].mean(axis=1)
    point = 1.0 - extra_b.mean() / extra_a.mean()
    return float(point), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def boot_mean(*, values: np.ndarray, seed: int = 188) -> tuple[float, float, float]:
    """Mean with a 95% interval from resampling events."""
    n = len(values)
    picks = np.random.default_rng(seed).integers(0, n, size=(N_DRAWS, n))
    draws = values[picks].mean(axis=1)
    return float(values.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def fmt(t: tuple[float, float, float]) -> str:
    """Format a point estimate with its interval."""
    return f"{t[0]:+.3f} ({t[1]:+.3f} to {t[2]:+.3f})"


def location_tables() -> None:
    """Location error of the alarm time, the detector's step stamp, and the located onset."""
    for kind, pattern in (("injected", "events_world_*.parquet"), ("real", "events_real.parquet")):
        events = read_all(pattern=pattern, folder=P2).with_columns(
            alarm_time=pl.col("known") - pl.col("true_onset"),
            step=pl.col("onset") - pl.col("true_onset"),
            located_err=pl.col("located") - pl.col("true_onset"),
            wait=pl.col("first_origin") - pl.col("known"),
        )
        print(f"\n== location error, {kind} (half-hours; signed median, median |e|, p90 |e|) ==")
        for family in ("PAIR", "RONC"):
            sub = events.filter(pl.col("family") == family)
            line = f"{kind} {family}: n={sub.height}"
            for col in ("alarm_time", "step", "located_err"):
                v = sub[col].to_numpy().astype(float)
                stats = (
                    float(np.median(v)),
                    float(np.median(np.abs(v))),
                    float(np.quantile(np.abs(v), 0.9)),
                )
                line += f"; {col} {stats[0]:+.1f}/{stats[1]:.1f}/{stats[2]:.1f}"
                REPORT[f"loc|{kind}|{family}|{col}"] = list(stats)
            wait = float(np.mean(sub["wait"].to_numpy()))
            line += f"; mean wait to first origin {wait:.1f}"
            REPORT[f"loc|{kind}|{family}|wait"] = wait
            better = float(np.mean((sub["located_err"].abs() < sub["step"].abs()).to_numpy()))
            worse = float(np.mean((sub["located_err"].abs() > sub["step"].abs()).to_numpy()))
            line += f"; located closer than step {better:.2f}, farther {worse:.2f}"
            print(line)


def per_event(*, scores: pl.DataFrame) -> pl.DataFrame:
    """Mean quantile score of each arm for each (tag, family, site, event), one column per arm."""
    keys = ["tag", "family", "site", "event"]
    wide = (
        scores.group_by([*keys, "arm"])
        .agg(qs=pl.col("qs").mean())
        .pivot(on="arm", index=keys, values="qs")
    )
    err = (
        scores.group_by([*keys, "arm"])
        .agg(ae=pl.col("err_mean").abs().mean())
        .pivot(on="arm", index=keys, values="ae")
        .rename({a: f"ae_{a}" for a in ARMS if a in scores["arm"].unique().to_list()})
    )
    return (
        wide.join(err, on=keys)
        .sort(keys)
        .with_columns(injected=pl.col("tag").str.starts_with("world_"))
    )


def mask_tables(*, events: pl.DataFrame) -> None:
    """Scores, extra errors and the decisive ratio for injected and real transfers."""
    for kind, sub_all in (
        ("injected", events.filter(pl.col("injected"))),
        ("real", events.filter(~pl.col("injected"))),
    ):
        for family in ("PAIR", "RONC", "BOTH"):
            sub = sub_all if family == "BOTH" else sub_all.filter(pl.col("family") == family)
            if sub.height < 2:
                continue
            name = f"{kind} {family}"
            print(f"\n== {name}: {sub.height} found transfers ==")
            for arm in ARMS:
                if arm in sub.columns and sub[arm].null_count() < sub.height:
                    print(f"  mean score {arm}: {sub[arm].mean():.3f}")
                    REPORT[f"score|{name}|{arm}"] = float(np.mean(sub[arm].to_numpy()))
            extra = {
                a: (sub[a] - sub["MASK_TRUE"]).to_numpy()
                for a in ("MASK_ALARM", "MASK_STEP", "MASK_LOCATED", "K0")
            }
            for a, v in extra.items():
                t = boot_mean(values=v)
                print(f"  extra error {a} minus MASK_TRUE: {fmt(t)}")
                REPORT[f"extra|{name}|{a}"] = list(t)
            ratio = boot_ratio(extra_a=extra["MASK_ALARM"], extra_b=extra["MASK_LOCATED"])
            print(f"  C2 reduction in extra error, located vs alarm time: {fmt(ratio)}")
            REPORT[f"C2|{name}"] = list(ratio)
            ratio_step = boot_ratio(extra_a=extra["MASK_ALARM"], extra_b=extra["MASK_STEP"])
            print(f"  same for the detector's step stamp: {fmt(ratio_step)}")
            REPORT[f"C2 step|{name}"] = list(ratio_step)
            for a, b in (
                ("MASK_ALARM", "MASK_LOCATED"),
                ("MASK_LOCATED", "MASK_STEP"),
                ("MASK_LOCATED", "MASK_TRUE"),
            ):
                t = boot_mean(values=(sub[a] - sub[b]).to_numpy())
                print(f"  {a} minus {b}: {fmt(t)}")
                REPORT[f"diff|{name}|{a} - {b}"] = list(t)
            ae = boot_mean(values=(sub["ae_MASK_ALARM"] - sub["ae_MASK_LOCATED"]).to_numpy())
            print(f"  error of the mean forecast, MASK_ALARM minus MASK_LOCATED: {fmt(ae)}")
            REPORT[f"mean error|{name}"] = list(ae)


def decide(*, events: pl.DataFrame) -> None:
    """Apply the pre-registered decision rule to the injected transfers of both families."""
    sub = events.filter(pl.col("injected"))
    extra_a = (sub["MASK_ALARM"] - sub["MASK_TRUE"]).to_numpy()
    extra_b = (sub["MASK_LOCATED"] - sub["MASK_TRUE"]).to_numpy()
    point, lo, hi = boot_ratio(extra_a=extra_a, extra_b=extra_b)
    if point <= 0:
        verdict = "refuted"
    elif lo > 0 and point >= 1 / 3:
        verdict = "supported"
    elif lo > 0:
        verdict = "partly supported"
    else:
        verdict = "inconclusive"
    print(f"\n== DECISION: C2 {point:+.3f} ({lo:+.3f} to {hi:+.3f}); verdict {verdict} ==")
    REPORT["verdict_part2"] = verdict


def sensitivity(*, events: pl.DataFrame) -> None:
    """Post hoc: resample whole worlds, leave one site out, and each site alone."""
    sub = events.filter(pl.col("injected"))
    worlds = sorted(sub["tag"].unique().to_list())
    by_world = {
        w: (
            (
                sub.filter(pl.col("tag") == w)["MASK_ALARM"]
                - sub.filter(pl.col("tag") == w)["MASK_TRUE"]
            ).sum(),
            (
                sub.filter(pl.col("tag") == w)["MASK_LOCATED"]
                - sub.filter(pl.col("tag") == w)["MASK_TRUE"]
            ).sum(),
        )
        for w in worlds
    }
    a = np.array([by_world[w][0] for w in worlds])
    b = np.array([by_world[w][1] for w in worlds])
    picks = np.random.default_rng(188).integers(0, len(worlds), size=(N_DRAWS, len(worlds)))
    draws = 1.0 - b[picks].sum(axis=1) / a[picks].sum(axis=1)
    point = 1.0 - b.sum() / a.sum()
    print(
        f"\nC2 resampling whole worlds ({len(worlds)}): "
        f"{fmt((point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))))}"
    )
    for site in sorted(sub["site"].unique().to_list()):
        for label, part in (
            ("without", sub.filter(pl.col("site") != site)),
            ("only", sub.filter(pl.col("site") == site)),
        ):
            if part.height < 3:
                continue
            ea = (part["MASK_ALARM"] - part["MASK_TRUE"]).to_numpy()
            eb = (part["MASK_LOCATED"] - part["MASK_TRUE"]).to_numpy()
            t = boot_ratio(extra_a=ea, extra_b=eb)
            print(f"  C2 {label} {site}: {fmt(t)} (n={part.height})")
            REPORT[f"C2 {label} {site}"] = list(t)


def tails(*, scores: pl.DataFrame) -> None:
    """Coverage and pinball loss at q01, q05, q95, q99 for the injected rows, by arm."""
    inj = scores.filter(pl.col("tag").str.starts_with("world_"))
    print(
        "\n== tail calibration, injected rows: coverage (share of readings at or below the "
        "quantile) and pinball loss =="
    )
    for arm in ARMS:
        sub = inj.filter(pl.col("arm") == arm)
        if sub.is_empty():
            continue
        line = f"{arm}:"
        for col, tau in TAUS.items():
            diff = (sub["actual"] - sub[col]).to_numpy()
            cover = float(np.mean(diff <= 0))
            pin = float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))
            line += f" {col} cov {cover:.3f} pin {pin:.3f};"
            REPORT[f"tail|{arm}|{col}"] = [cover, pin]
        print(line)


def checks(*, scores: pl.DataFrame) -> None:
    """Median minus mean forecast per site (identical for every arm up to the correction)."""
    print("\n== median minus mean forecast, per site, MASK_LOCATED, all rows ==")
    sub = scores.filter(pl.col("arm") == "MASK_LOCATED")
    for site in sorted(sub["site"].unique().to_list()):
        s = sub.filter(pl.col("site") == site)
        d = float(np.mean((s["err_median"] - s["err_mean"]).to_numpy()))
        print(f"  {site}: {d:+.3f}")
        REPORT[f"median minus mean|{site}"] = d


def main() -> None:
    """Print every number of part 2."""
    location_tables()
    scores = read_all(pattern="*.parquet", folder=SCORES)
    print("\nrows scored:", scores.height, "files:", len(list(SCORES.glob("*.parquet"))))
    events = per_event(scores=scores)
    print("found transfers scored:", events.height)
    mask_tables(events=events)
    decide(events=events)
    sensitivity(events=events)
    tails(scores=scores)
    checks(scores=scores)
    events.write_parquet(WORK_DIR / "p2_event_scores.parquet")
    (WORK_DIR / "report_p2.json").write_text(json.dumps(REPORT, indent=1, default=str))


if __name__ == "__main__":
    main()
