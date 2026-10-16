"""Print every number the write-up of experiment 65 quotes, and the verdict the decision rule gives.

Reads the tables that `e00065_run.py` wrote: MW-weighted recall of every arm under both end rules
for each injection shape, the six planned contrasts (LR and LOGIT against D0M, D1, and D2, with
the PE end rule) with month-resampled intervals pooled over S2, S3, and S4, recall by donor step
split by metered receiver, the labelled real transfers, alarm counts, and the fitted scores.

    uv run python ml-experiments/00065-*/e00065_report.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00065_common import ML_EXPERIMENTS_DATA_DIR, WORK_DIR, stamp_months, stamp_year_months
from e00065_inject import HEADLINE_BAND
from e00065_profile import NON_MATCHING, SHAPE_NAMES, SHAPES
from e00065_run import ARMS, RULES, setting

EXPERIMENT_58_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00058-sw2-8b-donor-step-recut-and-stronger-veto"
)
N_RESAMPLES: Final[int] = 20_000
N_CONTRASTS: Final[int] = 6
LEVEL: Final[float] = 1 - 0.05 / N_CONTRASTS
"""Two-sided level of the intervals: Bonferroni across the six planned contrasts (99.17%)."""
SEED: Final[int] = 6500
SUPPORT_AT_LEAST: Final[float] = 0.02
ALARM_MARGIN: Final[int] = 10
STEP_EDGES: Final[tuple[float, ...]] = (0.0, 2.5, 4.0, 6.0, 10.0, float("inf"))
VERSIONS: Final[tuple[str, ...]] = ("LR", "LOGIT")
BENCHMARKS: Final[tuple[str, ...]] = ("D0M", "D1", "D2")
POST_HOC: Final[tuple[str, ...]] = ("LOGIT_SAME", "D2V1")


def population(*, out: Path) -> pl.DataFrame:
    """The 620 headline-band even-month transfers: a row per transfer, shape, and setting.

    The weight of a transfer is its mean moved MW under S1.
    """
    events = pl.read_parquet(EXPERIMENT_58_DIR / "events.parquet")
    onset = events["onset"].to_numpy()
    events = events.with_columns(
        month=pl.Series(stamp_months()[onset]),
        year_month=pl.Series(stamp_year_months()[onset]),
        headline=(pl.col("size_band") == HEADLINE_BAND[0])
        & (pl.col("duration_band") == HEADLINE_BAND[1]),
    ).filter(pl.col("headline"), pl.col("month") % 2 == 0, pl.col("set") != "holiday")
    results = pl.read_parquet(out / "event_results.parquet")
    weight = (
        results.filter(pl.col("shape") == "S1", pl.col("setting") == setting(arm="D0M", rule="TR"))
        .select("event_id", weight=pl.col("moved_mw"))
        .unique("event_id")
    )
    return (
        results.join(events, on="event_id", validate="m:1")
        .join(weight, on="event_id", validate="m:1")
        .sort("shape", "setting", "event_id")
    )


def wide(*, data: pl.DataFrame, shape: str) -> pl.DataFrame:
    """One row per transfer with every setting's `found` as a column named by the setting."""
    one = data.filter(pl.col("shape") == shape)
    base = one.filter(pl.col("setting") == setting(arm="D0M", rule="TR")).select(
        "event_id", "weight", "year_month", "metered_receivers", "donor_step_z"
    )
    found = one.pivot(on="setting", index="event_id", values="found")
    return base.join(found, on="event_id", validate="1:1").sort("event_id")


def interval(*, resampled: np.ndarray) -> tuple[float, float]:
    """The two-sided interval of resampled statistics at `LEVEL`."""
    tail = 100 * (1 - LEVEL) / 2
    low, high = np.percentile(resampled, [tail, 100 - tail])
    return float(low), float(high)


def month_sums(*, frame: pl.DataFrame, found: str) -> tuple[np.ndarray, np.ndarray]:
    """Per calendar month: the weight-times-found sum and the weight sum."""
    grouped = (
        frame.with_columns(wf=pl.col("weight") * pl.col(found).cast(pl.Float64))
        .group_by("year_month")
        .agg(pl.col("wf").sum(), pl.col("weight").sum())
        .sort("year_month")
    )
    return grouped["wf"].to_numpy(), grouped["weight"].to_numpy()


def diffs(
    *, frames: list[pl.DataFrame], arm: str, base: str, picks: np.ndarray | None
) -> np.ndarray:
    """MW-weighted recall of `arm` minus `base`, averaged over shapes; `picks` resamples months."""
    total = 0.0
    for frame in frames:
        wf_a, w = month_sums(frame=frame, found=arm)
        wf_b, _ = month_sums(frame=frame, found=base)
        if picks is None:
            total = total + (wf_a.sum() - wf_b.sum()) / w.sum()
        else:
            total = total + (wf_a[picks].sum(axis=1) - wf_b[picks].sum(axis=1)) / w[picks].sum(
                axis=1
            )
    return np.atleast_1d(total / len(frames))


def contrast(*, frames: list[pl.DataFrame], arm: str, base: str) -> tuple[float, float, float]:
    """Point value and interval of the shape-averaged MW-weighted recall difference."""
    months = np.unique(frames[0]["year_month"].to_numpy())
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(months), size=(N_RESAMPLES, len(months)))
    point = float(diffs(frames=frames, arm=arm, base=base, picks=None)[0])
    low, high = interval(resampled=diffs(frames=frames, arm=arm, base=base, picks=picks))
    return point, low, high


def fmt(point: float, low: float, high: float) -> str:
    """A point value with its interval."""
    return f"{point:+.3f} ({low:+.3f} to {high:+.3f})"


def recall(*, frame: pl.DataFrame, column: str) -> tuple[float, float]:
    """MW-weighted and unweighted recall of one setting."""
    w, f = frame["weight"].to_numpy(), frame[column].to_numpy().astype(float)
    return float((w * f).sum() / w.sum()), float(f.mean())


def print_shapes(*, data: pl.DataFrame) -> None:
    """Recall of every setting for each shape, with the share of the moved MW missed."""
    print("\nMW-weighted recall | unweighted recall (even months); missed MW = 1 - first number")
    for shape in SHAPES:
        frame = wide(data=data, shape=shape)
        print(f"{shape} {SHAPE_NAMES[shape]}, n={frame.height}")
        for rule in RULES:
            cells = []
            for arm in ARMS:
                weighted, unweighted = recall(frame=frame, column=setting(arm=arm, rule=rule))
                cells.append(f"{arm} {weighted:.3f}|{unweighted:.3f}")
            print(f"   {rule}: " + "  ".join(cells))
    pooled = [wide(data=data, shape=s) for s in NON_MATCHING]
    for rule in RULES:
        cells = []
        for arm in ARMS:
            value = np.mean(
                [recall(frame=f, column=setting(arm=arm, rule=rule))[0] for f in pooled]
            )
            cells.append(f"{arm} {value:.3f} (missed {1 - value:.3f})")
        print(f"pooled S2-S4 {rule}: " + "  ".join(cells))


def planned_contrasts(*, data: pl.DataFrame, rule: str) -> dict[str, tuple[float, float, float]]:
    """The six planned contrasts (and the post hoc ones) under one end rule."""
    frames = [wide(data=data, shape=s) for s in NON_MATCHING]
    out = {}
    for arm in (*VERSIONS, *POST_HOC):
        for base in BENCHMARKS:
            if arm == "D2V1" and base == "D2":
                continue
            out[f"{arm}-{base}"] = contrast(
                frames=frames,
                arm=setting(arm=arm, rule=rule),
                base=setting(arm=base, rule=rule),
            )
    return out


def print_by_step(*, data: pl.DataFrame) -> None:
    """Standard table: found/transfers by donor step in noise units, split by metered receiver."""
    print(
        "\nFound / transfers by donor step in noise units (PE end rule); arms: " + " | ".join(ARMS)
    )
    for shape in SHAPES:
        frame = wide(data=data, shape=shape)
        for flag, label in ((True, "metered"), (False, "unmetered")):
            part = frame.filter((pl.col("metered_receivers") > 0) == flag)
            cells = []
            for low, high in itertools.pairwise(STEP_EDGES):
                bin_ = part.filter(pl.col("donor_step_z").abs().is_between(low, high, "left"))
                counts = "|".join(str(bin_[setting(arm=a, rule="PE")].sum()) for a in ARMS)
                cells.append(f"[{low:g},{high:g}) {counts}/{bin_.height}")
            print(f"{shape} {label:9s} " + "; ".join(cells))


def print_real(*, out: Path) -> None:
    """The labelled real transfers of more than 1 day, found by each setting, by onset parity."""
    real = pl.read_parquet(out / "real_results.parquet").filter(pl.col("hours") > 24)
    n = real.filter(pl.col("setting") == setting(arm="D0M", rule="TR")).height
    print(f"\nLabelled real transfers of more than 1 day: {n}")
    for rule in RULES:
        for arm in ARMS:
            part = real.filter(pl.col("setting") == setting(arm=arm, rule=rule))
            even = part.filter(pl.col("month") % 2 == 0)
            odd = part.filter(pl.col("month") % 2 == 1)
            print(
                f"  {arm}_{rule}: found {part['found'].sum()}, onset found "
                f"{part['onset_found'].sum()}; even-month onsets "
                f"{even['found'].sum()}/{even.height}, odd {odd['found'].sum()}/{odd.height}"
            )


def print_alarms(*, out: Path) -> dict[str, int]:
    """Unexplained alarm counts on the real residual by even and odd month."""
    alarms = pl.read_parquet(out / "alarms.parquet")
    print(f"\nthresholds {json.loads((out / 'thresholds.json').read_text())}")
    counts = {}
    for arm in ARMS:
        part = alarms.filter(
            pl.col("setting") == setting(arm=arm, rule="PE"), ~pl.col("logged_matched")
        )
        counts[arm] = part.filter(pl.col("month") % 2 == 0).height
        print(
            f"{arm}: unexplained alarms even months {counts[arm]}, "
            f"odd {part.filter(pl.col('month') % 2 == 1).height}"
        )
    return counts


def print_models(*, out: Path) -> None:
    """The fitted log-likelihood ratios and logistic coefficients."""
    models = pickle.loads((out / "models.pkl").read_bytes())
    print("\nLR x1 bins (<3.0, 3.0..., ...):", np.round(models["LR"].x1, 2).tolist())
    print("LR x2 bins:", np.round(models["LR"].x2, 2).tolist())
    for name in ("LOGIT", "LOGIT_SAME"):
        m = models[name]
        print(
            name,
            dict(zip(m.features, np.round(m.coef, 3).tolist(), strict=True)),
            round(m.intercept, 3),
        )


def verdict(*, version: str, pooled: dict, alarms: dict[str, int]) -> str:
    """Apply the pre-registered decision rule to one version."""
    point, low, high = pooled[f"{version}-D0M"]
    if high < SUPPORT_AT_LEAST:
        return "refuted"
    beats_gate_and_veto = all(pooled[f"{version}-{b}"][1] > 0 for b in ("D1", "D2"))
    quiet = alarms[version] <= alarms["D0M"] + ALARM_MARGIN
    if low > 0 and point >= SUPPORT_AT_LEAST and beats_gate_and_veto and quiet:
        return "supported"
    if beats_gate_and_veto and low <= 0:
        return "partly supported"
    return "inconclusive"


def main() -> None:
    """Print the tables and the verdict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    out = parser.parse_args().out
    data = population(out=out)
    one = data.filter(pl.col("shape") == "S1", pl.col("setting") == "D0M_TR")
    print(f"{one.height} transfers; {one.filter(pl.col('weight') <= 0).height} move no MW")
    print_shapes(data=data)
    results = {}
    for rule in RULES:
        results[rule] = planned_contrasts(data=data, rule=rule)
        print(f"\nContrasts pooled over {NON_MATCHING}, end rule {rule}, level {LEVEL:.4f}:")
        for name, value in results[rule].items():
            post_hoc = " [post hoc, outside the verdict]" if name.split("-")[0] in POST_HOC else ""
            print(f"  {name}: {fmt(*value)}{post_hoc}")
    print_by_step(data=data)
    print_real(out=out)
    alarms = print_alarms(out=out)
    print_models(out=out)
    verdicts = {v: verdict(version=v, pooled=results["PE"], alarms=alarms) for v in VERSIONS}
    rank = ["refuted", "inconclusive", "partly supported", "supported"]
    best = max(verdicts.values(), key=rank.index)
    print(f"\nVERDICT per version: {verdicts}")
    print(
        f"EXPERIMENT VERDICT (more favourable version; outside the Bonferroni correction): {best}"
    )


if __name__ == "__main__":
    main()
