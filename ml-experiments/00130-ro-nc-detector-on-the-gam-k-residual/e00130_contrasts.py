r"""Print every number of the experiment 130 write-up, and the verdict the decision rule gives.

Reads the tables that `e00130_run.py` wrote for each residual source (`A15`, `B`, `C`, and the
sanity source `A16`). The arms are named `<source>_<detector>`, where the detector is `PF14`, `X`,
`LOGIT_SAME`, or `XL` (RO-NC). Arm D is `B_LOGIT_SAME`, the logistic score on one model without the
refit offset. The planned contrast C1 is `B_XL` minus `A15_XL`; C2 is `C_XL` minus `A15_XL`.

    uv run --no-sync python ml-experiments/00130-*/e00130_contrasts.py
"""

from __future__ import annotations

import itertools
import json
from typing import Final

import numpy as np
import polars as pl
from e00130_common import WORK_DIR, stamp_months, stamp_year_months
from e00130_profile import NON_MATCHING, SHAPES

N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 0.95
SEED: Final[int] = 130
THRESHOLD: Final[float] = 0.03
REFUTED_BELOW: Final[float] = 0.02
SOURCES: Final[tuple[str, ...]] = ("A15", "B", "C")
DETECTORS: Final[tuple[str, ...]] = ("PF14", "X", "LOGIT_SAME", "XL")
ARMS: Final[tuple[str, ...]] = tuple(f"{s}_{d}" for s in SOURCES for d in DETECTORS)
STEP_EDGES: Final[tuple[float, ...]] = (0.0, 2.0, 4.0, 6.0, 10.0, float("inf"))
BASE_ARM: Final[str] = "A15_PF14"
BLANK_SERIES: Final[int] = 12
EXPECTED_BASE: Final[int] = 1160


def load(*, source: str) -> pl.DataFrame:
    """One source's event results, with its settings renamed `<source>_<detector>`."""
    return pl.read_parquet(WORK_DIR / source / "event_results.parquet").with_columns(
        setting=pl.lit(f"{source}_") + pl.col("setting"), source=pl.lit(source)
    )


def populations() -> pl.DataFrame:
    """A row per transfer, shape, and arm for the fresh `base` and `long` draws."""
    events = pl.read_parquet(WORK_DIR / "A15" / "events.parquet")
    for source in ("B", "C", "A16"):
        other = pl.read_parquet(WORK_DIR / source / "events.parquet")
        if not other.equals(events):
            raise ValueError(f"source {source} drew different transfers")
    onset = events["onset"].to_numpy()
    month = stamp_months()[onset]
    events = pl.read_parquet(WORK_DIR / "A15" / "events.parquet").with_columns(
        year_month=pl.Series(stamp_year_months()[onset]),
        parity=pl.Series(np.where(month % 2 == 0, "even", "odd")),
    )
    events = events.filter(pl.col("donor") != BLANK_SERIES)
    results = pl.concat([load(source=s) for s in SOURCES])
    weight = (
        results.filter(pl.col("shape") == "S1", pl.col("setting") == BASE_ARM)
        .select("event_id", weight=pl.col("moved_mw"))
        .unique("event_id")
    )
    return results.join(events, on="event_id", validate="m:1").join(
        weight, on="event_id", validate="m:1"
    )


def wide(*, data: pl.DataFrame, shape: str, population: str, parity: str) -> pl.DataFrame:
    """One row per transfer with every arm's `found` (columns named by arm) for one shape."""
    one = data.filter(
        pl.col("shape") == shape, pl.col("population") == population, pl.col("parity") == parity
    )
    frame = one.filter(pl.col("setting") == BASE_ARM).select(
        "event_id",
        "weight",
        "year_month",
        "metered_receivers",
        "donor_step_z",
        "hours",
        **{BASE_ARM: "found", f"onset_{BASE_ARM}": "onset_found"},
    )
    for arm in ARMS:
        if arm == BASE_ARM:
            continue
        part = one.filter(pl.col("setting") == arm)
        frame = frame.join(
            part.select("event_id", **{arm: "found", f"onset_{arm}": "onset_found"}),
            on="event_id",
            validate="1:1",
        )
    return frame.sort("event_id")


def pooled(*, data: pl.DataFrame, population: str, parity: str) -> list[pl.DataFrame]:
    """The three non-matching shapes' wide frames."""
    return [wide(data=data, shape=s, population=population, parity=parity) for s in NON_MATCHING]


def month_sums(*, frame: pl.DataFrame, found: str) -> tuple[np.ndarray, np.ndarray]:
    """Per calendar month: the weight-times-found sum and the weight sum."""
    grouped = (
        frame.with_columns(wf=pl.col("weight") * pl.col(found).cast(pl.Float64))
        .group_by("year_month")
        .agg(pl.col("wf").sum(), pl.col("weight").sum())
        .sort("year_month")
    )
    return grouped["wf"].to_numpy(), grouped["weight"].to_numpy()


def stat(
    *, frames: list[pl.DataFrame], arm: str, base: str | None, picks: np.ndarray | None
) -> np.ndarray:
    """MW-weighted recall of `arm` (minus `base`) over shapes; `picks` resamples months."""
    total = 0.0
    for frame in frames:
        wf_a, w = month_sums(frame=frame, found=arm)
        wf_b = month_sums(frame=frame, found=base)[0] if base else np.zeros_like(wf_a)
        if picks is None:
            total = total + (wf_a.sum() - wf_b.sum()) / w.sum()
        else:
            total = total + (wf_a[picks].sum(axis=1) - wf_b[picks].sum(axis=1)) / w[picks].sum(
                axis=1
            )
    return np.atleast_1d(total / len(frames))


def contrast(
    *, frames: list[pl.DataFrame], arm: str, base: str | None = None
) -> tuple[float, float, float]:
    """Point value and 95% interval of the shape-averaged MW-weighted recall (difference)."""
    months = np.unique(frames[0]["year_month"].to_numpy())
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(months), size=(N_RESAMPLES, len(months)))
    point = float(stat(frames=frames, arm=arm, base=base, picks=None)[0])
    tail = 100 * (1 - LEVEL) / 2
    low, high = np.percentile(
        stat(frames=frames, arm=arm, base=base, picks=picks), [tail, 100 - tail]
    )
    return point, float(low), float(high)


def fmt(point: float, low: float, high: float) -> str:
    """A point value with its interval."""
    return f"{point:+.3f} ({low:+.3f} to {high:+.3f})"


def decide(*, c1: tuple[float, float, float]) -> str:
    """Apply the pre-registered verdict rule to C1 (point, lower end, upper end)."""
    if c1[2] < REFUTED_BELOW:
        return "refuted"
    if c1[1] > THRESHOLD:
        return "supported"
    if c1[1] > 0:
        return "small gain"
    return "inconclusive"


def print_recall(*, data: pl.DataFrame, population: str, parity: str) -> None:
    """Absolute MW-weighted recall per arm with the interval, and the missed shares."""
    frames = pooled(data=data, population=population, parity=parity)
    print(f"\n== {population} transfers, {parity} months: MW-weighted recall (S2, S3, S4 pooled)")
    print(f"transfers per shape: {frames[0].height}; MW weight {frames[0]['weight'].sum():.1f}")
    for arm in ARMS:
        value, low, high = contrast(frames=frames, arm=arm)
        no_start = np.mean(
            [
                float((f["weight"] * (~f[f"onset_{arm}"]).cast(pl.Float64)).sum())
                / float(f["weight"].sum())
                for f in frames
            ]
        )
        print(
            f"{arm:14s} {value:.3f} ({low:.3f} to {high:.3f}) missed {1 - value:.3f}, "
            f"no start {no_start:.3f}"
        )


def print_by_step(*, data: pl.DataFrame) -> None:
    """Transfers found by donor step in noise units, split by metered receiver (S2 to S4 summed)."""
    print("\nTransfers found by donor step (base, even months): n, then found per RO-NC arm")
    for flag, label in ((True, "metered"), (False, "unmetered")):
        for low, high in itertools.pairwise(STEP_EDGES):
            n, counts = 0, dict.fromkeys(ARMS, 0)
            for frame in pooled(data=data, population="base", parity="even"):
                part = frame.filter(
                    (pl.col("metered_receivers") > 0) == flag,
                    pl.col("donor_step_z").abs().is_between(low, high, "left"),
                )
                n += part.height
                for arm in ARMS:
                    counts[arm] += int(part[arm].sum())
            print(
                f"{label:9s} [{low:g},{high:g}) n={n}: "
                + " ".join(f"{a} {counts[a]}" for a in ARMS if a.endswith("XL"))
            )


def print_misses_in_band(*, data: pl.DataFrame) -> None:
    """Share of each RO-NC arm's missed transfers whose donor step is 2 to 4 noise units."""
    print(
        "\nShare of missed transfers with a donor step of 2 to 4 noise units (S2 to S4, base, even)"
    )
    for arm in ("A15_XL", "B_XL", "C_XL"):
        missed = in_band = 0
        for frame in pooled(data=data, population="base", parity="even"):
            miss = frame.filter(~pl.col(arm))
            missed += miss.height
            in_band += miss.filter(pl.col("donor_step_z").abs().is_between(2.0, 4.0, "left")).height
        print(f"  {arm}: {in_band} of {missed} = {in_band / max(missed, 1):.3f}")


def print_real() -> None:
    """The labelled real transfers of more than 1 day, found by each arm."""
    print("\nLabelled real transfers of more than 1 day (descriptive)")
    for source in ("A16", *SOURCES):
        real = pl.read_parquet(WORK_DIR / source / "real_results.parquet").filter(
            pl.col("hours") > 24
        )
        n = real.filter(pl.col("setting") == "PF14").height
        for setting in DETECTORS:
            part = real.filter(pl.col("setting") == setting)
            print(f"  {source}_{setting:10s} found {int(part['found'].sum())} of {n}")


def print_alarms() -> None:
    """Thresholds and unexplained odd-month alarm counts."""
    for source in ("A16", *SOURCES):
        alarms = pl.read_parquet(WORK_DIR / source / "alarms.parquet")
        thresholds = json.loads((WORK_DIR / source / "thresholds.json").read_text())
        print(f"\n{source} thresholds {thresholds}")
        for arm in DETECTORS:
            odd = alarms.filter(
                pl.col("setting") == arm, ~pl.col("logged_matched"), pl.col("month") % 2 == 1
            ).height
            print(f"  {arm}: odd-month unexplained alarms {odd}")


def sanity_a16() -> None:
    """Experiment 82's all-16-series arms on its own draw: RO-NC 0.770, PF14 0.656 expected."""
    results = load(source="A16")
    events = pl.read_parquet(WORK_DIR / "A16" / "events.parquet")
    onset = events["onset"].to_numpy()
    month = stamp_months()[onset]
    events = events.with_columns(
        year_month=pl.Series(stamp_year_months()[onset]),
        parity=pl.Series(np.where(month % 2 == 0, "even", "odd")),
    )
    weight = (
        results.filter(pl.col("shape") == "S1", pl.col("setting") == "A16_PF14")
        .select("event_id", weight=pl.col("moved_mw"))
        .unique("event_id")
    )
    data = results.join(events, on="event_id").join(weight, on="event_id")
    print("\nSanity: all 16 series, base even months (experiment 82: RO-NC 0.770, PF14 0.656)")
    for detector in ("PF14", "XL"):
        frames = []
        for shape in NON_MATCHING:
            one = data.filter(
                pl.col("shape") == shape,
                pl.col("population") == "base",
                pl.col("parity") == "even",
                pl.col("setting") == f"A16_{detector}",
            ).select("event_id", "weight", "year_month", found="found")
            frames.append(one)
        value = np.mean(
            [
                float((f["weight"] * f["found"].cast(pl.Float64)).sum() / f["weight"].sum())
                for f in frames
            ]
        )
        print(f"  A16_{detector}: {value:.3f}")


def main() -> None:
    """Print the tables and the verdict."""
    data = populations()
    if data.filter(pl.col("population") == "base")["event_id"].n_unique() < 0.9 * EXPECTED_BASE:
        raise ValueError("fewer base transfers than expected")
    sanity_a16()
    for population, parity in itertools.product(("base", "long"), ("even", "odd")):
        print_recall(data=data, population=population, parity=parity)
    base = pooled(data=data, population="base", parity="even")
    long = pooled(data=data, population="long", parity="even")
    c1 = contrast(frames=base, arm="B_XL", base="A15_XL")
    print("\nPlanned contrast C1, arm B minus arm A, RO-NC (95% interval, even months)")
    print(f"  C1: {fmt(*c1)}")
    print("Descriptive contrasts")
    for label, frames, arm, ref in (
        ("base, C2 arm C minus arm A", base, "C_XL", "A15_XL"),
        ("base, arm B minus arm C", base, "B_XL", "C_XL"),
        (
            "base, arm D minus arm B (logistic, one model, minus RO-NC)",
            base,
            "B_LOGIT_SAME",
            "B_XL",
        ),
        ("base, arm D minus arm A PF14", base, "B_LOGIT_SAME", "A15_PF14"),
        ("long, arm B minus arm A", long, "B_XL", "A15_XL"),
        ("long, arm C minus arm A", long, "C_XL", "A15_XL"),
    ):
        print(f"  {label}: {fmt(*contrast(frames=frames, arm=arm, base=ref))}")
    print("Arm B minus arm A by shape, base, even months")
    for shape in SHAPES:
        frame = wide(data=data, shape=shape, population="base", parity="even")
        print(f"  {shape}: {fmt(*contrast(frames=[frame], arm='B_XL', base='A15_XL'))}")
    print_by_step(data=data)
    print_misses_in_band(data=data)
    print_real()
    print_alarms()
    print(f"VERDICT: {decide(c1=c1)}")


if __name__ == "__main__":
    main()
