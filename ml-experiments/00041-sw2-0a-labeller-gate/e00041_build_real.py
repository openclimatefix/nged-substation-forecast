"""Build the SW2-0b cards: real logged operations, edge events, short closures, detector alarms.

The cards also hold hidden known-answer cards (placebo and injected) built from the same operation
table, so that a labeller cannot tell the two kinds apart.

    uv run python ml-experiments/00041-sw2-0a-labeller-gate/e00041_build_real.py
"""

from __future__ import annotations

import json
import pickle
import random
import secrets
from collections import Counter
from datetime import datetime, timedelta
from typing import cast

import numpy as np
import polars as pl
from e00041_build_sets import hourly_table
from e00041_cards import (
    QUIET_RULE,
    apply_injection,
    make_injection,
    make_placebo,
    render_card,
    revert_injection,
)
from e00041_common import WORK_DIR
from e00041_features import CardSpec, compute_features
from e00041_panel import INDEX, N_STAMPS, PAD_STAMPS, Panel, load_panel, time_of
from e00041_rules import label_card

INTERVALS = WORK_DIR.parent / "00037-switching-first-look" / "intervals.parquet"
ALARM_FILES = ("alarms_D0_8_6.0", "alarms_D1_8_3.75", "alarms_D0_48_4.0", "alarms_D1_48_2.5")
OUT = WORK_DIR / "real"
N_ALARMS = 150
N_ALARM_PLACEBOS = 40
N_SHORT = 50
N_SENTINEL_PLACEBO = 100
N_SENTINEL_INJECTED = 80
N_SENTINEL_JITTERED = 20
EDGE_HOURS = 72
SEED = 4100


def _sides(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        a=pl.col("side_a").cast(pl.Int64, strict=False),
        b=pl.col("side_b").cast(pl.Int64, strict=False),
    )


def operation_table(*, min_hours: float, max_hours: float) -> pl.DataFrame:
    """One row per logged operation: first close, last open, and the two commonest metered sides."""
    frame = _sides(pl.read_parquet(INTERVALS)).filter(pl.col("end_utc").is_not_null())
    rows = []
    for _, group in frame.group_by("operation"):
        start = cast("datetime", group["start_utc"].min())
        end = cast("datetime", group["end_utc"].max())
        hours = (end - start).total_seconds() / 3600
        if not min_hours <= hours <= max_hours:
            continue
        counts = Counter(
            s
            for s in (*group["a"].to_list(), *group["b"].to_list())
            if s is not None and s in INDEX
        )
        top = [s for s, _ in counts.most_common(2)]
        if not top:
            continue
        rows.append(
            {
                "operation": int(group["operation"][0]),
                "a": top[0],
                "b": top[1] if len(top) > 1 else None,
                "start_utc": start,
                "end_utc": end,
                "hours": hours,
            }
        )
    return pl.DataFrame(rows, schema_overrides={"b": pl.Int64})


def _interval_rows(*, min_hours: float, max_hours: float) -> list[dict]:
    frame = _sides(pl.read_parquet(INTERVALS)).filter(
        pl.col("end_utc").is_not_null(), pl.col("hours").is_between(min_hours, max_hours)
    )
    out = []
    for row in frame.iter_rows(named=True):
        sides = [s for s in (row["a"], row["b"]) if s is not None and s in INDEX]
        if sides:
            out.append({**row, "a": sides[0], "b": sides[1] if len(sides) > 1 else None})
    return out


def _spec(*, a: int, b: int | None, close_utc, open_utc) -> CardSpec:  # noqa: ANN001
    return CardSpec(
        card_id=secrets.token_hex(5), side_a=a, side_b=b, close_utc=close_utc, open_utc=open_utc
    )


def _alarm_specs(
    *, panel: Panel, rng: random.Random, durations: list[float]
) -> list[tuple[CardSpec, dict]]:
    """Alarm cards: the alarm's stamp stands in for the log time and a real duration is drawn."""
    per_file = N_ALARMS // len(ALARM_FILES)
    out: list[tuple[CardSpec, dict]] = []
    base = WORK_DIR.parent / "00037-switching-first-look"
    for name in ALARM_FILES:
        alarms = pl.read_parquet(base / f"{name}.parquet").filter(
            pl.col("series").is_in(list(INDEX))
        )
        keep = [
            r
            for r in alarms.iter_rows(named=True)
            if 5 * 336 < r["stamp"] < N_STAMPS - 400
            and not panel.excluded[INDEX[r["series"]], r["stamp"]]
        ]
        rng.shuffle(keep)
        for r in keep[:per_file]:
            hours = rng.choice(durations)
            close = time_of(stamp=r["stamp"] - 1) - timedelta(minutes=rng.uniform(1, 29))
            partner = str(r["partners"]).split(":")[0].split(",")[0].strip()
            b = int(partner) if partner.isdigit() and int(partner) in INDEX else None
            spec = _spec(
                a=r["series"], b=b, close_utc=close, open_utc=close + timedelta(hours=hours)
            )
            out.append((spec, {"kind": "alarm", "detector": name, "alarm_stamp": int(r["stamp"])}))
    return out


def _alarm_placebo_specs(
    *, panel: Panel, rng: random.Random, durations: list[float], alarm_stamps: dict[int, np.ndarray]
) -> list[tuple[CardSpec, dict]]:
    out: list[tuple[CardSpec, dict]] = []
    series = list(INDEX)
    while len(out) < N_ALARM_PLACEBOS:
        sid = rng.choice(series)
        stamp = rng.randrange(5 * 336, N_STAMPS - 400)
        row = INDEX[sid]
        near = alarm_stamps.get(sid, np.array([], dtype=int))
        if panel.excluded[row, stamp - 96 : stamp + 400].any() or (
            len(near) and np.abs(near - stamp).min() < 96
        ):
            continue
        hours = rng.choice(durations)
        close = time_of(stamp=stamp - 1) - timedelta(minutes=rng.uniform(1, 29))
        spec = _spec(a=sid, b=None, close_utc=close, open_utc=close + timedelta(hours=hours))
        out.append((spec, {"kind": "alarm_placebo"}))
    return out


NAN_BINS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.7, 1.5, 3.0, 6.0, 12.0, 1e9)
"""Edges of the unavailable-value fraction bins that sentinel cards are matched to."""
CANDIDATES_PER_SENTINEL = 6


def _na_per_row(text: str) -> float:
    """Unavailable values per hourly row of a rendered card (what sentinels are matched on)."""
    rows = [line for line in text.splitlines() if line.startswith("| ") and "UTC" not in line]
    return text.count("n/a") / max(len(rows), 1)


def _sentinels(
    *, panel: Panel, ops: pl.DataFrame, real_fractions: list[float], nprng: np.random.Generator
) -> list[tuple[CardSpec, dict, dict | None, tuple[str, dict, object]]]:
    """Draw sentinel cards whose unavailable values per row follow the real operations'."""
    QUIET_RULE["relaxed"] = True
    kinds = (
        ["placebo"] * N_SENTINEL_PLACEBO
        + ["injected"] * N_SENTINEL_INJECTED
        + ["jittered"] * N_SENTINEL_JITTERED
    )
    out: list[tuple[CardSpec, dict, dict | None, tuple[str, dict, object]]] = []
    real_hist, _ = np.histogram(real_fractions, bins=NAN_BINS)
    real_p = real_hist / real_hist.sum()
    for kind in sorted(set(kinds)):
        wanted = kinds.count(kind)
        pool = []
        while len(pool) < wanted * CANDIDATES_PER_SENTINEL:
            if kind == "placebo":
                m = make_placebo(panel=panel, ops=ops, rng=nprng)
                made = None if m is None else (m[0], m[1], None)
            else:
                made = make_injection(panel=panel, ops=ops, rng=nprng, jitter=(kind == "jittered"))
            if made is None:
                continue
            spec, truth, inj = made
            built = _build_card(panel=panel, spec=spec, inj=inj, nprng=nprng)
            pool.append(
                (
                    spec,
                    {"kind": f"sentinel_{kind}", "truth": truth.__dict__},
                    inj,
                    built,
                    _na_per_row(built[0]),
                )
            )
        bins = np.digitize([c[4] for c in pool], NAN_BINS[1:-1])
        cand_p = np.bincount(bins, minlength=len(real_p)) / len(pool)
        weights = np.array([real_p[b] / max(cand_p[b], 1e-9) for b in bins])
        pick = nprng.choice(len(pool), size=wanted, replace=False, p=weights / weights.sum())
        out += [(pool[i][0], pool[i][1], pool[i][2], pool[i][3]) for i in pick]
    QUIET_RULE["relaxed"] = False
    return out


def _build_card(
    *, panel: Panel, spec: CardSpec, inj: dict | None, nprng: np.random.Generator
) -> tuple[str, dict, object]:
    rows = [INDEX[s] for s in (spec.side_a, spec.side_b) if s in INDEX]
    if inj is not None:
        rows = sorted(set(rows) | {INDEX[inj["donor"]]} | {INDEX[r] for r in inj["receivers"]})
    saved = panel.excluded[rows].copy()
    avoid = np.zeros(N_STAMPS, dtype=bool)
    avoid[max(spec.first - PAD_STAMPS - 48, 0) : min(spec.last + PAD_STAMPS + 48, N_STAMPS)] = True
    panel.excluded[
        rows, max(spec.first - PAD_STAMPS, 0) : min(spec.last + PAD_STAMPS + 1, N_STAMPS)
    ] = True
    if inj is not None:
        apply_injection(panel=panel, inj=inj)
    features = compute_features(
        panel=panel, spec=spec, avoid=avoid, seed=int(nprng.integers(1 << 30))
    )
    label = label_card(panel=panel, spec=spec, features=features)
    text = render_card(spec=spec, features=features, panel=panel) + hourly_table(
        panel=panel, spec=spec, features=features
    )
    if inj is not None:
        revert_injection(panel=panel, inj=inj)
    panel.excluded[rows] = saved
    return text, label, features


def main() -> None:
    """Build every card, write them with their hidden metadata, and cut batches of ten."""
    rng = random.Random(SEED)
    nprng = np.random.default_rng(SEED)
    panel = load_panel()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cards").mkdir(exist_ok=True)
    ops = operation_table(min_hours=2, max_hours=60 * 24)
    sentinel_ops = ops.filter(pl.col("hours") <= 30 * 24)
    durations = ops["hours"].to_list()
    todo: list[tuple[CardSpec, dict, dict | None]] = []
    todo.extend(
        (
            _spec(a=r["a"], b=r["b"], close_utc=r["start_utc"], open_utc=r["end_utc"]),
            {"kind": "real_operation", "operation": r["operation"]},
            None,
        )
        for r in ops.iter_rows(named=True)
    )
    for r in _interval_rows(min_hours=60 * 24, max_hours=1e9):
        edge = timedelta(hours=EDGE_HOURS)
        todo.append(
            (
                _spec(a=r["a"], b=r["b"], close_utc=r["start_utc"], open_utc=r["start_utc"] + edge),
                {"kind": "real_edge_close", "operation": r["operation"]},
                None,
            )
        )
        todo.append(
            (
                _spec(a=r["a"], b=r["b"], close_utc=r["end_utc"] - edge, open_utc=r["end_utc"]),
                {"kind": "real_edge_open", "operation": r["operation"]},
                None,
            )
        )
    shorts = _interval_rows(min_hours=0.5, max_hours=2)
    rng.shuffle(shorts)
    todo.extend(
        (
            _spec(a=r["a"], b=r["b"], close_utc=r["start_utc"], open_utc=r["end_utc"]),
            {"kind": "real_short", "operation": r["operation"]},
            None,
        )
        for r in shorts[:N_SHORT]
    )
    alarm_stamps: dict[int, np.ndarray] = {}
    for name in ALARM_FILES:
        frame = pl.read_parquet(WORK_DIR.parent / "00037-switching-first-look" / f"{name}.parquet")
        for sid, g in frame.group_by("series"):
            alarm_stamps[int(sid[0])] = np.concatenate(
                [alarm_stamps.get(int(sid[0]), np.array([], dtype=int)), g["stamp"].to_numpy()]
            )
    todo.extend(
        (spec, meta, None) for spec, meta in _alarm_specs(panel=panel, rng=rng, durations=durations)
    )
    todo.extend(
        (spec, meta, None)
        for spec, meta in _alarm_placebo_specs(
            panel=panel, rng=rng, durations=durations, alarm_stamps=alarm_stamps
        )
    )
    n_real = len(todo)
    prebuilt: dict[str, tuple[str, dict, object]] = {}
    for spec, _, inj in todo:
        prebuilt[spec.card_id] = _build_card(panel=panel, spec=spec, inj=inj, nprng=nprng)
    real_fractions = [
        _na_per_row(prebuilt[spec.card_id][0])
        for spec, meta, _ in todo
        if meta["kind"] == "real_operation"
    ]
    for spec, meta, inj, built in _sentinels(
        panel=panel, ops=sentinel_ops, real_fractions=real_fractions, nprng=nprng
    ):
        todo.append((spec, meta, inj))
        prebuilt[spec.card_id] = built
    rng.shuffle(todo)
    _write_outputs(todo=todo, prebuilt=prebuilt, n_real=n_real)


def _write_outputs(
    *,
    todo: list[tuple[CardSpec, dict, dict | None]],
    prebuilt: dict[str, tuple[str, dict, object]],
    n_real: int,
) -> None:
    """Write each card, its hidden metadata and labels, the features pickle, and the batches."""
    meta_all: dict[str, dict] = {}
    rules: dict[str, dict] = {}
    feats: dict[str, object] = {}
    specs: dict[str, object] = {}
    for i, (spec, meta, _) in enumerate(todo):
        text, label, features = prebuilt[spec.card_id]
        (OUT / "cards" / f"{spec.card_id}.md").write_text(text)
        meta_all[spec.card_id] = meta
        rules[spec.card_id] = label
        feats[spec.card_id] = features
        specs[spec.card_id] = spec
        if i % 50 == 0:
            print(i, len(todo), meta["kind"], flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta_all, indent=1, default=float))
    (OUT / "rules_labels.json").write_text(json.dumps(rules, indent=1, default=float))
    with (OUT / "features.pkl").open("wb") as handle:
        pickle.dump({"features": feats, "specs": specs}, handle)
    ids = list(meta_all)
    random.Random(SEED + 1).shuffle(ids)
    (OUT / "batches").mkdir(exist_ok=True)
    for i in range(0, len(ids), 10):
        lines = [str((OUT / "cards" / f"{c}.md").resolve()) for c in ids[i : i + 10]]
        (OUT / "batches" / f"batch_{i // 10:03d}.txt").write_text("\n".join(lines) + "\n")
    print("cards", len(ids), "real", n_real, "sentinels", len(ids) - n_real)


if __name__ == "__main__":
    main()
