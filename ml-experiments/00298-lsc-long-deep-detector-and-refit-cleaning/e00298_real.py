# ruff: noqa: E501, PERF401, B007
"""Part 1 on the real record: tune each variant on odd months, score on even months.

For each variant family the threshold is the smallest `k` that gives at most 2 false alarms per
Primary-year in odd months (D8 and D9 excluded from the pooled rate). Recall is read on all 17 block
days, all 29 D8 drops and all labelled real transfers over 1 day. False alarms are counted in even
months.

    uv run --no-sync python e00298_real.py > printed_real.txt
"""

from __future__ import annotations

import json
from typing import Final

import e00298_common as common
import e00298_detect as dt
import e00298_events as ev
import numpy as np
import polars as pl

BUDGET: Final[float] = 2.0
EXCLUDE_DAYS: Final[int] = 3
K_GRID: Final[tuple[float, ...]] = (
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    12.0,
    15.0,
    20.0,
)
FAMILIES: Final[tuple[tuple[str, dict], ...]] = (
    ("A2", {"kind": "A", "w": 2}),
    ("A3", {"kind": "A", "w": 3}),
    ("A7", {"kind": "A", "w": 7}),
    ("B", {"kind": "B"}),
    ("C", {"kind": "C"}),
    ("D-A3", {"kind": "D", "w": 3, "base": "A"}),
    ("D-A7", {"kind": "D", "w": 7, "base": "A"}),
    ("D-C", {"kind": "D", "base": "C"}),
    ("E3", {"kind": "E", "w": 3}),
    ("E7", {"kind": "E", "w": 7}),
    ("F-A3", {"kind": "F", "w": 3, "base": "A"}),
    ("F-A7", {"kind": "F", "w": 7, "base": "A"}),
    ("F-C", {"kind": "F", "base": "C"}),
)
FIRST_EVAL: Final[int] = ev.stamp_day(stamp=0)
LAST_EVAL: Final[int] = dt.N_DAYS - 1
SPECIAL: Final[tuple[str, ...]] = ("D8", "D9")


def build_streams() -> dict[str, dt.Stream]:
    """Return a stream per Primary from the clean panel."""
    panel = pl.read_parquet(common.PANEL_PATH)
    arrays = dt.panel_arrays(panel=panel)
    return {
        s: dt.make_stream(residual=a["residual"], level=dt.rolling_level(power=a["power"]))
        for s, a in arrays.items()
    }


def event_mask(*, events: list[dict], site: str, pad: int) -> np.ndarray:
    """True on grid days within `pad` of any known event that involves `site`."""
    mask = np.zeros(dt.N_DAYS, dtype=bool)
    for e in events:
        if site in e["sites"]:
            mask[max(0, e["start"] - pad) : e["end"] + pad + 1] = True
    return mask


def run_family(*, streams: dict[str, dt.Stream], variant: dt.Variant) -> dict[str, list[dt.Span]]:
    """Return the spans of `variant` at every Primary over the evaluation window."""
    out = {}
    for site, stream in streams.items():
        others = {n: streams[n] for n in common.NEIGHBOURS.get(site, ()) if n in streams}
        out[site] = dt.detect(
            variant=variant, stream=stream, others=others, first=FIRST_EVAL, last=LAST_EVAL
        )
    return out


def classify(
    *, spans: dict[str, list[dt.Span]], events: list[dict], closures: list[dict]
) -> pl.DataFrame:
    """Label every span as a false alarm (strict), a false alarm with no logged closure, or a hit."""
    rows = []
    for site, items in spans.items():
        known = event_mask(events=events, site=site, pad=EXCLUDE_DAYS)
        logged = np.zeros(dt.N_DAYS, dtype=bool)
        for c in closures:
            if site in c["sites"]:
                logged[max(0, c["start"] - 1) : c["end"] + 2] = True
        for s in items:
            near_event = bool(known[s.start : s.end + 1].any())
            near_log = bool(logged[s.start : s.end + 1].any())
            rows.append(
                {
                    "site": site,
                    "start": s.start,
                    "end": s.end,
                    "fire": s.fire,
                    "month": ev.month_of_day(day=s.fire),
                    "false_strict": not near_event,
                    "false_unlogged": (not near_event) and (not near_log),
                    "explained": s.explained,
                    "size_mw": s.size_mw,
                }
            )
    schema = {
        "site": pl.String,
        "start": pl.Int64,
        "end": pl.Int64,
        "fire": pl.Int64,
        "month": pl.Int64,
        "false_strict": pl.Boolean,
        "false_unlogged": pl.Boolean,
        "explained": pl.Float64,
        "size_mw": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def primary_years(*, streams: dict[str, dt.Stream], odd: bool, sites: list[str]) -> float:
    """Days with a valid z-score in odd (or even) months, over the evaluation window, in years."""
    months = np.array([ev.month_of_day(day=d) for d in range(dt.N_DAYS)])
    in_part = (months % 2 == 1) if odd else (months % 2 == 0)
    total = 0
    for s in sites:
        valid = np.isfinite(streams[s].z)
        valid[:FIRST_EVAL] = False
        total += int((valid & in_part).sum())
    return total / 365.25


def recall(
    *, spans: dict[str, list[dt.Span]], events: list[dict], cls: str
) -> tuple[int, int, list[dict]]:
    """Return (found, total, per-event detail) for the events of class `cls`."""
    found, detail = 0, []
    items = [e for e in events if e["cls"] == cls]
    for e in items:
        hit = None
        for site in e["sites"]:
            for s in spans.get(site, []):
                if s.start <= e["end"] + 1 and s.end >= e["start"] - 1:
                    hit = s if hit is None or s.fire < hit.fire else hit
        detail.append(
            {
                "cls": cls,
                "site": e["sites"][0],
                "start": e["start"],
                "end": e["end"],
                "found": hit is not None,
                "moved_mw": e["moved_mw"],
                "onset_err_h": (hit.start - e["start"]) * 24.0 if hit else np.nan,
                "end_err_h": (hit.end - e["end"]) * 24.0 if hit else np.nan,
                "alarm_lag_h": (hit.fire - e["start"]) * 24.0 if hit else np.nan,
                "partners": ",".join(hit.partners) if hit else "",
            }
        )
        found += hit is not None
    return found, len(items), detail


def main() -> None:
    """Tune and score every variant family and write the tables."""
    streams = build_streams()
    events, closures = ev.load_events(), ev.load_closures()
    pooled = [s for s in streams if s not in SPECIAL]
    years_odd = primary_years(streams=streams, odd=True, sites=pooled)
    years_even = primary_years(streams=streams, odd=False, sites=pooled)
    years_even_site = {s: primary_years(streams=streams, odd=False, sites=[s]) for s in SPECIAL}
    print(f"Primary-years: odd {years_odd:.2f}, even {years_even:.2f} (pooled, excl. D8 and D9)")
    summary, detail_all, curves = [], [], []
    for family, kwargs in FAMILIES:
        for k in K_GRID:
            variant = dt.Variant(k=k, **kwargs)
            spans = run_family(streams=streams, variant=variant)
            table = classify(spans=spans, events=events, closures=closures)
            fa = table.filter(pl.col("false_strict"), ~pl.col("site").is_in(SPECIAL))
            fa_odd = fa.filter(pl.col("month") % 2 == 1).height / years_odd
            fa_even = fa.filter(pl.col("month") % 2 == 0).height / years_even
            fa_unlogged_even = (
                table.filter(
                    pl.col("false_unlogged"),
                    ~pl.col("site").is_in(SPECIAL),
                    pl.col("month") % 2 == 0,
                ).height
                / years_even
            )
            specials = {
                s: table.filter(
                    pl.col("false_strict"), pl.col("site") == s, pl.col("month") % 2 == 0
                ).height
                / years_even_site[s]
                for s in SPECIAL
            }
            rec = {
                c: recall(spans=spans, events=events, cls=c) for c in ("block", "drop", "transfer")
            }
            curves.append(
                {
                    "family": family,
                    "k": k,
                    "fa_odd": fa_odd,
                    "fa_even": fa_even,
                    "fa_unlogged_even": fa_unlogged_even,
                    "fa_d8_even": specials["D8"],
                    "fa_d9_even": specials["D9"],
                    "block": rec["block"][0],
                    "drop": rec["drop"][0],
                    "transfer": rec["transfer"][0],
                    "n_spans": table.height,
                }
            )
    curve = pl.DataFrame(curves)
    curve.write_parquet(common.EXPERIMENT_DIR / "real_curves.parquet")
    chosen = {}
    for family, kwargs in FAMILIES:
        part = curve.filter(pl.col("family") == family, pl.col("fa_odd") <= BUDGET).sort("k")
        chosen[family] = float(part["k"][0]) if part.height else float(K_GRID[-1])
    (common.EXPERIMENT_DIR / "chosen_k.json").write_text(json.dumps(chosen, indent=2))
    for family, kwargs in FAMILIES:
        variant = dt.Variant(k=chosen[family], **kwargs)
        spans = run_family(streams=streams, variant=variant)
        table = classify(spans=spans, events=events, closures=closures)
        row = curve.filter(
            (pl.col("family") == family) & (pl.col("k") == chosen[family])
        ).to_dicts()[0]
        for cls in ("block", "drop", "transfer"):
            _, _, detail = recall(spans=spans, events=events, cls=cls)
            for d in detail:
                detail_all.append({"family": family, "k": chosen[family], **d})
        summary.append(row)
        print(
            f"{family:5s} k={chosen[family]:5.2f} fa_odd={row['fa_odd']:.2f} fa_even={row['fa_even']:.2f} "
            f"unlogged={row['fa_unlogged_even']:.2f} D8={row['fa_d8_even']:.1f} D9={row['fa_d9_even']:.1f} "
            f"blocks={row['block']}/17 drops={row['drop']}/29 transfers={row['transfer']}/"
            f"{sum(e['cls'] == 'transfer' for e in events)}"
        )
        table.write_parquet(common.EXPERIMENT_DIR / f"real_spans_{family}.parquet")
    pl.DataFrame(detail_all).write_parquet(common.EXPERIMENT_DIR / "real_event_detail.parquet")


if __name__ == "__main__":
    main()
