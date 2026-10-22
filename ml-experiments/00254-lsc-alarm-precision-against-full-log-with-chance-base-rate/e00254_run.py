# ruff: noqa: E501, PLR0915, PERF401, PLC0206
"""Class shares, log-near excess over a week-shifted base rate, and corrected false-alarm rates.

For each frozen alarm source (experiment 227's LR_b20, LR_b80, RO_b80, PG_g70) the script classes every
alarm, shifts the alarms by whole weeks to estimate how often a logged closure lies near an alarm by
chance, and turns the excess into a corrected precision with a month-resampled interval. Nothing is
tuned and nothing is trained.

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00254-*/e00254_run.py
"""

from __future__ import annotations

import json
from typing import Any, Final

import e00254_data as d
import numpy as np
import polars as pl

WRAP: Final[int] = (d.GRID_LENGTH // d.WEEK) * d.WEEK
"""Shifts wrap modulo a whole number of weeks (234), so a wrapped alarm keeps its weekday and hour."""
N_SHIFTS: Final[int] = 200
N_RESAMPLES: Final[int] = 1000
SHIFT_WEEKS: Final[tuple[int, int]] = (3, 100)
SHORT_SHIFT_WEEKS: Final[tuple[int, int]] = (3, 12)
"""Post hoc sensitivity: shifts short enough to keep the season of the year."""
NEAR_WINDOWS: Final[dict[str, int]] = {"6h": 12, "24h": 48, "48h": 96}
BALANCED_RANGE: Final[tuple[float, float]] = (0.4, 2.5)
SEED: Final[int] = 254
CLASSES: Final[tuple[str, ...]] = (
    "strict match",
    "log-near (24 h)",
    "balanced-opposite candidate",
    "D8 drop edge",
    "block day",
    "bank holiday or clock change",
    "none",
)


def _matches(*, table: pl.DataFrame, event: dict) -> pl.DataFrame:
    """Rows of the table that match a labelled load shift (experiment 195's rule)."""
    hits = table.filter(pl.col("series").is_in(event["series"]))
    keep = [
        abs(o - event["onset"]) <= d.ONSET_TOLERANCE
        and abs(e - event["end"]) <= d.end_tolerance(onset=event["onset"], end=event["end"])
        for o, e in zip(hits["onset"].to_list(), hits["end"].to_list(), strict=True)
    ]
    return hits.filter(pl.Series(keep, dtype=pl.Boolean)) if keep else hits.clear()


def alarm_set(*, source: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The source's alarms (one row per stepped series and stamp, weighted over folds) and its matched events.

    A learned source's alarm on a series comes from every fold that holds out a labelled Primary pair
    containing the series; the alarms from those folds count 1 / (number of such folds) each, so a series
    in pairs of two folds contributes one model's worth of alarms. An alarm that strictly matches a
    labelled load shift in the load shift's fold counts once with weight 1 and replaces its
    non-matching twins in the other folds. RO-NC and the pair alarm have one table. An alarm is
    `strict` when one of its table rows matches a labelled load shift of its fold.
    """
    tables = d.alarm_tables(source=source)
    foldless = source.startswith(("PG_", "RO_"))
    folds_of = d.series_folds()
    events = d.labelled_loads()
    fold_of_pair = d.pair_folds()
    strict_keys: dict[int, set[tuple[int, int]]] = {f: set() for f in tables}
    matched_events = []
    for e in events.iter_rows(named=True):
        fold = 0 if foldless else fold_of_pair[e["pair"]]
        hits = _matches(table=tables[fold], event=e)
        if hits.height:
            matched_events.append(
                {"card": e["card_id"], "onset": e["onset"], "fold": fold, "series": e["series"]}
            )
            strict_keys[fold] |= set(
                zip(hits["alarm_series"].to_list(), hits["onset"].to_list(), strict=True)
            )
    rows = []
    for fold, table in tables.items():
        if foldless and fold != 0:
            continue
        alarms = table.filter(pl.col("role") == "alarm") if "role" in table.columns else table
        for r in alarms.iter_rows(named=True):
            s = int(r["alarm_series"])
            if s not in folds_of:
                continue
            if not foldless and fold not in folds_of[s]:
                continue
            rows.append(
                {
                    "series": s,
                    "onset": int(r["onset"]),
                    "end": int(r["end"]),
                    "fold": fold,
                    "weight": 1.0 if foldless else 1.0 / len(folds_of[s]),
                    "strict": (s, int(r["onset"])) in strict_keys[fold],
                    "opp_ratio": r.get("opp_ratio"),
                    "n_opposite": r.get("n_opposite"),
                    "pair_alarm": source.startswith("PG_"),
                }
            )
    frame = pl.DataFrame(rows, infer_schema_length=None)
    hit = (
        frame.filter(pl.col("strict"))
        .select("series", "onset")
        .unique()
        .with_columns(has_strict=pl.lit(True))
    )
    frame = (
        frame.join(hit, on=["series", "onset"], how="left")
        .with_columns(pl.col("has_strict").fill_null(False))
        .filter(~pl.col("has_strict") | pl.col("strict"))
        .drop("has_strict")
        .group_by("series", "onset", "strict", maintain_order=True)
        .agg(pl.exclude("weight").first(), weight=pl.col("weight").sum())
        .with_columns(
            weight=pl.when(pl.col("strict"))
            .then(1.0)
            .otherwise(pl.col("weight").clip(upper_bound=1.0))
        )
    )
    return frame, pl.DataFrame(matched_events)


def near_flags(
    *, series: np.ndarray, stamps: np.ndarray, window: int, edges: dict[int, np.ndarray]
) -> np.ndarray:
    """True where a closure edge on the series or a neighbour lies within `window` half-hours of the stamp."""
    out = np.zeros(len(stamps), dtype=bool)
    for s in np.unique(series):
        sel = series == s
        e = edges[int(s)]
        if len(e) == 0:
            continue
        pos = np.searchsorted(e, stamps[sel])
        lo = e[np.clip(pos - 1, 0, len(e) - 1)]
        hi = e[np.clip(pos, 0, len(e) - 1)]
        out[sel] = np.minimum(np.abs(stamps[sel] - lo), np.abs(stamps[sel] - hi)) <= window
    return out


def classify(*, alarms: pl.DataFrame, edges: dict[int, np.ndarray]) -> pl.DataFrame:
    """Add the near-closure flags and the first-matching class of every alarm."""
    series, stamps = alarms["series"].to_numpy(), alarms["onset"].to_numpy()
    out = alarms
    for name, window in NEAR_WINDOWS.items():
        out = out.with_columns(
            pl.Series(
                f"near_{name}", near_flags(series=series, stamps=stamps, window=window, edges=edges)
            )
        )
    drop = d.d8_drop_edges()
    block = d.block_day_ranges()
    calendar = d.calendar_flags()
    n = len(stamps)
    is_d8 = np.array(
        [
            s == 26 and np.min(np.abs(drop - t)) <= NEAR_WINDOWS["6h"]
            for s, t in zip(series, stamps, strict=True)
        ],
        dtype=bool,
    )
    is_block = np.array(
        [
            any(
                a - d.BLOCK_TOLERANCE <= t <= b + d.BLOCK_TOLERANCE
                for a, b in block.get(int(s), [])
            )
            for s, t in zip(series, stamps, strict=True)
        ],
        dtype=bool,
    )
    is_cal = calendar[np.clip(stamps, 0, d.GRID_LENGTH - 1)]
    ratio = out["opp_ratio"].fill_null(0.0).to_numpy()
    n_opp = out["n_opposite"].fill_null(0.0).to_numpy()
    is_bal = ((n_opp >= 1) & (ratio >= BALANCED_RANGE[0]) & (ratio <= BALANCED_RANGE[1])) | out[
        "pair_alarm"
    ].to_numpy()
    strict, near = out["strict"].to_numpy(), out["near_24h"].to_numpy()
    labels = np.full(n, "none", dtype=object)
    for mask, name in (
        (is_cal, CLASSES[5]),
        (is_block, CLASSES[4]),
        (is_d8, CLASSES[3]),
        (is_bal, CLASSES[2]),
        (near, CLASSES[1]),
        (strict, CLASSES[0]),
    ):
        labels[mask] = name
    return out.with_columns(pl.Series("class", labels.astype(str)), is_d8_edge=pl.Series(is_d8))


def shifted_near(
    *,
    alarms: pl.DataFrame,
    edges: dict[int, np.ndarray],
    window: int,
    weeks_range: tuple[int, int] = SHIFT_WEEKS,
) -> np.ndarray:
    """Near-closure flags of every alarm under each of `N_SHIFTS` whole-week shifts: array (alarms, shifts)."""
    rng = np.random.default_rng(SEED)
    weeks = np.concatenate(
        [
            np.arange(weeks_range[0], weeks_range[1] + 1),
            -np.arange(weeks_range[0], weeks_range[1] + 1),
        ]
    )
    series, stamps = alarms["series"].to_numpy(), alarms["onset"].to_numpy()
    k = {s: rng.choice(weeks, size=N_SHIFTS, replace=True) for s in d.SERIES_IDS}
    out = np.zeros((len(stamps), N_SHIFTS), dtype=bool)
    for j in range(N_SHIFTS):
        shift = np.array([k[int(s)][j] for s in series]) * d.WEEK
        out[:, j] = near_flags(
            series=series, stamps=(stamps + shift) % WRAP, window=window, edges=edges
        )
    return out


def summarise(
    *,
    alarms: pl.DataFrame,
    chance: np.ndarray,
    months: np.ndarray,
    event_months: np.ndarray,
    rng: np.random.Generator,
    resamples: int = N_RESAMPLES,
) -> dict[str, Any]:
    """Weighted counts, excess log-near share, corrected precision, and month-resampled intervals.

    `chance` is (alarms, shifts); the log-near share is taken over the alarms that are not strict matches.
    `event_months` holds the month label of each matched load shift (the denominator of the false-alarm rate).
    """
    w = alarms["weight"].to_numpy()
    strict = alarms["strict"].to_numpy()
    near = alarms["near_24h"].to_numpy()

    def stats(idx: np.ndarray, cnt: np.ndarray, m_events: float) -> dict[str, Any]:
        wt = w[idx] * cnt
        s = strict[idx]
        a = wt.sum()
        n_ns = wt[~s].sum()
        obs = (wt[~s] * near[idx][~s]).sum() / n_ns if n_ns > 0 else np.nan
        per_shift = (
            (wt[~s, None] * chance[idx][~s]).sum(0) / n_ns
            if n_ns > 0
            else np.full(chance.shape[1], np.nan)
        )
        excess = obs - np.mean(per_shift)
        real = wt[s].sum() + max(excess, 0.0) * n_ns if n_ns > 0 else wt[s].sum()
        m = m_events
        return {
            "alarms": a,
            "strict_alarms": wt[s].sum(),
            "non_strict": n_ns,
            "obs_share": obs,
            "chance_share": float(np.mean(per_shift)),
            "chance_lo": float(np.quantile(per_shift, 0.025)),
            "chance_hi": float(np.quantile(per_shift, 0.975)),
            "excess_pp": 100 * excess,
            "precision": real / a if a > 0 else np.nan,
            "fa_per_match": (a - real) / m if m > 0 else np.nan,
            "fa_per_match_all_false": (a - wt[s].sum()) / m if m > 0 else np.nan,
            "matched_events": m,
        }

    n = len(w)
    point = stats(np.arange(n), np.ones(n), float(len(event_months)))
    uniq = np.unique(months)
    by_month = {m: np.flatnonzero(months == m) for m in uniq}
    boot = {"excess_pp": [], "precision": [], "fa_per_match": [], "obs_share": []}
    for _ in range(resamples):
        picks = rng.choice(uniq, size=len(uniq), replace=True)
        counts = np.bincount(np.searchsorted(uniq, picks), minlength=len(uniq))
        idx = np.concatenate([by_month[m] for m in uniq])
        cnt = np.concatenate([np.full(len(by_month[m]), counts[i]) for i, m in enumerate(uniq)])
        keep = cnt > 0
        ev_counts = np.bincount(np.searchsorted(uniq, picks), minlength=len(uniq))
        m_b = float(sum(ev_counts[np.searchsorted(uniq, m)] for m in event_months if m in by_month))
        r = stats(idx[keep], cnt[keep].astype(float), m_b)
        for key in boot:
            boot[key].append(r[key])
    for key, vals in boot.items():
        if not vals:
            continue
        v = np.array(vals, dtype=float)
        point[f"{key}_lo"], point[f"{key}_hi"] = (
            float(np.nanquantile(v, 0.025)),
            float(np.nanquantile(v, 0.975)),
        )
    return point


def main() -> None:
    """Run every source, print every quoted number, and store the tables."""
    d.WORK_DIR.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(300)
    pl.Config.set_float_precision(3)
    edges = d.closure_edges()
    ym = d.stamp_year_month()
    print(f"logged closures of 2 h to 60 days: {d.logged_closures().height}")
    class_rows, summary_rows, primary_rows, chance_rows, subset_rows = [], [], [], [], []
    for source in d.SOURCES:
        alarms, matched = alarm_set(source=source)
        alarms = classify(alarms=alarms, edges=edges)
        months = ym[np.clip(alarms["onset"].to_numpy(), 0, d.GRID_LENGTH - 1)]
        chance = shifted_near(alarms=alarms, edges=edges, window=NEAR_WINDOWS["24h"])
        rng = np.random.default_rng(SEED)
        ev_month = ym[np.clip(matched["onset"].to_numpy(), 0, d.GRID_LENGTH - 1)]
        total = summarise(
            alarms=alarms, chance=chance, months=months, event_months=ev_month, rng=rng
        )
        total.update(source=source, alarm_rows=alarms.height)
        summary_rows.append(total)
        for window in ("6h", "48h"):
            c = shifted_near(alarms=alarms, edges=edges, window=NEAR_WINDOWS[window])
            ns = ~alarms["strict"].to_numpy()
            w = alarms["weight"].to_numpy()[ns]
            obs = float((w * alarms[f"near_{window}"].to_numpy()[ns]).sum() / w.sum())
            per = (w[:, None] * c[ns]).sum(0) / w.sum()
            chance_rows.append(
                {
                    "source": source,
                    "window": window,
                    "obs_share": obs,
                    "chance_share": float(per.mean()),
                    "chance_lo": float(np.quantile(per, 0.025)),
                    "chance_hi": float(np.quantile(per, 0.975)),
                }
            )
        ns = ~alarms["strict"].to_numpy()
        w = alarms["weight"].to_numpy()
        short = shifted_near(
            alarms=alarms, edges=edges, window=NEAR_WINDOWS["24h"], weeks_range=SHORT_SHIFT_WEEKS
        )
        per_short = (w[ns, None] * short[ns]).sum(0) / w[ns].sum()
        chance_rows.append(
            {
                "source": source,
                "window": "24h, shifts of 3 to 12 weeks",
                "obs_share": float((w[ns] * alarms["near_24h"].to_numpy()[ns]).sum() / w[ns].sum()),
                "chance_share": float(per_short.mean()),
                "chance_lo": float(np.quantile(per_short, 0.025)),
                "chance_hi": float(np.quantile(per_short, 0.975)),
            }
        )
        per_shift = (w[ns, None] * chance[ns]).sum(0) / w[ns].sum()
        np.save(d.WORK_DIR / f"shifted_shares_{source}.npy", per_shift)
        for cls in CLASSES:
            class_rows.append(
                {
                    "source": source,
                    "class": cls,
                    "alarms": float(w[alarms["class"].to_numpy() == cls].sum()),
                }
            )
        for s in sorted(alarms["series"].unique().to_list()):
            sel = (alarms["series"] == s).to_numpy()
            sub = alarms.filter(pl.Series(sel))
            if sub["weight"].sum() < 5:
                continue
            mine = np.array([s in ser for ser in matched["series"].to_list()], dtype=bool)
            row = summarise(
                alarms=sub,
                chance=chance[sel],
                months=months[sel],
                event_months=ev_month[mine],
                rng=np.random.default_rng(SEED),
            )
            row.update(source=source, series=s, label=d.label_of(series=s))
            primary_rows.append(row)
        parity = np.array([m % 100 for m in months])
        for name, keep in (("even months", parity % 2 == 0), ("odd months", parity % 2 == 1)):
            sub = alarms.filter(pl.Series(keep))
            em = ev_month[(ev_month % 100) % 2 == (0 if name == "even months" else 1)]
            row = summarise(
                alarms=sub,
                chance=chance[keep],
                months=months[keep],
                event_months=em,
                rng=np.random.default_rng(SEED),
            )
            row.update(source=source, subset=name, alarm_rows=sub.height)
            subset_rows.append(row)
        if source == "LR_b20":
            for s_out in d.SERIES_IDS:
                keep = (alarms["series"] != s_out).to_numpy()
                if keep.all():
                    continue
                em = ev_month[
                    np.array([s_out not in ser for ser in matched["series"].to_list()], dtype=bool)
                ]
                row = summarise(
                    alarms=alarms.filter(pl.Series(keep)),
                    chance=chance[keep],
                    months=months[keep],
                    event_months=em,
                    rng=np.random.default_rng(SEED),
                    resamples=0,
                )
                row.update(
                    source=source,
                    subset=f"without {d.label_of(series=s_out)}",
                    alarm_rows=int(keep.sum()),
                )
                subset_rows.append(row)
        alarms.write_parquet(d.WORK_DIR / f"alarms_{source}.parquet")
    summary, classes, primary, chances = (
        pl.DataFrame(x, infer_schema_length=None)
        for x in (summary_rows, class_rows, primary_rows, chance_rows)
    )
    subsets = pl.DataFrame(subset_rows, infer_schema_length=None)
    subsets.write_parquet(d.WORK_DIR / "subsets.parquet")
    summary.write_parquet(d.WORK_DIR / "summary.parquet")
    classes.write_parquet(d.WORK_DIR / "classes.parquet")
    primary.write_parquet(d.WORK_DIR / "per_primary.parquet")
    chances.write_parquet(d.WORK_DIR / "other_windows.parquet")
    print(
        "== summary by source (24 h window; shares are over the alarms that are not strict matches) =="
    )
    print(
        summary.select(
            "source",
            "alarm_rows",
            "alarms",
            "strict_alarms",
            "matched_events",
            "non_strict",
            "obs_share",
            "obs_share_lo",
            "obs_share_hi",
            "chance_share",
            "chance_lo",
            "chance_hi",
            "excess_pp",
            "excess_pp_lo",
            "excess_pp_hi",
        )
    )
    print(
        summary.select(
            "source",
            "precision",
            "precision_lo",
            "precision_hi",
            "fa_per_match",
            "fa_per_match_lo",
            "fa_per_match_hi",
            "fa_per_match_all_false",
        )
    )
    print(
        "== even months, odd months (the threshold budget was set on odd months), and LR_b20 without one Primary =="
    )
    print(
        subsets.select(
            "source",
            "subset",
            "alarm_rows",
            "alarms",
            "strict_alarms",
            "obs_share",
            "chance_share",
            "excess_pp",
            "excess_pp_lo",
            "excess_pp_hi",
            "precision",
            "precision_lo",
            "precision_hi",
            "fa_per_match",
        )
    )
    print("== class counts ==")
    print(classes.pivot(on="class", index="source", values="alarms").select("source", *CLASSES))
    print("== 6 h and 48 h windows ==")
    print(chances)
    print("== per Primary ==")
    print(
        primary.select(
            "source",
            "label",
            "alarms",
            "strict_alarms",
            "matched_events",
            "non_strict",
            "obs_share",
            "chance_share",
            "excess_pp",
            "excess_pp_lo",
            "excess_pp_hi",
            "precision",
            "precision_lo",
            "precision_hi",
        )
    )
    verdict = summary.filter(pl.col("source") == "LR_b20").row(0, named=True)
    (d.WORK_DIR / "headline.json").write_text(
        json.dumps(
            {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in verdict.items()},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
