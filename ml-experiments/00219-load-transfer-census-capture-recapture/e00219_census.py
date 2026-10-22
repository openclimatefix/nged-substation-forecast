# ruff: noqa: E501, D103
"""Capture-recapture census of large load transfers from three detectors' intervals, plus the injected arm.

Reads `intervals.parquet` and `p99.parquet` written by `e00219_build.py`, and experiment 195's stored
injected found/not-found flags. No switching log is read. The 35 logged transfers enter only the final
comparison, as numbers quoted from earlier write-ups.

    OMP_NUM_THREADS=2 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00219-*/e00219_census.py
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import combinations
from typing import Final

import numpy as np
import polars as pl
from e00219_common import ML_EXPERIMENTS_DATA_DIR, WORK_DIR

E195_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00195-switching-neural-plus-detector-features"
E217_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00217-switching-label-free-thresholds-in-noise-units"
GRID_START: Final = datetime(2022, 1, 1, tzinfo=UTC)
MIN_HOURS: Final = 3.0
OVERLAP: Final = 0.5
SIZE_EDGES: Final = (0.2, 0.4)
"""Size strata, as a share of the Primary's 99th percentile of demand: below 20%, 20 to 40%, 40% and above."""
DURATION_EDGES_DAYS: Final = (3.0, 14.0)
N_BOOT: Final = 2000
DETECTORS: Final = ("RO-NC", "pair", "learned")
LOGGED: Final = {
    "RO-NC": (0.51, 0.657),
    "pair": (5 / 35, float("nan")),
    "learned": (23 / 35, float("nan")),
}
"""Recall on the 35 logged transfers (by count, MW-weighted), quoted from experiments 146, 156, and 195."""
INJECTED_ARMS: Final = {"RO-NC": "RO-NC", "pair": "Pair alarm", "learned": "R re-ranker"}
RNG: Final = np.random.default_rng(219)


# ----------------------------------------------------------------------------- estimators
def three_list(*, c: dict[tuple[int, int, int], float]) -> tuple[float, float]:
    """Unseen-cell estimates for three lists: no three-way interaction, and Chao's lower bound."""
    d = lambda a, b, cc: c.get((a, b, cc), 0.0)  # noqa: E731
    num = d(1, 0, 0) * d(0, 1, 0) * d(0, 0, 1) * d(1, 1, 1)
    den = d(1, 1, 0) * d(1, 0, 1) * d(0, 1, 1)
    nti = num / den if den > 0 else float("nan")
    f1 = d(1, 0, 0) + d(0, 1, 0) + d(0, 0, 1)
    f2 = d(1, 1, 0) + d(1, 0, 1) + d(0, 1, 1)
    chao = (2 / 3) * f1 * f1 / (2 * f2) if f2 > 0 else float("nan")
    return nti, chao


def summarise(*, flags: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """All estimates from an events x 3 boolean matrix with integer weights."""
    c: dict[tuple[int, int, int], float] = {}
    for pattern in np.unique(flags.astype(int), axis=0):
        key = (int(pattern[0]), int(pattern[1]), int(pattern[2]))
        c[key] = float(weights[(flags.astype(int) == pattern).all(axis=1)].sum())
    seen = float(weights[flags.any(axis=1)].sum())
    out = {"n_seen": seen}
    n = [float(weights[flags[:, i]].sum()) for i in range(3)]
    for i in range(3):
        out[f"n_{DETECTORS[i]}"] = n[i]
    for i, j in combinations(range(3), 2):
        m = float(weights[flags[:, i] & flags[:, j]].sum())
        out[f"m_{DETECTORS[i]}_{DETECTORS[j]}"] = m
        out[f"lp_{DETECTORS[i]}_{DETECTORS[j]}"] = n[i] * n[j] / m if m > 0 else float("nan")
        out[f"chapman_{DETECTORS[i]}_{DETECTORS[j]}"] = (n[i] + 1) * (n[j] + 1) / (m + 1) - 1
    nti, chao = three_list(c=c)
    out["nti"] = seen + nti if np.isfinite(nti) else float("nan")
    out["chao"] = seen + chao if np.isfinite(chao) else float("nan")
    primary = out["nti"] if np.isfinite(out["nti"]) else out["chao"]
    out["primary"] = max(primary, seen) if np.isfinite(primary) else float("nan")
    out["chao_fallback"] = float(not np.isfinite(out["nti"]))
    for i in range(3):
        out[f"recall_{DETECTORS[i]}"] = (
            n[i] / out["primary"] if out["primary"] > 0 else float("nan")
        )
    return out


# ----------------------------------------------------------------------------- real events
def year_month(*, stamp: np.ndarray) -> np.ndarray:
    return np.array(
        [
            (GRID_START + timedelta(minutes=30 * int(s))).year * 100
            + (GRID_START + timedelta(minutes=30 * int(s))).month
            for s in stamp
        ]
    )


def large_intervals(*, frame: pl.DataFrame, p99: dict[int, float], frac: float) -> pl.DataFrame:
    """Intervals of at least 3 hours whose step reaches `frac` of either Primary's 99th percentile."""
    rows = frame.with_columns(
        hours=(pl.col("end") - pl.col("onset")) / 2,
        p_alarm=pl.col("alarm_series").replace_strict(p99, default=None, return_dtype=pl.Float64),
        p_series=pl.col("series").replace_strict(p99, default=None, return_dtype=pl.Float64),
    )
    return rows.filter(
        (pl.col("hours") >= MIN_HOURS)
        & (pl.col("step_mw").abs() >= frac * pl.min_horizontal("p_alarm", "p_series"))
    ).with_columns(ratio=pl.col("step_mw").abs() / pl.min_horizontal("p_alarm", "p_series"))


def components(*, rows: pl.DataFrame) -> np.ndarray:
    """Union-find label of each interval: same Primary and overlap of at least half the shorter one."""
    n = rows.height
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    series, onset, end = rows["series"].to_numpy(), rows["onset"].to_numpy(), rows["end"].to_numpy()
    # A partner-side interval is the same transfer as its alarm-side interval of the same detector.
    alarm_series = rows["alarm_series"].to_numpy()
    detector = rows["detector"].to_numpy()
    first_of: dict[tuple, int] = {}
    for i in range(n):
        key = (detector[i], int(alarm_series[i]), int(onset[i]))
        if key in first_of:
            parent[find(i)] = find(first_of[key])
        else:
            first_of[key] = i
    order = np.argsort(onset)
    for a_pos, a in enumerate(order):
        for b in order[a_pos + 1 :]:
            if onset[b] >= end[a]:
                break
            if series[a] != series[b]:
                continue
            overlap = min(end[a], end[b]) - onset[b]
            if overlap >= OVERLAP * min(end[a] - onset[a], end[b] - onset[b]):
                parent[find(int(a))] = find(int(b))
    return np.array([find(i) for i in range(n)])


def event_table(*, rows: pl.DataFrame, detectors: dict[str, str]) -> pl.DataFrame:
    """One row per matched event, with a found flag per detector (`detectors` maps list name to interval name)."""
    rows = rows.with_columns(comp=pl.Series(components(rows=rows)))
    aggs = [(pl.col("detector") == name).any().alias(key) for key, name in detectors.items()]
    return (
        rows.group_by("comp")
        .agg(
            *aggs,
            ratio=pl.col("ratio").median(),
            days=((pl.col("end") - pl.col("onset")) / 48).median(),
            partner=(pl.col("role") == "partner").any(),
            onset=pl.col("onset").min(),
            n_series=pl.col("series").n_unique(),
        )
        .with_columns(
            ym=pl.Series(
                year_month(
                    stamp=rows.group_by("comp")
                    .agg(pl.col("onset").min())
                    .sort("comp")["onset"]
                    .to_numpy()
                )
            )
            if False
            else pl.lit(0)
        )
    )


def strata(*, events: pl.DataFrame) -> dict[str, np.ndarray]:
    ratio, days = events["ratio"].to_numpy(), events["days"].to_numpy()
    partner = events["partner"].to_numpy()
    out = {"all": np.ones(events.height, dtype=bool)}
    sz = np.digitize(ratio, SIZE_EDGES)
    for k, name in enumerate(("size 10-20% of p99", "size 20-40% of p99", "size 40%+ of p99")):
        out[name] = sz == k
    du = np.digitize(days, DURATION_EDGES_DAYS)
    for k, name in enumerate(("duration under 3 d", "duration 3-14 d", "duration 14 d+")):
        out[name] = du == k
    out["metered partner seen"] = partner
    out["no metered partner seen"] = ~partner
    return out


def bootstrap(*, flags: np.ndarray, groups: np.ndarray) -> dict[str, tuple[float, float]]:
    """95% intervals of the primary estimate and the recalls, resampling whole groups (calendar months)."""
    labels = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in labels}
    draws = []
    for _ in range(N_BOOT):
        pick = RNG.choice(labels, size=len(labels), replace=True)
        w = np.zeros(len(groups))
        for g in pick:
            w[index[g]] += 1
        draws.append(summarise(flags=flags, weights=w))
    keys = (
        "primary",
        "nti",
        "chao",
        "recall_RO-NC",
        "recall_pair",
        "recall_learned",
        "chao_fallback",
    )
    return {
        k: (
            float(np.nanpercentile([d[k] for d in draws], 2.5)),
            float(np.nanpercentile([d[k] for d in draws], 97.5)),
        )
        for k in keys
    }


def run_real(
    *, frame: pl.DataFrame, p99: dict, years: dict, frac: float, variant: str, fold: int
) -> list[dict]:
    learned = f"learned_f{fold}"
    sub = frame.filter(
        (pl.col("variant") == variant)
        & pl.col("detector").is_in(["RO-NC", "pair", learned])
        & pl.col("alarm_series").is_in(list(years))
        & pl.col("series").is_in(list(years))
    )
    rows = large_intervals(frame=sub, p99=p99, frac=frac)
    rows = rows.with_columns(ym=pl.Series(year_month(stamp=rows["onset"].to_numpy())))
    rows = rows.with_columns(comp=pl.Series(components(rows=rows)))
    events = rows.group_by("comp").agg(
        pl.col("detector").is_in(["RO-NC"]).any().alias("RO-NC"),
        pl.col("detector").is_in(["pair"]).any().alias("pair"),
        pl.col("detector").is_in([learned]).any().alias("learned"),
        ratio=pl.col("ratio").median(),
        days=((pl.col("end") - pl.col("onset")) / 48).median(),
        partner=(pl.col("role") == "partner").any(),
        onset=pl.col("onset").min(),
        ym=pl.col("ym").min(),
        n_series=pl.col("series").n_unique(),
        series=pl.col("alarm_series").first(),
    )
    events.write_parquet(WORK_DIR / f"events_{variant}_{int(frac * 100)}pct_f{fold}.parquet")
    flags = events.select(*DETECTORS).to_numpy().astype(bool)
    total_years = sum(years.values())
    out = []
    for name, mask in strata(events=events).items():
        if mask.sum() == 0:
            continue
        stat = summarise(flags=flags[mask], weights=np.ones(int(mask.sum())))
        row = {
            "variant": variant,
            "frac": frac,
            "fold": fold,
            "stratum": name,
            "n_events_seen": int(mask.sum()),
            **stat,
        }
        if (frac == 0.10) and variant == "L6_M48":
            ci = bootstrap(flags=flags[mask], groups=events["ym"].to_numpy()[mask])
            for k, (lo, hi) in ci.items():
                row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
        row["per_primary_year"] = row["primary"] / total_years
        out.append(row)
    return out


# ----------------------------------------------------------------------------- injected arm
def run_injected() -> list[dict]:
    x = pl.concat([pl.read_parquet(E195_DIR / f"injected_f{k}.parquet") for k in (0, 1, 2)]).filter(
        (pl.col("kind") == "eval_held") & (pl.col("hours") >= 24)
    )
    wide = (
        x.filter(pl.col("arm").is_in(list(INJECTED_ARMS.values())))
        .pivot(on="arm", index=["fold", "world", "event_id", "depth_z", "hours"], values="found")
        .drop_nulls()
    )
    flags = np.stack([wide[INJECTED_ARMS[d]].to_numpy().astype(bool) for d in DETECTORS], axis=1)
    edges = np.quantile(wide["depth_z"].to_numpy(), [1 / 3, 2 / 3])
    size_bin = np.digitize(wide["depth_z"].to_numpy(), edges)
    groups = (wide["fold"].to_numpy() * 100 + wide["world"].to_numpy()).astype(int)
    out = []
    for name, mask in {
        "all": np.ones(wide.height, dtype=bool),
        "small third": size_bin == 0,
        "middle third": size_bin == 1,
        "large third": size_bin == 2,
    }.items():
        known = int(mask.sum())
        stat = summarise(flags=flags[mask], weights=np.ones(known))
        ci = bootstrap(flags=flags[mask], groups=groups[mask])
        out.append(
            {
                "stratum": name,
                "known_n": known,
                **stat,
                **{
                    f"{k}_{b}": v for k, (lo, hi) in ci.items() for b, v in (("lo", lo), ("hi", hi))
                },
            }
        )
        print(
            f"injected {name}: known {known}, seen {stat['n_seen']:.0f}, primary {stat['primary']:.1f} "
            f"(err {100 * (stat['primary'] / known - 1):+.0f}%), nti {stat['nti']:.1f}, chao {stat['chao']:.1f}, "
            f"LP {stat['lp_RO-NC_pair']:.1f}/{stat['lp_RO-NC_learned']:.1f}/{stat['lp_pair_learned']:.1f}; "
            f"true recall RO-NC {flags[mask][:, 0].mean():.3f} est {stat['recall_RO-NC']:.3f}"
        )
    return out


def main() -> None:
    frame = pl.read_parquet(WORK_DIR / "intervals.parquet")
    p99 = dict(pl.read_parquet(WORK_DIR / "p99.parquet").iter_rows())
    rates = pl.read_parquet(E217_DIR / "rates.parquet").filter(
        (pl.col("detector") == "RO-NC/current") & (pl.col("group") == "all")
    )
    years = {int(s): float(y) for s, y in rates.select("series", "years").iter_rows()}
    print("Primaries:", len(years), "total Primary-years:", round(sum(years.values()), 2))
    all_rows: list[dict] = []
    for variant in ("L6_M48", "L6"):
        for frac in (0.10, 0.05, 0.20):
            for fold in range(3):
                all_rows += run_real(
                    frame=frame, p99=p99, years=years, frac=frac, variant=variant, fold=fold
                )
    table = pl.DataFrame(all_rows, infer_schema_length=None)
    table.write_parquet(WORK_DIR / "census.parquet")
    show = table.filter(
        (pl.col("variant") == "L6_M48") & (pl.col("frac") == 0.10) & (pl.col("stratum") == "all")
    )
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=250):
        print(
            show.select(
                "fold",
                "n_seen",
                "n_RO-NC",
                "n_pair",
                "n_learned",
                "m_RO-NC_pair",
                "m_RO-NC_learned",
                "m_pair_learned",
                "lp_RO-NC_pair",
                "lp_RO-NC_learned",
                "lp_pair_learned",
                "nti",
                "chao",
                "primary",
                "primary_lo",
                "primary_hi",
                "per_primary_year",
                "recall_RO-NC",
                "recall_RO-NC_lo",
                "recall_RO-NC_hi",
                "recall_pair",
                "recall_learned",
            )
        )
    inj = pl.DataFrame(run_injected(), infer_schema_length=None)
    inj.write_parquet(WORK_DIR / "injected_census.parquet")


if __name__ == "__main__":
    main()
