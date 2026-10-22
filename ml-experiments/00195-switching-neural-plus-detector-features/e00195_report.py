# ruff: noqa: E501, PLR0915
"""Pool the three folds, compute the headline tables and intervals, and print every number the page quotes.

    uv run --no-sync python ml-experiments/00195-*/e00195_report.py > ml-experiments/00195-*/printed_report.txt

Intervals come from resampling whole Primary pairs (23 pairs) with replacement, 10,000 draws.
"""

from __future__ import annotations

import json
import os
from typing import Final

import numpy as np
import polars as pl
from e00195_common import ML_EXPERIMENTS_DATA_DIR, real_transfers
from e00195_common import WORK_DIR as BASE_DIR
from e00195_data import Base
from e00195_run import end_tolerance, real_candidates

WORK_DIR = BASE_DIR / os.environ["E195_TAG"] if os.environ.get("E195_TAG") else BASE_DIR
N_RESAMPLES: Final[int] = 10_000
BASELINE: Final[str] = "RO-NC"
LIKE_FOR_LIKE: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00181-switching-block-detector-and-like-for-like-table"
    / "like_for_like.parquet"
)
ARM_ORDER: Final[tuple[str, ...]] = (
    "RO-NC",
    "Pair alarm",
    "U RO-NC + pair (rule)",
    "F logistic",
    "F tree",
    "F network",
    "R re-ranker",
    "C stack",
    "U F network + RO-NC",
    "U F network + pair",
    "U R re-ranker + pair",
    "I F network & RO-NC",
    "I F network & pair",
)


def load() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Pooled per-event hits, per-injected-event hits, and alarms."""
    folds = [0, 1, 2]
    real = pl.concat([pl.read_parquet(WORK_DIR / f"real_f{f}.parquet") for f in folds])
    injected = pl.concat([pl.read_parquet(WORK_DIR / f"injected_f{f}.parquet") for f in folds])
    alarms = pl.concat([pl.read_parquet(WORK_DIR / f"alarms_f{f}.parquet") for f in folds])
    transfers = real_transfers().select(
        "card_id", "hours", "moved_mw_p50", "pair", "onset", "end", "found_ronc"
    )
    return real.join(transfers, on="card_id"), injected, alarms


def bootstrap_difference(
    *, table: pl.DataFrame, arm: str, base: str, weighted: bool, seed: int = 195
) -> tuple[float, float, float]:
    """Point estimate and 95% interval of (arm - base) recall, resampling whole pairs."""
    w = table["moved_mw_p50"].fill_null(0.0).to_numpy() if weighted else np.ones(table.height)
    pairs = sorted(table["pair"].unique().to_list())
    key = {a: table.filter(pl.col("arm") == a).sort("card_id") for a in (arm, base)}
    wa = (
        key[arm]["moved_mw_p50"].fill_null(0.0).to_numpy() if weighted else np.ones(key[arm].height)
    )
    fa = key[arm]["found"].to_numpy().astype(float)
    fb = key[base]["found"].to_numpy().astype(float)
    pr = key[arm]["pair"].to_numpy()
    num_a = np.array([(wa * fa)[pr == p].sum() for p in pairs])
    num_b = np.array([(wa * fb)[pr == p].sum() for p in pairs])
    den = np.array([wa[pr == p].sum() for p in pairs])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(pairs), size=(N_RESAMPLES, len(pairs)))
    diff = (num_a[picks].sum(axis=1) - num_b[picks].sum(axis=1)) / np.maximum(
        den[picks].sum(axis=1), 1e-9
    )
    point = (num_a.sum() - num_b.sum()) / den.sum()
    del w
    return float(point), float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))


def bootstrap_recall(
    *, table: pl.DataFrame, arm: str, weighted: bool, seed: int = 196
) -> tuple[float, float, float]:
    """Point estimate and 95% interval of recall for one arm, resampling whole pairs."""
    sub = table.filter(pl.col("arm") == arm).sort("card_id")
    w = sub["moved_mw_p50"].fill_null(0.0).to_numpy() if weighted else np.ones(sub.height)
    f = sub["found"].to_numpy().astype(float)
    pr = sub["pair"].to_numpy()
    pairs = sorted(sub["pair"].unique().to_list())
    num = np.array([(w * f)[pr == p].sum() for p in pairs])
    den = np.array([w[pr == p].sum() for p in pairs])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(pairs), size=(N_RESAMPLES, len(pairs)))
    draws = num[picks].sum(axis=1) / np.maximum(den[picks].sum(axis=1), 1e-9)
    return (
        float(num.sum() / den.sum()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def main() -> None:  # noqa: PLR0912
    """Print the tables."""
    real, injected, alarms = load()
    long = real.filter(pl.col("hours") > 24)
    like = pl.read_parquet(LIKE_FOR_LIKE).select("card_id", "xgboost_has_readings")
    both = long.join(like, on="card_id", how="left")
    pairs_sorted = (
        long.filter(pl.col("arm") == BASELINE)
        .group_by("pair")
        .agg(mw=pl.col("moved_mw_p50").sum(), n=pl.len())
        .sort("mw", descending=True)
    )
    label = {p: chr(ord("A") + i) for i, p in enumerate(pairs_sorted["pair"].to_list())}
    print("== Events: 35 transfers over 1 day (pairs lettered A.. by moved MW) ==")
    print(pairs_sorted.with_columns(label=pl.col("pair").replace(label)).drop("pair"))
    ronc_found = long.filter(pl.col("arm") == BASELINE).sort("card_id")["found"].to_numpy()
    print("\n== Headline: arms at 80 unexplained odd-month alarms ==")
    header = "arm | found/35 | onset only/35 | found/66 | MW-weighted recall [95%] | minus RO-NC [95%] | gained | lost | median onset err h | found/29 both-see"
    print(header)
    sees = (
        both.filter(pl.col("arm") == BASELINE)
        .sort("card_id")["xgboost_has_readings"]
        .fill_null(False)
        .to_numpy()
    )
    for arm in ARM_ORDER:
        sub = long.filter(pl.col("arm") == arm).sort("card_id")
        found = sub["found"].to_numpy()
        point, lo, hi = bootstrap_recall(table=long, arm=arm, weighted=True)
        d = bootstrap_difference(table=long, arm=arm, base=BASELINE, weighted=True)
        err = sub.filter(pl.col("found"))["onset_error_h"].drop_nulls()
        all66 = real.filter(pl.col("arm") == arm)
        print(
            f"{arm} | {int(found.sum())} | {int(sub['onset_found'].sum())} | {int(all66['found'].sum())} | "
            f"{point:.3f} [{lo:.3f}, {hi:.3f}] | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | "
            f"{int((found & ~ronc_found).sum())} | {int((~found & ronc_found).sum())} | "
            f"{err.median() if len(err) else float('nan'):.2f} | {int((found & sees).sum())}/{int(sees.sum())}"
        )
    print("\n== Count difference vs RO-NC (events), 95% pair interval ==")
    for arm in ARM_ORDER:
        if arm == BASELINE:
            continue
        d = bootstrap_difference(table=long, arm=arm, base=BASELINE, weighted=False)
        print(
            f"{arm}: {d[0] * 35:+.1f} events [{d[1] * 35:+.1f}, {d[2] * 35:+.1f}] (unweighted share x 35)"
        )
    print("\n== Found per pair (35 events) ==")
    rows = []
    for arm in ARM_ORDER:
        sub = long.filter(pl.col("arm") == arm)
        per = {label[p]: int(g["found"].sum()) for (p,), g in sub.group_by("pair")}
        rows.append(f"{arm}: " + " ".join(f"{k}={per.get(k, 0)}" for k in sorted(label.values())))
    print("\n".join(rows))
    n_by = {label[p]: n for p, n in zip(pairs_sorted["pair"], pairs_sorted["n"], strict=True)}
    print("events per pair:", " ".join(f"{k}={v}" for k, v in sorted(n_by.items())))
    print(
        "\n== Leave one pair out: found/35 total with the pair removed (best learned arm vs RO-NC) =="
    )
    for arm in ("F network", "R re-ranker", "C stack", "U RO-NC + pair (rule)"):
        sub = long.filter(pl.col("arm") == arm)
        base = long.filter(pl.col("arm") == BASELINE)
        cells = []
        for p, k in label.items():
            gain = int(sub.filter(pl.col("pair") != p)["found"].sum()) - int(
                base.filter(pl.col("pair") != p)["found"].sum()
            )
            cells.append(f"{k}:{gain:+d}")
        print(f"{arm} minus RO-NC with one pair removed:", " ".join(cells))
    print("\n== Injected transfers (odd months): recall of start and end, by arm ==")
    for kind in ("eval_held", "eval_fitted"):
        for arm in ARM_ORDER:
            sub = injected.filter((pl.col("arm") == arm) & (pl.col("kind") == kind))
            unmet = sub.filter(pl.col("n_receivers") == 0)
            met = sub.filter(pl.col("n_receivers") > 0)
            long_i = sub.filter(pl.col("hours") > 24)
            print(
                f"{kind} | {arm} | n={sub.height} | found {sub['found'].mean():.3f} | onset {sub['onset_found'].mean():.3f} | "
                f"over 1 day {long_i['found'].mean():.3f} | unmetered receiver {unmet['found'].mean():.3f} (n={unmet.height}) | "
                f"metered {met['found'].mean():.3f} (n={met.height})"
            )
    print("\n== Alarm accounting (mean over folds) ==")
    for arm in ARM_ORDER:
        sub = alarms.filter(pl.col("arm") == arm)
        unx = sub.filter(~pl.col("explained") & pl.col("odd"))
        bal = unx.filter((pl.col("n_opposite") >= 1) & pl.col("opp_ratio").is_between(0.4, 2.5))
        allm = sub.filter(~pl.col("explained"))
        print(
            f"{arm} | alarms (all months) {sub.height / 3:.0f} | unexplained odd {unx.height / 3:.0f} | balanced-opposite among those {bal.height / 3:.1f} | "
            f"unexplained all months {allm.height / 3:.0f} | per Primary-month (odd months) {unx.height / 3 / (16 * 27):.3f}"
        )
    print("\n== Candidate ceilings (real record) ==")
    table = real_candidates(base=Base())[0]
    for name, srcs in (("RO-NC candidates (x)", ("x",)), ("x + pair candidates", ("x", "p"))):
        cand = table.filter(pl.col("source").is_in(list(srcs)))
        onset = found = 0
        for e in real_transfers().filter(pl.col("hours") > 24).iter_rows(named=True):
            hit = cand.filter(
                pl.col("series").is_in(e["series"]) & ((pl.col("stamp") - e["onset"]).abs() <= 4)
            )
            onset += hit.height > 0
            found += any(
                abs(x - e["end"]) <= end_tolerance(onset=e["onset"], end=e["end"])
                for x in hit["end_est"].to_list()
            )
        print(
            f"{name}: onset candidate for {onset}/35, start-and-end candidate for {found}/35, {cand.height} candidates"
        )
    print("\n== Which events each arm finds that RO-NC misses (lettered pair, hours, MW) ==")
    base_found = set(
        long.filter((pl.col("arm") == BASELINE) & pl.col("found"))["card_id"].to_list()
    )
    for arm in ARM_ORDER:
        if arm == BASELINE:
            continue
        gained = long.filter(
            (pl.col("arm") == arm) & pl.col("found") & ~pl.col("card_id").is_in(list(base_found))
        )
        lost = base_found - set(
            long.filter((pl.col("arm") == arm) & pl.col("found"))["card_id"].to_list()
        )
        text = "; ".join(
            f"{label[r['pair']]} {r['hours']:.0f} h {r['moved_mw_p50']:.1f} MW"
            for r in gained.iter_rows(named=True)
        )
        print(f"{arm}: gains {gained.height} [{text}]; loses {len(lost)}")
    for fold in range(3):
        print(
            f"fold {fold}:",
            json.dumps(
                {
                    k: v
                    for k, v in json.loads((WORK_DIR / f"summary_f{fold}.json").read_text()).items()
                    if k
                    in (
                        "tree_config",
                        "n_parameters",
                        "stack_coefficients",
                        "train_rows",
                        "train_positives",
                    )
                }
            ),
        )


if __name__ == "__main__":
    main()
