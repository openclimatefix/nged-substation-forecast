# ruff: noqa: E501, PLR0912, PLR0915
"""Contrasts of the degraded arms and the new heads against BAND_HEAD, with intervals.

Reads experiment 297's stored scores (K0, BAND_HEAD, BAND_HEAD_HELDOUT, PERFECT_ADD), this
experiment's degraded-arm scores (`deg/`) and head scores (`heads/`), and prints every number the
page quotes. Rows, groups, weights, curtailment cost, and intervals are experiment 297's (whole
shifts or whole calendar months resampled, 95%, 20,000 draws).

    PYTHONPATH=... uv run --no-sync python e00312_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import e00297_report as r297
import numpy as np
import polars as pl
from e00312_common import DEG_DIR, DEGRADED_ARMS, E297_DIR, HEADS_DIR, RESULTS_DIR

BAR: Final[str] = "BAND_HEAD"
COLS: Final[tuple[str, ...]] = (
    "qs_obs",
    "tw95_obs",
    "q01",
    "q05",
    "q95",
    "q99",
    "ae_median_obs",
    "ae_mean_obs",
)
META: Final[list[str]] = [
    "site", "world", "period", "row", "position", "run", "power_fcst_init_time", "lead_hours",
    "event", "phase", "in_shift", "event_hours", "fraction", "onset_month", "actual_nra", "actual_obs",
]  # fmt: skip
HEAD_VARIANTS: Final[tuple[str, ...]] = ("BASE", "PIT", "IND", "PIT_IND", "IND_MW")
FRACTIONS: Final[tuple[float, ...]] = r297.FRACTIONS


def read_long(*, directory: Path, arms: list[str] | None) -> pl.DataFrame:
    """Concatenate score files, keeping only `arms` and the score columns."""
    parts = []
    for path in sorted(directory.glob("D*_*.parquet")):
        frame = pl.read_parquet(path)
        if arms is not None:
            frame = frame.filter(pl.col("arm").is_in(arms))
        have = [c for c in COLS if c in frame.columns]
        parts.append(frame.select("arm", "site", "world", "row", *have))
    return pl.concat(parts)


def load_wide() -> pl.DataFrame:
    """One row per scored row, every arm's score columns side by side as `<col>__<arm>`."""
    first = pl.concat(
        [
            pl.read_parquet(p).filter(pl.col("arm") == "K0")
            for p in sorted((E297_DIR / "scores").glob("D*_*.parquet"))
        ]
    )
    wide = first.select(META)
    sources = [
        read_long(directory=E297_DIR / "scores", arms=["K0", "PERFECT_ADD"]),
        read_long(directory=E297_DIR / "scores2", arms=[BAR]),
        read_long(directory=E297_DIR / "scores3", arms=None),
        read_long(directory=DEG_DIR, arms=None),
        read_long(directory=HEADS_DIR, arms=None),
    ]
    for long in sources:
        for arm in sorted(long["arm"].unique().to_list()):
            part = long.filter(pl.col("arm") == arm)
            part = part.select(
                "site",
                "world",
                "row",
                *[pl.col(c).alias(f"{c}__{arm}") for c in COLS if c in part.columns],
            )
            wide = wide.join(part, on=["site", "world", "row"], how="left", validate="1:1")
    return wide


def cost_of(
    *, wide: pl.DataFrame, constants: dict[str, Any], arm: str, fraction: float
) -> np.ndarray:
    """Node-only curtailment cost per row of an arm against observed power (NaN outside 12 to 36 h)."""
    cap = np.array([constants[s]["capacity_mw"] for s in wide["site"]]) / 100.0
    peak = np.array([constants[s]["max_before_points"] for s in wide["site"]]) * cap
    lead = wide["lead_hours"].to_numpy()
    inside = (lead >= r297.LEAD_MIN_HOURS) & (lead < r297.LEAD_MAX_HOURS)
    limit = fraction * peak
    actual = wide["actual_obs"].to_numpy().astype(np.float64) * cap
    q95 = wide[f"q95__{arm}"].to_numpy().astype(np.float64) * cap
    planned = np.maximum(q95 - limit, 0.0) * 0.5
    breach = np.maximum(actual - planned / 0.5 - limit, 0.0) * 0.5
    cost = r297.PRICE_GBP_PER_MWH * (planned + r297.BREACH_RATIO * breach)
    return np.where(inside, cost, np.nan)


def main() -> None:
    """Compute and print every number the page quotes."""
    import json as _json

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    constants = _json.loads((E297_DIR / "scores" / "site_constants.json").read_text())
    wide = load_wide()
    print("rows", wide.height, flush=True)
    rng = np.random.default_rng(312)
    phase = wide["phase"].to_numpy()
    weight = np.where(phase == "ORDINARY", r297.ORDINARY_WEIGHT, 1.0)
    units = r297.units_of(wide=wide)
    groups = r297.group_masks(wide=wide)
    sites = wide["site"].to_numpy()
    world = wide["world"].to_numpy()
    odd_world = world % 2 == 1
    origin_month = wide["power_fcst_init_time"].dt.month().to_numpy()
    even_month = origin_month % 2 == 0
    out: dict[str, Any] = {"n_rows": wide.height}
    metric_cache: dict[tuple[str, str], np.ndarray] = {}

    def metric(arm: str, name: str) -> np.ndarray:
        key = (arm, name)
        if key not in metric_cache:
            if name.startswith("cost"):
                metric_cache[key] = cost_of(
                    wide=wide, constants=constants, arm=arm, fraction=float(name[4:])
                )
            elif f"{name}__{arm}" in wide.columns:
                metric_cache[key] = wide[f"{name}__{arm}"].to_numpy().astype(np.float64)
            else:
                metric_cache[key] = np.full(wide.height, np.nan)
        return metric_cache[key]

    def gain(*, base: str, arm: str, name: str, mask: np.ndarray) -> dict[str, float]:
        b, a = metric(base, name), metric(arm, name)
        return r297.interval(
            gain=(b - a)[mask], base=b[mask], weight=weight[mask], units=units[mask], rng=rng
        )

    def show(tag: str, res: dict[str, float]) -> None:
        r297.show(tag, res)

    # 0. Reproduction checks.
    print("=== checks: ADD_PERFECT against stored PERFECT_ADD; HEAD_BASE_cross against BAND_HEAD")
    for new, old in (("ADD_PERFECT", "PERFECT_ADD"), ("HEAD_BASE_cross", "BAND_HEAD")):
        a, b = metric(new, "qs_obs"), metric(old, "qs_obs")
        ok = np.isfinite(a) & np.isfinite(b)
        out[f"repro_{new}"] = {
            "max_abs_diff": float(np.max(np.abs(a[ok] - b[ok]))),
            "n": int(ok.sum()),
        }
        print(new, out[f"repro_{new}"], flush=True)
    # 1. Degraded arms against the bar.
    print(
        "=== degraded arms against BAND_HEAD (gain = BAND_HEAD minus arm; positive is better for the arm)"
    )
    cells = {
        "cost0.5 G5": ("cost0.5", "G5 near"),
        "cost0.75 G5": ("cost0.75", "G5 near"),
        "cost0.9 G5": ("cost0.9", "G5 near"),
        "cost0.5 Gs": ("cost0.5", "Gs shift rows"),
        "cost0.5 G2": ("cost0.5", "G2 before alarm"),
        "cost0.5 G3": ("cost0.5", "G3 during"),
        "cost0.5 G4": ("cost0.5", "G4 after"),
        "tw95 G5": ("tw95_obs", "G5 near"),
        "tw95 Gs": ("tw95_obs", "Gs shift rows"),
        "tw95 G2": ("tw95_obs", "G2 before alarm"),
        "tw95 G3": ("tw95_obs", "G3 during"),
        "tw95 G4": ("tw95_obs", "G4 after"),
    }
    sweep = []
    for arm in DEGRADED_ARMS:
        for cell, (name, group) in cells.items():
            res = gain(base=BAR, arm=arm, name=name, mask=groups[group])
            sweep.append({"arm": arm, "cell": cell, **res})
            show(f"{arm:12s} vs bar {cell:12s}", res)
        for label, extra in (("without D11", sites != "D11"), ("without D16", sites != "D16")):
            res = gain(base=BAR, arm=arm, name="cost0.5", mask=groups["G5 near"] & extra)
            sweep.append({"arm": arm, "cell": f"cost0.5 G5 {label}", **res})
            show(f"{arm:12s} vs bar cost0.5 G5 {label:12s}", res)
        # held-out check: odd-seed worlds with the stored held-out head
        res = gain(
            base="BAND_HEAD_HELDOUT", arm=arm, name="cost0.5", mask=groups["G5 near"] & odd_world
        )
        sweep.append({"arm": arm, "cell": "cost0.5 G5 odd worlds vs held-out head", **res})
        show(f"{arm:12s} vs held-out head cost0.5 G5 odd", res)
        # ordinary-row harm: arm minus perfect arm, and K0 as control
        for name in ("qs_obs", "cost0.5"):
            res = gain(base="ADD_PERFECT", arm=arm, name=name, mask=groups["G1 ordinary"])
            sweep.append({"arm": arm, "cell": f"G1 {name} vs ADD_PERFECT", **res})
            show(f"{arm:12s} vs ADD_PERFECT G1 {name}", res)
    out["sweep"] = sweep
    print("=== reference arms against the bar")
    refs = []
    for arm in ("K0", "ADD_PERFECT"):
        for cell in ("cost0.5 G5", "tw95 G5", "cost0.5 Gs"):
            name, group = cells[cell]
            res = gain(base=BAR, arm=arm, name=name, mask=groups[group])
            refs.append({"arm": arm, "cell": cell, **res})
            show(f"{arm:12s} vs bar {cell}", res)
        for name in ("qs_obs", "cost0.5"):
            res = gain(base=BAR, arm=arm, name=name, mask=groups["G1 ordinary"])
            refs.append({"arm": arm, "cell": f"G1 {name}", **res})
            show(f"{arm:12s} vs bar G1 {name}", res)
    out["refs"] = refs
    # 2. Heads against the matching base head.
    print("=== heads against the base head of the same mode (gain = base minus head)")
    head_rows = []
    modes = {
        "cross": np.ones(wide.height, bool),
        "held": odd_world,
        "size": np.ones(wide.height, bool),
    }
    for mode, scope in modes.items():
        for variant in HEAD_VARIANTS[1:]:
            arm, base = f"HEAD_{variant}_{mode}", f"HEAD_BASE_{mode}"
            if f"qs_obs__{arm}" not in wide.columns:
                continue
            both = np.isfinite(metric(arm, "qs_obs")) & np.isfinite(metric(base, "qs_obs")) & scope
            for read, extra in (
                ("all months", np.ones(wide.height, bool)),
                ("even origin months", even_month),
            ):
                for cell in ("tw95 G5", "cost0.5 G5", "tw95 Gs", "tw95 G4", "cost0.5 Gs"):
                    name, group = cells[cell]
                    res = gain(base=base, arm=arm, name=name, mask=groups[group] & both & extra)
                    head_rows.append(
                        {"mode": mode, "variant": variant, "read": read, "cell": cell, **res}
                    )
                    show(f"{arm:20s} vs {base} [{read}] {cell}", res)
                res = gain(
                    base=base, arm=arm, name="qs_obs", mask=groups["G1 ordinary"] & both & extra
                )
                head_rows.append(
                    {"mode": mode, "variant": variant, "read": read, "cell": "G1 qs", **res}
                )
                show(f"{arm:20s} vs {base} [{read}] G1 qs", res)
    out["heads"] = head_rows
    # 3. Heads against K0 and the bar (cross), headline numbers beside the stored head.
    print("=== heads and base head against K0 (cross-fitted)")
    for arm in ("BAND_HEAD", *[f"HEAD_{v}_cross" for v in HEAD_VARIANTS]):
        for cell in ("tw95 G5", "cost0.5 G5"):
            name, group = cells[cell]
            show(f"{arm:20s} vs K0 {cell}", gain(base="K0", arm=arm, name=name, mask=groups[group]))
    # 4. Tail coverage after a shift (G4) and on ordinary rows (G1), observed power.
    print("=== tail coverage against observed power (share outside the band)")
    coverage = []
    arms_cov = [
        "K0",
        BAR,
        "ADD_PERFECT",
        "ADD_R50",
        "ADD_P50",
        "ADD_B50",
        *[f"HEAD_{v}_cross" for v in HEAD_VARIANTS],
    ]
    actual = wide["actual_obs"].to_numpy().astype(np.float64)
    for arm in arms_cov:
        for gname in ("G4 after", "G1 ordinary"):
            mask = groups[gname]
            for side, level, ind in (
                ("below q01", 0.01, actual < metric(arm, "q01")),
                ("below q05", 0.05, actual < metric(arm, "q05")),
                ("above q95", 0.05, actual > metric(arm, "q95")),
                ("above q99", 0.01, actual > metric(arm, "q99")),
            ):
                res = r297.interval(
                    gain=ind[mask].astype(np.float64),
                    base=np.ones(mask.sum()),
                    weight=weight[mask],
                    units=units[mask],
                    rng=rng,
                )
                coverage.append({"arm": arm, "group": gname, "side": side, "nominal": level, **res})
                print(
                    f"{arm:16s} {gname:12s} {side}: {100 * res['mean']:.2f}% ({100 * res['mean_lo']:.2f}, {100 * res['mean_hi']:.2f}) nominal {100 * level:.0f}%"
                )
    out["coverage"] = coverage
    # 5. Absolute scores and checks.
    print(
        "=== absolute mean scores (points of effective capacity; cost in GBP per scored half-hour)"
    )
    absolute = []
    all_arms = sorted({c.split("__")[1] for c in wide.columns if "__" in c})
    for arm in all_arms:
        for gname in ("G1 ordinary", "G4 after", "G5 near"):
            m = groups[gname]
            row: dict[str, Any] = {"arm": arm, "group": gname}
            for name in ("qs_obs", "tw95_obs", "cost0.5"):
                v = metric(arm, name)[m]
                ok = np.isfinite(v)
                row[name] = (
                    float(np.sum(v[ok] * weight[m][ok]) / np.sum(weight[m][ok]))
                    if ok.any()
                    else float("nan")
                )
            absolute.append(row)
            print(arm, gname, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)})
    out["absolute"] = absolute
    print("=== median-forecast minus mean-forecast absolute error, per site (points)")
    checks = []
    for site in sorted(set(sites)):
        for arm in ("K0", BAR, "ADD_PERFECT", "HEAD_PIT_cross", "HEAD_IND_MW_cross"):
            m = sites == site
            diff = float(
                np.nanmean(metric(arm, "ae_median_obs")[m] - metric(arm, "ae_mean_obs")[m])
            )
            checks.append({"site": site, "arm": arm, "diff": diff})
            print(f"{site:4s} {arm:20s} {diff:+.4f}")
    out["checks"] = checks
    out["counts"] = {
        "worlds": len(set(world.tolist())),
        "shifts": len(np.unique(units[units > 0])),
        "rows_by_phase": {p: int((phase == p).sum()) for p in sorted(set(phase))},
    }
    print(json.dumps(out["counts"], indent=2))
    (RESULTS_DIR / "report.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
