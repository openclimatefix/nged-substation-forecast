"""Tune, gate, and build the arms of experiment 134.

Stages (each reads only what the earlier stages stored):

    tune     score every profile setting on the tuning rows' D+1 median error and choose
    gate     inject a Saturday-evening shift into real residuals and measure its recovery
    frames   build each arm's correction, spread, and frame for every site

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \\
        uv run --no-sync python ml-experiments/00134-*/e00134_run.py <stage> [--workers 4]
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Final

import e00134_arms as arms
import e00134_common as common
import e00134_profiles as profiles
import e00134_site as site_module
import numpy as np
import polars as pl
import torch
from studies.dp_ens_weather import N_LEAD_BANDS

SITES: Final[tuple[str, ...]] = common.SCORED_SITES
FRAMES_DIR: Final[Path] = common.WORK_DIR / "frames"
TUNED_PATH: Final[Path] = common.WORK_DIR / "tuned.json"
GATE_PATH: Final[Path] = common.WORK_DIR / "gate.json"
SRC_FRAMES_DIR: Final[Path] = common.SRC_115_WORK_DIR / "frames"
KAPPAS: Final[tuple[float, ...]] = (0.5, 0.75, 1.0)
WEIGHTS: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75, 1.0)
WEEKS: Final[tuple[int, ...]] = (2, 3, 4)
R_KAPPA: Final[float] = 0.75
INJECTION_SHIFT: Final[float] = 3.0
INJECTION_FIRST_SATURDAY: Final[datetime] = datetime(2024, 11, 2, tzinfo=UTC)
INJECTION_WEEKS: Final[int] = 8
INJECTION_HALF_HOURS: Final[range] = range(34, 42)
"""Local 17:00 to 20:30 start times: the half-hours of 17:00 to 21:00."""
FIRST_SCORED_SATURDAY: Final[int] = 3
"""Recovery is read from the third injected Saturday on, when two earlier ones carry the shift."""


@lru_cache(maxsize=2)
def get_site(site: str) -> site_module.SiteData:
    """Load a site once per worker process."""
    torch.set_num_threads(1)
    return site_module.load_site(site=site)


def load_k0(*, site: str) -> tuple[arms.Correction, np.ndarray]:
    """Return experiment 115's K0 correction and K0's median forecast at every row."""
    with np.load(common.SRC_115_WORK_DIR / "corrections" / f"K0_{site}.npz") as stored:
        k0 = arms.Correction(
            level=stored["level"], anomaly=stored["anomaly"], shape=stored["shape"]
        )
    with np.load(common.SRC_115_WORK_DIR / "spread" / f"K0_{site}.npz") as stored:
        tables, factors = stored["tables"], stored["factors"]
    from studies.dp_gam import ResidualSpread
    from studies.dp_ens import LeadHourSpread

    spread = LeadHourSpread(
        by_lead_band=tuple(ResidualSpread(table=t) for t in tables), factors=factors
    )
    data = get_site(site)
    quantiles, rows = arms.quantiles_for_runs(
        data=data, correction=k0.total, spread=spread, run_indices=np.arange(len(data.runs))
    )
    q50 = np.full(data.n_rows, np.nan)
    q50[rows] = quantiles[:, 5]
    return k0, q50


def spec_table() -> dict[str, profiles.ProfileSpec]:
    """Return every candidate setting by name."""
    table: dict[str, profiles.ProfileSpec] = {}
    for kappa in KAPPAS:
        table[f"Rh_k{kappa:g}"] = profiles.ProfileSpec(kappa=kappa)
        for weeks in (1, *WEEKS):
            table[f"RW_n{weeks}_k{kappa:g}"] = profiles.ProfileSpec(
                kappa=kappa, weeks=weeks, weight=1.0
            )
        for weeks in WEEKS:
            for weight in WEIGHTS[1:-1]:
                for group, label in (("weekday", "RB"), ("day_type", "RBdt")):
                    table[f"{label}_n{weeks}_l{weight:g}_k{kappa:g}"] = profiles.ProfileSpec(
                        kappa=kappa, weeks=weeks, weight=weight, group=group
                    )
    return table


def _tune_job(site: str) -> dict:
    data = get_site(site)
    k0, q50 = load_k0(site=site)
    rows = data.period_rows(period="tuning") & data.scorable & data.d1_rows()
    blocks, targets = profiles.day_blocks(data=data, k0_q50=q50, actual=data.actual)
    out: dict[str, list[float]] = {}
    for name, spec in spec_table().items():
        by_run = profiles.profile_shift_by_run(blocks=blocks, targets=targets, spec=spec)
        shift = profiles.shift_at_rows(data=data, by_run=by_run)
        error = np.abs(data.actual[rows] - (q50[rows] + shift[rows]))
        out[name] = [float(error.sum()), int(rows.sum())]
    base = np.abs(data.actual[rows] - q50[rows])
    out["K0"] = [float(base.sum()), int(rows.sum())]
    return {"site": site, "scores": out}


def stage_tune(*, pool: ProcessPoolExecutor) -> None:
    """Choose the settings of Rh, RW, RB, and RB-day-types on the tuning rows."""
    results = list(pool.map(_tune_job, SITES))
    pooled: dict[str, float] = {}
    for name in results[0]["scores"]:
        total = np.zeros(2)
        for result in results:
            if result["site"] in common.DECISION_SITES:
                total += np.array(result["scores"][name])
        pooled[name] = float(total[0] / total[1])
    families = {
        "Rh": lambda n: n.startswith("Rh_"),
        "L1": lambda n: n.startswith("RW_n1_"),
        "RW": lambda n: n.startswith("RW_n") and not n.startswith("RW_n1_"),
        "RB": lambda n: n.startswith("RB_"),
        "RBdt": lambda n: n.startswith("RBdt_"),
    }
    chosen = {
        family: min((n for n in pooled if test(n)), key=lambda n: pooled[n])
        for family, test in families.items()
    }
    content = {"pooled_mae": pooled, "chosen": chosen, "per_site": results}
    common.write_json_atomic(path=TUNED_PATH, content=content)
    for family, name in chosen.items():
        print(f"{family}: {name} MAE {pooled[name]:.4f} (K0 {pooled['K0']:.4f})")


def chosen_spec(*, family: str) -> profiles.ProfileSpec:
    """Return the tuned setting of an arm family."""
    name = json.loads(TUNED_PATH.read_text())["chosen"][family]
    return spec_table()[name]


def saturdays() -> list[int]:
    """Return the UTC day numbers of the injected Saturdays."""
    first = int((np.datetime64(INJECTION_FIRST_SATURDAY.date(), "D") - profiles.EPOCH).astype(int))
    return [first + 7 * i for i in range(INJECTION_WEEKS)]


def _gate_job(site: str) -> dict:
    data = get_site(site)
    k0, q50 = load_k0(site=site)
    days = saturdays()
    injected = data.actual.copy()
    origin_days = data.run_origins.astype("datetime64[D]")
    target_day = (origin_days + np.timedelta64(1, "D") - profiles.EPOCH).astype(np.int64)
    # Rows of the injected Saturdays' valid times: every row whose valid UTC day is injected.
    valid_day = (
        data.cache.end_times[data.position].astype("datetime64[D]") - profiles.EPOCH
    ).astype(np.int64)
    local_half_hour = data.row_half_hour
    hit = np.isin(valid_day, days) & np.isin(local_half_hour, list(INJECTION_HALF_HOURS))
    injected[hit] = injected[hit] + INJECTION_SHIFT
    recovery: dict[str, float] = {}
    specs = {
        "Rh": chosen_spec(family="Rh"),
        "R": profiles.ProfileSpec(kappa=R_KAPPA, holiday_rule=False),
        "RB": chosen_spec(family="RB"),
        "RW": chosen_spec(family="RW"),
        "RBdt": chosen_spec(family="RBdt"),
    }
    placebo = chosen_spec(family="RB")
    specs["RB-placebo"] = profiles.ProfileSpec(
        kappa=placebo.kappa, weeks=placebo.weeks, weight=placebo.weight, group="placebo"
    )
    measured_runs = np.flatnonzero(np.isin(target_day, days[FIRST_SCORED_SATURDAY - 1 :]))
    for name, spec in specs.items():
        base = profiles.profile_shift_by_run(
            blocks=profiles.day_blocks(data=data, k0_q50=q50, actual=data.actual)[0],
            targets=target_day,
            spec=spec,
        )
        moved = profiles.profile_shift_by_run(
            blocks=profiles.day_blocks(data=data, k0_q50=q50, actual=injected)[0],
            targets=target_day,
            spec=spec,
        )
        delta = (moved - base)[measured_runs][:, list(INJECTION_HALF_HOURS)]
        recovery[name] = float(delta.mean() / INJECTION_SHIFT)
    return {"site": site, "recovery": recovery, "runs": int(len(measured_runs))}


def stage_gate(*, pool: ProcessPoolExecutor) -> None:
    """Measure the share of an injected Saturday-evening shift each correction recovers."""
    results = list(pool.map(_gate_job, SITES))
    names = list(results[0]["recovery"])
    median = {
        name: float(
            np.median([r["recovery"][name] for r in results if r["site"] in common.DECISION_SITES])
        )
        for name in names
    }
    common.write_json_atomic(path=GATE_PATH, content={"median_recovery": median, "sites": results})
    print(json.dumps(median, indent=1))


def main() -> None:
    """Run one stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["tune", "gate", "frames"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.stage == "tune":
            stage_tune(pool=pool)
        elif args.stage == "gate":
            stage_gate(pool=pool)
    print(f"{args.stage}: {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
