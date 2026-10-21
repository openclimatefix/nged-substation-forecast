r"""Check, tune, and gate the arms of experiment 145.

Stages (each reads only what the earlier stages stored):

    check    confirm that recipe R's 14-day profile reproduces experiment 115's stored arm R
    tune     score every analog setting on the tuning rows' D+1 median error and choose
    gate     inject known patterns into real residuals and measure how much each arm recovers

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \\
        uv run --no-sync python ml-experiments/00145-*/e00145_run.py <stage> [--workers 4]
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Final

import e00145_analog as analog
import e00145_arms as arms
import e00145_common as common
import e00145_site as site_module
import numpy as np
from studies.dp_ens import LeadHourSpread
from studies.dp_gam import ResidualSpread

SITES: Final[tuple[str, ...]] = common.SCORED_SITES
TUNED_PATH: Final[Path] = common.WORK_DIR / "tuned.json"
GATE_PATH: Final[Path] = common.WORK_DIR / "gate.json"
SRC_FRAMES_DIR: Final[Path] = common.SRC_115_WORK_DIR / "frames"
KAPPAS: Final[tuple[float, ...]] = (0.5, 0.75, 1.0, 1.25)
NEIGHBOURS: Final[tuple[int, ...]] = (3, 5, 8, 12, 20)
SEASON_WEEKS: Final[tuple[int, ...]] = (4, 8, 13)
WEIGHTS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
R_KAPPA: Final[float] = 0.75
RECENT_DAYS: Final[int] = 300
"""The post hoc recent-only arm leaves out source days older than this, so no day of the previous
year enters."""
R_SPEC: Final[analog.AnalogSpec] = analog.AnalogSpec(kappa=R_KAPPA, k=1, weeks=1, weight=0.0)
"""Recipe R: weight 0 returns the 14-day profile alone."""
HOLIDAY_SPEC: Final[analog.AnalogSpec] = analog.AnalogSpec(
    kappa=R_KAPPA, k=3, weeks=0, weight=0.5, holiday_only=True
)
"""ANALOG-holiday: fixed before any fit; the tuning months hold too few repeated holidays."""
REFERENCE_PURE: Final[analog.AnalogSpec] = analog.AnalogSpec(kappa=0.75, k=8, weeks=8, weight=1.0)
REFERENCE_MIXED: Final[analog.AnalogSpec] = analog.AnalogSpec(kappa=0.75, k=8, weeks=8, weight=0.5)
"""The fixed settings at which the gate's pass conditions are read."""
INJECTION_SHIFT: Final[float] = 3.0
INJECTION_HALF_HOURS: Final[range] = range(34, 42)
"""Local 17:00 to 20:30 start times: the half-hours of 17:00 to 21:00."""
SATURDAY_FIRST: Final[date] = date(2024, 11, 2)
SATURDAY_COUNT: Final[int] = 8
FIRST_SCORED_SATURDAY: Final[int] = 3
"""Recovery is read from the third injected Saturday on, when two earlier ones carry the shift."""
HOLIDAY_SOURCE: Final[date] = date(2024, 5, 27)
HOLIDAY_TARGET: Final[date] = date(2025, 5, 26)
COLD_LIMIT_C: Final[float] = 5.0
COLD_MONTHS: Final[frozenset[tuple[int, int]]] = frozenset({(2024, 11), (2025, 1)})
FIRST_SCORED_COLD_DAY: Final[int] = 6
"""Cold-day recovery is read from the sixth injected day on."""
FAMILIES: Final[dict[str, tuple[float, bool]]] = {
    "ANALOG-k": (1.0, True),
    "ANALOG+R": (-1.0, True),
    "ANALOG+R-notemp": (-1.0, False),
}
"""Tuned families: weight 1 fixes the weight, -1 tunes it; True ranks by temperature."""


@lru_cache(maxsize=2)
def get_site(site: str) -> site_module.SiteData:
    """Load a site once per worker process."""
    return site_module.load_site(site=site)


def load_k0(*, site: str) -> tuple[arms.Correction, np.ndarray]:
    """Return experiment 115's K0 correction and K0's median forecast at every row."""
    with np.load(common.SRC_115_WORK_DIR / "corrections" / f"K0_{site}.npz") as stored:
        k0 = arms.Correction(
            level=stored["level"], anomaly=stored["anomaly"], shape=stored["shape"]
        )
    with np.load(common.SRC_115_WORK_DIR / "spread" / f"K0_{site}.npz") as stored:
        tables, factors = stored["tables"], stored["factors"]
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


def site_days(*, site: str, actual: np.ndarray | None = None) -> analog.AnalogDays:
    """Return a site's source days for the real readings, or for an injected copy."""
    data = get_site(site)
    _k0, q50 = load_k0(site=site)
    return analog.build_days(
        data=data,
        k0_q50=q50,
        actual=data.actual if actual is None else actual,
        temperature_c=analog.forecast_temperature(data=data),
    )


def spec_name(*, family: str, spec: analog.AnalogSpec) -> str:
    """Return a candidate setting's name."""
    return f"{family}_k{spec.k}_n{spec.weeks}_l{spec.weight:g}_kappa{spec.kappa:g}"


def spec_table() -> dict[str, tuple[str, analog.AnalogSpec]]:
    """Return every tuned candidate, by name, with its family."""
    table: dict[str, tuple[str, analog.AnalogSpec]] = {}
    for family, (fixed_weight, temperature) in FAMILIES.items():
        weights = (fixed_weight,) if fixed_weight > 0 else WEIGHTS
        for kappa in KAPPAS:
            for k in NEIGHBOURS:
                for weeks in SEASON_WEEKS:
                    for weight in weights:
                        spec = analog.AnalogSpec(
                            kappa=kappa, k=k, weeks=weeks, weight=weight, temperature=temperature
                        )
                        table[spec_name(family=family, spec=spec)] = (family, spec)
    return table


def _mae(
    *, data: site_module.SiteData, q50: np.ndarray, shift: np.ndarray, rows: np.ndarray
) -> list[float]:
    error = np.abs(data.actual[rows] - (q50[rows] + shift[rows]))
    return [float(error.sum()), int(rows.sum())]


def _tune_job(site: str) -> dict:
    data = get_site(site)
    _k0, q50 = load_k0(site=site)
    days = site_days(site=site)
    rows = data.period_rows(period="tuning") & data.scorable & data.d1_rows()
    out: dict[str, list[float]] = {
        "K0": _mae(data=data, q50=q50, shift=np.zeros(data.n_rows), rows=rows)
    }
    extra = {
        "R-exact": R_SPEC,
        "ANALOG-holiday": HOLIDAY_SPEC,
        "ANALOG-k-ref": REFERENCE_PURE,
        "ANALOG+R-ref": REFERENCE_MIXED,
    }
    for kappa in KAPPAS:
        extra[f"R-kappa{kappa:g}"] = analog.AnalogSpec(kappa=kappa, k=1, weeks=1, weight=0.0)
    for name, spec in extra.items():
        shift = analog.shift_at_rows(
            data=data, by_run=analog.analog_shift_by_run(days=days, spec=spec)
        )
        out[name] = _mae(data=data, q50=q50, shift=shift, rows=rows)
    for name, (_family, spec) in spec_table().items():
        shift = analog.shift_at_rows(
            data=data, by_run=analog.analog_shift_by_run(days=days, spec=spec)
        )
        out[name] = _mae(data=data, q50=q50, shift=shift, rows=rows)
    return {"site": site, "scores": out}


def stage_check(*, pool: ProcessPoolExecutor) -> None:
    """Confirm the 14-day profile reproduces experiment 115's stored recipe R at every site."""
    for line in pool.map(_check_job, SITES):
        print(line)


def _check_job(site: str) -> str:
    data = get_site(site)
    k0, _q50 = load_k0(site=site)
    days = site_days(site=site)
    mine = analog.shift_at_rows(
        data=data, by_run=analog.analog_shift_by_run(days=days, spec=R_SPEC)
    )
    with np.load(common.SRC_115_WORK_DIR / "corrections" / f"R_{site}.npz") as stored:
        theirs = stored["shape"]
        same_level = np.allclose(stored["level"], k0.level) and np.allclose(
            stored["anomaly"], k0.anomaly
        )
    worst = float(np.max(np.abs(mine - theirs)))
    return f"{site}: max |shift difference| {worst:.2e}, level and anomaly equal K0's: {same_level}"


def stage_tune(*, pool: ProcessPoolExecutor) -> None:
    """Choose the settings of ANALOG-k, ANALOG+R, and ANALOG+R-notemp on the tuning rows."""
    results = list(pool.map(_tune_job, SITES))
    pooled: dict[str, float] = {}
    for name in results[0]["scores"]:
        total = np.zeros(2)
        for result in results:
            if result["site"] in common.DECISION_SITES:
                total += np.array(result["scores"][name])
        pooled[name] = float(total[0] / total[1])
    chosen = {
        family: min((n for n in pooled if n.startswith(f"{family}_k")), key=lambda n: pooled[n])
        for family in FAMILIES
    }
    content = {"pooled_mae": pooled, "chosen": chosen, "per_site": results}
    common.write_json_atomic(path=TUNED_PATH, content=content)
    for family, name in chosen.items():
        print(f"{family}: {name} MAE {pooled[name]:.4f} (K0 {pooled['K0']:.4f})")
    for name in (
        "R-exact",
        "R-kappa0.5",
        "R-kappa1",
        "R-kappa1.25",
        "ANALOG-holiday",
        "ANALOG-k-ref",
        "ANALOG+R-ref",
    ):
        print(f"{name}: MAE {pooled[name]:.4f}")


def chosen_spec(*, family: str) -> analog.AnalogSpec:
    """Return the tuned setting of a family, or the fixed setting of ANALOG-holiday."""
    if family == "ANALOG-holiday":
        return HOLIDAY_SPEC
    if family == "R":
        return R_SPEC
    if family == "ANALOG+R-recent":
        return replace(chosen_spec(family="ANALOG+R"), max_age_days=RECENT_DAYS)
    name = json.loads(TUNED_PATH.read_text())["chosen"][family]
    return spec_table()[name][1]


def _day_number(day: date) -> int:
    return (day - analog.EPOCH_DATE).days


def _valid_day(*, data: site_module.SiteData) -> np.ndarray:
    """Return the UTC day number of each row's valid time."""
    return (data.cache.end_times[data.position].astype("datetime64[D]") - analog.EPOCH).astype(
        np.int64
    )


def _recovery(
    *,
    base: analog.AnalogDays,
    injected: analog.AnalogDays,
    spec: analog.AnalogSpec,
    runs: np.ndarray,
) -> float:
    """Return the mean share of the shift recovered at the given runs' injected half-hours."""
    half_hours = list(INJECTION_HALF_HOURS)
    before = analog.analog_shift_by_run(days=base, spec=spec)
    after = analog.analog_shift_by_run(days=injected, spec=spec)
    delta = (after - before)[runs][:, half_hours]
    return float(delta.mean() / INJECTION_SHIFT)


def _inject(*, data: site_module.SiteData, valid_days: list[int]) -> np.ndarray:
    injected = data.actual.copy()
    hit = np.isin(_valid_day(data=data), valid_days) & np.isin(
        data.row_half_hour, list(INJECTION_HALF_HOURS)
    )
    injected[hit] = injected[hit] + INJECTION_SHIFT
    return injected


def _placebo(spec: analog.AnalogSpec) -> analog.AnalogSpec:
    return analog.AnalogSpec(
        kappa=spec.kappa,
        k=spec.k,
        weeks=spec.weeks,
        weight=spec.weight,
        temperature=spec.temperature,
        placebo_shift=3,
    )


def _gate_job(site: str) -> dict:
    data = get_site(site)
    base = site_days(site=site)
    tuned = {family: chosen_spec(family=family) for family in FAMILIES}
    # G1: a recurring day type (Saturday evenings).
    saturdays = [_day_number(SATURDAY_FIRST) + 7 * i for i in range(SATURDAY_COUNT)]
    injected = site_days(site=site, actual=_inject(data=data, valid_days=saturdays))
    measured = np.flatnonzero(np.isin(base.targets, saturdays[FIRST_SCORED_SATURDAY - 1 :]))
    g1_specs = {
        "R": R_SPEC,
        "ANALOG-k-ref": REFERENCE_PURE,
        "ANALOG+R-ref": REFERENCE_MIXED,
        "ANALOG-k-ref-placebo": _placebo(REFERENCE_PURE),
        "ANALOG+R-ref-placebo": _placebo(REFERENCE_MIXED),
        "ANALOG-k-tuned": tuned["ANALOG-k"],
        "ANALOG+R-tuned": tuned["ANALOG+R"],
        "ANALOG+R-tuned-placebo": _placebo(tuned["ANALOG+R"]),
    }
    g1_specs["ANALOG-k-k3-n4"] = analog.AnalogSpec(kappa=0.75, k=3, weeks=4, weight=1.0)
    g1_specs["ANALOG-k-k3-n4-notemp"] = analog.AnalogSpec(
        kappa=0.75, k=3, weeks=4, weight=1.0, temperature=False
    )
    g1 = {
        name: _recovery(base=base, injected=injected, spec=spec, runs=measured)
        for name, spec in g1_specs.items()
    }
    # G0: readings of the last Saturday's own day and of the origin's day must not move its profile.
    last = measured[-1:]
    final = int(base.targets[last[0]])
    leak_days = [final, final - 1]
    leaked = site_days(site=site, actual=_inject(data=data, valid_days=leak_days))
    leak = {}
    for name, spec in g1_specs.items():
        before = analog.analog_shift_by_run(days=base, spec=spec)[last]
        after = analog.analog_shift_by_run(days=leaked, spec=spec)[last]
        leak[name] = float(np.max(np.abs(after - before)))
    # G2: a recurring holiday (the evening of the Spring Bank Holiday, a year apart).
    holiday = site_days(
        site=site, actual=_inject(data=data, valid_days=[_day_number(HOLIDAY_SOURCE)])
    )
    run = np.flatnonzero(base.targets == _day_number(HOLIDAY_TARGET))
    g2_specs = {"R": R_SPEC, "ANALOG-holiday": HOLIDAY_SPEC, "ANALOG+R-tuned": tuned["ANALOG+R"]}
    g2 = {
        name: _recovery(base=base, injected=holiday, spec=spec, runs=run)
        for name, spec in g2_specs.items()
    }
    # G3: a recurring weather state (cold days of the tuning months).
    dates = [date.fromordinal(analog.EPOCH_DATE.toordinal() + int(t)) for t in base.targets]
    cold = [
        int(t)
        for t, d, temperature in zip(base.targets, dates, base.temperature_c, strict=True)
        if (d.year, d.month) in COLD_MONTHS and temperature < COLD_LIMIT_C
    ]
    g3: dict[str, float] = {}
    if len(cold) > FIRST_SCORED_COLD_DAY:
        cold_injected = site_days(site=site, actual=_inject(data=data, valid_days=cold))
        cold_runs = np.flatnonzero(np.isin(base.targets, cold[FIRST_SCORED_COLD_DAY - 1 :]))
        g3_specs = {
            "ANALOG+R-ref": REFERENCE_MIXED,
            "ANALOG+R-ref-notemp": analog.AnalogSpec(
                kappa=0.75, k=8, weeks=8, weight=0.5, temperature=False
            ),
        }
        g3 = {
            name: _recovery(base=base, injected=cold_injected, spec=spec, runs=cold_runs)
            for name, spec in g3_specs.items()
        }
    return {
        "site": site,
        "g0_max_change": leak,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "n_cold_days": len(cold),
    }


def stage_gate(*, pool: ProcessPoolExecutor) -> None:
    """Measure the share of each injected pattern that each correction recovers."""
    results = list(pool.map(_gate_job, SITES))
    median: dict[str, dict[str, float]] = {}
    for gate in ("g1", "g2", "g3"):
        names = {n for r in results for n in r[gate]}
        median[gate] = {
            name: float(
                np.median(
                    [
                        r[gate][name]
                        for r in results
                        if r["site"] in common.DECISION_SITES and name in r[gate]
                    ]
                )
            )
            for name in sorted(names)
        }
    common.write_json_atomic(path=GATE_PATH, content={"median_recovery": median, "sites": results})
    print(json.dumps(median, indent=1))


def main() -> None:
    """Run one stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "tune", "gate"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        {"check": stage_check, "tune": stage_tune, "gate": stage_gate}[args.stage](pool=pool)
    print(f"{args.stage}: {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
