"""Score the gate's draws against the five pre-registered readings, for experiment 118.

A reading passes if at least 80% of the draws of its scenarios pass it, separately in every noise
form and regressor condition. The draws of the five decision sites are pooled (100 per scenario,
form, and condition). Readings 1, 3, 4, and 5 block: if any fails, `gate.json` records the failures
and the experiment's replicate and scored runs do not start. Reading 2 (heating tracking) does not
block: a failure labels the heating weight "not interpretable".

1. **PV tracking (S1, S3):** the filtered PV weight 200 days after the ramp starts recovers at least
   70% of the injected change, and its daily correlation with the true path over days 0 to 200 is
   at least 0.7.
2. **Heating tracking (S2, S3):** the mean filtered heating weight 120 days after its ramp starts
   recovers at least 50% of the injected change.
3. **No false movement (S0):** after day 60, no PV, heating, or holiday weight is further than 0.10
   from 1, and the fitted PV and heating `q` are under a tenth of their median in the drift
   scenarios.
4. **No cross-talk (S1, S2):** with only PV moving, the heating weights' mean absolute change after
   day 60 is under 25% of the PV change; with only heating moving, the PV weight's is under 25% of
   the heating change.
5. **Forecast skill follows (S1 to S3):** the day-ahead mean absolute error of KEZ on the stream is
   below GK's.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Final

import e00118_common as common
import e00118_gate as gate
import numpy as np

PV_RECOVERY: Final[float] = 0.70
PV_CORRELATION: Final[float] = 0.70
HEATING_RECOVERY: Final[float] = 0.50
FALSE_MOVEMENT: Final[float] = 0.10
CROSS_TALK_SHARE: Final[float] = 0.25
Q_SHARE: Final[float] = 0.10
POWER_SDS: Final[float] = 3.0
HEATING_READ_INDEX: Final[int] = 92 + gate.HEATING_READ_DAY - 1
PV_READ_INDEX: Final[int] = gate.PV_READ_DAY - 1
BLOCKING_READINGS: Final[tuple[str, ...]] = ("1", "3", "4", "5")


def load_draws() -> dict[tuple[str, str, str], list[dict]]:
    """Return every stored draw, grouped by (scenario, form, condition)."""
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for path in sorted(Path(gate.GATE_DIR).glob("*/*/*/*.json")):
        draw = json.loads(path.read_text())
        grouped[(draw["scenario"], draw["form"], draw["condition"])].append(draw)
    return grouped


def amplitudes(*, scenario: str) -> tuple[float, float]:
    """Return the PV and heating amplitudes of a scenario."""
    return {**gate.SCENARIOS, **gate.EXTRA_SCENARIOS}[scenario]


def true_pv_path(*, amplitude: float, n: int) -> np.ndarray:
    """Return the injected PV weight at the end of each of the first `n` days."""
    return 1.0 + amplitude * gate.PV_CHANGE * np.clip(np.arange(1, n + 1) / gate.PV_RAMP_DAYS, 0, 1)


def pv_tracking(*, draw: dict, amplitude: float) -> bool:
    """Return whether a draw passes reading 1."""
    weight = np.array(draw["pv_weight"])
    recovery = (weight[PV_READ_INDEX] - 1.0) / (amplitude * gate.PV_CHANGE)
    truth = true_pv_path(amplitude=amplitude, n=gate.PV_READ_DAY)
    correlation = np.corrcoef(weight[: gate.PV_READ_DAY], truth)[0, 1]
    return bool(recovery >= PV_RECOVERY and correlation >= PV_CORRELATION)


def heating_tracking(*, draw: dict, amplitude: float) -> bool:
    """Return whether a draw passes reading 2."""
    weight = np.array(draw["heating_weights"])[HEATING_READ_INDEX].mean()
    recovery = (1.0 - weight) / (amplitude * abs(gate.HEATING_CHANGE))
    return bool(recovery >= HEATING_RECOVERY)


def no_false_movement(*, draw: dict, median_pv_q: float, median_heating_q: float) -> bool:
    """Return whether an S0 draw passes reading 3."""
    after = slice(gate.SETTLE_DAYS, None)
    deviations = [
        np.abs(np.array(draw["pv_weight"])[after] - 1.0).max(),
        np.abs(np.array(draw["heating_weights"])[after] - 1.0).max(),
        np.abs(np.array(draw["holiday_weight"])[after] - 1.0).max(),
    ]
    q = np.array(draw["q"])
    small_q = (
        q[gate.PV_INDEX] < Q_SHARE * median_pv_q
        and q[gate.HEATING].mean() < Q_SHARE * median_heating_q
    )
    return bool(max(deviations) < FALSE_MOVEMENT and small_q)


def no_cross_talk(*, draw: dict, scenario: str) -> bool:
    """Return whether an S1 or S2 draw passes reading 4."""
    after = slice(gate.SETTLE_DAYS, None)
    pv_amplitude, heating_amplitude = amplitudes(scenario=scenario)
    if heating_amplitude == 0.0:
        moved = np.abs(np.array(draw["heating_weights"])[after] - 1.0).mean()
        return bool(moved < CROSS_TALK_SHARE * pv_amplitude * gate.PV_CHANGE)
    moved = np.abs(np.array(draw["pv_weight"])[after] - 1.0).mean()
    return bool(moved < CROSS_TALK_SHARE * heating_amplitude * abs(gate.HEATING_CHANGE))


def median_drift_q(*, grouped: dict, form: str, condition: str) -> tuple[float, float]:
    """Return the median fitted PV q (S1, S3) and mean heating q (S2, S3) of the drift scenarios."""
    pv, heating = [], []
    for scenario in ("S1", "S3"):
        pv += [d["q"][gate.PV_INDEX] for d in grouped.get((scenario, form, condition), [])]
    for scenario in ("S2", "S3"):
        heating += [
            float(np.mean(np.array(d["q"])[gate.HEATING]))
            for d in grouped.get((scenario, form, condition), [])
        ]
    return float(np.median(pv)) if pv else np.nan, float(np.median(heating)) if heating else np.nan


def pass_rates(*, grouped: dict) -> list[dict]:
    """Return one row per reading, scenario, form, and condition: draws, passes, and the rate."""
    rows = []
    for (scenario, form, condition), draws in sorted(grouped.items()):
        pv_amplitude, heating_amplitude = amplitudes(scenario=scenario)
        median_pv, median_heating = median_drift_q(grouped=grouped, form=form, condition=condition)
        checks: dict[str, list[bool]] = {}
        if pv_amplitude > 0:
            checks["1"] = [pv_tracking(draw=d, amplitude=pv_amplitude) for d in draws]
        if heating_amplitude > 0:
            checks["2"] = [heating_tracking(draw=d, amplitude=heating_amplitude) for d in draws]
        if scenario == "S0":
            checks["3"] = [
                no_false_movement(draw=d, median_pv_q=median_pv, median_heating_q=median_heating)
                for d in draws
            ]
        if scenario in ("S1", "S2"):
            checks["4"] = [no_cross_talk(draw=d, scenario=scenario) for d in draws]
        if scenario in ("S1", "S2", "S3"):
            checks["5"] = [d["mae_kez"] < d["mae_gk"] for d in draws]
        for reading, passed in checks.items():
            rate = float(np.mean(passed))
            half_width = 1.96 * float(np.sqrt(rate * (1 - rate) / len(passed)))
            rows.append(
                {
                    "reading": reading,
                    "scenario": scenario,
                    "form": form,
                    "condition": condition,
                    "draws": len(passed),
                    "passes": int(np.sum(passed)),
                    "rate": rate,
                    "half_width": half_width,
                    "passed": rate >= gate.PASS_RATE,
                }
            )
    return rows


def labels(*, rows: list[dict]) -> dict[str, str]:
    """Return the PV and heating weights' labels from readings 1 and 2 and the power check."""
    power = json.loads((common.WORK_DIR / "gate_power.json").read_text())
    out = {}
    for name, reading, key in (("pv", "1", "pv_in_sds"), ("heating", "2", "heating_in_sds")):
        selected = [
            r for r in rows if r["reading"] == reading and r["scenario"] in ("S1", "S2", "S3")
        ]
        failed = any(not r["passed"] for r in selected)
        underpowered = any(site[key] < POWER_SDS for site in power.values())
        out[name] = (
            "not interpretable"
            if failed or underpowered or not selected
            else "interpretable at the real amplitude"
        )
        out[f"{name}_underpowered"] = str(underpowered)
    return out


def main() -> None:
    """Print the readings and write `gate.json`."""
    grouped = load_draws()
    rows = pass_rates(grouped=grouped)
    for row in rows:
        mark = "pass" if row["passed"] else "FAIL"
        print(
            f"reading {row['reading']} {row['scenario']:6s} {row['form']:6s} {row['condition']:5s} "
            f"{row['passes']:3d}/{row['draws']:3d} = {row['rate']:.2f} "
            f"+/- {row['half_width']:.2f} {mark}"
        )
    real = [r for r in rows if r["scenario"] in gate.SCENARIOS]
    failures = sorted(
        {r["reading"] for r in real if not r["passed"] and r["reading"] in BLOCKING_READINGS}
    )
    content = {
        "blocking_failed": len(failures),
        "failed_readings": failures,
        "rows": rows,
        "labels": labels(rows=rows) if (common.WORK_DIR / "gate_power.json").exists() else {},
        "not_run": ["level step", "holiday step", "identity noise matrix"],
    }
    common.write_json_atomic(path=gate.GATE_JSON, content=content)
    print(f"blocking readings failed: {failures or 'none'}")
