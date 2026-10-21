"""Score the gate's streams against the plan's four readings and the gate rule, for experiment 128.

For each noise draw the S1 and S1x3 streams are paired with the S0 stream on the identical noise,
and an arm's tracked quantity is its S1 trace minus its own S0 trace. The tracked quantity is the
midday band's level alone (primary); the midday level plus d is the secondary quantity.

1. **Tracking (S1, S1x3).** The paired midday level, averaged over the 7 days up to the read day
   `k*`, recovers at least 70% of the truth `I_{k*}`, and its daily correlation with `I_k` over days
   1 to 200 is at least 0.70.
2. **No cross-talk (S1, S1x3).** The paired night level and the paired d each stay below 25% of the
   peak of `|I_k|` after day 60. The reading also fails an arm whose secondary quantity passes
   reading 1 while its primary quantity does not, because d then carried the change.
3. **No false movement (S0).** The largest change of the midday level after day 60 (maximum minus
   minimum), in units of the peak of `|I_k|` at the decision amplitude, taken as a median over
   draws, is at most 1.25 times GAM-K's.
4. **Forecast error follows (S1, S1x3).** The arm's mean absolute day-ahead error on the stream
   after day 60 is below GAM-K's.

**The gate passes** when KDS meets all of these in the real form at the decision amplitude (S1, or
S1x3 when the power check finds S1 underpowered at 2 or more of the 5 substations): the real-form
pass rate of readings 1, 2, and 4 is at least 0.8 times KDS's own pass rate in the AR(1) form; KDS's
tracking pass rate exceeds GAM-K's by at least 0.10 with a paired 95% interval (resampling the
draws) that excludes zero; and reading 3 holds. The fixed pass rate of 0.80 is reported beside every
rate as a labelled exploration. **The stop rule** fires when KDS tracks worse than GAM-K by more
than 0.10 in the AR(1) form.
"""

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Final

import e00128_common as common
import e00128_gate as gate
import numpy as np

RECOVERY: Final[float] = 0.70
CORRELATION: Final[float] = 0.70
CROSS_TALK_SHARE: Final[float] = 0.25
MOVEMENT_RATIO: Final[float] = 1.25
RELATIVE_BAR: Final[float] = 0.80
TRACKING_MARGIN: Final[float] = 0.10
POWER_SDS: Final[float] = 3.0
N_UNDERPOWERED: Final[int] = 2
FIXED_PASS_RATE: Final[float] = 0.80
N_RESAMPLES: Final[int] = 5000
BOOTSTRAP_SEED: Final[int] = 128
READINGS: Final[tuple[str, ...]] = ("1", "2", "4")
Key = tuple[str, str, str, int]


def load_records() -> dict[Key, dict]:
    """Return every stored stream, keyed by (form, scenario, site, draw)."""
    out: dict[Key, dict] = {}
    for path in sorted(Path(gate.GATE_DIR).glob("*/*/*.json")):
        record = json.loads(path.read_text())
        out[(record["form"], record["scenario"], record["site"], record["draw"])] = record
    return out


@lru_cache(maxsize=8)
def site_truth(site: str) -> tuple[np.ndarray, int]:
    """Return a site's truth path at amplitude 1 and its read day."""
    truth = gate.truth_path(inputs=gate.gate_inputs(site))
    return truth, gate.read_day(truth=truth)


def decision_scenario() -> tuple[str, bool]:
    """Return the decision scenario (`S1` or `S1x3`) and whether S1 is underpowered."""
    power = json.loads(gate.POWER_JSON.read_text())
    weak = sum(entry["primary_in_sds"] < POWER_SDS for entry in power.values())
    underpowered = weak >= N_UNDERPOWERED
    return ("S1x3" if underpowered else "S1"), underpowered


def series(*, record: dict, arm: str, name: str) -> np.ndarray:
    """Return an arm's stored end-of-day series as an array (`None` becomes NaN)."""
    return np.array(record["arms"][arm][name], dtype=np.float64)


def paired(*, treated: dict, base: dict, arm: str, name: str) -> np.ndarray:
    """Return an arm's series on a scenario stream minus the same series on the paired S0 stream."""
    return series(record=treated, arm=arm, name=name) - series(record=base, arm=arm, name=name)


def tracks(*, tracked: np.ndarray, truth: np.ndarray, k_star: int) -> bool:
    """Return whether a paired series passes reading 1 against a truth path."""
    recent = tracked[k_star - gate.READ_AVERAGE_DAYS + 1 : k_star + 1]
    if not np.isfinite(recent).any():
        return False
    recovery = np.nanmean(recent) / truth[k_star]
    both = np.isfinite(tracked[: gate.READ_WINDOW[1]]) & np.isfinite(truth[: gate.READ_WINDOW[1]])
    if both.sum() < 3:
        return False
    correlation = np.corrcoef(
        tracked[: gate.READ_WINDOW[1]][both], truth[: gate.READ_WINDOW[1]][both]
    )[0, 1]
    return bool(recovery >= RECOVERY and correlation >= CORRELATION)


def readings_of_stream(*, treated: dict, base: dict, amplitude: float, arm: str) -> dict:
    """Return an arm's pass/fail on readings 1, 2, and 4 for one paired stream."""
    truth_1, k_star = site_truth(treated["site"])
    truth = amplitude * truth_1
    mid = paired(treated=treated, base=base, arm=arm, name="mid")
    day = paired(treated=treated, base=base, arm=arm, name="d")
    night = paired(treated=treated, base=base, arm=arm, name="night")
    primary = tracks(tracked=mid, truth=truth, k_star=k_star)
    secondary = tracks(tracked=mid + day, truth=truth, k_star=k_star)
    limit = CROSS_TALK_SHARE * float(np.nanmax(np.abs(truth)))
    after = slice(gate.SETTLE_DAYS, None)
    quiet = bool(
        np.nanmax(np.abs(night[after])) < limit
        and (np.all(np.isnan(day[after])) or np.nanmax(np.abs(day[after])) < limit)
    )
    gk_mae = treated["arms"]["GK"]["mae"]
    return {
        "1": primary,
        "2": quiet and not (secondary and not primary),
        "4": treated["arms"][arm]["mae"] < gk_mae,
    }


def rate_row(*, passed: list[bool], **labels: str) -> dict:
    """Return a pass-rate row: draws, passes, rate, and a 95% binomial half-width."""
    rate = float(np.mean(passed))
    return {
        **labels,
        "draws": len(passed),
        "passes": int(np.sum(passed)),
        "rate": rate,
        "half_width": 1.96 * float(np.sqrt(rate * (1.0 - rate) / len(passed))),
        "passes_fixed_bar": rate >= FIXED_PASS_RATE,
    }


def pass_table(*, records: dict[Key, dict]) -> tuple[list[dict], dict[tuple, list[bool]]]:
    """Return one row per reading, arm, form, and scenario, and the per-draw outcomes.

    The per-draw outcomes are keyed by (reading, arm, form, scenario) and ordered by site and draw,
    so two arms' lists pair draw by draw.
    """
    outcomes: dict[tuple, list[bool]] = defaultdict(list)
    for (form, scenario, site, draw), record in sorted(records.items()):
        if scenario == "S0":
            continue
        base = records.get((form, "S0", site, draw))
        if base is None:
            continue
        amplitude = gate.SCENARIOS[scenario]
        for arm in gate.GATE_ARMS:
            result = readings_of_stream(treated=record, base=base, amplitude=amplitude, arm=arm)
            for reading in READINGS:
                if reading == "4" and arm == "GK":
                    continue
                outcomes[(reading, arm, form, scenario)].append(bool(result[reading]))
    rows = [
        rate_row(passed=passed, reading=reading, arm=arm, form=form, scenario=scenario)
        for (reading, arm, form, scenario), passed in sorted(outcomes.items())
    ]
    return rows, outcomes


def false_movement(*, records: dict[Key, dict], scenario: str) -> dict:
    """Return, per arm and form, the median largest change of the midday level on S0 streams.

    The change is in units of the peak of `|I_k|` at the decision amplitude. The ratio to GAM-K's
    median is included.
    """
    amplitude = gate.SCENARIOS[scenario]
    out: dict = {}
    for form in gate.FORMS:
        out[form] = {}
        for arm in gate.GATE_ARMS:
            values = []
            for (f, s, site, _), record in records.items():
                if f != form or s != "S0":
                    continue
                peak = amplitude * float(np.nanmax(np.abs(site_truth(site)[0])))
                mid = series(record=record, arm=arm, name="mid")[gate.SETTLE_DAYS :]
                values.append(float(np.nanmax(mid) - np.nanmin(mid)) / peak)
            out[form][arm] = float(np.median(values)) if values else None
        gk = out[form]["GK"]
        for arm in gate.GATE_ARMS:
            value = out[form][arm]
            out[form][arm] = {
                "median": value,
                "ratio_to_gk": None if gk in (None, 0.0) or value is None else value / gk,
            }
    return out


def paired_interval(*, difference: np.ndarray) -> tuple[float, float]:
    """Return the 95% interval of the mean of per-draw differences, resampling draws."""
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    picks = generator.integers(len(difference), size=(N_RESAMPLES, len(difference)))
    means = difference[picks].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def tracking_difference(*, outcomes: dict[tuple, list[bool]], form: str, scenario: str) -> dict:
    """Return KDS minus GAM-K in the tracking pass rate, with a paired interval over draws."""
    kds = np.array(outcomes[("1", "KDS", form, scenario)], dtype=np.float64)
    gk = np.array(outcomes[("1", "GK", form, scenario)], dtype=np.float64)
    low, high = paired_interval(difference=kds - gk)
    return {"difference": float(np.mean(kds - gk)), "lower": low, "upper": high}


def gate_rule(
    *, rows: list[dict], outcomes: dict[tuple, list[bool]], movement: dict, scenario: str
) -> dict:
    """Return the gate rule's three conditions, the stop rule, and whether the gate passes."""
    rate = {(r["reading"], r["arm"], r["form"], r["scenario"]): r["rate"] for r in rows}
    relative = {}
    for reading in READINGS:
        real = rate[(reading, "KDS", "real", scenario)]
        reference = rate[(reading, "KDS", "ar1", scenario)]
        relative[reading] = {
            "real": real,
            "ar1": reference,
            "bar": RELATIVE_BAR * reference,
            "passed": real >= RELATIVE_BAR * reference,
        }
    real_tracking = tracking_difference(outcomes=outcomes, form="real", scenario=scenario)
    ar1_tracking = tracking_difference(outcomes=outcomes, form="ar1", scenario=scenario)
    tracking_ok = bool(
        real_tracking["difference"] >= TRACKING_MARGIN and real_tracking["lower"] > 0.0
    )
    ratio = movement["real"]["KDS"]["ratio_to_gk"]
    movement_ok = bool(ratio is not None and ratio <= MOVEMENT_RATIO)
    return {
        "scenario": scenario,
        "relative_to_ar1": relative,
        "tracking_real": real_tracking,
        "tracking_ar1": ar1_tracking,
        "tracking_ok": tracking_ok,
        "movement_ratio_real": ratio,
        "movement_ok": movement_ok,
        "passed": bool(all(r["passed"] for r in relative.values()) and tracking_ok and movement_ok),
        "stop": bool(ar1_tracking["difference"] < -TRACKING_MARGIN),
    }


def main() -> None:
    """Print the readings and write `gate.json`."""
    records = load_records()
    scenario, underpowered = decision_scenario()
    rows, outcomes = pass_table(records=records)
    for row in rows:
        mark = "ok" if row["passes_fixed_bar"] else "below 0.80"
        print(
            f"reading {row['reading']} {row['arm']:4s} {row['form']:4s} {row['scenario']:5s} "
            f"{row['passes']:3d}/{row['draws']:3d} = {row['rate']:.2f} "
            f"+/- {row['half_width']:.2f} ({mark})"
        )
    movement = false_movement(records=records, scenario=scenario)
    rule = gate_rule(rows=rows, outcomes=outcomes, movement=movement, scenario=scenario)
    content = {
        "decision_scenario": scenario,
        "s1_underpowered": underpowered,
        "rows": rows,
        "false_movement": movement,
        "rule": rule,
        "passed": rule["passed"],
        "stop": rule["stop"],
        "power": json.loads(gate.POWER_JSON.read_text()),
    }
    common.write_json_atomic(path=gate.GATE_JSON, content=content)
    print(json.dumps(rule, indent=2))
    print(f"gate passed: {rule['passed']}; stop rule: {rule['stop']}")
