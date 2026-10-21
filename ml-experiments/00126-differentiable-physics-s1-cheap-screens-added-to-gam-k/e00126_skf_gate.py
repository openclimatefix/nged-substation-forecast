"""G-skf's pass rows (detection, delay, recovery, false alarms, power) for experiment 126.

**Held-out year.** For a cut `c`, the injected streams are the out-of-sample residuals of T-all's
frozen fit at `c` over [c, c + 12 months), preceded by 14 days of burn-in. EM and the alarm
threshold use the earlier year's residuals (the replica), never the held-out year.

**Events** are injected into copies of the held-out residual stream, one per stream, starting at
least 14 days after `c`, away from the hand-filtered events:

- `transfer_6h`, `transfer_24h`, `transfer_72h`: a step of ±10%, 25%, or 40% of the substation's
  median demand (points of effective capacity), lasting 6, 24, or 72 hours;
- `stuck_24h`: the meter repeats its first reading for 24 hours;
- `scale_30d`, `scale_90d`: the reading is multiplied by 1 ± 10%, 25%, or 40% for 30 or 90 days.

Each cell has `SEEDS` streams per substation. Two placebos run beside the cells: the clean stream,
and the clean stream's cold-snap days.

**Detectors.** The alarm rule is common: the statistic exceeds `k` for 3 consecutive half-hours,
with `k` set so that the detector raises 1 alarm per 100 non-event days on the replica year. The
statistic is the standardised absolute innovation of the plain filter, or the shift-or-outlier
probability of G-skf. Recall counts an alarm between the event's start and 48 hours after its end;
delay is the hours from start to the first alarm (48 if none); recovery is the hours after the
event's end until the absolute innovation is back under 1.5 normal standard deviations for 3
consecutive half-hours (7 days if never).

    uv run --no-sync python ml-experiments/00126-*/e00126_skf_gate.py run --cut 2023-07-01
    uv run --no-sync python ml-experiments/00126-*/e00126_skf_gate.py power
"""

import argparse
import json
from datetime import UTC, datetime
from typing import Any, Final

import e00126_common as common
import e00126_fmix as fmix
import e00126_pipeline as pipeline
import e00126_skf as skf
import e00126_tune as tune
import numpy as np
import polars as pl
from studies.dp_assimilation import FilterHyperparameters
from studies.dp_features import FeatureCache
from studies.dp_pinned_capacity import read_vintages

SEEDS: Final[int] = 100
BURN_IN_DAYS: Final[int] = 14
AMPLITUDES: Final[tuple[float, ...]] = (0.10, 0.25, 0.40)
CELLS: Final[tuple[tuple[str, float], ...]] = (
    *(
        (kind, amplitude)
        for kind in ("transfer_6h", "transfer_24h", "transfer_72h", "scale_30d", "scale_90d")
        for amplitude in AMPLITUDES
    ),
    ("stuck_24h", 0.0),
)
DURATION_HOURS: Final[dict[str, float]] = {
    "transfer_6h": 6.0,
    "transfer_24h": 24.0,
    "transfer_72h": 72.0,
    "stuck_24h": 24.0,
    "scale_30d": 30 * 24.0,
    "scale_90d": 90 * 24.0,
}
ALARM_RUN: Final[int] = 3
ALARMS_PER_100_DAYS: Final[float] = 1.0
DELAY_CENSOR_HOURS: Final[float] = 48.0
RECOVERY_CENSOR_HOURS: Final[float] = 7 * 24.0
RECOVERY_SD: Final[float] = 1.5
HALF_HOURS_PER_DAY: Final[int] = 48
MIN_RECALL_GAIN: Final[float] = 0.10
MAX_PLACEBO_ALARMS_PER_100_DAYS: Final[float] = 2.0
GATE_SEED_TAG: Final[int] = 4
POWER_DRAWS: Final[int] = 100
POWER_LEVEL: Final[float] = 98.75
POWER_MIN_RECOVERY: Final[float] = 0.80
CUT_LABELS: Final[dict[str, str]] = {
    "2023-07-01": "tuning",
    "2024-07-01": "replicate",
    "2025-07-01": "scored",
}


def _grid_statistic(
    *, results: list[skf.SwitchingResult], positions: list[np.ndarray], size: int, kind: str
) -> np.ndarray:
    """Assemble a per-stream statistic on the window's grid rows, `NaN` where there is none."""
    streams = results[0].level.shape[0]
    out = np.full((streams, size), np.nan)
    for result, rows in zip(results, positions, strict=True):
        source = result.p_bad if kind == "p_bad" else np.abs(result.innovation)
        out[:, rows] = source
    return out


def alarm_onsets(*, statistic: np.ndarray, threshold: float) -> np.ndarray:
    """Return True where an alarm starts (3 consecutive half-hours above the threshold)."""
    above = statistic > threshold
    run = above[:, 2:] & above[:, 1:-1] & above[:, :-2]
    onset = np.zeros_like(above)
    onset[:, 2:] = run & ~np.concatenate(
        [np.zeros((run.shape[0], 1), dtype=bool), run[:, :-1]], axis=1
    )
    return onset


def calibrate_threshold(*, statistic: np.ndarray, days: float) -> float:
    """Return the threshold giving `ALARMS_PER_100_DAYS` alarm onsets per 100 days on a stream."""
    finite = statistic[np.isfinite(statistic)]
    low, high = float(np.percentile(finite, 50.0)), float(finite.max())
    target = ALARMS_PER_100_DAYS * days / 100.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        count = float(alarm_onsets(statistic=statistic, threshold=middle).sum())
        if count > target:
            low = middle
        else:
            high = middle
    return high


def inject(
    *,
    residual: np.ndarray,
    y: np.ndarray,
    kind: str,
    amplitude: float,
    onset: int,
    length: int,
    sign: float,
) -> np.ndarray:
    """Return a copy of the residual stream with one event injected."""
    out = residual.copy()
    span = slice(onset, onset + length)
    if kind.startswith("transfer"):
        out[span] += sign * amplitude * np.nanmedian(np.abs(y))
    elif kind.startswith("scale"):
        out[span] += sign * amplitude * y[span]
    elif kind == "stuck_24h":
        first = y[onset] if np.isfinite(y[onset]) else np.nanmedian(y)
        out[span] += first - y[span]
    return out


def _window(*, cache: FeatureCache, cut: datetime) -> slice:
    """Return the grid slice of the burn-in plus the held-out year."""
    start = common.to_64(moment=cut) - np.timedelta64(BURN_IN_DAYS, "D")
    stop = common.to_64(moment=common.year_end(cut=cut))
    return slice(
        int(np.searchsorted(cache.end_times, start, side="right")),
        int(np.searchsorted(cache.end_times, stop, side="right")),
    )


def run_site(  # noqa: PLR0912, PLR0915 (one pass over cells, detectors, and metrics)
    *, cache: FeatureCache, vintages: pl.DataFrame, site: str, cut: datetime
) -> list[dict[str, Any]]:
    """Run every cell and the placebos at one substation and cut and return the rows."""
    decisions = json.loads((common.WORK_DIR / "decisions.json").read_text())
    hyper = FilterHyperparameters.from_array(
        values=np.array(decisions["filter_hyper"]["t_all"][site])
    )
    label = CUT_LABELS[cut.strftime("%Y-%m-%d")]
    weight = tune.t_all_weight()
    runs = {}
    for fit_label in (pipeline.PREVIOUS_LABEL[label], label):
        run = pipeline.fit_year(
            cache=cache,
            vintages=vintages,
            site=site,
            fit_label=fit_label,
            arm_name="t_all",
            penalty_weight=weight,
        )
        assert run is not None
        runs[fit_label] = run
    features = cache.sites[site]
    band = cache.calendar.band
    window = _window(cache=cache, cut=cut)
    # Calibration year: the replica's out-of-sample residuals, event rows masked.
    replica_stream = runs[pipeline.PREVIOUS_LABEL[label]].residual_stream(cache=cache)
    replica_stream[features.switching_mask] = np.nan
    transition = skf.fit_transition(residual=replica_stream[None, :], band=band, hyper=hyper)
    rep_window = slice(
        *(
            int(np.searchsorted(cache.end_times, common.to_64(moment=moment), side="right"))
            for moment in (runs[pipeline.PREVIOUS_LABEL[label]].cut, cut)
        )
    )
    detectors = {
        "plain": (hyper, skf.PRIOR_TRANSITION, (1.0, 1.0), "innovation"),
        "gskf": (hyper, transition, (skf.SHIFT_MULTIPLIER, skf.OUTLIER_MULTIPLIER), "p_bad"),
    }
    held = runs[label].residual_stream(cache=cache)
    y = features.y
    out: list[dict[str, Any]] = []
    sd_normal = float(np.nanstd(replica_stream[rep_window]))
    calibration_days = (rep_window.stop - rep_window.start) / HALF_HOURS_PER_DAY
    thresholds: dict[str, float] = {}
    for name, (h, a, mult, kind) in detectors.items():
        res, pos = skf.run_gpb1(
            residual=replica_stream[None, rep_window],
            band=band[rep_window],
            hyper=h,
            transition=a,
            multipliers=mult,
        )
        stat = _grid_statistic(
            results=res, positions=pos, size=rep_window.stop - rep_window.start, kind=kind
        )
        if kind == "innovation":
            stat = stat / sd_normal
        thresholds[name] = calibrate_threshold(statistic=stat, days=calibration_days)
    rng = np.random.default_rng(
        [common.EXPERIMENT_ID, GATE_SEED_TAG, common.SCORED_SITES.index(site), int(cut.year)]
    )
    hours = 0.5
    size = window.stop - window.start
    regimes = fmix.regime_map(cache=cache, site=site).regime[window]
    padded = features.switching_mask[window]
    valid_onsets = np.flatnonzero(~padded)
    base_stream = held[window]
    base_y = y[window]
    base_band = band[window]
    cells = [("clean", 0.0), *CELLS]
    for kind, amplitude in cells:
        length = 0 if kind == "clean" else int(DURATION_HOURS[kind] / hours)
        streams, spans = [], []
        for _ in range(SEEDS):
            if kind == "clean":
                streams.append(base_stream.copy())
                spans.append((0, 0))
                continue
            for _attempt in range(200):
                onset = int(
                    rng.integers(
                        BURN_IN_DAYS * HALF_HOURS_PER_DAY + 48,
                        size - length - 3 * HALF_HOURS_PER_DAY,
                    )
                )
                if not padded[onset - 48 : onset + length + 48].any():
                    break
            sign = 1.0 if rng.random() < 0.5 else -1.0
            streams.append(
                inject(
                    residual=base_stream,
                    y=base_y,
                    kind=kind,
                    amplitude=amplitude,
                    onset=onset,
                    length=length,
                    sign=sign,
                )
            )
            spans.append((onset, onset + length))
        batch = np.stack(streams)
        for name, (h, a, mult, stat_kind) in detectors.items():
            res, pos = skf.run_gpb1(
                residual=batch, band=base_band, hyper=h, transition=a, multipliers=mult
            )
            stat = _grid_statistic(results=res, positions=pos, size=size, kind=stat_kind)
            innovation = (
                _grid_statistic(results=res, positions=pos, size=size, kind="innovation")
                / sd_normal
            )
            if stat_kind == "innovation":
                stat = innovation
            onsets = alarm_onsets(statistic=stat, threshold=thresholds[name])
            for s, (start, stop) in enumerate(spans):
                row: dict[str, Any] = {
                    "site": site,
                    "cut": label,
                    "cell": kind,
                    "amplitude": amplitude,
                    "detector": name,
                    "seed": s,
                }
                if kind == "clean":
                    non_event = ~padded
                    row["alarms"] = int(onsets[s][non_event].sum())
                    row["days"] = float(non_event.sum() / HALF_HOURS_PER_DAY)
                    cold = (regimes == fmix.COLD_SNAP) & non_event
                    row["cold_alarms"] = int(onsets[s][cold].sum())
                    row["cold_days"] = float(cold.sum() / HALF_HOURS_PER_DAY)
                    christmas = (regimes == fmix.HOLIDAY) & non_event
                    row["holiday_alarms"] = int(onsets[s][christmas].sum())
                    row["holiday_days"] = float(christmas.sum() / HALF_HOURS_PER_DAY)
                else:
                    stop_watch = min(stop + int(DELAY_CENSOR_HOURS / hours), size)
                    hits = np.flatnonzero(onsets[s, start:stop_watch])
                    row["detected"] = bool(len(hits))
                    row["delay_hours"] = (hits[0] * hours) if len(hits) else DELAY_CENSOR_HOURS
                    after = np.abs(innovation[s, stop : stop + int(RECOVERY_CENSOR_HOURS / hours)])
                    calm = np.nan_to_num(after, nan=0.0) < RECOVERY_SD
                    run = (
                        calm[2:] & calm[1:-1] & calm[:-2]
                        if len(calm) > 2
                        else np.array([], dtype=bool)
                    )
                    first = np.flatnonzero(run)
                    row["recovery_hours"] = (
                        (first[0] * hours) if len(first) else RECOVERY_CENSOR_HOURS
                    )
                out.append(row)
        print(f"{site} {label} {kind} {amplitude}", flush=True)
    del valid_onsets
    return out


def run(*, cut_text: str) -> None:
    """Run every decision substation at one cut and store the rows."""
    common.configure_threads()
    cache = common.load_cache()
    vintages = read_vintages()
    cut = datetime.fromisoformat(cut_text).replace(tzinfo=UTC)
    rows = [
        row
        for site in common.DECISION_SITES
        for row in run_site(cache=cache, vintages=vintages, site=site, cut=cut)
    ]
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(
        common.WORK_DIR / f"skf_gate_{cut_text}.parquet"
    )


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--cut", default="2023-07-01", choices=list(CUT_LABELS))
    arguments = parser.parse_args()
    run(cut_text=arguments.cut)


if __name__ == "__main__":
    main()
