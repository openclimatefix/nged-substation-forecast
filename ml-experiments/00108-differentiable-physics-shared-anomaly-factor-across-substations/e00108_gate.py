"""The known-answer gate: inject anomalies of known size into the tuning year and check the factor.

Tuning is frozen (`e00108_tune.py`). Twenty episodes go into the tuning year's out-of-sample
residual streams of the five decision sites, at least 48 hours from every hand-filtered switching
row and at least 10 days apart, with site loadings `lambda_s` drawn from 0.5 to 1.5 (seeded):

- 8 common multi-day episodes: `+a lambda_s` points in every band for 2 or 3 days, `a` in {2, 4};
- 4 common midday episodes: `+3 lambda_s` points in the midday band for 2 days;
- 4 common one-evening episodes: `+3 lambda_s` points over 16:00 to 23:59 local on one day;
- 4 local episodes: `+4` points at one site only, for 3 days.

The tests are in the module docstring of the plan (`result.md`, "Known-answer gate"). A failed test
stops scoring. No replicate-year or scored-year row is read.

    uv run python ml-experiments/00108-*/e00108_gate.py
"""

import json
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import e00108_common as common
import e00108_factor as factor_lib
import e00108_pipeline as pipeline
import e00108_tune as tune_lib
import numpy as np
from studies.dp_assimilation import day_ahead_initialisation
from studies.dp_features import FeatureCache

GATE_PATH: Final[Path] = common.WORK_DIR / "gate.json"
BAND_MIDDAY: Final[int] = 2
BAND_EVENING: Final[int] = 3
MIN_DISTANCE_TO_TRANSFER_DAYS: Final[int] = 2
MIN_GAP_DAYS: Final[int] = 10
WARM_UP_DAYS: Final[int] = 28
LOADING_RANGE: Final[tuple[float, float]] = (0.5, 1.5)
N_ATTEMPTS: Final[int] = 5000
KINDS: Final[tuple[tuple[str, int], ...]] = (
    ("multi", 8),
    ("midday", 4),
    ("evening", 4),
    ("local", 4),
)


@dataclass(frozen=True)
class Episode:
    """One injected anomaly.

    Attributes:
        kind: `multi`, `midday`, `evening`, or `local`.
        first_day: The first local date, as `datetime64[D]`.
        n_days: How many local days it lasts.
        amplitude: `a` points for a common episode, or the size at the one site of a local one.
        bands: The hour bands it is injected in.
        loading: Each site's `lambda_s` (common episodes) or 1 at the site (local episode).
    """

    kind: str
    first_day: np.datetime64
    n_days: int
    amplitude: float
    bands: tuple[int, ...]
    loading: dict[str, float]

    @property
    def last_day(self) -> np.datetime64:
        """The last local date of the episode."""
        return self.first_day + np.timedelta64(self.n_days - 1, "D")


def free_windows(*, cache: FeatureCache) -> tuple[np.ndarray, np.ndarray]:
    """Return the tuning year's local dates and a mask of those at least 48 h from a transfer."""
    window = tune_lib.tuning_window(cache=cache)
    dates = cache.calendar.local_date[window]
    flagged = np.zeros(len(dates), dtype=bool)
    for site in common.DECISION_SITES:
        features = cache.sites[site]
        flagged |= features.switching_mask[window] & features.present[window]
    days = np.unique(dates)
    flagged_days = np.unique(dates[flagged])
    close = np.array(
        [
            np.any(np.abs((flagged_days - day).astype(int)) <= MIN_DISTANCE_TO_TRANSFER_DAYS)
            for day in days
        ]
    )
    return days, ~close


def place_episodes(*, cache: FeatureCache, seed: int) -> list[Episode]:
    """Place the 20 episodes by randomised sequential packing into the free runs of days.

    Args:
        cache: The feature cache.
        seed: The random seed.

    Returns:
        The episodes, ordered by first day.

    Raises:
        ValueError: If no packing is found in `N_ATTEMPTS` attempts.
    """
    rng = np.random.default_rng(seed)
    days, free = free_windows(cache=cache)
    free_set = set(days[free].astype(int).tolist())
    first_ok = int(days[0].astype(int)) + WARM_UP_DAYS
    last_ok = int(days[-1].astype(int)) - 7
    shapes: list[tuple[str, int, float, tuple[int, ...]]] = []
    amplitudes = [2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0]
    rng.shuffle(amplitudes)
    shapes += [
        ("multi", int(rng.integers(2, 4)), amplitude, (0, 1, 2, 3)) for amplitude in amplitudes
    ]
    shapes += [("midday", 2, 3.0, (BAND_MIDDAY,))] * 4
    shapes += [("evening", 1, 3.0, (BAND_EVENING,))] * 4
    shapes += [("local", 3, 4.0, (0, 1, 2, 3))] * 4
    for _ in range(N_ATTEMPTS):
        order = rng.permutation(len(shapes))
        cursor = first_ok
        placed: list[tuple[int, int]] = []
        ok = True
        for position in order:
            n_days = shapes[position][1]
            start = cursor + int(rng.integers(0, 4))
            while start <= last_ok and not all(
                day in free_set for day in range(start, start + n_days)
            ):
                start += 1
            if start + n_days - 1 > last_ok:
                ok = False
                break
            placed.append((position, start))
            cursor = start + n_days - 1 + MIN_GAP_DAYS + 1
        if ok:
            episodes = []
            sites = list(common.DECISION_SITES)
            for position, start in sorted(placed, key=lambda item: item[1]):
                kind, n_days, amplitude, bands = shapes[position]
                if kind == "local":
                    loading = dict.fromkeys(sites, 0.0)
                    loading[str(rng.choice(sites))] = 1.0
                else:
                    loading = {s: float(rng.uniform(*LOADING_RANGE)) for s in sites}
                episodes.append(
                    Episode(
                        kind=kind,
                        first_day=np.datetime64(start, "D"),
                        n_days=n_days,
                        amplitude=amplitude,
                        bands=bands,
                        loading=loading,
                    )
                )
            return episodes
    msg = "no packing of the 20 episodes into the free days was found"
    raise ValueError(msg)


def episode_mask(*, cache: FeatureCache, episode: Episode) -> np.ndarray:
    """Return the grid rows the episode covers (its days, in its bands)."""
    dates = cache.calendar.local_date
    return (
        (dates >= episode.first_day)
        & (dates <= episode.last_day)
        & np.isin(cache.calendar.band, episode.bands)
    )


def injection(*, cache: FeatureCache, episodes: list[Episode]) -> dict[str, np.ndarray]:
    """Return each contributor's injected points on the grid."""
    injected = {site: np.zeros(len(cache.end_times)) for site in common.SCORED_SITES}
    for episode in episodes:
        mask = episode_mask(cache=cache, episode=episode)
        for site, loading in episode.loading.items():
            injected[site][mask] += episode.amplitude * loading
    return injected


@dataclass(frozen=True)
class Outcome:
    """One site's day-ahead corrections in the tuning year, for A2 and KF.

    Attributes:
        a2: A2's total correction at the tuning year's rows.
        kf_local: KF's local part.
        kf_shared: KF's shared part (already times `beta sigma`).
        kf_total: KF's total correction.
    """

    a2: np.ndarray
    kf_local: np.ndarray
    kf_shared: np.ndarray
    kf_total: np.ndarray


def corrections_with(
    *,
    cache: FeatureCache,
    all_fits: dict[str, dict[int, factor_lib.SiteFit]],
    panel: factor_lib.Panel,
    setup: pipeline.FactorSetup,
    a0_hyper: dict,
    injected: dict[str, np.ndarray] | None,
) -> dict[str, Outcome]:
    """Return every decision site's day-ahead corrections, with the injection added to streams."""
    if injected is None:
        use_panel, use_setup = panel, setup
    else:
        residual = {site: panel.residual[site] + injected.get(site, 0.0) for site in panel.residual}
        use_panel = factor_lib.panel_from_residuals(residual=residual, sigma=panel.sigma)
        use_setup = replace(setup, factor=use_panel.factor)
    out: dict[str, Outcome] = {}
    for site in common.DECISION_SITES:
        inputs = tuning_inputs(cache=cache, all_fits=all_fits, site=site)
        init = day_ahead_initialisation(end_times=cache.end_times[inputs.index], days_ahead=1)
        parts = {}
        for spec in (pipeline.A2_SPEC, pipeline.KF_SPEC):
            terms = pipeline.arm_terms(
                cache=cache,
                all_fits=all_fits,
                panel=use_panel,
                setup=use_setup,
                a0_hyper=a0_hyper,
                inputs=inputs,
                spec=spec,
            )
            parts[spec.name] = pipeline.forecast_corrections(
                cache=cache,
                inputs=inputs,
                terms=terms,
                spec=spec,
                hindsight_factor=None,
                initialisation=init,
            )
        out[site] = Outcome(
            a2=parts["a2"].total.total,
            kf_local=parts["kf"].local.total,
            kf_shared=parts["kf"].shared.total,
            kf_total=parts["kf"].total.total,
        )
    return out


def case_mask(
    *, cache: FeatureCache, index: np.ndarray, episode: Episode, first_day: int, last_day: int
) -> np.ndarray:
    """Return a mask over the cases of days `first_day` to `last_day` of the episode, in its bands.

    Day 0 is the episode's first local day.
    """
    dates = cache.calendar.local_date[index]
    start = episode.first_day + np.timedelta64(first_day, "D")
    end = episode.first_day + np.timedelta64(last_day, "D")
    return (dates >= start) & (dates <= end) & np.isin(cache.calendar.band[index], episode.bands)


@dataclass(frozen=True)
class Gate:
    """Everything the tests share.

    Attributes:
        cache: The feature cache.
        all_fits: The stored fits.
        panel: The residual panel.
        setup: The frozen factor setup.
        a0_hyper: Experiment 106's `a0` filter hyperparameters.
        episodes: The injected episodes.
        injected: Each contributor's injected points on the grid.
        inputs: Each decision site's tuning-year inputs.
        base: Day-ahead corrections without injection.
        everything: Day-ahead corrections with every episode injected.
    """

    cache: FeatureCache
    all_fits: dict[str, dict[int, factor_lib.SiteFit]]
    panel: factor_lib.Panel
    setup: pipeline.FactorSetup
    a0_hyper: dict
    episodes: list[Episode]
    injected: dict[str, np.ndarray]
    inputs: dict[str, pipeline.YearInputs]
    base: dict[str, Outcome]
    everything: dict[str, Outcome]

    def run(self, *, episodes: list[Episode]) -> dict[str, Outcome]:
        """Return corrections with only the given episodes injected."""
        return corrections_with(
            cache=self.cache,
            all_fits=self.all_fits,
            panel=self.panel,
            setup=self.setup,
            a0_hyper=self.a0_hyper,
            injected=injection(cache=self.cache, episodes=episodes),
        )


def tuning_inputs(
    *, cache: FeatureCache, all_fits: dict[str, dict[int, factor_lib.SiteFit]], site: str
) -> pipeline.YearInputs:
    """Return a decision site's tuning-year inputs, which every decision site has."""
    inputs = pipeline.year_inputs(cache=cache, all_fits=all_fits, site=site, label="tuning")
    assert inputs is not None
    return inputs


def build_gate() -> Gate:
    """Load the frozen settings, place the episodes, and run the base and full-injection passes."""
    cache = common.load_cache()
    all_fits = factor_lib.load_fits(cache=cache)
    panel = factor_lib.build_panel(cache=cache, all_fits=all_fits)
    setup = tune_lib.real_setup(tuning=tune_lib.load_real_tuning(), panel=panel)
    a0_hyper = pipeline.a0_hyper_from_part4()
    episodes = place_episodes(cache=cache, seed=common.SEED)
    injected = injection(cache=cache, episodes=episodes)
    kwargs = {
        "cache": cache,
        "all_fits": all_fits,
        "panel": panel,
        "setup": setup,
        "a0_hyper": a0_hyper,
    }
    return Gate(
        cache=cache,
        all_fits=all_fits,
        panel=panel,
        setup=setup,
        a0_hyper=a0_hyper,
        episodes=episodes,
        injected=injected,
        inputs={
            site: tuning_inputs(cache=cache, all_fits=all_fits, site=site)
            for site in common.DECISION_SITES
        },
        base=corrections_with(injected=None, **kwargs),
        everything=corrections_with(injected=injected, **kwargs),
    )


def test_recovery(*, gate: Gate) -> dict:
    """Test 1: the injected change in the all-site median regresses on its expected size."""
    sites = common.DECISION_SITES
    residual = {s: gate.panel.residual[s] + gate.injected.get(s, 0.0) for s in gate.panel.residual}
    panel_inj = factor_lib.panel_from_residuals(residual=residual, sigma=gate.panel.sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_base = np.nanmedian(np.stack([gate.panel.scaled[s] for s in sites]), axis=0)
        median_inj = np.nanmedian(np.stack([panel_inj.scaled[s] for s in sites]), axis=0)
        expected = np.nanmedian(
            np.stack([gate.injected[s] / gate.panel.sigma[s] for s in sites]), axis=0
        )
    covered = np.zeros(len(gate.cache.end_times), dtype=bool)
    for episode in gate.episodes:
        if episode.kind != "local":
            covered |= episode_mask(cache=gate.cache, episode=episode)
    use = covered & np.isfinite(median_base) & np.isfinite(expected)
    slope = float(np.polyfit(expected[use], (median_inj - median_base)[use], 1)[0])
    return {"slope": slope, "pass": 0.8 <= slope <= 1.2, "range": [0.8, 1.2]}


def common_episodes(*, gate: Gate) -> list[Episode]:
    """Return the multi-day and midday episodes."""
    return [e for e in gate.episodes if e.kind in {"multi", "midday"}]


def test_carry(*, gate: Gate) -> dict:
    """Test 2: the shared part alone carries 40% to 120% of the injected points from day 2."""
    numerator = denominator = reference = 0.0
    by_day: dict[int, list[float]] = {}
    for episode in common_episodes(gate=gate):
        for site in common.DECISION_SITES:
            index = gate.inputs[site].index
            truth = gate.injected[site][index]
            delta = gate.everything[site].kf_shared - gate.base[site].kf_shared
            others = [
                gate.injected[o][index] / gate.panel.sigma[o][index]
                for o in common.DECISION_SITES
                if o != site
            ]
            perfect = (
                gate.setup.beta[site] * gate.panel.sigma[site][index] * np.median(others, axis=0)
            )
            second_on = case_mask(
                cache=gate.cache, index=index, episode=episode, first_day=1, last_day=episode.n_days
            )
            numerator += float(delta[second_on].sum())
            denominator += float(truth[second_on].sum())
            reference += float(perfect[second_on].sum())
            for day in range(1, episode.n_days):
                mask = case_mask(
                    cache=gate.cache, index=index, episode=episode, first_day=day, last_day=day
                )
                by_day.setdefault(day, []).append(float(delta[mask].sum() / truth[mask].sum()))
    ratio = numerator / denominator
    return {
        "ratio": ratio,
        "pass": 0.4 <= ratio <= 1.2,
        "range": [0.4, 1.2],
        "diagnostic_ratio_to_perfect_tracking_of_the_factor": numerator / reference,
        "diagnostic_ratio_by_episode_day": {d: float(np.mean(v)) for d, v in by_day.items()},
    }


def test_release(*, gate: Gate) -> dict:
    """Test 3: three days after an episode ends, the shared part is under 25% of its peak."""
    rows = []
    for episode in common_episodes(gate=gate):
        daily = []
        for day in range(1, episode.n_days + 2):
            values = []
            for site in common.DECISION_SITES:
                index = gate.inputs[site].index
                delta = gate.everything[site].kf_shared - gate.base[site].kf_shared
                mask = case_mask(
                    cache=gate.cache, index=index, episode=episode, first_day=day, last_day=day
                )
                values.append(float(delta[mask].mean()))
            daily.append(float(np.mean(values)))
        late = []
        for site in common.DECISION_SITES:
            index = gate.inputs[site].index
            delta = gate.everything[site].kf_shared - gate.base[site].kf_shared
            mask = case_mask(
                cache=gate.cache,
                index=index,
                episode=episode,
                first_day=episode.n_days + 2,
                last_day=episode.n_days + 2,
            )
            late.append(float(np.abs(delta[mask]).mean()))
        peak = max(daily)
        rows.append(
            {
                "episode": f"{episode.kind} {episode.first_day}",
                "peak": peak,
                "three_days_after_end": float(np.mean(late)),
                "ratio": float(np.mean(late)) / peak if peak > 0 else float("inf"),
            }
        )
    worst = max(float(row["ratio"]) for row in rows)
    return {"worst_ratio": worst, "pass": worst < 0.25, "episodes": rows}


def test_factor_value(*, gate: Gate) -> dict:
    """Test 4: over days 2 and 3, KF's correction error is below A2's."""
    kf_error: list[float] = []
    a2_error: list[float] = []
    for episode in common_episodes(gate=gate):
        for site in common.DECISION_SITES:
            index = gate.inputs[site].index
            truth = gate.injected[site][index]
            window = case_mask(
                cache=gate.cache, index=index, episode=episode, first_day=1, last_day=2
            )
            delta_kf = gate.everything[site].kf_total - gate.base[site].kf_total
            delta_a2 = gate.everything[site].a2 - gate.base[site].a2
            kf_error += np.abs(truth[window] - delta_kf[window]).tolist()
            a2_error += np.abs(truth[window] - delta_a2[window]).tolist()
    kf_mean, a2_mean = float(np.mean(kf_error)), float(np.mean(a2_error))
    return {
        "kf_mean_abs_error": kf_mean,
        "a2_mean_abs_error": a2_mean,
        "pass": kf_mean < a2_mean,
    }


def test_no_foresight(*, gate: Gate) -> dict:
    """Test 5: one evening alone leaves every earlier-issued forecast bit-identical."""
    identical = True
    checked = 0
    for episode in (e for e in gate.episodes if e.kind == "evening"):
        alone = gate.run(episodes=[episode])
        first = int(np.flatnonzero(episode_mask(cache=gate.cache, episode=episode))[0])
        for site in common.DECISION_SITES:
            index = gate.inputs[site].index
            initialisation = day_ahead_initialisation(
                end_times=gate.cache.end_times[index], days_ahead=1
            )
            cutoff = (
                np.searchsorted(
                    gate.cache.end_times,
                    initialisation - np.timedelta64(round(common.DELAY_HOURS * 3600), "s"),
                    side="right",
                )
                - 1
            )
            before = cutoff < first
            checked += int(before.sum())
            for field in ("a2", "kf_local", "kf_shared", "kf_total"):
                same = np.array_equal(
                    getattr(alone[site], field)[before], getattr(gate.base[site], field)[before]
                )
                identical &= bool(same)
    return {"cases_checked": checked, "bit_identical": identical, "pass": identical}


def test_locality(*, gate: Gate) -> dict:
    """Test 6: a local episode leaves its own site's shared part alone and barely moves others."""
    own: list[float] = []
    others: list[float] = []
    for episode in (e for e in gate.episodes if e.kind == "local"):
        alone = gate.run(episodes=[episode])
        injected_site = max(episode.loading, key=lambda site: episode.loading[site])
        for site in common.DECISION_SITES:
            index = gate.inputs[site].index
            window = case_mask(
                cache=gate.cache,
                index=index,
                episode=episode,
                first_day=0,
                last_day=episode.n_days + 1,
            )
            if site == injected_site:
                own.append(float(np.abs(alone[site].kf_shared - gate.base[site].kf_shared).max()))
            else:
                move = np.abs(alone[site].kf_total - gate.base[site].kf_total)
                others.append(float(move[window].mean()))
    return {
        "own_site_shared_max_change": max(own),
        "others_mean_abs_move_points": float(np.mean(others)),
        "pass": max(own) == 0.0 and float(np.mean(others)) < 0.3,
    }


def test_nesting(*, gate: Gate) -> dict:
    """Test 7a: KF with beta zero and the `a0` filters equals A2."""
    zero = replace(gate.setup, beta=dict.fromkeys(gate.setup.beta, 0.0), local_hyper=gate.a0_hyper)
    worst = 0.0
    for site in common.DECISION_SITES:
        frames = {}
        for name, spec, setup in (
            ("a2", pipeline.A2_SPEC, None),
            ("kf", pipeline.KF_SPEC, zero),
        ):
            frames[name] = pipeline.filter_arm_frames(
                cache=gate.cache,
                all_fits=gate.all_fits,
                panel=gate.panel,
                setup=setup,
                a0_hyper=gate.a0_hyper,
                inputs=gate.inputs[site],
                spec=spec,
            )
        for suffix in ("", "_d3"):
            for column in ("mean", "q01", "q05", "q50", "q95", "q99"):
                gap = np.abs(
                    frames["a2"][suffix][column].to_numpy()
                    - frames["kf"][suffix][column].to_numpy()
                ).max()
                worst = max(worst, float(gap))
    return {"max_abs_gap": worst, "pass": worst <= 1e-6, "tolerance": 1e-6}


def main() -> int:
    """Run the gate and write `work/gate.json`; return 1 if a blocking test fails."""
    gate = build_gate()
    for episode in gate.episodes:
        who = (
            max(episode.loading, key=lambda site: episode.loading[site])
            if episode.kind == "local"
            else " ".join(f"{v:.2f}" for v in episode.loading.values())
        )
        print(
            f"  {episode.kind:>8} {episode.first_day} x{episode.n_days}d a={episode.amplitude}", who
        )
    results: dict[str, dict] = {
        "1_recovery": test_recovery(gate=gate),
        "2_shared_carry": test_carry(gate=gate),
        "3_shared_release": test_release(gate=gate),
        "4_factor_adds_value": test_factor_value(gate=gate),
        "5_no_foresight": test_no_foresight(gate=gate),
        "6_locality": test_locality(gate=gate),
        "7a_nesting": test_nesting(gate=gate),
    }
    episodes = [
        {
            "kind": e.kind,
            "first_day": str(e.first_day),
            "n_days": e.n_days,
            "amplitude": e.amplitude,
            "loading": e.loading,
        }
        for e in gate.episodes
    ]
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    GATE_PATH.write_text(
        json.dumps({**results, "episodes": episodes}, indent=2, default=float) + "\n"
    )
    tests = results
    for name, value in tests.items():
        detail = {k: v for k, v in value.items() if k not in {"episodes", "pass"}}
        print(f"{name}: {'PASS' if value['pass'] else 'FAIL'} {detail}")
    failed = [name for name, value in tests.items() if not value["pass"]]
    print("GATE FAILED: " + ", ".join(failed) if failed else "gate passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
