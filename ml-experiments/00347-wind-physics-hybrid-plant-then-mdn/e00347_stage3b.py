"""Stage 3b of experiment 00347: fit the plant twice, masking the periods the first plant flags.

Pass 1 is the Stage 3 plant (all training half-hours, experiment 210's idle-band rows weight 0).
The pre-registered rule of `e00347_flags` (3 robust standard deviations; 4 and 2 as sensitivity)
flags short periods on which the farm made much less than pass 1 expects. Pass 2 refits the plant
with the flagged periods masked. Controls: a placebo that masks the same number of day blocks at
random within each month, and an injection test that scales the observed power down on random
blocks and asks whether the rule finds them. A third pass flags with the pass-2 plant and refits.

    export PYTHONPATH=ml-experiments/00230-wind-physics-core-part-1:\
        ml-experiments/00347-wind-physics-hybrid-plant-then-mdn
    uv run --no-sync python \
        ml-experiments/00347-wind-physics-hybrid-plant-then-mdn/e00347_stage3b.py
"""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from typing import Final  # noqa: E402

import e00347_common as common  # noqa: E402
import e00347_eval as evaluation  # noqa: E402
import e00347_flags as flags  # noqa: E402
import e00347_plant as plant  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

RULES: Final[tuple[float, ...]] = (4.0, 3.0, 2.0)
MAIN_RULE: Final[float] = 3.0
PLACEBO_DRAWS: Final[int] = 5
INJECTION_TRIALS: Final[int] = 20
INJECTION_BLOCKS: Final[int] = 4
INJECTION_LENGTHS: Final[tuple[int, int]] = (3, 14)
INJECTION_GAP_DAYS: Final[int] = 5
INJECTED_SHARES: Final[dict[str, dict[str, float]]] = {
    "W1": {"one turbine (1/13)": 1 / 13, "25%": 0.25},
    "W2": {"one turbine (1/8)": 1 / 8, "25%": 0.25},
    "W3": {"one turbine (1/2)": 1 / 2, "25%": 0.25},
}
WORKERS: Final[int] = 9
_STATE: dict = {}


def _init() -> None:
    """Load the farms once per worker process."""
    _STATE["farms"] = common.load_farms()
    _STATE["priors"] = common.prior_scales()


def fit_with_mask(
    *, farm: common.FarmData, prior_scale: float, masked: np.ndarray, start: plant.PlantParameters
) -> plant.PlantParameters:
    """Refit the Stage 3 plant on the training rows, giving weight 0 to the `masked` rows too."""
    train = farm.train
    weight = (~(farm.idle_mask | masked))[: train.time.size].astype(np.float64)
    return plant.fit_plant(
        weather=train.weather,
        target=train.target,
        weight=weight,
        prior_scale=prior_scale,
        with_direction=True,
        start=start,
    ).parameters


def refit_task(args: tuple[str, str, np.ndarray, dict]) -> tuple[str, str, dict]:
    """Worker: refit one farm with one mask, starting from the given parameters."""
    key, label, masked_index, start = args
    farm = _STATE["farms"][label]
    masked = np.zeros(farm.time.size, dtype=bool)
    masked[masked_index] = True
    parameters = fit_with_mask(
        farm=farm,
        prior_scale=_STATE["priors"][label],
        masked=masked,
        start=plant.PlantParameters.from_dict(start),
    )
    return key, label, parameters.as_dict()


def direction_factor_summary(parameters: plant.PlantParameters) -> tuple[float, float]:
    """Return the deepest loss of wind speed (a share) and its direction in degrees."""
    grid = np.radians(np.arange(360))
    loss = np.zeros(360)
    for depth, centre, kappa in zip(
        parameters.depths, parameters.centres_deg, parameters.kappas, strict=True
    ):
        loss += depth * np.exp(kappa * (np.cos(grid - np.radians(centre)) - 1.0))
    return float(loss.max()), float(np.argmax(loss))


def shift_line(
    *, label: str, name: str, before: plant.PlantParameters, after: plant.PlantParameters
) -> str:
    """Return one report line on how the parameters moved from `before` to `after`."""
    depth_a, centre_a = direction_factor_summary(before)
    depth_b, centre_b = direction_factor_summary(after)
    return (
        f"{label} {name}: scale {before.scale:.3f} -> {after.scale:.3f} "
        f"({after.scale - before.scale:+.3f}); "
        f"rated {before.rated:.2f} -> {after.rated:.2f} ({after.rated - before.rated:+.2f} m/s); "
        f"cut_in {before.cut_in:.2f} -> {after.cut_in:.2f}; hub_height "
        f"{before.hub_height:.1f} -> {after.hub_height:.1f}; "
        f"blur {before.blur:.2f} -> {after.blur:.2f}; "
        f"deepest direction loss of wind speed {depth_a:.3f} at {centre_a:.0f} deg -> "
        f"{depth_b:.3f} at {centre_b:.0f} deg"
    )


def placebo_days(
    *,
    farm: common.FarmData,
    blocks: list[tuple[int, int]],
    scored: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Return random day blocks of the same lengths in the same months, and the number unplaced."""
    month_of_day = {int(d): str(np.datetime64(int(d), "D"))[:7] for d in scored}
    days_by_month: dict[str, list[int]] = {}
    for d, m in month_of_day.items():
        days_by_month.setdefault(m, []).append(d)
    scored_set = {int(d) for d in scored}
    used: set[int] = set()
    chosen: list[int] = []
    failures = 0
    for first, last in blocks:
        length = last - first + 1
        candidates = days_by_month.get(str(np.datetime64(first, "D"))[:7], [])
        placed = False
        for _ in range(100):
            if not candidates:
                break
            start = int(candidates[rng.integers(0, len(candidates))])
            span = set(range(start, start + length))
            if span <= scored_set and not (span & used):
                used |= span
                chosen.extend(sorted(span))
                placed = True
                break
        failures += 0 if placed else 1
    return np.array(sorted(chosen), dtype=np.int64), failures


def injection_trial(
    *,
    farm: common.FarmData,
    evidence: flags.Evidence,
    scored_days: np.ndarray,
    share: float,
    k: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Scale the observed power down on random day blocks and report what the rule finds."""
    available = {int(d) for d in scored_days}
    taken: set[int] = set()
    blocks: list[tuple[int, int]] = []
    for _ in range(INJECTION_BLOCKS):
        for _attempt in range(200):
            length = int(rng.integers(INJECTION_LENGTHS[0], INJECTION_LENGTHS[1] + 1))
            start = int(scored_days[rng.integers(0, scored_days.size)])
            span = range(start - INJECTION_GAP_DAYS, start + length + INJECTION_GAP_DAYS)
            if all(d in available for d in range(start, start + length)) and not (
                set(span) & taken
            ):
                taken |= set(span)
                blocks.append((start, start + length - 1))
                break
    injected_days = np.concatenate([np.arange(a, b + 1) for a, b in blocks])
    in_block = np.isin(farm.day, injected_days)
    target = np.where(in_block, farm.target * (1.0 - share), farm.target)
    floor = flags.FLOOR_SHARE * evidence.scale
    log_ratio = np.where(
        evidence.weight > 0,
        np.log(np.maximum(target, floor) / np.maximum(evidence.expected, 1e-6)),
        np.nan,
    )
    series = flags.daily_series(
        farm=farm, log_ratio=log_ratio, weight=evidence.weight, rows=farm.is_train
    )
    reference = flags.reference_of(series)
    flagged = {int(d) for d in flags.flagged_days(series=series, reference=reference, k=k)}
    scored = {int(d) for d in series.day}
    detected, usable, any_day = 0, 0, 0
    for first, last in blocks:
        inside = [d for d in range(first, last + 1) if d in scored]
        if not inside:
            continue
        usable += 1
        hits = sum(d in flagged for d in inside)
        detected += hits * 2 >= len(inside)
        any_day += hits >= 1
    outside = [d for d in scored if d not in set(injected_days.tolist())]
    false_share = float(np.mean([d in flagged for d in outside]))
    return {"detected": detected, "any_day": any_day, "usable": usable, "false_share": false_share}


def main() -> None:  # noqa: PLR0912, PLR0915
    """Run Stage 3b and write the plants, the private flagged-blocks table, and the report."""
    farms = common.load_farms()
    priors = common.prior_scales()
    pass1 = {
        label: plant.PlantParameters.from_dict(v)
        for label, v in common.read_json(name="plant_stage3.json").items()
    }
    report = ["STAGE 3b: fit the plant twice (pass 1 = Stage 3; flag; mask; pass 2)"]
    evidence1, series1, reference1, flagged_by_rule, masks = {}, {}, {}, {}, {}
    idle1 = {}
    block_rows = []
    for label, farm in farms.items():
        evidence1[label] = flags.build_evidence(
            farm=farm, parameters=pass1[label], prior_scale=priors[label]
        )
        series1[label] = flags.daily_series(
            farm=farm,
            log_ratio=evidence1[label].log_ratio,
            weight=evidence1[label].weight,
            rows=farm.is_train,
        )
        reference1[label] = flags.reference_of(series1[label])
        idle1[label] = flags.idle_run_rows(farm=farm, rated=pass1[label].rated) & farm.is_train
        e = evidence1[label]
        report.append(
            f"{label}: evidence model: relative ERA5 speed error sigma_v {e.sigma_speed:.3f}, "
            f"power noise sigma_p {e.sigma_power:.3f} (log ratio); "
            f"{series1[label].day.size} scored training days; median daily log ratio "
            f"{reference1[label].median:+.4f}, robust sd {reference1[label].sd:.4f}"
        )
        n_train = farm.is_train.sum()
        for k in RULES:
            days = flags.flagged_days(series=series1[label], reference=reference1[label], k=k)
            blocks = flags.blocks_of(days=days)
            masked = flags.rows_in_days(farm=farm, days=days) & farm.is_train | idle1[label]
            flagged_by_rule[(label, k)] = days
            masks[(label, k)] = masked
            longest = max((b - a + 1 for a, b in blocks), default=0)
            overlap = float((masked & farm.idle_mask).sum() / max(masked.sum(), 1))
            report.append(
                f"{label} rule {k:.0f} sigma: flagged {days.size} of "
                f"{series1[label].day.size} scored days "
                f"({days.size / series1[label].day.size * 100:.1f}%) in {len(blocks)} blocks, "
                f"longest block {longest} days; "
                f"idle-floor runs above rated speed: {int(idle1[label].sum())} half-hours; "
                f"masked {int(masked.sum())} of {n_train} training half-hours "
                f"({masked.sum() / n_train * 100:.1f}%), "
                f"of which {overlap * 100:.0f}% already carry weight 0 through experiment "
                "210's mask"
            )
            for first, last in blocks:
                inside = series1[label].ratio[
                    np.isin(series1[label].day, np.arange(first, last + 1))
                ]
                block_rows.append(
                    {
                        "farm": label,
                        "rule": f"pass1_{k:.0f}sigma",
                        "reason": "daily_log_ratio",
                        "start_date": np.datetime64(first, "D").astype(object),
                        "end_date": np.datetime64(last, "D").astype(object),
                        "days": last - first + 1,
                        "mean_daily_log_ratio": float(inside.mean()),
                        "share_of_scale_lost": float(1 - np.exp(inside.mean())),
                    }
                )
        runs = np.flatnonzero(
            np.diff(np.concatenate([[0], idle1[label].astype(int), [0]])) != 0
        ).reshape(-1, 2)
        for a, b in runs:
            lost = 1 - farm.target[a:b].sum() / max(evidence1[label].expected[a:b].sum(), 1e-9)
            block_rows.append(
                {
                    "farm": label,
                    "rule": "pass1_idle_run",
                    "reason": "idle_floor_above_rated_speed",
                    "start_date": farm.time[a].astype("datetime64[D]").astype(object),
                    "end_date": farm.time[b - 1].astype("datetime64[D]").astype(object),
                    "days": int((b - a) / 48) + 1,
                    "mean_daily_log_ratio": float("nan"),
                    "share_of_scale_lost": float(lost),
                }
            )
        report.append(f"{label}: idle-floor runs above rated speed: {len(runs)} runs")
    pl.DataFrame(
        block_rows, schema_overrides={"start_date": pl.Date, "end_date": pl.Date}
    ).write_parquet(common.DATA_DIR / "flagged_blocks_private.parquet")
    # Placebo masks (3 sigma day blocks only).
    rng = np.random.default_rng(347)
    tasks = []
    placebo_info: dict[tuple[str, int], int] = {}
    for label, farm in farms.items():
        for k in RULES:
            tasks.append(  # noqa: PERF401
                (f"pass2_{k:.0f}", label, np.flatnonzero(masks[(label, k)]), pass1[label].as_dict())
            )
        blocks = flags.blocks_of(days=flagged_by_rule[(label, MAIN_RULE)])
        for draw in range(PLACEBO_DRAWS):
            days, _failed = placebo_days(
                farm=farm, blocks=blocks, scored=series1[label].day, rng=rng
            )
            placebo_rows = flags.rows_in_days(farm=farm, days=days) & farm.is_train
            placebo_info[(label, draw)] = int(placebo_rows.sum())
            tasks.append(
                (f"placebo_{draw}", label, np.flatnonzero(placebo_rows), pass1[label].as_dict())
            )
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=get_context("spawn"), initializer=_init
    ) as pool:
        results = list(pool.map(refit_task, tasks))
        fits = {(key, label): plant.PlantParameters.from_dict(p) for key, label, p in results}
        # Third pass: flag with the pass-2 plant.
        evidence2, flagged2, masks3 = {}, {}, {}
        tasks3 = []
        for label, farm in farms.items():
            p2 = fits[(f"pass2_{MAIN_RULE:.0f}", label)]
            evidence2[label] = flags.build_evidence(
                farm=farm, parameters=p2, prior_scale=priors[label]
            )
            series2 = flags.daily_series(
                farm=farm,
                log_ratio=evidence2[label].log_ratio,
                weight=evidence2[label].weight,
                rows=farm.is_train,
            )
            days2 = flags.flagged_days(
                series=series2, reference=flags.reference_of(series2), k=MAIN_RULE
            )
            flagged2[label] = days2
            masks3[label] = flags.rows_in_days(farm=farm, days=days2) & farm.is_train | (
                flags.idle_run_rows(farm=farm, rated=p2.rated) & farm.is_train
            )
            tasks3.append(("pass3", label, np.flatnonzero(masks3[label]), p2.as_dict()))
        for key, label, p in pool.map(refit_task, tasks3):
            fits[(key, label)] = plant.PlantParameters.from_dict(p)
    for label in common.FARMS:
        fits[("pass1", label)] = pass1[label]
    for key, name in (
        ("pass2_3", "plant_stage3b_pass2.json"),
        ("pass2_4", "plant_stage3b_pass2_4sigma.json"),
        ("pass2_2", "plant_stage3b_pass2_2sigma.json"),
        ("pass3", "plant_stage3b_pass3.json"),
    ):
        common.write_json(
            name=name,
            payload={
                label: {**fits[(key, label)].as_dict(), "prior_scale": priors[label]}
                for label in common.FARMS
            },
        )
    # Parameter shifts.
    report.append("-- pass-1 to pass-2 parameter shifts (3 sigma rule) --")
    for label in common.FARMS:
        report.append(
            shift_line(
                label=label,
                name="pass 1 -> pass 2",
                before=pass1[label],
                after=fits[("pass2_3", label)],
            )
        )
        for k in (4, 2):
            report.append(  # noqa: PERF401
                shift_line(
                    label=label,
                    name=f"pass 1 -> pass 2 ({k} sigma)",
                    before=pass1[label],
                    after=fits[(f"pass2_{k}", label)],
                )
            )
        report.append(
            shift_line(
                label=label,
                name="pass 2 -> pass 3",
                before=fits[("pass2_3", label)],
                after=fits[("pass3", label)],
            )
        )
    # Hold-out scores.
    holdout_flag_days, holdout_unflagged = {}, {}
    for label, farm in farms.items():
        series_all = flags.daily_series(
            farm=farm,
            log_ratio=evidence1[label].log_ratio,
            weight=evidence1[label].weight,
            rows=~farm.is_train,
        )
        days = flags.flagged_days(series=series_all, reference=reference1[label], k=MAIN_RULE)
        idle_hold = flags.idle_run_rows(farm=farm, rated=pass1[label].rated) & ~farm.is_train
        holdout_flag_days[label] = days
        holdout_unflagged[label] = ~(flags.rows_in_days(farm=farm, days=days) | idle_hold)[
            ~farm.is_train
        ]
        report.append(
            f"{label}: hold-out (2024-04 to 2025-06) days flagged by the pass-1 plant at 3 "
            "sigma (reference from training): "
            f"{days.size} of {series_all.day.size} scored days "
            f"({days.size / max(series_all.day.size, 1) * 100:.1f}%); "
            f"unflagged hold-out half-hours {holdout_unflagged[label].mean() * 100:.1f}% of all"
        )
    report.append(
        "-- ERA5 hold-out quantile score in points (smaller is better); offsets always "
        "fitted on all training rows; evaluation rows are never masked --"
    )
    keys = [
        "pass1",
        "pass2_4",
        "pass2_3",
        "pass2_2",
        "pass3",
        *[f"placebo_{d}" for d in range(PLACEBO_DRAWS)],
    ]
    evals = {
        (key, label): evaluation.evaluate_plant(
            farm=farms[label], parameters=fits[(key, label)], prior_scale=priors[label]
        )
        for key in keys
        for label in common.FARMS
    }
    for label in common.FARMS:
        kept = evals[("pass1", label)].kept
        for key in keys[:5]:
            ev = evals[(key, label)]
            unflagged = kept & holdout_unflagged[label]
            report.append(
                f"{label} {key}: kept rows {ev.mean_score(rows=kept):.3f}, all rows "
                f"{ev.mean_score():.3f}, unflagged kept rows "
                f"{ev.mean_score(rows=unflagged):.3f}; "
                f"R2 kept {ev.r_squared(rows=kept):.3f}; median-forecast bias on kept rows "
                f"{float((ev.median[kept] - ev.target[kept]).mean() * 100):+.2f} points"
            )
    pooled = {
        key: {
            scope: np.concatenate(
                [evals[(key, label)].score[rows(label)] for label in common.FARMS]
            )
            for scope, rows in (
                ("kept", lambda lb: evals[("pass1", lb)].kept),
                ("all", lambda lb: np.ones_like(evals[("pass1", lb)].kept)),
                ("unflagged kept", lambda lb: evals[("pass1", lb)].kept & holdout_unflagged[lb]),
            )
        }
        for key in keys
    }
    months = {
        scope: np.concatenate(
            [evals[("pass1", label)].month[rows(label)] for label in common.FARMS]
        )
        for scope, rows in (
            ("kept", lambda lb: evals[("pass1", lb)].kept),
            ("all", lambda lb: np.ones_like(evals[("pass1", lb)].kept)),
            ("unflagged kept", lambda lb: evals[("pass1", lb)].kept & holdout_unflagged[lb]),
        )
    }
    for key in keys[1:5]:
        for scope in ("kept", "all", "unflagged kept"):
            gap = common.month_block_gap(
                losses_a=pooled[key][scope], losses_b=pooled["pass1"][scope], months=months[scope]
            )
            report.append(
                f"3 farms pooled, {scope} rows: {key} minus pass1 {gap[0]:+.3f} (95% "
                f"{gap[1]:+.3f} to {gap[2]:+.3f}); pass1 {pooled['pass1'][scope].mean():.3f}, "
                f"{key} {pooled[key][scope].mean():.3f}"
            )
    # Placebo comparison.
    report.append(
        "-- placebo: the same number of day blocks masked at random within each month (5 "
        "draws per farm) --"
    )
    for label in common.FARMS:
        real_shift = fits[("pass2_3", label)].scale - pass1[label].scale
        rated_shift = fits[("pass2_3", label)].rated - pass1[label].rated
        placebo_scale = np.array(
            [fits[(f"placebo_{d}", label)].scale - pass1[label].scale for d in range(PLACEBO_DRAWS)]
        )
        placebo_rated = np.array(
            [fits[(f"placebo_{d}", label)].rated - pass1[label].rated for d in range(PLACEBO_DRAWS)]
        )
        kept = evals[("pass1", label)].kept
        real_gap = evals[("pass2_3", label)].mean_score(rows=kept) - evals[
            ("pass1", label)
        ].mean_score(rows=kept)
        placebo_gap = np.array(
            [
                evals[(f"placebo_{d}", label)].mean_score(rows=kept)
                - evals[("pass1", label)].mean_score(rows=kept)
                for d in range(PLACEBO_DRAWS)
            ]
        )
        masked_real = int(masks[(label, MAIN_RULE)].sum())
        report.append(
            f"{label}: scale shift real mask {real_shift:+.4f}, placebo mean "
            f"{placebo_scale.mean():+.4f} (sd {placebo_scale.std(ddof=1):.4f}, range "
            f"{placebo_scale.min():+.4f} to {placebo_scale.max():+.4f}); "
            f"rated-speed shift real {rated_shift:+.3f}, placebo mean "
            f"{placebo_rated.mean():+.3f} (sd {placebo_rated.std(ddof=1):.3f}); "
            f"kept-row hold-out score change real {real_gap:+.3f}, placebo mean "
            f"{placebo_gap.mean():+.3f} (sd {placebo_gap.std(ddof=1):.3f}); "
            f"real mask {masked_real} half-hours (includes idle-floor runs), placebo masks "
            f"{placebo_info[(label, 0)]} to "
            f"{max(placebo_info[(label, d)] for d in range(PLACEBO_DRAWS))} half-hours"
        )
        z_scale = (real_shift - placebo_scale.mean()) / max(placebo_scale.std(ddof=1), 1e-9)
        z_rated = (rated_shift - placebo_rated.mean()) / max(placebo_rated.std(ddof=1), 1e-9)
        z_score = (real_gap - placebo_gap.mean()) / max(placebo_gap.std(ddof=1), 1e-9)
        moved = [
            name
            for name, z in (
                ("scale", z_scale),
                ("rated speed", z_rated),
                ("hold-out score", z_score),
            )
            if abs(z) >= 2.0
        ]
        report.append(
            (
                f"{label}: real mask minus placebo mean, in placebo sd: scale {z_scale:+.1f}, "
                f"rated speed {z_rated:+.1f}, hold-out score {z_score:+.1f}; "
            )
            + (
                "beyond 2 placebo sd for "
                + ", ".join(moved)
                + ": the flagged blocks carry some information that random thinning does not"
                if moved
                else (
                    "within 2 placebo sd on all three: the flagged blocks are not "
                    "distinguishable from random thinning here, so they are not shown to be "
                    "hardware failures"
                )
            )
        )
    # Known-answer injection test.
    report.append(
        "-- injection test (pass-1 plant fixed; 20 trials x 4 blocks of 3 to 14 days; "
        "observed power scaled by 1 - share; a block counts as flagged when at least half "
        "its scored days are) --"
    )
    for label, farm in farms.items():
        for rule in RULES:
            base_false = flagged_by_rule[(label, rule)].size / series1[label].day.size
            report.append(
                f"{label}: unmodified series: {base_false * 100:.1f}% of scored days flagged "
                f"at {rule:.0f} sigma"
            )
        for size_index, (size, share) in enumerate(INJECTED_SHARES[label].items()):
            for rule in (MAIN_RULE, 4.0, 2.0):
                rng = np.random.default_rng([3470, common.FARMS.index(label), size_index])
                outcomes = [
                    injection_trial(
                        farm=farm,
                        evidence=evidence1[label],
                        scored_days=series1[label].day,
                        share=share,
                        k=rule,
                        rng=rng,
                    )
                    for _ in range(INJECTION_TRIALS)
                ]
                usable = sum(o["usable"] for o in outcomes)
                detected = sum(o["detected"] for o in outcomes)
                any_day = sum(o["any_day"] for o in outcomes)
                false_share = float(np.mean([o["false_share"] for o in outcomes]))
                criterion = ""
                if label in ("W1", "W2") and rule == MAIN_RULE:
                    criterion = (
                        "; pre-registered criterion (>= 80% of blocks, < 5% false days): "
                        f"{'met' if detected / usable >= 0.8 and false_share < 0.05 else 'NOT met'}"
                    )
                report.append(
                    f"{label} {size}, {rule:.0f} sigma rule: blocks detected {detected} of "
                    f"{usable} ({detected / usable * 100:.0f}%), blocks with at least one "
                    f"flagged day {any_day} of {usable} "
                    f"({any_day / usable * 100:.0f}%); false-flag share of unmodified days in "
                    f"the injected series {false_share * 100:.1f}%{criterion}"
                )
    # Stability across passes.
    report.append("-- stability of the flagged day set across passes (3 sigma) --")
    for label, farm in farms.items():
        s1 = {int(d) for d in flagged_by_rule[(label, MAIN_RULE)]}
        s2 = {int(d) for d in flagged2[label]}
        p3 = fits[("pass3", label)]
        e3 = flags.build_evidence(farm=farm, parameters=p3, prior_scale=priors[label])
        series3 = flags.daily_series(
            farm=farm, log_ratio=e3.log_ratio, weight=e3.weight, rows=farm.is_train
        )
        s3 = {
            int(d)
            for d in flags.flagged_days(
                series=series3, reference=flags.reference_of(series3), k=MAIN_RULE
            )
        }

        def jaccard(a: set[int], b: set[int]) -> float:
            return len(a & b) / max(len(a | b), 1)

        report.append(
            f"{label}: days flagged by the pass-1 plant {len(s1)}, pass-2 plant {len(s2)}, "
            f"pass-3 plant {len(s3)}; Jaccard pass1-pass2 {jaccard(s1, s2):.2f}, pass2-pass3 "
            f"{jaccard(s2, s3):.2f}; "
            "share of pass-1 days also flagged by the pass-2 plant "
            f"{len(s1 & s2) / max(len(s1), 1):.2f}"
        )
        ev3 = evals[("pass3", label)]
        ev2 = evals[("pass2_3", label)]
        kept = ev3.kept
        report.append(
            f"{label}: pass 3 vs pass 2 hold-out kept score {ev3.mean_score(rows=kept):.3f} "
            f"vs {ev2.mean_score(rows=kept):.3f}"
        )
    common.write_report(name="stage3b_report.txt", lines=report)
    print("\n".join(report))


if __name__ == "__main__":
    main()
