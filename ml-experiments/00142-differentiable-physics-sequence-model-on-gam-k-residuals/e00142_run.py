r"""Build the samples, run the known-answer gate, fit the networks, and write the arms' frames.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \\
        uv run --no-sync python ml-experiments/00142-*/e00142_run.py <stage>

Stages: `samples`, `gate`, `fit`, `frames`.
"""

import argparse
import json
import pickle
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Final

import e00142_arms as arms
import e00142_common as common
import e00142_data as data_module
import e00142_ens as ens
import e00142_model as model
import e00142_profiles as profiles
import e00142_ridge as ridge
import e00142_site as site_module
import numpy as np
import polars as pl
import torch
from studies.dp_ens_weather import N_LEAD_BANDS

SITES: Final[tuple[str, ...]] = common.SCORED_SITES
SAMPLES_PATH: Final[Path] = common.WORK_DIR / "samples.pkl"
GATE_PATH: Final[Path] = common.WORK_DIR / "gate.json"
PREDICTIONS_PATH: Final[Path] = common.WORK_DIR / "predictions.pkl"
DEVICE: Final[str] = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS: Final[tuple[int, ...]] = (0, 1)
SCORED_FOLDS: Final[int] = 5
TUNING_FOLDS: Final[int] = 3
SCORED_EMBARGO_DAY: Final[int] = int(
    (np.datetime64("2025-06-28", "D") - data_module.EPOCH).astype(int)
)
"""Training targets end on this UTC day, 2 days before the scored year, so no scored reading is
read through a training target or window."""
INJECTION_SHIFT: Final[float] = 3.0
INJECTION_START: Final[datetime] = datetime(2024, 4, 14, tzinfo=UTC)
INJECTION_WEEKS: Final[int] = 8
SATURDAY_BINS: Final[range] = range(34, 42)
"""UTC half-hours 17:00 to 21:00."""


@lru_cache(maxsize=2)
def get_site(site: str) -> site_module.SiteData:
    """Load a site once per process."""
    torch.set_num_threads(1)
    return site_module.load_site(site=site)


def build_all(*, injection: str | None = None) -> list[data_module.Samples]:
    """Build every substation's samples, with an optional injection (`level` or `saturday`)."""
    out = []
    for site in SITES:
        data = get_site(site)
        _k0, q50, quantiles = data_module.load_k0_with_quantiles(data=data, site=site)
        setup = ens.load_setup(cache=data.cache, site=site)
        actual, stream = None, None
        if injection is not None:
            add = injection_signal(data=data, kind=injection)
            actual, stream = data.actual + add[data.position], data.stream + add
        samples = data_module.build_samples(
            data=data,
            k0_q50=q50,
            k0_quantiles=quantiles,
            setup_runs=setup.runs,
            actual=actual,
            stream=stream,
        )
        if injection is None:
            data_module.assert_no_lookahead(data=data, samples=samples)
        out.append(samples)
    return out


def injection_signal(*, data: site_module.SiteData, kind: str) -> np.ndarray:
    """Return the injected shift at every grid position: a level step or a Saturday evening."""
    ends = data.cache.end_times
    start = np.datetime64(INJECTION_START.replace(tzinfo=None), "us")
    stop = start + np.timedelta64(INJECTION_WEEKS * 7, "D")
    inside = (ends > start) & (ends <= stop)
    if kind == "level":
        return np.where(inside, INJECTION_SHIFT, 0.0)
    first_day = (ends[0] - np.timedelta64(30, "m")).astype("datetime64[D]")
    positions = np.arange(len(ends))
    day_number = (first_day - data_module.EPOCH).astype(int) + positions // data_module.N_BINS
    weekday = (day_number + 3) % 7
    saturday = (weekday == 5) & np.isin(positions % data_module.N_BINS, list(SATURDAY_BINS))
    return np.where(inside & saturday, INJECTION_SHIFT, 0.0)


def training_candidates(*, pooled: model.Pooled, kind: str) -> np.ndarray:
    """Return the sample indices a fit may train on: `tuning` months or all pre-scored months."""
    has_target = np.isfinite(pooled.y).any(axis=1)
    if kind == "tuning":
        months = np.array([100 * y + m for y, m in common.TUNING_MONTHS])
        keep = np.isin(pooled.month_code, months) & (pooled.period == "tuning")
    else:
        keep = pooled.target_day <= SCORED_EMBARGO_DAY
    return np.flatnonzero(keep & has_target)


def fit_ensemble(
    *,
    pooled: model.Pooled,
    candidates: np.ndarray,
    n_folds: int,
    mode: str,
    use_history: bool = True,
) -> tuple[list[model.SequenceNet], dict, dict[int, np.ndarray]]:
    """Train fold models and return them, their logs, and the out-of-fold samples of each fold."""
    blocks = model.month_blocks(month_codes=pooled.month_code[candidates], n_folds=n_folds)
    nets, logs, held_out = [], [], {}
    for fold, block in enumerate(blocks):
        train, validation = model.purged_split(pooled=pooled, candidates=candidates, block=block)
        for seed in SEEDS:
            net, log = model.fit(
                pooled=pooled,
                train=train,
                validation=validation,
                n_sites=len(SITES),
                mode=mode,
                seed=1000 * fold + seed,
                device=DEVICE,
                use_history=use_history,
            )
            nets.append(net)
            logs.append({"fold": fold, "seed": seed, **log, "n_train": len(train)})
        held_out[fold] = validation
    return nets, {"folds": logs}, held_out


def out_of_fold(
    *, pooled: model.Pooled, nets: list[model.SequenceNet], held_out: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict each held-out sample with its own fold's networks.

    Args:
        pooled: The pooled samples.
        nets: The fold networks, in fold order with equal numbers per fold.
        held_out: Each fold's held-out sample indices.

    Returns:
        The sample indices, and their correction, lower scale, and upper scale.
    """
    per_fold = len(nets) // len(held_out)
    indices, parts = [], []
    for fold, validation in held_out.items():
        fold_nets = nets[fold * per_fold : (fold + 1) * per_fold]
        parts.append(model.predict(nets=fold_nets, pooled=pooled, index=validation, device=DEVICE))
        indices.append(validation)
    index = np.concatenate(indices)
    return (
        index,
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
        np.concatenate([p[2] for p in parts]),
    )


def recovery(
    *, pooled: model.Pooled, nets: list[model.SequenceNet], clean: model.Pooled, kind: str
) -> dict:
    """Return the share of the injected shift the nets' median correction recovers.

    The correction on injected inputs minus the correction on clean inputs, at the injected bins,
    divided by the shift, for the third day (level) or the third Saturday (shape) onwards.
    """
    index = np.arange(len(pooled.y))
    inj_c, _, _ = model.predict(nets=nets, pooled=pooled, index=index, device=DEVICE)
    cln_c, _, _ = model.predict(nets=nets, pooled=clean, index=index, device=DEVICE)
    start_day = int(
        (np.datetime64(INJECTION_START.replace(tzinfo=None), "D") - data_module.EPOCH).astype(int)
    )
    since = pooled.target_day - start_day
    weekday = (pooled.target_day + 3) % 7
    if kind == "level":
        rows = (since >= 3) & (since < INJECTION_WEEKS * 7)
        bins = np.arange(48)
    else:
        saturdays = np.flatnonzero((weekday == 5) & (since >= 0) & (since < INJECTION_WEEKS * 7))
        rows = np.zeros(len(since), dtype=bool)
        rows[saturdays[np.argsort(pooled.target_day[saturdays], kind="stable")]] = True
        rows &= since >= 14  # third Saturday on
        bins = np.array(list(SATURDAY_BINS))
    share = (inj_c[rows][:, bins] - cln_c[rows][:, bins]) / INJECTION_SHIFT
    return {"share_median": float(np.median(share)), "n": int(rows.sum())}


def r_recovery(*, kind: str) -> float:
    """Return the median share of the injection recipe R (kappa 0.75) recovers, over substations."""
    shares = []
    spec = profiles.ProfileSpec(kappa=0.75, holiday_rule=False)
    start_day = int(
        (np.datetime64(INJECTION_START.replace(tzinfo=None), "D") - data_module.EPOCH).astype(int)
    )
    for site in SITES:
        data = get_site(site)
        _k0, q50, _ = data_module.load_k0_with_quantiles(data=data, site=site)
        add = injection_signal(data=data, kind=kind)
        injected = data.actual + add[data.position]
        clean_blocks, targets = profiles.day_blocks(data=data, k0_q50=q50, actual=data.actual)
        inj_blocks, _ = profiles.day_blocks(data=data, k0_q50=q50, actual=injected)
        clean_shift = profiles.profile_shift_by_run(blocks=clean_blocks, targets=targets, spec=spec)
        inj_shift = profiles.profile_shift_by_run(blocks=inj_blocks, targets=targets, spec=spec)
        since = targets - start_day
        weekday = (targets + 3) % 7
        if kind == "level":
            rows, bins = (since >= 3) & (since < INJECTION_WEEKS * 7), np.arange(48)
        else:
            rows = (weekday == 5) & (since >= 14) & (since < INJECTION_WEEKS * 7)
            bins = np.array(list(SATURDAY_BINS))
        # R's shift is by local half-hour; here a UTC-bin proxy suffices because the gate compares
        # shares, and BST moves the Saturday bins by two half-hours at most.
        delta = -(inj_shift[rows][:, bins] - clean_shift[rows][:, bins]) / INJECTION_SHIFT
        shares.append(float(np.median(delta)))
    return float(np.median(shares))


def stage_samples() -> None:
    """Build and store the clean samples."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_all()
    SAMPLES_PATH.write_bytes(pickle.dumps(samples))
    print({s.site: len(s.run) for s in samples})


def stage_gate() -> None:
    """Run the known-answer gate on real residuals (no replicate or scored forecast is read)."""
    clean = model.pool(samples=pickle.loads(SAMPLES_PATH.read_bytes()))
    candidates = training_candidates(pooled=clean, kind="tuning")
    nets, log, _held = fit_ensemble(
        pooled=clean, candidates=candidates, n_folds=TUNING_FOLDS, mode="median"
    )
    result: dict = {"train_log": log}
    for kind in ("level", "saturday"):
        injected = model.pool(samples=build_all(injection=kind))
        window = np.flatnonzero(
            (
                injected.target_day
                - int(
                    (
                        np.datetime64(INJECTION_START.replace(tzinfo=None), "D") - data_module.EPOCH
                    ).astype(int)
                )
                >= 0
            )
            & (
                injected.target_day
                < int(
                    (
                        np.datetime64(INJECTION_START.replace(tzinfo=None), "D") - data_module.EPOCH
                    ).astype(int)
                )
                + INJECTION_WEEKS * 7
            )
        )
        sub = _subset(pooled=injected, index=window)
        sub_clean = _subset(pooled=clean, index=window)
        result[kind] = {
            "network": recovery(pooled=sub, nets=nets, clean=sub_clean, kind=kind),
            "R": r_recovery(kind=kind),
        }
    result["absorption"] = absorption_test(
        clean=clean,
        injected=model.pool(samples=build_all(injection="saturday")),
        baseline=nets,
    )
    result["passed"] = gate_passed(result)
    common.write_json_atomic(path=GATE_PATH, content=result)
    print(json.dumps({k: v for k, v in result.items() if k != "train_log"}, indent=1))


def gate_passed(result: dict) -> bool:
    """Apply the plan's pass conditions to the gate numbers."""
    level = result["level"]["network"]["share_median"]
    shape = result["saturday"]["network"]["share_median"]
    return bool(
        level >= 0.30
        and level > result["level"]["R"]
        and shape >= 0.15
        and result["absorption"]["clean_share"] < 0.25
    )


def _subset(*, pooled: model.Pooled, index: np.ndarray) -> model.Pooled:
    return model.Pooled(
        **{
            name: getattr(pooled, name)[index]
            for name in (
                "history",
                "target",
                "offsets",
                "y",
                "site",
                "target_day",
                "month_code",
                "period",
                "where",
            )
        }
    )


def absorption_test(*, clean: model.Pooled, injected: model.Pooled, baseline: list) -> dict:
    """Train with the injection in weeks A, evaluate weeks B with and without the injection."""
    start_day = int(
        (np.datetime64(INJECTION_START.replace(tzinfo=None), "D") - data_module.EPOCH).astype(int)
    )
    since = injected.target_day - start_day
    weeks_a = np.flatnonzero((since >= 0) & (since < 28))
    weeks_b = np.flatnonzero((since >= 28) & (since < 56))
    base = training_candidates(pooled=clean, kind="tuning")
    stacked = model.Pooled(
        **{
            name: np.concatenate([getattr(clean, name)[base], getattr(injected, name)[weeks_a]])
            for name in (
                "history",
                "target",
                "offsets",
                "y",
                "site",
                "target_day",
                "month_code",
                "period",
                "where",
            )
        }
    )
    candidates = np.arange(len(stacked.y))
    nets, _log, _ = fit_ensemble(
        pooled=stacked, candidates=candidates, n_folds=TUNING_FOLDS, mode="median"
    )
    b_inj, b_clean = _subset(pooled=injected, index=weeks_b), _subset(pooled=clean, index=weeks_b)
    c_inj, _, _ = model.predict(
        nets=nets, pooled=b_inj, index=np.arange(len(weeks_b)), device=DEVICE
    )
    c_cln, _, _ = model.predict(
        nets=nets, pooled=b_clean, index=np.arange(len(weeks_b)), device=DEVICE
    )
    c_base, _, _ = model.predict(
        nets=baseline, pooled=b_clean, index=np.arange(len(weeks_b)), device=DEVICE
    )
    saturday = ((b_clean.target_day + 3) % 7) == 5
    bins = np.array(list(SATURDAY_BINS))
    return {
        "clean_share": float(np.median((c_cln - c_base)[saturday][:, bins]) / INJECTION_SHIFT),
        "injected_share": float(np.median((c_inj - c_base)[saturday][:, bins]) / INJECTION_SHIFT),
        "n_saturdays": int(saturday.sum()),
    }


def _fit_period(*, pooled: model.Pooled, kind: str, n_folds: int) -> dict:
    """Fit every arm's models for one training set and return their predictions and logs.

    The result holds, per arm, `oof` (sample index -> correction, lower, upper for the samples a
    fold held out) and `ensemble` (the fold models' mean prediction for every sample).
    """
    candidates = training_candidates(pooled=pooled, kind=kind)
    everyone = np.arange(len(pooled.y))
    out: dict = {"n_candidates": len(candidates)}
    for arm, mode, use_history in (
        ("SEQ-median", "median", True),
        ("SEQ-quantile", "quantile", True),
        ("SEQ-blind", "quantile", False),
    ):
        nets, log, held = fit_ensemble(
            pooled=pooled,
            candidates=candidates,
            n_folds=n_folds,
            mode=mode,
            use_history=use_history,
        )
        index, c, lo, up = out_of_fold(pooled=pooled, nets=nets, held_out=held)
        ensemble = model.predict(nets=nets, pooled=pooled, index=everyone, device=DEVICE)
        out[arm] = {"oof": (index, c, lo, up), "ensemble": ensemble, "log": log}
        print(
            arm,
            kind,
            [
                (f["fold"], round(f["train"], 3), round(f["validation"], 3), f["epochs"])
                for f in log["folds"]
            ],
            flush=True,
        )
    fits, held, oof, alpha = ridge.fit_folds(
        pooled=pooled, candidates=candidates, n_folds=n_folds, n_sites=len(SITES)
    )
    x_all = ridge.design(pooled=pooled, index=everyone, n_sites=len(SITES))
    mean_c = np.mean([ridge.predict_ridge(fitted=f, x=x_all) for f in fits], axis=0)
    ones = np.ones_like(mean_c)
    index = np.concatenate([held[f] for f in sorted(held)])
    c_oof = np.concatenate([oof[f] for f in sorted(oof)])
    out["SEQ-ridge"] = {
        "oof": (index, c_oof, np.ones_like(c_oof), np.ones_like(c_oof)),
        "ensemble": (mean_c, ones, ones),
        "log": {"alpha": alpha},
    }
    return out


def stage_fit() -> None:
    """Fit the replicate-period and scored-year networks (the gate must have passed)."""
    gate = json.loads(GATE_PATH.read_text())
    assert gate["passed"], "the known-answer gate did not pass; no scored fit"
    pooled = model.pool(samples=pickle.loads(SAMPLES_PATH.read_bytes()))
    result = {
        "tuning": _fit_period(pooled=pooled, kind="tuning", n_folds=TUNING_FOLDS),
        "scored": _fit_period(pooled=pooled, kind="scored", n_folds=SCORED_FOLDS),
    }
    PREDICTIONS_PATH.write_bytes(pickle.dumps(result))


def sample_parameters(*, result: dict, arm: str, pooled: model.Pooled) -> tuple[np.ndarray, ...]:
    """Return each pooled sample's correction and scales as its own period's nets give them.

    A tuning-period sample takes its out-of-fold prediction from the replicate fit, a replicate
    sample the replicate fit's ensemble, a scored sample the scored fit's ensemble, and any other
    sample no correction (K0 as it is).
    """
    n = len(pooled.y)
    c, lo, up = np.zeros((n, 48)), np.ones((n, 48)), np.ones((n, 48))
    ens_t, ens_s = result["tuning"][arm]["ensemble"], result["scored"][arm]["ensemble"]
    replicate = pooled.period == "replicate"
    scored = pooled.period == "scored"
    for target, source in ((c, 0), (lo, 1), (up, 2)):
        target[replicate] = ens_t[source][replicate]
        target[scored] = ens_s[source][scored]
    index, oc, olo, oup = result["tuning"][arm]["oof"]
    c[index], lo[index], up[index] = oc, olo, oup
    return c, lo, up


def _rows_correction(
    *, data: site_module.SiteData, samples: data_module.Samples, c: np.ndarray
) -> arms.Correction:
    k0, _q50, _ = data_module.load_k0_with_quantiles(data=data, site=data.site)
    shift = k0.shape.copy()
    shift[samples.rows] += c
    return arms.Correction(level=k0.level, anomaly=k0.anomaly, shape=shift)


def _quantile_frame(
    *,
    data: site_module.SiteData,
    samples: data_module.Samples,
    c: np.ndarray,
    lo: np.ndarray,
    up: np.ndarray,
    arm: str,
) -> pl.DataFrame:
    """Return K0's frame with D+1 quantiles moved to the network's: median shift, scaled offsets."""
    reference = pl.read_parquet(common.frame_path(arm="K0", site=data.site))
    ends = data.cache.end_times
    origin = np.repeat(data.run_origins[samples.run], 48)
    valid = ends[data.position[samples.rows.reshape(-1)]]
    table = pl.DataFrame(
        {
            "power_fcst_init_time": pl.Series(origin).dt.replace_time_zone("UTC"),
            "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
            "c": c.reshape(-1),
            "lo": lo.reshape(-1),
            "up": up.reshape(-1),
        }
    )
    joined = reference.join(
        table, on=["power_fcst_init_time", "valid_time"], how="left"
    ).with_columns(
        pl.col("c").fill_null(0.0), pl.col("lo").fill_null(1.0), pl.col("up").fill_null(1.0)
    )
    columns = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
    exprs = []
    for position, name in enumerate(columns):
        side = "lo" if position < model.MEDIAN_INDEX else "up"
        offset = pl.col(name) - pl.col("q50")
        exprs.append(
            (
                pl.col("q50")
                + pl.col("c")
                + (pl.col(side) * offset if position != model.MEDIAN_INDEX else 0.0)
            )
            .cast(pl.Float32)
            .alias(name)
        )
    return joined.with_columns(
        *exprs,
        (pl.col("mean") + pl.col("c")).cast(pl.Float32).alias("mean"),
        pl.lit(arm).alias("arm"),
    ).drop("c", "lo", "up")


def stage_frames() -> None:
    """Write every network arm's forecast frame, and link K0 and R from experiment 115."""
    result = pickle.loads(PREDICTIONS_PATH.read_bytes())
    all_samples = pickle.loads(SAMPLES_PATH.read_bytes())
    pooled = model.pool(samples=all_samples)
    for arm in ("K0", "R"):
        for site in SITES:
            target = common.frame_path(arm=arm, site=site)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(common.SRC_115_WORK_DIR / "frames" / f"{arm}_{site}.parquet")
    for arm in ("SEQ-quantile", "SEQ-blind"):
        c, lo, up = sample_parameters(result=result, arm=arm, pooled=pooled)
        for i, samples in enumerate(all_samples):
            mine = pooled.where[:, 0] == i
            frame = _quantile_frame(
                data=get_site(samples.site),
                samples=samples,
                c=c[mine],
                lo=lo[mine],
                up=up[mine],
                arm=arm,
            )
            frame.write_parquet(common.frame_path(arm=arm, site=samples.site))
    for arm in ("SEQ-median", "SEQ-ridge"):
        c, _lo, _up = sample_parameters(result=result, arm=arm, pooled=pooled)
        corrections, spreads, scores = {}, {}, {}
        for i, samples in enumerate(all_samples):
            data = get_site(samples.site)
            correction = _rows_correction(data=data, samples=samples, c=c[pooled.where[:, 0] == i])
            spread = arms.fit_spread(data=data, correction=correction.total)
            assert spread is not None
            corrections[samples.site], spreads[samples.site] = correction, spread
            scores[samples.site] = {
                str(f): arms.tuning_band_scores(
                    data=data,
                    correction=correction.total,
                    spread=spread.with_factors(factors=[f] * N_LEAD_BANDS),
                ).tolist()
                for f in arms.FACTOR_GRID
            }
        totals = np.zeros((len(arms.FACTOR_GRID), N_LEAD_BANDS, 2))
        for site in common.DECISION_SITES:
            for position, f in enumerate(arms.FACTOR_GRID):
                totals[position] += np.array(scores[site][str(f)])
        factors = [
            float(arms.FACTOR_GRID[k]) for k in (totals[:, :, 0] / totals[:, :, 1]).argmin(axis=0)
        ]
        print(arm, "factors", factors, flush=True)
        for samples in all_samples:
            data = get_site(samples.site)
            frame = arms.frame_for_arm(
                data=data,
                arm=arm,
                correction=corrections[samples.site],
                spread=spreads[samples.site].with_factors(factors=factors),
            )
            arms.replace_period_masks(frame=frame, data=data).write_parquet(
                common.frame_path(arm=arm, site=samples.site)
            )


def main() -> None:
    """Run a stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["samples", "gate", "fit", "frames"])
    args = parser.parse_args()
    {"samples": stage_samples, "gate": stage_gate, "fit": stage_fit, "frames": stage_frames}[
        args.stage
    ]()


if __name__ == "__main__":
    main()
