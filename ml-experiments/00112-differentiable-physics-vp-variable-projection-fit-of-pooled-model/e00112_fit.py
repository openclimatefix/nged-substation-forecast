"""Fit the pooled and per-site variable-projection arms, and P-map-same, at one cut.

For a cut the script (1) works out round 1 of the noise standard deviation from the per-site convex
model, (2) fits VP-P from the default start at round 1 and takes round 2 from its residuals, (3)
continues that fit at round 2 and fits VP-P from the second start at round 2, (4) fits VP-S, the
same model one MW substation at a time, from both starts, and (5) fits P-map-same, the gradient fit
on the VP objective, and P-map-polish, its non-linear parameters with the linear coefficients
solved exactly. Each fit is saved under `work/models/`, and its record under
`work/fit_records.json`. A stage whose fits are saved already is skipped.

    uv run --no-sync python ml-experiments/00112-*/e00112_fit.py --cut 2025-07-01 [--stages ...]
"""

import argparse
from datetime import UTC, datetime
from typing import Final

import e00112_common as common
import e00112_vp as vp
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import TrainingWindow, training_rows
from studies.dp_pooled_vp import VpSettings

STAGES: Final[tuple[str, ...]] = ("sigma", "pooled", "per_site", "adam_same")
MIN_SITE_ROWS: Final[int] = 1440
"""A substation with fewer training rows than this (30 days) has no per-site fit at the cut."""
EARLIEST_ROW: Final[datetime] = datetime(2021, 7, 1, tzinfo=UTC)
"""The replicate year's replicas train from here, after the lockdown windows."""


def parse_cut(text: str) -> datetime:
    """Return the cut named on the command line, which must be one of `common.CUTS`."""
    cut = datetime.fromisoformat(text).replace(tzinfo=UTC)
    if cut not in common.CUTS:
        msg = f"{text} is not one of the cuts {[c.date().isoformat() for c in common.CUTS]}"
        raise ValueError(msg)
    return cut


def pool_batch(*, cache: FeatureCache, sites: tuple[str, ...], cut: datetime):  # noqa: ANN201
    """Return the float64 training rows of `sites` at `cut`, checking the cache reaches back."""
    if cache.end_times[0] > common.to_64(moment=EARLIEST_ROW):
        msg = "the feature cache does not cover the replicate year's training window"
        raise ValueError(msg)
    return common.pooled_batch_for(
        cache=cache, sites=sites, cut=cut, device=vp.DEVICE, dtype=common.BATCH_DTYPE
    )


def stage_sigma(*, cache: FeatureCache, cut: datetime) -> None:
    """Store round 1 of the noise standard deviation, from the per-site convex model."""
    if "round1" in common.read_sigma_table().get(cut.date().isoformat(), {}):
        return
    sites = tuple(
        site
        for site in common.MW_POOL
        if len(
            training_rows(
                features=cache.sites[site],
                cache=cache,
                cut=cut,
                window=TrainingWindow(months=common.WINDOW_MONTHS),
            )
        )
        >= MIN_SITE_ROWS
    )
    value = vp.convex_training_sd(cache=cache, sites=sites, cut=cut)
    common.write_sigma(cut=cut, round_name="round1", value=value)
    print(f"sigma round 1 at {cut.date()}: {value:.3f} points from {sites}", flush=True)


def stage_pooled(*, cache: FeatureCache, cut: datetime) -> None:
    """Fit VP-P from both starts; take round 2 of the noise standard deviation from the first."""
    batch = pool_batch(cache=cache, sites=common.POOL, cut=cut)
    first = "vp_p_s0_r1"
    if not vp.has_fit(name=first, cut=cut):
        settings = common.vp_settings(cut=cut, round_name="round1")
        model, result = vp.fit_pooled_arm(
            batch=batch,
            settings=settings,
            start="s0",
            fallback_steps=common.FALLBACK_STEPS_PRIMARY,
        )
        vp.save_fit(
            name=first, cut=cut, model=model, record=vp.record_of(result=result, rows=batch.n_rows)
        )
        common.write_sigma(
            cut=cut, round_name="round2", value=vp.training_residual_sd(model=model, batch=batch)
        )
        print(
            f"{first} at {cut.date()}: {result.seconds:.0f} s, converged {result.converged}",
            flush=True,
        )
    settings = common.vp_settings(cut=cut, round_name="round2")
    if not vp.has_fit(name="vp_p_s0", cut=cut):
        state = torch.load(vp.model_path(name=first, cut=cut), map_location=batch.y.device)
        model, result = vp.fit_pooled_arm(
            batch=batch,
            settings=settings,
            start="s0",
            fallback_steps=common.FALLBACK_STEPS_SECONDARY,
            init_state=state,
        )
        vp.save_fit(
            name="vp_p_s0",
            cut=cut,
            model=model,
            record=vp.record_of(result=result, rows=batch.n_rows),
        )
        print(
            f"vp_p_s0 at {cut.date()}: {result.seconds:.0f} s, converged {result.converged}",
            flush=True,
        )
    if not vp.has_fit(name="vp_p_s1", cut=cut):
        model, result = vp.fit_pooled_arm(
            batch=batch,
            settings=settings,
            start="s1",
            fallback_steps=common.FALLBACK_STEPS_PRIMARY,
        )
        vp.save_fit(
            name="vp_p_s1",
            cut=cut,
            model=model,
            record=vp.record_of(result=result, rows=batch.n_rows),
        )
        print(
            f"vp_p_s1 at {cut.date()}: {result.seconds:.0f} s, converged {result.converged}",
            flush=True,
        )


def stage_per_site(*, cache: FeatureCache, cut: datetime) -> None:
    """Fit VP-S: the same model one MW substation at a time, from both starts."""
    settings = common.vp_settings(cut=cut, round_name="round2")
    for site in common.MW_POOL:
        batch = pool_batch(cache=cache, sites=(site,), cut=cut)
        if batch.n_rows < MIN_SITE_ROWS:
            continue
        for start in common.STARTS:
            name = f"vp_s_{site.lower()}_{start}"
            if vp.has_fit(name=name, cut=cut):
                continue
            model, result = vp.fit_pooled_arm(
                batch=batch,
                settings=settings,
                start=start,
                fallback_steps=common.FALLBACK_STEPS_SECONDARY,
            )
            vp.save_fit(
                name=name,
                cut=cut,
                model=model,
                record=vp.record_of(result=result, rows=batch.n_rows),
            )
            print(
                f"{name} at {cut.date()}: {result.seconds:.0f} s, converged {result.converged}",
                flush=True,
            )


def stage_adam_same(*, cache: FeatureCache, cut: datetime) -> None:
    """Fit P-map-same and P-map-polish; record both objectives, in nats, on the VP objective."""
    batch = pool_batch(cache=cache, sites=common.POOL, cut=cut)
    settings: VpSettings = common.vp_settings(cut=cut, round_name="round2")
    if not vp.has_fit(name="p_map_same", cut=cut):
        model, record = vp.fit_adam_same(batch=batch, settings=settings)
        vp.save_fit(name="p_map_same", cut=cut, model=model, record=record)
        print(f"p_map_same at {cut.date()}: {record['objective']:.0f} nats", flush=True)
    if not vp.has_fit(name="p_map_polish", cut=cut):
        model = vp.load_fit(name="p_map_same", cut=cut, batch=batch)
        same = vp.read_record(name="p_map_same", cut=cut)["objective"]
        polished = vp.polish(model=model, batch=batch, settings=settings)
        record = {"objective": polished, "objective_p_map_same": same, "rows": batch.n_rows}
        vp.save_fit(name="p_map_polish", cut=cut, model=model, record=record)
        print(f"p_map_polish at {cut.date()}: {polished:.0f} nats against {same:.0f}", flush=True)
        if polished > same + 1.0:
            msg = "control failed: the exact solve raised the objective of P-map-same"
            raise RuntimeError(msg)


def main() -> None:
    """Run the requested stages at the requested cut."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut", required=True)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    arguments = parser.parse_args()
    cut = parse_cut(arguments.cut)
    torch.set_num_threads(2)
    cache = common.load_cache()
    runners = {
        "sigma": stage_sigma,
        "pooled": stage_pooled,
        "per_site": stage_per_site,
        "adam_same": stage_adam_same,
    }
    for stage in STAGES:
        if stage in arguments.stages:
            runners[stage](cache=cache, cut=cut)


if __name__ == "__main__":
    main()
