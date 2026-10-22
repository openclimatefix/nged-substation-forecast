"""Compute the solar forecast of every run's half-hour, for the network's inputs and training.

For each run of experiment 136's stored forecasts, the ensemble-mean weather goes through the
unclipped fit that experiment 136 chose for that run's origin. Each half-hour gets the unclipped
solar term `s_raw`, the term under a 60% clip `s_clip`, the term under the profile-fit cap of
experiment 247 `s_learned`, the clear-sky index of the ensemble-mean irradiance, and the fit's cut.
Writes `work/inputs/<site>_<period>.parquet`, keyed by run and valid time.

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00250-*/e00250_inputs.py --workers 4
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00250_common as common
import e00250_ens as ens
import e00250_tall as tall
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_ens import choose_fit
from studies.dp_ens_weather import Calibration, RunWindow
from studies.dp_gam import ClipSpec, pinned_solar_points
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D16")
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
HAND_SET_SHARE: Final[float] = 0.6
OUT_DIR: Final = common.WORK_DIR / "inputs"
LEARNED_CAPS: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00247-dp-learned-export-cap-for-the-solar-clip"
    / "work"
    / "learned_caps.json"
)


def load_context(*, site: str, cache) -> ens.SiteSetup:  # noqa: ANN001
    """Return a site's ensemble setup with experiment 111's calibration and solar shrink."""
    tuned = json.loads((common.SOURCE_WORK_DIR / "tuned.json").read_text())
    with np.load(common.SOURCE_WORK_DIR / "calibration" / f"{site}.npz") as stored:
        calibration = Calibration(
            temperature_shift_c=stored["shift"],
            irradiance_ratio=stored["ratio"],
            solar_shrink=stored["shrink"],
        ).with_solar_shrink(solar_shrink=tuned["solar_shrink"])
    return ens.load_setup(cache=cache, site=site).with_calibration(calibration=calibration)


def run_inputs(*, cache, setup, fits, learned: dict[str, float], index: int) -> dict:  # noqa: ANN001
    """Return one run's solar terms and clear-sky index at every half-hour of its window."""
    run = setup.runs[index]
    fit = choose_fit(fits=fits, origin=setup.origins[index])
    window = RunWindow.build(cache=cache, context=setup.context, run=run)
    n = window.window.stop - window.window.start
    grid = np.arange(n)
    driven = window.driven_caches(context=setup.context, run=ens.mean_run(run=run), members=[0])[0]
    features = driven.sites[setup.site]
    pinned = fit.pinned
    if pinned is None:
        msg = "the control fit has no pinned capacity"
        raise ValueError(msg)
    cut_key = fit.cut.strftime("%Y-%m-%d")

    def solar(clip: ClipSpec | None) -> np.ndarray:
        return pinned_solar_points(
            cache=driven, features=features, pinned=pinned, index=grid, clip=clip
        )

    clear = window.sun.clear_sky_w_m2
    with np.errstate(all="ignore"):
        csi = np.where(clear > 20.0, features.ghi_w_m2 / clear, 0.0)
    return {
        "window_start": window.window.start,
        "s_raw": solar(None),
        "s_clip": solar(ClipSpec(p_max_share=HAND_SET_SHARE)),
        "s_learned": solar(ClipSpec(p_max_share=learned[cut_key])),
        "csi": csi,
        "held_points": np.full(n, float(pinned.held_points)),
        "cut": np.full(n, fit.cut.strftime("%Y-%m-%d")),
    }


def _job(job: tuple[str, str]) -> str:
    torch.set_num_threads(1)
    site, period = job
    started = time.monotonic()
    cache = common.load_cache()
    setup = load_context(site=site, cache=cache)
    fits = tall.load_control_fits(site=site)
    learned = json.loads(LEARNED_CAPS.read_text())[site]["p"]
    frame = pl.read_parquet(common.E136_DIR / "dp_preds" / f"a2_{period}_{site}.parquet")
    pieces = []
    for run_index, rows in frame.group_by("run", maintain_order=True):
        full = run_inputs(
            cache=cache, setup=setup, fits=fits, learned=learned, index=int(run_index[0])
        )
        start = full.pop("window_start")
        position = (
            np.searchsorted(
                cache.end_times, rows["valid_time"].dt.replace_time_zone(None).to_numpy()
            )
            - start
        )
        if position.min() < 0 or position.max() >= len(full["s_raw"]):
            msg = "a stored valid time lies outside the run's window"
            raise ValueError(msg)
        pieces.append(
            pl.DataFrame({name: values[position] for name, values in full.items()}).with_columns(
                run=rows["run"], valid_time=rows["valid_time"]
            )
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / f"{site}_{period}.parquet"
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    out = pl.concat(pieces)
    out.write_parquet(temporary)
    temporary.rename(target)
    return f"{site} {period}: {out.height:,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Compute the inputs of every site and period."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sites", nargs="+", default=list(SITES))
    args = parser.parse_args()
    jobs = [(s, p) for p in PERIODS for s in args.sites]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
