"""Fit experiment 00285's variants, one worker per farm, and store their quantile forecasts.

Each (variant, seed) writes, per farm, `work/preds/<name>__<farm>__s<seed>__era5.npy` (11 quantiles
for the scored cases with ERA5 as the weather) and, for variants with `ens=True`,
`work/preds/<name>__<farm>__s<seed>__ens.npy` (the equal-weight mixture of the 51 ensemble
members' quantiles, before the spread factor). Existing files are skipped, so a crashed run
resumes. Fitted values go to `work/facts/<name>__<farm>.json`.

    cd ml-experiments/00285-wind-effective-wind-adapters
    uv run --no-sync python e00285_run.py [--farms W1,W2,W3] [--no-ens] NAME [NAME ...]
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("E00285_TORCH_THREADS", "2")

import e00285_common as common
import e00285_models as models
import numpy as np
import polars as pl

PRED_DIR = common.WORK_DIR / "preds"
FACT_DIR = common.WORK_DIR / "facts"


def _write_atomic(*, path: Path, array: np.ndarray) -> None:
    """Write `array` to `path` through a temporary file."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.rename(path)


def farm_frames(
    *, farm: str, need_ens: bool
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame | None, pl.DataFrame]:
    """Return one farm's (training rows, ERA5 case frame, mask, member frame or None, cases)."""
    sites = common.site_table().filter(pl.col("site") == farm)
    era5 = common.era5_rows(sites=sites)
    train = models.add_derived_columns(frame=common.training_table(sites=sites, era5=era5))
    cases = common.cases_table(sites=sites)
    era5_cases = models.add_derived_columns(
        frame=cases.join(era5, on=["time_series_id", "valid_time"], how="left", validate="m:1")
    )
    if era5_cases["wind_speed_100m"].null_count():
        msg = f"{farm}: some cases have no ERA5 weather"
        raise ValueError(msg)
    members = None
    if need_ens:
        members = common.ens_member_rows(sites=sites).join(
            cases.select(*common.KEY_COLUMNS), on=list(common.KEY_COLUMNS), how="semi"
        )
        members = models.add_derived_columns(frame=members)
        if members.height != cases.height * common.MEMBERS or not members.gather_every(
            common.MEMBERS
        ).select(common.KEY_COLUMNS).equals(cases.select(common.KEY_COLUMNS)):
            msg = f"{farm}: the members do not line up with the cases"
            raise ValueError(msg)
    return train, era5_cases, common.idle_mask(), members, cases


def worker(*, farm: str, names: list[str], with_ens: bool) -> None:
    """Fit and store every requested variant for one farm."""
    specs = models.build_specs()
    chosen = [specs[n] for n in names]
    need_ens = with_ens and any(s.ens for s in chosen)
    train_all, era5_cases, mask, members, _ = farm_frames(farm=farm, need_ens=need_ens)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    FACT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in chosen:
        train = models.select_training_rows(train=train_all, spec=spec, mask=mask)
        for seed in spec.seeds:
            targets = {"era5": PRED_DIR / f"{spec.name}__{farm}__s{seed}__era5.npy"}
            if spec.ens and with_ens:
                targets["ens"] = PRED_DIR / f"{spec.name}__{farm}__s{seed}__ens.npy"
            if all(path.exists() for path in targets.values()):
                continue
            started = time.time()
            evals: dict[str, pl.DataFrame] = {"era5": era5_cases}
            if "ens" in targets and members is not None:
                evals["ens"] = members
            predicted, facts = models.fit_predict(
                spec=spec, train=train, evals=evals, seed=seed, with_idle_layer=farm != "W2"
            )
            for name, quantiles in predicted.items():
                stored = (
                    common.mixture_quantiles(
                        member_quantiles=quantiles.reshape(-1, common.MEMBERS, 11)
                    )
                    if name == "ens"
                    else quantiles
                )
                _write_atomic(path=targets[name], array=stored.astype(np.float32))
            if seed == spec.seeds[0]:
                (FACT_DIR / f"{spec.name}__{farm}.json").write_text(
                    json.dumps({**facts, "train_rows": train.height})
                )
            print(f"{farm} {spec.name} seed {seed}: {time.time() - started:.0f} s", flush=True)


def main() -> None:
    """Start one worker process per farm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default="W1,W2,W3", help="comma-separated farm labels")
    parser.add_argument("--no-ens", action="store_true")
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    unknown = set(args.names) - set(models.build_specs())
    if unknown:
        sys.exit(f"unknown variants: {sorted(unknown)}")
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=worker, kwargs={"farm": farm, "names": args.names, "with_ens": not args.no_ens}
        )
        for farm in args.farms.split(",")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if any(process.exitcode for process in processes):
        sys.exit("a worker failed")


if __name__ == "__main__":
    main()
