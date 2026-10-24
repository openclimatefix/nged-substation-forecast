r"""Stage C: retrain experiment 201's median-correction network on a variant's own rows.

Per variant: rebuild the past-weather arm D (experiment 201's `build_site`, with the variant's
refitted coefficients installed in `e00201_tall.predict_tall`), join it to the variant's
ensemble-era arm D (stage B), add the network's inputs, and train arm 2 (arm D's inputs plus the
calendar and holiday flags) with three seeds on the rows before the scored year. The output holds
the network's median shift `delta` for every scored-year row. `v0r` is the control: with no
response it must reproduce experiment 201's stored `delta` up to the network's training noise.

    source /tmp/env325.sh && systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00325-*/e00325_net.py --variants v0r f1 f1s
"""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import e00201_base as base201  # ty: ignore[unresolved-import]
import e00201_common as c201  # ty: ignore[unresolved-import]
import e00201_data as data201  # ty: ignore[unresolved-import]
import e00201_features as features201  # ty: ignore[unresolved-import]
import e00201_join as join201  # ty: ignore[unresolved-import]
import e00201_nn as nn201  # ty: ignore[unresolved-import]
import e00325_common as common
import e00325_variant as variant_module
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

SEEDS = (0, 1, 2)
ARM = 2
KEYS = ("site", "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "era")


def variant_dir(*, variant: str) -> Path:
    """Return the folder that holds a variant's network files."""
    return common.NET_DIR / variant


def point_201_at(*, variant: str) -> None:
    """Redirect experiment 201's modules to this variant's folders."""
    root = variant_dir(variant=variant)
    c201.WORK_DIR = root
    base201.BASE_DIR = root / "base"
    join201.OUT_DIR = root / "gam_d"
    features201.IN_DIR = root / "gam_d"
    features201.OUT_DIR = root / "features"

    def ens_frame(*, site: str) -> pl.DataFrame:
        stored = pl.read_parquet(common.ARM_D_DIR / variant / f"{site}.parquet")
        return stored.with_columns(arm=pl.lit("d"), era=pl.lit("ens"))

    join201.ens_frame = ens_frame


def early_job(arguments: tuple[str, str]) -> str:
    """Build one Primary's past-weather rows, join, and add the network inputs."""
    variant, site = arguments
    torch.set_num_threads(1)
    variant_module.install(variant=variant)
    point_201_at(variant=variant)
    started = time.monotonic()
    line = base201.build_site(site=site)
    line += " | " + join201.join_site(site=site)
    frame = features201.add_features(site=site)
    features201.OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = features201.OUT_DIR / f"{site}.parquet"
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)
    return f"{variant} {site}: {line} ({time.monotonic() - started:.0f} s)"


def train(*, variant: str) -> str:
    """Train arm 2 with three seeds on the variant's rows and write the scored-year shifts."""
    point_201_at(variant=variant)
    started = time.monotonic()
    frame = data201.load_rows()
    x = data201.design_columns(frame=frame, arm=ARM)
    split = data201.splits(frame=frame)
    q = nn201.quantile_matrix(frame=frame)
    actual = frame["actual"].to_numpy().astype(np.float64)
    test = split == "test"
    keys = frame.select(*KEYS, "actual").filter(pl.Series(test))
    out_dir = variant_dir(variant=variant) / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    notes = []
    for seed in SEEDS:
        fit = nn201.fit_predict(x=x, q=q, actual=actual, split=split, seed=seed)
        part = keys.with_columns(
            *[
                pl.Series(name, fit.quantiles[test][:, i])
                for i, name in enumerate(nn201.quantile_columns())
            ],
            pl.Series("delta", fit.delta[test]),
            pl.Series("log_scale", fit.log_scale[test]),
        )
        target = out_dir / f"arm{ARM}_seed{seed}.parquet"
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        part.write_parquet(temporary)
        temporary.rename(target)
        notes.append(f"seed {seed}: {fit.epochs} epochs, val {fit.best_val:.4f}/{fit.base_val:.4f}")
    return f"{variant} network: {'; '.join(notes)} ({time.monotonic() - started:.0f} s)"


def main() -> None:
    """Rebuild the past-weather rows in parallel, then train each variant's network."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(common.VARIANTS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-early", action="store_true")
    args = parser.parse_args()
    if not args.skip_early:
        jobs = [(v, s) for v in args.variants for s in common.SITES]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for line in pool.imap_unordered(early_job, jobs, chunksize=1):
                print(line, flush=True)
    for variant in args.variants:
        print(train(variant=variant), flush=True)


if __name__ == "__main__":
    main()
