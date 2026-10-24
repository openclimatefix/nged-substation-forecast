# ruff: noqa: E501
"""Load experiment 334's dense tensors and pick the pool, validation, and test runs of one refit.

`combine()` joins the six Primaries' arrays into `all_*.npy` files once; `load()` memory-maps them.
The pool, early-stopping split, and test selection follow experiment 330 (monthly refits on a
24-month pool, every fourth pool month held out), except that a run joins the pool only when its
whole 240-hour window ends before the refit date.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl

import e00334_common as common  # isort: skip


def combine() -> None:
    """Concatenate the per-site files (run order: site order of `common.SITES`, then origin)."""
    runs = pl.concat(
        [pl.read_parquet(common.PREP_DIR / f"{s}_runs.parquet") for s in common.SITES]
    ).with_columns(
        site_id=pl.col("site").replace_strict(
            {s: i for i, s in enumerate(common.SITES)}, return_dtype=pl.Int32
        )
    )
    runs.write_parquet(common.PREP_DIR / "all_runs.parquet")
    for name, suffix in (
        ("x", "x"),
        ("y", "y"),
        ("keep", "keep"),
        ("hist", "hist"),
        ("temp24", "valid_ens_temp24"),
    ):
        parts = [np.load(common.PREP_DIR / f"{s}_{suffix}.npy") for s in common.SITES]
        np.save(common.PREP_DIR / f"all_{name}.npy", np.concatenate(parts, axis=0))


@dataclass(frozen=True)
class Data:
    """The memory-mapped arrays and run table."""

    runs: pl.DataFrame
    x: np.ndarray
    y: np.ndarray
    keep: np.ndarray
    hist: np.ndarray
    temp24: np.ndarray


def load() -> Data:
    """Memory-map the combined arrays."""
    return Data(
        runs=pl.read_parquet(common.PREP_DIR / "all_runs.parquet"),
        x=np.load(common.PREP_DIR / "all_x.npy", mmap_mode="r"),
        y=np.load(common.PREP_DIR / "all_y.npy", mmap_mode="r"),
        keep=np.load(common.PREP_DIR / "all_keep.npy", mmap_mode="r"),
        hist=np.load(common.PREP_DIR / "all_hist.npy", mmap_mode="r"),
        temp24=np.load(common.PREP_DIR / "all_temp24.npy", mmap_mode="r"),
    )


def month_start(*, label: str) -> datetime:
    """Return the first instant of a `YYYY-MM` month, UTC."""
    year, month = map(int, label.split("-"))
    return datetime(year, month, 1, tzinfo=UTC)


def months_before(*, when: datetime, months: int) -> datetime:
    """Return the first of the month `months` before `when`."""
    total = when.year * 12 + when.month - 1 - months
    return datetime(total // 12, total % 12 + 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Split:
    """Run indices of one refit."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_for(*, runs: pl.DataFrame, label: str) -> Split:
    """Return the train, validation, and test runs of the refit at the start of month `label`."""
    cut = month_start(label=label)
    origin = runs["power_fcst_init_time"]
    pool = ((origin + timedelta(hours=240)) < cut) & (
        origin >= months_before(when=cut, months=common.POOL_MONTHS)
    )
    pool_np = pool.to_numpy()
    months = sorted(runs.filter(pool)["month_key"].unique().to_list())
    val_months = {m for i, m in enumerate(months) if i % common.VAL_MONTH_STRIDE == 3}
    is_val = runs["month_key"].is_in(list(val_months)).to_numpy()
    test = (runs["month_key"] == label).to_numpy()
    if label in common.REPLICATE_CUTS:
        test &= (runs["period"] == "replicate").to_numpy()
    last = runs.filter(pool)["power_fcst_init_time"].max()
    assert isinstance(last, datetime), "the training pool is empty"
    assert last + timedelta(hours=240) < cut, "a training target reaches the refit date"
    # Purge: a training run within 10 days before, or 7 days after, a validation month shares targets with it.
    near = np.zeros(runs.height, dtype=bool)
    origin_np = origin.dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    for m in val_months:
        start = np.datetime64(f"{m}-01", "us")
        end = (np.datetime64(f"{m}-01", "M") + np.timedelta64(1, "M")).astype("datetime64[us]")
        near |= (origin_np >= start - np.timedelta64(10, "D")) & (
            origin_np < end + np.timedelta64(7, "D")
        )
    return Split(
        train=np.flatnonzero(pool_np & ~is_val & ~near),
        val=np.flatnonzero(pool_np & is_val),
        test=np.flatnonzero(test),
    )


def pinned_capacity_points(*, label: str, runs: pl.DataFrame, index: np.ndarray) -> np.ndarray:
    """Return the pinned capacity (points) of each (run, step) of the runs `index`, shape (runs, steps)."""
    from studies.dp_features import SiteTable
    from studies.dp_pinned_capacity import pinned_capacity, read_vintages

    sites = json.loads((common.PREP_DIR / "sites.json").read_text())
    vintages = read_vintages()
    cut = month_start(label=label)
    out = np.zeros((len(index), common.STEPS))
    site_id = runs["site_id"].to_numpy()[index]
    origin = (
        runs["power_fcst_init_time"]
        .dt.replace_time_zone(None)
        .to_numpy()
        .astype("datetime64[us]")[index]
    )
    for sid, site in enumerate(common.SITES):
        info = sites[site]
        table = SiteTable(
            site=site,
            time_series_id=info["time_series_id"],
            units="MW",
            role="mw_long",
            latitude=0.0,
            longitude=0.0,
            effective_capacity_mw=info["effective_capacity_mw"],
            first_usable=cut,
        )
        pinned = pinned_capacity(vintages=vintages, table=table, cut=cut)
        rows = np.flatnonzero(site_id == sid)
        ends = (
            origin[rows, None] + (np.arange(1, common.STEPS + 1) * np.timedelta64(30, "m"))[None, :]
        )
        shim = SimpleNamespace(end_times=ends.ravel())
        out[rows] = pinned.at_rows(cache=shim, index=np.arange(ends.size)).reshape(ends.shape)  # ty: ignore[invalid-argument-type]
    return out
