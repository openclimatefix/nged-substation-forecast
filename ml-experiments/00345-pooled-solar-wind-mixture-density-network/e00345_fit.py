# ruff: noqa: D102, E501
"""Fit the pooled mixture-density networks of experiment 345 on the GPU and store their quantiles.

Every network is experiment 310's `MixtureHead` (three Gaussians on the capacity-normalised target,
no plant base) reading standardised inputs from the unified layout of `e00345_common`. Columns a
family lacks are 0 after standardisation, and `is_wind` and `temp_missing` tell the network which
rows they are. Solar jobs follow experiment 310's refit rule (train on rows whose valid time is at
least one day before the refit, predict the runs of the refit's window). Wind jobs train once on
rows before 2025-06-17, as experiment 313 did, and predict W1 to W3's scored cases with ERA5 weather
and with each of the 51 ensemble members (the members' quantiles are mixed with equal weights).

    PYTHONPATH=ml-experiments/00285-wind-effective-wind-adapters:ml-experiments/00310-metered-farm-long-lead-scoring \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00345-*/e00345_fit.py [solar|wind|ARM|own_history|zero_history ...]
"""

import multiprocessing
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00285_common as c285  # noqa: E402  # ty: ignore[unresolved-import]
import e00310_common as c310  # noqa: E402  # ty: ignore[unresolved-import]
import e00310_features as feats  # noqa: E402  # ty: ignore[unresolved-import]
import e00310_fit as f310  # noqa: E402  # ty: ignore[unresolved-import]
import e00345_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from studies.tail_calibration import level_column  # noqa: E402

N_PROCESSES: Final[int] = int(os.environ.get("E345_PROCESSES", "5"))
DEVICE: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEVELS_T: Final = torch.tensor(c310.LEVELS, dtype=torch.float64)
Piece = dict[str, np.ndarray]
MONTHLY_STARTS: Final[tuple[datetime, ...]] = tuple(
    f310.month_add(start=c310.SCORED_START, months=i) for i in range(12)
)
ONCE_START: Final[datetime] = c310.SCORED_START


# ---------------------------------------------------------------- data


@cache
def solar_base(farm: str, band: str) -> pl.DataFrame:
    """Return a farm-band's rows in the unified layout (plant-free columns, plant unit mean kept)."""
    frame = feats.feature_frame(farm=feats.load_band_farm(farm, band), scale=1.0, ceiling=1e9)
    return frame.with_columns(
        temperature=pl.col("temperature_mean"),
        hour_sin=(2 * np.pi * pl.col("hour") / 24.0).sin(),
        hour_cos=(2 * np.pi * pl.col("hour") / 24.0).cos(),
        doy_sin=(2 * np.pi * pl.col("doy") / 366.0).sin(),
        doy_cos=(2 * np.pi * pl.col("doy") / 366.0).cos(),
        temp_missing=pl.lit(0.0),
        is_wind=pl.lit(0.0),
    ).select(
        "time",
        "init_time",
        "output",
        "unit_mean",
        *{*common.SHARED, *common.SOLAR_ONLY, *common.FLAGS},
    )


@cache
def solar_params(farm: str, band: str, cutoff: datetime) -> tuple[float, float]:
    """Return the plant scale and ceiling fitted on the farm-band's rows before the cutoff."""
    return feats.scale_and_ceiling(farm=feats.load_band_farm(farm, band), cutoff=cutoff)


def test_params(*, setting: str, farm: str, band: str, cutoff: datetime) -> tuple[float, float]:
    """Return a test farm's plant scale and ceiling: its own, or the median of the other farms."""
    if setting == "own_history":
        return solar_params(farm, band, cutoff)
    others = [solar_params(s, band, cutoff) for s in c310.FARMS if s != farm]
    return float(np.median([p[0] for p in others])), float(np.median([p[1] for p in others]))


def numpy_columns(*, frame: pl.DataFrame, rows: np.ndarray | None = None) -> Piece:
    """Return the frame's unified columns as float32 arrays (all rows, or the chosen rows)."""
    wanted = [*common.SHARED, *common.SOLAR_ONLY, *common.WIND_ONLY, *common.FLAGS]
    out: Piece = {}
    for c in wanted:
        if c in frame.columns and frame[c].null_count() < frame.height:
            values = frame[c].to_numpy().astype(np.float32)
            out[c] = values if rows is None else values[rows]
    return out


def solar_piece(
    *, farm: str, band: str, cutoff: datetime, with_plant: bool, is_train: bool
) -> Piece:
    """Return one farm-band's training rows before the cutoff, with targets and a validation flag."""
    frame = solar_base(farm, band)
    time_ = frame["time"]
    rows = (time_ < cutoff).to_numpy()
    piece = numpy_columns(frame=frame, rows=rows)
    piece["y"] = frame["output"].to_numpy().astype(np.float32)[rows]
    piece["is_val"] = (time_ >= cutoff - timedelta(days=common.VALIDATION_DAYS)).to_numpy()[rows]
    if with_plant:
        scale, ceiling = solar_params(farm, band, cutoff)
        piece["mu"] = np.minimum(scale * frame["unit_mean"].to_numpy(), ceiling).astype(np.float32)[
            rows
        ]
    return piece


@cache
def wind_tables() -> dict[str, pl.DataFrame]:
    """Return the prepared wind training tables."""
    return {
        "metered": pl.read_parquet(common.PREP_DIR / "wind_train_metered.parquet"),
        "public": pl.read_parquet(common.PREP_DIR / "wind_train_public.parquet"),
    }


def wind_pieces(*, metered_farms: tuple[str, ...]) -> list[Piece]:
    """Return one piece per wind farm: the listed metered farms and every public farm."""
    tables = wind_tables()
    pieces = []
    for kind, farms in (("metered", metered_farms), ("public", None)):
        table = tables[kind]
        for label in table["farm"].unique(maintain_order=True).to_list():
            if farms is not None and label not in farms:
                continue
            frame = table.filter(pl.col("farm") == label).sort("valid_time")
            last = frame["valid_time"].max()
            assert isinstance(last, datetime)
            piece = numpy_columns(frame=frame)
            piece["y"] = frame["target"].to_numpy().astype(np.float32)
            piece["is_val"] = (
                frame["valid_time"] >= last - timedelta(days=common.VALIDATION_DAYS)
            ).to_numpy()
            pieces.append(piece)
    return pieces


def standardiser(
    *, pieces: list[Piece], columns: tuple[str, ...]
) -> dict[str, tuple[float, float]]:
    """Return each column's mean and deviation over the pieces that hold the column."""
    stats: dict[str, tuple[float, float]] = {}
    for c in columns:
        if c in common.FLAGS:
            continue
        total = count = square = 0.0
        for piece in pieces:
            if c in piece:
                v = piece[c].astype(np.float64)
                total, square, count = total + v.sum(), square + (v**2).sum(), count + v.size
        if count == 0:
            stats[c] = (0.0, 1.0)
            continue
        mean = total / count
        stats[c] = (mean, float(np.sqrt(max(square / count - mean**2, 0.0))) + 1e-6)
    return stats


def matrix(*, piece: Piece, columns: tuple[str, ...], stats: dict) -> np.ndarray:
    """Return the piece's standardised inputs; a column the piece lacks is 0."""
    n = next(len(v) for k, v in piece.items() if k not in ("y", "is_val"))
    out = np.zeros((n, len(columns)), dtype=np.float32)
    for j, c in enumerate(columns):
        if c in common.FLAGS:
            if c in piece:
                out[:, j] = piece[c]
            elif c == "temp_missing":
                out[:, j] = 1.0
        elif c in piece:
            mean, std = stats[c]
            out[:, j] = (piece[c] - mean) / std
    return out


# ---------------------------------------------------------------- network


def train_network(
    *, pieces: list[Piece], arm: common.Arm, seed: int
) -> tuple[f310.MixtureHead, dict]:
    """Fit one network on the pieces and return it with the standardiser."""
    columns = arm.columns
    stats = standardiser(pieces=pieces, columns=columns)
    x = np.concatenate([matrix(piece=p, columns=columns, stats=stats) for p in pieces])
    y = np.concatenate([p["y"] for p in pieces])
    is_val = np.concatenate([p["is_val"] for p in pieces])
    wind = np.concatenate(
        [np.full(len(p["y"]), p.get("is_wind", np.zeros(1))[0] == 1.0) for p in pieces]
    )
    rng = np.random.default_rng(1000 + seed)
    if arm.shuffle_wind_target:
        index = np.flatnonzero(wind)
        y[index] = y[rng.permutation(index)]
    if arm.shuffle_fuel and "is_wind" in columns:
        j = columns.index("is_wind")
        x[:, j] = x[rng.permutation(len(x)), j]
    torch.manual_seed(seed)
    xt = torch.tensor(x, device=DEVICE)
    del x
    yt = torch.tensor(y, device=DEVICE)
    fit_rows = torch.tensor(np.flatnonzero(~is_val), device=DEVICE)
    val_rows = torch.tensor(np.flatnonzero(is_val), device=DEVICE)
    zero = torch.zeros(len(y), device=DEVICE)
    model = f310.MixtureHead(len(columns)).to(DEVICE)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=common.LEARNING_RATE, weight_decay=common.WEIGHT_DECAY
    )
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    best, best_state = float("inf"), None
    for _ in range(common.EPOCHS):
        model.train()
        order = fit_rows[torch.randperm(len(fit_rows), generator=generator, device=DEVICE)]
        for start in range(0, len(order), common.BATCH):
            idx = order[start : start + common.BATCH]
            loss = model.nll({"x": xt[idx], "mu": zero[idx], "actual": yt[idx]})
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(
                model.nll({"x": xt[val_rows], "mu": zero[val_rows], "actual": yt[val_rows]})
            )
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    del xt, yt
    return model, stats


@torch.no_grad()
def quantiles(
    *, model: f310.MixtureHead, x: np.ndarray, ceiling: float, chunk: int = 300_000
) -> np.ndarray:
    """Return the mixture's 11 quantiles per row by bisection on its CDF, clamped to [0, ceiling]."""
    levels = LEVELS_T.to(DEVICE)
    out = np.empty((len(x), levels.numel()), dtype=np.float32)
    for start in range(0, len(x), chunk):
        xb = torch.tensor(x[start : start + chunk], device=DEVICE)
        log_w, mean, std = model.parameters_of(xb)
        w, mean, std = log_w.exp().double(), mean.double(), std.double()
        lo = torch.full((len(xb), levels.numel()), -2.0, dtype=torch.float64, device=DEVICE)
        hi = torch.full_like(lo, 2.0)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            z = (mid[:, :, None] - mean[:, None, :]) / std[:, None, :]
            cdf = (w[:, None, :] * torch.special.ndtr(z)).sum(dim=2)
            below = cdf < levels[None, :]
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        out[start : start + chunk] = (0.5 * (lo + hi)).clamp(0.0, ceiling).cpu().numpy()
    return out


# ---------------------------------------------------------------- jobs


@dataclass(frozen=True)
class SolarJob:
    """One solar refit: an arm, a history setting, a held-out farm (own history: none), a window."""

    arm: str
    setting: str
    held: str
    start: datetime
    end: datetime

    @property
    def cutoff(self) -> datetime:
        return self.start - timedelta(days=f310.REFIT_GAP_DAYS)

    @property
    def seeds(self) -> tuple[int, ...]:
        if self.arm in common.OWN_ONLY_SOLAR:
            return common.FEW_SEEDS
        return common.OWN_SEEDS if self.setting == "own_history" else common.ZERO_SEEDS

    @property
    def test_farms(self) -> tuple[str, ...]:
        return c310.FARMS if self.setting == "own_history" else (self.held,)


@dataclass(frozen=True)
class WindJob:
    """One wind network trained once; predicts W1 to W3 for each seed."""

    arm: str
    setting: str

    @property
    def seeds(self) -> tuple[int, ...]:
        return common.OWN_SEEDS


def solar_out(*, job: SolarJob, band: str, farm: str, seed: int) -> Path:
    """Return a solar prediction file."""
    return (
        common.PRED_DIR / band / job.setting / f"{job.arm}_s{seed}_{farm}_{job.start:%Y-%m}.parquet"
    )


def wind_out(*, arm: str, setting: str, farm: str, seed: int, weather: str) -> Path:
    """Return a wind prediction file."""
    return common.WIND_DIR / f"{arm}__{setting}__{farm}__s{seed}__{weather}.npy"


def write_frame(*, path: Path, frame: pl.DataFrame) -> None:
    """Write a parquet file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def write_array(*, path: Path, array: np.ndarray) -> None:
    """Write an array atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array.astype(np.float32))
    tmp.replace(path)


def solar_job_pieces(*, job: SolarJob, arm: common.Arm) -> list[Piece]:
    """Return a solar job's training pieces."""
    pieces: list[Piece] = []
    if arm.solar_rows:
        train_farms = [s for s in c310.FARMS if job.setting == "own_history" or s != job.held]
        pieces += [
            solar_piece(
                farm=s, band=b, cutoff=job.cutoff, with_plant="mu" in arm.columns, is_train=True
            )
            for s in train_farms
            for b in c310.BAND_ORDER
        ]
    if arm.wind_rows:
        pieces += wind_pieces(metered_farms=common.WIND_FARMS)
    return pieces


def test_x(
    *, job: SolarJob, arm: common.Arm, farm: str, band: str, stats: dict, seed: int
) -> tuple[pl.DataFrame, np.ndarray, float]:
    """Return a test farm-band's scored rows, their inputs, and the output ceiling."""
    frame = solar_base(farm, band)
    keep = (
        ((frame["init_time"] >= job.start) & (frame["init_time"] < job.end))
        & (frame["time"] < c310.SCORED_END)
    ).to_numpy()
    rows = frame.filter(pl.Series(keep))
    scale, ceiling = test_params(setting=job.setting, farm=farm, band=band, cutoff=job.cutoff)
    piece = numpy_columns(frame=rows)
    if "mu" in arm.columns:
        piece["mu"] = np.minimum(scale * rows["unit_mean"].to_numpy(), ceiling).astype(np.float32)
    piece["y"] = np.zeros(rows.height, dtype=np.float32)
    x = matrix(piece=piece, columns=arm.columns, stats=stats)
    if arm.shuffle_fuel and "is_wind" in arm.columns:
        share = float(stats.get("_wind_share", (0.5, 0))[0])
        rng = np.random.default_rng(777 + seed)
        x[:, arm.columns.index("is_wind")] = (rng.random(rows.height) < share).astype(np.float32)
    return rows, x, ceiling


def run_solar(job: SolarJob) -> str:
    """Fit every seed of one solar job and write its scored rows."""
    arm = common.ARMS[job.arm]
    started = time.time()
    todo = [
        s
        for s in job.seeds
        if not all(
            solar_out(job=job, band=b, farm=f, seed=s).exists()
            for b in c310.BAND_ORDER
            for f in job.test_farms
        )
    ]
    if not todo:
        return f"solar {job.arm} {job.setting} {job.held or '-'} {job.start:%Y-%m} skipped"
    pieces = solar_job_pieces(job=job, arm=arm)
    wind_share = sum(len(p["y"]) for p in pieces if p.get("is_wind", np.zeros(1))[0] == 1.0) / max(
        1, sum(len(p["y"]) for p in pieces)
    )
    for seed in todo:
        model, stats = train_network(pieces=pieces, arm=arm, seed=seed)
        stats = {**stats, "_wind_share": (wind_share, 0.0)}
        for farm in job.test_farms:
            for band in c310.BAND_ORDER:
                rows, x, ceiling = test_x(
                    job=job, arm=arm, farm=farm, band=band, stats=stats, seed=seed
                )
                q = quantiles(model=model, x=x, ceiling=ceiling)
                frame = rows.select("time", "init_time", "lead_hours").with_columns(
                    time_series_id=pl.lit(farm),
                    arm=pl.lit(job.arm),
                    seed=pl.lit(seed),
                    setting=pl.lit(job.setting),
                    band=pl.lit(band),
                    actual=rows["output"].cast(pl.Float64),
                    **{
                        level_column(level=lv): pl.Series(q[:, i])
                        for i, lv in enumerate(c310.LEVELS)
                    },
                )
                write_frame(path=solar_out(job=job, band=band, farm=farm, seed=seed), frame=frame)
    return (
        f"solar {job.arm} {job.setting} {job.held or '-'} {job.start:%Y-%m} "
        f"{len(todo)} seeds {time.time() - started:.0f}s"
    )


@cache
def wind_eval(farm: str, weather: str) -> Piece:
    """Return a wind farm's evaluation inputs (cases with ERA5, or their ensemble members)."""
    frame = pl.read_parquet(common.PREP_DIR / f"wind_eval_{farm}_{weather}.parquet")
    return numpy_columns(frame=frame)


def run_wind(job: WindJob) -> str:
    """Fit every seed of one wind network and write W1 to W3's mixtures."""
    arm = common.ARMS[job.arm]
    started = time.time()
    todo = [
        s
        for s in job.seeds
        if not all(
            wind_out(arm=job.arm, setting=job.setting, farm=f, seed=s, weather=w).exists()
            for f in common.WIND_FARMS
            for w in ("era5", "ens")
        )
    ]
    if not todo:
        return f"wind {job.arm} {job.setting} skipped"
    pieces: list[Piece] = []
    if arm.solar_rows:
        pieces += [
            solar_piece(
                farm=s, band=b, cutoff=common.WIND_TRAIN_END, with_plant=False, is_train=True
            )
            for s in c310.FARMS
            for b in c310.BAND_ORDER
        ]
    pieces += wind_pieces(metered_farms=common.WIND_FARMS if job.setting == "own_history" else ())
    wind_share = sum(len(p["y"]) for p in pieces if p.get("is_wind", np.zeros(1))[0] == 1.0) / sum(
        len(p["y"]) for p in pieces
    )
    for seed in todo:
        model, stats = train_network(pieces=pieces, arm=arm, seed=seed)
        for farm in common.WIND_FARMS:
            for weather in ("era5", "ens"):
                piece = dict(wind_eval(farm, weather))
                if job.setting == "zero_history":
                    # No wind row the network trained on has a temperature, so hide it at test too.
                    piece.pop("temperature", None)
                    piece.pop("temp_missing", None)
                piece["y"] = np.zeros(len(piece["wind10"]), dtype=np.float32)
                x = matrix(piece=piece, columns=arm.columns, stats=stats)
                if arm.shuffle_fuel and "is_wind" in arm.columns:
                    rng = np.random.default_rng(777 + seed)
                    x[:, arm.columns.index("is_wind")] = (rng.random(len(x)) < wind_share).astype(
                        np.float32
                    )
                q = quantiles(model=model, x=x, ceiling=1.0)
                if weather == "ens":
                    q = c285.mixture_quantiles(member_quantiles=q.reshape(-1, c285.MEMBERS, 11))
                write_array(
                    path=wind_out(
                        arm=job.arm, setting=job.setting, farm=farm, seed=seed, weather=weather
                    ),
                    array=q,
                )
    return f"wind {job.arm} {job.setting} {len(todo)} seeds {time.time() - started:.0f}s"


def run_any(job: SolarJob | WindJob) -> str:
    """Run a job; a crash returns its traceback instead of killing the pool."""
    torch.set_num_threads(1)
    try:
        return run_solar(job) if isinstance(job, SolarJob) else run_wind(job)
    except Exception:  # noqa: BLE001
        return "FAILED " + repr(job) + "\n" + traceback.format_exc()


def all_jobs() -> list[SolarJob | WindJob]:
    """Return every job, the biggest first."""
    jobs: list[SolarJob | WindJob] = [
        WindJob(arm=a, setting=s)
        for a in common.WIND_ARM_NAMES
        for s in ("zero_history", "own_history")
        if not (a in common.CONTROL_ARMS and s == "zero_history")
    ]
    cadence = {
        "both_once": ((ONCE_START, f310.month_add(start=ONCE_START, months=12)),),
        "both_monthly": tuple((s, f310.month_add(start=s, months=1)) for s in MONTHLY_STARTS),
    }
    quarterly = tuple(
        (s, f310.month_add(start=s, months=f310.REFIT_MONTHS)) for s in f310.REFIT_STARTS
    )
    for arm in common.SOLAR_ARM_NAMES:
        for setting in ("own_history", "zero_history"):
            if setting == "zero_history" and arm in common.OWN_ONLY_SOLAR:
                continue
            held_options = ("",) if setting == "own_history" else c310.FARMS
            for held in held_options:
                for start, end in cadence.get(arm, quarterly):
                    jobs.append(SolarJob(arm=arm, setting=setting, held=held, start=start, end=end))
    return jobs


def main() -> None:
    """Run all jobs (or those matching the arguments) in a pool of spawned processes."""
    only = set(sys.argv[1:])
    jobs = all_jobs()
    if only:
        jobs = [
            j
            for j in jobs
            if (
                ("solar" in only and isinstance(j, SolarJob))
                or ("wind" in only and isinstance(j, WindJob))
                or j.arm in only
                or j.setting in only
            )
        ]
    failures = 0
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES, maxtasksperchild=40) as pool:
        for done in pool.imap_unordered(run_any, jobs):
            failures += done.startswith("FAILED")
            print("finished", done, flush=True)
    print(f"all {len(jobs)} jobs done, {failures} failed", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
