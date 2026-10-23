"""Part B of experiment 00303: correct each ensemble member's wind speed, then forecast.

For every farm this fits the physics core and XGBoost on ERA5 before `common.BASE_END` (as
experiment 00259 did), builds four versions of the ensemble members' wind speed (raw, B1 quantile
mapping by lead day, B2 mapping by lead day and direction sector, B3 a small neural mapping), runs
each model on each version of the hindcast members (2024-04-01 to 2025-06-16) and the scored
members (the 53 Tuesday runs), mixes the 51 members with equal weights, and stores the mixtures.

Output: `work/ens_mix/<model>_<arm>_s<seed>_<set>_<farm>.npy`, shape (cases, 11), where the model
is `core` or `xgb`, the arm is `B0` to `B3`, and the set is `hindcast` or `scored`. The spread
factor and the offset table are applied by `e00303_ens_final.py`.

    systemd-run --user --scope -p MemoryMax=14G -p OOMPolicy=continue \
      uv run --no-sync python e00303_ens.py [--farms W1,W2,W3]
"""

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
os.environ.setdefault("E00303_TORCH_THREADS", "2")

import e00303_common as common  # noqa: E402
import e00303_core as core_module  # noqa: E402
import e00303_models as models  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402  # ty: ignore[unresolved-import]
import xgboost as xgb  # noqa: E402
from torch import nn  # noqa: E402  # ty: ignore[unresolved-import]

OUT_DIR: Final[Path] = common.WORK_DIR / "ens_mix"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
CASE_CHUNK: Final[int] = 20_000
KNOTS: Final[np.ndarray] = np.linspace(0.0, 1.0, 101)
MIN_GROUP_ROWS: Final[int] = 20_000
SECTOR_COUNT: Final[int] = 8
MAX_LEAD_DAY: Final[int] = 15
VALIDATION_DAYS: Final[int] = 60
TRAIN_EVERY: Final[int] = 3
EPOCHS: Final[int] = 4
BATCH: Final[int] = 262_144
DEVICE: Final[str] = "cuda" if torch.cuda.is_available() else "cpu"


def hindcast_rows(*, series: int) -> pl.DataFrame:
    """Return one farm's hindcast member rows, sorted by case and member."""
    paths = sorted(common.HINDCAST_DIR.glob("init=*.parquet"))
    return (
        pl.concat([pl.scan_parquet(p).filter(pl.col("time_series_id") == series) for p in paths])
        .sort(*common.KEY_COLUMNS, "ensemble_member")
        .collect()
    )


def check_alignment(*, members: pl.DataFrame, cases: pl.DataFrame, farm: str) -> None:
    """Raise unless `members` holds members 0 to 50 of each case in `cases`, in order."""
    if members.height != cases.height * common.MEMBERS or not (
        members.gather_every(common.MEMBERS)
        .select(common.KEY_COLUMNS)
        .equals(cases.select(common.KEY_COLUMNS))
    ):
        msg = f"{farm}: the members do not line up with the cases"
        raise ValueError(msg)


def add_member_columns(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Add `lead_day`, `sector` (of the member's own 100 m direction), and the hour of day."""
    return models.add_derived_columns(frame=rows).with_columns(
        lead_day=common.lead_day(
            power_fcst_init_time=pl.col("power_fcst_init_time"), valid_time=pl.col("valid_time")
        )
        .clip(0, MAX_LEAD_DAY)
        .cast(pl.Int32),
        sector=(
            (
                (
                    pl.arctan2(pl.col("wind_dir_sin_100m"), pl.col("wind_dir_cos_100m")).degrees()
                    % 360
                )
                + 180 / SECTOR_COUNT
            )
            // (360 / SECTOR_COUNT)
        ).cast(pl.Int32)
        % SECTOR_COUNT,
    )


def with_speed(*, rows: pl.DataFrame, speed_100m: np.ndarray) -> pl.DataFrame:
    """Return `rows` with a new 100 m speed and the 10 m speed scaled by the same ratio."""
    old = rows["wind_speed_100m"].to_numpy()
    ratio = np.where(old > 0.1, speed_100m / np.maximum(old, 0.1), 1.0)
    return rows.with_columns(
        wind_speed_100m=pl.Series(speed_100m.astype(np.float32)),
        wind_speed_10m=pl.Series((rows["wind_speed_10m"].to_numpy() * ratio).astype(np.float32)),
    )


def _map_group(
    *, speed: np.ndarray, member_knots: np.ndarray, era5_knots: np.ndarray
) -> np.ndarray:
    """Return `speed` mapped through the knots, shifting beyond the ends instead of clamping."""
    mapped = np.interp(speed, member_knots, era5_knots)
    mapped = np.where(speed > member_knots[-1], speed + (era5_knots[-1] - member_knots[-1]), mapped)
    return np.maximum(mapped, 0.0)


class QuantileMapping:
    """Per-group empirical quantile mapping of the members' 100 m speed to ERA5's."""

    def __init__(self, *, rows: pl.DataFrame, era5_speed: np.ndarray, by_sector: bool) -> None:
        """Fit the mapping on hindcast member rows and the ERA5 100 m speed at the same rows."""
        self.by_sector = by_sector
        speed = rows["wind_speed_100m"].to_numpy()
        lead = rows["lead_day"].to_numpy()
        sector = rows["sector"].to_numpy()
        self.overall: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.table: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for day in np.unique(lead):
            in_day = lead == day
            self.overall[int(day)] = (
                np.maximum.accumulate(np.quantile(speed[in_day], KNOTS)),
                np.maximum.accumulate(np.quantile(era5_speed[in_day], KNOTS)),
            )
            if by_sector:
                for s in range(SECTOR_COUNT):
                    chosen = in_day & (sector == s)
                    if chosen.sum() >= MIN_GROUP_ROWS:
                        self.table[(int(day), s)] = (
                            np.maximum.accumulate(np.quantile(speed[chosen], KNOTS)),
                            np.maximum.accumulate(np.quantile(era5_speed[chosen], KNOTS)),
                        )

    def __call__(self, rows: pl.DataFrame) -> np.ndarray:
        """Return the mapped 100 m speed of `rows`."""
        speed = rows["wind_speed_100m"].to_numpy()
        lead = rows["lead_day"].to_numpy()
        sector = rows["sector"].to_numpy()
        out = np.empty_like(speed, dtype=np.float64)
        for day in np.unique(lead):
            in_day = lead == day
            days = self.overall.get(int(day)) or self.overall[max(self.overall)]
            if self.by_sector:
                done = np.zeros(speed.size, dtype=bool)
                for s in range(SECTOR_COUNT):
                    chosen = in_day & (sector == s)
                    if (int(day), s) in self.table:
                        out[chosen] = _map_group(
                            speed=speed[chosen],
                            member_knots=self.table[(int(day), s)][0],
                            era5_knots=self.table[(int(day), s)][1],
                        )
                        done |= chosen
                rest = in_day & ~done
            else:
                rest = in_day
            out[rest] = _map_group(speed=speed[rest], member_knots=days[0], era5_knots=days[1])
        return out


def network_inputs(*, rows: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the neural mapping's inputs (rows, 11) and the member's rank level in its case."""
    speed = rows["wind_speed_100m"].to_numpy().astype(np.float32)
    cases = speed.reshape(-1, common.MEMBERS)
    spread = np.repeat(cases.std(axis=1), common.MEMBERS)
    rank = cases.argsort(axis=1).argsort(axis=1).reshape(-1)
    level = ((rank + 0.5) / common.MEMBERS).astype(np.float32)
    ten = rows["wind_speed_10m"].to_numpy().astype(np.float32)
    ratio = np.where(ten > 0.5, np.clip(ten / np.maximum(speed, 0.5), 0.05, 1.5), 0.5)
    angle = rows["hour"].to_numpy().astype(np.float32) * (2 * np.pi / 24.0)
    inputs = np.stack(
        [
            speed / 15.0,
            ratio,
            rows["wind_dir_sin_100m"].to_numpy(),
            rows["wind_dir_cos_100m"].to_numpy(),
            (rows["temperature_c"].to_numpy() - 10.0) / 10.0,
            np.sin(angle),
            np.cos(angle),
            rows["lead_day"].to_numpy() / float(MAX_LEAD_DAY),
            spread / 5.0,
            level,
            cases.mean(axis=1).repeat(common.MEMBERS) / 15.0,
        ],
        axis=1,
    ).astype(np.float32)
    return inputs, level


class NeuralMapping(nn.Module):
    """Two layers of 32 units giving a speed scale (0.8 to 1.25) and an offset (within 1 m/s)."""

    def __init__(self) -> None:
        """Start at the identity mapping."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2)
        )
        last = self.net[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(
                torch.tensor(
                    [core_module._inverse_sigmoid_range(1.0, *core_module.SCALE_RANGE), 0.0]
                )
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the mapped 100 m speed; input column 0 is the speed divided by 15."""
        raw = self.net(inputs)
        scale = core_module._sigmoid_range(raw[:, 0], *core_module.SCALE_RANGE)
        return (inputs[:, 0] * 15.0 * scale + torch.tanh(raw[:, 1])).clamp(min=0.0)


def pinball(*, predicted: torch.Tensor, actual: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss with a per-row level."""
    difference = actual - predicted
    return torch.maximum(level * difference, (level - 1) * difference).mean()


def fit_neural_mapping(
    *, rows: pl.DataFrame, era5_speed: np.ndarray, seed: int
) -> tuple[NeuralMapping, float]:
    """Train the mapping so each member is the quantile of ERA5 speed its rank implies."""
    torch.manual_seed(seed)
    inputs, level = network_inputs(rows=rows)
    target = era5_speed.astype(np.float32)
    cases = rows.height // common.MEMBERS
    case_index = np.arange(cases)
    valid_time = rows["valid_time"].to_numpy()[:: common.MEMBERS]
    cutoff = valid_time.max() - np.timedelta64(VALIDATION_DAYS, "D")
    train_cases = case_index[
        (valid_time < cutoff) & (case_index % TRAIN_EVERY == seed % TRAIN_EVERY)
    ]
    validation_cases = case_index[valid_time >= cutoff][::2]

    def rows_of(chosen: np.ndarray) -> np.ndarray:
        return (chosen[:, None] * common.MEMBERS + np.arange(common.MEMBERS)).reshape(-1)

    def tensors(chosen: np.ndarray) -> tuple[torch.Tensor, ...]:
        index = rows_of(chosen)
        return (
            torch.as_tensor(inputs[index], device=DEVICE),
            torch.as_tensor(target[index], device=DEVICE),
            torch.as_tensor(level[index], device=DEVICE),
        )

    train_x, train_y, train_level = tensors(train_cases)
    validation_x, validation_y, validation_level = tensors(validation_cases)
    network = NeuralMapping().to(DEVICE)
    optimiser = torch.optim.Adam(network.parameters(), lr=3e-3)
    steps = EPOCHS * (train_x.shape[0] // BATCH + 1)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    best, best_state = np.inf, None
    for _ in range(EPOCHS):
        order = torch.randperm(train_x.shape[0], device=DEVICE)
        for start in range(0, order.numel(), BATCH):
            chosen = order[start : start + BATCH]
            optimiser.zero_grad()
            loss = pinball(
                predicted=network(train_x[chosen]),
                actual=train_y[chosen],
                level=train_level[chosen],
            )
            loss.backward()
            optimiser.step()
            schedule.step()
        with torch.no_grad():
            held = float(
                pinball(
                    predicted=network(validation_x), actual=validation_y, level=validation_level
                )
            )
        if held < best:
            best = held
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
    if best_state is None:
        raise ValueError
    network.load_state_dict(best_state)
    return network.eval(), best


def apply_neural(*, network: NeuralMapping, rows: pl.DataFrame) -> np.ndarray:
    """Return the neural mapping's 100 m speed for `rows`."""
    inputs, _ = network_inputs(rows=rows)
    out = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], BATCH):
            chunk = torch.as_tensor(inputs[start : start + BATCH], device=DEVICE)
            out.append(network(chunk).cpu().numpy())
    return np.concatenate(out)


def mixtures(*, rows: pl.DataFrame, predict: Callable[[pl.DataFrame], np.ndarray]) -> np.ndarray:
    """Return the equal-weight member mixture of `predict(chunk)` for every case in `rows`."""
    pieces = []
    step = CASE_CHUNK * common.MEMBERS
    for start in range(0, rows.height, step):
        chunk = rows.slice(start, step)
        pieces.append(
            common.mixture_quantiles(
                member_quantiles=predict(chunk).reshape(-1, common.MEMBERS, 11)
            )
        )
    return np.concatenate(pieces).astype(np.float32)


def write_atomic(*, path: Path, array: np.ndarray) -> None:
    """Write `array` to `path` through a temporary file."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.rename(path)


def run_farm(*, farm: str) -> None:  # noqa: PLR0915
    """Fit and store every Part B mixture for one farm."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = common.site_table().filter(pl.col("site") == farm)
    series = int(sites["time_series_id"][0])
    era5 = common.era5_rows(sites=sites)
    train = models.add_derived_columns(frame=common.training_table(sites=sites, era5=era5)).filter(
        pl.col("valid_time") < common.BASE_END
    )
    target = train["target"].to_numpy()
    # base models
    weather = models.weather_arrays(frame=train, options=core_module.DEFAULT_OPTIONS)
    core = core_module.fit_core(
        weather=weather, target=target, with_idle_layer=farm != "W2", seed=0
    )
    residuals = core_module.ResidualQuantiles.fit(
        expected=core_module.predict_expected_power(core=core, weather=weather), observed=target
    )
    centres, curve = models.binned_curve(speed=train["wind_speed_100m"].to_numpy(), target=target)
    boosters = {}
    for seed in SEEDS:
        boosters[seed] = xgb.train(
            {**models.XGBOOST_PARAMETERS, "seed": seed},
            xgb.DMatrix(
                train.select(common.XGBOOST_FEATURES).to_numpy(),
                label=target,
                base_margin=np.tile(
                    np.interp(train["wind_speed_100m"].to_numpy(), centres, curve)[:, None], (1, 11)
                ),
            ),
            num_boost_round=models.N_ROUNDS,
        )

    def predict_core(chunk: pl.DataFrame) -> np.ndarray:
        weather_chunk = models.weather_arrays(frame=chunk, options=core_module.DEFAULT_OPTIONS)
        return residuals.quantiles(
            expected=core_module.predict_expected_power(core=core, weather=weather_chunk)
        )

    def predict_xgb(chunk: pl.DataFrame, seed: int) -> np.ndarray:
        margin = np.tile(
            np.interp(chunk["wind_speed_100m"].to_numpy(), centres, curve)[:, None], (1, 11)
        )
        predicted = boosters[seed].predict(
            xgb.DMatrix(chunk.select(common.XGBOOST_FEATURES).to_numpy(), base_margin=margin)
        )
        return np.sort(np.clip(predicted, 0.0, 1.0), axis=1).astype(np.float32)

    # member sets
    cases = common.cases_table(sites=sites)
    scored = common.ens_member_rows(sites=sites).join(
        cases.select(*common.KEY_COLUMNS), on=list(common.KEY_COLUMNS), how="semi"
    )
    check_alignment(members=scored, cases=cases, farm=farm)
    hind = hindcast_rows(series=series)
    hind_cases = hind.gather_every(common.MEMBERS).select(common.KEY_COLUMNS)
    check_alignment(members=hind, cases=hind_cases, farm=farm)
    scored = add_member_columns(rows=scored)
    hind = add_member_columns(rows=hind)
    era5_speed = hind.join(
        era5.select("time_series_id", "valid_time", era5_speed=pl.col("wind_speed_100m")),
        on=["time_series_id", "valid_time"],
        how="left",
        maintain_order="left",
    )["era5_speed"].to_numpy()
    if np.isnan(era5_speed).any():
        msg = f"{farm}: hindcast cases without ERA5"
        raise ValueError(msg)
    corrections: dict[str, dict] = {}
    b1 = QuantileMapping(rows=hind, era5_speed=era5_speed, by_sector=False)
    b2 = QuantileMapping(rows=hind, era5_speed=era5_speed, by_sector=True)
    facts: dict = {
        "hindcast_cases": hind_cases.height,
        "raw_mean_speed": float(hind["wind_speed_100m"].to_numpy().mean()),
        "era5_mean_speed": float(era5_speed.mean()),
    }
    corrections["B0"] = {s: (lambda r: r["wind_speed_100m"].to_numpy()) for s in SEEDS}
    corrections["B1"] = dict.fromkeys(SEEDS, b1)
    corrections["B2"] = dict.fromkeys(SEEDS, b2)
    networks = {}
    for seed in SEEDS:
        networks[seed], held_out = fit_neural_mapping(rows=hind, era5_speed=era5_speed, seed=seed)
        facts[f"b3_validation_pinball_seed{seed}"] = held_out
    corrections["B3"] = {
        s: (lambda r, n=networks[s]: apply_neural(network=n, rows=r)) for s in SEEDS
    }
    for arm, by_seed in corrections.items():
        for seed in SEEDS:
            core_seed = seed if arm == "B3" else 0
            for set_name, rows in (("hindcast", hind), ("scored", scored)):
                core_path = OUT_DIR / f"core_{arm}_s{core_seed}_{set_name}_{farm}.npy"
                xgb_path = OUT_DIR / f"xgb_{arm}_s{seed}_{set_name}_{farm}.npy"
                if core_path.exists() and xgb_path.exists():
                    continue
                corrected = with_speed(rows=rows, speed_100m=by_seed[seed](rows))
                if not core_path.exists():
                    write_atomic(
                        path=core_path, array=mixtures(rows=corrected, predict=predict_core)
                    )
                if not xgb_path.exists():
                    write_atomic(
                        path=xgb_path,
                        array=mixtures(rows=corrected, predict=lambda c, s=seed: predict_xgb(c, s)),
                    )
                print(f"{farm} {arm} seed {seed} {set_name} done", flush=True)
    (common.WORK_DIR / f"ens_facts_{farm}.json").write_text(json.dumps(facts, indent=1))


def main() -> None:
    """Run the requested farms one after another."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default="W1,W2,W3")
    for farm in parser.parse_args().farms.split(","):
        run_farm(farm=farm)


if __name__ == "__main__":
    main()
