"""Fit experiment 00303's ensemble arms for one farm and store their member mixtures.

Arms stored in `work/mix/<arm>_s<seed>_<hindcast|scored>_<farm>.npy`, shape (cases, 11), the
equal-weight mixture of the 51 members' quantiles, before the spread factor:

- `P2`: the ERA5-trained adapter A4 (experiment 00285), run on every member.
- `P8`: XGBoost with the A4 effective hub-height speed as a seventh column, run on every member.
- `P4`: core and adapter fitted on the ensemble hindcast alone (adapter also sees lead and spread).
- `P5`: P2 fine-tuned on the ensemble hindcast.
- `P7`: the blind adapter (100 m speed only) pre-trained on ERA5, fine-tuned like P5.

P4, P5, and P7 are cross-fitted: the hindcast mixture of each comes from models fitted without the
held-out third of the hindcast runs, so the spread factor and the residual quantiles never see an
in-sample prediction. The scored mixture comes from a model fitted on the whole hindcast.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python e00303_fit.py --farms W1 [--arms P2,P8,P4,P5,P7]
"""

import argparse
import os
import time
from datetime import timedelta
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
os.environ.setdefault("E00303_TORCH_THREADS", "2")

import e00303_common as common  # noqa: E402
import e00303_core as core_module  # noqa: E402
import e00303_ens as ens  # noqa: E402
import e00303_models as models  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402  # ty: ignore[unresolved-import]
import xgboost as xgb  # noqa: E402

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
DEVICE: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDS: Final[int] = 3
MEMBERS_PER_CASE: Final[int] = 10
"""Members drawn per hindcast case for a fit; each draw is seeded."""
VALIDATION_DAYS: Final[int] = 90
SCRATCH_STEPS: Final[int] = 1_500
FINE_TUNE_STEPS: Final[int] = 300
FINE_TUNE_RATE_FACTOR: Final[float] = 0.1
EVAL_EVERY: Final[int] = 25
PREDICT_CHUNK: Final[int] = 2_000_000
CASE_CHUNK: Final[int] = 20_000
ARMS: Final[tuple[str, ...]] = ("P2", "P8", "P4", "P5", "P7")


def member_columns(*, rows: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the weather columns of member rows as float32 arrays, with lead and spread."""
    speed = rows["wind_speed_100m"].to_numpy().astype(np.float32)
    spread = np.repeat(speed.reshape(-1, common.MEMBERS).std(axis=1), common.MEMBERS)
    return {
        "speed_10m": rows["wind_speed_10m"].to_numpy().astype(np.float32),
        "speed_100m": speed,
        "temperature_c": rows["temperature_c"].to_numpy().astype(np.float32),
        "dir_sin_10m": rows["wind_dir_sin_10m"].to_numpy().astype(np.float32),
        "dir_cos_10m": rows["wind_dir_cos_10m"].to_numpy().astype(np.float32),
        "direction_sin": rows["wind_dir_sin_100m"].to_numpy().astype(np.float32),
        "direction_cos": rows["wind_dir_cos_100m"].to_numpy().astype(np.float32),
        "hour": rows["hour"].to_numpy().astype(np.float32),
        "lead_day": (rows["lead_day"].to_numpy() / float(ens.MAX_LEAD_DAY)).astype(np.float32),
        "spread": (spread / 5.0).astype(np.float32),
    }


def weather_on_device(
    *, columns: dict[str, np.ndarray], index: slice | np.ndarray
) -> core_module.WeatherArrays:
    """Return the weather arrays of `index` as tensors on the device."""

    def tensor(name: str) -> torch.Tensor:
        return torch.as_tensor(columns[name][index], dtype=torch.float32, device=DEVICE)

    return core_module.WeatherArrays(
        speed_10m=tensor("speed_10m"),
        speed_100m=tensor("speed_100m"),
        temperature_c=tensor("temperature_c"),
        direction_sin=tensor("direction_sin"),
        direction_cos=tensor("direction_cos"),
        hour=tensor("hour"),
        lead_day=tensor("lead_day"),
        spread=tensor("spread"),
    )


def predict_columns(
    *, core: core_module.WindCore, columns: dict[str, np.ndarray], speed: bool = False
) -> np.ndarray:
    """Return the core's expected power (or hub-height speed) for every row of `columns`."""
    core = core.to(DEVICE).eval()
    size = columns["speed_100m"].size
    pieces = []
    with torch.no_grad():
        for start in range(0, size, PREDICT_CHUNK):
            weather = weather_on_device(columns=columns, index=slice(start, start + PREDICT_CHUNK))
            out = core.hub_height_speed(weather=weather) if speed else core(weather=weather)
            pieces.append(out.cpu().numpy())
    return np.concatenate(pieces).astype(np.float32)


def mixture_of_expected(
    *, expected: np.ndarray, residuals: core_module.ResidualQuantiles
) -> np.ndarray:
    """Return the equal-weight member mixtures (cases, 11) of expected power plus residual offsets."""
    pieces = []
    step = CASE_CHUNK * common.MEMBERS
    for start in range(0, expected.size, step):
        member = residuals.quantiles(expected=expected[start : start + step])
        pieces.append(
            common.mixture_quantiles(member_quantiles=member.reshape(-1, common.MEMBERS, 11))
        )
    return np.concatenate(pieces).astype(np.float32)


def mixture_of_member_quantiles(*, member_quantiles: np.ndarray) -> np.ndarray:
    """Return the equal-weight member mixtures (cases, 11) of (rows, 11) member quantiles."""
    pieces = []
    step = CASE_CHUNK * common.MEMBERS
    for start in range(0, member_quantiles.shape[0], step):
        chunk = member_quantiles[start : start + step]
        pieces.append(
            common.mixture_quantiles(member_quantiles=chunk.reshape(-1, common.MEMBERS, 11))
        )
    return np.concatenate(pieces).astype(np.float32)


def store(*, arm: str, seed: int, set_name: str, farm: str, mixture: np.ndarray) -> None:
    """Write one mixture atomically."""
    common.MIX_DIR.mkdir(parents=True, exist_ok=True)
    path = common.MIX_DIR / f"{arm}_s{seed}_{set_name}_{farm}.npy"
    ens.write_atomic(path=path, array=mixture)


def done(*, arm: str, farm: str) -> bool:
    """Return whether every mixture of `arm` for `farm` is stored."""
    return all(
        (common.MIX_DIR / f"{arm}_s{seed}_{name}_{farm}.npy").exists()
        for seed in SEEDS
        for name in ("hindcast", "scored")
    )


def fit_era5_core(
    *, train: pl.DataFrame, adapter: str, seed: int, with_idle_layer: bool
) -> tuple[core_module.WindCore, core_module.ResidualQuantiles]:
    """Fit the core with an ERA5-trained adapter and return it with its ERA5 residual quantiles."""
    options = core_module.CoreOptions(adapter=adapter)
    weather = models.weather_arrays(frame=train, options=options)
    target = train["target"].to_numpy()
    core = core_module.fit_core(
        weather=weather, target=target, with_idle_layer=with_idle_layer, seed=seed, options=options
    )
    residuals = core_module.ResidualQuantiles.fit(
        expected=core_module.predict_expected_power(core=core, weather=weather), observed=target
    )
    return core, residuals


def transfer(*, source: core_module.WindCore, target: core_module.WindCore) -> None:
    """Copy `source`'s parameters into `target`; new adapter inputs start with zero weight."""
    state = target.state_dict()
    for name, value in source.state_dict().items():
        if name not in state:
            continue
        if state[name].shape == value.shape:
            state[name] = value.clone()
        elif name == "adapter_net.0.weight":
            widened = torch.zeros_like(state[name])
            widened[:, : value.shape[1]] = value
            state[name] = widened
    target.load_state_dict(state)


def l1(*, core: core_module.WindCore, weather: core_module.WeatherArrays, goal: torch.Tensor):
    """Return the mean absolute error of the core's expected power."""
    return (core(weather=weather) - goal).abs().mean()


def train_core(
    *,
    adapter: str,
    start: core_module.WindCore | None,
    idle_start: float,
    with_idle_layer: bool,
    train_columns: dict[str, np.ndarray],
    train_target: np.ndarray,
    validation_columns: dict[str, np.ndarray],
    validation_target: np.ndarray,
    seed: int,
) -> core_module.WindCore:
    """Fit (from scratch) or fine-tune (from `start`) a core on member rows, with early stopping."""
    torch.manual_seed(seed)
    options = core_module.CoreOptions(adapter=adapter)
    core = core_module.WindCore(
        with_idle_layer=with_idle_layer, idle_floor=idle_start, options=options
    )
    scratch = start is None
    if start is not None:
        transfer(source=start.cpu(), target=core)
    core = core.to(DEVICE)
    adapter_names = ("adapter_net", "sector_log", "shear_adapter")
    adapter_parameters = [p for n, p in core.named_parameters() if n.startswith(adapter_names)]
    other_parameters = [p for n, p in core.named_parameters() if not n.startswith(adapter_names)]
    factor = 1.0 if scratch else FINE_TUNE_RATE_FACTOR
    steps = SCRATCH_STEPS if scratch else FINE_TUNE_STEPS
    optimiser = torch.optim.AdamW(
        [
            {"params": other_parameters, "lr": core_module.LEARNING_RATE * factor, "weight_decay": 0},
            {
                "params": adapter_parameters,
                "lr": core_module.ADAPTER_LEARNING_RATE * factor,
                "weight_decay": core_module.ADAPTER_WEIGHT_DECAY,
            },
        ]
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    weather = weather_on_device(columns=train_columns, index=slice(None))
    goal = torch.as_tensor(train_target, dtype=torch.float32, device=DEVICE)
    val_weather = weather_on_device(columns=validation_columns, index=slice(None))
    val_goal = torch.as_tensor(validation_target, dtype=torch.float32, device=DEVICE)
    log_prior = np.log(core_module.HUB_HEIGHT_PRIOR_M)
    best, best_state = float("inf"), None
    for step in range(steps + 1):
        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                held = float(l1(core=core, weather=val_weather, goal=val_goal))
            if held < best:
                best = held
                best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}
        if step == steps:
            break
        optimiser.zero_grad()
        loss = l1(core=core, weather=weather, goal=goal)
        penalty = (
            (torch.log(core.hub_height_m) - log_prior) ** 2
            / (2 * core_module.HUB_HEIGHT_PRIOR_LOG_STANDARD_DEVIATION**2)
            / goal.numel()
        )
        (loss + penalty).backward()
        optimiser.step()
        schedule.step()
    if best_state is None:
        msg = "no step produced a finite validation loss"
        raise RuntimeError(msg)
    core.load_state_dict(best_state)
    return core.eval()


class Hindcast:
    """One farm's hindcast and scored member weather, with the case arrays that split it."""

    def __init__(self, *, farm: str) -> None:
        """Load the farm's rows, check alignment, and build the arrays."""
        self.farm = farm
        sites = common.site_table().filter(pl.col("site") == farm)
        self.sites = sites
        series = int(sites["time_series_id"][0])
        self.era5 = common.era5_rows(sites=sites)
        self.train = models.add_derived_columns(
            frame=common.training_table(sites=sites, era5=self.era5)
        ).filter(pl.col("valid_time") < common.BASE_END)
        cases = common.cases_table(sites=sites)
        scored = common.ens_member_rows(sites=sites).join(
            cases.select(*common.KEY_COLUMNS), on=list(common.KEY_COLUMNS), how="semi"
        )
        ens.check_alignment(members=scored, cases=cases, farm=farm)
        hind = ens.hindcast_rows(series=series)
        keys = hind.gather_every(common.MEMBERS).select(common.KEY_COLUMNS)
        ens.check_alignment(members=hind, cases=keys, farm=farm)
        scored = ens.add_member_columns(rows=scored)
        hind = ens.add_member_columns(rows=hind)
        self.scored_columns = member_columns(rows=scored)
        self.hind_columns = member_columns(rows=hind)
        self.hind_keys = keys
        actual = common.observed_training_power(sites=sites).rename({"target": "actual_norm"})
        self.hind_actual = (
            keys.join(actual, on=["time_series_id", "valid_time"], how="left", validate="m:1")[
                "actual_norm"
            ]
            .to_numpy()
            .astype(np.float32)
        )
        self.usable = np.isfinite(self.hind_actual)
        self.init = keys["power_fcst_init_time"].to_numpy()
        self.valid = keys["valid_time"].to_numpy()
        self.scored_cases = cases.height
        self.hind_cases = keys.height
        self.with_idle_layer = farm != "W2"
        del scored, hind

    def rows_of(self, *, cases: np.ndarray, seed: int) -> np.ndarray:
        """Return the row indices of `MEMBERS_PER_CASE` seeded random members of each case."""
        rng = np.random.default_rng(seed)
        draws = np.argsort(rng.random((cases.size, common.MEMBERS)), axis=1)[:, :MEMBERS_PER_CASE]
        return (cases[:, None] * common.MEMBERS + draws).reshape(-1)

    def subset(self, *, cases: np.ndarray, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Return the member columns and the observed fraction of seeded draws from `cases`."""
        rows = self.rows_of(cases=cases, seed=seed)
        columns = {name: values[rows] for name, values in self.hind_columns.items()}
        return columns, np.repeat(self.hind_actual[cases], MEMBERS_PER_CASE)

    def blocks(self) -> np.ndarray:
        """Return the block (0 to FOLDS - 1) of each hindcast case, contiguous in run date."""
        days = np.unique(self.init)
        block_of_day = np.minimum(np.arange(days.size) * FOLDS // days.size, FOLDS - 1)
        return block_of_day[np.searchsorted(days, self.init)]

    def split(self, *, held_out: int | None) -> tuple[np.ndarray, np.ndarray]:
        """Return (training cases, early-stopping cases) for a fold, or for the final fit."""
        eligible = self.usable.copy()
        if held_out is not None:
            block = self.blocks()
            inside = block == held_out
            start, end = self.init[inside].min(), self.valid[inside].max()
            eligible &= (block != held_out) & ~((self.valid >= start) & (self.valid <= end))
        cutoff = self.init[eligible].max() - np.timedelta64(VALIDATION_DAYS, "D")
        validation = eligible & (self.init >= cutoff)
        training = eligible & (self.init < cutoff) & (self.valid < cutoff)
        return np.flatnonzero(training), np.flatnonzero(validation)


def run_era5_arms(*, data: Hindcast, arms: list[str]) -> dict[str, dict[int, core_module.WindCore]]:
    """Fit and store P2 and P8, and return the ERA5-trained cores for the fine-tuned arms."""
    farm = data.farm
    cores: dict[str, dict[int, core_module.WindCore]] = {"net": {}, "blind": {}}
    residuals_by: dict[tuple[str, int], core_module.ResidualQuantiles] = {}
    need_net = any(a in arms for a in ("P2", "P8", "P5"))
    need_blind = "P7" in arms
    for seed in SEEDS:
        for adapter, wanted in (("net", need_net), ("blind", need_blind)):
            if not wanted:
                continue
            started = time.time()
            core, residuals = fit_era5_core(
                train=data.train,
                adapter=adapter,
                seed=seed,
                with_idle_layer=data.with_idle_layer,
            )
            cores[adapter][seed] = core
            residuals_by[(adapter, seed)] = residuals
            print(f"{farm} ERA5 {adapter} seed {seed}: {time.time() - started:.0f} s", flush=True)
    if "P2" in arms and not done(arm="P2", farm=farm):
        for seed in SEEDS:
            for set_name, columns in (("hindcast", data.hind_columns), ("scored", data.scored_columns)):
                expected = predict_columns(core=cores["net"][seed], columns=columns)
                store(
                    arm="P2",
                    seed=seed,
                    set_name=set_name,
                    farm=farm,
                    mixture=mixture_of_expected(
                        expected=expected, residuals=residuals_by[("net", seed)]
                    ),
                )
            print(f"{farm} P2 seed {seed} stored", flush=True)
    if "P8" in arms and not done(arm="P8", farm=farm):
        run_p8(data=data, cores=cores["net"])
    return cores


XGB_COLUMN_ORDER: Final[tuple[str, ...]] = (
    "speed_10m",
    "speed_100m",
    "dir_sin_10m",
    "dir_cos_10m",
    "direction_sin",
    "direction_cos",
)
"""`common.XGBOOST_FEATURES` in the names of `member_columns`; the effective speed follows."""


def xgb_features(*, columns: dict[str, np.ndarray], effective: np.ndarray) -> np.ndarray:
    """Return XGBoost's six wind columns plus the effective hub-height speed."""
    return np.column_stack([*(columns[name] for name in XGB_COLUMN_ORDER), effective])


def run_p8(*, data: Hindcast, cores: dict[int, core_module.WindCore]) -> None:
    """Fit XGBoost with the effective speed as an extra column and store its mixtures."""
    farm = data.farm
    train = data.train
    target = train["target"].to_numpy()
    centres, curve = models.binned_curve(speed=train["wind_speed_100m"].to_numpy(), target=target)
    base = np.interp(train["wind_speed_100m"].to_numpy(), centres, curve)
    train_columns = {
        "speed_10m": train["wind_speed_10m"].to_numpy(),
        "speed_100m": train["wind_speed_100m"].to_numpy(),
        "temperature_c": train["temperature_c"].to_numpy(),
        "dir_sin_10m": train["wind_dir_sin_10m"].to_numpy(),
        "dir_cos_10m": train["wind_dir_cos_10m"].to_numpy(),
        "direction_sin": train["wind_dir_sin_100m"].to_numpy(),
        "direction_cos": train["wind_dir_cos_100m"].to_numpy(),
        "hour": train["hour"].to_numpy(),
        "lead_day": np.zeros(train.height, dtype=np.float32),
        "spread": np.zeros(train.height, dtype=np.float32),
    }
    for seed in SEEDS:
        effective = predict_columns(core=cores[seed], columns=train_columns, speed=True)
        booster = xgb.train(
            {**models.XGBOOST_PARAMETERS, "seed": seed},
            xgb.DMatrix(
                xgb_features(columns=train_columns, effective=effective),
                label=target,
                base_margin=np.tile(base[:, None], (1, 11)),
            ),
            num_boost_round=models.N_ROUNDS,
        )
        for set_name, columns in (("hindcast", data.hind_columns), ("scored", data.scored_columns)):
            eff = predict_columns(core=cores[seed], columns=columns, speed=True)
            size = columns["speed_100m"].size
            member = np.empty((size, 11), dtype=np.float32)
            for start in range(0, size, PREDICT_CHUNK):
                sl = slice(start, start + PREDICT_CHUNK)
                chunk = {name: values[sl] for name, values in columns.items()}
                member[sl] = _xgb_member(
                    booster=booster,
                    features=xgb_features(columns=chunk, effective=eff[sl]),
                    speed100=chunk["speed_100m"],
                    centres=centres,
                    curve=curve,
                )
            store(
                arm="P8",
                seed=seed,
                set_name=set_name,
                farm=farm,
                mixture=mixture_of_member_quantiles(member_quantiles=member),
            )
        print(f"{farm} P8 seed {seed} stored", flush=True)


def _xgb_member(
    *,
    booster: xgb.Booster,
    features: np.ndarray,
    speed100: np.ndarray,
    centres: np.ndarray,
    curve: np.ndarray,
) -> np.ndarray:
    """Return sorted, clipped XGBoost quantiles with the binned-curve base margin."""
    margin = np.tile(np.interp(speed100, centres, curve)[:, None], (1, 11))
    predicted = booster.predict(xgb.DMatrix(features, base_margin=margin))
    return np.sort(np.clip(predicted, 0.0, 1.0), axis=1).astype(np.float32)


def run_hindcast_arm(
    *,
    data: Hindcast,
    arm: str,
    cores: dict[str, dict[int, core_module.WindCore]],
) -> None:
    """Fit P4, P5, or P7 with a three-fold cross-fit, and store both mixtures per seed."""
    farm = data.farm
    adapter = {"P4": "net_ens", "P5": "net_ens", "P7": "blind"}[arm]
    pretrained = {"P4": None, "P5": "net", "P7": "blind"}[arm]
    idle_start = core_module.idle_floor_start(
        speed_100m=data.train["wind_speed_100m"].to_numpy(), target=data.train["target"].to_numpy()
    )
    block = data.blocks()
    for seed in SEEDS:
        started = time.time()
        expected_oof = np.empty(data.hind_columns["speed_100m"].size, dtype=np.float32)
        final_core = None
        for held_out in (*range(FOLDS), None):
            training, validation = data.split(held_out=held_out)
            train_columns, train_target = data.subset(cases=training, seed=seed)
            val_columns, val_target = data.subset(cases=validation, seed=seed + 100)
            core = train_core(
                adapter=adapter,
                start=None if pretrained is None else cores[pretrained][seed],
                idle_start=idle_start,
                with_idle_layer=data.with_idle_layer,
                train_columns=train_columns,
                train_target=train_target,
                validation_columns=val_columns,
                validation_target=val_target,
                seed=seed,
            )
            if held_out is None:
                final_core = core
                break
            cases = np.flatnonzero(block == held_out)
            rows = (cases[:, None] * common.MEMBERS + np.arange(common.MEMBERS)).reshape(-1)
            sub = {name: values[rows] for name, values in data.hind_columns.items()}
            expected_oof[rows] = predict_columns(core=core, columns=sub)
        if final_core is None:
            raise RuntimeError
        # residual quantiles from the out-of-fold case-mean expected power
        mean_expected = expected_oof.reshape(-1, common.MEMBERS).mean(axis=1)
        usable = data.usable
        residuals = core_module.ResidualQuantiles.fit(
            expected=mean_expected[usable], observed=data.hind_actual[usable].astype(np.float64)
        )
        store(
            arm=arm,
            seed=seed,
            set_name="hindcast",
            farm=farm,
            mixture=mixture_of_expected(expected=expected_oof, residuals=residuals),
        )
        expected_scored = predict_columns(core=final_core, columns=data.scored_columns)
        store(
            arm=arm,
            seed=seed,
            set_name="scored",
            farm=farm,
            mixture=mixture_of_expected(expected=expected_scored, residuals=residuals),
        )
        print(f"{farm} {arm} seed {seed}: {time.time() - started:.0f} s", flush=True)


def main() -> None:
    """Fit the requested arms for the requested farms, one farm after another."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default="W1,W2,W3")
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()
    arms = args.arms.split(",")
    for farm in args.farms.split(","):
        pending = [a for a in arms if not done(arm=a, farm=farm)]
        if not pending:
            continue
        data = Hindcast(farm=farm)
        cores = run_era5_arms(data=data, arms=[a for a in pending if a in ARMS])
        for arm in ("P4", "P5", "P7"):
            if arm in pending:
                run_hindcast_arm(data=data, arm=arm, cores=cores)


if __name__ == "__main__":
    main()
