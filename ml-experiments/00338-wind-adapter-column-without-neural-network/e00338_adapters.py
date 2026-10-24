"""Non-neural effective-wind-speed adapters for experiment 00338.

All three fit to the physics core's loss on ERA5 before 2024-04-01, as experiment 303's neural
adapter did, and the column handed to XGBoost is the core's effective hub-height speed.

- `TableCore` (x1): a speed scale looked up by direction sector, speed-ratio bin, and temperature
  bin, fitted jointly with the core by the optimiser that fitted the neural adapter. Setting a
  dimension's bin count to one drops that dimension (the drop-one arms).
- `fit_boosted_scale` (x2): a boosted tree for the log speed scale, trained through the core's
  loss by a custom XGBoost objective, with two passes that alternate with a refit of the core.
- `distilled_table` (x3): x1's cells filled with the median scale of the neural adapter.
"""

from collections.abc import Callable
from dataclasses import dataclass

import e00303_core as core_module  # isort: skip  # ty: ignore[unresolved-import]
import e00303_fit as fit303  # isort: skip  # ty: ignore[unresolved-import]
import e00338_common as common
import numpy as np
import torch  # ty: ignore[unresolved-import]
import xgboost as xgb
from scipy.stats import spearmanr
from torch import nn  # ty: ignore[unresolved-import]

SCALE_RANGE = core_module.SCALE_RANGE
MIN_SPEED = core_module.MIN_SHEAR_SPEED_M_S
PSEUDO_HUBER_DELTA = 0.1
BOOST_ROUNDS = (60, 30)
BOOST_PARAMETERS = {
    "tree_method": "hist",
    "max_depth": 3,
    "eta": 0.1,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "colsample_bylevel": 1.0,
    "colsample_bynode": 1.0,
    "base_score": 0.0,
}


def speed_ratio(*, speed_10m: np.ndarray, speed_100m: np.ndarray) -> np.ndarray:
    """Return the 10 m over 100 m speed ratio, as `WindCore._adapt` computes it."""
    usable = (speed_10m > MIN_SPEED) & (speed_100m > MIN_SPEED)
    ratio = np.where(usable, speed_10m / np.where(usable, speed_100m, 1.0), 0.5)
    return np.clip(ratio, 0.05, 1.5).astype(np.float32)


def sector_of(*, sin: np.ndarray, cos: np.ndarray) -> np.ndarray:
    """Return the direction sector (0 to 7) of the 100 m wind, as `WindCore._adapt` computes it."""
    degrees = np.degrees(np.arctan2(sin, cos)) % 360.0
    width = 360.0 / common.DIRECTION_SECTORS
    return (np.floor((degrees + width / 2) / width).astype(np.int64)) % common.DIRECTION_SECTORS


@dataclass(frozen=True)
class Edges:
    """Interior bin edges of the table's speed-ratio and temperature dimensions."""

    ratio: np.ndarray
    temperature: np.ndarray


def make_edges(*, columns: dict[str, np.ndarray], ratio_bins: int, temperature_bins: int) -> Edges:
    """Return quantile edges of the ERA5 training rows' speed ratio and temperature."""
    ratio = speed_ratio(speed_10m=columns["speed_10m"], speed_100m=columns["speed_100m"])
    return Edges(
        ratio=np.quantile(ratio, np.arange(1, ratio_bins) / ratio_bins).astype(np.float32),
        temperature=np.quantile(
            columns["temperature_c"], np.arange(1, temperature_bins) / temperature_bins
        ).astype(np.float32),
    )


def cell_of(*, columns: dict[str, np.ndarray], edges: Edges, use_sector: bool = True) -> np.ndarray:
    """Return each row's flat table cell: sector, then ratio bin, then temperature bin."""
    sector = (
        sector_of(sin=columns["direction_sin"], cos=columns["direction_cos"])
        if use_sector
        else np.zeros(columns["speed_100m"].size, dtype=np.int64)
    )
    ratio_bin = np.searchsorted(
        edges.ratio,
        speed_ratio(speed_10m=columns["speed_10m"], speed_100m=columns["speed_100m"]),
    )
    temperature_bin = np.searchsorted(edges.temperature, columns["temperature_c"])
    return (sector * (edges.ratio.size + 1) + ratio_bin) * (edges.temperature.size + 1) + (
        temperature_bin
    )


class TableCore(core_module.WindCore):
    """The physics core with a lookup-table speed scale in place of the neural adapter."""

    def __init__(
        self, *, with_idle_layer: bool, idle_floor: float, edges: Edges, use_sector: bool
    ) -> None:
        """Set the starting parameters; the table starts at a scale of one."""
        super().__init__(
            with_idle_layer=with_idle_layer,
            idle_floor=idle_floor,
            options=core_module.CoreOptions(adapter="none"),
        )
        self.edges = edges
        self.use_sector = use_sector
        cells = (common.DIRECTION_SECTORS if use_sector else 1) * (edges.ratio.size + 1)
        cells *= edges.temperature.size + 1
        self.table_raw = nn.Parameter(
            torch.full((cells,), core_module._inverse_sigmoid_range(1.0, *SCALE_RANGE))
        )

    def table_scale(self) -> torch.Tensor:
        """Return the fitted speed scale of every cell."""
        return core_module._sigmoid_range(self.table_raw, *SCALE_RANGE)

    def _adapt(self, *, speed: torch.Tensor, weather: core_module.WeatherArrays) -> torch.Tensor:
        """Return `speed` times the table's scale for each row's cell."""
        columns = {
            "speed_10m": core_module._t(weather.speed_10m).cpu().numpy(),
            "speed_100m": core_module._t(weather.speed_100m).cpu().numpy(),
            "temperature_c": core_module._t(weather.temperature_c).cpu().numpy(),
            "direction_sin": core_module._t(weather.direction_sin).cpu().numpy(),
            "direction_cos": core_module._t(weather.direction_cos).cpu().numpy(),
        }
        cell = torch.as_tensor(
            cell_of(columns=columns, edges=self.edges, use_sector=self.use_sector),
            device=speed.device,
        )
        return (speed * self.table_scale().to(speed.device)[cell]).clamp(min=0.0)


def adam_fit(
    *,
    core: core_module.WindCore,
    weather: core_module.WeatherArrays,
    target: np.ndarray,
    adapter_names: tuple[str, ...],
    seed: int,
) -> None:
    """Fit `core` in place by full-batch AdamW on the L1 loss, as `core_module.fit_core` does."""
    torch.manual_seed(seed)
    torch.set_num_threads(core_module.THREADS)
    goal = torch.as_tensor(target, dtype=torch.float32)
    named = list(core.named_parameters())
    adapter = [p for n, p in named if n.startswith(adapter_names)]
    others = [p for n, p in named if not n.startswith(adapter_names)]
    steps = core_module.ADAPTER_STEPS
    groups = [{"params": others, "lr": core_module.LEARNING_RATE, "weight_decay": 0.0}]
    if adapter:
        groups.append(
            {
                "params": adapter,
                "lr": core_module.ADAPTER_LEARNING_RATE,
                "weight_decay": core_module.ADAPTER_WEIGHT_DECAY,
            }
        )
    optimiser = torch.optim.AdamW(groups)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    log_prior = np.log(core_module.HUB_HEIGHT_PRIOR_M)
    for _ in range(steps):
        optimiser.zero_grad()
        loss = (core(weather=weather) - goal).abs().mean()
        penalty = (
            (torch.log(core.hub_height_m) - log_prior) ** 2
            / (2 * core_module.HUB_HEIGHT_PRIOR_LOG_STANDARD_DEVIATION**2)
            / goal.numel()
        )
        (loss + penalty).backward()
        optimiser.step()
        schedule.step()
    core.eval()


def table_weather(*, train_columns: dict[str, np.ndarray]) -> core_module.WeatherArrays:
    """Return the core's weather arrays from the ERA5 training columns."""
    return core_module.WeatherArrays(
        speed_10m=train_columns["speed_10m"],
        speed_100m=train_columns["speed_100m"],
        temperature_c=train_columns["temperature_c"],
        direction_sin=train_columns["direction_sin"],
        direction_cos=train_columns["direction_cos"],
        hour=train_columns["hour"],
    )


def fit_table_core(
    *,
    train_columns: dict[str, np.ndarray],
    target: np.ndarray,
    edges: Edges,
    with_idle_layer: bool,
    seed: int,
    use_sector: bool = True,
) -> TableCore:
    """Fit x1: the core and its lookup table jointly on the ERA5 training rows."""
    core = TableCore(
        with_idle_layer=with_idle_layer,
        idle_floor=core_module.idle_floor_start(
            speed_100m=train_columns["speed_100m"], target=target
        ),
        edges=edges,
        use_sector=use_sector,
    )
    adam_fit(
        core=core,
        weather=table_weather(train_columns=train_columns),
        target=target,
        adapter_names=("table_raw",),
        seed=seed,
    )
    return core


def power_from_speed(*, core: core_module.WindCore, speed: torch.Tensor) -> torch.Tensor:
    """Return the core's expected power at hub-height `speed` (the tail of `WindCore.forward`)."""
    active = core.active_power(speed=speed)
    if not core.with_idle_layer:
        return active
    floor = nn.functional.softplus(core.raw_idle_floor)
    dip = nn.functional.softplus(core.raw_dip)
    position = speed / core.cut_in_m_s
    return torch.sqrt(active**2 + floor**2) - dip * position * torch.exp(1.0 - position)


def scale_features(*, columns: dict[str, np.ndarray]) -> np.ndarray:
    """Return x2's inputs: direction sine and cosine, speed ratio, temperature, hour terms."""
    angle = columns["hour"] * (2 * np.pi / 24.0)
    return np.column_stack(
        [
            columns["direction_sin"],
            columns["direction_cos"],
            speed_ratio(speed_10m=columns["speed_10m"], speed_100m=columns["speed_100m"]),
            columns["temperature_c"],
            np.sin(angle),
            np.cos(angle),
        ]
    ).astype(np.float32)


def hub_speed(*, core: core_module.WindCore, columns: dict[str, np.ndarray]) -> np.ndarray:
    """Return the core's hub-height speed for every row, leaving the core on the CPU."""
    speed = fit303.predict_columns(core=core, columns=columns, speed=True)
    core.cpu()
    return speed


def base_speed(*, core: core_module.WindCore, columns: dict[str, np.ndarray]) -> np.ndarray:
    """Return the no-adapter core's hub-height speed for every row of `columns`."""
    return hub_speed(core=core, columns=columns)


@dataclass
class BoostedScale:
    """x2's fitted pieces: the final core and the boosters whose log scales add."""

    core: core_module.WindCore
    boosters: list[xgb.Booster]
    constraints: str

    def log_scale(self, *, columns: dict[str, np.ndarray]) -> np.ndarray:
        """Return the summed boosters' log speed scale, clipped to the adapter's range."""
        features = xgb.DMatrix(scale_features(columns=columns))
        margin = sum(b.predict(features) for b in self.boosters)
        return np.clip(margin, np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1])).astype(np.float32)

    def effective_speed(self, *, columns: dict[str, np.ndarray]) -> np.ndarray:
        """Return the hub-height speed after the boosted scale, as XGBoost's extra column."""
        speed = base_speed(core=self.core, columns=columns) * np.exp(
            self.log_scale(columns=columns)
        )
        return np.clip(speed, 0.0, None).astype(np.float32)


def monotone_constraints(*, scale: np.ndarray, columns: dict[str, np.ndarray]) -> str:
    """Return XGBoost's monotone constraints: the sign of x1's fitted scale against ratio and T."""
    ratio = speed_ratio(speed_10m=columns["speed_10m"], speed_100m=columns["speed_100m"])
    signs = []
    for values in (ratio, columns["temperature_c"]):
        rho = spearmanr(values, scale).statistic
        signs.append(int(np.sign(rho)))
    return f"(0,0,{signs[0]},{signs[1]},0,0)"


def _objective(
    *, core: core_module.WindCore, base: torch.Tensor, target: torch.Tensor, offset: np.ndarray
) -> Callable[[np.ndarray, xgb.DMatrix], tuple[np.ndarray, np.ndarray]]:
    """Return a custom XGBoost objective: pseudo-Huber loss of the core's power at base * exp(m)."""

    def objective(predicted: np.ndarray, _data: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        margin = torch.as_tensor(predicted + offset, dtype=torch.float32).requires_grad_(True)
        scale = torch.exp(margin.clamp(np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1])))
        power = power_from_speed(core=core, speed=(base * scale).clamp(min=0.0))
        (grad,) = torch.autograd.grad(power.sum(), margin)
        jacobian = grad.numpy().astype(np.float64)
        residual = (power.detach() - target).numpy().astype(np.float64)
        root = np.sqrt(1.0 + (residual / PSEUDO_HUBER_DELTA) ** 2)
        gradient = jacobian * residual / root
        hessian = jacobian**2 / root**3 + 1e-4
        return gradient, hessian

    return objective


def fit_boosted_scale(
    *,
    core: core_module.WindCore,
    train_columns: dict[str, np.ndarray],
    target: np.ndarray,
    constraints: str,
    seed: int,
) -> BoostedScale:
    """Fit x2: boost the log scale through the core's loss, refit the core, and boost again."""
    features = xgb.DMatrix(scale_features(columns=train_columns))
    goal = torch.as_tensor(target, dtype=torch.float32)
    boosters: list[xgb.Booster] = []
    margin = np.zeros(target.size)
    weather = table_weather(train_columns=train_columns)
    for rounds in BOOST_ROUNDS:
        base = torch.as_tensor(base_speed(core=core, columns=train_columns))
        booster = xgb.train(
            {**BOOST_PARAMETERS, "monotone_constraints": constraints, "seed": seed},
            features,
            num_boost_round=rounds,
            obj=_objective(core=core, base=base, target=goal, offset=margin),
        )
        boosters.append(booster)
        margin = margin + booster.predict(features)
        margin = np.clip(margin, np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1]))
        fixed = torch.as_tensor(np.exp(margin), dtype=torch.float32)
        core._adapt = lambda *, speed, weather, fixed=fixed: (speed * fixed).clamp(min=0.0)
        adam_fit_core_only(core=core, weather=weather, target=target, seed=seed)
        del core._adapt
    return BoostedScale(core=core, boosters=boosters, constraints=constraints)


def adam_fit_core_only(
    *,
    core: core_module.WindCore,
    weather: core_module.WeatherArrays,
    target: np.ndarray,
    seed: int,
) -> None:
    """Refit the core's curve parameters with its (patched) scale held fixed."""
    adam_fit(core=core, weather=weather, target=target, adapter_names=("__none__",), seed=seed)


def distilled_table(
    *,
    net_core: core_module.WindCore,
    train_columns: dict[str, np.ndarray],
    edges: Edges,
    with_idle_layer: bool,
) -> tuple[TableCore, np.ndarray]:
    """Fit x3: x1's cells filled with the median scale of the neural adapter on ERA5 rows.

    A cell's median is clipped to the table's scale range, which the neural adapter's offset can
    exceed at low speeds. The returned core is the neural adapter's own core (curve and hub
    height) with its adapter replaced by the table, so the column differs from (c) only in the
    adapter step. The second return value is the count of ERA5 rows in each cell.
    """
    effective = hub_speed(core=net_core, columns=train_columns)
    net_core._adapt = lambda *, speed, weather: speed
    before = hub_speed(core=net_core, columns=train_columns)
    del net_core._adapt
    usable = before > MIN_SPEED
    cell = cell_of(columns=train_columns, edges=edges)
    cells = (common.DIRECTION_SECTORS) * (edges.ratio.size + 1) * (edges.temperature.size + 1)
    scale = np.ones(cells, dtype=np.float32)
    counts = np.zeros(cells, dtype=np.int64)
    ratio = effective[usable] / before[usable]
    for index in range(cells):
        inside = cell[usable] == index
        counts[index] = int(inside.sum())
        if counts[index]:
            scale[index] = np.median(ratio[inside])
    margin = 1e-4
    scale = np.clip(scale, SCALE_RANGE[0] + margin, SCALE_RANGE[1] - margin)
    table = TableCore(
        with_idle_layer=with_idle_layer,
        idle_floor=net_core.raw_idle_floor.item(),
        edges=edges,
        use_sector=True,
    )
    table.load_state_dict(
        {
            **{k: v for k, v in net_core.state_dict().items() if k in table.state_dict()},
            "table_raw": torch.as_tensor(
                np.log((scale - SCALE_RANGE[0]) / (SCALE_RANGE[1] - scale)), dtype=torch.float32
            ),
        }
    )
    return table.eval(), counts
