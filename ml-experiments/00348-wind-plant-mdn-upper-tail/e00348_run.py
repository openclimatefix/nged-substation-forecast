# ruff: noqa: E501
"""Step 1 of experiment 00348: train the upper-tail heads on experiment 00347's plant summaries.

Every arm is experiment 00347's pooled three-component Gaussian mixture on the residual from the
Stage 3 plant's median (inputs: the summary of the plant's 51 member powers, the lead terms, the
farm), with the tail change of the arm:

- `y2p`: pinball loss over the 11 levels, all weights 1 (y2's head with the loss changed only).
- `z1`: the pinball loss weighted 3 at q01, q90, q95, q99; ceiling 1.0 as in y2.
- `z2`: unweighted pinball loss, a point mass at the farm's plant scale with weight `pi` output by
  the head, and every quantile capped at the scale.
- `z3`: weighted pinball loss, the mixture supplies q01 to q90, a small network (plant p90,
  saturation share, lead) supplies q95 and q99 as non-negative increments, capped at the scale.
- `z4`: z1's weighted loss with z2's point mass and cap.
- `y2` (refit with 00347's own code, the reproduction gate).

`<arm>_basefarm` is the base-only control (plant median, lead terms, farm; the z3 network gets the
median and lead). Gradients pass through a mixture quantile by one Newton step on the CDF taken at
the bisection root, which gives the implicit derivative of the quantile with respect to the
mixture's parameters.

    PYTHONPATH=<348>:<347>:<230>:<303>:<310>:<326>:<336 folders> \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00348-*/e00348_run.py --arms z1,z2 [--first-only]
"""

import os
import sys
import time
from datetime import datetime
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_common as c303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00310_fit as f310  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00347_s4_data as data  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00347_s4_run as run347  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00348_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402  # ty: ignore[unresolved-import]
from torch import nn  # noqa: E402  # ty: ignore[unresolved-import]

DEVICE: Final = run347.DEVICE
EPOCHS: Final[int] = run347.EPOCHS
BATCH: Final[int] = run347.BATCH
LEARNING_RATE: Final[float] = run347.LEARNING_RATE
WEIGHT_DECAY: Final[float] = run347.WEIGHT_DECAY
COMPONENTS: Final[int] = f310.COMPONENTS
BISECTIONS: Final[int] = 30
MIN_DENSITY: Final[float] = 0.1
"""Floor on the density in the Newton step, so a tail quantile's gradient cannot explode."""
LEVELS: Final = torch.tensor(c303.LEVELS, dtype=torch.float32)
WEIGHTED: Final = torch.tensor([3, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3], dtype=torch.float32)
UNWEIGHTED: Final = torch.ones(11)
ARMS: Final[dict[str, dict]] = {
    "y2p": {"weights": UNWEIGHTED, "point_mass": False, "tail_net": False, "capped": False},
    "z1": {"weights": WEIGHTED, "point_mass": False, "tail_net": False, "capped": False},
    "z2": {"weights": UNWEIGHTED, "point_mass": True, "tail_net": False, "capped": True},
    "z3": {"weights": WEIGHTED, "point_mass": False, "tail_net": True, "capped": True},
    "z4": {"weights": WEIGHTED, "point_mass": True, "tail_net": False, "capped": True},
}
TAIL_INPUTS: Final[dict[str, list[int]]] = {"full": [4, 5, 8], "basefarm": [0, 3]}
"""Columns of the standardised input read by z3's increment network: p90, saturation share, lead
(full), or median and lead (base-only control)."""
PLACEBO_SEED: Final[int] = run347.PLACEBO_SEED


class TailHead(nn.Module):
    """Three-component Gaussian mixture with an optional point mass and increment network."""

    def __init__(self, *, n_in: int, n_tail_in: int, point_mass: bool, tail_net: bool) -> None:
        """Build the network; the last layer starts at the mixture of experiment 310."""
        super().__init__()
        self.point_mass, self.has_tail = point_mass, tail_net
        outputs = 3 * COMPONENTS + (1 if point_mass else 0)
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, outputs)
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        bias = [0.0] * 3 + [-0.1, 0.0, 0.1] + [float(np.log(0.08))] * 3
        if point_mass:
            bias.append(-3.0)
        with torch.no_grad():
            last.bias.copy_(torch.tensor(bias))
        if tail_net:
            self.tail = nn.Sequential(nn.Linear(n_tail_in, 16), nn.GELU(), nn.Linear(16, 2))
            end = self.tail[-1]
            assert isinstance(end, nn.Linear)
            nn.init.zeros_(end.weight)
            with torch.no_grad():
                end.bias.copy_(torch.tensor([-3.5, -3.5]))  # softplus(-3.5) = 0.03

    def quantiles(
        self,
        *,
        x: torch.Tensor,
        x_tail: torch.Tensor,
        mu: torch.Tensor,
        upper: torch.Tensor,
        differentiable: bool,
    ) -> torch.Tensor:
        """Return the 11 quantiles (rows, 11) in capacity fractions, clamped to [0, upper]."""
        out = self.net(x)
        log_w = torch.log_softmax(out[:, :COMPONENTS], dim=1)
        mean = out[:, COMPONENTS : 2 * COMPONENTS]
        std = torch.exp(out[:, 2 * COMPONENTS : 3 * COMPONENTS].clamp(-6.0, 1.0))
        tau = LEVELS.to(x.device)[None, :].expand(len(x), 11)
        if self.point_mass:
            pi = torch.sigmoid(out[:, 3 * COMPONENTS]).clamp(max=0.5)
            at_mass = tau >= (1.0 - pi)[:, None]
            tau = (tau / (1.0 - pi)[:, None]).clamp(max=1.0 - 1e-4)
        weights = log_w.exp()
        centre = mu[:, None] + mean  # (rows, components)

        def cdf(q: torch.Tensor) -> torch.Tensor:
            z = (q[:, :, None] - centre[:, None, :]) / std[:, None, :]
            return (weights[:, None, :] * 0.5 * (1.0 + torch.erf(z / 2**0.5))).sum(dim=2)

        with torch.no_grad():
            lo, hi = (
                (mu[:, None] - 2.0).expand_as(tau).clone(),
                (mu[:, None] + 2.0).expand_as(tau).clone(),
            )
            tau_d = tau.detach()
            for _ in range(BISECTIONS):
                mid = 0.5 * (lo + hi)
                below = cdf(mid) < tau_d
                lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
            root = 0.5 * (lo + hi)
            z0 = (root[:, :, None] - centre[:, None, :]) / std[:, None, :]
            density = (
                (
                    weights[:, None, :]
                    * torch.exp(-0.5 * z0**2)
                    / (std[:, None, :] * (2 * np.pi) ** 0.5)
                )
                .sum(dim=2)
                .clamp_min(MIN_DENSITY)
            )
        q = root if not differentiable else root + (tau - cdf(root)) / density
        if self.has_tail:
            inc = nn.functional.softplus(self.tail(x_tail))
            q90 = q[:, 8]
            q = torch.cat(
                [q[:, :9], (q90 + inc[:, 0])[:, None], (q90 + inc[:, 0] + inc[:, 1])[:, None]],
                dim=1,
            )
        if self.point_mass:
            q = torch.where(at_mass, upper[:, None].expand_as(q), q)
        return torch.minimum(q, upper[:, None]).clamp_min(0.0)


def pinball(*, q: torch.Tensor, actual: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return the weighted mean pinball loss over rows and the 11 levels."""
    levels = LEVELS.to(q.device)[None, :]
    error = actual[:, None] - q
    loss = torch.maximum(levels * error, (levels - 1.0) * error)
    return (loss * weights.to(q.device)[None, :]).mean()


def train_and_predict(  # noqa: PLR0915
    *,
    arm: str,
    hindcast: pl.DataFrame,
    summary_h: np.ndarray,
    cutoff: datetime,
    valid_limit: datetime | None,
    kind: str,
    targets: dict[str, tuple[pl.DataFrame, np.ndarray]],
    seed: int,
) -> dict[str, np.ndarray]:
    """Train one head on the hindcast's kept rows and return each target set's 11 quantiles."""
    config = ARMS[arm]
    torch.manual_seed(seed)
    usable = (~hindcast["masked"]).to_numpy()
    init, valid = hindcast["power_fcst_init_time"], hindcast["valid_time"]
    train = usable & ((init < cutoff) & (valid < cutoff)).to_numpy()
    stop = usable & (init >= cutoff).to_numpy()
    if valid_limit is not None:
        stop &= (valid < valid_limit).to_numpy()
    scales = common.farm_scales()
    x_h = run347.build_features(summary=summary_h, frame=hindcast, kind=kind, farm_columns=True)
    mean, std = x_h[train].mean(axis=0), np.maximum(x_h[train].std(axis=0), 1e-6)

    def tensor(x: np.ndarray) -> torch.Tensor:
        return torch.tensor((x - mean) / std, dtype=torch.float32, device=DEVICE)

    def upper_of(frame: pl.DataFrame) -> torch.Tensor:
        cap = [scales[s] if config["capped"] else 1.0 for s in frame["site"].to_list()]
        return torch.tensor(cap, dtype=torch.float32, device=DEVICE)

    tail_cols = TAIL_INPUTS[kind if kind == "basefarm" else "full"]
    mu_t = torch.tensor(summary_h[:, 0], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(hindcast["actual_norm"].to_numpy(), dtype=torch.float32, device=DEVICE)
    up_t = upper_of(hindcast)
    x_t = tensor(x_h)
    model = TailHead(
        n_in=x_h.shape[1],
        n_tail_in=len(tail_cols),
        point_mass=config["point_mass"],
        tail_net=config["tail_net"],
    ).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_i = torch.tensor(np.flatnonzero(train), device=DEVICE)
    stop_i = torch.tensor(np.flatnonzero(stop), device=DEVICE)
    generator = torch.Generator().manual_seed(seed)

    def loss_on(i: torch.Tensor, *, differentiable: bool) -> torch.Tensor:
        q = model.quantiles(
            x=x_t[i],
            x_tail=x_t[i][:, tail_cols],
            mu=mu_t[i],
            upper=up_t[i],
            differentiable=differentiable,
        )
        return pinball(q=q, actual=y_t[i], weights=config["weights"])

    best, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        order = train_i[torch.randperm(len(train_i), generator=generator).to(DEVICE)]
        for start in range(0, len(order), BATCH):
            loss = loss_on(order[start : start + BATCH], differentiable=True)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            parts = [
                float(loss_on(stop_i[s : s + 8192], differentiable=False))
                * len(stop_i[s : s + 8192])
                for s in range(0, len(stop_i), 8192)
            ]
        score = sum(parts) / len(stop_i)
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    out = {}
    for name, (frame, summary_s) in targets.items():
        xs_t = tensor(
            run347.build_features(summary=summary_s, frame=frame, kind=kind, farm_columns=True)
        )
        mu_s = torch.tensor(summary_s[:, 0], dtype=torch.float32, device=DEVICE)
        up_s = upper_of(frame)
        result = np.empty((len(frame), 11), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(frame), 20_000):
                sl = slice(start, start + 20_000)
                q = model.quantiles(
                    x=xs_t[sl],
                    x_tail=xs_t[sl][:, tail_cols],
                    mu=mu_s[sl],
                    upper=up_s[sl],
                    differentiable=False,
                )
                result[sl] = q.cpu().numpy()
        out[name] = np.sort(result, axis=1)
    return out


def save(*, name: str, array: np.ndarray) -> None:
    """Write an array to the parts folder atomically."""
    common.PARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = common.PARTS_DIR / f"{name}.npy"
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array.astype(np.float32))
    tmp.replace(path)


def fit_y2(
    *, arm: str, kind: str, hindcast: pl.DataFrame, scored: pl.DataFrame, started: float
) -> None:
    """Refit experiment 00347's y2 head (or its base-only control) with 00347's own code."""
    summary_h = data.summaries(arm_plant="y2", set_name="hindcast")
    summary_s = data.summaries(arm_plant="y2", set_name="scored")
    placebo_s = run347.shuffled(summary=summary_s, frame=scored, seed=PLACEBO_SEED)
    for seed in common.SEEDS:
        if (common.PARTS_DIR / f"{arm}_h{seed}.npy").exists():
            continue
        out = run347.train_and_predict(
            hindcast=hindcast,
            summary_h=summary_h,
            rows=np.ones(len(hindcast), dtype=bool),
            cutoff=data.MAIN_CUTOFF,
            valid_limit=None,
            kind=kind,
            farm_columns=True,
            targets={"scored": (scored, summary_s), "placebo": (scored, placebo_s)},
            seed=seed,
        )
        save(name=f"{arm}_h{seed}", array=out["scored"])
        save(name=f"{arm}_placebo_h{seed}", array=out["placebo"])
        print(f"{arm} seed {seed} done {time.time() - started:.0f} s", flush=True)


def main() -> None:
    """Train the arms named by `--arms` (comma list of y2, y2p, z1 to z4) and store their quantiles."""
    wanted = sys.argv[sys.argv.index("--arms") + 1].split(",")
    first_only = "--first-only" in sys.argv
    started = time.time()
    hindcast = data.hindcast_frame()
    scored = data.scored_frame()
    in_rep = (hindcast["power_fcst_init_time"] >= data.REPLICATE_START).to_numpy()
    replicate = hindcast.filter(pl.Series(in_rep))
    summary_h = data.summaries(arm_plant="y2", set_name="hindcast")
    summary_s = data.summaries(arm_plant="y2", set_name="scored")
    placebo_s = run347.shuffled(summary=summary_s, frame=scored, seed=PLACEBO_SEED)
    for arm in wanted:
        if arm == "y2":
            fit_y2(arm="y2", kind="full", hindcast=hindcast, scored=scored, started=started)
            fit_y2(
                arm="y2_basefarm",
                kind="basefarm",
                hindcast=hindcast,
                scored=scored,
                started=started,
            )
            for seed in common.REPLICATE_SEEDS:
                if (common.PARTS_DIR / f"y2_rep_h{seed}.npy").exists():
                    continue
                fit = ~in_rep
                out = run347.train_and_predict(
                    hindcast=hindcast.filter(pl.Series(fit)),
                    summary_h=summary_h[fit],
                    rows=np.ones(int(fit.sum()), dtype=bool),
                    cutoff=data.REPLICATE_CUTOFF,
                    valid_limit=data.REPLICATE_START,
                    kind="full",
                    farm_columns=True,
                    targets={"rep": (replicate, summary_h[in_rep])},
                    seed=seed,
                )
                save(name=f"y2_rep_h{seed}", array=out["rep"])
                print(f"y2 replicate seed {seed} done {time.time() - started:.0f} s", flush=True)
            continue
        for kind, suffix in (("full", ""), ("basefarm", "_basefarm")):
            for seed in common.SEEDS:
                if (common.PARTS_DIR / f"{arm}{suffix}_h{seed}.npy").exists():
                    continue
                targets = {"scored": (scored, summary_s)}
                if kind == "full":
                    targets["placebo"] = (scored, placebo_s)
                out = train_and_predict(
                    arm=arm,
                    hindcast=hindcast,
                    summary_h=summary_h,
                    cutoff=data.MAIN_CUTOFF,
                    valid_limit=None,
                    kind=kind,
                    targets=targets,
                    seed=seed,
                )
                save(name=f"{arm}{suffix}_h{seed}", array=out["scored"])
                if kind == "full":
                    save(name=f"{arm}_placebo_h{seed}", array=out["placebo"])
                print(f"{arm}{suffix} seed {seed} done {time.time() - started:.0f} s", flush=True)
                if first_only:
                    return
        for seed in common.REPLICATE_SEEDS:
            if (common.PARTS_DIR / f"{arm}_rep_h{seed}.npy").exists():
                continue
            fit = ~in_rep
            out = train_and_predict(
                arm=arm,
                hindcast=hindcast.filter(pl.Series(fit)),
                summary_h=summary_h[fit],
                cutoff=data.REPLICATE_CUTOFF,
                valid_limit=data.REPLICATE_START,
                kind="full",
                targets={"rep": (replicate, summary_h[in_rep])},
                seed=seed,
            )
            save(name=f"{arm}_rep_h{seed}", array=out["rep"])
            print(f"{arm} replicate seed {seed} done {time.time() - started:.0f} s", flush=True)
    print("all heads done", flush=True)


if __name__ == "__main__":
    main()
