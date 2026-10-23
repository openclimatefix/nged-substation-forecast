# ruff: noqa: E501, D107
"""Fit every arm of experiment 243 in both settings and write each job's scored-year quantiles.

Settings: `own_history` fits on the scored farm's own tuning months; `zero_history` fits on the other
five farms' tuning months only, with the plant scale and ceiling the median over those five farms.
One job is one (setting, farm, arm, seed); its output file is written atomically and skipped when it
exists, so a crashed run resumes. Run:

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00243-*/e00243_fit.py
"""

import itertools
import multiprocessing
import sys
import time
from functools import cache

import e00243_common as common
import e00243_models as models
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
import xgboost as xgb
from studies.tail_calibration import level_column

N_PROCESSES = 4
EPOCHS = 60
BATCH = 512
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
LEVELS_T = None


@cache
def farm(site: str) -> common.Farm:
    """Load a farm once per process."""
    return common.load_farm(site)


def lag_residual(*, f: common.Farm, mu: np.ndarray) -> np.ndarray:
    """Return observed power one day earlier minus the plant forecast then (0 when unobserved)."""
    t = f.frame["time"].dt.epoch("us").to_numpy()
    days = np.where(f.frame["lead_hours"].to_numpy() <= 24, 1, 2)
    lag_t = t - days * 86_400_000_000
    idx = np.searchsorted(t, lag_t)
    idx_c = np.minimum(idx, len(t) - 1)
    found = (t[idx_c] == lag_t) & (f.frame["lag_missing"].to_numpy() == 0)
    return np.where(found, f.frame["lag_power"].to_numpy() - mu[idx_c], 0.0)


def features(*, f: common.Farm, scale: float, ceiling: float) -> dict[str, np.ndarray]:
    """Return a farm's raw, head, and blind feature matrices and the plant arrays at the setting's scale."""
    frame = f.frame
    mu = common.plant_forecast(farm=f, scale=scale, ceiling=ceiling)
    raw = frame.select(list(common.RAW_FEATURES)).to_numpy().astype(np.float64)
    ghi = frame["ghi_mean"].to_numpy()
    lagres = lag_residual(f=f, mu=mu)
    spread = frame["ghi_std"].to_numpy() / (ghi + 1.0)
    head = np.column_stack(
        [
            mu,
            spread,
            frame["ghi_std"].to_numpy(),
            lagres,
            frame["lag_missing"].to_numpy(),
            frame["elevation_deg"].to_numpy(),
            frame["csi"].to_numpy(),
            frame["lead_hours"].to_numpy(),
            frame["temperature_mean"].to_numpy(),
        ]
    )
    blind = np.column_stack(
        [
            ghi,
            frame["ghi_std"].to_numpy(),
            frame["dni_mean"].to_numpy(),
            frame["dhi_mean"].to_numpy(),
            frame["temperature_mean"].to_numpy(),
            frame["elevation_deg"].to_numpy(),
            frame["csi"].to_numpy(),
            frame["lead_hours"].to_numpy(),
            frame["lag_power"].to_numpy(),
            frame["lag_missing"].to_numpy(),
        ]
    )
    return {"mu": mu, "raw": raw, "head": head, "blind": blind}


class Setting:
    """The training and test arrays of one (setting, scored farm)."""

    def __init__(self, *, setting: str, site: str) -> None:
        self.site = site
        others = [s for s in common.FARMS if s != site]
        train_sites = [site] if setting == "own_history" else others
        if setting == "own_history":
            self.scale, self.ceiling = farm(site).own_scale, farm(site).own_ceiling
        else:
            self.scale = float(np.median([farm(s).own_scale for s in others]))
            self.ceiling = float(np.median([farm(s).own_ceiling for s in others]))
        parts = {"train": [], "val": []}
        for s in train_sites:
            f = farm(s)
            feats = features(f=f, scale=self.scale, ceiling=self.ceiling)
            output = f.frame["output"].to_numpy().astype(np.float64)
            for key, mask in (
                ("train", f.tuning & ~f.validation),
                ("val", f.tuning & f.validation),
            ):
                parts[key].append(
                    {"output": output[mask], **{k: v[mask] for k, v in feats.items()}}
                )
        cat = {
            key: {k: np.concatenate([p[k] for p in parts[key]]) for k in parts[key][0]}
            for key in parts
        }
        mu_all = np.concatenate([cat["train"]["mu"], cat["val"]["mu"]])
        out_all = np.concatenate([cat["train"]["output"], cat["val"]["output"]])
        self.edges, self.table = common.bin_offsets(mu=mu_all, output=out_all)
        self.train, self.val = cat["train"], cat["val"]
        test = farm(site)
        feats = features(f=test, scale=self.scale, ceiling=self.ceiling)
        mask = test.scored
        self.test = {
            "output": test.frame["output"].to_numpy().astype(np.float64)[mask],
            **{k: v[mask] for k, v in feats.items()},
        }
        self.test_frame = test.frame.filter(pl.Series(mask))
        self.test_frame_extra = {
            "dni": test.frame["dni_mean"].to_numpy()[mask],
            "dhi": test.frame["dhi_mean"].to_numpy()[mask],
            "temperature": test.frame["temperature_mean"].to_numpy()[mask],
            "zenith": test.frame["zenith_deg"].to_numpy()[mask],
            "azimuth": test.frame["sun_azimuth_deg"].to_numpy()[mask],
        }
        self.extra: dict[str, dict[str, np.ndarray]] = {}
        # Physics inputs (beam, diffuse, temperature, angles) for the adapter, per split.
        for key, mask_fn in (
            ("train", lambda f: f.tuning & ~f.validation),
            ("val", lambda f: f.tuning & f.validation),
        ):
            pieces = []
            for s in train_sites:
                f = farm(s)
                m = mask_fn(f)
                pieces.append(
                    {
                        "dni": f.frame["dni_mean"].to_numpy()[m],
                        "dhi": f.frame["dhi_mean"].to_numpy()[m],
                        "temperature": f.frame["temperature_mean"].to_numpy()[m],
                        "zenith": f.frame["zenith_deg"].to_numpy()[m],
                        "azimuth": f.frame["sun_azimuth_deg"].to_numpy()[m],
                    }
                )
            self.extra[key] = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        self.extra["test"] = self.test_frame_extra
        # Standardisation from the training rows.
        self.stats = {}
        for name in ("raw", "head", "blind"):
            x = self.train[name]
            self.stats[name] = (x.mean(axis=0), x.std(axis=0) + 1e-6)


def make_batch(*, st: Setting, split: str, device: str) -> dict[str, torch.Tensor]:
    """Return a split's tensors: standardised inputs, plant arrays, offsets, and physics inputs."""
    ds = getattr(st, split)
    ex = st.extra[split]
    out: dict[str, torch.Tensor] = {}
    for name in ("raw", "head", "blind"):
        mean, std = st.stats[name]
        out[name] = torch.tensor((ds[name] - mean) / std, dtype=torch.float32, device=device)
    n = len(ds["mu"])
    out["mu"] = torch.tensor(ds["mu"], dtype=torch.float32, device=device)
    out["offsets"] = torch.tensor(
        common.offsets_for(mu=ds["mu"], edges=st.edges, table=st.table),
        dtype=torch.float32,
        device=device,
    )
    out["ceiling"] = torch.full((n,), st.ceiling, dtype=torch.float32, device=device)
    out["scale"] = torch.full((n,), st.scale, dtype=torch.float32, device=device)
    out["dni"] = torch.tensor(ex["dni"], dtype=torch.float32, device=device)
    out["dhi"] = torch.tensor(ex["dhi"], dtype=torch.float32, device=device)
    out["temperature"] = torch.tensor(ex["temperature"], dtype=torch.float32, device=device)
    out["zenith_rad"] = torch.tensor(np.radians(ex["zenith"]), dtype=torch.float32, device=device)
    out["sun_azimuth_rad"] = torch.tensor(
        np.radians(ex["azimuth"]), dtype=torch.float32, device=device
    )
    out["actual"] = torch.tensor(ds["output"], dtype=torch.float32, device=device)
    return out


def pinball(*, q: torch.Tensor, actual: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss over rows and the 11 levels."""
    error = actual[:, None] - q
    return torch.mean(torch.maximum(levels * error, (levels - 1.0) * error))


def build_model(*, arm: str, st: Setting) -> torch.nn.Module:  # noqa: PLR0911
    """Return the untrained network for an arm."""
    n_raw, n_head, n_blind = (
        st.train["raw"].shape[1],
        st.train["head"].shape[1],
        st.train["blind"].shape[1],
    )
    if arm == "nn_pure":
        return models.PureNetwork(n_raw)
    if arm == "h1_scale":
        return models.ScaleHead(n_head)
    if arm == "h2_residual":
        return models.ResidualHead(n_head)
    if arm == "h5_quantile":
        return models.QuantileHead(n_head)
    if arm == "h6_blind":
        return models.ResidualHead(n_blind)
    if arm == "h3_adapter":
        return models.Adapter(n_raw, with_scale_head=False, n_head_in=n_head)
    if arm == "h4_both":
        return models.Adapter(n_raw, with_scale_head=True, n_head_in=n_head)
    raise ValueError(arm)


def forward(*, arm: str, model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run a model, giving the blind arm its blind inputs as the head inputs."""
    if arm == "h6_blind":
        batch = {**batch, "head": batch["blind"]}
    return model(batch)


def train_neural(*, arm: str, st: Setting, seed: int) -> np.ndarray:
    """Fit one network and return the scored farm's (rows, 11) quantiles."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    levels = torch.tensor(common.LEVELS, dtype=torch.float32, device=device)
    train, val, test = (make_batch(st=st, split=s, device=device) for s in ("train", "val", "test"))
    model = build_model(arm=arm, st=st).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    n = len(train["mu"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(n, generator=generator).to(device)
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
            batch = {k: v[idx] for k, v in train.items()}
            loss = pinball(
                q=forward(arm=arm, model=model, batch=batch), actual=batch["actual"], levels=levels
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(
                pinball(
                    q=forward(arm=arm, model=model, batch=val), actual=val["actual"], levels=levels
                )
            )
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return forward(arm=arm, model=model, batch=test).cpu().numpy().astype(np.float64)


def plant_quantiles(*, st: Setting) -> np.ndarray:
    """Return the plant alone: forecast plus the tuning-month offsets, clamped to [0, ceiling]."""
    q = st.test["mu"][:, None] + common.offsets_for(
        mu=st.test["mu"], edges=st.edges, table=st.table
    )
    return np.minimum(np.maximum(np.sort(q, axis=1), 0.0), st.ceiling)


def xgboost_quantiles(*, st: Setting, setting: str) -> np.ndarray:
    """Fit XGBoost with 11 quantile outputs on the same training rows and return the scored quantiles."""

    def matrix(ds: dict[str, np.ndarray], raw_feats: np.ndarray) -> np.ndarray:
        return np.column_stack([raw_feats, ds["mu"]])

    train_x = np.concatenate([matrix(st.train, st.train["raw"]), matrix(st.val, st.val["raw"])])
    train_y = np.concatenate([st.train["output"], st.val["output"]])
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=common.LEVELS,
        tree_method="hist",
        n_estimators=150,
        learning_rate=0.15,
        max_depth=5,
        min_child_weight=10,
        subsample=0.8,
        n_jobs=1,
        random_state=243,
    )
    model.fit(train_x, train_y)
    predicted = np.asarray(model.predict(matrix(st.test, st.test["raw"]))).reshape(
        len(st.test["mu"]), -1
    )
    return np.maximum(np.sort(predicted, axis=1), 0.0)


def out_path(*, setting: str, site: str, arm: str, seed: int):  # noqa: ANN201
    """Return a job's output file."""
    return common.PRED_DIR / setting / f"{arm}_s{seed}_{site}.parquet"


def write(*, st: Setting, setting: str, arm: str, seed: int, quantiles: np.ndarray) -> None:
    """Write a job's scored rows atomically."""
    path = out_path(setting=setting, site=st.site, arm=arm, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = st.test_frame.select("time", "init_time", "lead_hours").with_columns(
        time_series_id=pl.lit(st.site),
        arm=pl.lit(arm),
        seed=pl.lit(seed),
        setting=pl.lit(setting),
        actual=pl.Series(st.test["output"]),
        **{
            level_column(level=lv): pl.Series(quantiles[:, i]) for i, lv in enumerate(common.LEVELS)
        },
    )
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def run_group(task: tuple[str, str]) -> str:
    """Run every arm and seed for one (setting, farm), skipping jobs whose file exists."""
    torch.set_num_threads(1)
    setting, site = task
    st = Setting(setting=setting, site=site)
    started = time.time()
    for arm, seed in [
        ("plant", 0),
        ("xgboost", 0),
        *itertools.product(common.NEURAL_ARMS, common.SEEDS),
    ]:
        if out_path(setting=setting, site=site, arm=arm, seed=seed).exists():
            continue
        if arm == "plant":
            q = plant_quantiles(st=st)
        elif arm == "xgboost":
            q = xgboost_quantiles(st=st, setting=setting)
        else:
            q = train_neural(arm=arm, st=st, seed=seed)
        write(st=st, setting=setting, arm=arm, seed=seed, quantiles=q)
        print(f"{setting} {site} {arm} s{seed} done {time.time() - started:.0f}s", flush=True)
    return f"{setting} {site}"


def main() -> None:
    """Run all (setting, farm) groups in parallel."""
    tasks = list(itertools.product(common.SETTINGS, common.FARMS))
    only = sys.argv[1:]
    if only:
        tasks = [t for t in tasks if t[1] in only]
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for done in pool.imap_unordered(run_group, tasks):
            print("group finished", done, flush=True)


if __name__ == "__main__":
    main()
