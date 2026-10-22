# ruff: noqa: E501
"""The learned scores: logistic regression, gradient-boosted tree, small neural network, re-ranker, stack.

All are fitted on injected worlds only (positives) and the real record's unexplained candidates in
even months (negatives, inside the same worlds). Network sizes, stopping epochs, and the stack are
chosen on separate validation worlds. Inputs are the columns of `FEATURE_NAMES`; no series identifier,
no Primary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00227_s_data import WORLD_DIR
from e00227_s_ronc import FEATURE_NAMES, RAW_LENGTH, pair_scores, ronc_scores
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn  # ty: ignore[unresolved-import]

N_SEEDS: Final[int] = 3
POS_WEIGHT_CAP: Final[float] = 10.0
Frame = tuple[pl.DataFrame, np.ndarray]


def load_worlds(*, fold: int, kind: str, count: int) -> Frame:
    """Concatenate a fold's worlds of one kind: candidate table (with a `world` column) and windows."""
    tables, windows = [], []
    for i in range(count):
        stem = f"f{fold}_{kind}_{i}"
        tables.append(pl.read_parquet(WORLD_DIR / f"{stem}.parquet").with_columns(world=pl.lit(i)))
        windows.append(np.load(WORLD_DIR / f"{stem}_windows.npy"))
    return pl.concat(tables), np.concatenate(windows)


def usable(*, frame: Frame, sources: tuple[str, ...]) -> Frame:
    """Candidates fit for training or selection: label 0 or 1, even-month, of the given sources."""
    table, windows = frame
    keep = (
        (table["label"] >= 0).to_numpy()
        & ~table["odd"].to_numpy()
        & table["source"].is_in(list(sources)).to_numpy()
    )
    return table.filter(pl.Series(keep)), windows[keep]


def matrix(*, table: pl.DataFrame) -> np.ndarray:
    """The feature matrix."""
    return np.nan_to_num(table.select(FEATURE_NAMES).to_numpy().astype(np.float32))


@dataclass
class Standard:
    """Mean and standard deviation of the training features."""

    mean: np.ndarray
    std: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Standardise `x`."""
        return ((x - self.mean) / self.std).astype(np.float32)


def fit_standard(*, x: np.ndarray) -> Standard:
    """Standardiser fitted on `x`."""
    return Standard(x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6))


class LogisticScore:
    """Logistic regression on the features."""

    def __init__(self, *, train: Frame) -> None:
        """Fit with balanced class weights."""
        x = matrix(table=train[0])
        self.standard = fit_standard(x=x)
        self.model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        self.model.fit(self.standard(x), train[0]["label"].to_numpy())

    def __call__(self, frame: Frame) -> np.ndarray:
        """Log-odds."""
        return self.model.decision_function(self.standard(matrix(table=frame[0])))


class TreeScore:
    """Gradient-boosted tree: depth and rounds chosen on validation log-loss."""

    def __init__(self, *, train: Frame, valid: Frame) -> None:
        """Fit a small grid and keep the best by validation log-loss."""
        x, y = matrix(table=train[0]), train[0]["label"].to_numpy()
        xv, yv = matrix(table=valid[0]), valid[0]["label"].to_numpy()
        best = np.inf
        weight = np.where(y == 1, min(POS_WEIGHT_CAP, (y == 0).sum() / max((y == 1).sum(), 1)), 1.0)
        for depth in (2, 3, 4):
            for rounds in (100, 300):
                model = HistGradientBoostingClassifier(
                    max_depth=depth,
                    max_iter=rounds,
                    learning_rate=0.05,
                    l2_regularization=1.0,
                    random_state=0,
                )
                model.fit(x, y, sample_weight=weight)
                p = np.clip(model.predict_proba(xv)[:, 1], 1e-6, 1 - 1e-6)
                loss = float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))
                if loss < best:
                    best, self.model, self.config = loss, model, (depth, rounds)

    def __call__(self, frame: Frame) -> np.ndarray:
        """Log-odds."""
        p = np.clip(self.model.predict_proba(matrix(table=frame[0]))[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))


class FeatureNet(nn.Module):
    """A small multilayer perceptron on the features (about 2,000 weights)."""

    def __init__(self, *, n_in: int) -> None:
        """Two hidden layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 48), nn.GELU(), nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 1)
        )

    def forward(self, x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
        """Logit."""
        return self.net(x).squeeze(-1)


class RerankNet(nn.Module):
    """Features plus a one-dimensional convolution over the raw residual window (about 10,000 weights)."""

    def __init__(self, *, n_in: int) -> None:
        """Three strided convolutions, pooled, joined to a feature branch."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 16, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 24, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(24, 32, 5, stride=2, padding=2),
            nn.GELU(),
        )
        self.features = nn.Sequential(nn.Linear(n_in, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(64 + 32, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Logit."""
        h = self.conv(w)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.head(torch.cat([pooled, self.features(x)], dim=1)).squeeze(-1)


def n_parameters(*, model: nn.Module) -> int:
    """Count learned weights."""
    return sum(p.numel() for p in model.parameters())


class NetScore:
    """A network trained with early stopping on validation log-loss, averaged over seeds."""

    def __init__(
        self, *, train: Frame, valid: Frame, kind: str, epochs: int = 50, threads: int = 4
    ) -> None:
        """Train `N_SEEDS` networks (`kind` is `features` or `rerank`)."""
        torch.set_num_threads(threads)
        self.kind = kind
        x = matrix(table=train[0])
        self.standard = fit_standard(x=x)
        xt, yt = (
            torch.tensor(self.standard(x)),
            torch.tensor(train[0]["label"].to_numpy(), dtype=torch.float32),
        )
        xv = torch.tensor(self.standard(matrix(table=valid[0])))
        yv = torch.tensor(valid[0]["label"].to_numpy(), dtype=torch.float32)
        wt = torch.tensor(train[1].astype(np.float32)) if kind == "rerank" else None
        wv = torch.tensor(valid[1].astype(np.float32)) if kind == "rerank" else None
        pos_weight = torch.tensor(
            min(POS_WEIGHT_CAP, float((yt == 0).sum() / max((yt == 1).sum(), 1)))
        )
        self.models: list[nn.Module] = []
        self.history: list[dict] = []
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            model: nn.Module = (
                RerankNet(n_in=x.shape[1]) if kind == "rerank" else FeatureNet(n_in=x.shape[1])
            )
            optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            best, best_state, bad = np.inf, None, 0
            for epoch in range(epochs):  # noqa: B007
                model.train()
                order = torch.randperm(len(yt))
                for i in range(0, len(yt), 256):
                    idx = order[i : i + 256]
                    out = model(xt[idx], wt[idx] if wt is not None else None)
                    loss = loss_fn(out, yt[idx])
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                model.eval()
                with torch.no_grad():
                    logits = torch.cat(
                        [
                            model(xv[j : j + 2048], wv[j : j + 2048] if wv is not None else None)
                            for j in range(0, len(yv), 2048)
                        ]
                    )
                    vloss = float(loss_fn(logits, yv))
                if vloss < best - 1e-4:
                    best, bad = vloss, 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 6:
                        break
            assert best_state is not None
            model.load_state_dict(best_state)
            model.eval()
            self.models.append(model)
            self.history.append({"seed": seed, "epochs": epoch + 1, "valid_loss": best})
        self.n_parameters = n_parameters(model=self.models[0])

    def __call__(self, frame: Frame) -> np.ndarray:
        """Mean logit over seeds."""
        x = torch.tensor(self.standard(matrix(table=frame[0])))
        w = torch.tensor(frame[1].astype(np.float32)) if self.kind == "rerank" else None
        outs = []
        with torch.no_grad():
            for model in self.models:
                outs.append(  # noqa: PERF401
                    torch.cat(
                        [
                            model(x[j : j + 2048], w[j : j + 2048] if w is not None else None)
                            for j in range(0, len(x), 2048)
                        ]
                    ).numpy()
                )
        return np.mean(outs, axis=0)


def stack_matrix(*, frame: Frame, net_logit: np.ndarray) -> np.ndarray:
    """Columns for the stacked score: the network's logit, RO-NC's log-odds, the pair statistic."""
    table = frame[0]
    ronc = ronc_scores(table=table)
    finite = np.isfinite(ronc)
    filled = np.where(finite, ronc, -4.0)
    return np.column_stack(
        [net_logit, filled, (~finite).astype(float), np.clip(pair_scores(table=table), -5, 15)]
    )


class StackScore:
    """Logistic combination of the network's probability and the detectors' statistics (validation worlds only)."""

    def __init__(self, *, net: NetScore, valid: Frame) -> None:
        """Fit on validation worlds, which neither the network nor the detectors saw."""
        self.net = net
        x = stack_matrix(frame=valid, net_logit=net(valid))
        self.mean, self.std = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
        self.model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        self.model.fit((x - self.mean) / self.std, valid[0]["label"].to_numpy())
        self.coef = self.model.coef_.ravel().tolist()

    def __call__(self, frame: Frame) -> np.ndarray:
        """Log-odds of the stack."""
        x = stack_matrix(frame=frame, net_logit=self.net(frame))
        return self.model.decision_function((x - self.mean) / self.std)


assert RAW_LENGTH == 288
