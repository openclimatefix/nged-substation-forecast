"""Mix the ECMWF and AIFS member mixtures and climatology, then apply the spread factor and table.

Every arm follows the same steps on the rows of experiment 00259's scored set that have both
ensembles' forecasts (`e00262_members.py`):

1. **Weights.** A source is a set of atoms (the 199 quantiles of an ensemble's mixture, or the 11
   quantiles of climatology). The pooled forecast is the linear pool of the sources' atoms with one
   weight per source and lead day, fitted monthly on earlier cases (valid time at least 3.5 hours
   before the month's first run) by the continuous ranked probability score. The quantile-averaged
   arms (M1q) fit one weight per lead day on the quantile score. The neural gates read per-case
   inputs instead of the lead day.
2. **Spread factor.** Experiment 00259's `apply_spread`, fitted month by month on earlier cases of
   the arm's own pooled forecast.
3. **Offset table.** Experiment 00259's recipe (the quantile of observed minus forecast, per lead
   day, level, and farm), refitted monthly on the arm's earlier spread-adjusted forecasts.

July to September 2025 rows share the weights, spread factors, and tables fitted at the first
scored month (October 2025), so every scored month's inputs are fitted on earlier cases only.

Output, in `work/arms/`: `<arm>.parquet` and `<arm>+T.parquet` (without and with the table), with
key columns, `seed`, and `q01` to `q99`; and `weights.json`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00262-*/e00262_mix.py
"""

import itertools
import json
import os
from datetime import UTC, datetime
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00262_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
MEMBER_DIR: Final = common.WORK_DIR / "members"
OUT_DIR: Final = common.WORK_DIR / "arms"
STRIDE: Final[int] = 4
"""Every fourth of the 199 fine levels (50 atoms) enters the weight-fitting statistics."""
WEIGHT_STEP: Final[float] = 0.05
FIRST_SCORED_MONTH: Final[str] = "2025-10"
END_OF_SCORING: Final = datetime(2026, 7, 1, tzinfo=UTC)
"""No experiment scores a valid time on or after this date."""
MIN_TABLE_ROWS: Final[int] = 200
GATE_HIDDEN: Final[int] = 16
GATE_EPOCHS: Final[int] = 200
GATE_LEARNING_RATE: Final[float] = 0.02
GATE_WEIGHT_DECAY: Final[float] = 1e-3
SOURCE_NAMES: Final[tuple[str, ...]] = ("ens", "aifs", "clim")
LEAD_DAYS: Final[range] = range(16)
CHUNK: Final[int] = 2_000


def grid(*, free: tuple[bool, ...]) -> np.ndarray:
    """Return every weight vector on `WEIGHT_STEP` steps that sums to 1 over the `free` sources."""
    steps = round(1 / WEIGHT_STEP)
    count = sum(free)
    points = [c for c in itertools.product(range(steps + 1), repeat=count) if sum(c) == steps]
    out = np.zeros((len(points), len(free)))
    out[:, np.flatnonzero(free)] = np.array(points) / steps
    return out


def levels_from_fine(*, fine: np.ndarray) -> np.ndarray:
    """Return the 11 `LEVELS` of a quantile function stored at the 199 fine levels."""
    position = common.LEVELS * 200 - 1
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, common.FINE_LEVELS.size - 1)
    fraction = (position - low).astype(np.float32)
    return fine[:, low] * (1 - fraction) + fine[:, high] * fraction


def pool_quantiles(
    *, atoms: list[np.ndarray], masses: list[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    """Return the 11 quantiles of each case's linear pool of the sources' atoms.

    Args:
        atoms: Per source, shape (cases, atoms) values.
        masses: Per source, shape (atoms,) probability masses that sum to 1.
        weights: Shape (cases, sources) mixture weights; each row sums to 1.

    Returns:
        Shape (cases, 11) float32 quantiles, by linear interpolation between atom mid-points as in
        `members.mixture_fine`.
    """
    cases = weights.shape[0]
    out = np.empty((cases, len(common.LEVELS)), dtype=np.float32)
    for start in range(0, cases, CHUNK):
        sl = slice(start, start + CHUNK)
        values = np.concatenate([a[sl] for a in atoms], axis=1)
        mass = np.concatenate(
            [weights[sl, s : s + 1] * masses[s][None, :] for s in range(len(atoms))], axis=1
        )
        order = np.argsort(values, axis=1, kind="stable")
        values = np.take_along_axis(values, order, axis=1)
        mass = np.take_along_axis(mass, order, axis=1)
        mid = np.cumsum(mass, axis=1) - mass / 2
        width = values.shape[1]
        for k, level in enumerate(common.LEVELS):
            j = np.clip((mid < level).sum(axis=1), 1, width - 1)
            row = np.arange(values.shape[0])
            low_mid, high_mid = mid[row, j - 1], mid[row, j]
            low_val, high_val = values[row, j - 1], values[row, j]
            gap = high_mid - low_mid
            fraction = np.where(
                gap > 0, np.clip((level - low_mid) / np.where(gap > 0, gap, 1), 0, 1), 1.0
            )
            out[start : start + values.shape[0], k] = low_val + fraction * (high_val - low_val)
    return np.sort(np.clip(out, 0.0, 1.0), axis=1)


def crps_statistics(
    *, atoms: list[np.ndarray], masses: list[np.ndarray], actual: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-case `a[s] = sum_k m_k |x_sk - y|` and the pairwise `D[s, t]` of the CRPS.

    `D[s, t]` is `sum_kl m_k m_l |x_sk - x_tl|`.
    """
    cases, sources = actual.size, len(atoms)
    a = np.empty((cases, sources))
    d = np.empty((cases, sources, sources))
    for start in range(0, cases, CHUNK):
        sl = slice(start, start + CHUNK)
        for s in range(sources):
            a[sl, s] = (np.abs(atoms[s][sl] - actual[sl, None]) * masses[s]).sum(axis=1)
            for t in range(s, sources):
                gap = np.abs(atoms[s][sl, :, None] - atoms[t][sl, None, :])
                d[sl, s, t] = d[sl, t, s] = np.einsum("nkl,k,l->n", gap, masses[s], masses[t])
    return a, d


def crps_of_weights(*, a: np.ndarray, d: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the CRPS of the pool at each case: `w . a - 0.5 w' D w`."""
    return (weights * a).sum(axis=1) - 0.5 * np.einsum("ns,nst,nt->n", weights, d, weights)


def best_weights(
    *, a: np.ndarray, d: np.ndarray, free: tuple[bool, ...], rows: np.ndarray
) -> np.ndarray:
    """Return the grid weight vector that minimises the mean CRPS over `rows`."""
    candidates = grid(free=free)
    mean_a, mean_d = a[rows].mean(axis=0), d[rows].mean(axis=0)
    loss = candidates @ mean_a - 0.5 * np.einsum("cs,st,ct->c", candidates, mean_d, candidates)
    return candidates[int(np.argmin(loss))]


class Frame:
    """The rows every arm scores, with the sources' atoms for one seed."""

    def __init__(self, *, seed: int, sources: tuple[str, ...] = ("ens", "aifs")) -> None:
        """Load the rows, atoms, and per-case statistics for `seed`."""
        sites = common.site_table()
        cases = common.cases_table(sites=sites).filter(
            pl.col("actual_norm").is_finite() & (pl.col("valid_time") < END_OF_SCORING)
        )
        keys = {s: pl.read_parquet(MEMBER_DIR / f"{s}_keys.parquet") for s in sources}
        master = cases
        for source_keys in keys.values():
            master = master.join(source_keys, on=list(common.KEY_COLUMNS), how="semi")
        self.master = master.sort(*common.KEY_COLUMNS)
        self.seed = seed
        fine_columns = [f"f{k:03d}" for k in range(1, 200)]
        self.fine: dict[str, np.ndarray] = {}
        self.summary: dict[str, pl.DataFrame] = {}
        for source in sources:
            raw = pl.read_parquet(MEMBER_DIR / f"{source}_seed{seed}.parquet")
            index = (
                self.master.select(common.KEY_COLUMNS)
                .join(
                    raw.select(common.KEY_COLUMNS).with_row_index("i"),
                    on=list(common.KEY_COLUMNS),
                    how="left",
                    validate="1:1",
                )["i"]
                .to_numpy()
            )
            self.fine[source] = raw.select(fine_columns).to_numpy()[index]
            self.summary[source] = (
                self.master.select(common.KEY_COLUMNS)
                .join(
                    pl.read_parquet(MEMBER_DIR / f"{source}_summary.parquet"),
                    on=list(common.KEY_COLUMNS),
                    how="left",
                    validate="1:1",
                )
                .select("ws_mean", "ws_std")
            )
        clim = (
            pl.read_parquet(
                common.ML_EXPERIMENTS_DATA_DIR
                / "00230-wind-physics-core-part-1"
                / "work"
                / "arms"
                / "CLIM_spread.parquet"
            )
            .filter(pl.col("seed") == 0)
            .select(*common.KEY_COLUMNS, *common.LEVEL_COLUMNS)
        )
        joined = self.master.select(common.KEY_COLUMNS).join(
            clim, on=list(common.KEY_COLUMNS), how="left", validate="1:1"
        )
        self.clim = joined.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float32)
        if np.isnan(self.clim).any():
            msg = "climatology is missing for some scored rows"
            raise ValueError(msg)
        self.actual = self.master["actual_norm"].to_numpy().astype(np.float64)
        self.lead_day = self.master["lead_day"].to_numpy()
        self.valid = self.master["valid_time"].to_numpy()
        self.init = self.master["power_fcst_init_time"].to_numpy()
        self.month = self.master["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
        self.sources = sources
        self.atoms_full = [self.fine[s] for s in sources]
        self.masses_full = [np.full(199, 1 / 199) for _ in sources]
        self.atoms_stat = [self.fine[s][:, STRIDE - 1 :: STRIDE] for s in sources]
        self.masses_stat = [np.full(a.shape[1], 1 / a.shape[1]) for a in self.atoms_stat]
        self.atoms_clim = self.clim.astype(np.float64)
        self.refit_months = sorted(set(self.month))
        self.key = np.array([self.refit_key(m) for m in self.month])

    def refit_key(self, month: str) -> str:
        """Return the month whose fit a row of `month` uses."""
        return max(month, FIRST_SCORED_MONTH)

    def train_rows(self, key: str) -> np.ndarray:
        """Return indices of the cases known 3.5 hours before the first run of month `key`."""
        cutoff = self.init[self.month == key].min() - common.TELEMETRY_DELAY
        return np.flatnonzero(self.valid <= cutoff)


def stat_arrays(*, data: Frame, with_clim: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return the CRPS statistics of the ensembles (and climatology) for `data`."""
    atoms = [*data.atoms_stat, data.atoms_clim] if with_clim else list(data.atoms_stat)
    masses = [*data.masses_stat, common.ATOM_MASSES] if with_clim else list(data.masses_stat)
    return crps_statistics(atoms=atoms, masses=masses, actual=data.actual)


def lead_day_weights(
    *, data: Frame, a: np.ndarray, d: np.ndarray, free: tuple[bool, ...]
) -> tuple[np.ndarray, dict]:
    """Return per-row weights from per-lead-day weights refitted monthly, and the fitted table."""
    weights = np.zeros((data.actual.size, len(free)))
    table: dict = {}
    for key in sorted(set(data.key)):
        train = data.train_rows(key)
        rows_of_key = np.flatnonzero(data.key == key)
        pooled = best_weights(a=a, d=d, free=free, rows=train)
        for day in LEAD_DAYS:
            day_train = train[data.lead_day[train] == day]
            fitted = (
                best_weights(a=a, d=d, free=free, rows=day_train)
                if day_train.size >= MIN_TABLE_ROWS
                else pooled
            )
            table[f"{key} day {day}"] = fitted.tolist()
            target = rows_of_key[data.lead_day[rows_of_key] == day]
            weights[target] = fitted
    return weights, table


def quantile_average_weights(*, data: Frame, free_clim: bool) -> tuple[np.ndarray, dict]:
    """Return per-row climatology weights for M1q, by the lowest quantile score on the grid."""
    q_ens = levels_from_fine(fine=data.fine["ens"])
    candidates = np.arange(0, 1 + 1e-9, WEIGHT_STEP)
    weights = np.zeros(data.actual.size)
    table: dict = {}
    for key in sorted(set(data.key)):
        train = data.train_rows(key)
        rows_of_key = np.flatnonzero(data.key == key)
        for day in LEAD_DAYS:
            day_train = train[data.lead_day[train] == day]
            if day_train.size < MIN_TABLE_ROWS:
                day_train = train
            losses = [
                common.quantile_score(
                    quantiles=(1 - w) * q_ens[day_train] + w * data.clim[day_train],
                    actual=data.actual[day_train],
                ).mean()
                for w in candidates
            ]
            best = float(candidates[int(np.argmin(losses))])
            table[f"{key} day {day}"] = best
            target = rows_of_key[data.lead_day[rows_of_key] == day]
            weights[target] = best
    return weights, table


class Gate(torch.nn.Module):
    """One hidden layer of at most 16 units and a softmax over the sources."""

    def __init__(self, *, inputs: int, sources: int) -> None:
        """Set the layer sizes."""
        super().__init__()
        self.hidden = torch.nn.Linear(inputs, GATE_HIDDEN)
        self.out = torch.nn.Linear(GATE_HIDDEN, sources)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return mixture weights, shape (rows, sources)."""
        return torch.softmax(self.out(torch.tanh(self.hidden(x))), dim=1)


def gate_inputs(*, data: Frame, blind: bool) -> np.ndarray:
    """Return the gate's inputs: lead alone (blind), or lead, both spreads, and the mean gap."""
    lead = data.master["lead_hours"].to_numpy().astype(np.float64) / 100.0
    if blind:
        return lead[:, None]
    ens, aifs = data.summary["ens"], data.summary["aifs"]
    return np.column_stack(
        [
            lead,
            ens["ws_std"].to_numpy(),
            aifs["ws_std"].to_numpy(),
            ens["ws_mean"].to_numpy() - aifs["ws_mean"].to_numpy(),
        ]
    ).astype(np.float64)


def gate_weights(*, data: Frame, a: np.ndarray, d: np.ndarray, blind: bool) -> np.ndarray:
    """Return per-row weights from a gate refitted monthly on earlier cases."""
    x = gate_inputs(data=data, blind=blind)
    weights = np.zeros((data.actual.size, a.shape[1]))
    torch.manual_seed(data.seed)
    for key in sorted(set(data.key)):
        train = data.train_rows(key)
        mean, std = x[train].mean(axis=0), x[train].std(axis=0) + 1e-9
        tx = torch.tensor((x - mean) / std, dtype=torch.float32)
        ta, td = (torch.tensor(v, dtype=torch.float32) for v in (a, d))
        gate = Gate(inputs=x.shape[1], sources=a.shape[1])
        optimiser = torch.optim.Adam(
            gate.parameters(), lr=GATE_LEARNING_RATE, weight_decay=GATE_WEIGHT_DECAY
        )
        train_index = torch.tensor(train)
        for _ in range(GATE_EPOCHS):
            optimiser.zero_grad()
            w = gate(tx[train_index])
            loss = (
                (w * ta[train_index]).sum(1)
                - 0.5 * torch.einsum("ns,nst,nt->n", w, td[train_index], w)
            ).mean()
            loss.backward()
            optimiser.step()
        rows_of_key = np.flatnonzero(data.key == key)
        with torch.no_grad():
            weights[rows_of_key] = gate(tx[torch.tensor(rows_of_key)]).numpy()
    return weights


def offset_table(*, data: Frame, quantiles: np.ndarray) -> np.ndarray:
    """Return `quantiles` plus offsets fitted monthly on earlier spread-adjusted rows."""
    out = quantiles.copy()
    sites = data.master["site"].to_numpy()
    for key in sorted(set(data.key)):
        train = data.train_rows(key)
        rows_of_key = np.flatnonzero(data.key == key)
        residual = data.actual[:, None] - quantiles
        for site in np.unique(sites):
            for day in LEAD_DAYS:
                cell = train[(sites[train] == site) & (data.lead_day[train] == day)]
                target = rows_of_key[
                    (sites[rows_of_key] == site) & (data.lead_day[rows_of_key] == day)
                ]
                if target.size == 0:
                    continue
                offsets = (
                    np.array(
                        [
                            np.quantile(residual[cell, k], level)
                            for k, level in enumerate(common.LEVELS)
                        ]
                    )
                    if cell.size >= MIN_TABLE_ROWS
                    else np.zeros(len(common.LEVELS))
                )
                out[target] = np.clip(np.sort(quantiles[target] + offsets, axis=1), 0.0, 1.0)
    return out


def spread(*, data: Frame, quantiles: np.ndarray) -> np.ndarray:
    """Return `quantiles` after experiment 00259's monthly spread factor."""
    cases = data.master.select("power_fcst_init_time", "valid_time", "lead_hours", "actual_norm")
    return common.apply_spread(cases=cases, mixture=quantiles.astype(np.float64)).astype(np.float32)


def hindcast_table(*, seed: int) -> pl.DataFrame:
    """Return experiment 00259's table recipe fitted on its stored A0r (R0) hindcast forecasts."""
    prior = common.WORK_DIR.parent.parent / "00259-wind-plant-plus-quantile-offset-head" / "work"
    forecasts = pl.read_parquet(prior / "base" / f"R0_hindcast_seed{seed}.parquet")
    summary = pl.read_parquet(prior / "base" / "summaries_hindcast.parquet").select(
        *common.KEY_COLUMNS, "site", "actual_norm", "lead_day"
    )
    joined = forecasts.join(summary, on=list(common.KEY_COLUMNS), validate="1:1")
    residual = joined.select(
        "site",
        "lead_day",
        *[(pl.col("actual_norm") - pl.col(q)).alias(q) for q in common.LEVEL_COLUMNS],
    )
    return residual.group_by("site", "lead_day").agg(
        [
            pl.col(q).quantile(level, interpolation="linear")
            for q, level in zip(common.LEVEL_COLUMNS, common.LEVELS, strict=True)
        ]
    )


def apply_hindcast_table(*, data: Frame, quantiles: np.ndarray, table: pl.DataFrame) -> np.ndarray:
    """Return `quantiles` plus the hindcast table's offsets (zero where a cell is absent)."""
    joined = data.master.select("site", "lead_day").join(
        table, on=["site", "lead_day"], how="left", maintain_order="left"
    )
    offsets = joined.select([pl.col(q).fill_null(0.0) for q in common.LEVEL_COLUMNS]).to_numpy()
    return np.clip(np.sort(quantiles + offsets, axis=1), 0.0, 1.0)


def main() -> None:
    """Write every arm, with and without the table, for every seed."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights_log: dict = {}
    collected: dict[str, list[pl.DataFrame]] = {}
    for seed in SEEDS:
        data = Frame(seed=seed)
        a3, d3 = stat_arrays(data=data, with_clim=True)
        ens_only = [data.atoms_full[0]]
        raw: dict[str, np.ndarray] = {}

        def pooled(*, w: np.ndarray, with_clim: bool, data: Frame = data) -> np.ndarray:
            atoms = [*data.atoms_full, data.atoms_clim] if with_clim else list(data.atoms_full)
            masses = (
                [*data.masses_full, common.ATOM_MASSES] if with_clim else list(data.masses_full)
            )
            return pool_quantiles(atoms=atoms, masses=masses, weights=w)

        n = data.actual.size
        one = np.ones((n, 1))
        raw["M0"] = pool_quantiles(atoms=ens_only, masses=[data.masses_full[0]], weights=one)
        raw["M2"] = pool_quantiles(
            atoms=[data.atoms_full[1]], masses=[data.masses_full[1]], weights=one
        )
        # M1p: ensemble and climatology; the AIFS column of the 3-source statistics is not free.
        w, table = lead_day_weights(data=data, a=a3, d=d3, free=(True, False, True))
        raw["M1p"] = pooled(w=w, with_clim=True)
        weights_log[f"seed{seed}_M1p"] = table
        wq, table = quantile_average_weights(data=data, free_clim=True)
        q_ens = levels_from_fine(fine=data.fine["ens"])
        raw["M1q"] = np.sort((1 - wq[:, None]) * q_ens + wq[:, None] * data.clim, axis=1).astype(
            np.float32
        )
        weights_log[f"seed{seed}_M1q"] = table
        w, table = lead_day_weights(data=data, a=a3, d=d3, free=(True, True, False))
        raw["M3"] = pooled(w=w, with_clim=True)
        weights_log[f"seed{seed}_M3"] = table
        w, table = lead_day_weights(data=data, a=a3, d=d3, free=(True, True, True))
        raw["M4"] = pooled(w=w, with_clim=True)
        weights_log[f"seed{seed}_M4"] = table
        for name, blind in (("M5", False), ("M5b", True)):
            w = gate_weights(data=data, a=a3, d=d3, blind=blind)
            raw[name] = pooled(w=w, with_clim=True)
            weights_log[f"seed{seed}_{name}_mean_weights"] = w.mean(axis=0).tolist()
        for name, quantiles in raw.items():
            adjusted = spread(data=data, quantiles=quantiles)
            tabled = offset_table(data=data, quantiles=adjusted)
            frames = {
                name: adjusted,
                f"{name}+T": tabled,
            }
            if name == "M0":
                frames["M0+Thindcast"] = apply_hindcast_table(
                    data=data, quantiles=adjusted, table=hindcast_table(seed=seed)
                )
            for label, q in frames.items():
                frame = (
                    data.master.select(*common.KEY_COLUMNS)
                    .hstack(pl.DataFrame(q.astype(np.float32), schema=list(common.LEVEL_COLUMNS)))
                    .with_columns(seed=pl.lit(seed, dtype=pl.Int32))
                )
                collected.setdefault(label, []).append(frame)
        previous = Frame(seed=seed, sources=("ens", "aifs_prev"))
        a_prev, d_prev = stat_arrays(data=previous, with_clim=True)
        w, table = lead_day_weights(data=previous, a=a_prev, d=d_prev, free=(True, True, False))
        weights_log[f"seed{seed}_M3prev"] = table
        quantiles = pool_quantiles(
            atoms=[*previous.atoms_full, previous.atoms_clim],
            masses=[*previous.masses_full, common.ATOM_MASSES],
            weights=w,
        )
        adjusted = spread(data=previous, quantiles=quantiles)
        for label, q in (
            ("M3prev", adjusted),
            ("M3prev+T", offset_table(data=previous, quantiles=adjusted)),
        ):
            collected.setdefault(label, []).append(
                previous.master.select(*common.KEY_COLUMNS)
                .hstack(pl.DataFrame(q.astype(np.float32), schema=list(common.LEVEL_COLUMNS)))
                .with_columns(seed=pl.lit(seed, dtype=pl.Int32))
            )
        print(f"seed {seed} done", flush=True)
    for label, frames in collected.items():
        pl.concat(frames).write_parquet(OUT_DIR / f"{label}.parquet")
    (OUT_DIR / "weights.json").write_text(json.dumps(weights_log))


if __name__ == "__main__":
    main()
