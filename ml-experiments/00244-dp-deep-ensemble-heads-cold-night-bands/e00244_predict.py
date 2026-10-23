"""Experiment 244: predict with every trained network and write the evaluation rows.

Writes under `work/`:

- `rows.parquet`: one row per (substation, run, valid time) with experiment 169 arm A's quantiles
  (E0), arm D's median and offsets, the actual, hour band, the ERA5 cold-day flag, and the head set
  that is out of sample for the row (`full` for the scored year, `cfx` and `cfy` for the replicate
  year). Only rows with lead 15.5 to 39 hours outside the hand-filtered switching events.
- `preds_<set>_<kind>_<scheme>.npz`: median corrections `c` and log-scales `s`, shape
  (20, rows of that set), and for the `full` set the same on the gate's injected rows.

    uv run --no-sync python e00244_predict.py
"""

import json
from typing import Final

import e00244_data as data_module
import e00244_fit as fit_module
import e00244_model as model_module
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR = data_module.WORK_DIR
E169_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00169-calibration-combined-fixes-and-ensemble-spread" / "work"
)
CACHE_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
PERIODS: Final = ("tuning", "replicate", "scored")
INJECT_FRACTION: Final[float] = 0.02
INJECT_SEED: Final[int] = 244
INJECT_C: Final[float] = 4.0
COLD_C: Final[float] = 2.0
WINTER: Final = (11, 12, 1, 2, 3)
NIGHT_BANDS: Final = (0, 3)
"""Band indices of `night` and `evening` in `studies.dp_calendar.BAND_NAMES`."""


def daily_mean(*, temperature: np.ndarray, end_times: np.ndarray) -> np.ndarray:
    """ERA5 temperature's mean over each half-hour's UTC date, on the grid."""
    days = end_times.astype("datetime64[D]")
    codes = np.unique(days, return_inverse=True)[1]
    t = temperature.astype(np.float64)
    finite = np.isfinite(t)
    total = np.bincount(codes, weights=np.where(finite, t, 0.0))
    count = np.bincount(codes, weights=finite.astype(np.float64))
    return (total / np.maximum(count, 1))[codes]


def head_set_of(*, init: np.ndarray) -> np.ndarray:
    """The head set that is out of sample for each run origin."""
    out = np.full(len(init), "", dtype="U4")
    out[init >= fit_module.SCORED_START] = "full"
    for name, (low, high) in fit_module.HOLD.items():
        out[(init >= low) & (init < high)] = name
    return out


def build_rows() -> tuple[pl.DataFrame, dict]:
    """Evaluation rows with their index into each substation's feature arrays."""
    state = fit_module.load_state()
    feats, labels = state["feats"], state["labels"]
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)
    pieces, slopes = [], {}
    for k, (f, label) in enumerate(zip(feats, labels, strict=True)):
        features = cache.sites[label]
        index = pl.DataFrame(
            {
                "idx": np.arange(len(f["init"])),
                "init": f["init"],
                "valid": f["valid"],
            }
        )
        frames = []
        for period in PERIODS:
            frame = pl.read_parquet(E169_DIR / "dp_preds" / f"a_{period}_{label}.parquet")
            frames.append(frame.with_columns(period=pl.lit(period)))
        arm_a = (
            pl.concat(frames)
            .filter(pl.col("seed") == 0)
            .filter(~pl.col("switching_mask") & pl.col("actual").is_finite())
            .filter(pl.col("lead_hours").is_between(15.5, 39.0))
            .with_columns(
                init=pl.col("power_fcst_init_time")
                .dt.replace_time_zone(None)
                .cast(pl.Datetime("us")),
                valid=pl.col("valid_time").dt.replace_time_zone(None).cast(pl.Datetime("us")),
            )
            .unique(subset=["init", "valid"])
        )
        joined = arm_a.join(index, on=["init", "valid"], how="inner").with_columns(
            site_index=pl.lit(k), site=pl.lit(label)
        )
        target = np.searchsorted(cache.end_times, joined["valid"].to_numpy())
        daily = daily_mean(temperature=features.temperature_c, end_times=cache.end_times)
        joined = joined.with_columns(
            cold2=pl.Series(np.nan_to_num(daily[target], nan=99.0) < COLD_C),
            era5_c=pl.Series(features.temperature_c[target].astype(np.float64)),
        )
        pieces.append(joined)
        # Cold-night slope of arm D's forecast level on ERA5 temperature, training months only.
        train = f["init"] < fit_module.TRAIN_END
        t_all = np.searchsorted(cache.end_times, f["valid"])
        t_all = np.clip(t_all, 0, len(cache.end_times) - 1)
        temp = features.temperature_c[t_all].astype(np.float64)
        month = f["valid"].astype("datetime64[M]").astype(int) % 12 + 1
        band = cache.calendar.band[t_all]
        sel = train & np.isin(month, WINTER) & np.isin(band, NIGHT_BANDS) & np.isfinite(temp)
        slopes[label] = float(np.polyfit(temp[sel], f["mean"][sel].astype(np.float64), 1)[0])
    rows = pl.concat(pieces, how="diagonal_relaxed")
    init = rows["init"].to_numpy()
    rows = rows.with_columns(
        head_set=pl.Series(head_set_of(init=init)),
        month=pl.col("valid").dt.month(),
        day=pl.col("valid").dt.date(),
        row_id=pl.int_range(pl.len()),
    )
    return rows, slopes


def choose_injection(*, rows: pl.DataFrame) -> np.ndarray:
    """Row ids of 2% of the scored year's cold-night rows, drawn with a fixed seed."""
    cold = rows.filter(
        (pl.col("head_set") == "full")
        & pl.col("cold2")
        & pl.col("month").is_in(list(WINTER))
        & pl.col("hour_band").is_in(list(NIGHT_BANDS))
    )["row_id"].to_numpy()
    rng = np.random.default_rng(INJECT_SEED)
    return np.sort(rng.choice(cold, size=max(1, round(INJECT_FRACTION * len(cold))), replace=False))


def predict_all() -> None:
    """Write rows and per-combination predictions."""
    state = fit_module.load_state()
    feats = state["feats"]
    rows, slopes = build_rows()
    injected = choose_injection(rows=rows)
    rows = rows.with_columns(injected=pl.col("row_id").is_in(injected.tolist()))
    arm_d_q50 = np.empty(rows.height, dtype=np.float32)
    offsets = np.empty((rows.height, 11), dtype=np.float32)
    site_idx = rows["site_index"].to_numpy()
    idx = rows["idx"].to_numpy()
    for k, f in enumerate(feats):
        m = site_idx == k
        arm_d_q50[m] = f["q50"][idx[m]]
        offsets[m] = f["q_offsets"][idx[m]]
    rows = rows.with_columns(
        d_q50=pl.Series(arm_d_q50),
        **{f"d_off_{q}": pl.Series(offsets[:, i]) for i, q in enumerate(data_module.QUANTILES)},
    )
    rows.drop(["idx"]).write_parquet(WORK_DIR / "rows.parquet")
    (WORK_DIR / "slopes.json").write_text(json.dumps(slopes, indent=2))
    labels = state["labels"]
    shift = np.array([INJECT_C * slopes[labels[k]] for k in site_idx])
    sets = rows["head_set"].to_numpy()
    periods = rows["period"].to_numpy()
    groups = {
        "full": ("full", sets == "full"),
        "fullin": ("full", periods == "tuning"),
        "cfx": ("cfx", sets == "cfx"),
        "cfy": ("cfy", sets == "cfy"),
    }
    torch.set_num_threads(4)
    for group, (head_set, in_group) in groups.items():
        pick = np.flatnonzero(in_group)
        order = np.concatenate([pick[site_idx[pick] == k] for k in range(len(feats))])
        base = np.concatenate(
            [state["x"][k][idx[order][site_idx[order] == k]] for k in range(len(feats))]
        )
        site_codes = site_idx[order]
        for kind in fit_module.KINDS:
            zero = list(fit_module.ERROR_COLUMNS) if kind == "ref" else []
            x = base.copy()
            x[:, zero] = 0.0
            x_inj = x.copy()
            x_inj[:, 28] += (shift[order] / 100.0).astype(np.float32)
            for scheme in fit_module.SCHEMES:
                c = np.empty((fit_module.N_SEEDS, len(order)), np.float32)
                s = np.empty_like(c)
                ci = np.empty_like(c)
                si = np.empty_like(c)
                for seed in range(fit_module.N_SEEDS):
                    net = model_module.Head(n_sites=len(feats))
                    net.load_state_dict(
                        torch.load(
                            fit_module.NETS_DIR / f"{head_set}_{kind}_{scheme}_{seed}.pt",
                            weights_only=True,
                        )
                    )
                    net.eval()
                    c[seed], s[seed] = model_module.predict(
                        net=net, x=x, site=site_codes, device="cpu"
                    )
                    if group == "full":
                        ci[seed], si[seed] = model_module.predict(
                            net=net, x=x_inj, site=site_codes, device="cpu"
                        )
                np.savez(
                    WORK_DIR / f"preds_{group}_{kind}_{scheme}.npz",
                    row_id=rows["row_id"].to_numpy()[order],
                    c=c,
                    s=s,
                    c_injected=ci,
                    s_injected=si,
                )
                print(group, kind, scheme, c.shape, flush=True)


if __name__ == "__main__":
    predict_all()
