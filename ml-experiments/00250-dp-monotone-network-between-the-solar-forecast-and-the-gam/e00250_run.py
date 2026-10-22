"""Fit the monotone solar network at every monthly cut of the scored year and write its forecasts.

At the first day of each scored month, the network trains on the stage 2 forecasts of experiments
180 and 214 whose valid time falls in the 12 months before that day. The forecasts of the month then
replace the solar term of the demand forecast with the network's output. Arms:

- `n`: the unclipped forecast (experiment 180's control `c`) with its solar term replaced.
- `nc`: the 60%-clip forecast (experiment 214's `k60`) with its clipped solar term replaced.
- `n2`: as `n`, plus the mean and standard deviation of the 51 members' solar responses
  (experiment 237), at D5 and D9 only.

The band of each arm is the base arm's band scaled by one factor per lead band, fitted on the
training rows. Writes `work/arms/<arm>/dp_preds_s2/d_scored_<site>.parquet`, in the format of the
stage 2 files.

    systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00250-*/e00250_run.py --workers 4
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Final

import e00250_common as common
import e00250_inputs as inputs
import e00250_nn as nn
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
WINDOW: Final[timedelta] = timedelta(days=365)
MAX_LEAD_HOURS: Final[float] = 168.0
SUN_POINTS: Final[float] = 0.01
LEAD_BANDS: Final[tuple[float, ...]] = (24.0, 72.0, 168.0)
FACTORS: Final[np.ndarray] = np.arange(0.5, 2.01, 0.05)
TWO_NUMBER_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D16")
MEMBERS_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00184-dp-ens-daylight-calibration" / "work" / "arms"
HAND_SET_SHARE: Final[float] = 0.6
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
KEY: Final[list[str]] = ["run", "valid_time"]


def load_rows(*, site: str) -> pl.DataFrame:
    """Return every period's rows of the control and the 60% clip, with the solar inputs."""
    pieces = []
    for period in PERIODS:
        control = pl.read_parquet(common.arm_file(arm="c", period=period, site=site))
        clip = pl.read_parquet(common.arm_file(arm="k60", period=period, site=site)).select(
            *KEY, **{f"k_{q}": pl.col(q) for q in QUANTILES}
        )
        solar = pl.read_parquet(inputs.OUT_DIR / f"{site}_{period}.parquet")
        joined = control.join(clip, on=KEY, how="inner", validate="1:1").join(
            solar, on=KEY, how="inner", validate="1:1"
        )
        if joined.height != control.height:
            msg = f"{site} {period}: {joined.height} joined rows for {control.height}"
            raise ValueError(msg)
        pieces.append(joined.with_columns(period=pl.lit(period)))
    return pl.concat(pieces)


def with_two_number(*, rows: pl.DataFrame, site: str) -> pl.DataFrame:
    """Add the mean and standard deviation of the members' solar responses (points)."""
    pieces = []
    for period in PERIODS:
        frame = pl.read_parquet(MEMBERS_DIR / f"{period}_frame.parquet").with_row_index("row")
        delta = np.load(MEMBERS_DIR / f"{period}_delta_m.npy", mmap_mode="r")
        keep = frame.filter(pl.col("site") == site)
        member = delta[keep["row"].to_numpy()].astype(np.float64)
        pieces.append(
            keep.select(
                "run",
                "valid_time",
                m_mean=pl.Series(np.nanmean(member, axis=1)),
                m_sd=pl.Series(np.nanstd(member, axis=1)),
            )
        )
    both = pl.concat(pieces)
    out = rows.join(both, on=KEY, how="left", validate="1:1")
    missing = out["m_mean"].is_null().sum() / out.height
    if missing > 0.02:
        msg = f"{site}: {missing:.1%} of rows lack the two numbers"
        raise ValueError(msg)
    return out.with_columns(pl.col("m_mean", "m_sd").fill_null(0.0))


def side_inputs(*, rows: pl.DataFrame, cap: float, two_number: bool) -> np.ndarray:
    """Return the network's side inputs: clear-sky index, hour, cap, optionally the two numbers."""
    hour = (
        rows["valid_time"].dt.convert_time_zone("Europe/London").dt.hour().to_numpy()
        + rows["valid_time"].dt.convert_time_zone("Europe/London").dt.minute().to_numpy() / 60.0
    )
    columns = [
        np.clip(rows["csi"].to_numpy(), 0.0, 1.5),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.full(rows.height, cap),
    ]
    if two_number:
        columns += [rows["m_mean"].to_numpy() / 100.0, rows["m_sd"].to_numpy() / 100.0]
    return np.stack(columns, axis=1)


def pinball(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return each row's pinball loss averaged over the 11 levels."""
    error = actual[:, None] - quantiles
    return np.mean(np.maximum(LEVELS * error, (LEVELS - 1.0) * error), axis=1)


def band_factors(*, train: pl.DataFrame, median: np.ndarray, prefix: str) -> dict[int, float]:
    """Return the band-width factor of each lead band, fitted on the training rows."""
    base = train.select([f"{prefix}{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    offsets = base - base[:, [5]]
    actual = train["actual"].to_numpy().astype(np.float64)
    lead = train["lead_hours"].to_numpy()
    out = {}
    low = 0.0
    for band, high in enumerate(LEAD_BANDS):
        mask = (lead > low) & (lead <= high)
        low = high
        if mask.sum() < 200:
            out[band] = 1.0
            continue
        losses = [
            float(
                pinball(
                    actual=actual[mask], quantiles=median[mask, None] + f * offsets[mask]
                ).mean()
            )
            for f in FACTORS
        ]
        out[band] = float(FACTORS[int(np.argmin(losses))])
    return out


def forecast_arm(
    *,
    rows: pl.DataFrame,
    cut: datetime,
    cap: float,
    solar: str,
    prefix: str,
    two_number: bool,
) -> pl.DataFrame:
    """Return the forecast rows of the cut's month under one arm, with the network's quantiles."""
    start = cut - WINDOW
    train = rows.filter(
        (pl.col("valid_time") >= start)
        & (pl.col("valid_time") < cut)
        & (pl.col("lead_hours") <= MAX_LEAD_HOURS)
        & ~pl.col("switching_mask")
        & pl.col("actual").is_finite()
        & (pl.col(solar) > SUN_POINTS)
    ).sort("valid_time")
    month = rows.filter(
        pl.col("power_fcst_init_time").dt.truncate("1mo").dt.replace_time_zone("UTC") == cut
    )
    q50 = f"{prefix}q50"

    def pieces(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = frame[solar].to_numpy().astype(np.float64) / 100.0
        base = (frame[q50].to_numpy().astype(np.float64) + frame[solar].to_numpy()) / 100.0
        x = side_inputs(rows=frame, cap=cap, two_number=two_number)
        target = frame["actual"].to_numpy().astype(np.float64) / 100.0
        return s, x, base, target

    s, x, base, target = pieces(train)
    fitted = nn.fit_network(s=s, x=x, base=base, target=target)
    train_all = rows.filter(
        (pl.col("valid_time") >= start)
        & (pl.col("valid_time") < cut)
        & (pl.col("lead_hours") <= MAX_LEAD_HOURS)
        & ~pl.col("switching_mask")
        & pl.col("actual").is_finite()
    )
    s_t, x_t, base_t, _ = pieces(train_all)
    median_train = (base_t - fitted.predict(s=s_t, x=x_t)) * 100.0
    factors = band_factors(train=train_all, median=median_train, prefix=prefix)
    s_m, x_m, base_m, _ = pieces(month)
    median = (base_m - fitted.predict(s=s_m, x=x_m)) * 100.0
    quantiles = month.select([f"{prefix}{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    offsets = quantiles - quantiles[:, [5]]
    lead = month["lead_hours"].to_numpy()
    scale = np.ones(month.height)
    low = 0.0
    for band, high in enumerate(LEAD_BANDS):
        scale[(lead > low) & (lead <= high)] = factors[band]
        low = high
    new = median[:, None] + scale[:, None] * offsets
    return month.with_columns([pl.Series(q, new[:, i]) for i, q in enumerate(QUANTILES)]).select(
        common_columns(month=month)
    )


def common_columns(*, month: pl.DataFrame) -> list[str]:
    """Return the stage 2 columns, in the control file's order."""
    return [
        c
        for c in month.columns
        if not c.startswith(("k_", "s_", "csi", "held", "cut", "m_")) and c != "period"
    ]


def _job(site: str) -> str:
    rows = load_rows(site=site)
    caps = json.loads(inputs.LEARNED_CAPS.read_text())[site]["p"]
    two_number_rows = with_two_number(rows=rows, site=site) if site in TWO_NUMBER_SITES else None
    arms: dict[str, list[pl.DataFrame]] = {"n": [], "nc": [], "n2": []}
    cuts = [
        datetime(2025 if m >= 7 else 2026, m, 1, tzinfo=UTC) for m in (*range(7, 13), *range(1, 7))
    ]
    for cut in cuts:
        cap = float(caps[cut.strftime("%Y-%m-%d")])
        arms["n"].append(
            forecast_arm(rows=rows, cut=cut, cap=cap, solar="s_raw", prefix="", two_number=False)
        )
        arms["nc"].append(
            forecast_arm(rows=rows, cut=cut, cap=cap, solar="s_clip", prefix="k_", two_number=False)
        )
        if two_number_rows is not None:
            arms["n2"].append(
                forecast_arm(
                    rows=two_number_rows,
                    cut=cut,
                    cap=cap,
                    solar="s_raw",
                    prefix="",
                    two_number=True,
                )
            )
    scored = rows.filter(pl.col("period") == "scored")
    for arm, frames in arms.items():
        if not frames:
            continue
        out = pl.concat(frames)
        if out.height != scored.height:
            msg = f"{site} {arm}: {out.height} rows for {scored.height} scored rows"
            raise ValueError(msg)
        target = common.WORK_DIR / "arms" / arm / "dp_preds_s2" / f"d_scored_{site}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        out.write_parquet(temporary)
        temporary.rename(target)
    return f"{site} done"


def main() -> None:
    """Fit and forecast every site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sites", nargs="+", default=list(inputs.SITES))
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, args.sites):
            print(line, flush=True)


if __name__ == "__main__":
    main()
