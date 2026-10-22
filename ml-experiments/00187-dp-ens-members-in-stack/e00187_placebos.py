"""Score arm M against three further placebos and report the share of M's gain each recovers.

The placebos replace each daylight row's 51 member responses (`delta_m`, from experiment 184's
stored arrays) and refit the four lead-band factors on the tuning months:

- `mp_strong`: responses taken from a random row of the same substation, calendar month, hour band,
  lead day (0 to 6), and clear-sky band.
- `gauss`: 51 Gaussian quantile points with the row's own mean and standard deviation of the
  responses.
- `flip`: the responses reflected about their row mean, which reverses the skew.

    uv run --no-sync python ml-experiments/00187-*/e00187_placebos.py
"""

from collections.abc import Callable
from typing import Final

import e00187_common as common
import e00187_data as data
import e00187_fit as fit
import e00187_score as score
import numpy as np
import polars as pl
from scipy.stats import norm

SHUFFLE_SEED: Final[int] = 7
CUTS: Final[tuple[str, ...]] = (
    "daylight",
    "daylight, under 24 h",
    "daylight, 24 to 72 h",
    "daylight, 72 to 168 h",
    "daylight, overcast (under 0.4)",
    "daylight, partly cloudy (0.4 to 0.7)",
    "daylight, clear (0.7 and over)",
    "sunniest third (experiment 178)",
)
GAUSS_POINTS: Final[np.ndarray] = norm.ppf((np.arange(51) + 0.5) / 51)


def labelled(*, period: str) -> tuple[pl.DataFrame, np.ndarray]:
    """Return a period's labelled frame and the stored member responses."""
    frame = score.label_columns(frame=pl.read_parquet(fit.ARM_DIR / f"{period}_frame.parquet"))
    return frame, np.load(fit.ARM_DIR / f"{period}_delta_m.npy").astype(np.float64)


def strong_shuffle(frame: pl.DataFrame, delta: np.ndarray) -> np.ndarray:
    """Shuffle whole rows of responses within substation, month, hour band, lead day, cloud band."""
    day = (frame["kind"] == "daylight").to_numpy()
    lead_day = np.minimum((frame["lead_hours"].to_numpy() // 24).astype(int), 6)
    group = (
        frame["site"]
        + "|"
        + frame["valid_time"].dt.month().cast(pl.String)
        + "|"
        + frame["hour_band"].cast(pl.String)
        + "|"
        + pl.Series(lead_day.astype(str))
        + "|"
        + frame["csi_band"].fill_null("na")
    ).to_numpy()
    rng = np.random.default_rng(SHUFFLE_SEED)
    out = delta.copy()
    for key in np.unique(group[day]):
        rows = np.flatnonzero(day & (group == key))
        out[rows] = delta[rng.permutation(rows)]
    return out


def gaussian(frame: pl.DataFrame, delta: np.ndarray) -> np.ndarray:
    """Return Gaussian points with each row's own mean and standard deviation of the responses."""
    mean = np.nanmean(delta, axis=1, keepdims=True)
    return mean + np.nanstd(delta, axis=1, keepdims=True) * GAUSS_POINTS[None, :]


def reflected(frame: pl.DataFrame, delta: np.ndarray) -> np.ndarray:
    """Return the responses reflected about their row mean."""
    mean = np.nanmean(delta, axis=1, keepdims=True)
    return 2.0 * mean - delta


VARIANTS: Final[dict[str, Callable[[pl.DataFrame, np.ndarray], np.ndarray]]] = {
    "m": lambda frame, delta: delta,
    "mp_strong": strong_shuffle,
    "gauss": gaussian,
    "flip": reflected,
}


def main() -> None:
    """Fit each variant's factors on the tuning months, score both periods, write the shares."""
    tuning_frame, tuning_delta = labelled(period="tuning")
    factors = {
        name: fit.fit_factors(
            frame=tuning_frame, delta=make(tuning_frame, tuning_delta), centre=False
        )
        for name, make in VARIANTS.items()
    }
    out: dict = {"factors": factors}
    for period in ("scored", "replicate"):
        frame, delta = labelled(period=period)
        stored = np.load(fit.ARM_DIR / f"{period}_mp.npy").astype(np.float64)
        arms = {"a": frame.select(list(data.QUANTILES)).to_numpy().astype(np.float64), "mp": stored}
        for name, make in VARIANTS.items():
            arms[name] = np.sort(
                fit.apply_member_arm(
                    frame=frame, delta=make(frame, delta), factors=factors[name], centre=False
                ),
                axis=1,
            )
        masks = score.subsets(frame=frame)
        metrics = {arm: score.row_metrics(frame=frame, quantiles=q) for arm, q in arms.items()}
        month = frame["month"].to_numpy()
        period_out: dict = {}
        for cut in CUTS:
            mask = masks[cut]
            row: dict = {}
            for arm in arms:
                if arm == "a":
                    continue
                change = score.percent_change(
                    month=month[mask],
                    arm=metrics[arm]["qs"][mask],
                    reference=metrics["a"]["qs"][mask],
                    level=95.0,
                    seed=184,
                )
                row[arm] = {**change, "inside90": float(metrics[arm]["inside90"][mask].mean())}
            for arm in ("mp", "mp_strong", "gauss", "flip"):
                row[arm]["share_of_m"] = row[arm]["point"] / row["m"]["point"]
            period_out[cut] = row
        out[period] = period_out
        day = period_out["daylight"]
        print(
            period,
            {
                k: (round(v["point"], 2), round(v["share_of_m"], 2) if "share_of_m" in v else None)
                for k, v in day.items()
            },
        )
    common.write_json_atomic(path=common.WORK_DIR / "report" / "placebos.json", content=out)


if __name__ == "__main__":
    main()
