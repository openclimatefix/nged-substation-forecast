"""Score the arms of experiment 00094 on all rows, in cold hours, and in the tails.

Prints, and writes to `work/results/`:

- `contrasts`: the four pre-registered contrasts C1 to C4 (treatment minus `ref`, quantile score in
  points of capacity) at 98.75%, from 20,000 resamples of whole ISO weeks of valid time, paired on
  series, run, and valid time, plus the exploratory contrasts at 95%.
- `slices`: each arm's quantile score, mean signed error of the median (observed minus median, so
  positive means the forecast was too low), pinball loss at q05 and q95, and the share of
  observations above q95 and below q05, in each temperature slice and lead band.
- `weeks`: how many ISO weeks each slice holds, since the interval rests on them.

Cold means the ensemble-mean ENS 2 m temperature of the case is below 3 C (the temperature
available at forecast time). The check slice uses the ERA5 temperature at the valid time.

    uv run python ml-experiments/00094-*/e00094_stats.py
"""

from dataclasses import dataclass
from typing import Final

import e00094_common as common
import numpy as np
import polars as pl

N_RESAMPLES: Final[int] = 20_000
BONFERRONI_LEVEL: Final[float] = 98.75
EXPLORATORY_LEVEL: Final[float] = 95.0
COLD_C: Final[float] = 3.0
SHORT_LEAD_HOURS: Final[float] = 48.0
SEED: Final[int] = 20261018
ARM_NAMES: Final[tuple[str, ...]] = ("ref", "ht", "mono")


@dataclass(frozen=True)
class Contrast:
    """A paired difference: treatment minus `ref` on a slice.

    Attributes:
        label: The contrast's label.
        treatment: The treatment arm.
        slice_name: A key of `slice_masks()`.
        registered: Whether it belongs to the pre-registered Bonferroni family.
    """

    label: str
    treatment: str
    slice_name: str
    registered: bool


CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast("C1 ht, cold, up to 48 h", "ht", "cold_short", True),
    Contrast("C2 ht, all rows", "ht", "all", True),
    Contrast("C3 mono, cold, up to 48 h", "mono", "cold_short", True),
    Contrast("C4 mono, all rows", "mono", "all", True),
    Contrast("ht, cold, all leads", "ht", "cold", False),
    Contrast("ht, cold, over 48 h", "ht", "cold_long", False),
    Contrast("ht, cold by ERA5, up to 48 h", "ht", "cold_era5_short", False),
    Contrast("ht, mild winter, all leads", "ht", "mild_winter", False),
    Contrast("ht, summer (Jun to Aug), all leads", "ht", "summer", False),
    Contrast("mono, cold, all leads", "mono", "cold", False),
    Contrast("ht minus mono, cold, up to 48 h", "ht", "cold_short", False),
)


def temperature_table() -> pl.DataFrame:
    """Return the ensemble-mean ENS temperature of each case, from the 365 demand frames."""
    target = common.RESULTS_DIR / "ens_temperature.parquet"
    if target.exists():
        return pl.read_parquet(target)
    pieces = []
    for run in common.run_dates(subset="all"):
        frame = pl.read_parquet(
            common.demand_frame_path(run=run),
            columns=[*common.KEYS, "temperature_2m"],
        )
        pieces.append(
            frame.group_by(*common.KEYS).agg(ens_temperature=pl.col("temperature_2m").mean())
        )
    table = pl.concat(pieces)
    common.atomic_write(frame=table, target=target)
    return table


def case_table(*, name: str) -> pl.DataFrame:
    """Return one arm's per-case metrics, keyed by series, run, and valid time."""
    frame = (
        pl.read_parquet(common.ARMS_DIR / f"{name}.parquet")
        .filter(pl.col("actual").is_finite() & (pl.col("valid_time") < common.HOLDOUT))
        .sort(*common.KEYS, "seed")
    )
    quantiles = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1)
    actual = frame["actual"].to_numpy().astype(np.float64)
    miss = actual[:, None] - quantiles
    pinball = np.maximum(common.LEVELS * miss, (common.LEVELS - 1.0) * miss) * 100.0
    return frame.select(*common.KEYS).with_columns(
        qs=pl.Series(common.pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0),
        bias=pl.Series(miss[:, common.MEDIAN] * 100.0),
        pin05=pl.Series(pinball[:, 0]),
        pin95=pl.Series(pinball[:, 8]),
        above95=pl.Series((actual > quantiles[:, 8]).astype(float)),
        below05=pl.Series((actual < quantiles[:, 0]).astype(float)),
    )


def joined_cases() -> pl.DataFrame:
    """Return every arm's metrics side by side on the cases all arms share, with slice columns."""
    tables = {name: case_table(name=name) for name in ARM_NAMES}
    shared = tables["ref"].select(common.KEYS)
    for name in ARM_NAMES[1:]:
        shared = shared.join(tables[name].select(common.KEYS), on=list(common.KEYS), how="inner")
    columns = ("qs", "bias", "pin05", "pin95", "above95", "below05")
    frame = shared
    for name, table in tables.items():
        frame = frame.join(
            table.rename({c: f"{c}_{name}" for c in columns}),
            on=list(common.KEYS),
            how="left",
            validate="1:1",
        )
    era5 = pl.read_parquet(
        common.ERA5_PATH, columns=["time_series_id", "valid_time", "temperature_2m"]
    )
    frame = (
        frame.join(temperature_table(), on=list(common.KEYS), how="left", validate="1:1")
        .join(
            era5.rename({"temperature_2m": "era5_temperature"}),
            on=["time_series_id", "valid_time"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
            local_month=pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.month(),
            week=pl.col("valid_time").dt.strftime("%G-W%V"),
        )
    )
    if frame["ens_temperature"].null_count():
        msg = "some cases have no ENS temperature"
        raise ValueError(msg)
    return frame


def slice_masks(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return a boolean mask for each slice."""
    cold = (frame["ens_temperature"] < COLD_C).to_numpy()
    cold_era5 = (frame["era5_temperature"] < COLD_C).fill_null(False).to_numpy()
    short = (frame["lead_hours"] <= SHORT_LEAD_HOURS).to_numpy()
    winter = frame["local_month"].is_in([12, 1, 2]).to_numpy()
    summer = frame["local_month"].is_in([6, 7, 8]).to_numpy()
    return {
        "all": np.ones(frame.height, dtype=bool),
        "cold": cold,
        "mild_winter": winter & ~cold,
        "cold_short": cold & short,
        "cold_long": cold & ~short,
        "mild_winter_short": winter & ~cold & short,
        "mild_winter_long": winter & ~cold & ~short,
        "cold_era5_short": cold_era5 & short,
        "summer": summer,
    }


def week_bounds(
    *, values: np.ndarray, weeks: np.ndarray, level: float, generator: np.random.Generator
) -> tuple[float, float, int]:
    """Return the interval of the mean of `values`, resampling whole ISO weeks of valid time.

    Args:
        values: The per-case values (a difference of two arms, or one arm's metric).
        weeks: The ISO week label of each case.
        level: The confidence level in percent.
        generator: The random stream.

    Returns:
        The lower and upper ends, and the number of weeks holding at least one case.
    """
    _unique, inverse = np.unique(weeks, return_inverse=True)
    sums = np.bincount(inverse, weights=values.astype(np.float64))
    counts = np.bincount(inverse)
    drawn = generator.integers(0, len(sums), size=(N_RESAMPLES, len(sums)))
    resampled = sums[drawn].sum(axis=1) / counts[drawn].sum(axis=1)
    tail = (100.0 - level) / 2.0
    return (
        float(np.percentile(resampled, tail)),
        float(np.percentile(resampled, 100.0 - tail)),
        len(sums),
    )


def contrast_rows(*, frame: pl.DataFrame, masks: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return every contrast with its difference and interval."""
    rows = []
    for contrast in CONTRASTS:
        mask = masks[contrast.slice_name]
        subset = frame.filter(pl.Series(mask))
        reference = "mono" if contrast.label.startswith("ht minus mono") else "ref"
        difference = (subset[f"qs_{contrast.treatment}"] - subset[f"qs_{reference}"]).to_numpy()
        level = BONFERRONI_LEVEL if contrast.registered else EXPLORATORY_LEVEL
        generator = np.random.default_rng(SEED)
        lower, upper, n_weeks = week_bounds(
            values=difference, weeks=subset["week"].to_numpy(), level=level, generator=generator
        )
        rows.append(
            {
                "contrast": contrast.label,
                "registered": contrast.registered,
                "level": level,
                "difference": float(difference.mean()),
                "lower": lower,
                "upper": upper,
                "rows": subset.height,
                "weeks": n_weeks,
            }
        )
    return pl.DataFrame(rows)


def slice_rows(*, frame: pl.DataFrame, masks: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return each arm's mean metrics in each slice."""
    rows = []
    for slice_name, mask in masks.items():
        subset = frame.filter(pl.Series(mask))
        rows.extend(
            {
                "slice": slice_name,
                "arm": arm,
                "rows": subset.height,
                **{
                    metric: subset[f"{metric}_{arm}"].mean()
                    for metric in ("qs", "bias", "pin05", "pin95", "above95", "below05")
                },
            }
            for arm in ARM_NAMES
        )
    return pl.DataFrame(rows)


def lead_bias_rows(*, frame: pl.DataFrame, masks: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return each arm's bias and q95 exceedance by lead band, for cold and mild-winter cases."""
    rows = []
    for slice_name in ("cold", "mild_winter"):
        subset = frame.filter(pl.Series(masks[slice_name]))
        for label, low, high in common.LEAD_BANDS:
            band = subset.filter((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))
            rows.extend(
                {
                    "slice": slice_name,
                    "band": label,
                    "arm": arm,
                    "rows": band.height,
                    **{
                        metric: band[f"{metric}_{arm}"].mean()
                        for metric in ("qs", "bias", "above95")
                    },
                }
                for arm in ARM_NAMES
            )
    return pl.DataFrame(rows)


def main() -> None:
    """Print the scores and write them under `work/results/`."""
    frame = joined_cases()
    masks = slice_masks(frame=frame)
    print(f"{frame.height:,} shared cases, {frame['week'].n_unique()} ISO weeks", flush=True)
    outputs = {
        "contrasts": contrast_rows(frame=frame, masks=masks),
        "slices": slice_rows(frame=frame, masks=masks),
        "lead_bias": lead_bias_rows(frame=frame, masks=masks),
    }
    cold_weeks = frame.filter(pl.Series(masks["cold"]))["week"].value_counts().sort("week")
    outputs["cold_weeks"] = cold_weeks
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=60, tbl_width_chars=200):
        for name, table in outputs.items():
            print(f"\n== {name}", flush=True)
            print(table)
            common.atomic_write(frame=table, target=common.RESULTS_DIR / f"{name}.parquet")


if __name__ == "__main__":
    main()
