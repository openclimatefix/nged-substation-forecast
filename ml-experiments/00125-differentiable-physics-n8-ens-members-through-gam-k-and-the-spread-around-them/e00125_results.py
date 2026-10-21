"""Score experiment 125's arms and write `work/results.json`.

Rows are the decision sites' (D5, D10, D11, D14, D16) scored half-hours outside the hand-filtered
switching events, 0.5 to 168 hours from the origin, shared by every arm. D9 is reported beside them.
Intervals resample whole calendar months, the same months for both arms of a contrast; the winter
rows resample days in blocks of 7 within months.

    uv run python ml-experiments/00125-*/e00125_results.py
"""

from typing import Final, cast

import e00125_common as common
import e00125_post as post
import numpy as np
import polars as pl
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours
from studies.ens_members import MIXTURE_LEVELS

ARMS: Final[tuple[str, ...]] = ("K0", "K0-w", "MP", "MP-s", "MM", "MO", "MP-1")
PERIODS: Final[tuple[str, ...]] = ("replicate", "scored")
QUANTILES: Final[tuple[str, ...]] = tuple(f"q{round(p * 100):02d}" for p in MIXTURE_LEVELS)
TAIL_COLUMNS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
N_RESAMPLES: Final[int] = 2000
SEED: Final[int] = 20_251_125
BLOCK_DAYS: Final[int] = 7
MARGIN_POINTS: Final[float] = 0.02
BONFERRONI_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 3.0)
"""The interval level of the three planned contrasts, 98.3%."""
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)


def load(*, arm: str, period: str, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return an arm's scored rows for the sites, with local time columns added."""
    frames = [
        pl.read_parquet(common.WORK_DIR / "preds" / f"{arm}_{period}_{site}.parquet")
        for site in sites
    ]
    return (
        pl.concat(frames)
        .filter(~pl.col("switching_mask"))
        .with_columns(local=pl.col("valid_time").dt.convert_time_zone("Europe/London"))
        .with_columns(
            local_date=pl.col("local").dt.date(),
            local_hour=pl.col("local").dt.hour(),
            local_month=pl.col("local").dt.month(),
            month_label=pl.col("local").dt.year() * 100 + pl.col("local").dt.month(),
        )
    )


def with_bands(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the lead band column."""
    return frame.with_columns(
        lead_band=pl.Series(lead_band_of_hours(lead_hours=frame["lead_hours"].to_numpy())).cast(
            pl.Int8
        )
    )


def per_row(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return per-row quantile score and tail pinball."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    quantiles = frame.select(list(QUANTILES)).to_numpy().astype(np.float64)
    rows = post.Rows(
        quantiles=quantiles,
        actual=actual,
        run=np.zeros(len(actual), dtype=np.int64),
        valid_time=np.zeros(len(actual), dtype="datetime64[us]"),
        lead_hours=frame["lead_hours"].to_numpy(),
        hour_band=np.zeros(len(actual)),
        quarter=np.zeros(len(actual)),
        switching_mask=np.zeros(len(actual), dtype=bool),
    )
    return {
        "score": post.quantile_score(actual=actual, quantiles=quantiles),
        "tail": post.tail_pinball(rows=rows),
        "mae_median": np.abs(actual - quantiles[:, 5]),
    }


def block_ids(*, frame: pl.DataFrame, days: bool) -> np.ndarray:
    """Return a resampling block per row: the month, or a 7-day block within the month."""
    month = frame["month_label"].to_numpy().astype(np.int64)
    if not days:
        return month
    day = frame["local_date"].dt.day().to_numpy().astype(np.int64)
    return month * 10 + (day - 1) // BLOCK_DAYS


def bootstrap(*, values: np.ndarray, blocks: np.ndarray, level: float) -> tuple[float, float]:
    """Return the percentile interval of the mean of `values`, resampling whole blocks."""
    unique, inverse = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    generator = np.random.default_rng(SEED)
    draws = generator.integers(0, len(unique), size=(N_RESAMPLES, len(unique)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (100.0 - level) / 2.0
    return float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))


def calibration_cells(*, frame: pl.DataFrame) -> dict:
    """Return coverage, n_eff, and in-band flags for 4 tail levels x 4 lead bands."""
    cells = []
    in_band = 0
    for band in range(N_LEAD_BANDS):
        chosen = frame.filter(pl.col("lead_band") == band)
        n_eff = chosen.select(pl.struct("site", "local_date").n_unique()).item()
        row = []
        for column, level in zip(TAIL_COLUMNS, TAIL_LEVELS, strict=True):
            coverage = cast("float", (chosen["actual"] <= chosen[column]).mean())
            half = 2.5 * np.sqrt(level * (1 - level) / n_eff)
            ok = bool(abs(coverage - level) <= half)
            in_band += int(ok)
            row.append(
                {"level": level, "coverage": coverage, "half_width": float(half), "in_band": ok}
            )
        cells.append(row)
    return {"cells": cells, "in_band": in_band}


def summarise(*, frame: pl.DataFrame) -> dict:
    """Return the headline numbers of one arm on one set of rows."""
    values = per_row(frame=frame)
    out = {
        "n_rows": frame.height,
        "quantile_score": cast("float", values["score"].mean()),
        "tail_pinball": cast("float", values["tail"].mean()),
        "mae_median": cast("float", values["mae_median"].mean()),
        "calibration": calibration_cells(frame=frame),
    }
    out["by_band"] = {
        str(band): {
            "quantile_score": cast(
                "float", values["score"][frame["lead_band"].to_numpy() == band].mean()
            ),
            "tail_pinball": cast(
                "float", values["tail"][frame["lead_band"].to_numpy() == band].mean()
            ),
        }
        for band in range(N_LEAD_BANDS)
    }
    out["coverage_80_90"] = {
        "central_80": cast(
            "float", ((frame["actual"] >= frame["q10"]) & (frame["actual"] <= frame["q90"])).mean()
        ),
        "central_90": cast(
            "float", ((frame["actual"] >= frame["q05"]) & (frame["actual"] <= frame["q95"])).mean()
        ),
    }
    return out


def contrast(*, treatment: pl.DataFrame, reference: pl.DataFrame, winter: bool = False) -> dict:
    """Return the paired differences of the quantile score and tail pinball with intervals."""
    a, b = per_row(frame=treatment), per_row(frame=reference)
    blocks = block_ids(frame=treatment, days=winter)
    out = {}
    for name, level in (("score", 95.0), ("tail", BONFERRONI_LEVEL)):
        difference = a[name] - b[name]
        lower, upper = bootstrap(values=difference, blocks=blocks, level=level)
        out[name] = {"difference": cast("float", difference.mean()), "lower": lower, "upper": upper}
    out["n_blocks"] = len(np.unique(blocks))
    return out


def main() -> None:
    """Score every arm and write `work/results.json`."""
    decision = common.DECISION_SITES
    results: dict = {}
    for period in PERIODS:
        frames = {
            arm: with_bands(frame=load(arm=arm, period=period, sites=common.SCORED_SITES))
            for arm in ARMS
        }
        frames = {
            arm: frame.filter((pl.col("lead_hours") >= 0.5) & (pl.col("lead_hours") <= 168.0))
            for arm, frame in frames.items()
        }
        block: dict = {"arms": {}, "contrasts": {}, "winter": {}, "d9": {}, "by_site": {}}
        for arm, frame in frames.items():
            on_decision = frame.filter(pl.col("site").is_in(decision))
            block["arms"][arm] = summarise(frame=on_decision)
            block["d9"][arm] = summarise(frame=frame.filter(pl.col("site") == "D9"))
            winter = on_decision.filter(pl.col("local_month").is_in(WINTER_MONTHS))
            block["winter"][arm] = summarise(frame=winter)
            block["by_site"][arm] = {
                site: {
                    "quantile_score": float(
                        per_row(frame=frame.filter(pl.col("site") == site))["score"].mean()
                    ),
                    "tail_pinball": float(
                        per_row(frame=frame.filter(pl.col("site") == site))["tail"].mean()
                    ),
                }
                for site in common.SCORED_SITES
            }
        pairs = {
            "C1 MP-K0-w": ("MP", "K0-w"),
            "C1b MP-K0": ("MP", "K0"),
            "C2 MP-MM": ("MP", "MM"),
            "C4 MPs-MP": ("MP-s", "MP"),
            "K0w-K0": ("K0-w", "K0"),
            "MP-MO": ("MP", "MO"),
            "MP1-MP": ("MP-1", "MP"),
        }
        for name, (treatment, reference) in pairs.items():
            first = frames[treatment].filter(pl.col("site").is_in(decision))
            second = frames[reference].filter(pl.col("site").is_in(decision))
            block["contrasts"][name] = contrast(treatment=first, reference=second)
            block["contrasts"][name + " winter"] = contrast(
                treatment=first.filter(pl.col("local_month").is_in(WINTER_MONTHS)),
                reference=second.filter(pl.col("local_month").is_in(WINTER_MONTHS)),
                winter=(period == "scored"),
            )
        sites_better = sum(
            block["by_site"]["MP"][site]["tail_pinball"]
            < block["by_site"]["K0-w"][site]["tail_pinball"]
            for site in decision
        )
        block["sites_improving_tail_pinball_MP_vs_K0w"] = int(sites_better)
        results[period] = block
    common.write_json_atomic(path=common.WORK_DIR / "results.json", content=results)
    for period in PERIODS:
        print(period)
        for arm in ARMS:
            item = results[period]["arms"][arm]
            print(
                f"  {arm:5s} score {item['quantile_score']:.4f} tail {item['tail_pinball']:.4f} "
                f"cells {item['calibration']['in_band']}/16 mae {item['mae_median']:.4f}"
            )
        for name, item in results[period]["contrasts"].items():
            print(
                f"  {name:18s} score {item['score']['difference']:+.4f} "
                f"[{item['score']['lower']:+.4f},{item['score']['upper']:+.4f}] "
                f"tail {item['tail']['difference']:+.4f} "
                f"[{item['tail']['lower']:+.4f},{item['tail']['upper']:+.4f}]"
            )
        print("  sites improving", results[period]["sites_improving_tail_pinball_MP_vs_K0w"])


if __name__ == "__main__":
    main()
