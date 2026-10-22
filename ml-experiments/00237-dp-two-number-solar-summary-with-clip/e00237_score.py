"""Score the two-number summary with and without the clip, with month-resampled intervals.

The score is the quantile score: twice the mean pinball loss over the 11 quantiles, in points of
effective capacity. The gain of an arm is its daylight total minus arm A's daylight total, as a
percentage of arm A's total (negative is better). The retained share is the gain with the clip
divided by the gain without it. Months of the period are resampled as whole blocks, in pairs
across the two conditions.

    uv run --no-sync python ml-experiments/00237-*/e00237_score.py
"""

import json
from typing import Final

import e00237_common as common
import e00237_data as data
import numpy as np
import polars as pl

OUT_DIR = common.WORK_DIR / "report"
SITES: Final[tuple[str, ...]] = ("D5", "D9")
ARMS: Final[tuple[str, ...]] = ("a", "m", "q2", "sg", "rs")
N_DRAWS: Final[int] = 20_000
LEVEL: Final[float] = 95.0
DAYLIGHT_DEG: Final[float] = 5.0
SEED: Final[int] = 237


def prepare(*, name: str, period: str) -> pl.DataFrame:
    """Return a condition's rows with the usable, daylight, and sunny flags."""
    frame = pl.read_parquet(OUT_DIR / f"{name}_{period}.parquet")
    usable = pl.Series(
        data.usable_rows(frame=frame.with_columns(switching_mask=pl.col("switching_mask")))
    )
    frame = frame.with_columns(usable=usable)
    edges = {}
    for (site,), group in frame.filter(
        pl.col("usable") & (pl.col("unit") > data.SUN_UP_UNIT_SHAPE)
    ).group_by("site"):
        edges[site] = float(np.quantile(group["unit"].to_numpy(), 2 / 3))
    return frame.with_columns(
        daylight=pl.col("usable")
        & (pl.col("elevation") >= DAYLIGHT_DEG)
        & (pl.col("n_members") > 0),
        sunny=pl.col("usable")
        & (pl.col("unit") > data.SUN_UP_UNIT_SHAPE)
        & (pl.col("unit") >= pl.col("site").replace_strict(edges, return_dtype=pl.Float64)),
    )


def blocks(*, frame: pl.DataFrame, mask: str, months: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-month sums of each arm's quantile score and median error, and the row counts."""
    out = {}
    part = frame.filter(pl.col(mask))
    grouped = part.group_by("month").agg(
        [pl.col(f"qs_{a}").sum().alias(f"qs_{a}") for a in ARMS]
        + [pl.col(f"abs_{a}").sum().alias(f"abs_{a}") for a in ARMS]
        + [pl.len().alias("n")]
    )
    table = pl.DataFrame({"month": months}).join(grouped, on="month", how="left").fill_null(0)
    for column in grouped.columns:
        if column != "month":
            out[column] = table[column].to_numpy().astype(np.float64)
    return out


def gain(
    *, b: dict[str, np.ndarray], arm: str, idx: np.ndarray, base: str = "a", column: str = "qs"
) -> float:
    """Return an arm's change against `base` as a percentage of the base total, on months `idx`."""
    return (
        100.0
        * (b[f"{column}_{arm}"][idx].sum() - b[f"{column}_{base}"][idx].sum())
        / b[f"{column}_{base}"][idx].sum()
    )


def interval(*, values: np.ndarray) -> list[float]:
    """Return the point estimate (first draw) and its 95% percentile interval."""
    low, high = np.nanpercentile(values[1:], [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2])
    return [float(values[0]), float(low), float(high)]


def score_period(*, period: str) -> dict:
    """Return the scores of one period."""
    frames = {name: prepare(name=name, period=period) for name in ("noclip", "clip")}
    months = np.array(sorted(frames["noclip"]["month"].unique().to_list()))
    rng = np.random.default_rng(SEED)
    draws = np.vstack(
        [np.arange(len(months)), rng.integers(0, len(months), size=(N_DRAWS, len(months)))]
    )
    result: dict = {}
    for scope, sites in (("D5", ["D5"]), ("D9", ["D9"]), ("D5 and D9", ["D5", "D9"])):
        for mask in ("daylight", "sunny"):
            cut = {n: f.filter(pl.col("site").is_in(sites)) for n, f in frames.items()}
            b = {n: blocks(frame=f, mask=mask, months=months) for n, f in cut.items()}
            cell: dict = {"rows": int(b["noclip"]["n"].sum())}
            for name in ("noclip", "clip"):
                for arm in ARMS:
                    cell[f"qs_{name}_{arm}"] = float(
                        b[name][f"qs_{arm}"].sum() / b[name]["n"].sum()
                    )
                    cell[f"mae_{name}_{arm}"] = float(
                        b[name][f"abs_{arm}"].sum() / b[name]["n"].sum()
                    )
                for arm in ("m", "q2", "sg", "rs"):
                    cell[f"gain_{name}_{arm}"] = interval(
                        values=np.array([gain(b=b[name], arm=arm, idx=d) for d in draws])
                    )
                    cell[f"mae_change_{name}_{arm}"] = interval(
                        values=np.array(
                            [gain(b=b[name], arm=arm, idx=d, column="abs") for d in draws]
                        )
                    )
            for arm in ("m", "q2", "sg", "rs"):
                ratio = np.array(
                    [
                        gain(b=b["clip"], arm=arm, idx=d) / gain(b=b["noclip"], arm=arm, idx=d)
                        for d in draws
                    ]
                )
                cell[f"retained_{arm}"] = interval(values=ratio)
            cell["q2_minus_m_clip"] = interval(
                values=np.array([gain(b=b["clip"], arm="q2", idx=d, base="m") for d in draws])
            )
            cell["clip_effect_on_a"] = interval(
                values=np.array(
                    [
                        100.0
                        * (b["clip"]["qs_a"][d].sum() - b["noclip"]["qs_a"][d].sum())
                        / b["noclip"]["qs_a"][d].sum()
                        for d in draws
                    ]
                )
            )
            result[f"{scope}|{mask}"] = cell
    return result


def main() -> None:
    """Score the replicate and scored periods and write the table."""
    out = {period: score_period(period=period) for period in ("replicate", "scored")}
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=1))
    for period, cells in out.items():
        for key, cell in cells.items():
            print(
                period,
                key,
                cell["rows"],
                "Q2 gain noclip",
                [round(x, 2) for x in cell["gain_noclip_q2"]],
                "clip",
                [round(x, 2) for x in cell["gain_clip_q2"]],
                "retained",
                [round(x, 2) for x in cell["retained_q2"]],
            )


if __name__ == "__main__":
    main()
