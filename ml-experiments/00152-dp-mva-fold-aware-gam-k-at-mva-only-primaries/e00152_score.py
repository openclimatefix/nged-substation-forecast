"""Score stage 2: the pre-registered contrasts R1 to R3, slices, and tail calibration.

Reads `work/preds/real/{period}_{site}.parquet`; writes `work/score.json`.
"""

import json
from typing import Final

import e00152_common as common
import numpy as np
import polars as pl
from studies.dp_gam import QUANTILE_COLUMNS

N_DRAWS: Final[int] = 10_000
LOWER_PERCENT: Final[float] = 0.625
UPPER_PERCENT: Final[float] = 99.375
"""The 98.75% interval: 5% split over four planned contrasts."""
SEED: Final[int] = 152
QUANTILES: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
LEVEL_OF: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}


def load(*, period: str, sites: list[str]) -> pl.DataFrame:
    """Return the stored predictions of the sites for one period, with a score and a month."""
    frames = [
        pl.read_parquet(
            common.WORK_DIR / "preds" / "real" / f"{period}_{site}.parquet"
        ).with_columns(pl.lit(site).alias("site"))
        for site in sites
    ]
    return pl.concat(frames)


def with_score(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the row quantile score (points) and the calendar month."""
    quantiles = frame.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    score = common.quantile_score(actual=actual, quantiles=quantiles)
    return frame.with_columns(
        pl.Series("score", score),
        pl.col("valid_time").dt.strftime("%Y-%m").alias("month"),
    )


def paired(*, frame: pl.DataFrame, treatment: str, reference: str) -> pl.DataFrame:
    """Return per-row score differences (treatment minus reference) on the shared rows."""
    keys = ["site", "run", "valid_time"]
    if frame.height == 0:
        return pl.DataFrame(
            schema={"month": pl.String, "sunny_midday": pl.Boolean, "difference": pl.Float64}
        )
    wide = (
        frame.filter(pl.col("arm").is_in([treatment, reference]))
        .select(*keys, "arm", "score", "month", "sunny_midday", "lead_hours")
        .pivot(on="arm", index=[*keys, "month", "sunny_midday", "lead_hours"], values="score")
        .drop_nulls()
    )
    return wide.with_columns((pl.col(treatment) - pl.col(reference)).alias("difference"))


def month_block(*, difference: pl.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    """Return the mean difference and its 98.75% interval from resampling whole calendar months."""
    if difference.height == 0:
        return {"difference": float("nan"), "lower": float("nan"), "upper": float("nan"), "n": 0}
    monthly = difference.group_by("month").agg(
        pl.col("difference").sum().alias("total"), pl.len().alias("count")
    )
    totals = monthly["total"].to_numpy()
    counts = monthly["count"].to_numpy().astype(np.float64)
    picks = rng.integers(0, len(totals), size=(N_DRAWS, len(totals)))
    draws = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lower, upper = np.percentile(draws, [LOWER_PERCENT, UPPER_PERCENT])
    return {
        "difference": float(totals.sum() / counts.sum()),
        "lower": float(lower),
        "upper": float(upper),
        "n": int(counts.sum()),
        "months": len(totals),
    }


def contrast(
    *, frame: pl.DataFrame, treatment: str, reference: str, sites: list[str], where: str = "all"
) -> dict[str, float]:
    """Return one paired contrast pooled over the sites."""
    rng = np.random.default_rng(SEED)
    difference = paired(
        frame=frame.filter(pl.col("site").is_in(sites)), treatment=treatment, reference=reference
    )
    if where == "sunny_midday":
        difference = difference.filter(pl.col("sunny_midday"))
    return month_block(difference=difference, rng=rng)


def tail_calibration(*, frame: pl.DataFrame) -> list[dict]:
    """Return the share of actuals below q01 and q05 and above q95 and q99, per site and arm."""
    rows = []
    for (site, arm), group in frame.group_by(["site", "arm"]):
        row = {"site": site, "arm": arm, "n": group.height}
        for column in QUANTILES:
            below = column in ("q01", "q05")
            row[column] = float(
                (  # ty: ignore[invalid-argument-type]
                    (group["actual"] < group[column])
                    if below
                    else (group["actual"] > group[column])
                ).mean()
            )
        row["mean_error"] = float((group["mean"] - group["actual"]).mean())  # ty: ignore[invalid-argument-type]
        row["median_minus_mean"] = float((group["q50"] - group["mean"]).mean())  # ty: ignore[invalid-argument-type]
        rows.append(row)
    return rows


def main() -> None:
    """Score both periods and write `score.json`."""
    others = [s for s in common.MVA_ONLY_PRIMARIES if s not in ("D12", "D8")]
    pooled = [*others, "D8"]
    out: dict = {}
    for period in ("scored", "replicate"):
        frame = with_score(
            frame=load(
                period=period,
                sites=[
                    s
                    for s in common.MVA_ONLY_PRIMARIES
                    if (common.WORK_DIR / "preds" / "real" / f"{period}_{s}.parquet").exists()
                ],
            )
        )
        result = {
            "R1_F_minus_N_D12": contrast(
                frame=frame, treatment="F_fold", reference="N", sites=["D12"]
            ),
            "R2_F_minus_A_D12_sunny": contrast(
                frame=frame, treatment="F_fold", reference="A", sites=["D12"], where="sunny_midday"
            ),
            "R2_all_rows": contrast(frame=frame, treatment="F_fold", reference="A", sites=["D12"]),
            "R3_F_minus_N_pooled": contrast(
                frame=frame, treatment="F_fold", reference="N", sites=pooled
            ),
            "A_minus_N_D12": contrast(frame=frame, treatment="A", reference="N", sites=["D12"]),
            "per_site_F_minus_N": {
                s: contrast(frame=frame, treatment="F_fold", reference="N", sites=[s])
                for s in common.MVA_ONLY_PRIMARIES
            },
            "per_site_F_minus_A": {
                s: contrast(frame=frame, treatment="F_fold", reference="A", sites=[s])
                for s in common.MVA_ONLY_PRIMARIES
            },
            "absolute_score": {
                f"{site}|{arm}": float(g["score"].mean())  # ty: ignore[invalid-argument-type]
                for (site, arm), g in frame.group_by(["site", "arm"])
            },
            "tail": tail_calibration(frame=frame),
            "tail_D12_sunny": tail_calibration(
                frame=frame.filter((pl.col("site") == "D12") & pl.col("sunny_midday"))
            ),
        }
        leave_out = {}
        for left in others:
            leave_out[left] = contrast(
                frame=frame,
                treatment="F_fold",
                reference="N",
                sites=[s for s in pooled if s != left],
            )
        result["R3_leave_one_out"] = leave_out
        out[period] = result
    (common.WORK_DIR / "score.json").write_text(json.dumps(out, indent=1))
    for period, result in out.items():
        for key in (
            "R1_F_minus_N_D12",
            "R2_F_minus_A_D12_sunny",
            "R2_all_rows",
            "R3_F_minus_N_pooled",
        ):
            print(period, key, result[key])


if __name__ == "__main__":
    main()
