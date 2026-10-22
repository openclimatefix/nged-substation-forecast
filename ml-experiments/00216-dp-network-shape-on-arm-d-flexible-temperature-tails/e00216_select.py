"""Choose the tail ridge ratio on the tuning months only (the plan's pre-registered rule).

The criterion is the mean absolute error of the filtered mean forecast (`base_preds`) over leads
0.5 to 168 hours on tuning-month rows in deciles 1 and 10 of the site's 24-hour temperature, outside
switching events, pooled over the substations with at least 200 such rows. A larger ratio wins only
if its error is lower by more than 0.5%; otherwise the smaller ratio wins. The error on all tuning
rows is printed beside it.

    uv run --no-sync python ml-experiments/00216-*/e00216_select.py
"""

import json
from typing import Final

import e00216_common as common
import e00216_temperature as temperature
import polars as pl

RATIOS: Final[tuple[str, ...]] = ("1", "0.1", "0.01")
MIN_TAIL_ROWS: Final[int] = 200
TIE_MARGIN: Final[float] = 0.005


def mean_of(series: pl.Series) -> float:
    """Return a series' mean as a float."""
    return float(series.to_numpy().mean())


def tuning_rows(*, arm: str) -> pl.DataFrame:
    """Return an arm's tuning-month base forecasts with the decile of each valid time."""
    deciles = temperature.all_frames()
    frames = [
        pl.read_parquet(common.WORK_DIR / "base_preds" / f"{arm}_{site}.parquet").with_columns(
            site=pl.lit(site)
        )
        for site in common.SCORED_SITES
    ]
    return (
        pl.concat(frames)
        .filter(
            (pl.col("period") == "tuning")
            & ~pl.col("switching_mask")
            & pl.col("lead_hours").is_between(0.5, 168.0)
        )
        .join(deciles, on=["site", "valid_time"], how="inner")
        .with_columns(abs_error=(pl.col("actual") - pl.col("mean")).abs())
    )


def main() -> None:
    """Print each ratio's errors and write the choice to `work/tail_ratio.json`."""
    errors: dict[str, dict[str, float]] = {}
    usable_sites: set[str] | None = None
    frames = {ratio: tuning_rows(arm=f"f1_{ratio}") for ratio in RATIOS}
    for frame in frames.values():
        counts = (
            frame.filter(pl.col("decile").is_in([1, 10]))
            .group_by("site")
            .len()
            .filter(pl.col("len") >= MIN_TAIL_ROWS)
        )
        sites = set(counts["site"].to_list())
        usable_sites = sites if usable_sites is None else usable_sites & sites
    assert usable_sites is not None
    control = tuning_rows(arm="r0")
    errors["r0"] = _errors(frame=control, sites=usable_sites)
    for ratio, frame in frames.items():
        errors[ratio] = _errors(frame=frame, sites=usable_sites)
    for name, value in errors.items():
        print(name, value)
    chosen = RATIOS[-1]
    for ratio in reversed(RATIOS[:-1]):
        if errors[ratio]["tail"] < errors[chosen]["tail"] * (1.0 - TIE_MARGIN):
            chosen = ratio
    print("chosen ratio:", chosen, "sites:", sorted(usable_sites))
    common.write_json_atomic(
        path=common.WORK_DIR / "tail_ratio.json",
        content={"chosen": chosen, "errors": errors, "sites": sorted(usable_sites)},
    )
    print(json.dumps({"chosen": chosen}))


def _errors(*, frame: pl.DataFrame, sites: set[str]) -> dict[str, float]:
    kept = frame.filter(pl.col("site").is_in(sorted(sites)))
    tail = kept.filter(pl.col("decile").is_in([1, 10]))
    return {"tail": mean_of(tail["abs_error"]), "all": mean_of(kept["abs_error"])}


if __name__ == "__main__":
    main()
