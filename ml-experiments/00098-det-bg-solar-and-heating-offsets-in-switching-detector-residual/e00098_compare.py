"""Compare the arms: detector recall, contrasts, residual noise, and forecast error.

Every arm scores the same transfers (experiment 82's seeds), so the arms are paired on the
transfer and on the calendar month. Reads each arm's `detector/<arm>/` folder and `fits/<arm>/`.

    uv run python ml-experiments/00098-*/e00098_compare.py --arms h05_stored ctrl era5_solar \
        cams_solar heat
"""

from __future__ import annotations

import argparse
import itertools
import json
from typing import Final

import numpy as np
import polars as pl
from e00098_common import BASE_DIR, FITS_ROOT, SERIES_IDS
from e00098_fit import SOLAR_SERIES
from e00098_profile import NON_MATCHING
from e00098_report import STEP_EDGES, contrast, fmt, populations, wide

DETECTORS: Final[tuple[str, ...]] = ("XL", "PF14", "LOGIT_SAME", "X")
"""RO-NC is `XL`; `PF14` and `LOGIT_SAME` are the older detectors."""
N_CONTRASTS: Final[int] = 4
SCORE_START: Final[str] = "2022-01-01"
MIDDAY_HOURS: Final[tuple[int, ...]] = (10, 11, 12, 13)
"""UTC hours of the summer-midday band."""
RESULTS_PATH: Final = BASE_DIR / "results" / "compare.json"


def arm_frames(*, arm: str, detector: str) -> list[pl.DataFrame]:
    """The S2, S3, and S4 frames of one arm, with `detector`'s found flag as column `arm`."""
    data = populations(out=BASE_DIR / "detector" / arm)
    donors = pl.read_parquet(BASE_DIR / "detector" / arm / "events.parquet").select(
        "event_id", "donor"
    )
    frames = []
    for shape in NON_MATCHING:
        frame = wide(data=data, shape=shape, population="base", parity="even")
        frames.append(
            frame.select("event_id", "weight", "year_month", "metered_receivers", "donor_step_z")
            .join(donors, on="event_id", validate="1:1")
            .with_columns(found=frame[detector])
            .rename({"found": arm, "donor_step_z": f"z_{arm}"})
        )
    return frames


def paired(*, arms: list[str], detector: str) -> list[pl.DataFrame]:
    """One frame per shape holding every arm's found flag, joined on the transfer."""
    per_arm = {arm: arm_frames(arm=arm, detector=detector) for arm in arms}
    out = []
    for index in range(len(NON_MATCHING)):
        frame = per_arm[arms[0]][index]
        for arm in arms[1:]:
            other = per_arm[arm][index]
            same = other.height == frame.height and all(
                other[c].equals(frame[c])
                for c in (
                    "event_id",
                    "weight",
                    "year_month",
                    "metered_receivers",
                    "donor",
                )
            )
            if not same:
                raise ValueError(f"{arm} scores different transfers from {arms[0]}")
            frame = frame.with_columns(other[arm], other[f"z_{arm}"])
        out.append(frame)
    return out


def residual_table(*, arm: str) -> pl.DataFrame:
    """Absolute residual of the model refitted on the 1st, in fractions of effective capacity."""
    arm_dir = FITS_ROOT / arm / "first"
    parts = []
    for series in SERIES_IDS:
        path = arm_dir / f"{series}.parquet"
        if not path.exists():
            path = FITS_ROOT / "ctrl" / "first" / f"{series}.parquet"
        parts.append(pl.read_parquet(path).with_columns(series=pl.lit(series)))
    return (
        pl.concat(parts)
        .filter(pl.col("valid_time") >= pl.lit(SCORE_START).str.to_datetime(time_zone="UTC"))
        .with_columns(
            abs_residual=(pl.col("target") - pl.col("pred")).abs(),
            month=pl.col("valid_time").dt.month(),
            hour=pl.col("valid_time").dt.hour(),
        )
        .with_columns(
            season=pl.when(pl.col("month").is_in([12, 1, 2]))
            .then(pl.lit("winter"))
            .when(pl.col("month").is_in([3, 4, 5]))
            .then(pl.lit("spring"))
            .when(pl.col("month").is_in([6, 7, 8]))
            .then(pl.lit("summer"))
            .otherwise(pl.lit("autumn")),
            band=pl.when(pl.col("hour").is_between(0, 5))
            .then(pl.lit("night 00-06"))
            .when(pl.col("hour").is_between(6, 9))
            .then(pl.lit("morning 06-10"))
            .when(pl.col("hour").is_between(10, 13))
            .then(pl.lit("midday 10-14"))
            .when(pl.col("hour").is_between(14, 17))
            .then(pl.lit("afternoon 14-18"))
            .otherwise(pl.lit("evening 18-24")),
        )
    )


def noise_rows(*, arms: list[str]) -> pl.DataFrame:
    """Median and mean absolute residual by arm, series group, season, and hour band."""
    rows = []
    for arm in arms:
        if arm == "h05_stored":
            continue
        table = residual_table(arm=arm)
        for group, ids in (("solar series", SOLAR_SERIES), ("all series", SERIES_IDS)):
            part = table.filter(pl.col("series").is_in(list(ids)))
            for (season, band), cell in part.group_by("season", "band"):
                rows.append(
                    {
                        "arm": arm,
                        "group": group,
                        "season": season,
                        "band": band,
                        "median_abs": float(np.median(cell["abs_residual"].to_numpy())) * 100,
                        "mae": float(np.mean(cell["abs_residual"].to_numpy())) * 100,
                        "rows": cell.height,
                    }
                )
    return pl.DataFrame(rows)


def odd_summer_midday(*, arm: str) -> float:
    """Median absolute residual, April to September 10:00 to 14:00 UTC, odd months, solar series."""
    table = residual_table(arm=arm).filter(
        pl.col("series").is_in(list(SOLAR_SERIES)),
        pl.col("month").is_between(4, 9),
        pl.col("month") % 2 == 1,
        pl.col("hour").is_in(list(MIDDAY_HOURS)),
    )
    return float(np.median(table["abs_residual"].to_numpy())) * 100


def by_step(*, frames: list[pl.DataFrame], arms: list[str]) -> list[dict]:
    """Transfers found by donor step in each arm's own noise units (S2 to S4 summed).

    The step in noise units depends on the arm's residual noise, so each arm bins by its own step
    and reports its own transfer count `n_<arm>` as well as the number found.
    """
    out = []
    for flag, label in ((True, "metered"), (False, "unmetered")):
        for low, high in itertools.pairwise(STEP_EDGES):
            row: dict = {"receivers": label, "low": low, "high": high}
            for arm in arms:
                found = n = 0
                for frame in frames:
                    part = frame.filter(
                        (pl.col("metered_receivers") > 0) == flag,
                        pl.col(f"z_{arm}").abs().is_between(low, high, "left"),
                    )
                    n += part.height
                    found += int(part[arm].sum())
                row[arm], row[f"n_{arm}"] = found, n
            out.append(row)
    return out


def main() -> None:
    """Print and store every cross-arm table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--heat-parent", default="era5_solar")
    args = parser.parse_args()
    arms = args.arms
    results: dict = {"arms": arms, "recall": {}, "contrasts": {}, "steps": {}}
    for detector in DETECTORS:
        frames = paired(arms=arms, detector=detector)
        results["recall"][detector] = {}
        print(f"\n== {detector}: MW-weighted recall, even months, S2 to S4 pooled")
        for arm in arms:
            value = contrast(frames=frames, arm=arm)
            results["recall"][detector][arm] = list(value)
            print(f"  {arm:12s} {value[0]:.3f} ({value[1]:.3f} to {value[2]:.3f})")
        if detector == "XL":
            results["steps"] = by_step(frames=frames, arms=arms)
            for row in results["steps"]:
                print("  step", row)
    frames = paired(arms=arms, detector="XL")
    print(f"\nPlanned contrasts for RO-NC (level 1-0.05/{N_CONTRASTS} intervals via report LEVEL)")
    pairs = {
        "C1 era5_solar - ctrl": ("era5_solar", "ctrl"),
        "C2 cams_solar - ctrl": ("cams_solar", "ctrl"),
        "C3 heat - ctrl": ("heat", "ctrl"),
        "C4 heat - solar parent": ("heat", args.heat_parent),
        "descriptive ctrl - h05_stored": ("ctrl", "h05_stored"),
    }
    for name, (arm, base) in pairs.items():
        if arm in arms and base in arms:
            value = contrast(frames=frames, arm=arm, base=base)
            results["contrasts"][name] = list(value)
            print(f"  {name}: {fmt(*value)}")
    print("\nExploratory: RO-NC recall split by whether the donor has a solar term")
    results["donor_split"] = {}
    for label, keep in (("solar donors", True), ("other donors", False)):
        part = [f.filter(pl.col("donor").is_in(list(SOLAR_SERIES)) == keep) for f in frames]
        results["donor_split"][label] = {"transfers": part[0].height}
        for arm in arms:
            value = contrast(frames=part, arm=arm)
            results["donor_split"][label][arm] = list(value)
            print(f"  {label} (n={part[0].height}) {arm:12s} {fmt(*value)}")
        for arm, base in (("era5_solar", "ctrl"), ("cams_solar", "ctrl"), ("heat", "ctrl")):
            if arm in arms and base in arms:
                value = contrast(frames=part, arm=arm, base=base)
                results["donor_split"][label][f"{arm} - {base}"] = list(value)
                print(f"    {arm} - {base}: {fmt(*value)}")
    noise = noise_rows(arms=arms)
    print("\nOdd-month summer-midday median |residual| at the solar series (points of capacity)")
    results["selection"] = {}
    for arm in arms:
        if arm != "h05_stored":
            results["selection"][arm] = odd_summer_midday(arm=arm)
            print(f"  {arm:12s} {results['selection'][arm]:.4f}")
    print(
        noise.filter(pl.col("group") == "solar series", pl.col("season") == "summer").sort(
            "arm", "band"
        )
    )
    print(
        noise.filter(pl.col("group") == "all series")
        .group_by("arm")
        .agg(mae=(pl.col("mae") * pl.col("rows")).sum() / pl.col("rows").sum())
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=1))
    noise.write_parquet(RESULTS_PATH.with_name("noise.parquet"))


if __name__ == "__main__":
    main()
