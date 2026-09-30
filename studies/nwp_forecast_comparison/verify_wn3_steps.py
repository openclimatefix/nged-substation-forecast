"""Check WeatherNext 3's hour convention against ERA5, from the built inputs, before any fit.

One-off throwaway script for the WeatherNext 3 arms of
<https://github.com/openclimatefix/nged-substation-forecast/issues/974>. It reads and never fits,
and writes `<output-dir>/verification/wn3_steps.md`. It exits non-zero if a check fails.

WeatherNext 3's radiation is the mean of the hour ending at the valid time, and its wind is the
value at the valid time. For each time offset of -1, 0 and +1 h, the check shifts ERA5's series by
the offset and scores the built `wn3_mean_day<N>` column against it, pooled over the sites, at
days 1 and 2:

1. **Solar.** The mean absolute difference between `wn3_mean_day<N>_ghi` and ERA5's hour-ending
   global irradiance. The check fails unless offset 0 has the lowest difference.
2. **Wind.** The correlation of `wn3_mean_day<N>_speed_100m` with ERA5's 100 m speed. The check
   fails unless offset 0 has the highest correlation.

Every offset is scored on the same rows: the valid times at which ERA5 holds all three shifted
values. Only pooled statistics and the anonymised `site` label are printed.

Run it with `uv run python studies/nwp_forecast_comparison/verify_wn3_steps.py --published-dir
PUBLISHED --output-dir DIR`, where `DIR` holds the built `<domain>_wn3_inputs.parquet`.
"""

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Final

import polars as pl
from build_forecast_inputs import DomainType
from studies.guards import refuse_to_overwrite
from verify_aifs_steps import era5_column

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

OFFSETS: Final[tuple[int, ...]] = (-1, 0, 1)
"""The time shifts, in hours, applied to ERA5's series."""

VERIFY_DAYS: Final[tuple[int, ...]] = (1, 2)
"""The days scored: the diurnal cycle and the day-to-day weather still show at these leads."""

DOMAIN_COLUMNS: Final[dict[str, str]] = {"solar": "ghi", "wind": "speed_100m"}
"""The `wn3_mean_day<N>_<field>` field scored for each technology."""


def offset_scores(
    *, inputs: pl.DataFrame, era5: pl.DataFrame, domain: DomainType, day: int
) -> dict[int | str, float]:
    """Score one built column against ERA5 shifted by each offset, on common rows.

    Args:
        inputs: `<domain>_wn3_inputs.parquet`.
        era5: `site`, `time` and `era5`, from `era5_column`.
        domain: `solar` or `wind`.
        day: The day whose `wn3_mean_day<N>` column is scored.

    Returns:
        Each offset's mean absolute difference (solar) or correlation (wind), plus `n_rows`.
    """
    column = f"wn3_mean_day{day}_{DOMAIN_COLUMNS[domain]}"
    frame = inputs.select("site", "time", wn3=pl.col(column)).drop_nulls()
    for offset in OFFSETS:
        shifted = era5.select(
            "site",
            time=pl.col("time") - timedelta(hours=offset),
            **{f"era5_{offset}": pl.col("era5")},
        )
        frame = frame.join(shifted, on=["site", "time"], how="inner")
    scores: dict[int | str, float] = {"n_rows": frame.height}
    for offset in OFFSETS:
        if domain == "solar":
            scores[offset] = frame.select((pl.col("wn3") - pl.col(f"era5_{offset}")).abs().mean())[
                0, 0
            ]
        else:
            scores[offset] = frame.select(pl.corr("wn3", f"era5_{offset}"))[0, 0]
    return scores


def best_is_zero(*, scores: dict[int | str, float], domain: DomainType) -> bool:
    """Return whether offset 0 has the lowest difference (solar) or highest correlation (wind)."""
    values = {offset: scores[offset] for offset in OFFSETS}
    pick = min if domain == "solar" else max
    return pick(values, key=lambda offset: values[offset]) == 0


def main() -> int:
    """Run the checks, write the report, and return 1 if any check failed."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.resolve() == args.published_dir.resolve():
        msg = "the verification output must not be the published folder"
        raise ValueError(msg)
    verification = args.output_dir / "verification"
    report_path = verification / "wn3_steps.md"
    refuse_to_overwrite(paths=[report_path])
    lines = ["# WeatherNext 3 hour convention against ERA5", ""]
    failures: list[str] = []
    for domain in ("solar", "wind"):
        inputs = pl.read_parquet(args.output_dir / f"{domain}_wn3_inputs.parquet")
        sites = sorted(inputs["site"].unique().to_list())
        era5 = era5_column(domain=domain, sites=sites)
        metric = "mean absolute difference (W/m2)" if domain == "solar" else "correlation"
        lines += [
            f"## {domain.capitalize()}: {metric} by ERA5 shift",
            "",
            "| Day | Rows | -1 h | 0 h | +1 h | Offset 0 best |",
            "|---|---|---|---|---|---|",
        ]
        for day in VERIFY_DAYS:
            scores = offset_scores(inputs=inputs, era5=era5, domain=domain, day=day)
            ok = best_is_zero(scores=scores, domain=domain)
            if not ok:
                failures.append(f"{domain} day {day}")
            lines.append(
                f"| {day} | {scores['n_rows']} | "
                + " | ".join(f"{scores[offset]:.4f}" for offset in OFFSETS)
                + f" | {ok} |"
            )
        lines.append("")
    verification.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    _LOG.info("wrote %s", report_path)
    if failures:
        _LOG.error("failed: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
