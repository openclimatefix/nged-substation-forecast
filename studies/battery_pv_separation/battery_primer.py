"""Primer numbers: what the four batteries do, from the downloaded market data.

Builds one half-hourly frame per battery: settled output (B1610), the planned output (the
physical notification, half-hourly mean), the net accepted balancing volume (BOAV, signed so a
bid is negative and an offer positive), the remainder (output minus both), the grid frequency's
deviation from 50 Hz, and whether the battery had a balancing acceptance (BOALF) or a response
contract (NESO's Enduring Auction Capability, EAC) in the half-hour. The auction unit is matched
to the battery only by identifier, so the response contract is a candidate match.

Writes `primer_frames.parquet`, `primer_summary.parquet`, and `report_primer.md`.

Run: `OMP_NUM_THREADS=2 uv run python studies/battery_pv_separation/battery_primer.py`.
"""

from datetime import timedelta
from typing import Any, Final

import numpy as np
import polars as pl
from battery_inputs import (
    BATTERIES,
    EXPECTED_ROWS,
    MINUTES_PER_HALF_HOUR,
    NAMES,
    OUTPUT_DIR,
    WINDOW_START,
    battery_output,
    market_frame,
)
from studies.sources import MARKET_DOWNLOADS_DIR

NOMINAL_FREQUENCY_HZ: Final[float] = 50.0
HOURS_PER_HALF_HOUR: Final[float] = 0.5
RESPONSE_SERVICE: Final[str] = "Response"
"""NESO's service type for Dynamic Containment, Moderation, and Regulation."""
FREQUENCY_BINS: Final[int] = 25
"""The binned-mean chart of the remainder against frequency uses this many equal-count bins."""
MIN_BIN_COUNT: Final[int] = 30
TAIL_HZ: Final[float] = 0.08
"""The frequency deviation, in hertz, beyond which a half-hour counts as a high or low tail."""
SEASONS: Final[dict[str, tuple[int, ...]]] = {
    "Winter (December to February)": (12, 1, 2),
    "Summer (June to August)": (6, 7, 8),
}
"""The calendar months of each season in the typical-day figure."""
FRAME_PATH: Final = OUTPUT_DIR / "primer_frames.parquet"
SUMMARY_PATH: Final = OUTPUT_DIR / "primer_summary.parquet"


def _market(*, folder: str, name: str) -> pl.DataFrame:
    """Read one downloaded market table.

    Args:
        folder: The folder under the market downloads.
        name: The parquet file's stem.

    Returns:
        The table.
    """
    return pl.read_parquet(MARKET_DOWNLOADS_DIR / folder / f"{name}.parquet")


def _grid() -> pl.DataFrame:
    """Return the study window's half-hour starts.

    Returns:
        A frame with one column, `time`.
    """
    return pl.DataFrame(
        {
            "time": pl.datetime_range(
                WINDOW_START,
                WINDOW_START + timedelta(days=365) - timedelta(minutes=MINUTES_PER_HALF_HOUR),
                interval="30m",
                time_unit="us",
                eager=True,
            )
        }
    )


def _acceptance_half_hours(*, bmu_id: str) -> pl.DataFrame:
    """Return the half-hours in which a BOALF acceptance ramp touches the battery.

    Args:
        bmu_id: The BMU.

    Returns:
        A frame with `time` (half-hour start) and `has_acceptance` true, one row per half-hour.
    """
    boalf = (
        _market(folder="elexon_boalf", name="elexon_boalf")
        .filter((pl.col("bmu_id") == bmu_id) & (pl.col("amendment_flag") != "DEL"))
        .select(
            first=pl.col("time_from").dt.truncate("30m"),
            last=pl.max_horizontal("time_from", "time_to").dt.truncate("30m"),
        )
    )
    return (
        boalf.select(time=pl.datetime_ranges("first", "last", interval="30m"))
        .explode("time")
        .unique()
        .with_columns(has_acceptance=pl.lit(True))
    )


def _response_half_hours(*, auction_unit: str, service: str) -> pl.DataFrame:
    """Return the half-hours inside a contract block of an auction unit, and the volume held.

    Args:
        auction_unit: NESO's auction unit identifier.
        service: The NESO service type.

    Returns:
        A frame with `time` (half-hour start) and `contracted_mw`, the sum of the unit's accepted
        volumes over the service's products in the half-hour.
    """
    rows = _market(folder="neso_eac_response_reserve", name="neso_eac_response_reserve").filter(
        (pl.col("auction_unit") == auction_unit) & (pl.col("service_type") == service)
    )
    return (
        rows.select(
            time=pl.datetime_ranges("time", "time_end", interval="30m", closed="left"),
            contracted_mw="executed_quantity_mw",
        )
        .explode("time")
        .group_by("time")
        .agg(pl.col("contracted_mw").sum())
    )


def eac_auction_unit(*, bmu_id: str) -> str | None:
    """Return the candidate NESO auction unit of a battery, by identifier match.

    Args:
        bmu_id: The Elexon BMU identifier.

    Returns:
        The auction unit, or None if no unit matches.
    """
    mapping = pl.read_csv(
        MARKET_DOWNLOADS_DIR / "neso_eac_unit_to_bmu" / "eac_unit_to_bmu.csv"
    ).filter(pl.col("candidate_bmu_id") == bmu_id)
    return None if mapping.is_empty() else mapping["auction_unit"][0]


def battery_decomposition(*, bmu_id: str) -> pl.DataFrame:
    """Return a battery's half-hourly output, plan, accepted volume, and remainder.

    Args:
        bmu_id: The Elexon BMU identifier.

    Returns:
        One row per half-hour start in the window with columns `time`, `output_mw` (B1610 times
        two), `plan_mw` (the half-hourly mean physical notification), `accepted_mw` (the net
        accepted volume times two; zero where there was no acceptance), `remainder_mw` (output
        minus plan minus accepted), `frequency_deviation_hz` (50 minus the half-hour's mean),
        `has_acceptance`, `contracted_response_mw`, `contracted_reserve_mw`, and `bmu_id`.
        Output, plan, remainder, and the frequency have nulls where the source has no value.
    """
    plan = (
        _market(folder="elexon_pn", name="elexon_pn_half_hourly")
        .filter((pl.col("bmu_id") == bmu_id) & (pl.col("covered_seconds") == 1800))
        .select("time", plan_mw="mean_level_mw")
    )
    accepted = (
        _market(folder="elexon_boav", name="elexon_boav")
        .filter(pl.col("bmu_id") == bmu_id)
        .group_by("time")
        .agg(accepted_mw=pl.col("total_volume_accepted_mwh").sum() / HOURS_PER_HALF_HOUR)
    )
    frequency = _market(folder="elexon_frequency", name="elexon_frequency").select(
        "time", frequency_deviation_hz=NOMINAL_FREQUENCY_HZ - pl.col("frequency_mean_hz")
    )
    unit = eac_auction_unit(bmu_id=bmu_id)
    response = (
        _response_half_hours(auction_unit=unit, service=RESPONSE_SERVICE)
        if unit
        else pl.DataFrame(schema={"time": pl.Datetime("us", "UTC"), "contracted_mw": pl.Float64})
    ).rename({"contracted_mw": "contracted_response_mw"})
    reserve = (
        pl.concat(
            _response_half_hours(auction_unit=unit, service=service)
            for service in ("Quick Reserve", "Slow Reserve", "Balancing Reserve")
        )
        .group_by("time")
        .agg(contracted_reserve_mw=pl.col("contracted_mw").sum())
        if unit
        else pl.DataFrame(
            schema={"time": pl.Datetime("us", "UTC"), "contracted_reserve_mw": pl.Float64}
        )
    )
    output = battery_output(bmu_id=bmu_id).select("time", "output_mw")
    return (
        _grid()
        .join(output, on="time", how="left")
        .join(plan, on="time", how="left")
        .join(accepted, on="time", how="left")
        .join(frequency, on="time", how="left")
        .join(_acceptance_half_hours(bmu_id=bmu_id), on="time", how="left")
        .join(response, on="time", how="left")
        .join(reserve, on="time", how="left")
        .with_columns(
            accepted_mw=pl.col("accepted_mw").fill_null(0.0),
            has_acceptance=pl.col("has_acceptance").fill_null(value=False),
            contracted_response_mw=pl.col("contracted_response_mw").fill_null(0.0),
            contracted_reserve_mw=pl.col("contracted_reserve_mw").fill_null(0.0),
        )
        .with_columns(remainder_mw=pl.col("output_mw") - pl.col("plan_mw") - pl.col("accepted_mw"))
        .with_columns(bmu_id=pl.lit(bmu_id))
    )


def variance_explained(*, output: np.ndarray, explained: np.ndarray) -> float:
    """Return the share of the output's variance that a prediction removes, with no fitting.

    Args:
        output: The metered output.
        explained: The prediction, with the same length.

    Returns:
        One minus the sum of squared differences over the sum of squared deviations from the mean.
    """
    return float(1.0 - ((output - explained) ** 2).sum() / ((output - output.mean()) ** 2).sum())


def decomposition_summary(*, frame: pl.DataFrame) -> dict[str, float]:
    """Summarise one battery's decomposition over the half-hours with all of its inputs.

    Args:
        frame: `battery_decomposition`'s output for one battery.

    Returns:
        The number of half-hours used, the variance of output explained by the plan alone and by
        the plan plus accepted volume, the remainder's mean and standard deviation, and the
        output's standard deviation.
    """
    used = frame.drop_nulls(["output_mw", "plan_mw"])
    output = used["output_mw"].to_numpy()
    plan = used["plan_mw"].to_numpy()
    accepted = used["accepted_mw"].to_numpy()
    remainder = output - plan - accepted
    return {
        "half_hours": float(len(used)),
        "output_sd_mw": float(output.std()),
        "plan_only_r2": variance_explained(output=output, explained=plan),
        "plan_plus_accepted_r2": variance_explained(output=output, explained=plan + accepted),
        "plan_correlation_squared": float(np.corrcoef(output, plan)[0, 1] ** 2),
        "remainder_mean_mw": float(remainder.mean()),
        "remainder_sd_mw": float(remainder.std()),
        "accepted_mean_abs_mw": float(np.abs(accepted).mean()),
    }


def frequency_slope(*, frame: pl.DataFrame) -> dict[str, float]:
    """Fit the remainder against the frequency deviation by least squares.

    Args:
        frame: `battery_decomposition`'s output for one battery.

    Returns:
        The correlation, the slope in megawatts per 0.1 hertz, and the half-hours used.
    """
    used = frame.drop_nulls(["remainder_mw", "frequency_deviation_hz"])
    x = used["frequency_deviation_hz"].to_numpy()
    y = used["remainder_mw"].to_numpy()
    slope = float(np.polyfit(x, y, 1)[0])
    return {
        "frequency_correlation": float(np.corrcoef(x, y)[0, 1]),
        "mw_per_tenth_hz": slope * 0.1,
        "frequency_half_hours": float(len(used)),
    }


def response_comparison(*, frame: pl.DataFrame) -> dict[str, float]:
    """Compare the remainder in half-hours inside and outside a response-contract block.

    Args:
        frame: `battery_decomposition`'s output for one battery.

    Returns:
        The half-hour counts, the remainder's standard deviation and mean absolute value, and the
        frequency slope of each group. Values are NaN for an empty group.
    """
    used = frame.drop_nulls(["remainder_mw", "frequency_deviation_hz"])
    result: dict[str, float] = {}
    for label, mask in (
        ("with_response", used["contracted_response_mw"] > 0),
        ("without_response", used["contracted_response_mw"] == 0),
    ):
        part = used.filter(mask)
        result[f"{label}_half_hours"] = float(part.height)
        if part.height < MIN_BIN_COUNT:
            result |= dict.fromkeys(
                (
                    f"{label}_remainder_sd_mw",
                    f"{label}_remainder_mean_abs_mw",
                    f"{label}_slope",
                    f"{label}_high_frequency_mean_mw",
                    f"{label}_low_frequency_mean_mw",
                ),
                float("nan"),
            )
            continue
        remainder = part["remainder_mw"].to_numpy()
        x = part["frequency_deviation_hz"].to_numpy()
        result[f"{label}_remainder_sd_mw"] = float(remainder.std())
        result[f"{label}_remainder_mean_abs_mw"] = float(np.abs(remainder).mean())
        result[f"{label}_slope"] = float(np.polyfit(x, remainder, 1)[0] * 0.1)
        result[f"{label}_high_frequency_mean_mw"] = float(remainder[x <= -TAIL_HZ].mean())
        result[f"{label}_low_frequency_mean_mw"] = float(remainder[x >= TAIL_HZ].mean())
        result[f"{label}_high_frequency_half_hours"] = float((x <= -TAIL_HZ).sum())
        result[f"{label}_low_frequency_half_hours"] = float((x >= TAIL_HZ).sum())
    return result


def behaviour_summary(*, frame: pl.DataFrame) -> dict[str, float]:
    """Summarise what a battery does: acceptances, accepted energy, and contract blocks.

    Args:
        frame: `battery_decomposition`'s output for one battery.

    Returns:
        The shares of half-hours with an acceptance, inside a response block, and inside a reserve
        block, and the mean absolute accepted energy over half-hours with an acceptance.
    """
    accepted_mwh = frame.filter(pl.col("has_acceptance"))["accepted_mw"].abs() * HOURS_PER_HALF_HOUR
    return {
        "share_with_acceptance": float(frame["has_acceptance"].mean()),  # ty: ignore[invalid-argument-type]
        "share_in_response_block": float((frame["contracted_response_mw"] > 0).mean()),  # ty: ignore[invalid-argument-type]
        "share_in_reserve_block": float((frame["contracted_reserve_mw"] > 0).mean()),  # ty: ignore[invalid-argument-type]
        "mean_abs_accepted_mwh_when_accepted": float(accepted_mwh.mean()),  # ty: ignore[invalid-argument-type]
        "mean_response_mw_when_contracted": float(
            frame.filter(pl.col("contracted_response_mw") > 0)["contracted_response_mw"].mean()  # ty: ignore[invalid-argument-type]
            or 0.0
        ),
    }


def typical_day(*, frames: pl.DataFrame) -> pl.DataFrame:
    """Return the mean output and mean day-ahead price by half-hour of the UTC day and season.

    Args:
        frames: The batteries' decomposition frames stacked.

    Returns:
        Columns `season`, `tod` (0 to 47), `bmu_id`, `output_mw`, and `day_ahead_gbp_per_mwh`.
    """
    market = market_frame().select("time", "day_ahead_gbp_per_mwh")
    joined = frames.join(market, on="time", how="left").with_columns(
        month=pl.col("time").dt.month(),
        tod=pl.col("time").dt.hour().cast(pl.Int32) * 2 + pl.col("time").dt.minute() // 30,
    )
    parts = []
    for season, months in SEASONS.items():
        parts.append(
            joined.filter(pl.col("month").is_in(months))
            .group_by("bmu_id", "tod")
            .agg(
                pl.col("output_mw").mean(),
                pl.col("day_ahead_gbp_per_mwh").mean(),
            )
            .with_columns(season=pl.lit(season))
        )
    return pl.concat(parts).sort("season", "bmu_id", "tod")


def _percent(*, value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    """Build the frames, the summary table, and the report fragment."""
    frames = pl.concat([battery_decomposition(bmu_id=bmu_id) for bmu_id in BATTERIES])
    frames.write_parquet(FRAME_PATH)
    typical_day(frames=frames).write_parquet(OUTPUT_DIR / "primer_typical_day.parquet")
    lines = [
        "## Primer: what the four batteries do",
        "",
        (
            "Output is B1610 settled output as megawatts (energy times two). The plan is the "
            "half-hourly mean of the Elexon physical notification. The accepted volume is the sum "
            "over acceptances of BOAV `total_volume_accepted_mwh` (a bid is negative and an offer "
            "positive, the sign of a change in output) times two. The remainder is output minus "
            "plan minus accepted volume. Frequency deviation is 50 Hz minus the half-hour's mean "
            "frequency, so a positive value means low frequency, when a battery should export "
            "more. Every figure uses the window 1 September 2025 to 31 August 2026, in UTC."
        ),
        "",
        f"Half-hours in the window: {EXPECTED_ROWS}.",
        "",
    ]
    rows: list[dict[str, Any]] = []
    for bmu_id in BATTERIES:
        frame = frames.filter(pl.col("bmu_id") == bmu_id)
        unit = eac_auction_unit(bmu_id=bmu_id)
        summary = {
            "bmu_id": bmu_id,
            "auction_unit": unit or "",
            "output_missing": frame["output_mw"].null_count(),
            "plan_missing": frame["plan_mw"].null_count(),
            "frequency_missing": frame["frequency_deviation_hz"].null_count(),
            **decomposition_summary(frame=frame),
            **frequency_slope(frame=frame),
            **response_comparison(frame=frame),
            **behaviour_summary(frame=frame),
        }
        rows.append(summary)
    table = pl.DataFrame(rows)
    table.write_parquet(SUMMARY_PATH)
    lines += [
        "### P2: how much of the output the plan and the accepted volumes explain",
        "",
        (
            "Variance explained is one minus the sum of squared differences between output and "
            "the prediction, over the sum of squared deviations of output from its mean; nothing "
            "is fitted. Half-hours with no output or no plan are left out."
        ),
        "",
        (
            "| Battery | Half-hours | Output SD (MW) | Plan alone | Plan plus accepted | "
            "Squared correlation with plan | Remainder mean (MW) | Remainder SD (MW) |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {NAMES[row['bmu_id']]} | {row['half_hours']:.0f} | {row['output_sd_mw']:.1f} | "
            f"{_percent(value=row['plan_only_r2'])} | "
            f"{_percent(value=row['plan_plus_accepted_r2'])} "
            f"| {_percent(value=row['plan_correlation_squared'])} | "
            f"{row['remainder_mean_mw']:.2f} | {row['remainder_sd_mw']:.2f} |"
        )
    lines += [
        "",
        "Missing half-hours (output, plan, frequency): "
        + "; ".join(
            f"{NAMES[r['bmu_id']]} {r['output_missing']}, {r['plan_missing']}, "
            f"{r['frequency_missing']}"
            for r in rows
        )
        + ".",
        "",
        "### P3: the remainder against the grid frequency",
        "",
        (
            "Least-squares slope of the remainder on the frequency deviation, in megawatts per "
            "0.1 Hz of deviation; positive means the battery exports more when frequency is low. "
            "Response-contract status is a candidate: the NESO auction unit is matched to the "
            "battery by identifier only (see `neso_eac_unit_to_bmu/README.md`), and a half-hour "
            "counts as inside a block when any Dynamic Containment, Moderation, or Regulation "
            "delivery period of the candidate unit covers it."
        ),
        "",
        (
            "| Battery | Candidate auction unit | Correlation | Slope, all (MW per 0.1 Hz) | "
            "Half-hours with response | Remainder SD with | Remainder SD without | "
            "Slope with | Slope without |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {NAMES[row['bmu_id']]} | {row['auction_unit']} (candidate) | "
            f"{row['frequency_correlation']:.3f} | {row['mw_per_tenth_hz']:.3f} | "
            f"{row['with_response_half_hours']:.0f} | {row['with_response_remainder_sd_mw']:.2f} | "
            f"{row['without_response_remainder_sd_mw']:.2f} | "
            f"{row['with_response_slope']:.3f} | {row['without_response_slope']:.3f} |"
        )
    lines += [
        "",
        "Half-hours without a response block: "
        + "; ".join(f"{NAMES[r['bmu_id']]} {r['without_response_half_hours']:.0f}" for r in rows)
        + ". Mean absolute remainder with and without: "
        + "; ".join(
            f"{NAMES[r['bmu_id']]} {r['with_response_remainder_mean_abs_mw']:.2f} and "
            f"{r['without_response_remainder_mean_abs_mw']:.2f} MW"
            for r in rows
        )
        + ".",
        "",
        (
            f"Mean remainder (MW) in half-hours with frequency at least {TAIL_HZ} Hz above 50 "
            f"(deviation at most -{TAIL_HZ}, the high-frequency tail) and at least {TAIL_HZ} Hz "
            "below 50 (the low-frequency tail), inside and outside a response block, with "
            "half-hour counts in brackets:"
        ),
        "",
        (
            "| Battery | High frequency, inside | High frequency, outside "
            "| Low frequency, inside | Low frequency, outside |"
        ),
        "|---|---|---|---|---|",
        *[
            f"| {NAMES[r['bmu_id']]} | "
            f"{r['with_response_high_frequency_mean_mw']:.2f} "
            f"({r['with_response_high_frequency_half_hours']:.0f}) | "
            f"{r['without_response_high_frequency_mean_mw']:.2f} "
            f"({r['without_response_high_frequency_half_hours']:.0f}) | "
            f"{r['with_response_low_frequency_mean_mw']:.2f} "
            f"({r['with_response_low_frequency_half_hours']:.0f}) | "
            f"{r['without_response_low_frequency_mean_mw']:.2f} "
            f"({r['without_response_low_frequency_half_hours']:.0f}) |"
            for r in rows
        ],
        "",
        "### P4: what each battery does",
        "",
        (
            "Share of the window's half-hours with a BOALF acceptance ramp touching them "
            "(deleted amendments excluded), mean absolute accepted energy over those half-hours, "
            "and the shares inside a candidate response block and inside a candidate reserve "
            "block (Quick, Slow, or Balancing Reserve)."
        ),
        "",
        (
            "| Battery | With a BM acceptance | Mean absolute accepted MWh | In a response block "
            "(candidate) | In a reserve block (candidate) | Mean response MW when contracted |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {NAMES[row['bmu_id']]} | {_percent(value=row['share_with_acceptance'])} | "
            f"{row['mean_abs_accepted_mwh_when_accepted']:.2f} | "
            f"{_percent(value=row['share_in_response_block'])} | "
            f"{_percent(value=row['share_in_reserve_block'])} | "
            f"{row['mean_response_mw_when_contracted']:.1f} |"
        )
    day = pl.read_parquet(OUTPUT_DIR / "primer_typical_day.parquet")
    lines += ["", "### P1: the typical day", ""]
    for season in SEASONS:
        part = day.filter(pl.col("season") == season)
        price = part.filter(pl.col("bmu_id") == BATTERIES[0]).sort("tod")
        cheapest = int(price["day_ahead_gbp_per_mwh"].arg_min())  # ty: ignore[invalid-argument-type]
        dearest = int(price["day_ahead_gbp_per_mwh"].arg_max())  # ty: ignore[invalid-argument-type]
        lines.append(
            f"- {season}: mean day-ahead price lowest at UTC half-hour {cheapest} "
            f"({price['day_ahead_gbp_per_mwh'][cheapest]:.1f} £ per MWh) and highest at "
            f"half-hour {dearest} ({price['day_ahead_gbp_per_mwh'][dearest]:.1f}). Mean output "
            "lowest and highest half-hour, by battery: "
            + "; ".join(
                f"{NAMES[b]} {int(sub['tod'][int(sub['output_mw'].arg_min())])} "  # ty: ignore[invalid-argument-type]
                f"({sub['output_mw'].min():.1f} MW) and "
                f"{int(sub['tod'][int(sub['output_mw'].arg_max())])} "  # ty: ignore[invalid-argument-type]
                f"({sub['output_mw'].max():.1f} MW)"
                for b in BATTERIES
                for sub in [part.filter(pl.col("bmu_id") == b).sort("tod")]
            )
            + "."
        )
    (OUTPUT_DIR / "report_primer.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
