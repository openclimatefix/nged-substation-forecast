"""Summarise rung 7b: write `report_rung7b.md` from `rung7b_fits.parquet`. Nothing is refitted.

Run: `uv run python studies/battery_pv_separation/battery_rung7b_summary.py`.
"""

from typing import Final

import polars as pl
from battery_inputs import OUTPUT_DIR
from battery_rung7 import CLOUD_SIGNAL_BMUS
from battery_rung7_summary import (
    SOLAR_STUDY_DIFFERENCE_MW,
    SOLAR_STUDY_PHYSICAL_MW,
    _capacity_table,
    _powers_header,
)
from battery_rung7b import POWERS_MW

COMPARISON_POWERS_MW: Final[tuple[float, float]] = (50.0, 200.0)


def _total_row(*, frame: pl.DataFrame, label: str, column: str, scale: float, digits: int) -> str:
    """Return a table row holding the sum of `column` over the BMUs at each assumed power."""
    total = frame.group_by("power_mw").agg(pl.col(column).sum()).sort("power_mw")
    return f"| {label} | " + " | ".join(f"{v / scale:.{digits}f}" for v in total[column]) + " |"


def _capacity(*, frame: pl.DataFrame, bmu: str, power: float) -> float:
    """Return one BMU's fitted solar capacity at one assumed power."""
    return float(
        frame.filter((pl.col("bmu") == bmu) & (pl.col("power_mw") == power))["fitted_ac_mw"][0]
    )


def main() -> None:
    """Write the report fragment."""
    fits = pl.read_parquet(OUTPUT_DIR / "rung7b_fits.parquet")
    bmus = sorted(fits["bmu"].unique().to_list())
    cloud = [b for b in bmus if b in CLOUD_SIGNAL_BMUS]
    other = [b for b in bmus if b not in CLOUD_SIGNAL_BMUS]
    real = fits.filter(pl.col("source") == "real")
    replica = fits.filter(pl.col("source") == "replica")
    header, rule = _powers_header(powers=POWERS_MW)
    header = header.replace("| BMU |", "| Quantity |")
    lines = [
        "## Rung 7b: the joint model with a calendar baseline on the real aggregate BMUs",
        "",
        (
            "Rung 7 had no calendar baseline, so its solar part absorbed every regular daily and "
            "seasonal shape of a BMU's output, and it could not test the solar study's totals. "
            "Rung 7b fits `output = solar + calendar baseline + battery` to the same 25 BMUs. "
            "Solar is the four fleet curves under the same regional CAMS sky as the solar study's "
            "stage 3. The baseline is the solar study's seasonal design (an indicator for each "
            "local half-hour of the day and day type, plus two annual harmonics for each "
            "half-hour) with free signed coefficients. The solar study fitted that baseline with "
            "a small ridge penalty. A linear programme cannot hold a ridge penalty, so the fit "
            "uses a small absolute penalty (0.001 per unit of coefficient, against 1 per "
            "megawatt of residual) instead. The fit minimises the absolute residual of "
            "half-hourly levels in windows of about 4 weeks, with the solar weights and the "
            "baseline coefficients shared by all windows. The battery has power `P`, energy "
            "`2 h x P`, and a one-way efficiency of 0.92, and `P = 0` has no battery. Each "
            "BMU's calendar replica (the mean output of its month, local half-hour, and day "
            "type) is fitted the same way, and it has no cloud, so solar fitted to a replica "
            "could be shape alone."
        ),
        "",
        (
            f"The solar study's totals over the same 25 BMUs are {SOLAR_STUDY_DIFFERENCE_MW:,.0f} "
            f"MW (difference separation) and {SOLAR_STUDY_PHYSICAL_MW:,.0f} MW (physical fit). "
            "The solar study's difference separation fitted four-hour changes, and its physical "
            "fit used a free plant model. The `P = 0` column here minimises the absolute "
            "residual of half-hourly levels, so it is a different fit from both."
        ),
        "",
        "### Total fitted solar capacity against assumed battery power",
        "",
        header,
        rule,
        _total_row(
            frame=real,
            label="All 25 BMUs, real output (MW)",
            column="fitted_ac_mw",
            scale=1,
            digits=1,
        ),
        _total_row(
            frame=replica,
            label="All 25 BMUs, calendar replicas (MW)",
            column="fitted_ac_mw",
            scale=1,
            digits=1,
        ),
        _total_row(
            frame=real.filter(pl.col("bmu").is_in(cloud)),
            label="Nine cloud-signal BMUs, real output (MW)",
            column="fitted_ac_mw",
            scale=1,
            digits=1,
        ),
        _total_row(
            frame=replica.filter(pl.col("bmu").is_in(cloud)),
            label="Nine cloud-signal BMUs, calendar replicas (MW)",
            column="fitted_ac_mw",
            scale=1,
            digits=1,
        ),
        _total_row(
            frame=real,
            label="Fitted solar energy, all 25 BMUs, real output (GWh)",
            column="solar_energy_mwh",
            scale=1000,
            digits=1,
        ),
        _total_row(
            frame=real,
            label="Battery energy throughput, all 25 BMUs, real output (GWh)",
            column="battery_throughput_mwh",
            scale=1000,
            digits=1,
        ),
    ]
    gap = (
        real.join(replica, on=["bmu", "power_mw"], suffix="_replica")
        .with_columns(gap=pl.col("fitted_ac_mw") - pl.col("fitted_ac_mw_replica"))
        .group_by("power_mw")
        .agg(pl.col("gap").sum())
        .sort("power_mw")
    )
    lines.append(
        "| Real minus replica fitted solar capacity, all 25 BMUs (MW) | "
        + " | ".join(f"{v:.1f}" for v in gap["gap"])
        + " |"
    )
    for source, frame in (("real output", real), ("calendar replica", replica)):
        residual = (
            frame.group_by("power_mw")
            .agg(pl.col("residual_relative_to_no_battery").median())
            .sort("power_mw")
        )
        lines.append(
            f"| Residual relative to P = 0, median over BMUs, {source} | "
            + " | ".join(f"{v:.2f}" for v in residual["residual_relative_to_no_battery"])
            + " |"
        )
    lines += [
        "",
        (
            "Fitted capacity over all 25 BMUs compared with the solar study's totals: at "
            f"`P = 0` the real-output total is {_total(real, 0.0):,.1f} MW, "
            f"{_total(real, 0.0) / SOLAR_STUDY_DIFFERENCE_MW:.2f} times the difference "
            f"separation's {SOLAR_STUDY_DIFFERENCE_MW:,.0f} MW and "
            f"{_total(real, 0.0) / SOLAR_STUDY_PHYSICAL_MW:.2f} times the physical fit's "
            f"{SOLAR_STUDY_PHYSICAL_MW:,.0f} MW; at `P = 200` MW it is "
            f"{_total(real, 200.0):,.1f} MW. The calendar replicas total "
            f"{_total(replica, 0.0):,.1f} MW at `P = 0` and {_total(replica, 200.0):,.1f} MW at "
            "`P = 200` MW."
        ),
        "",
        "### Cloud-signal BMUs: fitted solar capacity (MW) against assumed battery power (MW)",
        "",
        "Real output:",
        "",
        *_capacity_table(fits=fits, source="real", bmus=cloud, powers=POWERS_MW),
        "",
        "Calendar replicas:",
        "",
        *_capacity_table(fits=fits, source="replica", bmus=cloud, powers=POWERS_MW),
        "",
        "### The other 16 BMUs: fitted solar capacity (MW)",
        "",
        "Real output:",
        "",
        *_capacity_table(fits=fits, source="real", bmus=other, powers=POWERS_MW),
        "",
        "Calendar replicas:",
        "",
        *_capacity_table(fits=fits, source="replica", bmus=other, powers=POWERS_MW),
        "",
        "### Cloud-signal BMUs: change with the assumed battery",
        "",
        (
            "| BMU | P = 0 (MW) | P = 50 (MW) | P = 200 (MW) | Ratio at 50 MW | Ratio at 200 MW | "
            "Replica at P = 0 (MW) | Replica at P = 200 (MW) | Share of half-hours at the "
            "battery limit, P = 50 |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for bmu in cloud:
        base = _capacity(frame=real, bmu=bmu, power=0.0)
        mid = _capacity(frame=real, bmu=bmu, power=COMPARISON_POWERS_MW[0])
        high = _capacity(frame=real, bmu=bmu, power=COMPARISON_POWERS_MW[1])
        limit = real.filter((pl.col("bmu") == bmu) & (pl.col("power_mw") == 50.0))[
            "share_at_battery_limit"
        ][0]
        lines.append(
            f"| {bmu} | {base:.1f} | {mid:.1f} | {high:.1f} | "
            f"{mid / base if base > 0 else float('nan'):.2f} | "
            f"{high / base if base > 0 else float('nan'):.2f} | "
            f"{_capacity(frame=replica, bmu=bmu, power=0.0):.1f} | "
            f"{_capacity(frame=replica, bmu=bmu, power=200.0):.1f} | {limit:.1%} |"
        )
    (OUTPUT_DIR / "report_rung7b.md").write_text("\n".join(lines))
    print("\n".join(lines))


def _total(frame: pl.DataFrame, power: float) -> float:
    """Return the total fitted solar capacity over the BMUs at one assumed power."""
    return float(frame.filter(pl.col("power_mw") == power)["fitted_ac_mw"].sum())


if __name__ == "__main__":
    main()
