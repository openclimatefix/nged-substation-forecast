"""Summarise rung 7: write `report_rung7.md` from `rung7_fits.parquet`. Nothing is refitted.

Run: `uv run python studies/battery_pv_separation/battery_rung7_summary.py`.
"""

from typing import Final

import polars as pl
from battery_inputs import OUTPUT_DIR
from battery_rung7 import CLOUD_SIGNAL_BMUS, POWERS_MW

SOLAR_STUDY_PHYSICAL_MW: Final[float] = 1163.0
SOLAR_STUDY_DIFFERENCE_MW: Final[float] = 1089.0
"""The solar study's totals over the same 25 BMUs, from its physical fit and its regional
difference separation."""
COMPARISON_POWERS_MW: Final[tuple[float, float]] = (50.0, 200.0)


def _powers_header(*, powers: tuple[float, ...] = POWERS_MW) -> tuple[str, str]:
    """Return the markdown header and rule of a table with one column per assumed power."""
    labels = " | ".join(f"{p:g}" for p in powers)
    return f"| BMU | {labels} |", "|---|" + "---|" * len(powers)


def _capacity_table(
    *, fits: pl.DataFrame, source: str, bmus: list[str], powers: tuple[float, ...] = POWERS_MW
) -> list[str]:
    """Return a table of fitted solar capacity by BMU and assumed power.

    Args:
        fits: `rung7_fits.parquet`.
        source: `real` or `replica`.
        bmus: The BMUs to list, in order.
        powers: The assumed powers, one column each.

    Returns:
        The markdown lines, with a total row.
    """
    header, rule = _powers_header(powers=powers)
    lines = [header, rule]
    wide = fits.filter(pl.col("source") == source).pivot(
        on="power_mw", index="bmu", values="fitted_ac_mw"
    )
    for bmu in bmus:
        row = wide.filter(pl.col("bmu") == bmu).row(0, named=True)
        lines.append(f"| {bmu} | " + " | ".join(f"{row[str(float(p))]:.1f}" for p in powers) + " |")
    totals = wide.filter(pl.col("bmu").is_in(bmus)).select(
        [pl.col(str(float(p))).sum() for p in powers]
    )
    lines.append(
        "| **Total** | " + " | ".join(f"{totals[str(float(p))][0]:.1f}" for p in powers) + " |"
    )
    return lines


def main() -> None:
    """Write the report fragment."""
    fits = pl.read_parquet(OUTPUT_DIR / "rung7_fits.parquet")
    bmus = sorted(fits["bmu"].unique().to_list())
    cloud = [b for b in bmus if b in CLOUD_SIGNAL_BMUS]
    other = [b for b in bmus if b not in CLOUD_SIGNAL_BMUS]
    real = fits.filter(pl.col("source") == "real")
    replica = fits.filter(pl.col("source") == "replica")
    p0_total = float(real.filter(pl.col("power_mw") == 0.0)["fitted_ac_mw"].sum())
    lines = [
        "## Rung 7: the joint model on the real aggregate BMUs",
        "",
        (
            "The solar study could not tell solar from batteries in its 25 aggregate BMUs. Each "
            "BMU's real output is fitted with the joint model of rung 6: four fleet curves under "
            "the regional CAMS sky of the BMU's GSP group, plus a battery of assumed power `P` "
            "with energy `2 h x P` and a one-way efficiency of 0.92. The state of charge starts "
            "free in each window of about 4 weeks. `P = 0` is solar only. **The battery size of "
            "these BMUs is not known and the fit cannot identify it, and no ground truth exists "
            "for their solar capacity.** The tables show how far the fitted solar capacity moves "
            "with the assumed battery, not how close it is to a true value."
        ),
        "",
        (
            "The solar study's physical fit totals "
            f"{SOLAR_STUDY_PHYSICAL_MW:,.0f} MW over the same 25 BMUs, and its regional "
            f"difference separation {SOLAR_STUDY_DIFFERENCE_MW:,.0f} MW. Those methods fitted a "
            "seasonal calendar baseline as well, so the `P = 0` column here is a different "
            "model, fitted to half-hourly levels, and not a reproduction of either. Its total "
            f"over the 25 BMUs is {p0_total:,.1f} MW. The reason for the difference was not "
            "investigated."
        ),
        "",
        "### Fitted solar capacity (MW) against assumed battery power (MW)",
        "",
        "The nine BMUs with a cloud-correlated component, real output:",
        "",
        *_capacity_table(fits=fits, source="real", bmus=cloud),
        "",
        "The other 16 BMUs, real output:",
        "",
        *_capacity_table(fits=fits, source="real", bmus=other),
        "",
        "All 25 BMUs, real output, are in the two tables above; the total over all 25:",
        "",
    ]
    header, rule = _powers_header()
    header = header.replace("| BMU |", "| Quantity |")
    lines += [header, rule]
    for source, frame in (("Real output", real), ("Calendar replica", replica)):
        total = frame.group_by("power_mw").agg(pl.col("fitted_ac_mw").sum()).sort("power_mw")
        lines.append(
            f"| Fitted solar capacity, {source.lower()} (MW) | "
            + " | ".join(f"{v:.1f}" for v in total["fitted_ac_mw"])
            + " |"
        )
    gap = (
        real.join(replica, on=["bmu", "power_mw"], suffix="_replica")
        .with_columns(gap=pl.col("fitted_ac_mw") - pl.col("fitted_ac_mw_replica"))
        .filter(pl.col("bmu").is_in(cloud))
        .group_by("power_mw")
        .agg(pl.col("gap").sum())
        .sort("power_mw")
    )
    lines.append(
        "| Real minus replica fitted solar capacity, nine cloud-signal BMUs (MW) | "
        + " | ".join(f"{v:.1f}" for v in gap["gap"])
        + " |"
    )
    for source, frame in (("real output", real), ("calendar replica", replica)):
        energy = frame.group_by("power_mw").agg(pl.col("solar_energy_mwh").sum()).sort("power_mw")
        lines.append(
            f"| Fitted solar energy, {source} (GWh) | "
            + " | ".join(f"{v / 1000:.1f}" for v in energy["solar_energy_mwh"])
            + " |"
        )
    throughput = (
        real.group_by("power_mw").agg(pl.col("battery_throughput_mwh").sum()).sort("power_mw")
    )
    lines.append(
        "| Battery energy throughput, real output (GWh) | "
        + " | ".join(f"{v / 1000:.1f}" for v in throughput["battery_throughput_mwh"])
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
            "The residual is the sum of absolute half-hourly differences between the output and "
            "the fit, divided by the same sum at `P = 0`. A battery can follow any smooth daily "
            "shape, so the residual falls with `P` for the calendar replica too, which has no "
            "solar cloud signal. A lower residual with a bigger battery is therefore not evidence "
            "that the battery exists."
        ),
        "",
        "### Cloud-signal BMUs: change in fitted solar capacity",
        "",
        (
            "| BMU | P = 0 (MW) | P = 50 (MW) | P = 200 (MW) | Ratio at 50 MW | Ratio at 200 MW | "
            "Replica at P = 0 (MW) | Replica at 200 MW (MW) |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]

    def ratio(value: float, base: float) -> str:
        return f"{value / base:.2f}" if base > 0 else "n/a"

    def capacity(*, frame: pl.DataFrame, bmu: str, power: float) -> float:
        return float(
            frame.filter((pl.col("bmu") == bmu) & (pl.col("power_mw") == power))["fitted_ac_mw"][0]
        )

    for bmu in cloud:
        base = capacity(frame=real, bmu=bmu, power=0.0)
        mid = capacity(frame=real, bmu=bmu, power=COMPARISON_POWERS_MW[0])
        high = capacity(frame=real, bmu=bmu, power=COMPARISON_POWERS_MW[1])
        lines.append(
            f"| {bmu} | {base:.1f} | {mid:.1f} | {high:.1f} | {ratio(mid, base)} "
            f"| {ratio(high, base)} "
            f"| {capacity(frame=replica, bmu=bmu, power=0.0):.1f} "
            f"| {capacity(frame=replica, bmu=bmu, power=200.0):.1f} |"
        )
    ratios = [
        capacity(frame=real, bmu=b, power=200.0) / capacity(frame=real, bmu=b, power=0.0)
        for b in cloud
        if capacity(frame=real, bmu=b, power=0.0) > 0
    ]
    cloud_base = sum(capacity(frame=real, bmu=b, power=0.0) for b in cloud)
    cloud_high = sum(capacity(frame=real, bmu=b, power=200.0) for b in cloud)
    lines += [
        "",
        (
            f"Over the nine BMUs the fitted capacity goes from {cloud_base:.1f} MW at `P = 0` to "
            f"{cloud_high:.1f} MW at `P = 200` MW (ratio {cloud_high / cloud_base:.2f}); the "
            f"ratio at 200 MW for the {len(ratios)} BMUs with a nonzero fit at `P = 0` ranges from "
            f"{min(ratios):.2f} to {max(ratios):.2f}."
        ),
        "",
        (
            "**What the tables show.** The fitted solar capacity of a BMU falls as the assumed "
            "battery grows for the four BMUs with the largest fits (2__ATGPL000, 2__HTGPL000, "
            "2__DRWED000, 2__LTGPL000), and by a similar share in their calendar replicas. A "
            "replica has no cloud, so most of the fitted solar capacity in this model comes "
            "from the BMU's daily and seasonal shape, which a battery can also produce, and "
            "not from the cloud signal. The replica control therefore does not stay near "
            "zero, unlike the battery-only aggregates of rung 6. The gap between a real BMU "
            "and its replica is the part that follows the real weather; its total over the "
            "nine BMUs is in the totals table above. Nothing here says which battery size is "
            "right: the residual falls steadily with `P` for real output and replica alike."
        ),
        "",
        "### Calendar replicas (controls)",
        "",
        "The replicas of the nine cloud-signal BMUs:",
        "",
        *_capacity_table(fits=fits, source="replica", bmus=cloud),
        "",
        "The replicas of the other 16 BMUs:",
        "",
        *_capacity_table(fits=fits, source="replica", bmus=other),
    ]
    (OUTPUT_DIR / "report_rung7.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
