"""Summarise rung 7c, run 1: write `report_rung7c_costs.md` from `rung7c_costs_fits.parquet`.

Nothing is refitted.

Run: `uv run python studies/battery_pv_separation/battery_rung7c_costs_summary.py`.
"""

import polars as pl
from battery_inputs import OUTPUT_DIR
from battery_rung7 import CLOUD_SIGNAL_BMUS


def cost_table(*, fits: pl.DataFrame) -> pl.DataFrame:
    """Return one row per cost setting and assumed power with the totals the report tables show.

    Args:
        fits: `rung7c_costs_fits.parquet`.

    Returns:
        Columns `power_mw`, `throughput_cost`, `solar_cost`, the total fitted solar capacity of the
        real outputs of all 25 BMUs and of the 9 cloud-signal BMUs, the same for the calendar
        replicas, and the median over BMUs of the real outputs' residual relative to the `P = 0`
        fit with the same solar cost, and of the share of half-hours that fit exactly.
    """
    reference = (
        fits.filter(pl.col("power_mw") == 0.0)
        .select("bmu", "source", "solar_cost", no_battery_residual="residual_abs_sum_mw")
        .unique()
    )
    scored = fits.join(reference, on=["bmu", "source", "solar_cost"]).with_columns(
        relative_residual=pl.col("residual_abs_sum_mw") / pl.col("no_battery_residual"),
        cloud=pl.col("bmu").is_in(CLOUD_SIGNAL_BMUS),
    )
    keys = ["power_mw", "throughput_cost", "solar_cost"]
    real = scored.filter(pl.col("source") == "real")
    replica = scored.filter(pl.col("source") == "replica")
    return (
        real.group_by(keys)
        .agg(
            real_all_mw=pl.col("fitted_ac_mw").sum(),
            real_cloud_mw=pl.col("fitted_ac_mw").filter(pl.col("cloud")).sum(),
            median_relative_residual=pl.col("relative_residual").median(),
            median_share_exact=pl.col("share_exact_fit").median(),
        )
        .join(
            replica.group_by(keys).agg(
                replica_all_mw=pl.col("fitted_ac_mw").sum(),
                replica_cloud_mw=pl.col("fitted_ac_mw").filter(pl.col("cloud")).sum(),
            ),
            on=keys,
        )
        .sort(keys)
    )


def main() -> None:
    """Write the report fragment."""
    fits = pl.read_parquet(OUTPUT_DIR / "rung7c_costs_fits.parquet")
    table = cost_table(fits=fits)
    lines = [
        "## Rung 7c: cost sensitivity",
        "",
        (
            "Rung 7b's fitted solar capacity stays near 1.15 to 1.22 GW at every assumed battery "
            "power. The linear programme charges 0.1 per megawatt of battery throughput (against "
            "1 per megawatt of residual), nothing for solar capacity, and 0.001 per unit of "
            "baseline coefficient. At a large battery power the median residual falls to 0 in "
            "many half-hours, so many solutions fit equally well and the costs choose among "
            "them. This run refits rung 7b's model on the 25 real aggregate BMUs and their "
            "calendar replicas at `P = 50` and `200` MW, varying the battery throughput cost "
            "and adding a cost per megawatt of fitted solar capacity. Each solar cost also has a "
            "`P = 0` fit, which has no battery and is the reference for the residual."
        ),
        "",
        (
            "| Throughput cost | Solar cost | P (MW) | Real, 25 BMUs (MW) | Real, 9 cloud-signal "
            "BMUs (MW) | Replicas, 25 BMUs (MW) | Replicas, 9 cloud-signal BMUs (MW) | Median "
            "residual relative to P = 0 | Median share of half-hours fitted exactly |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table.iter_rows(named=True):
        throughput = "none" if row["power_mw"] == 0 else f"{row['throughput_cost']:g}"
        lines.append(
            f"| {throughput} | {row['solar_cost']:g} | {row['power_mw']:g} | "
            f"{row['real_all_mw']:.0f} | {row['real_cloud_mw']:.0f} | "
            f"{row['replica_all_mw']:.0f} | {row['replica_cloud_mw']:.0f} | "
            f"{row['median_relative_residual']:.3f} | {row['median_share_exact']:.1%} |"
        )
    (OUTPUT_DIR / "report_rung7c_costs.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
