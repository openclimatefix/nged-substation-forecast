"""Rung 2: a simulated fleet of small batteries that move as one.

No ground truth exists for any real domestic fleet, so this rung is simulated. Each fleet is built
home by home (`studies.battery_templates.simulate_fleet`): a 3 to 5 kW battery per home, a usable
duration, a round-trip efficiency, and a tariff drawn for each home. The draws are centred away
from the estimator's priors (a median duration of 2.6 hours against the prior's 2.0, and a mean
round-trip efficiency of 0.82 against 0.87). One fleet is drawn for each share and scaled to every
series so that its rated power is that share of the series' 99th percentile absolute flow.

Rung 2 is 9 series x 4 blocks x 7 shares = 252 sums. Each is fitted twice: with the real tariff
windows and with every window moved one hour earlier (the negative control: a real window-edge
signal should beat the moved windows). Writes `rung2_posteriors.parquet` (real windows),
`rung2_shifted_posteriors.parquet` (windows one hour early), and `rung2_fleets.parquet`.

Run: `OMP_NUM_THREADS=2 uv run python studies/unmetered_battery_capacity/capacity_rung2.py`
"""

import time

import numpy as np
import polars as pl
from capacity_inputs import OUTPUT_DIR, agile_prices, demand_series, window_half_hours
from capacity_runs import N_BLOCKS, SHARES, fit_block, p99_flow, posterior_row
from capacity_state_space import estimator
from studies.battery_templates import FleetSpec, simulate_fleet

SEED = 20261010
TARIFF_SHARES: dict = {
    "intelligent_octopus_go": 0.4,
    "octopus_go": 0.2,
    "octopus_flux": 0.1,
    "agile": 0.3,
}
"""The share of homes on each tariff in every simulated fleet. The shares are assumptions, not
measurements: no register of home tariffs is available."""
MEAN_HOME_POWER_MW: float = 0.004
SHIFT_HOURS: float = -1.0


def build() -> tuple[np.ndarray, list[list[dict]], list[dict]]:
    """Simulate the fleets and the aggregates.

    Returns:
        Aggregates of shape (series, shares, 17,520), their identifiers and truth, and one row per
        fleet describing it.
    """
    series = demand_series()
    reference_p99 = float(np.median([p99_flow(v) for v in series.values()]))
    fleets = {}
    fleet_rows = []
    for k, share in enumerate(SHARES):
        n_homes = max(5, round(share * reference_p99 / MEAN_HOME_POWER_MW))
        started = time.monotonic()
        fleets[share] = simulate_fleet(
            half_hour_end_time=window_half_hours(),
            agile_prices=agile_prices(),
            n_homes=n_homes,
            spec=FleetSpec(tariff_shares=TARIFF_SHARES),
            rng=np.random.default_rng(SEED + k),
        )
        homes = fleets[share].homes
        rated = float(homes["power_mw"].sum())
        energy = float((homes["power_mw"] * homes["duration_hours"]).sum())
        fleet_rows.append(
            {
                "share": share,
                "n_homes": n_homes,
                "rated_power_per_reference_mw": rated,
                "usable_hours_power_weighted": energy / rated,
                "round_trip_power_weighted": float(
                    (homes["power_mw"] * homes["round_trip_efficiency"]).sum()
                )
                / rated,
                "seconds": time.monotonic() - started,
            }
        )
        print(
            f"fleet {share:.3f}: {n_homes} homes in {time.monotonic() - started:.0f} s", flush=True
        )
    aggregates, metadata = [], []
    for label, demand in series.items():
        p99 = p99_flow(demand)
        lanes, meta = [], []
        for share in SHARES:
            fleet = fleets[share]
            homes = fleet.homes
            rated = float(homes["power_mw"].sum())
            scale = share * p99 / rated
            lanes.append(demand - scale * fleet.output_mw)
            meta.append(
                {
                    "rung": "rung2",
                    "series": label,
                    "share": share,
                    "true_power_mw": share * p99,
                    "true_energy_mwh": scale
                    * float((homes["power_mw"] * homes["duration_hours"]).sum()),
                    "coincident_peak_mw": scale * float(np.quantile(np.abs(fleet.output_mw), 0.99)),
                    "n_homes": homes.height,
                }
            )
        aggregates.append(np.stack(lanes))
        metadata.append(meta)
    return np.stack(aggregates), metadata, fleet_rows


def fit(
    *, aggregates: np.ndarray, metadata: list[list[dict]], window_shift_hours: float
) -> pl.DataFrame:
    """Fit every block with the windows moved by `window_shift_hours`."""
    rows = []
    for block in range(N_BLOCKS):
        model = estimator(setting="standard", block=block, window_shift_hours=window_shift_hours)
        result, seconds = fit_block(block=block, aggregates=aggregates, model=model)
        print(
            f"rung 2 (shift {window_shift_hours:+.0f} h) block {block}: {seconds:.0f} s", flush=True
        )
        for g, group in enumerate(metadata):
            for lane, meta in enumerate(group):
                rows.append(
                    {
                        **meta,
                        "block": block,
                        "window_shift_hours": window_shift_hours,
                        "fit_seconds": seconds,
                        **posterior_row(fit=result, group=g, lane=lane, seed=1000 * block + g),
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def main() -> None:
    """Fit rung 2 with the real windows and with the moved windows."""
    aggregates, metadata, fleet_rows = build()
    pl.DataFrame(fleet_rows).write_parquet(OUTPUT_DIR / "rung2_fleets.parquet")
    fit(aggregates=aggregates, metadata=metadata, window_shift_hours=0.0).write_parquet(
        OUTPUT_DIR / "rung2_posteriors.parquet"
    )
    fit(aggregates=aggregates, metadata=metadata, window_shift_hours=SHIFT_HOURS).write_parquet(
        OUTPUT_DIR / "rung2_shifted_posteriors.parquet"
    )


if __name__ == "__main__":
    main()
