# ruff: noqa: E501
"""Refit the plant's parameters on forecast weather and on observed weather (experiment 270, arm P).

`p_ens` fits tilt, azimuth, heat loss, scale, and ceiling on the farm's tuning months with the
ensemble-mean weather as input. `p_cams` fits the same parameters on the same months with the
Copernicus Atmosphere Monitoring Service (CAMS) irradiance and ERA5-style temperature stored in
experiment 218's farm table, which is observed (estimated past) weather and is not available at
forecast time. Both parameter sets are then applied to the ensemble-mean weather of the scored
year, with the four-bin residual offsets recomputed on the tuning months' ensemble forecast, so the
two arms differ only in the parameters. Own history only. Run:

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00270-*/e00270_plantfit.py
"""

import itertools
from typing import Final

import e00270_common as common
import e00270_features as feats
import e00270_fit as fit
import numpy as np
import polars as pl
from studies.pv_physics import SunAndSky, plane_of_array_w_m2

TILTS_DEG: Final[tuple[float, ...]] = (10.0, 20.0, 30.0, 35.0, 40.0, 50.0, 60.0)
AZIMUTHS_DEG: Final[tuple[float, ...]] = (120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0)
HEAT_PER_KELVIN: Final[tuple[float, ...]] = (0.0, 0.002, 0.004, 0.006)
RISE_K: Final[float] = 25.0


def unit_power(
    *, sky: dict[str, np.ndarray], temperature: np.ndarray, tilt: float, azimuth: float, heat: float
) -> np.ndarray:
    """Return the plant's unit response (plane-of-array over 1000 W/m2 times the heat derating)."""
    plane = plane_of_array_w_m2(
        sky=SunAndSky(
            half_hour_end_time=np.zeros(len(temperature)),
            ghi_w_m2=sky["ghi"],
            dni_w_m2=sky["dni"],
            dhi_w_m2=sky["dhi"],
            zenith_deg=sky["zenith"],
            sun_azimuth_deg=sky["azimuth"],
        ),
        tilt_deg=tilt,
        azimuth_deg=azimuth,
        tracker=False,
    )
    unit = plane / 1000.0
    cell = temperature + RISE_K * unit
    return unit * np.clip(1.0 - heat * (cell - 25.0), 0.0, None)


def best_parameters(
    *, sky: dict[str, np.ndarray], temperature: np.ndarray, output: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Return (tilt, azimuth, heat, scale, ceiling) with the lowest squared error on the given rows."""
    best = (np.inf, 0.0, 0.0, 0.0, 0.0)
    for tilt, azimuth, heat in itertools.product(TILTS_DEG, AZIMUTHS_DEG, HEAT_PER_KELVIN):
        x = unit_power(sky=sky, temperature=temperature, tilt=tilt, azimuth=azimuth, heat=heat)
        scale = float(np.sum(output * x) / np.sum(x * x))
        error = float(np.sum((output - np.minimum(scale * x, common_ceiling(output))) ** 2))
        if error < best[0]:
            best = (error, tilt, azimuth, heat, scale)
    return best[1], best[2], best[3], best[4], common_ceiling(output)


def common_ceiling(output: np.ndarray) -> float:
    """Return the ceiling: the 0.995 quantile of the tuning months' output, as in experiment 243."""
    return float(np.quantile(output, common.CEILING_QUANTILE))


def sky_of(*, frame: pl.DataFrame, names: tuple[str, str, str, str, str]) -> dict[str, np.ndarray]:
    """Return the sun and sky arrays from the named ghi, dni, dhi, zenith, azimuth columns."""
    return {
        key: frame[name].to_numpy().astype(float)
        for key, name in zip(("ghi", "dni", "dhi", "zenith", "azimuth"), names, strict=True)
    }


def run(site: str) -> list[str]:
    """Fit and write both arms for one farm."""
    f = common.load_farm(site)
    frame = f.frame
    cams = (
        pl.read_parquet(common.FARM_ROWS)
        .filter(pl.col("site") == site)
        .select("time", "ghi_w_m2", "dni_w_m2", "dhi_w_m2", "temperature_c")
    )
    joined = frame.join(cams, on="time", how="left")
    tuning = f.tuning & joined["ghi_w_m2"].is_not_null().to_numpy()
    output = joined["output"].to_numpy().astype(float)
    ens_sky = sky_of(
        frame=joined, names=("ghi_mean", "dni_mean", "dhi_mean", "zenith_deg", "sun_azimuth_deg")
    )
    cams_sky = sky_of(
        frame=joined.fill_null(0.0),
        names=("ghi_w_m2", "dni_w_m2", "dhi_w_m2", "zenith_deg", "sun_azimuth_deg"),
    )
    ens_t = joined["temperature_mean"].to_numpy()
    cams_t = joined["temperature_c"].fill_null(0.0).to_numpy()
    sub = lambda sky, m: {k: v[m] for k, v in sky.items()}  # noqa: E731
    parameters = {
        "p_ens": best_parameters(
            sky=sub(ens_sky, tuning), temperature=ens_t[tuning], output=output[tuning]
        ),
        "p_cams": best_parameters(
            sky=sub(cams_sky, tuning), temperature=cams_t[tuning], output=output[tuning]
        ),
    }
    st = fit.Setting(setting="own_history", site=site)
    lines = []
    for arm, (tilt, azimuth, heat, scale, ceiling) in parameters.items():
        mu_all = np.minimum(
            scale
            * unit_power(sky=ens_sky, temperature=ens_t, tilt=tilt, azimuth=azimuth, heat=heat),
            ceiling,
        )
        edges, table = common.bin_offsets(mu=mu_all[f.tuning], output=output[f.tuning])
        mu = mu_all[f.scored]
        q = mu[:, None] + common.offsets_for(mu=mu, edges=edges, table=table)
        q = np.minimum(np.maximum(np.sort(q, axis=1), 0.0), ceiling)
        fit.write(st=st, setting="own_history", arm=arm, seed=0, quantiles=q, crossing=0.0)
        lines.append(
            f"{site} {arm} tilt={tilt} azimuth={azimuth} heat={heat} scale={scale:.3f} ceiling={ceiling:.3f}"
        )
    lines += [
        binned_offsets_arm(f=f, st=st, output=output, spec=spec) for spec in BINNED_PLANT_ARMS
    ]
    return lines


def binned_offsets_arm(
    *, f: common.Farm, st: fit.Setting, output: np.ndarray, spec: tuple[str, str, int, int]
) -> str:
    """Write one plant arm whose offsets are binned by the forecast and by a second variable.

    The plant forecast is experiment 243's (fleet shape, own scale). Its 11 residual quantiles come
    from the tuning months, in `n_mu` quantile bins of the forecast crossed with `n_var` quantile
    bins of the second variable (a cell with fewer than `common.MIN_BIN_ROWS` rows falls back to the
    four-bin forecast table). It tests whether the gain of XGBoost over the plant is only a band
    whose width depends on one more plant input. The bins come from tuning rows only.
    """
    name, variable, n_mu, n_var = spec
    mu_all = common.plant_forecast(farm=f, scale=st.scale, ceiling=st.ceiling)
    if variable == "plant_spread":
        values = np.minimum(st.scale * feats.members_unit(f.site).std(axis=1), st.ceiling)
    else:
        values = f.frame[variable].to_numpy().astype(float)
    _, fallback = common.bin_offsets(mu=mu_all[f.tuning], output=output[f.tuning])
    usable = f.tuning & (mu_all > common.MIN_FORECAST_POWER)
    mu_edges = np.quantile(mu_all[usable], np.linspace(0, 1, n_mu + 1)[1:-1])
    var_edges = np.quantile(values[usable], np.linspace(0, 1, n_var + 1)[1:-1])
    mu_bin, var_bin = np.digitize(mu_all, mu_edges), np.digitize(values, var_edges)
    four_bin_edges, _ = common.bin_offsets(mu=mu_all[f.tuning], output=output[f.tuning])
    coarse = np.digitize(mu_all, four_bin_edges)
    residual = output - mu_all
    cells = np.empty((n_mu, n_var, len(common.LEVELS)))
    for i in range(n_mu):
        for j in range(n_var):
            rows = f.tuning & (mu_bin == i) & (var_bin == j)
            cells[i, j] = (
                np.quantile(residual[rows], common.LEVELS)
                if rows.sum() >= common.MIN_BIN_ROWS
                else fallback[int(np.bincount(coarse[rows]).argmax()) if rows.any() else 0]
            )
    scored = f.scored
    q = mu_all[scored][:, None] + cells[mu_bin[scored], var_bin[scored]]
    q = np.minimum(np.maximum(np.sort(q, axis=1), 0.0), st.ceiling)
    fit.write(st=st, setting="own_history", arm=name, seed=0, quantiles=q, crossing=0.0)
    return f"{f.site} {name} written"


BINNED_PLANT_ARMS: Final[tuple[tuple[str, str, int, int], ...]] = (
    ("pq_elev", "elevation_deg", 4, 4),
    ("pq_csi", "csi", 8, 4),
    ("pq_spread", "plant_spread", 6, 3),
)


def main() -> None:
    """Fit every farm and write the parameters beside the predictions."""
    rows = []
    for site in common.FARMS:
        for line in run(site):
            print(line, flush=True)
            rows.append(line)
    (common.DATA_DIR / "plant_parameters.txt").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
