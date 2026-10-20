"""The plumbing controls P0 to P4 of experiment 118, run on the tuning months only.

- **P0, initial state:** prints the starting spread of every weight and the pinned-capacity vintage
  used for each monthly cut against the cut's date.
- **P1, nesting:** the effects filter with the 10 non-level weights held equals experiment 111's
  per-band filter. Part (a) compares the level and anomaly with `studies.dp_assimilation.run_kalman`
  on the same stream in float64 (1e-9). Part (b) compares the forecast frames of arm GK with
  experiment 111's stored `gamk_tuning_<site>.parquet` (quantiles to 1e-4; the stored level and
  anomaly are float32, so they agree to 1e-5).
- **P2, decomposition:** the regressors plus the remainder equal `predict_gam` for every member of
  every tuning run (1e-6 points), and each regressor is zero where its term is zero.
- **P3, causality:** (a) changing every reading after the cutoff, and every ensemble run issued
  after the origin, leaves a forecast unchanged; (b) the regressor scale and the noise fit do not
  read the stream after 2025-07-01.
- **P4, noise fit:** a stream simulated from known random-walk variances on the real regressors is
  fitted within a factor of 3 for every group whose weight has a posterior standard deviation under
  0.5 at the end, and a group simulated at 0 fits below 1e-7.

    uv run python ml-experiments/00118-*/e00118_controls.py p0 p1 p2 p3 p4
"""

import argparse
from dataclasses import replace
from typing import Final

import e00118_arms as arms
import e00118_common as common
import e00118_forecast as fc
import e00118_gate as gate
import e00118_noise as noise_io
import e00118_streams as streams
import numpy as np
import polars as pl
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    FilterHyperparameters,
    correction_at,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_effects_filter import (
    HEATING,
    HOLIDAY_INDEX,
    N_WEIGHTS,
    PROFILE,
    PV_INDEX,
    FilterNoise,
    fit_effects_noise,
    initial_sd_for,
    run_effects_filter,
    state_at,
)
from studies.dp_ens import choose_fit
from studies.dp_ens_weather import RunWindow
from studies.dp_features import naive_utc
from studies.dp_gam import predict_gam

P1_FILTER_TOLERANCE: Final[float] = 1e-9
P1_QUANTILE_TOLERANCE: Final[float] = 1e-4
P1_STORED_FLOAT32_TOLERANCE: Final[float] = 1e-5
P2_TOLERANCE: Final[float] = 1e-6
P4_FACTOR: Final[float] = 3.0
P4_ZERO_Q: Final[float] = 1e-7
P4_MAX_SD: Final[float] = 0.5
P4_TRUE_Q: Final[dict[str, float]] = {"pv": 1e-4, "heating": 1e-6, "profile": 0.0, "holiday": 0.0}
"""The simulated random-walk variances in weight units per day, for P4."""
CHECK_RUNS: Final[int] = 40


def p0() -> dict:
    """Print the starting state and the vintage used by each cut."""
    out: dict = {"vintages": {}, "initial_sd": {}}
    cache = common.load_cache()
    for site in common.SCORED_SITES:
        fits = common.load_fits(site=site)
        out["vintages"][site] = [
            {
                "cut": f"{fit.cut:%Y-%m-%d}",
                "source": fit.pinned.source,
                "vintage_origin": str(fit.pinned.origin),
                "days_vintage_after_cut": (
                    np.datetime64(fit.pinned.origin) - np.datetime64(fit.cut.date())
                )
                .astype(int)
                .item(),
            }
            for fit in fits
            if fit.pinned is not None
        ]
        late = [v for v in out["vintages"][site] if v["days_vintage_after_cut"] > 0]
        print(f"{site}: {len(fits)} cuts, {len(late)} with a vintage origin after the cut")
        data = streams.build_site_streams(
            cache=cache, setup=common.load_setup(cache=cache, site=site), fits=fits
        )
        out["initial_sd"][site] = initial_sd_for(
            scale=data.scale, moving=np.ones(N_WEIGHTS, dtype=bool)
        ).tolist()
    return out


def p1() -> dict:
    """Check nesting against `run_kalman` and against experiment 111's stored frames."""
    cache = common.load_cache()
    result: dict = {}
    for site in common.SCORED_SITES:
        setup = common.load_setup(cache=cache, site=site)
        fits = common.load_fits(site=site)
        data = streams.build_site_streams(cache=cache, setup=setup, fits=fits)
        hyper = common.tuned_111()["filter_hyper"]["gamk"][site]
        noise = arms.gk_noise(hyper=hyper, scale=data.scale)
        trace = run_effects_filter(
            residual=data.residual, regressors=data.standardised, band=data.band, noise=noise
        )
        reference = run_kalman(
            residual=data.residual,
            band=data.band,
            hyper=FilterHyperparameters(
                log_q=hyper[0], logit_phi=hyper[1], log_sigma_z=hyper[2], log_sigma_eps=hyper[3]
            ),
        )
        runs = setup.indices(periods={"tuning"})[:CHECK_RUNS]
        difference = 0.0
        for index in runs:
            members = fc.run_members(cache=cache, setup=setup, fits=fits, index=index)
            if members is None:
                continue
            target = np.arange(members.window.start, members.window.stop)
            band = data.band[target]
            cutoff = usable_cutoff_positions(
                end_times=cache.end_times,
                initialisation=np.array([naive_utc(moment=setup.origins[index])]),
                delay_hours=DEFAULT_DELAY_HOURS,
            )
            expected = correction_at(
                trace=reference, cutoff=np.full(len(target), cutoff[0]), target=target, band=band
            )
            mine = fc.corrected_members(
                cache=cache,
                setup=setup,
                data=data,
                filtered=fc.ArmFilter(name="GK", trace=trace, noise=noise),
                members=members,
            )
            difference = max(
                difference,
                float(np.abs(mine.level - expected.level).max()),
                float(np.abs(mine.anomaly - expected.anomaly).max()),
            )
        stored = pl.read_parquet(common.E111_WORK / "dp_preds" / f"gamk_tuning_{site}.parquet")
        spread = common.load_gamk_spread(site=site)
        quantile_difference = 0.0
        stored_difference = 0.0
        joined_rows = 0
        for index in runs:
            members = fc.run_members(cache=cache, setup=setup, fits=fits, index=index)
            if members is None:
                continue
            filtered = fc.ArmFilter(name="GK", trace=trace, noise=noise)
            corrected = fc.corrected_members(
                cache=cache, setup=setup, data=data, filtered=filtered, members=members
            )
            quantiles = fc.quantiles_of(
                cache=cache, members=members, corrected=corrected.means, spread=spread
            )
            frame = fc.frame_of(
                cache=cache,
                setup=setup,
                members=members,
                corrected=corrected,
                quantiles=quantiles,
                arm="GK",
            )
            if frame is None:
                continue
            joined = frame.join(
                stored.filter(pl.col("run") == index), on=["run", "valid_time"], suffix="_111"
            )
            joined_rows += joined.height
            for column in ("q01", "q25", "q50", "q75", "q99"):
                quantile_difference = max(
                    quantile_difference,
                    float((joined[column] - joined[f"{column}_111"]).abs().max()),
                )
            for column in ("level", "anomaly"):
                stored_difference = max(
                    stored_difference,
                    float((joined[column] - joined[f"{column}_111"]).abs().max()),
                )
        result[site] = {
            "level_anomaly_vs_run_kalman": difference,
            "quantiles_vs_stored": quantile_difference,
            "level_anomaly_vs_stored_float32": stored_difference,
            "rows_compared": joined_rows,
            "passed": joined_rows > 0
            and difference < P1_FILTER_TOLERANCE
            and quantile_difference < P1_QUANTILE_TOLERANCE
            and stored_difference < P1_STORED_FLOAT32_TOLERANCE,
        }
        print(site, result[site])
    return result


def p2() -> dict:
    """Check the decomposition against `predict_gam` for every member of the tuning runs."""
    cache = common.load_cache()
    result: dict = {}
    for site in common.SCORED_SITES:
        setup = common.load_setup(cache=cache, site=site)
        fits = common.load_fits(site=site)
        worst = 0.0
        pv_night = 0.0
        for index in setup.indices(periods={"tuning"}):
            members = fc.run_members(cache=cache, setup=setup, fits=fits, index=index)
            if members is None:
                continue
            run = setup.runs[index]
            fit = choose_fit(fits=fits, origin=setup.origins[index])
            window = RunWindow.build(cache=cache, context=setup.context, run=run)
            driven = window.driven_caches(context=setup.context, run=run, members=members.members)
            rows = np.arange(window.window.stop - window.window.start)
            for position, member_cache in enumerate(driven):
                expected = predict_gam(cache=member_cache, fit=fit, index=rows)
                worst = max(worst, float(np.abs(expected - members.base_mean[position]).max()))
                shape = member_cache.sites[site].pv_unit_shape
                pv_night = max(
                    pv_night,
                    float(np.abs(members.contributions[position][shape == 0, PV_INDEX]).max())
                    if (shape == 0).any()
                    else 0.0,
                )
        result[site] = {
            "max_abs_difference": worst,
            "max_pv_at_night": pv_night,
            "passed": worst < P2_TOLERANCE and pv_night == 0.0,
        }
        print(site, result[site])
    return result


def p3() -> dict:
    """Check that a forecast ignores later readings and runs, and that fits stop at the cut."""
    cache = common.load_cache()
    site = "D10"
    setup = common.load_setup(cache=cache, site=site)
    fits = common.load_fits(site=site)
    data = streams.build_site_streams(cache=cache, setup=setup, fits=fits)
    stored = noise_io.read_noise(arm="KEZ")["sites"][site]
    noise = FilterNoise(
        q=np.array(stored["q"]),
        phi=stored["phi"],
        sigma_z=stored["sigma_z"],
        sigma_eps=stored["sigma_eps"],
        initial_sd=np.array(stored["initial_sd"]),
        use_anomaly=True,
    )
    index = setup.indices(periods={"tuning"})[60]
    origin = setup.origins[index]
    cutoff = int(
        usable_cutoff_positions(
            end_times=cache.end_times,
            initialisation=np.array([naive_utc(moment=origin)]),
            delay_hours=DEFAULT_DELAY_HOURS,
        )[0]
    )

    def forecast(*, residual: np.ndarray, regressors: np.ndarray) -> np.ndarray:
        trace = run_effects_filter(
            residual=residual, regressors=regressors, band=data.band, noise=noise
        )
        state = state_at(trace=trace, cutoff=cutoff)
        return np.concatenate([state.delta, state.anomaly])

    changed_residual = data.residual.copy()
    changed_residual[cutoff + 1 :] += 25.0
    changed_regressors = data.standardised.copy()
    changed_regressors[cutoff + 1 :] *= 3.0
    first = forecast(residual=data.residual, regressors=data.standardised)
    second = forecast(residual=changed_residual, regressors=changed_regressors)
    members = fc.run_members(cache=cache, setup=setup, fits=fits, index=index)
    later = replace(
        setup,
        runs=[
            run
            if position <= index
            else replace(
                run, temperature_c=run.temperature_c + 9.0, shortwave_w_m2=run.shortwave_w_m2 * 0.1
            )
            for position, run in enumerate(setup.runs)
        ],
    )
    members_later = fc.run_members(cache=cache, setup=later, fits=fits, index=index)
    fit_cut_end = data.fit_residual(cache=cache)
    beyond = fit_cut_end[cache.end_times > naive_utc(moment=streams.FIT_END)]
    result = {
        "state_unchanged_by_later_readings": bool(np.array_equal(first, second)),
        "members_unchanged_by_later_runs": bool(
            members is not None
            and members_later is not None
            and np.array_equal(members.base_mean, members_later.base_mean)
            and np.array_equal(members.contributions, members_later.contributions)
        ),
        "fit_stream_empty_after_cut": bool(not np.isfinite(beyond).any()),
        "scale_uses_tuning_mask_only": bool(
            np.array_equal(
                data.scale,
                streams.regressor_scale(contributions=data.contributions, mask=data.tuning_mask),
            )
        ),
    }
    result["passed"] = all(result.values())
    print(result)
    return result


def p4() -> dict:
    """Fit streams simulated from known random-walk variances and compare the fitted ones."""
    result: dict = {}
    for site in gate.SITES:
        inputs = gate.gate_inputs(site)
        generator = np.random.default_rng(gate.draw_seed(parts=("p4", site)))
        weight_q = np.zeros(N_WEIGHTS)
        weight_q[PV_INDEX] = P4_TRUE_Q["pv"]
        weight_q[HEATING] = P4_TRUE_Q["heating"]
        weight_q[PROFILE] = P4_TRUE_Q["profile"]
        weight_q[HOLIDAY_INDEX] = P4_TRUE_Q["holiday"]
        steps = generator.normal(size=(len(inputs.rows), N_WEIGHTS)) * np.sqrt(weight_q / 48.0)
        deviation = np.cumsum(steps, axis=0)
        stream = (deviation * inputs.seen).sum(axis=1) + gate.ar1_noise(
            inputs=inputs, generator=generator
        )
        stream = np.where(np.isfinite(inputs.seen).all(axis=1), stream, np.nan)
        fitted = fit_effects_noise(
            residual=stream,
            regressors=inputs.seen / inputs.scale,
            band=inputs.band,
            scale=inputs.scale,
            variant=arms.FITTED_ARMS["KEZ"].variant,
            phi=inputs.noise.phi,
            sigma_z=inputs.noise.sigma_z,
            sigma_eps_start=inputs.noise.sigma_eps,
            likelihood_mask=inputs.likelihood_mask & np.isfinite(stream),
            prior_mean=np.full(N_WEIGHTS, -3.0),
            prior_spread=np.full(N_WEIGHTS, 20.0),
        )
        trace = run_effects_filter(
            residual=stream,
            regressors=inputs.seen / inputs.scale,
            band=inputs.band,
            noise=fitted.noise,
        )
        fitted_weight_q = fitted.noise.q / inputs.scale**2
        end_sd = trace.sd[-1] / inputs.scale
        rows = {}
        for name, index in (
            ("pv", PV_INDEX),
            ("heating", HEATING.start),
            ("holiday", HOLIDAY_INDEX),
        ):
            true = weight_q[index]
            value = float(fitted_weight_q[index])
            identified = bool(end_sd[index] < P4_MAX_SD and not fitted.set_by_prior[index])
            ok = (
                value < P4_ZERO_Q
                if true == 0.0
                else (true / P4_FACTOR <= value <= true * P4_FACTOR) or not identified
            )
            rows[name] = {"true": true, "fitted": value, "identified": identified, "ok": bool(ok)}
        result[site] = rows
        print(site, rows)
    result["passed"] = all(r["ok"] for site in gate.SITES for r in result[site].values())
    return result


STEPS: Final = {"p0": p0, "p1": p1, "p2": p2, "p3": p3, "p4": p4}


def main() -> None:
    """Run the chosen controls and write their results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controls", nargs="+", choices=list(STEPS))
    args = parser.parse_args()
    for name in args.controls:
        content = STEPS[name]()
        common.write_json_atomic(path=common.WORK_DIR / f"control_{name}.json", content=content)
        print(f"{name}: written")


if __name__ == "__main__":
    main()
