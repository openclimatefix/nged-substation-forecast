"""Unit tests of the bundle loader, the ENS summary, and the neural network's read-out."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl
import pytest
import torch
from _step5_bundle_test_data import write_bundle
from substation_demand_forecaster.bundle import Bundle, load_bundle, require_servable_bundle
from substation_demand_forecaster.ens_summary import SOURCE_COLUMNS, summarise_runs
from substation_demand_forecaster.features import STEP5_COLUMNS, ExpectedRows
from substation_demand_forecaster.neural_network import forecast, forward, mixture_quantiles

_LEVELS = (0.01, 0.1, 0.5, 0.9, 0.99)


def test_one_component_mixture_gives_the_normal_quantiles(bundle: Bundle) -> None:
    raw = torch.tensor([[60.0, 0.0, 0.0, 0.7, 0.0, 0.0, math.log(1.3), 0.0, 0.0]])
    out = mixture_quantiles(raw=raw, levels=_LEVELS, config=bundle.network)
    expected = [NormalDist(mu=0.7, sigma=1.3).inv_cdf(level) for level in _LEVELS]
    np.testing.assert_allclose(out[0].numpy(), expected, rtol=0, atol=1e-5)


def test_three_component_mixture_clamps_log_std_and_inverts_the_cdf(bundle: Bundle) -> None:
    logits, means, log_stds = [0.2, -0.4, 1.1], [-1.0, 0.5, 2.0], [-4.0, 0.1, 3.5]
    raw = torch.tensor([logits + means + log_stds])
    # The widest component's standard deviation of e^3 puts the outer levels beyond the
    # bisection range of 40, so the test reads the central levels.
    levels = (0.2, 0.35, 0.5, 0.65, 0.8)
    out = mixture_quantiles(raw=raw, levels=levels, config=bundle.network)[0].numpy()
    weights = np.exp(logits) / np.exp(logits).sum()
    stds = np.exp(np.clip(log_stds, -3.0, 3.0))
    for q, level in zip(out, levels, strict=True):
        cdf = sum(
            w * NormalDist(mu=m, sigma=s).cdf(q)
            for w, m, s in zip(weights, means, stds, strict=True)
        )
        assert math.isclose(cdf, level, abs_tol=1e-6)


def test_forecast_scales_each_seed_by_its_primary_and_averages_the_seeds(bundle: Bundle) -> None:
    rows = 50
    rng = np.random.default_rng(2)
    matrix = rng.standard_normal((rows, len(STEP5_COLUMNS))).astype(np.float32)
    primary = np.arange(rows, dtype=np.int64) % 2
    matrix[:, 43:45] = -1  # both lags missing, so both read the zero expected-weather row
    matrix[:, 45] = rng.uniform(-1.0, 5.0, rows)  # knot position, beyond both ends of the knots
    no_rows = ExpectedRows(
        primary_index=np.zeros(0, dtype=np.int64),
        grid_index=np.zeros(0, dtype=np.int64),
        run_init_time=np.zeros(0, dtype="datetime64[us]"),
        raw=np.zeros((0, 28), dtype=np.float32),
        standardised=np.zeros((0, 28), dtype=np.float16),
    )
    out = forecast(
        bundle=bundle, matrix=matrix, primary_index=primary, expected_rows=no_rows, levels=_LEVELS
    )
    zeros = torch.zeros((rows, 28), dtype=torch.float16)
    mean = bundle.standardisers.target_mean[primary][:, None]
    std = bundle.standardisers.target_std[primary][:, None]
    for seed, weights in enumerate(bundle.weights):
        raw = forward(
            w=weights, x=torch.from_numpy(matrix), primary=torch.from_numpy(primary),
            expected=(zeros, zeros), config=bundle.network,
        )  # fmt: skip
        standardised = mixture_quantiles(raw=raw, levels=_LEVELS, config=bundle.network).numpy()
        expected = np.sort(standardised * std + mean, axis=1)
        np.testing.assert_allclose(out.per_seed[seed], expected, rtol=0, atol=1e-6)
    assert not np.allclose(out.per_seed[0], out.per_seed[1])
    np.testing.assert_allclose(out.mean, np.sort(out.per_seed.mean(axis=0), axis=1), atol=1e-7)
    assert (np.diff(out.mean, axis=-1) >= 0).all()


def test_forecast_raises_when_a_lag_has_no_expected_row(bundle: Bundle) -> None:
    matrix = np.zeros((1, len(STEP5_COLUMNS)), dtype=np.float32)
    matrix[0, 43] = 7
    empty = ExpectedRows(
        primary_index=np.zeros(0, dtype=np.int64),
        grid_index=np.zeros(0, dtype=np.int64),
        run_init_time=np.zeros(0, dtype="datetime64[us]"),
        raw=np.zeros((0, 28), dtype=np.float32),
        standardised=np.zeros((0, 28), dtype=np.float16),
    )
    with pytest.raises(ValueError, match="no expected-weather row"):
        forecast(
            bundle=bundle,
            matrix=matrix,
            primary_index=np.zeros(1, dtype=np.int64),
            expected_rows=empty,
            levels=_LEVELS,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"bundle_format_version": "1"}, "bundle_format_version"),
        ({"components": {"step5": {"kind": "demand_network_step6"}}}, "not implemented"),
    ],
)
def test_load_bundle_rejects_a_bundle_it_cannot_serve(
    tmp_path: Path, overrides: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        load_bundle(write_bundle(tmp_path / "bundle", manifest_overrides=overrides))


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        (("shared", "native_origin_hour_utc"), 6),
        (("components", "step5", "b1_constants", "cams_columns_source"), "arm replay"),
    ],
)
def test_a_changed_config_constant_is_refused(
    tmp_path: Path, keys: tuple[str, ...], value: object
) -> None:
    root = write_bundle(tmp_path / "bundle")
    config = json.loads((root / "config.json").read_text())
    *parents, leaf = keys
    node = config
    for key in parents:
        node = node[key]
    node[leaf] = value
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=leaf):
        require_servable_bundle(load_bundle(root))


def test_ens_summary_takes_wind_speed_per_member_before_averaging(bundle: Bundle) -> None:
    init = datetime(2026, 4, 17, tzinfo=UTC)
    valid = [init + timedelta(hours=h) for h in range(25)]
    frames = []
    for member, sign, temperature in [(0, 1.0, 1.0), (1, -1.0, 3.0)]:
        frames.append(
            pl.DataFrame(
                {
                    "nwp_model_id": "ECMWF_ENS_0_25_degree",
                    "init_time": init,
                    "valid_time": valid,
                    "ensemble_member": member,
                    "h3_index": bundle.primaries["h3_index"][0],
                    "wind_u_10m": 3.0 * sign,
                    "wind_v_10m": 4.0 * sign,
                }
                | dict.fromkeys(SOURCE_COLUMNS.values(), 0.0)
                | {"temperature_2m": temperature}
            )
        )
    nwp = pl.concat(frames).with_columns(
        pl.col("init_time", "valid_time").cast(pl.Datetime("us", "UTC")),
        pl.col("ensemble_member").cast(pl.Int8),
        pl.col(*SOURCE_COLUMNS.values(), "wind_u_10m", "wind_v_10m").cast(pl.Float32),
    )
    summary = summarise_runs(nwp.lazy(), bundle.primaries).collect()
    # The 15 hours from O + 30 minutes to init + 24 hours.
    assert summary.height == 30
    assert summary["time_series_id"].unique().to_list() == [101]
    assert summary["valid_time"].min() == init + timedelta(hours=9, minutes=30)
    np.testing.assert_allclose(summary["mean_wind_speed_10m"].to_numpy(), 5.0, rtol=1e-6)
    np.testing.assert_allclose(summary["std_wind_speed_10m"].to_numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(summary["mean_temperature_2m"].to_numpy(), 2.0, rtol=1e-6)
    # The population standard deviation of 1 and 3.
    np.testing.assert_allclose(summary["std_temperature_2m"].to_numpy(), 1.0, rtol=1e-6)
