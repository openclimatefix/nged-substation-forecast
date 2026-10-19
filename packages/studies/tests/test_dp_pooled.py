"""Tests of the pooled PyTorch configuration of the differentiable-physics series."""

from dataclasses import replace
from typing import Final

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from studies.dp_features import FeatureCache  # noqa: E402
from studies.dp_pooled import (  # noqa: E402
    INTERACTION_INPUTS,
    DpConfig,
    PooledDemandModel,
    ResidualStates,
    build_pooled_batch,
    fit_pooled,
)

CUT: Final[np.datetime64] = np.datetime64("2024-07-01T00:00:00", "us")


def _two_site_cache(cache: FeatureCache) -> FeatureCache:
    """Return the synthetic cache with an MVA twin of its one substation."""
    mw = cache.sites["S1"]
    mva_table = replace(mw.table, site="S2", units="MVA", time_series_id=2)
    mva = replace(mw, table=mva_table, dark=np.nan_to_num(mw.ghi_w_m2) < 1.0)
    return replace(cache, sites={"S1": mw, "S2": mva}, node_ids={"S1": ("S1",), "S2": ("S2",)})


def _model(*, batch, config: DpConfig, seed: int = 0) -> PooledDemandModel:  # noqa: ANN001
    torch.manual_seed(seed)
    return PooledDemandModel(
        config=config,
        n_sites=len(batch.site_labels),
        n_season_years=batch.n_season_years,
        initial_level=torch.full((len(batch.site_labels),), 50.0),
    )


@pytest.mark.parametrize(
    "switch", [{"variational": True}, {"clip": True}, {"generator_node": True}, {"drift": "d1"}]
)
def test_a_switch_for_a_module_that_is_not_built_raises(switch: dict):
    with pytest.raises(NotImplementedError, match="not built yet"):
        DpConfig(**switch).check_supported()


def test_the_default_configuration_is_supported():
    DpConfig().check_supported()
    DpConfig(interaction_hidden=(64, 64)).check_supported()


def test_the_batch_holds_default_mw_rows_and_dark_mva_rows_before_the_cut(
    dp_synthetic_cache: FeatureCache,
):
    cache = _two_site_cache(dp_synthetic_cache)
    batch = build_pooled_batch(cache=cache, sites=("S1", "S2"), cut_end=CUT)
    mw_rows = int((batch.site == 0).sum())
    mva_rows = int((batch.site == 1).sum())
    assert not batch.is_apparent[batch.site == 0].any()
    assert batch.is_apparent[batch.site == 1].all()
    assert not batch.solar_gate[batch.site == 1].any()
    assert batch.solar_gate[batch.site == 0].all()
    assert 0 < mva_rows < mw_rows
    dark_in_window = int(
        (
            cache.sites["S2"].dark
            & cache.sites["S2"].default_rows
            & (cache.end_times <= CUT)
            & np.isfinite(cache.sites["S2"].pv_unit_shape)
            & np.isfinite(cache.sites["S2"].smoothed_temperature_c).all(axis=1)
        ).sum()
    )
    assert mva_rows == dark_in_window
    # Dark rows have almost no sun; the gate switches the solar term off on them anyway.
    assert float(batch.pv_shape[batch.site == 1].max()) < 0.01


def test_the_interaction_network_starts_at_zero_so_it_changes_nothing_at_first(
    dp_synthetic_cache: FeatureCache,
):
    batch = build_pooled_batch(
        cache=_two_site_cache(dp_synthetic_cache), sites=("S1", "S2"), cut_end=CUT
    )
    plain = _model(batch=batch, config=DpConfig())
    with_network = _model(batch=batch, config=DpConfig(interaction_hidden=(8, 8)))
    assert torch.allclose(plain(batch), with_network(batch))
    assert with_network.interaction is not None


def test_a_mva_row_with_no_reactive_power_and_no_solar_reads_the_demand(
    dp_synthetic_cache: FeatureCache,
):
    batch = build_pooled_batch(
        cache=_two_site_cache(dp_synthetic_cache), sites=("S1", "S2"), cut_end=CUT
    )
    model = _model(batch=batch, config=DpConfig())
    with torch.no_grad():
        model.tan_phi.zero_()
        model.q0.zero_()
        reading = model(batch)
        demand = model.demand(batch=batch)
    mva = batch.is_apparent
    assert torch.allclose(reading[mva], demand[mva].abs(), atol=1e-2)


def test_the_capacity_of_every_substation_never_decreases(dp_synthetic_cache: FeatureCache):
    batch = build_pooled_batch(
        cache=_two_site_cache(dp_synthetic_cache), sites=("S1", "S2"), cut_end=CUT
    )
    model = _model(batch=batch, config=DpConfig())
    with torch.no_grad():
        model.capacity_raw.copy_(torch.randn_like(model.capacity_raw))
        capacities = model.capacities()
    assert (torch.diff(capacities, dim=1) >= 0.0).all()


def test_fitting_lowers_the_loss_and_reports_time_and_memory(dp_synthetic_cache: FeatureCache):
    batch = build_pooled_batch(
        cache=_two_site_cache(dp_synthetic_cache), sites=("S1", "S2"), cut_end=CUT
    )
    model = _model(batch=batch, config=DpConfig(interaction_hidden=(16, 16)))
    result = fit_pooled(model=model, batch=batch, steps=25, learning_rate=0.05)
    assert len(result.losses) == 25
    assert result.losses[-1] < result.losses[0]
    assert result.seconds > 0.0
    assert result.peak_memory_bytes == 0


def test_residual_states_start_empty_and_addressable_by_site():
    states = ResidualStates()
    states.level["D10"] = np.zeros(4)
    assert states.anomaly == {}
    assert states.level["D10"].shape == (4,)


def test_cpu_and_gpu_pooled_fits_agree(dp_synthetic_cache: FeatureCache):
    if not torch.cuda.is_available():
        pytest.skip("no GPU in this environment")
    cache = _two_site_cache(dp_synthetic_cache)
    cpu_batch = build_pooled_batch(cache=cache, sites=("S1", "S2"), cut_end=CUT)
    gpu_batch = build_pooled_batch(cache=cache, sites=("S1", "S2"), cut_end=CUT, device="cuda")
    cpu_model = _model(batch=cpu_batch, config=DpConfig(), seed=1)
    gpu_model = _model(batch=cpu_batch, config=DpConfig(), seed=1).to("cuda")
    with torch.no_grad():
        assert torch.allclose(cpu_model(cpu_batch), gpu_model(gpu_batch).cpu(), atol=1e-3)


def test_the_calendar_terms_of_demand_are_centred_per_site_year(dp_synthetic_cache: FeatureCache):
    batch = build_pooled_batch(
        cache=_two_site_cache(dp_synthetic_cache), sites=("S1", "S2"), cut_end=CUT
    )
    model = _model(batch=batch, config=DpConfig())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.level.fill_(50.0)
        model.slope_raw.fill_(-30.0)
        model.holiday.fill_(5.0)  # a constant calendar term: centring must remove all of it
    model.train()
    demand = model.demand(batch=batch)
    assert torch.allclose(demand, torch.full_like(demand, 50.0), atol=1e-3)


def test_a_forecast_year_is_shifted_by_the_stored_training_mean_and_never_by_its_own_mean(
    dp_synthetic_cache: FeatureCache,
):
    cache = _two_site_cache(dp_synthetic_cache)
    train = build_pooled_batch(cache=cache, sites=("S1", "S2"), cut_end=CUT)
    forecast = build_pooled_batch(
        cache=cache,
        sites=("S1", "S2"),
        cut_end=np.datetime64("2025-07-01T00:00:00", "us"),
        after=CUT,
    )
    model = _model(batch=train, config=DpConfig(interaction_hidden=(8,)))
    fit_pooled(model=model, batch=train, steps=3, learning_rate=0.01)
    # The two dark half-hours that end at the 1 July cut start in the next season-year, which
    # must not count as trained.
    last_trained = int(model.trained[0].nonzero().max())
    assert int(forecast.season_slot.min()) % forecast.n_season_years > last_trained
    stored = model.centre_means.reshape(model.n_sites, model.n_season_years)
    # The forecast year takes the mean of the site's last trained year, not zero.
    assert (stored[:, last_trained + 1] == stored[:, last_trained]).all()
    assert (stored[:, last_trained] != 0.0).all()
    model.eval()
    values = torch.randn(forecast.n_rows) + 3.0
    centred = model._centred(values=values, slot=forecast.season_slot)
    assert torch.allclose(centred, values - stored.reshape(-1)[forecast.season_slot])
    # A live mean would leave the centred values with mean zero; the stored training mean does not.
    assert abs(float(centred.mean())) > 0.5
    # Shifting every forecast value shifts the centred values by the same amount: no live mean.
    shifted = model._centred(values=values + 10.0, slot=forecast.season_slot)
    assert torch.allclose(shifted - centred, torch.full_like(values, 10.0), atol=1e-4)


def test_the_interaction_network_does_not_see_the_suns_elevation(dp_synthetic_cache: FeatureCache):
    batch = build_pooled_batch(
        cache=_two_site_cache(dp_synthetic_cache), sites=("S1", "S2"), cut_end=CUT
    )
    mixed = torch.zeros(batch.n_rows)
    inputs = PooledDemandModel._interaction_inputs(batch=batch, mixed=mixed)
    assert inputs.shape[1] == INTERACTION_INPUTS
    changed = replace(batch, elevation=batch.elevation + 30.0)
    assert torch.equal(PooledDemandModel._interaction_inputs(batch=changed, mixed=mixed), inputs)


def test_a_season_year_without_training_rows_adds_no_capacity(dp_synthetic_cache: FeatureCache):
    batch = build_pooled_batch(
        cache=_two_site_cache(dp_synthetic_cache), sites=("S1", "S2"), cut_end=CUT
    )
    model = _model(batch=batch, config=DpConfig())
    model.mark_trained_years(batch=batch)
    with torch.no_grad():
        model.capacity_raw.fill_(2.0)
    capacities = model.capacities()
    last_trained = int(batch.season_slot.max()) % batch.n_season_years
    assert (
        capacities[:, last_trained + 1 :] == capacities[:, last_trained : last_trained + 1]
    ).all()
    assert (capacities[:, last_trained] > capacities[:, 0]).all()


def test_a_float64_batch_fits_without_a_dtype_mismatch(dp_synthetic_cache: FeatureCache):
    batch = build_pooled_batch(
        cache=_two_site_cache(dp_synthetic_cache),
        sites=("S1", "S2"),
        cut_end=CUT,
        dtype=torch.float64,
    )
    model = _model(batch=batch, config=DpConfig())
    result = fit_pooled(model=model, batch=batch, steps=3, learning_rate=0.05)
    assert len(result.losses) == 3
