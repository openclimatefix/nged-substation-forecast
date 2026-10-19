"""Tests of the pooled PyTorch configuration of the differentiable-physics series."""

from dataclasses import replace
from typing import Final

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from studies.dp_features import FeatureCache  # noqa: E402
from studies.dp_pooled import (  # noqa: E402
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
