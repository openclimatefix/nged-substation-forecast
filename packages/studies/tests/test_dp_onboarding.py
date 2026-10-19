"""Tests of the batched pooled fits, the site priors, onboarding, and the pooled spread."""

from dataclasses import replace
from typing import Final

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from studies.dp_features import FeatureCache  # noqa: E402
from studies.dp_onboarding import (  # noqa: E402
    FitCopy,
    PooledResiduals,
    after_mask,
    cross_fitted_spread,
    fit_many,
    onboard_site,
    pooled_residuals,
    predict_pooled,
    select_rows,
    site_mask,
    with_targets,
)
from studies.dp_pooled import (  # noqa: E402
    SITE_FAMILIES,
    DpConfig,
    PooledBatch,
    PopulationScales,
    build_pooled_batch,
)

CUT: Final[np.datetime64] = np.datetime64("2024-07-01T00:00:00", "us")
STEPS: Final[int] = 20
SITES: Final[tuple[str, ...]] = ("S1", "S2", "S3")


def _cache(cache: FeatureCache) -> FeatureCache:
    """Return the synthetic cache with an MVA twin (S2) and a shifted MW twin (S3) of S1."""
    mw = cache.sites["S1"]
    mva = replace(
        mw,
        table=replace(mw.table, site="S2", units="MVA", time_series_id=2),
        dark=np.nan_to_num(mw.ghi_w_m2) < 1.0,
    )
    twin = replace(mw, table=replace(mw.table, site="S3", time_series_id=3), y=mw.y * 1.1 + 2.0)
    return replace(
        cache,
        sites={"S1": mw, "S2": mva, "S3": twin},
        node_ids={name: (name,) for name in SITES},
    )


@pytest.fixture(scope="module")
def batch(dp_synthetic_cache: FeatureCache) -> PooledBatch:
    return build_pooled_batch(cache=_cache(dp_synthetic_cache), sites=SITES, cut_end=CUT)


def _level(batch: PooledBatch) -> torch.Tensor:
    return torch.full((len(batch.site_labels),), 50.0)


def _fit(batch: PooledBatch, config: DpConfig, copies: list[FitCopy]):  # noqa: ANN202
    return fit_many(
        batch=batch, copies=copies, config=config, initial_level=_level(batch), steps=STEPS
    )


@pytest.mark.parametrize(
    "config",
    [DpConfig(variational=True), DpConfig(population_scales=PopulationScales())],
    ids=["variational", "map"],
)
def test_a_site_prior_fit_lowers_the_loss_and_forecasts_deterministically(
    batch: PooledBatch, config: DpConfig
):
    (model,), (result,) = _fit(batch, config, [FitCopy(name="all")])
    assert result.losses[-1] < result.losses[0]
    first = predict_pooled(model=model, batch=batch)
    assert np.array_equal(first, predict_pooled(model=model, batch=batch))
    assert np.isfinite(first).all()


def test_the_variational_fit_learns_the_population_scales_and_the_map_fit_fixes_them(
    batch: PooledBatch,
):
    (variational,), _ = _fit(batch, DpConfig(variational=True), [FitCopy(name="v")])
    (fixed,), _ = _fit(batch, DpConfig(population_scales=PopulationScales()), [FitCopy(name="m")])
    start = PopulationScales().as_dict()
    moved = [
        abs(float(variational.population_log_scale[n].detach()) - np.log(start[n])) > 1e-4
        for n in SITE_FAMILIES
    ]
    assert any(moved)
    for name in SITE_FAMILIES:
        assert float(fixed.population_log_scale[name].detach()) == pytest.approx(
            np.log(start[name])
        )


def test_a_held_out_sites_targets_cannot_change_a_fit_that_drops_its_rows(batch: PooledBatch):
    keep = ~site_mask(batch=batch, label="S3")
    config = DpConfig(variational=True)
    changed = with_targets(
        batch=batch, y=torch.where(site_mask(batch=batch, label="S3"), batch.y + 40.0, batch.y)
    )
    (base,), _ = _fit(batch, config, [FitCopy(name="a", keep=keep)])
    (other,), _ = _fit(changed, config, [FitCopy(name="b", keep=keep)])
    assert torch.allclose(
        predict_pooled_tensor(base, batch), predict_pooled_tensor(other, batch), atol=1e-6
    )


def predict_pooled_tensor(model, batch):  # noqa: ANN001
    return torch.as_tensor(predict_pooled(model=model, batch=batch))


def test_batched_copies_match_separate_fits(batch: PooledBatch):
    config = DpConfig(variational=True)
    copies = [
        FitCopy(name="drop S1", keep=~site_mask(batch=batch, label="S1")),
        FitCopy(name="drop S3", keep=~site_mask(batch=batch, label="S3")),
    ]
    together, _ = _fit(batch, config, copies)
    for copy, joint in zip(copies, together, strict=True):
        (alone,), _ = _fit(batch, config, [copy])
        assert np.allclose(
            predict_pooled(model=joint, batch=batch), predict_pooled(model=alone, batch=batch)
        )


def test_mva_rows_train_demand_only_so_their_solar_gradient_is_zero(batch: PooledBatch):
    from studies.dp_pooled import PooledDemandModel

    torch.manual_seed(0)
    model = PooledDemandModel(
        config=DpConfig(),
        n_sites=3,
        n_season_years=batch.n_season_years,
        initial_level=_level(batch),
    )
    mva_only = select_rows(batch=batch, keep=site_mask(batch=batch, label="S2"))
    model.mark_trained_years(batch=mva_only)
    model.loss(batch=mva_only).backward()
    grad = model.capacity_raw.grad
    assert grad is not None
    assert float(grad.abs().sum()) == 0.0


def test_onboarding_changes_only_the_new_sites_parameters(batch: PooledBatch):
    config = DpConfig(variational=True, interaction_hidden=(8,))
    keep = ~site_mask(batch=batch, label="S3")
    (model,), _ = _fit(batch, config, [FitCopy(name="loso", keep=keep)])
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    year = int(batch.first_season_year + batch.n_season_years - 1)
    own = select_rows(batch=batch, keep=site_mask(batch=batch, label="S3"))
    info = onboard_site(model=model, batch=own, label="S3", forecast_season_year=year, steps=STEPS)
    assert info["rows"] == own.n_rows
    site = batch.site_labels.index("S3")
    for name, parameter in model.named_parameters():
        change = (parameter.detach() - before[name]).abs()
        if name.split(".")[0] in (*SITE_FAMILIES, "tan_phi", "q0") or name.startswith(
            "posterior_log_sd"
        ):
            others = torch.arange(change.shape[0]) != site
            assert float(change[others].sum()) == 0.0, name
        else:
            assert float(change.sum()) == 0.0, name
    assert float(model.trained[site].sum()) == 1.0
    assert np.isfinite(predict_pooled(model=model, batch=own)).all()


def test_zero_weeks_of_history_onboards_at_the_population_mean(batch: PooledBatch):
    config = DpConfig(population_scales=PopulationScales())
    keep = ~site_mask(batch=batch, label="S3")
    (model,), _ = _fit(batch, config, [FitCopy(name="loso", keep=keep)])
    empty = select_rows(batch=batch, keep=torch.zeros_like(keep))
    year = int(batch.first_season_year + batch.n_season_years - 1)
    onboard_site(model=model, batch=empty, label="S3", forecast_season_year=year, steps=STEPS)
    site = batch.site_labels.index("S3")
    assert float(model.level[site].detach()) == pytest.approx(
        float(model.population_mean["level"].detach())
    )
    own = select_rows(batch=batch, keep=site_mask(batch=batch, label="S3"))
    assert np.isfinite(predict_pooled(model=model, batch=own)).all()


def test_onboarding_rejects_the_decay_prior_and_another_sites_rows(batch: PooledBatch):
    (decay,), _ = _fit(batch, DpConfig(), [FitCopy(name="d")])
    own = select_rows(batch=batch, keep=site_mask(batch=batch, label="S3"))
    with pytest.raises(ValueError, match="population prior"):
        onboard_site(model=decay, batch=own, label="S3", forecast_season_year=2024, steps=1)
    (model,), _ = _fit(batch, DpConfig(population_scales=PopulationScales()), [FitCopy(name="m")])
    with pytest.raises(ValueError, match="must all belong"):
        onboard_site(model=model, batch=batch, label="S3", forecast_season_year=2024, steps=1)


def test_the_cross_fitted_spread_never_uses_the_held_out_sites_residuals():
    residuals = PooledResiduals(
        site=np.array(["A"] * 400 + ["B"] * 400),
        residual=np.concatenate([np.full(400, 100.0), np.linspace(-1, 1, 400)]),
        band=np.zeros(800, dtype=np.int64),
        quarter=np.zeros(800, dtype=np.int64),
    )
    spread = cross_fitted_spread(residuals=residuals, held_out="A")
    assert spread.table.max() <= 1.0
    only_held_out = replace(residuals, site=np.array(["A"] * 800))
    with pytest.raises(ValueError, match="no residuals"):
        cross_fitted_spread(residuals=only_held_out, held_out="A")


def test_the_pooled_residuals_are_actual_minus_the_predicted_mean(batch: PooledBatch):
    (model,), _ = _fit(batch, DpConfig(population_scales=PopulationScales()), [FitCopy(name="m")])
    result = pooled_residuals(model=model, batch=batch)
    expected = batch.y.numpy() - predict_pooled(model=model, batch=batch)
    assert np.allclose(result.residual, expected)
    assert set(result.site) == set(SITES)


def test_after_mask_keeps_only_rows_ending_after_the_instant(
    batch: PooledBatch, dp_synthetic_cache: FeatureCache
):
    cache = _cache(dp_synthetic_cache)
    cutoff = np.datetime64("2023-07-01T00:00:00", "us")
    mask = after_mask(batch=batch, cache=cache, after=cutoff)
    ends = cache.end_times[batch.row.numpy()]
    assert bool(mask.all() == (ends > cutoff).all())
    assert int(mask.sum()) == int((ends > cutoff).sum())
