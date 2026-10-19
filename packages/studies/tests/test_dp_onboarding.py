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
    PooledDemandModel,
    PopulationScales,
    build_pooled_batch,
)

pytestmark = pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")

CUT: Final[np.datetime64] = np.datetime64("2024-07-01T00:00:00", "us")
STEPS: Final[int] = 20
SITES: Final[tuple[str, ...]] = ("S1", "S2", "S3")
HELD_OUT: Final[str] = "S3"
FORECAST_YEAR: Final[int] = 2024


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
def cache(dp_synthetic_cache: FeatureCache) -> FeatureCache:
    return _cache(dp_synthetic_cache)


@pytest.fixture(scope="module")
def batch(cache: FeatureCache) -> PooledBatch:
    return build_pooled_batch(cache=cache, sites=SITES, cut_end=CUT)


@pytest.fixture(scope="module")
def solar_batch(batch: PooledBatch) -> PooledBatch:
    """The batch with a known truth: a daily shape, and a solar farm at each MW substation."""
    angle = 2.0 * np.pi * batch.half_hour.numpy() / 48.0
    farm = torch.where(batch.site == 0, 8.0, 10.0)
    truth = 50.0 + 3.0 * torch.as_tensor(np.cos(angle), dtype=batch.y.dtype)
    return with_targets(batch=batch, y=truth - farm * batch.pv_shape * batch.solar_gate)


def _level(batch: PooledBatch) -> torch.Tensor:
    return torch.full((len(batch.site_labels),), 50.0)


def _fit(batch: PooledBatch, config: DpConfig, copies: list[FitCopy], steps: int = STEPS):  # noqa: ANN202
    return fit_many(
        batch=batch,
        copies=copies,
        config=config,
        initial_level=_level(batch),
        steps=steps,
        learning_rate=0.1,
    )


def _held_out_keep(batch: PooledBatch) -> torch.Tensor:
    return ~site_mask(batch=batch, label=HELD_OUT)


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
        for n in start
    ]
    assert any(moved)
    for name, scale in start.items():
        assert float(fixed.population_log_scale[name].detach()) == pytest.approx(np.log(scale))


def test_the_kl_term_matches_its_closed_form():
    model = PooledDemandModel(
        config=DpConfig(variational=True), n_sites=2, n_season_years=2, initial_level=torch.zeros(2)
    )
    mean = torch.tensor([1.0, 2.0])
    log_sd = torch.log(torch.tensor([0.5, 2.0]))
    with torch.no_grad():
        model.population_mean["level"].fill_(0.5)
        model.population_log_scale["level"].fill_(float(np.log(3.0)))
    term = model._family_term(name="level", mean=mean, log_sd=log_sd, active=torch.ones(2))
    sd, centre, scale = np.array([0.5, 2.0]), 0.5, 3.0
    expected = np.log(scale / sd) + (sd**2 + (np.array([1.0, 2.0]) - centre) ** 2) / (2 * scale**2)
    assert float(term) == pytest.approx(float((expected - 0.5).sum()), rel=1e-5)


def test_a_site_specific_capacity_prior_centres_the_first_capacity_increment(batch: PooledBatch):
    (model,), _ = _fit(batch, DpConfig(population_scales=PopulationScales()), [FitCopy(name="m")])
    site = batch.site_labels.index("S1")
    base = float(model.site_prior_term())
    with torch.no_grad():
        model.capacity_prior[site] = float(model.capacity_raw[site].max())
        near = float(model.site_prior_term())
        model.capacity_prior[site] = 50.0
        far = float(model.site_prior_term())
    assert far > near
    assert near != base or far != base


def test_the_loss_is_the_negative_elbo_per_row(batch: PooledBatch):
    (model,), _ = _fit(batch, DpConfig(variational=True), [FitCopy(name="v")])
    model.eval()
    with torch.no_grad():
        residual = model(batch) - batch.y
        noise_variance = torch.exp(2.0 * model.log_noise)
        expected = (
            (residual**2).mean() / (2.0 * noise_variance)
            + model.log_noise
            + (model.penalty() + model.site_prior_term()) / batch.n_rows
        )
        assert float(model.loss(batch=batch)) == pytest.approx(float(expected), rel=1e-5)


def test_a_held_out_sites_targets_and_starting_values_cannot_change_a_fit(batch: PooledBatch):
    keep = _held_out_keep(batch)
    config = DpConfig(variational=True)
    mask = site_mask(batch=batch, label=HELD_OUT)
    changed = with_targets(batch=batch, y=torch.where(mask, batch.y + 40.0, batch.y))
    (base,), _ = _fit(batch, config, [FitCopy(name="a", keep=keep)])
    (other,), _ = _fit(changed, config, [FitCopy(name="b", keep=keep)])
    assert np.allclose(
        predict_pooled(model=base, batch=batch),
        predict_pooled(model=other, batch=batch),
        atol=1e-3,
    )
    site = batch.site_labels.index(HELD_OUT)
    start_low = torch.full((3,), 50.0)
    start_high = torch.tensor([50.0, 50.0, 500.0])
    models = []
    for level in (start_low, start_high):
        torch.manual_seed(0)
        model = PooledDemandModel(
            config=config, n_sites=3, n_season_years=batch.n_season_years, initial_level=level
        )
        from studies.dp_pooled import fit_pooled

        fit_pooled(model=model, batch=select_rows(batch=batch, keep=keep), steps=STEPS)
        models.append(model)
    for name in ("level", "threshold", "slope_raw", "capacity_raw", "capacity_first"):
        assert float(models[0].population_mean[name]) == pytest.approx(
            float(models[1].population_mean[name]), rel=1e-4, abs=1e-4
        )
    assert float(models[0].level[site]) == pytest.approx(float(models[1].level[site]), rel=1e-4)


def test_the_site_prior_ignores_a_site_with_no_rows(batch: PooledBatch):
    (model,), _ = _fit(
        batch, DpConfig(variational=True), [FitCopy(name="l", keep=_held_out_keep(batch))]
    )
    site = batch.site_labels.index(HELD_OUT)
    before = float(model.site_prior_term())
    with torch.no_grad():
        model.deviation[site] += 100.0
        model.threshold[site] += 100.0
        model.capacity_raw[site] += 100.0
    assert float(model.site_prior_term()) == before


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


def _onboarded(
    solar_batch: PooledBatch,
    cache: FeatureCache,
    *,
    last_days: int = 60,
    config: DpConfig | None = None,
) -> tuple[PooledDemandModel, PooledBatch, dict[str, float], dict[str, torch.Tensor]]:
    """Fit a model without S3, then onboard S3 from its last `last_days` days."""
    config = config or DpConfig(variational=True, interaction_hidden=(8,))
    (model,), _ = _fit(
        solar_batch, config, [FitCopy(name="loso", keep=_held_out_keep(solar_batch))], steps=150
    )
    start = np.datetime64("2024-07-01T00:00:00", "us") - np.timedelta64(last_days, "D")
    keep = site_mask(batch=solar_batch, label=HELD_OUT) & after_mask(
        batch=solar_batch, cache=cache, after=start
    )
    own = select_rows(batch=solar_batch, keep=keep)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    info = onboard_site(
        model=model,
        batch=own,
        label=HELD_OUT,
        cache=cache,
        history_end=CUT,
        forecast_season_year=FORECAST_YEAR,
        steps=150,
    )
    return model, own, info, before


def test_onboarding_fits_the_capacity_of_the_season_year_its_rows_fall_in(
    solar_batch: PooledBatch, cache: FeatureCache
):
    model, own, info, before = _onboarded(solar_batch, cache)
    site = solar_batch.site_labels.index(HELD_OUT)
    year = int(own.season_slot.max() % model.n_season_years)
    assert info["rows"] == own.n_rows
    assert float(model.trained[site, year]) == 1.0
    assert float(model.trained[site, FORECAST_YEAR - solar_batch.first_season_year]) == 1.0
    moved = model.capacity_raw[site, year] - before["capacity_raw"][site, year]
    assert abs(float(moved)) > 0.5
    capacity = float(model.capacities()[site, -1])
    assert 4.0 < capacity < 20.0


def test_onboarding_changes_only_the_new_sites_parameters_and_leaves_other_sites_forecasts(
    solar_batch: PooledBatch, cache: FeatureCache
):
    config = DpConfig(variational=True, interaction_hidden=(8,))
    (model,), _ = _fit(
        solar_batch, config, [FitCopy(name="loso", keep=_held_out_keep(solar_batch))], steps=30
    )
    others = select_rows(batch=solar_batch, keep=_held_out_keep(solar_batch))
    before_forecast = predict_pooled(model=model, batch=others)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    centres = model.centre_means.clone()
    trained = model.trained.clone()
    own = select_rows(batch=solar_batch, keep=site_mask(batch=solar_batch, label=HELD_OUT))
    onboard_site(
        model=model,
        batch=own,
        label=HELD_OUT,
        cache=cache,
        history_end=CUT,
        forecast_season_year=FORECAST_YEAR,
        steps=30,
    )
    site = solar_batch.site_labels.index(HELD_OUT)
    for name, parameter in model.named_parameters():
        change = (parameter.detach() - before[name]).abs()
        if name.split(".")[0] in (*SITE_FAMILIES, "tan_phi", "q0", "posterior_log_sd"):
            others_changed = torch.arange(change.shape[0]) != site
            assert float(change[others_changed].sum()) == 0.0, name
        else:
            assert float(change.sum()) == 0.0, name
    assert np.array_equal(before_forecast, predict_pooled(model=model, batch=others))
    n_years = model.n_season_years
    kept = torch.arange(model.n_sites * n_years) // n_years != site
    assert torch.equal(model.centre_means[kept], centres[kept])
    assert torch.equal(
        model.trained[torch.arange(model.n_sites) != site],
        trained[torch.arange(model.n_sites) != site],
    )
    assert not torch.equal(model.centre_means[~kept], centres[~kept])


def test_onboarding_lowers_the_new_sites_loss_and_moves_its_parameters(
    solar_batch: PooledBatch, cache: FeatureCache
):
    model, own, info, before = _onboarded(solar_batch, cache)
    site = solar_batch.site_labels.index(HELD_OUT)
    assert np.isfinite(info["loss"])
    assert (
        float((model.level[site] - before["level"][site]).abs()) > 0.0
        or float((model.deviation[site] - before["deviation"][site]).abs().sum()) > 0.0
    )
    reading = predict_pooled(model=model, batch=own)
    assert float(np.abs(own.y.numpy() - reading).mean()) < 3.0


def test_the_onboarded_sites_centring_means_are_the_means_of_its_own_rows(
    solar_batch: PooledBatch, cache: FeatureCache
):
    model, own, _, _ = _onboarded(solar_batch, cache)
    site = solar_batch.site_labels.index(HELD_OUT)
    model.eval()
    with torch.no_grad():
        calendar, _ = model.calendar_terms(batch=own)
        centred = calendar - model.centre_means[own.season_slot]
    slot = int(own.season_slot[0])
    assert slot // model.n_season_years == site
    assert abs(float(centred[own.season_slot == slot].mean())) < 1e-4


def test_zero_weeks_of_history_onboards_at_the_population_mean(
    solar_batch: PooledBatch, cache: FeatureCache
):
    config = DpConfig(population_scales=PopulationScales())
    (model,), _ = _fit(
        solar_batch, config, [FitCopy(name="loso", keep=_held_out_keep(solar_batch))]
    )
    empty = select_rows(
        batch=solar_batch, keep=torch.zeros_like(solar_batch.site, dtype=torch.bool)
    )
    onboard_site(
        model=model,
        batch=empty,
        label=HELD_OUT,
        cache=cache,
        history_end=CUT,
        forecast_season_year=FORECAST_YEAR,
        steps=STEPS,
    )
    site = solar_batch.site_labels.index(HELD_OUT)
    assert float(model.level[site]) == pytest.approx(float(model.population_mean["level"]))
    assert float(
        model.capacity_raw[site, FORECAST_YEAR - solar_batch.first_season_year]
    ) == pytest.approx(float(model.population_mean["capacity_first"]))
    own = select_rows(batch=solar_batch, keep=site_mask(batch=solar_batch, label=HELD_OUT))
    assert np.isfinite(predict_pooled(model=model, batch=own)).all()


def test_onboarding_rejects_bad_inputs(batch: PooledBatch, cache: FeatureCache):
    (decay,), _ = _fit(batch, DpConfig(), [FitCopy(name="d")])
    own = select_rows(batch=batch, keep=site_mask(batch=batch, label=HELD_OUT))

    def onboard(model: PooledDemandModel, rows: PooledBatch, end: np.datetime64 = CUT) -> None:
        onboard_site(
            model=model,
            batch=rows,
            label=HELD_OUT,
            cache=cache,
            history_end=end,
            forecast_season_year=FORECAST_YEAR,
            steps=1,
        )

    with pytest.raises(ValueError, match="population prior"):
        onboard(decay, own)
    (full,), _ = _fit(batch, DpConfig(population_scales=PopulationScales()), [FitCopy(name="m")])
    with pytest.raises(ValueError, match="held out"):
        onboard(full, own)
    (loso,), _ = _fit(
        batch,
        DpConfig(population_scales=PopulationScales()),
        [FitCopy(name="l", keep=_held_out_keep(batch))],
    )
    with pytest.raises(ValueError, match="must all belong"):
        onboard(loso, batch)
    with pytest.raises(ValueError, match="ends after its history end"):
        onboard(loso, own, np.datetime64("2024-01-01T00:00:00", "us"))


def test_the_cross_fitted_spread_needs_a_fit_that_never_saw_the_held_out_site():
    residuals = PooledResiduals(
        site=np.array(["A"] * 400 + ["B"] * 400),
        residual=np.concatenate([np.full(400, 100.0), np.linspace(-1, 1, 400)]),
        band=np.zeros(800, dtype=np.int64),
        quarter=np.zeros(800, dtype=np.int64),
        fit_excluded=("A",),
        weeks_of_history=4,
    )
    spread = cross_fitted_spread(residuals=residuals, held_out="A", weeks_of_history=4)
    assert spread.table.max() <= 1.0
    with pytest.raises(ValueError, match="weeks of history"):
        cross_fitted_spread(residuals=residuals, held_out="A", weeks_of_history=8)
    with pytest.raises(ValueError, match="saw B"):
        cross_fitted_spread(residuals=residuals, held_out="B", weeks_of_history=4)
    seen = cross_fitted_spread(
        residuals=residuals, held_out="B", weeks_of_history=4, allow_seen=True
    )
    assert np.allclose(seen.table, 0.0)
    only_held_out = replace(residuals, site=np.array(["A"] * 800))
    with pytest.raises(ValueError, match="no residuals"):
        cross_fitted_spread(residuals=only_held_out, held_out="A", weeks_of_history=4)


def test_the_spread_is_centred_so_a_replica_years_bias_does_not_shift_the_forecast():
    generator = np.random.default_rng(0)
    residuals = PooledResiduals(
        site=np.array(["A"] * 2000 + ["B"] * 2000),
        residual=np.concatenate([np.zeros(2000), 10.0 + generator.normal(size=2000)]),
        band=np.tile(np.arange(4), 1000),
        quarter=np.zeros(4000, dtype=np.int64),
        fit_excluded=("A",),
        weeks_of_history=4,
    )
    spread = cross_fitted_spread(residuals=residuals, held_out="A", weeks_of_history=4)
    median = spread.table[:, 0, 5]
    assert np.abs(median).max() < 0.2
    assert spread.table[:, 0, 0].max() < -1.0
    assert spread.table[:, 0, -1].min() > 1.0


def test_the_pooled_residuals_are_actual_minus_the_predicted_mean_and_name_the_excluded_site(
    batch: PooledBatch,
):
    (model,), _ = _fit(
        batch,
        DpConfig(population_scales=PopulationScales()),
        [FitCopy(name="m", keep=_held_out_keep(batch))],
    )
    result = pooled_residuals(model=model, batch=batch, weeks_of_history=4)
    expected = batch.y.numpy() - predict_pooled(model=model, batch=batch)
    assert np.allclose(result.residual, expected)
    assert set(result.site) == set(SITES)
    assert result.fit_excluded == (HELD_OUT,)
    assert result.weeks_of_history == 4


def test_all_present_rows_adds_the_rows_the_default_rows_leave_out(cache: FeatureCache):
    masked = replace(
        cache.sites["S1"],
        default_rows=cache.sites["S1"].default_rows
        & (cache.end_times.astype("datetime64[M]").astype(int) % 2 == 0),
    )
    edited = replace(cache, sites={**cache.sites, "S1": masked})
    default = build_pooled_batch(cache=edited, sites=("S1",), cut_end=CUT)
    every = build_pooled_batch(cache=edited, sites=("S1",), cut_end=CUT, all_present_rows=True)
    assert every.n_rows > default.n_rows


def test_after_mask_keeps_only_rows_ending_after_the_instant(
    batch: PooledBatch, cache: FeatureCache
):
    cutoff = np.datetime64("2023-07-01T00:00:00", "us")
    mask = after_mask(batch=batch, cache=cache, after=cutoff)
    ends = cache.end_times[batch.row.numpy()]
    assert np.array_equal(mask.numpy(), ends > cutoff)
    assert 0 < int(mask.sum()) < batch.n_rows
