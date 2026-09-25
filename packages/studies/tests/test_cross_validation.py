from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Final, NamedTuple

import numpy as np
import polars as pl
import pytest
from studies.cross_validation import (
    ENS_HRES_WIND_ERA_FOLD_OFFSETS,
    ENS_HRES_WIND_ERA_START_MONTHS,
    N_FOLDS,
    PRIMARY_HYPER_PARAMETERS,
    QUANTILE_LEVELS,
    SEEDS,
    DeviceType,
    HyperParameters,
    assign_folds,
    booster_parameters,
    calendar_month_coverage,
    clamp_to_cap,
    crps,
    cut_eras,
    fit_one_fold,
    out_of_fold_losses,
    raise_on_uncovered_months,
    rotate_folds,
    search_fold_offsets,
    uncovered_months,
)

from studies import cross_validation


def _months(site: str, n_months: int) -> list[dict[str, object]]:
    return [{"site": site, "month": f"2025-{month + 1:02d}"} for month in range(n_months)]


def test_folds_match_the_published_scheme():
    # Pinned from the implementation the published results were cut with: a dense month rank per
    # site, scaled to five folds. A site with ten months takes two per fold; one with seven
    # still fills every fold.
    folds = assign_folds(dataset=pl.DataFrame(_months("A", 10) + _months("B", 7)))

    by_site = folds.group_by("site", maintain_order=True).agg(pl.col("fold"))

    assert by_site["fold"].to_list() == [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 0, 1, 2, 2, 3, 4]]


def test_many_rows_a_month_are_ranked_by_month_not_by_row():
    # Real months hold hundreds of rows each. Ranking rows rather than distinct months would push
    # almost every month into the last fold.
    rows = pl.DataFrame(sorted(_months("A", 10) * 3, key=lambda row: str(row["month"])))

    folds = assign_folds(dataset=rows)

    assert folds["fold"].to_list() == [fold for fold in range(N_FOLDS) for _ in range(6)]


def test_grouping_by_era_puts_every_era_in_every_fold():
    rows = pl.DataFrame(_months("A", 12)).with_columns(
        era=pl.when(pl.col("month") < "2025-08").then(pl.lit("before")).otherwise(pl.lit("after"))
    )

    folds = assign_folds(dataset=rows, by=("site", "era"))

    eras_per_fold = folds.group_by("fold").agg(pl.col("era").n_unique()).sort("fold")
    assert eras_per_fold["fold"].to_list() == list(range(N_FOLDS))
    assert eras_per_fold["era"].to_list() == [2] * N_FOLDS


def test_crps_of_a_forecast_every_level_puts_one_above_the_outcome():
    # Every level misses by 1 on the high side, so each contributes (1 - level), and twice the
    # level spacing times their sum is 2 * 0.1 * 4.5.
    quantiles = np.ones((1, 9))

    assert crps(actual=np.zeros(1), quantiles=quantiles) == pytest.approx([0.9])


def test_crps_weights_an_overshoot_by_one_minus_its_level():
    # Only the 0.9 quantile misses, by 1 on the high side, so the score is 2 * 0.1 * (1 - 0.9).
    # Swapping the two pinball weights would give 2 * 0.1 * 0.9 instead.
    quantiles = np.zeros((1, 9))
    quantiles[0, -1] = 1.0

    assert crps(actual=np.zeros(1), quantiles=quantiles) == pytest.approx([0.02])


def test_crps_sorts_crossing_quantiles_before_scoring():
    ordered = np.linspace(0.0, 2.0, 9)[None, :]

    assert crps(actual=np.ones(1), quantiles=ordered[:, ::-1]) == pytest.approx(
        crps(actual=np.ones(1), quantiles=ordered)
    )


def test_the_booster_settings_translate_exactly_and_never_subsample_columns():
    # An ignored seed would make every seed fit the same model and the seed spread read zero.
    assert booster_parameters(hyper_parameters=PRIMARY_HYPER_PARAMETERS, seed=3) == {
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "min_child_weight": 20.0,
        "lambda": 1.0,
        "tree_method": "hist",
        "seed": 3,
        "nthread": 4,
        "device": "cpu",
    }


def test_the_booster_device_is_passed_through_and_defaults_to_the_cpu():
    # A GPU fit is not bit-identical to a CPU fit, so a study that mixes the two must be able to
    # name the device of each fit; silently falling back to the CPU would put a GPU arm in a CPU
    # contrast.
    assert booster_parameters(hyper_parameters=PRIMARY_HYPER_PARAMETERS, seed=0)["device"] == "cpu"
    assert (
        booster_parameters(hyper_parameters=PRIMARY_HYPER_PARAMETERS, seed=0, device="cuda")[
            "device"
        ]
        == "cuda"
    )


def test_the_published_seeds_are_the_ones_fitted():
    assert SEEDS == (0, 1, 2)


def test_each_model_is_fitted_on_its_own_objective(monkeypatch: pytest.MonkeyPatch):
    # The point model is scored on absolute error, so it has to be fitted on it; the quantile model
    # has to be fitted at every level the score integrates over.
    calls: list[tuple[dict[str, object], int]] = []

    class _Booster:
        def predict(self, matrix: object) -> np.ndarray:
            return np.zeros((4, 9))

    def _train(parameters: dict[str, object], matrix: object, num_boost_round: int) -> _Booster:
        calls.append((parameters, num_boost_round))
        return _Booster()

    monkeypatch.setattr(cross_validation.xgb, "train", _train)
    site_rows = _site_rows()
    fit_one_fold(
        train=site_rows,
        test=site_rows.head(4),
        features=["x"],
        target="power_mw",
        hyper_parameters=PRIMARY_HYPER_PARAMETERS,
        seed=0,
        with_quantiles=True,
    )

    (point, point_rounds), (quantile, quantile_rounds) = calls
    assert point["objective"] == "reg:absoluteerror"
    assert quantile["objective"] == "reg:quantileerror"
    assert np.asarray(quantile["quantile_alpha"]).tolist() == list(QUANTILE_LEVELS)
    assert point_rounds == quantile_rounds == PRIMARY_HYPER_PARAMETERS["num_boost_round"]
    assert point["device"] == quantile["device"] == "cpu"


def test_the_requested_device_reaches_both_boosters(monkeypatch: pytest.MonkeyPatch):
    # A device dropped between `fit_one_fold` and `xgb.train` would fit a "GPU" arm on the CPU.
    devices: list[object] = []

    class _Booster:
        def predict(self, matrix: object) -> np.ndarray:
            return np.zeros((4, 9))

    def _train(parameters: dict[str, object], matrix: object, num_boost_round: int) -> _Booster:
        devices.append(parameters["device"])
        return _Booster()

    monkeypatch.setattr(cross_validation.xgb, "train", _train)
    site_rows = _site_rows()
    fit_one_fold(
        train=site_rows,
        test=site_rows.head(4),
        features=["x"],
        target="power_mw",
        hyper_parameters=PRIMARY_HYPER_PARAMETERS,
        seed=0,
        with_quantiles=True,
        device="cuda",
    )

    assert devices == ["cuda", "cuda"]


def test_the_clamp_holds_a_prediction_to_its_cap_and_leaves_uncapped_rows_alone():
    clamped = clamp_to_cap(prediction=np.array([5.0, 5.0, 1.0]), cap_mw=pl.Series([3.0, None, 3.0]))

    assert clamped.tolist() == [3.0, 5.0, 1.0]


def test_the_clamp_holds_each_quantile_row_to_its_own_cap():
    clamped = clamp_to_cap(
        prediction=np.array([[1.0, 5.0], [1.0, 5.0]]), cap_mw=pl.Series([3.0, None])
    )

    assert clamped.tolist() == [[1.0, 3.0], [1.0, 5.0]]


def _site_rows() -> pl.DataFrame:
    """Ten months of one site, four rows a month, with a fold-specific feature per fold."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    times = [start + timedelta(days=31 * month + day) for month in range(10) for day in range(4)]
    frame = pl.DataFrame(
        {
            "site": "A",
            "time": times,
            "month": [time.strftime("%Y-%m") for time in times],
            "power_mw": np.linspace(0.0, 3.0, len(times)).astype(np.float32),
            "x": np.arange(len(times), dtype=np.float64),
            # Unequal capacities, so dividing by a pooled capacity cannot pass for dividing each row
            # by its own.
            "effective_capacity_mw": np.where(np.arange(len(times)) % 2 == 0, 4.0, 5.0).astype(
                np.float32
            ),
            "cap_mw": [2.0 if index % 5 == 0 else None for index in range(len(times))],
            "constrained": [index % 7 == 0 for index in range(len(times))],
        }
    )
    return assign_folds(dataset=frame).with_columns(
        pl.col("x").alias(f"x_fold{fold}") for fold in range(N_FOLDS)
    )


class _Call(NamedTuple):
    train: pl.DataFrame
    test: pl.DataFrame
    features: list[str]


class _RecordingFit:
    """Stands in for the booster, recording what each fit was shown."""

    def __init__(self, *, offset_mw: float) -> None:
        self.offset_mw = offset_mw
        self.calls: list[_Call] = []
        self.devices: list[str] = []

    def __call__(
        self,
        *,
        train: pl.DataFrame,
        test: pl.DataFrame,
        features: list[str],
        target: str,
        hyper_parameters: object,
        seed: int,
        with_quantiles: bool,
        weight: str | None = None,
        device: str = "cpu",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self.calls.append(_Call(train=train, test=test, features=features))
        self.devices.append(device)
        point = test[target].to_numpy() + self.offset_mw
        return point, (np.repeat(point[:, None], 9, axis=1) if with_quantiles else None)


def _run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    site_rows: pl.DataFrame,
    offset_mw: float = 1.0,
    device: DeviceType = "cpu",
) -> tuple[pl.DataFrame, _RecordingFit]:
    fit = _RecordingFit(offset_mw=offset_mw)
    monkeypatch.setattr(cross_validation, "fit_one_fold", fit)
    losses = out_of_fold_losses(
        site_rows=site_rows,
        features=["x", "x_fold{fold}"],
        target="power_mw",
        hyper_parameters=PRIMARY_HYPER_PARAMETERS,
        with_quantiles=True,
        device=device,
    )
    return losses, fit


def test_every_fold_and_seed_is_fitted_on_the_requested_device(monkeypatch: pytest.MonkeyPatch):
    # A device that reached only the first fit would leave the rest on the CPU, and a study would
    # report a GPU arm that was mostly fitted on the CPU.
    _, fit = _run(monkeypatch, site_rows=_site_rows(), device="cuda")
    assert fit.devices
    assert set(fit.devices) == {"cuda"}
    _, cpu_fit = _run(monkeypatch, site_rows=_site_rows())
    assert set(cpu_fit.devices) == {"cpu"}


def test_no_fit_trains_on_the_fold_it_scores(monkeypatch: pytest.MonkeyPatch):
    _, fit = _run(monkeypatch, site_rows=_site_rows())

    for call in fit.calls:
        assert set(call.train["time"]).isdisjoint(set(call.test["time"]))


def test_constrained_rows_are_scored_but_never_trained_on(monkeypatch: pytest.MonkeyPatch):
    site_rows = _site_rows()
    losses, fit = _run(monkeypatch, site_rows=site_rows)

    for call in fit.calls:
        assert not call.train["constrained"].any()
    assert losses.filter(pl.col("constrained")).height == (
        site_rows["constrained"].sum() * len(SEEDS)
    )


def test_every_row_is_scored_once_per_seed(monkeypatch: pytest.MonkeyPatch):
    site_rows = _site_rows()
    losses, _ = _run(monkeypatch, site_rows=site_rows)

    assert losses.height == site_rows.height * len(SEEDS)
    assert losses.select("time", "seed").n_unique() == losses.height


def test_a_fold_placeholder_names_the_fold_being_scored(monkeypatch: pytest.MonkeyPatch):
    _, fit = _run(monkeypatch, site_rows=_site_rows())

    for call in fit.calls:
        assert call.features == ["x", f"x_fold{call.test['fold'][0]}"]


def test_a_fold_with_nothing_to_train_on_is_skipped(monkeypatch: pytest.MonkeyPatch):
    # Every row outside fold 0 is constrained, so fold 0 has no training rows at all.
    site_rows = _site_rows().with_columns(constrained=pl.col("fold") != 0)

    losses, _ = _run(monkeypatch, site_rows=site_rows)

    assert 0 not in losses["fold"].to_list()


def test_the_capped_losses_are_clamped_and_the_uncapped_are_not(monkeypatch: pytest.MonkeyPatch):
    # The stub predicts 10 MW above the outcome at every quantile level. Uncapped, every level
    # overshoots by 10, so the score is 2 * 0.1 * 10 * sum(1 - level) = 9.
    losses, _ = _run(monkeypatch, site_rows=_site_rows(), offset_mw=10.0)
    joined = losses.join(_site_rows().select("time", "cap_mw", "power_mw"), on="time")
    capped = joined.filter(pl.col("cap_mw").is_not_null())
    shortfall = capped["cap_mw"].to_numpy() - capped["power_mw"].to_numpy()

    assert joined["absolute_error_mw"].to_numpy() == pytest.approx(10.0)
    assert joined["signed_error_mw"].to_numpy() == pytest.approx(10.0)
    assert joined["crps_mw"].to_numpy() == pytest.approx(9.0)
    assert capped["absolute_error_capped_mw"].to_numpy() == pytest.approx(np.abs(shortfall))
    assert capped["signed_error_capped_mw"].to_numpy() == pytest.approx(shortfall)
    assert capped["crps_capped_mw"].to_numpy() == pytest.approx(
        crps(
            actual=capped["power_mw"].to_numpy(),
            quantiles=np.repeat(capped["cap_mw"].to_numpy()[:, None], 9, axis=1),
        )
    )


def test_every_loss_is_float64_whatever_the_target_precision(monkeypatch: pytest.MonkeyPatch):
    losses, _ = _run(monkeypatch, site_rows=_site_rows())

    loss_columns = [name for name in losses.columns if name.endswith(("_mw", "_of_capacity"))]
    assert {losses.schema[name] for name in loss_columns} == {pl.Float64}


def test_each_row_is_divided_by_its_own_capacity(monkeypatch: pytest.MonkeyPatch):
    losses, _ = _run(monkeypatch, site_rows=_site_rows())

    assert (
        losses["absolute_error_fraction_of_capacity"].to_list()
        == (losses["absolute_error_mw"] / losses["effective_capacity_mw"]).to_list()
    )
    assert (
        losses["absolute_error_capped_fraction_of_capacity"].to_list()
        == (losses["absolute_error_capped_mw"] / losses["effective_capacity_mw"]).to_list()
    )


def test_a_real_fit_is_reproducible_for_a_seed():
    site_rows = _site_rows()
    train = site_rows.filter(pl.col("fold") != 0)
    test = site_rows.filter(pl.col("fold") == 0)
    few_rounds = HyperParameters(**{**PRIMARY_HYPER_PARAMETERS, "num_boost_round": 5})

    first, _ = fit_one_fold(
        train=train,
        test=test,
        features=["x"],
        target="power_mw",
        hyper_parameters=few_rounds,
        seed=0,
        with_quantiles=False,
    )
    second, _ = fit_one_fold(
        train=train,
        test=test,
        features=["x"],
        target="power_mw",
        hyper_parameters=few_rounds,
        seed=0,
        with_quantiles=False,
    )

    assert first.tolist() == second.tolist()


# Two sites with unequal eras. Both start their second era in July 2025. Site A has six months
# before it and twelve after; site B has four before it and fourteen after. Cut into five folds
# per era, the dense month ranks give (rank - 1) * 5 // n_months, worked out by hand below.
ERA_START: Final[tuple[str, ...]] = ("2025-07",)
ROTATED: Final[MappingProxyType[int, int]] = MappingProxyType({0: 0, 1: 2})
UNROTATED: Final[MappingProxyType[int, int]] = MappingProxyType({0: 0, 1: 0})


def _monthly_rows(
    *, site: str, first_year_month: tuple[int, int], n_months: int, days: Sequence[int] = (15,)
) -> pl.DataFrame:
    """One row per listed day of each month, so a month's rows outnumber its years."""
    year, month = first_year_month
    steps = [divmod(month - 1 + step, 12) for step in range(n_months)]
    return pl.DataFrame(
        {
            "site": [site for _ in steps for _ in days],
            "month": [f"{year + carry}-{index + 1:02d}" for carry, index in steps for _ in days],
            "time": [
                datetime(year + carry, index + 1, day) for carry, index in steps for day in days
            ],
        }
    )


def _two_sites() -> pl.DataFrame:
    days = (1, 10, 20)
    return pl.concat(
        [
            _monthly_rows(site="A", first_year_month=(2025, 1), n_months=18, days=days),
            _monthly_rows(site="B", first_year_month=(2025, 3), n_months=18, days=days),
        ]
    )


def _folds_by_site(*, frame: pl.DataFrame) -> dict[str, list[int]]:
    """Each site's fold per month, in month order (a month's rows all share one fold)."""
    months = frame.unique(["site", "month"]).sort("site", "month")
    grouped = months.group_by("site", maintain_order=True).agg("fold")
    return {row["site"]: row["fold"] for row in grouped.iter_rows(named=True)}


def test_era_folds_match_a_hand_computed_layout_at_the_boundary_months():
    # Site A: eras Jan-Jun 2025 (6 months: 0,0,1,2,3,4) and Jul 2025-Jun 2026 (12 months:
    # 0,0,0,1,1,2,2,2,3,3,4,4). Site B: Mar-Jun 2025 (4 months: 0,1,2,3) and Jul 2025-Aug 2026
    # (14 months: 0,0,0,1,1,1,2,2,2,3,3,3,4,4). Rotating era 1 by 2 adds 2 modulo 5 to that era
    # only.
    # June 2025 is the last month of era 0 and July 2025 the first of era 1.
    cut = cut_eras(frame=_two_sites(), first_months=ERA_START, fold_offsets=ROTATED)

    assert _folds_by_site(frame=cut) == {
        "A": [0, 0, 1, 2, 3, 4, 2, 2, 2, 3, 3, 4, 4, 4, 0, 0, 1, 1],
        "B": [0, 1, 2, 3, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0, 1, 1],
    }
    boundary = cut.filter(pl.col("month").is_in(["2025-06", "2025-07"])).sort("site", "month")
    assert boundary.unique(["site", "month"]).sort("site", "month")["era_code"].to_list() == [
        0,
        1,
        0,
        1,
    ]


def test_no_rotation_leaves_every_era_cut_from_fold_zero():
    cut = cut_eras(frame=_two_sites(), first_months=ERA_START, fold_offsets=UNROTATED)

    assert _folds_by_site(frame=cut) == {
        "A": [0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4],
        "B": [0, 1, 2, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    }


def test_each_site_is_cut_independently_of_the_other_site():
    both = cut_eras(frame=_two_sites(), first_months=ERA_START, fold_offsets=ROTATED)
    alone = cut_eras(
        frame=_two_sites().filter(pl.col("site") == "B"),
        first_months=ERA_START,
        fold_offsets=ROTATED,
    )

    assert _folds_by_site(frame=both)["B"] == _folds_by_site(frame=alone)["B"]


def test_three_eras_take_their_offsets_from_the_era_code():
    frame = _monthly_rows(site="A", first_year_month=(2025, 1), n_months=15)

    cut = cut_eras(
        frame=frame, first_months=("2025-04", "2025-10"), fold_offsets={0: 0, 1: 1, 2: 3}
    )

    # Eras of 3, 6 and 6 months cut to (0,1,3), (0,0,1,2,3,4) and (0,0,1,2,3,4), then rotated by
    # 0, 1 and 3.
    assert cut["era_code"].to_list() == [0] * 3 + [1] * 6 + [2] * 6
    assert cut["fold"].to_list() == [0, 1, 3, 1, 1, 2, 3, 4, 0, 3, 3, 4, 0, 1, 2]


def test_rotation_wraps_modulo_the_number_of_folds():
    frame = pl.DataFrame({"era_code": [0, 1, 1], "fold": [4, 3, 4]}).cast(
        {"era_code": pl.Int8, "fold": pl.Int32}
    )

    rotated = rotate_folds(frame=frame, fold_offsets={0: 1, 1: 3})

    assert rotated["fold"].to_list() == [0, 1, 2]


def test_calendar_month_coverage_finds_the_holes_of_an_unrotated_design():
    # June occurs in 2025 and 2026 at both sites and lands in the same fold in both years, so
    # holding that fold out leaves no June to train on: site A's fold 4 and site B's fold 3.
    cut = cut_eras(frame=_two_sites(), first_months=ERA_START, fold_offsets=UNROTATED)

    holes = uncovered_months(coverage=calendar_month_coverage(frame=cut))

    assert holes.sort("site").select(
        "site", "fold", "calendar_month", "n_scored", "n_train"
    ).rows() == [
        ("A", 4, 6, 6, 0),
        ("B", 3, 6, 6, 0),
    ]


def test_every_cell_of_a_rotated_design_has_a_training_row_for_a_month_in_two_years():
    cut = cut_eras(frame=_two_sites(), first_months=ERA_START, fold_offsets=ROTATED)

    coverage = calendar_month_coverage(frame=cut)

    assert uncovered_months(coverage=coverage).height == 0
    two_years = coverage.filter(pl.col("n_years") == 2)
    assert two_years.height > 0
    assert (two_years["n_train"] > 0).all()


def test_coverage_counts_scored_and_training_rows_of_a_cell():
    # Rotated, site A's January is in fold 0 in 2025 and fold 4 in 2026: each fold scores one
    # January and trains on the other.
    cut = cut_eras(frame=_two_sites(), first_months=ERA_START, fold_offsets=ROTATED)

    january = (
        calendar_month_coverage(frame=cut)
        .filter(pl.col("site") == "A", pl.col("calendar_month") == 1)
        .sort("fold")
    )

    # Three rows a month: each fold scores 3 January rows and trains on the other year's 3. Counting
    # distinct years instead of rows would give a training count of 1.
    assert january.select("fold", "n_scored", "n_train", "n_years").rows() == [
        (0, 3, 3, 2),
        (4, 3, 3, 2),
    ]


def test_a_calendar_month_in_one_year_is_uncovered_but_is_not_a_failure():
    # Site A has July to December in 2025 only, and site B has January and February in 2026 only
    # and September to December in 2025 only: no fold design can cover them.
    cut = cut_eras(frame=_two_sites(), first_months=ERA_START, fold_offsets=ROTATED)
    coverage = calendar_month_coverage(frame=cut)

    single_year = coverage.filter(pl.col("n_years") == 1)

    months_by_site = single_year.group_by("site").agg(pl.col("calendar_month").unique().sort())
    assert dict(months_by_site.sort("site").rows()) == {
        "A": [7, 8, 9, 10, 11, 12],
        "B": [1, 2, 9, 10, 11, 12],
    }
    assert not single_year["covered"].any()
    assert uncovered_months(coverage=coverage).height == 0


def test_an_uncovered_scored_month_raises_naming_the_count_and_the_cell():
    cut = cut_eras(frame=_two_sites(), first_months=ERA_START, fold_offsets=UNROTATED)

    with pytest.raises(ValueError, match=r"^2 \(site, fold, calendar month\) cells .*'site': 'A'"):
        raise_on_uncovered_months(coverage=calendar_month_coverage(frame=cut))


def _coverage_with_holes(*, sites: Sequence[str]) -> pl.DataFrame:
    """Coverage rows in which each listed site has one uncovered calendar month in two years."""
    return pl.DataFrame(
        {
            "site": list(sites),
            "fold": [0] * len(sites),
            "calendar_month": [6] * len(sites),
            "n_scored": [2] * len(sites),
            "n_train": [0] * len(sites),
            "n_years": [2] * len(sites),
            "covered": [False] * len(sites),
        }
    )


def test_a_single_uncovered_cell_raises():
    with pytest.raises(ValueError, match=r"^1 \(site, fold, calendar month\) cells .*'site': 'A'"):
        raise_on_uncovered_months(coverage=_coverage_with_holes(sites=["A"]))


def test_the_raise_names_every_failing_cell_up_to_five():
    with pytest.raises(ValueError, match=r"^7 .*'site': 'A'.*'site': 'B'.*'site': 'E'") as raised:
        raise_on_uncovered_months(coverage=_coverage_with_holes(sites=list("ABCDEFG")))

    assert "'site': 'F'" not in str(raised.value)


def test_more_than_one_year_is_the_message_wording():
    with pytest.raises(ValueError, match="occurs in more than one year"):
        raise_on_uncovered_months(coverage=_coverage_with_holes(sites=["A"]))


def test_coverage_is_sorted_by_site_fold_and_calendar_month_whatever_the_row_order():
    cut = cut_eras(frame=_two_sites(), first_months=ERA_START, fold_offsets=ROTATED)
    shuffled = cut.sample(fraction=1.0, shuffle=True, seed=0)

    coverage = calendar_month_coverage(frame=shuffled)

    assert coverage.equals(coverage.sort("site", "fold", "calendar_month"))
    assert coverage.height > 1


def test_a_site_that_starts_after_the_first_boundary_has_only_the_later_eras():
    # Site C runs from 2025-09 to 2026-08. Era 1 holds its first 4 months, cut to folds 0,1,2,3 and
    # rotated by 2 to 2,3,4,0. Era 2 (from 2026-01) holds 8 months, cut to 0,0,1,1,2,3,3,4 and
    # rotated by 1 to 1,1,2,2,3,4,4,0. No row is in era 0, and no fold is cut for it.
    site_c = _monthly_rows(site="C", first_year_month=(2025, 9), n_months=12)

    cut = cut_eras(
        frame=site_c, first_months=("2025-07", "2026-01"), fold_offsets={0: 0, 1: 2, 2: 1}
    )

    assert set(cut["era_code"]) == {1, 2}
    assert cut["fold"].to_list() == [2, 3, 4, 0, 1, 1, 2, 2, 3, 4, 4, 0]


def test_cut_eras_raises_naming_an_era_without_an_offset():
    with pytest.raises(ValueError, match=r"missing \[1\], extra \[\]"):
        cut_eras(frame=_two_sites(), first_months=ERA_START, fold_offsets={0: 0})


def test_cut_eras_raises_naming_an_offset_for_an_era_that_does_not_exist():
    with pytest.raises(ValueError, match=r"missing \[\], extra \[2\]"):
        cut_eras(frame=_two_sites(), first_months=ERA_START, fold_offsets={0: 0, 1: 0, 2: 0})


@pytest.mark.parametrize("bad_month", ["2025-7", "2025-07-01", "July 2025", "25-07", ""])
def test_cut_eras_raises_on_a_first_month_that_is_not_a_year_month_label(bad_month: str):
    with pytest.raises(ValueError, match="%Y-%m"):
        cut_eras(frame=_two_sites(), first_months=(bad_month,), fold_offsets=ROTATED)


def test_no_first_months_gives_one_era_coded_int8():
    frame = _monthly_rows(site="A", first_year_month=(2025, 1), n_months=10)

    cut = cut_eras(frame=frame, first_months=(), fold_offsets={0: 0})

    assert cut.schema["era_code"] == pl.Int8
    assert cut["era_code"].to_list() == [0] * 10
    assert cut["fold"].to_list() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_rotate_folds_raises_naming_an_era_code_without_an_offset():
    frame = pl.DataFrame({"era_code": [0, 1, 2], "fold": [0, 0, 0]}).cast(
        {"era_code": pl.Int8, "fold": pl.Int32}
    )

    with pytest.raises(ValueError, match=r"no offset for era_code \[1, 2\]"):
        rotate_folds(frame=frame, fold_offsets={0: 0})


def test_the_search_finds_every_zero_hole_rotation_smallest_first():
    found = search_fold_offsets(frame=_two_sites(), first_months=ERA_START)

    # Each candidate is checked against the coverage function itself, so the list is exactly the
    # rotations of era 1 that leave no hole, and the rotation the tests use appears in it.
    covered = [
        offset
        for offset in range(N_FOLDS)
        if uncovered_months(
            coverage=calendar_month_coverage(
                frame=cut_eras(
                    frame=_two_sites(), first_months=ERA_START, fold_offsets={0: 0, 1: offset}
                )
            )
        ).is_empty()
    ]
    assert [dict(offsets) for offsets in found] == [{0: 0, 1: offset} for offset in covered]
    assert dict(ROTATED) in [dict(offsets) for offsets in found]
    assert {0: 0, 1: 0} not in [dict(offsets) for offsets in found]


def test_the_search_orders_by_non_zero_rotations_then_by_sum():
    frame = _monthly_rows(site="A", first_year_month=(2025, 1), n_months=30, days=(1, 10))

    found = search_fold_offsets(frame=frame, first_months=("2025-07", "2026-03"))

    keys = [
        (sum(v != 0 for v in offsets.values()), sum(offsets.values()), tuple(offsets.values()))
        for offsets in found
    ]
    assert len(found) > 2
    assert keys == sorted(keys)
    assert all(offsets[0] == 0 for offsets in found)


def test_the_search_tries_every_rotation_and_sorts_by_count_then_sum(
    monkeypatch: pytest.MonkeyPatch,
):
    # With every design accepted, the result is all 25 pairs of rotations of eras 1 and 2. The
    # pair (1, 3) sums to 4 and (2, 1) to 3, so ordering by sum puts (2, 1) first where ordering
    # by the rotations alone would not.
    monkeypatch.setattr(cross_validation, "uncovered_months", lambda *, coverage: coverage.clear())
    frame = _monthly_rows(site="A", first_year_month=(2025, 1), n_months=15)

    found = search_fold_offsets(frame=frame, first_months=("2025-04", "2025-10"))

    pairs = [(offsets[1], offsets[2]) for offsets in found]
    assert len(pairs) == 25
    assert pairs[:9] == [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (0, 4), (4, 0)]
    assert pairs.index((2, 1)) < pairs.index((1, 3))
    assert pairs[-1] == (4, 4)


def test_the_search_returns_an_empty_list_when_no_rotation_can_cover():
    # January 2025 and January 2026 are adjacent month ranks in the first of five folds, and no
    # rotation of a single era separates them, so no design covers January.
    months = ["2025-01", "2026-01", *(f"2026-{month:02d}" for month in range(2, 10))]
    frame = pl.DataFrame(
        {
            "site": "A",
            "month": months,
            "time": [datetime(int(m[:4]), int(m[5:]), 15) for m in months],
        }
    )

    assert search_fold_offsets(frame=frame, first_months=()) == []


def test_the_search_returns_read_only_mappings():
    found = search_fold_offsets(frame=_two_sites(), first_months=ERA_START)

    assert found
    with pytest.raises(TypeError):
        found[0][1] = 3  # ty: ignore[invalid-assignment]


def test_the_study_era_constants_are_read_only_and_consistent():
    assert ENS_HRES_WIND_ERA_START_MONTHS == ("2025-10", "2026-02")
    assert dict(ENS_HRES_WIND_ERA_FOLD_OFFSETS) == {0: 0, 1: 0, 2: 2}
    assert set(ENS_HRES_WIND_ERA_FOLD_OFFSETS) == set(
        range(len(ENS_HRES_WIND_ERA_START_MONTHS) + 1)
    )
    with pytest.raises(TypeError):
        ENS_HRES_WIND_ERA_FOLD_OFFSETS[0] = 1  # ty: ignore[invalid-assignment]


def test_a_covered_design_does_not_raise():
    cut = cut_eras(frame=_two_sites(), first_months=ERA_START, fold_offsets=ROTATED)

    raise_on_uncovered_months(coverage=calendar_month_coverage(frame=cut))
