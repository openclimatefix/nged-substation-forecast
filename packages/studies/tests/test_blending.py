from typing import TypedDict

import numpy as np
import polars as pl
import pytest
from studies.blending import StackedErrors, climatology_permutation, simplex_weights, stacked_errors

N_FOLDS = 5
ROWS_PER_FOLD = 400


def _weather(*, n_rows: int = 600) -> pl.DataFrame:
    generator = np.random.default_rng(11)
    direction = generator.uniform(0.0, 2.0 * np.pi, n_rows)
    return pl.DataFrame(
        {
            "site": generator.choice(["A", "B"], n_rows),
            "hour": generator.integers(0, 3, n_rows),
            "ghi": generator.normal(size=n_rows),
            "sin": np.sin(direction),
            "cos": np.cos(direction),
            "untouched": np.arange(n_rows, dtype=np.float64),
        }
    )


def test_the_permutation_keeps_each_groups_values_and_changes_their_order():
    frame = _weather()

    permuted = climatology_permutation(
        frame=frame, column_groups=[("ghi",)], by=("site", "hour"), seed=1
    )

    for _, group in permuted.group_by("site", "hour"):
        assert sorted(group["ghi_shuffled"]) == sorted(group["ghi"])
    assert (permuted["ghi_shuffled"] != permuted["ghi"]).any()
    assert permuted.drop("ghi_shuffled").equals(frame)
    assert "untouched_shuffled" not in permuted.columns


def test_a_single_column_is_permuted_as_polars_shuffle_permutes_it():
    frame = _weather()

    permuted = climatology_permutation(
        frame=frame, column_groups=[("ghi",)], by=("site", "hour"), seed=5
    )

    expected = frame.select(pl.col("ghi").shuffle(seed=5).over("site", "hour"))["ghi"]
    assert permuted["ghi_shuffled"].equals(expected)


def test_a_products_columns_move_together():
    permuted = climatology_permutation(
        frame=_weather(), column_groups=[("sin", "cos")], by=("site", "hour"), seed=2
    )

    radius = permuted["sin_shuffled"] ** 2 + permuted["cos_shuffled"] ** 2
    assert np.allclose(radius.to_numpy(), 1.0)
    assert (permuted["sin_shuffled"] != permuted["sin"]).any()


def test_two_products_get_different_permutations():
    frame = _weather().with_columns(ghi_copy=pl.col("ghi"))

    permuted = climatology_permutation(
        frame=frame, column_groups=[("ghi",), ("ghi_copy",)], by=("site", "hour"), seed=3
    )

    assert (permuted["ghi_shuffled"] != permuted["ghi_copy_shuffled"]).any()


class StackInputs(TypedDict):
    errors: np.ndarray
    sites: np.ndarray
    seeds: np.ndarray
    folds: np.ndarray
    fit_rows: np.ndarray


def _stack_inputs(
    *, variances: tuple[float, ...], n_sites: int = 1, n_seeds: int = 1, seed: int = 0
) -> StackInputs:
    generator = np.random.default_rng(seed)
    n_rows = N_FOLDS * ROWS_PER_FOLD * n_sites * n_seeds
    return {
        "errors": np.column_stack(
            [generator.normal(scale=np.sqrt(variance), size=n_rows) for variance in variances]
        ),
        "sites": np.repeat(np.arange(n_sites), N_FOLDS * ROWS_PER_FOLD * n_seeds),
        "seeds": np.tile(np.repeat(np.arange(n_seeds), N_FOLDS * ROWS_PER_FOLD), n_sites),
        "folds": np.tile(np.repeat(np.arange(N_FOLDS), ROWS_PER_FOLD), n_sites * n_seeds),
        "fit_rows": np.ones(n_rows, dtype=bool),
    }


def _stack(
    inputs: StackInputs, *, errors: np.ndarray | None = None, cross_fitted: bool = True
) -> StackedErrors:
    return stacked_errors(
        errors=inputs["errors"] if errors is None else errors,
        sites=inputs["sites"],
        folds=inputs["folds"],
        seeds=inputs["seeds"],
        fit_rows=inputs["fit_rows"],
        cross_fitted=cross_fitted,
    )


def _corrupted(inputs: StackInputs, *, rows: np.ndarray) -> np.ndarray:
    errors = inputs["errors"].copy()
    errors[rows, 0] *= 50.0
    return errors


def _rows(inputs: StackInputs, **where: int) -> np.ndarray:
    mask = np.ones(len(inputs["folds"]), dtype=bool)
    for key, value in where.items():
        mask &= {"sites": inputs["sites"], "seeds": inputs["seeds"], "folds": inputs["folds"]}[
            key
        ] == value
    return mask


def test_the_stack_recovers_inverse_variance_weights():
    variance_a, variance_b = 1.0, 4.0
    errors = _stack_inputs(variances=(variance_a, variance_b), seed=1)["errors"]
    errors = np.vstack([errors] * 20)

    weights = simplex_weights(errors=errors)

    total = variance_a + variance_b
    assert weights == pytest.approx([variance_b / total, variance_a / total], abs=0.01)


def test_a_useless_model_gets_no_weight():
    generator = np.random.default_rng(2)
    good = generator.normal(size=5000)
    errors = np.column_stack([good, good + 3.0])

    assert simplex_weights(errors=errors) == pytest.approx([1.0, 0.0], abs=1e-9)


def test_the_weights_are_non_negative_and_sum_to_exactly_one():
    inputs = _stack_inputs(variances=(1.0, 2.0, 0.5))

    result = _stack(inputs)

    assert (result.weights >= 0.0).all()
    assert np.abs(result.weights.sum(axis=1) - 1.0).max() < 1e-12
    assert result.errors == pytest.approx((result.weights * inputs["errors"]).sum(axis=1))


def test_the_stack_never_sees_the_fold_it_scores():
    inputs = _stack_inputs(variances=(1.0, 2.0))
    scored = _rows(inputs, folds=2)

    clean = _stack(inputs).weights
    dirty = _stack(inputs, errors=_corrupted(inputs, rows=scored)).weights

    assert np.array_equal(clean[scored], dirty[scored])
    for fold in (0, 1, 3, 4):
        assert not np.allclose(clean[_rows(inputs, folds=fold)], dirty[_rows(inputs, folds=fold)])


def test_the_in_sample_stack_sees_the_fold_it_scores():
    inputs = _stack_inputs(variances=(1.0, 2.0))
    scored = _rows(inputs, folds=2)

    clean = _stack(inputs, cross_fitted=False).weights
    dirty = _stack(inputs, errors=_corrupted(inputs, rows=scored), cross_fitted=False).weights

    assert not np.allclose(clean[scored], dirty[scored])


def test_one_generators_errors_do_not_change_another_generators_weights():
    inputs = _stack_inputs(variances=(1.0, 2.0), n_sites=2)

    clean = _stack(inputs).weights
    dirty = _stack(inputs, errors=_corrupted(inputs, rows=_rows(inputs, sites=1))).weights

    assert np.array_equal(clean[_rows(inputs, sites=0)], dirty[_rows(inputs, sites=0)])
    assert not np.allclose(clean[_rows(inputs, sites=1)], dirty[_rows(inputs, sites=1)])


def test_each_seed_is_stacked_from_its_own_errors_only():
    inputs = _stack_inputs(variances=(1.0, 2.0), n_seeds=2)

    clean = _stack(inputs).weights
    dirty = _stack(inputs, errors=_corrupted(inputs, rows=_rows(inputs, seeds=1))).weights

    assert np.array_equal(clean[_rows(inputs, seeds=0)], dirty[_rows(inputs, seeds=0)])
    assert not np.allclose(clean[_rows(inputs, seeds=1)], dirty[_rows(inputs, seeds=1)])


def test_constrained_rows_do_not_reach_the_fit_and_are_still_scored():
    inputs = _stack_inputs(variances=(1.0, 2.0))
    constrained = np.zeros(len(inputs["folds"]), dtype=bool)
    constrained[::7] = True
    inputs["fit_rows"] = ~constrained
    corrupted = _corrupted(inputs, rows=constrained)

    clean = _stack(inputs)
    dirty = _stack(inputs, errors=corrupted)

    assert np.array_equal(clean.weights, dirty.weights)
    assert dirty.errors[constrained] == pytest.approx(
        (dirty.weights * corrupted).sum(axis=1)[constrained]
    )
    assert not np.allclose(dirty.errors[constrained], clean.errors[constrained])


def test_a_fold_with_nothing_to_fit_on_raises():
    inputs = _stack_inputs(variances=(1.0, 2.0))
    inputs["fit_rows"] = _rows(inputs, folds=0)

    with pytest.raises(ValueError, match="no rows to fit"):
        _stack(inputs)
