"""Shared fixtures for the root integration tests.

The repo-root ``conftest.py`` (one level up) owns the ``--run-network`` gate; this one scopes
its fixtures to the ``tests/`` directory only, so nothing here touches the ``packages/*/tests``
unit suites.

Shared helpers live here as *fixtures*, not as plain functions a test module imports by name.
A test module can import ``tests/_nwp_test_data.py`` (a normal, uniquely-named module) by bare
name via the ``pythonpath = ["tests"]`` pytest setting, but ``conftest`` is not a unique name —
this repo also has a root-level ``conftest.py`` — so ``from conftest import ...`` is ambiguous
outside pytest's own module loading: ``ty`` resolves it to the *root* ``conftest.py`` and reports
an unresolved import. Fixture injection sidesteps the ambiguity entirely, since a test module
never has to import ``conftest`` by name to use one.
"""

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from dagster import DagsterInstance, RunConfig

from nged_substation_forecast.defs.jobs import RegisterExperimentConfig, register_experiment_job


@pytest.fixture
def dagster_instance() -> Iterator[DagsterInstance]:
    """A fresh ephemeral ``DagsterInstance``, disposed when the test finishes.

    Every test here should take this fixture rather than calling ``DagsterInstance.ephemeral()``
    itself; outside the test suite the equivalent is to use the instance as a context manager. An
    ephemeral
    instance's in-memory run storage and event-log storage each open one SQLAlchemy connection
    against an in-memory SQLite database and hold it for the life of the instance. Nothing closes
    those two connections unless ``dispose()`` is called, and ``DagsterInstance`` has no
    finaliser — so an instance handed to the garbage collector can survive to interpreter
    shutdown, where SQLAlchemy's connection-pool finaliser may run *after* SQLite has closed the
    underlying database. That prints a bare ``Exception during reset or similar`` traceback ending
    in ``sqlite3.ProgrammingError: Cannot operate on a closed database`` — two of them, one per
    held connection — after pytest's own summary line, where no test owns it.

    Entering the instance as a context manager makes ``__exit__`` call ``dispose()``, which closes
    both connections and removes the instance's temporary artifact directory, at the end of the
    test that created it.
    """
    with DagsterInstance.ephemeral() as instance:
        yield instance


@pytest.fixture
def register_experiment() -> Callable[[DagsterInstance, str], None]:
    """Return a callable that registers a full-CV experiment for its leaderboard fold.

    Every CV integration test in this suite (``trained_cv_model``, ``cv_power_forecasts``,
    ``metrics``) registers ``run_mode="full_cv"`` rather than ``"smoke_test"``, because each one
    drives the leaderboard fold's (``FOLD_ID``) window and synthetic data specifically. A fixture
    that returns a callable, rather than a plain module-level function, so every caller reaches
    it the same way — pytest injection — with no ``from conftest import ...`` (see the module
    docstring for why that import is ambiguous).
    """

    def _register(instance: DagsterInstance, experiment_name: str) -> None:
        result = register_experiment_job.execute_in_process(
            run_config=RunConfig(
                ops={
                    "register_experiment": RegisterExperimentConfig(
                        experiment_name=experiment_name,
                        base_model_config="conf/model/xgboost.yaml",
                        config_overrides={
                            "selected_features": ["temperature_2m"],
                            "n_estimators": 5,
                        },
                        run_mode="full_cv",
                    )
                }
            ),
            instance=instance,
        )
        assert result.success

    return _register


@pytest.fixture(autouse=True)
def _fail_on_an_undisposed_dagster_instance(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail the test that leaves a ``DagsterInstance.ephemeral()`` instance for the collector.

    Being autouse, this fixture is set up before the ones a test asks for by name and so torn down
    after them, which is what lets it see ``dagster_instance`` dispose.

    It catches an instance that *survives* the test, not a call that forgot to own one: CPython
    frees an unreferenced context as soon as the invoking helper returns, so a missing
    ``instance=`` still passes green until something — a captured traceback, most often — pins the
    frame holding it. Instances made by any route other than ``ephemeral()``, such as
    ``local_temp()``, are invisible to it.
    """
    undisposed: set[DagsterInstance] = set()
    make_ephemeral = DagsterInstance.ephemeral
    dispose = DagsterInstance.dispose

    def _tracked_ephemeral(*args: Any, **kwargs: Any) -> DagsterInstance:
        instance = make_ephemeral(*args, **kwargs)
        undisposed.add(instance)
        return instance

    def _tracked_dispose(self: DagsterInstance) -> None:
        undisposed.discard(self)
        dispose(self)

    monkeypatch.setattr(DagsterInstance, "ephemeral", _tracked_ephemeral)
    monkeypatch.setattr(DagsterInstance, "dispose", _tracked_dispose)

    yield

    assert not undisposed, (
        f"{len(undisposed)} ephemeral DagsterInstance(s) outlived this test. Take the "
        "`dagster_instance` fixture, or enter the instance as a context manager."
    )
