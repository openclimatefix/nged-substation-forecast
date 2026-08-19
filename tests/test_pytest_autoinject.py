"""End-to-end regression test for `_pytest_autoinject.py`.

Runs real pytest subprocesses against a throwaway project, because the bug this guards against —
`-n auto` silently never being injected — is invisible from inside a single process: pytest still
collects and runs every test either way, just serially instead of under xdist. Only a subprocess
run's own stdout (the `created: N/N workers` banner, present only under xdist) can tell the two
apart. `runpytest_subprocess` re-uses the current interpreter, so `pytest-xdist` is available.
"""

import shutil
from pathlib import Path

import pytest


def _install_autoinject_plugin(pytester: pytest.Pytester) -> None:
    shutil.copy(Path(__file__).parent / "_pytest_autoinject.py", pytester.path)
    pytester.makeini("[pytest]\naddopts = -p _pytest_autoinject\n")
    pytester.makepyfile(test_dummy="def test_one(): pass\ndef test_two(): pass\n")


def test_bare_invocation_runs_under_xdist(pytester: pytest.Pytester) -> None:
    _install_autoinject_plugin(pytester)
    result = pytester.runpytest_subprocess()
    result.stdout.fnmatch_lines(["created: */* workers"])


def test_a_node_id_target_stays_serial(pytester: pytest.Pytester) -> None:
    _install_autoinject_plugin(pytester)
    result = pytester.runpytest_subprocess("test_dummy.py::test_one")
    assert "workers" not in result.stdout.str()


def test_explicit_numprocesses_is_respected(pytester: pytest.Pytester) -> None:
    _install_autoinject_plugin(pytester)
    result = pytester.runpytest_subprocess("-n", "2")
    result.stdout.fnmatch_lines(["created: 2/2 workers"])


def test_a_value_taking_flag_does_not_suppress_injection(pytester: pytest.Pytester) -> None:
    _install_autoinject_plugin(pytester)
    result = pytester.runpytest_subprocess("-k", "test_one")
    result.stdout.fnmatch_lines(["created: */* workers"])


def test_no_xdist_falls_back_to_serial_without_erroring(pytester: pytest.Pytester) -> None:
    _install_autoinject_plugin(pytester)
    result = pytester.runpytest_subprocess("-p", "no:xdist")
    result.assert_outcomes(passed=2)
    assert "workers" not in result.stdout.str()
