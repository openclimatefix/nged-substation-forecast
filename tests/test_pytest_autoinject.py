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

# Enables the `pytester` fixture used below, which spawns real pytest subprocesses to test
# `_pytest_autoinject.py` end to end. `pytester` is a builtin plugin that isn't loaded by default;
# scoping it to this module (rather than the repo-root conftest.py) keeps the extra CLI options and
# ini setting it registers out of the other 750-odd tests that don't need them.
pytest_plugins = ["pytester"]


def _install_autoinject_plugin(pytester: pytest.Pytester) -> None:
    shutil.copy(Path(__file__).parent / "_pytest_autoinject.py", pytester.path)
    pytester.makeini("[pytest]\naddopts = -p _pytest_autoinject\n")
    pytester.makepyfile(test_dummy="def test_one(): pass\ndef test_two(): pass\n")


def test_bare_invocation_runs_under_xdist(pytester: pytest.Pytester) -> None:
    _install_autoinject_plugin(pytester)
    result = pytester.runpytest_subprocess()
    result.stdout.fnmatch_lines(["created: */* worker*"])


def test_a_node_id_target_stays_serial(pytester: pytest.Pytester) -> None:
    _install_autoinject_plugin(pytester)
    result = pytester.runpytest_subprocess("test_dummy.py::test_one")
    result.assert_outcomes(passed=1)
    assert "workers" not in result.stdout.str()


def test_explicit_numprocesses_is_respected(pytester: pytest.Pytester) -> None:
    """`-n 1`, not `-n 2`: on a runner where physical-core auto-detection also resolves to 2 (this
    repo's CI), a broken injection that appends `-n auto` after an explicit `-n 2` instead of
    prepending it would still show `2/2 workers` by coincidence. `-n 1` can't coincide with `auto`
    on any multi-core runner, so it actually catches append-instead-of-prepend."""
    _install_autoinject_plugin(pytester)
    result = pytester.runpytest_subprocess("-n", "1")
    result.stdout.fnmatch_lines(["created: 1/1 worker*"])


def test_a_value_taking_flag_does_not_suppress_injection(pytester: pytest.Pytester) -> None:
    _install_autoinject_plugin(pytester)
    result = pytester.runpytest_subprocess("-k", "test_one")
    result.stdout.fnmatch_lines(["created: */* worker*"])


def test_no_xdist_falls_back_to_serial_without_erroring(pytester: pytest.Pytester) -> None:
    _install_autoinject_plugin(pytester)
    result = pytester.runpytest_subprocess("-p", "no:xdist")
    result.assert_outcomes(passed=2)
    assert "workers" not in result.stdout.str()


@pytest.mark.parametrize("capture_flag", ["-s", "--capture=no"])
def test_capture_no_stays_serial(pytester: pytest.Pytester, capture_flag: str) -> None:
    """Neither spelling of "capture off" may be parallelised: a worker's captured-off stdout never
    reaches the controller, so injecting `-n auto` here would silently swallow the output the flag
    exists to show. Both spellings are covered, not just `-s`, because a hand-scan for the literal
    `-s` token would also pass this test while leaving `--capture=no` parallelised."""
    _install_autoinject_plugin(pytester)
    pytester.makepyfile(test_verbose="def test_prints():\n    print('CAPTURE_NO_MARKER')\n")
    result = pytester.runpytest_subprocess(capture_flag, "-k", "test_prints")
    result.assert_outcomes(passed=1)
    assert "workers" not in result.stdout.str()
    result.stdout.fnmatch_lines(["*CAPTURE_NO_MARKER*"])


def test_plugin_registered(pytestconfig: pytest.Config) -> None:
    """Guards the wiring itself: `pyproject.toml`'s `addopts` has to actually load this plugin by
    name (`-p _pytest_autoinject`), or every invocation in this repo silently goes serial exactly
    like the bug this file otherwise guards against, with every test above still green — they build
    their own throwaway project with their own `addopts`, so none of them exercises this repo's."""
    assert pytestconfig.pluginmanager.hasplugin("_pytest_autoinject")
