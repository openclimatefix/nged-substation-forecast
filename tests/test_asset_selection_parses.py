"""Guards the ANTLR runtime version that Dagster's asset-selection parser needs.

Dagster's parser is *generated* by ANTLR 4.13.2, and ANTLR changed its serialised-ATN encoding at
4.10 — the 4.9 runtime reads the ATN as characters, the 4.13 one as integers — so resolving a 4.9
runtime breaks every asset-selection string with ``TypeError: ord() expected string of length 1,
but int found``. Dagster itself only requires ``<4.14``, so the lower bound is ours to hold: the
``constraint-dependencies`` floor in ``pyproject.toml`` is what enforces it, and this test is the
backstop that says what a violation actually costs.

The failure it guards against is not subtle but it is easy to not notice, because nothing in the
test suite or the Dagster UI parses a selection string — only the CLI does (``dg launch --assets``,
``dagster asset materialize --select``).
"""

from dagster._core.definitions.asset_selection import AssetSelection


def test_an_asset_selection_string_parses() -> None:
    assert AssetSelection.from_string('key:"ecmwf_ens"') is not None
