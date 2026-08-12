"""Every Dagster asset declares which machine runs it, via the ``layer`` tag.

The point of the tag is that an operator can filter the Dagster UI down to what the AWS box runs,
so what matters is that the two selections partition the graph — an asset with no layer, or with
both, would leave a hole in exactly that filter.
"""

from collections.abc import Set as AbstractSet
from typing import NamedTuple

from dagster import AssetKey, AssetSelection

from nged_substation_forecast.defs._tags import LAYER_TAG_KEY


class _Layers(NamedTuple):
    """The two layer selections, resolved against the real asset graph, and every asset key.

    ``executable`` rather than every key in the graph: a typo in a ``deps=[...]`` string silently
    creates an external asset key, which can never carry a tag. ``test_definitions_resolve`` is
    where a broken dep belongs, so counting one here would blame the wrong thing.
    """

    production: AbstractSet[AssetKey]
    research: AbstractSet[AssetKey]
    executable: AbstractSet[AssetKey]


def _resolve_layers() -> _Layers:
    """Resolve both layer selections against the repository's real asset graph."""
    from nged_substation_forecast.definitions import defs

    asset_graph = defs.get_repository_def().asset_graph
    return _Layers(
        production=AssetSelection.tag(key=LAYER_TAG_KEY, value="production").resolve(asset_graph),
        research=AssetSelection.tag(key=LAYER_TAG_KEY, value="research").resolve(asset_graph),
        executable=asset_graph.executable_asset_keys,
    )


def test_the_two_layers_partition_the_asset_graph() -> None:
    """Every executable asset carries exactly one layer, so neither filter can miss one."""
    layers = _resolve_layers()

    assert layers.production | layers.research == layers.executable
    assert not layers.production & layers.research


def test_assets_are_classified_by_the_machine_that_runs_them() -> None:
    """The AWS box runs the forecast chain; everything else only ever runs on a laptop.

    Names the classifications that are a judgement call rather than the obvious ones, because the
    partition assertion above is value-agnostic: it demands that every asset carry a layer, not
    which one. ``h3_grid_weights`` is production despite having no schedule — the box materialises
    it by hand once, because ``ecmwf_ens`` cannot run without it. ``promoted_model`` and
    ``promotable_model_runs`` are research despite changing what production serves, because both
    need MLflow and the box has none: the model reaches it baked into the Docker image.
    """
    layers = _resolve_layers()

    assert {AssetKey(key) for key in ("h3_grid_weights", "live_forecasts")} <= layers.production
    assert {AssetKey(key) for key in ("promoted_model", "promotable_model_runs")} <= layers.research
    assert AssetKey("effective_capacity") in layers.research


def test_the_operator_facing_selection_string_selects_the_production_layer() -> None:
    """The string the docs tell an operator to paste resolves to the production layer.

    Unquoted on purpose. ``tag:"layer"="production"`` parses to the same selection normally, but
    under this repo's ``filterwarnings = ["error"]`` a ``BetaWarning`` inside Dagster's ANTLR parse
    path drops it into a fallback parser that keeps the quotes, silently selecting nothing. Do not
    "tidy" the quotes back in.

    This exercises that fallback rather than the ANTLR parser the UI and CLI use, since the same
    ``filterwarnings`` setting applies to every ``tag:`` string parsed in the suite.
    """
    from nged_substation_forecast.definitions import defs

    asset_graph = defs.get_repository_def().asset_graph

    selected = AssetSelection.from_string("tag:layer=production").resolve(asset_graph)

    assert selected == _resolve_layers().production
    # Both sides are empty if no asset is tagged, which would pass without asserting anything.
    assert selected
