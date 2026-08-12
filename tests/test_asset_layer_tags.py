"""Every Dagster asset declares whether the live service needs it, via the ``layer`` tag.

The two selections must partition the graph — an asset with no layer, or with both, leaves a hole
in exactly the filter the tag exists for. Compared against ``executable_asset_keys``: a typo in a
``deps=[...]`` string silently creates an external key that can never carry a tag, and
``test_definitions_resolve`` is where a broken dep belongs.
"""

from dagster import AssetKey, AssetSelection

from nged_substation_forecast.defs._tags import LAYER_TAG_KEY


def test_every_asset_carries_exactly_one_layer() -> None:
    """The judgement calls are pinned by name, because the partition assertion is value-agnostic.

    ``h3_grid_weights`` is production despite having no schedule — the deployment materialises it
    by hand once, because ``ecmwf_ens`` cannot run without it. ``promoted_model`` and
    ``promotable_model_runs`` are research despite changing what the service serves, because both
    need MLflow: the champion reaches production baked into the Docker image.
    """
    from nged_substation_forecast.definitions import defs

    asset_graph = defs.get_repository_def().asset_graph
    production = AssetSelection.tag(key=LAYER_TAG_KEY, value="production").resolve(asset_graph)
    research = AssetSelection.tag(key=LAYER_TAG_KEY, value="research").resolve(asset_graph)

    assert production | research == asset_graph.executable_asset_keys
    assert not production & research
    assert {AssetKey("h3_grid_weights"), AssetKey("live_forecasts")} <= production
    assert {AssetKey("promoted_model"), AssetKey("promotable_model_runs")} <= research


def test_the_documented_selection_string_selects_the_production_layer() -> None:
    """``tag:layer=production`` is what three docs pages tell a reader to type; keep them agreeing.

    Renaming ``LAYER_TAG_KEY`` without touching those pages leaves every other assertion here
    green, because they all resolve through the constant.
    """
    from nged_substation_forecast.definitions import defs

    asset_graph = defs.get_repository_def().asset_graph

    selected = AssetSelection.from_string("tag:layer=production").resolve(asset_graph)

    assert selected == AssetSelection.tag(key=LAYER_TAG_KEY, value="production").resolve(
        asset_graph
    )
    assert selected  # Both sides are empty if nothing is tagged, which would assert nothing.
