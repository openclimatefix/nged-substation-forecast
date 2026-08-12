"""Every Dagster asset declares whether the live service needs it, via the ``layer`` tag.

The two selections must partition the graph — an asset with no layer, or with both, leaves a hole
in exactly the filter the tag exists for. Compared against ``executable_asset_keys``: a typo in a
``deps=[...]`` string silently creates an external key that can never carry a tag, and
``test_definitions_resolve`` is where a broken dep belongs.
"""

from dagster import AssetKey, AssetSelection

from nged_substation_forecast.defs._tags import LAYER_TAG_KEY


def test_every_asset_carries_exactly_one_layer() -> None:
    """The production layer is pinned exactly; research is whatever is left, and nothing is both.

    Naming the production four rather than spot-checking them is what makes a mis-tag fail: an
    asset flipped to the wrong layer still carries exactly one tag, so a partition assertion alone
    cannot see it. This list is also the list
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/overview/> gives the
    reader, so adding a production asset should mean editing both.

    The two judgement calls: ``h3_grid_weights`` is production despite having no schedule, because
    the deployment materialises it by hand once and ``ecmwf_ens`` cannot run without it.
    ``promoted_model`` and ``promotable_model_runs`` are research despite changing what the service
    serves, because both need MLflow: the champion reaches production baked into the Docker image.
    """
    from nged_substation_forecast.definitions import defs

    asset_graph = defs.get_repository_def().asset_graph
    production = AssetSelection.tag(key=LAYER_TAG_KEY, value="production").resolve(asset_graph)
    research = AssetSelection.tag(key=LAYER_TAG_KEY, value="research").resolve(asset_graph)

    assert production == {
        AssetKey("power_time_series_and_metadata"),
        AssetKey("h3_grid_weights"),
        AssetKey("ecmwf_ens"),
        AssetKey("live_forecasts"),
    }
    assert research == asset_graph.executable_asset_keys - production
    assert not production & research


def test_the_documented_selection_strings_still_name_this_tag() -> None:
    """Catches ``LAYER_TAG_KEY`` being renamed while the docs keep saying ``tag:layer=…``.

    Every other assertion here resolves through the constant, so a rename leaves them all green.
    This only guards that direction: nothing reads the docs, so editing the strings *there* to
    something else still passes.
    """
    from nged_substation_forecast.definitions import defs

    asset_graph = defs.get_repository_def().asset_graph

    for value in ("production", "research"):
        from_string = AssetSelection.from_string(f"tag:layer={value}").resolve(asset_graph)

        assert from_string == AssetSelection.tag(key=LAYER_TAG_KEY, value=value).resolve(
            asset_graph
        )
        assert from_string  # Both sides are empty if nothing is tagged, asserting nothing.
