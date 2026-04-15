def test_static_registry_matches_patito_contract():
    # Get statically defined features from Patito
    # (Exclude base columns like timestamp/power and parameterized ones like lags)
    patito_static_features = {...}

    registry_features = set(STATIC_FEATURE_REGISTRY.keys())

    missing_in_registry = patito_static_features - registry_features
    assert not missing_in_registry, (
        f"You added {missing_in_registry} to the Patito AllFeatures model, "
        f"but forgot to write the Polars logic in STATIC_FEATURE_REGISTRY."
    )
