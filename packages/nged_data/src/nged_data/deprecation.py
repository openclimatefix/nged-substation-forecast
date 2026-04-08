import warnings


def warn_deprecated():
    """Issue a deprecation warning for the nged_data package."""
    warnings.warn(
        "The 'nged_data' package is deprecated in favor of 'nged_json_data'.",
        DeprecationWarning,
        stacklevel=2,
    )
