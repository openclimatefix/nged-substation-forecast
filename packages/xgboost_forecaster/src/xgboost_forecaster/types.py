from enum import Enum


class EnsembleSelection(str, Enum):
    """Selection method for weather ensemble members."""

    MEAN = "mean"
    SINGLE = "single"
    ALL = "all"
