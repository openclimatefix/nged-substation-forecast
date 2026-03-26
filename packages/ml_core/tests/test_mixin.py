import patito as pt
import pytest
from ml_core.assets import FeatureAsset
from ml_core.trainer import BaseDataRequirements, DataRequirementsMixin


class MockRequirements(BaseDataRequirements):
    substation_power_flows: pt.LazyFrame


class MockWithRequirements(DataRequirementsMixin):
    requirements_class = MockRequirements


def test_data_requirements_mixin():
    # Test that it correctly extracts assets from the requirements class
    assets = MockWithRequirements.data_requirements()
    assert len(assets) == 1
    assert assets[0] == FeatureAsset.SUBSTATION_POWER_FLOWS


def test_data_requirements_mixin_missing_class():
    # Test that it raises TypeError if requirements_class is missing
    class MissingReq(DataRequirementsMixin):
        pass

    with pytest.raises(TypeError, match="must define a 'requirements_class' attribute"):
        MissingReq.data_requirements()
