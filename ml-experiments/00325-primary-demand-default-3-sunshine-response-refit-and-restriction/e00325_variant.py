# ruff: noqa: E501
"""Make the T-all prediction of experiments 136 and 201 carry a sunshine-response variant.

Both experiments predict through their own `tall.predict_tall`, which the chain, the Kalman
filter's residual stream, and the past-weather rebuild all call. `install` replaces that function
with one that uses the variant's refitted coefficients (`theta` and `gamma`, looked up by Primary
and cut) and adds the response, so every later stage of those scripts runs unchanged on the
variant's forecasts. The control `v0r` installs nothing.
"""

import e00136_tall as tall136  # ty: ignore[unresolved-import]
import e00201_tall as tall201  # ty: ignore[unresolved-import]
import e00325_basis as sun
import e00325_fits as fits
import numpy as np


def make_predict(*, tall_module, variant: str):  # noqa: ANN001, ANN201
    """Return a `predict_tall` for one experiment's T-all module and a response variant."""
    table = fits.load_table(variant=variant)

    def predict_tall(*, cache, fit, index: np.ndarray) -> np.ndarray:  # noqa: ANN001
        features = cache.sites[fit.site]
        x0 = tall_module.design_matrix(
            cache=cache, features=features, layout=fit.layout, index=index
        )
        x = tall_module.design.arm_matrix(
            cache=cache, layout=fit.layout, site=fit.site, index=index, x0=x0, arm=fit.arm
        )
        s = sun.basis(
            ghi_w_m2=features.ghi_w_m2[index],
            elevation_deg=features.elevation_deg[index],
            variant=variant,
        )
        if not (np.isfinite(x).all() and np.isfinite(s).all()):
            msg = f"{fit.site} has a non-finite input among the rows to predict"
            raise ValueError(msg)
        theta, gamma = table[(fit.site, fit.cut)]
        return (
            x @ theta
            + s @ gamma
            - tall_module.pinned_solar_points(
                cache=cache, features=features, pinned=fit.pinned, index=index
            )
        )

    return predict_tall


def install(*, variant: str) -> None:
    """Patch experiments 136's and 201's `predict_tall` for a response variant (`v0r`: no change)."""
    if variant == "v0r":
        return
    for module in (tall136, tall201):
        module.predict_tall = make_predict(tall_module=module, variant=variant)
