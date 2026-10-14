"""Check experiment 00018's availability rules on synthetic inputs; exits non-zero on a breach.

For origins at 09:00 UTC it asserts, for every arm: no input reads a half-hour after the newest
usable one (origin minus the delay); ERA5T is used only at least 5 days before the origin; CAMS
only up to 00:00 UTC of the origin day; and a lag is null exactly when its time is after the
newest usable half-hour.

    uv run python \
        ml-experiments/00018-staged-chain-vs-n1-serving-latency-lag-anomalies-demand/e00018_check.py
"""

from datetime import UTC, datetime

import e00018_common as common
import numpy as np


def main() -> None:
    """Run the checks."""
    size = int(common.grid_index(times=datetime(2026, 8, 1, tzinfo=UTC)))
    rng = np.random.default_rng(0)
    power = rng.random(size).astype(np.float32)
    marker = {name: np.full(size, float(i)) for i, name in enumerate(common.SOURCES)}
    for day in (60, 400, 700, 800):
        origin = int(common.grid_index(times=datetime(2024, 4, 1, 9, tzinfo=UTC))) + day * 48
        assert origin % 48 == 18
        for name, arm in common.ARMS.items():
            if arm.source is None:
                continue
            residual = common.origin_residual(
                power=power,
                predictions={k: v.astype(np.float32) * 0 for k, v in marker.items()},
                source=arm.source,
                origin=origin,
                delay_steps=arm.delay_steps,
            )
            finite = np.flatnonzero(np.isfinite(residual))
            assert finite.max() <= origin - arm.delay_steps, name
            window = np.arange(origin - common.MAX_LOOKBACK_STEPS, origin - arm.delay_steps + 1)
            codes = common.source_codes(source=arm.source, origin=origin, window=window)
            era5, proxy, cams = (codes == common.SOURCES.index(s) for s in ("E", "P", "CP"))
            if arm.source in {"S", "SN"}:
                assert (window[era5] <= origin - common.ERA5T_STEPS).all(), name
                assert (window[proxy | cams] > origin - common.ERA5T_STEPS).all(), name
            assert (window[cams] <= origin - origin % 48).all(), name
            if arm.source != "S":
                assert not cams.any(), name
        target = origin + np.arange(1, 703)
        recent = common.recent_inputs(
            power=power,
            residual=None,
            target=target,
            origin=np.full(target.size, origin),
            delay_steps=common.DELAY_STEPS,
        )
        newest = origin - common.DELAY_STEPS
        for column, lag in zip(recent[:, 5:9].T, common.LAG_STEPS, strict=True):
            assert np.isnan(column[target - lag > newest]).all()
            assert np.isfinite(column[target - lag <= newest]).all()
        assert recent[0, 0] == power[newest]
    print("availability checks passed")


if __name__ == "__main__":
    main()
