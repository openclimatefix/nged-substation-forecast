"""Wiring check for experiment 190: on a noiseless stream the gated filters follow an offset.

A stream of zeros has a 7.8-point offset added for 36 hours (the size of 0.2 of a typical median
load, over 3 days of clean readings). With no noise, every variant must follow the offset with the
level state, and the shared-level variant (A3) must reach the bands whose own windows have not
yet been read. The check prints, for each variant and each band, the correction (level plus
transient) 24 hours after the offset starts, as seen by a forecast with the 3.5-hour delay.

    uv run --no-sync python ml-experiments/00190-*/e00190_wiring_check.py
"""

import e00190_base as base
import e00190_gated as gated
import numpy as np
from studies.dp_assimilation import correction_at

OFFSET = 7.8
N_DAYS = 6
HOURLY_BANDS = np.repeat(np.arange(4), 12)


def main() -> None:
    """Print the correction by band for each variant."""
    band = np.tile(HOURLY_BANDS, N_DAYS * 2 // 1)[: 48 * N_DAYS]
    stream = np.zeros(48 * N_DAYS)
    start = 48 * 2 + 20
    stream[start : start + 72] = OFFSET
    probe = start + 48
    cutoff = np.full(4, probe - 7)
    target = np.arange(probe, probe + 4 * 12, 12)
    for name, gate in {
        "a0": gated.Gate(mode="none"),
        "a1": gated.Gate(mode="gated", m=5.0, k=8, z=0.5),
        "a2": gated.Gate(mode="always", m=5.0),
        "a3": gated.Gate(mode="shared", m=5.0, k=8, z=0.5),
    }.items():
        run = gated.gated_stream(
            residual=stream, band=band, settings=base.pool_settings(), gate=gate
        )
        correction = correction_at(
            trace=run.stream.trace, cutoff=cutoff, target=target, band=band[target].astype(np.int64)
        )
        print(name, np.round(correction.total, 2), "offset", OFFSET, flush=True)


if __name__ == "__main__":
    main()
