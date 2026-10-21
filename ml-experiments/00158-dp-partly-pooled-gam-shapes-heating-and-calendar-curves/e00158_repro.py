"""Check that experiment 158's per-site fits (`k0`) reproduce experiment 111's `gamk` fits.

Prints the largest absolute coefficient difference over every cut and substation, and the number of
cuts and substations compared.

    uv run python ml-experiments/00158-*/e00158_repro.py
"""

import pickle

import e00158_common as common
import e00158_fits as fits
import numpy as np


def main() -> None:
    """Compare the stored fits of experiment 158's `k0` with experiment 111's `gamk`."""
    worst = 0.0
    compared = 0
    for site in common.FLEET:
        mine = fits.load_fits(config="k0", history="full", site=site)
        stored = pickle.loads((common.FITS_DIR / "gamk" / f"{site}.pkl").read_bytes())
        if [f.cut for f in mine] != [f.cut for f in stored]:
            msg = f"{site}: the cuts differ"
            raise ValueError(msg)
        for a, b in zip(mine, stored, strict=True):
            worst = max(worst, float(np.abs(np.asarray(a.theta) - np.asarray(b.theta)).max()))
            compared += 1
    print(
        f"{compared} fits ({len(common.FLEET)} substations); "
        f"largest coefficient difference {worst:.2e}"
    )


if __name__ == "__main__":
    main()
