"""Diagnostic (post hoc): why the registered edge test rejects true engine spells."""

import numpy as np
from e00199_common import G2_FIRST, g2_output, stamp
from e00199_inject import build
from e00199_pipeline import load_series
from e00199_scoring import flagged
from e00199_v2 import detect, edge_is_sharp, load_frozen_fit, spells


def main() -> None:
    """Print pass rates of the edge test at flagged and at true boundaries."""
    noise, params = load_frozen_fit()
    s = load_series(label="D8")
    g2 = g2_output(n=len(s.y))
    inj = build(y=s.y, g2=g2)
    post, x = detect(s=s.with_reading(inj.y), noise=noise, params=params, night=None)
    flag = flagged(smoothed=post.smoothed)
    truth = inj.truth_units
    sc = inj.scored
    rows = {"flagged start": [], "flagged end": [], "true start": [], "true end": []}
    offsets = {k: [] for k in range(-2, 3)}
    for a, b in spells(flag=flag):
        if not sc[a : b + 1].all() or truth[a : b + 1].max() == 0:
            continue
        # xs is x on the injected reading
        rows["flagged start"].append(edge_is_sharp(x=x, boundary=a - 1, unit=params.g))
        rows["flagged end"].append(edge_is_sharp(x=x, boundary=b, unit=params.g))
        for shift, passed in offsets.items():
            passed.append(edge_is_sharp(x=x, boundary=a - 1 + shift, unit=params.g))
    run = (truth > 0) & sc
    edges = np.flatnonzero(np.diff(np.concatenate([[0], run.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        rows["true start"].append(edge_is_sharp(x=x, boundary=int(a) - 1, unit=params.g))
        rows["true end"].append(edge_is_sharp(x=x, boundary=int(b) - 1, unit=params.g))
    for name, passed in rows.items():
        print(f"{name:14s} n={len(passed):4d} pass={np.mean(passed):.2f}")
    print(
        "flagged start, boundary shifted by -2..2:",
        {k: round(float(np.mean(v)), 2) for k, v in offsets.items()},
    )
    _ = G2_FIRST, stamp


if __name__ == "__main__":
    main()
