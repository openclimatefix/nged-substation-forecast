"""Fit the running arm's T-all monthly fits (a wrapper, so pickled classes live in `e00180_tall`).

E00180_ARM=c uv run --no-sync python ml-experiments/00180-*/e00180_fit.py \
    --cadences monthly --sites D5
"""

import e00180_tall as tall

if __name__ == "__main__":
    tall.main()
