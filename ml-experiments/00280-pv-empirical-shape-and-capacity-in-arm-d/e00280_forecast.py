"""Forecast the running arm (experiment 136's A2 chain) at the given sites.

E00280_ARM=c uv run --no-sync python ml-experiments/00180-*/e00280_forecast.py --sites D5
"""

import e00280_run as run

if __name__ == "__main__":
    run.main()
