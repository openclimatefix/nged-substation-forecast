"""Forecast the running arm (experiment 136's A2 chain) at the given sites.

E00214_ARM=c uv run --no-sync python ml-experiments/00180-*/e00214_forecast.py --sites D5
"""

import e00214_run as run

if __name__ == "__main__":
    run.main()
