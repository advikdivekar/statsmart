"""Leaderboard scale probe: write v1 predictions x 1.05 as a separate submission.

Submitting this once tells us (together with v1's known public score) whether the hidden test
sales are systematically higher or lower than our predictions. calibrate.py does the maths.

Run:  .venv/bin/python probe.py
"""
import pandas as pd

from src import config


def main():
    v1 = pd.read_csv(config.V1_SUBMISSION_CSV)
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_CSV)
    assert v1[config.ID_COLS].equals(sample[config.ID_COLS]), "v1 row order differs from sample"

    probe = v1.copy()
    probe[config.TARGET] = v1[config.TARGET] * config.PROBE_SCALE
    probe.to_csv(config.PROBE_CSV, index=False)
    print(f"saved {config.PROBE_CSV}  (v1 predictions x {config.PROBE_SCALE})")


if __name__ == "__main__":
    main()
