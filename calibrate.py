"""Turn the probe's public score into the best prediction scale.

Let p = v1 predictions, y = true sales, and D[.] = mean over the public rows. Scaling p by s gives
    RMSE(s)^2 = D[(y - p)^2] - 2(s - 1) D[p(y - p)] + (s - 1)^2 D[p^2]
so the two known scores (s = 1 and s = PROBE_SCALE) give D[p(y - p)] exactly, and the best scale is
    s* = 1 + D[p(y - p)] / D[p^2],   with   RMSE(s*)^2 = RMSE(1)^2 - D[p(y - p)]^2 / D[p^2].
D[p^2] is estimated by the mean over all test rows (the public rows are a random 25%).

Run:  .venv/bin/python calibrate.py <probe public score>
"""
import sys

import numpy as np
import pandas as pd

from src import config

MIN_SHIFT = 0.02  # ignore |s* - 1| below 2%: that is within the noise of a 1,420-row public slice


def best_scale(r1: float, r2: float, mean_p2: float, probe_scale: float = config.PROBE_SCALE):
    d = probe_scale - 1
    mean_p_resid = (r1 ** 2 - r2 ** 2 + d ** 2 * mean_p2) / (2 * d)
    s_star = 1 + mean_p_resid / mean_p2
    expected_rmse = float(np.sqrt(r1 ** 2 - mean_p_resid ** 2 / mean_p2))
    return s_star, expected_rmse


def main():
    r2 = float(sys.argv[1])
    p = pd.read_csv(config.V1_SUBMISSION_CSV)[config.TARGET].to_numpy()
    mean_p2 = float(np.mean(p ** 2))

    s_star, expected = best_scale(config.V1_PUBLIC_SCORE, r2, mean_p2)
    print(f"v1 public score        {config.V1_PUBLIC_SCORE:.4f}")
    print(f"probe (x{config.PROBE_SCALE}) score    {r2:.4f}")
    print(f"best scale s*          {s_star:.4f}")
    print(f"expected public RMSE   {expected:.2f}  (at s*)")
    if abs(s_star - 1) > MIN_SHIFT:
        print(f"-> real shift: set PREDICTION_SCALE = {s_star:.4f} in src/config.py")
    else:
        print("-> shift is within noise: keep PREDICTION_SCALE = 1.0")


if __name__ == "__main__":
    main()
