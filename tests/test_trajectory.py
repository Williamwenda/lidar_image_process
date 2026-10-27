"""
Check the batched trajectory sampling (odometry_icp.sample_trajectory) against pysteam's
const_vel pose / velocity interpolators, including extrapolation outside the knots.
Run:  python tests/test_trajectory.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deps_path  # noqa: F401

import numpy as np
from pylgmath import Transformation
from pysteam.evaluable.se3 import SE3StateVar
from pysteam.evaluable.vspace import VSpaceStateVar
from pysteam.trajectory import Time
from pysteam.trajectory.const_vel import Interface

from odometry_icp import sample_trajectory

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    t0 = 1_747_839_879_068_984_890  # realistic absolute stamps [ns]
    knot_times = [t0, t0 + 40_000_000, t0 + 100_000_000]

    traj = Interface(np.array([1.0, 0.1, 0.1, 0.1, 0.1, 1.0]))
    knots = []
    for t in knot_times:
        T = Transformation(xi_ab=rng.normal(size=(6, 1)) * 0.3)
        w = rng.normal(size=(6, 1))
        traj.add_knot(Time(nsecs=t), SE3StateVar(T), VSpaceStateVar(w))
        knots.append((t, T.matrix(), w.ravel()))

    times = np.concatenate([
        t0 + rng.integers(-5_000_000, 105_000_000, 300),  # before, between and after the knots
        np.array(knot_times),                              # exactly on the knots
    ])
    T_batch, w_batch = sample_trajectory(knots, times)

    T_err = w_err = 0.0
    for i, t in enumerate(times):
        T_ref = traj.get_pose_interpolator(Time(nsecs=int(t))).evaluate().matrix()
        w_ref = traj.get_velocity_interpolator(Time(nsecs=int(t))).evaluate().ravel()
        T_err = max(T_err, np.abs(T_batch[i] - T_ref).max())
        w_err = max(w_err, np.abs(w_batch[i] - w_ref).max())
    print(f"max |T - T_pysteam| = {T_err:.2e},  max |w - w_pysteam| = {w_err:.2e}")
    assert T_err < 1e-9 and w_err < 1e-9
    print("trajectory sampling test passed")
