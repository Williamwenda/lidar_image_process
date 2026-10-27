"""
Check the custom cost terms against pysteam's own per-point / per-factor implementation.
Run:  python tests/test_cost_terms.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import deps_path  # noqa: F401

import numpy as np
from pylgmath import Transformation
from pysteam.evaluable.p2p import P2PErrorEvaluator
from pysteam.evaluable.se3 import SE3StateVar, compose, inv
from pysteam.evaluable.se3 import se3_error
from pysteam.evaluable.vspace import VSpaceStateVar, vspace_error
from pysteam.problem import L2LossFunc, OptimizationProblem, StaticNoiseModel, WeightedLeastSquareCostTerm
from pysteam.trajectory import Time
from pysteam.trajectory.const_vel import Interface

from core.cost_terms import PointToPlaneCostTerm, StatePriorCostTerm


def make_trajectory(rng):
    traj = Interface(np.array([1.0, 0.1, 0.1, 0.1, 0.1, 1.0]))
    state_vars = []
    for k, t_ns in enumerate([0, 50_000_000, 100_000_000]):
        T_var = SE3StateVar(Transformation(xi_ab=rng.normal(size=(6, 1)) * 0.1))
        w_var = VSpaceStateVar(rng.normal(size=(6, 1)))
        traj.add_knot(Time(nsecs=t_ns), T_var, w_var)
        state_vars += [T_var, w_var]
    return traj, state_vars


def build(state_vars, cost_terms):
    problem = OptimizationProblem()
    problem.add_state_var(*state_vars)
    problem.add_cost_term(*cost_terms)
    problem.get_state_vector()
    A, b = problem.build_gauss_newton_terms()
    return A, b, problem.cost()


def test_point_to_plane(rng):
    traj, state_vars = make_trajectory(rng)
    T_s_r = SE3StateVar(Transformation(xi_ab=rng.normal(size=(6, 1))), locked=True)

    n_points, n_bins = 200, 7
    query = rng.normal(size=(n_points, 3)) * 5
    ref = query + rng.normal(size=(n_points, 3)) * 0.2
    normals = rng.normal(size=(n_points, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    scores = rng.uniform(0.1, 1.0, n_points)
    bin_ids = rng.integers(0, n_bins, n_points)
    bin_times = np.linspace(10_000_000, 90_000_000, n_bins).astype(np.int64)

    T_ms_bins = [inv(compose(T_s_r, traj.get_pose_interpolator(Time(nsecs=int(t))))) for t in bin_times]
    batched = PointToPlaneCostTerm(T_ms_bins, bin_ids, query, ref, normals, scores)

    reference = []
    for i in range(n_points):
        W = scores[i] * np.outer(normals[i], normals[i]) + 1e-5 * np.eye(3)
        error = P2PErrorEvaluator(T_ms_bins[bin_ids[i]], ref[i][:, None], query[i][:, None])
        reference.append(WeightedLeastSquareCostTerm(error, StaticNoiseModel(W, "information"), L2LossFunc()))

    A1, b1, c1 = build(state_vars, [batched])
    A2, b2, c2 = build(state_vars, reference)
    print(f"point-to-plane: max|dA|={np.abs(A1 - A2).max():.2e} (|A|={np.abs(A2).max():.1e})  "
          f"max|db|={np.abs(b1 - b2).max():.2e}  dcost={abs(c1 - c2):.2e}")
    assert np.allclose(A1, A2, rtol=1e-6, atol=1e-8) and np.allclose(b1, b2, rtol=1e-6, atol=1e-8)
    assert np.isclose(c1, c2)


def test_state_prior(rng):
    traj, state_vars = make_trajectory(rng)
    T_var, w_var = state_vars[0], state_vars[1]
    T_prior = Transformation(xi_ab=rng.normal(size=(6, 1)) * 0.1)
    w_prior = rng.normal(size=(6, 1))
    cov_pose = np.diag(rng.uniform(0.01, 0.1, 6))
    cov_vel = np.diag(rng.uniform(0.01, 0.1, 6))
    cov = np.zeros((12, 12))
    cov[:6, :6] = cov_pose
    cov[6:, 6:] = cov_vel

    joint = StatePriorCostTerm(T_var, w_var, T_prior, w_prior, cov)
    separate = [
        WeightedLeastSquareCostTerm(se3_error(T_var, T_prior), StaticNoiseModel(cov_pose, "covariance"), L2LossFunc()),
        WeightedLeastSquareCostTerm(vspace_error(w_var, w_prior), StaticNoiseModel(cov_vel, "covariance"), L2LossFunc()),
    ]
    A1, b1, c1 = build(state_vars, [joint])
    A2, b2, c2 = build(state_vars, separate)
    print(f"state prior   : max|dA|={np.abs(A1 - A2).max():.2e}  max|db|={np.abs(b1 - b2).max():.2e}  "
          f"dcost={abs(c1 - c2):.2e}")
    assert np.allclose(A1, A2) and np.allclose(b1, b2) and np.isclose(c1, c2)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    test_point_to_plane(rng)
    test_state_prior(rng)
    print("all cost term tests passed")
