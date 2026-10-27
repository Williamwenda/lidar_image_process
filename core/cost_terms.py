"""
Custom pysteam cost terms used by the lidar odometry.

PointToPlaneCostTerm
    vtr adds one cost term per point, each with its own pose interpolator. pysteam evaluates every
    cost term in Python, so that is far too slow here. Instead, points are grouped into time bins:
    all points of a bin share one interpolated pose, and their point-to-plane terms are summed with
    numpy before being chained to the trajectory states through that pose's Jacobian. The error
    and weight per point are the same as vtr's p2pError cost term.

StatePriorCostTerm
    Prior on one trajectory knot (pose + velocity) with a joint 12x12 covariance, like steam's
    const_vel::Interface::addStatePrior (pysteam only has separate pose / velocity priors).

Both follow pysteam's convention for build_gauss_newton_terms: A += J^T W J, b += -J^T W e.
"""
import numpy as np
from pysteam.evaluable import Jacobians
from pysteam.evaluable.se3 import se3_error
from pysteam.evaluable.vspace import vspace_error
from pysteam.problem import CostTerm


def hat(p):
    """(N, 3) -> (N, 3, 3) skew-symmetric matrices."""
    zero = np.zeros(len(p))
    return np.stack([
        np.stack([zero, -p[:, 2], p[:, 1]], axis=-1),
        np.stack([p[:, 2], zero, -p[:, 0]], axis=-1),
        np.stack([-p[:, 1], p[:, 0], zero], axis=-1),
    ], axis=1)


class PointToPlaneCostTerm(CostTerm):
    """cost = sum_i 0.5 * e_i^T W_i e_i
        e_i = ref_i - T_ms(t_bin(i)) @ query_i        (map point - aligned query point)
        W_i = score_i * n_i n_i^T + 1e-5 * I          (n_i, score_i: normal and planarity of ref_i)
    T_ms_bins: one pysteam evaluator of T_ms (sensor -> map) per time bin."""

    def __init__(self, T_ms_bins, bin_ids, query, ref, normals, scores):
        super().__init__()
        self.T_ms_bins = T_ms_bins
        self.bin_members = [np.nonzero(bin_ids == k)[0] for k in range(len(T_ms_bins))]
        self.query = query
        self.ref = ref
        self.W = scores[:, None, None] * np.einsum("ni,nj->nij", normals, normals) + 1e-5 * np.eye(3)

    @property
    def related_var_keys(self):
        keys = set()
        for T_ms in self.T_ms_bins:
            keys |= T_ms.related_var_keys
        return keys

    def errors(self, T_ms, idx):
        aligned = self.query[idx] @ T_ms[:3, :3].T + T_ms[:3, 3]
        return aligned, self.ref[idx] - aligned

    def cost(self):
        total = 0.0
        for T_ms_eval, idx in zip(self.T_ms_bins, self.bin_members):
            if len(idx) == 0:
                continue
            _, e = self.errors(T_ms_eval.evaluate().matrix(), idx)
            total += 0.5 * np.einsum("ni,nij,nj->", e, self.W[idx], e)
        return total

    def build_gauss_newton_terms(self, state_vector, A, b):
        for T_ms_eval, idx in zip(self.T_ms_bins, self.bin_members):
            if len(idx) == 0:
                continue
            # jacs[key] = d(T_ms) / d(state), both as left perturbations, (6, dim)
            jacs = Jacobians()
            T_ms = T_ms_eval.evaluate(np.eye(6), jacs).matrix()
            aligned, e = self.errors(T_ms, idx)

            # de_i / d(left perturbation of T_ms) = [-I, aligned_i^]   (3 x 6)
            J = np.zeros((len(idx), 3, 6))
            J[:, :, :3] = -np.eye(3)
            J[:, :, 3:] = hat(aligned)
            JtW = np.einsum("nai,nab->nib", J, self.W[idx])  # (n, 6, 3)
            H_bin = np.einsum("nib,nbj->ij", JtW, J)          # (6, 6), summed over the bin
            g_bin = np.einsum("nib,nb->i", JtW, e)            # (6,)

            for key1, J1 in jacs.items():
                idx1 = state_vector.get_state_indices(key1)
                b[idx1, :] += -(J1.T @ g_bin)[:, None]
                for key2, J2 in jacs.items():
                    idx2 = state_vector.get_state_indices(key2)
                    A[idx1, idx2] += J1.T @ H_bin @ J2


class StatePriorCostTerm(CostTerm):
    """cost = 0.5 * e^T cov^-1 e,  e = [se3_error(T_var, T_prior); vspace_error(w_var, w_prior)]."""

    def __init__(self, T_var, w_var, T_prior, w_prior, cov):
        super().__init__()
        self.T_var = T_var
        self.w_var = w_var
        self.pose_error = se3_error(T_var, T_prior)
        self.vel_error = vspace_error(w_var, w_prior)
        self.info = np.linalg.inv(cov)

    @property
    def related_var_keys(self):
        return self.T_var.related_var_keys | self.w_var.related_var_keys

    def error_and_jacobian(self):
        jacs_pose = Jacobians()
        jacs_vel = Jacobians()
        e_pose = self.pose_error.evaluate(np.eye(6), jacs_pose)
        e_vel = self.vel_error.evaluate(np.eye(6), jacs_vel)
        e = np.concatenate([e_pose.ravel(), e_vel.ravel()])[:, None]  # (12, 1)
        J = np.zeros((12, 12))                # columns: [T_var (6), w_var (6)]
        J[:6, :6] = jacs_pose[self.T_var.key]
        J[6:, 6:] = jacs_vel[self.w_var.key]
        return e, J

    def cost(self):
        e = np.concatenate([self.pose_error.evaluate().ravel(), self.vel_error.evaluate().ravel()])
        return 0.5 * e @ self.info @ e

    def build_gauss_newton_terms(self, state_vector, A, b):
        e, J = self.error_and_jacobian()
        H = J.T @ self.info @ J
        g = J.T @ self.info @ e
        slices = [state_vector.get_state_indices(self.T_var.key),
                  state_vector.get_state_indices(self.w_var.key)]
        for i, idx_i in enumerate(slices):
            b[idx_i, :] += -g[6 * i:6 * i + 6]
            for j, idx_j in enumerate(slices):
                A[idx_i, idx_j] += H[6 * i:6 * i + 6, 6 * j:6 * j + 6]
