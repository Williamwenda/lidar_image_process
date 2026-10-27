"""
Python port of vtr's lidar odometry (vtr_lidar/src/modules/odometry/odometry_icp_module.cpp):
scan-to-map point-to-plane ICP with a continuous-time (constant-velocity GP) trajectory, solved
with pysteam.

Notation (Barfoot / vtr):
    T_r_m       : pose of the robot, maps points from the map (odometry) frame m to the robot frame r
    w_m_r_in_r  : body velocity [rho; phi] (6,), T_r(t)_m = exp((t - t0) * w^) @ T_r(t0)_m
                  (driving forward gives a negative rho_x: the map moves backwards in the robot frame)
    T_s_r       : robot -> sensor (os_lidar), fixed

Per scan (same as vtr):
  1. Knots at the end of the previous scan and at the end of this scan (+ optional extra knots),
     initialized by extrapolating the previous estimate with the previous velocity.
  2. A 12x12 state prior on the first knot carries the information from the previous scan.
  3. ICP loop: align the query points with their interpolated poses, find nearest map points,
     filter by pairing / planar distance, take one Gauss-Newton step, check convergence. After a
     first coarse stage the distance thresholds are tightened (refinement stage).
  4. The pose / velocity at the scan stamp are the outputs; the last knot becomes the next prior.

Differences to vtr:
  * Points are grouped into `num_time_bins` time bins that share one interpolated pose (speed).
  * The next prior's covariance is the marginal covariance of the last knot taken from the full
    problem (mathematically the same as vtr's marginalization, without its extra solver step).
  * The outputs are evaluated at the scan stamp, extrapolating if the stamp is slightly before
    the end of the previous scan (vtr clamps the stamp instead).
  * No gyro cost terms.
"""
import numpy as np

from . import deps_path  # noqa: F401
from pylgmath import Transformation, se3op
from pysteam.evaluable import Jacobians
from pysteam.evaluable.se3 import SE3StateVar, compose, inv
from pysteam.evaluable.vspace import VSpaceStateVar
from pysteam.problem import OptimizationProblem
from pysteam.solver import Covariance, GaussNewtonSolver
from pysteam.trajectory import Time
from pysteam.trajectory.const_vel import Interface

from .cost_terms import PointToPlaneCostTerm, StatePriorCostTerm


class OdometryICP:
    def __init__(self, cfg, T_s_r):
        self.cfg = cfg
        self.T_s_r = Transformation(T_ba=T_s_r)
        self.initialized = False

        # prior carried from one scan to the next (vtr: T_r_m_odo_prior, w_m_r_in_r_odo_prior, ...)
        self.T_r_m_prior = None
        self.w_prior = None
        self.cov_prior = None
        self.time_prior_ns = None

    def first_scan(self, stamp_ns, points, point_times_ns, num_cols):
        """The first scan defines the map frame: the robot frame at the end of the scan (as in vtr,
        whose first prior is the identity at the time of the last point). Nothing to estimate."""
        self.T_r_m_prior = Transformation()
        self.w_prior = np.zeros((6, 1))
        self.cov_prior = np.diag(self.cfg.init_prior_cov_diag)
        self.time_prior_ns = int(point_times_ns.max())
        self.initialized = True

        T_m_s = self.T_s_r.inverse().matrix()
        return dict(points_in_map=points @ T_m_s[:3, :3].T + T_m_s[:3, 3])

    def run(self, stamp_ns, points, point_times_ns, point_map, col_times):
        cfg = self.cfg
        frame_start_ns = self.time_prior_ns
        frame_end_ns = int(point_times_ns.max())

        # ---- (1) trajectory knots, initialized by extrapolating the prior ----
        traj = Interface(np.asarray(cfg.traj_qc_diag, dtype=np.float64))
        num_states = cfg.traj_num_extra_states + 2
        knot_times_ns = [frame_start_ns + i * (frame_end_ns - frame_start_ns) // (num_states - 1)
                         for i in range(num_states)]
        knot_times_ns[-1] = frame_end_ns
        state_vars = []
        for knot_ns in knot_times_ns:
            dt = (knot_ns - frame_start_ns) * 1e-9
            T_init = Transformation(xi_ab=dt * self.w_prior) @ self.T_r_m_prior
            T_var = SE3StateVar(T_init)
            w_var = VSpaceStateVar(self.w_prior.copy())
            traj.add_knot(Time(nsecs=int(knot_ns)), T_var, w_var)
            state_vars += [T_var, w_var]

        # ---- (2) prior terms: GP motion prior between knots + state prior on the first knot ----
        prior_costs = traj.get_prior_cost_terms()
        prior_costs.append(StatePriorCostTerm(state_vars[0], state_vars[1],
                                              self.T_r_m_prior, self.w_prior, self.cov_prior))

        # ---- time bins: points in a bin share one interpolated pose ----
        T_s_r_var = SE3StateVar(self.T_s_r, locked=True)
        if cfg.use_trajectory_estimation:
            t_rel = point_times_ns - point_times_ns.min()
            bin_ids = (t_rel * cfg.num_time_bins) // (t_rel.max() + 1)
            _, bin_ids = np.unique(bin_ids, return_inverse=True)  # drop empty bins
            bin_times_ns = [int(point_times_ns.min() + t_rel[bin_ids == k].mean())
                            for k in range(bin_ids.max() + 1)]
        else:  # rigid: every point at the end of the scan
            bin_ids = np.zeros(len(points), dtype=np.int64)
            bin_times_ns = [frame_end_ns]
        T_ms_bins = [inv(compose(T_s_r_var, traj.get_pose_interpolator(Time(nsecs=t)))) for t in bin_times_ns]
        T_ms_stamp = inv(compose(T_s_r_var, traj.get_pose_interpolator(Time(nsecs=int(stamp_ns)))))

        # ---- (3) ICP loop ----
        max_iter = cfg.initial_max_iter
        max_pair_dist = cfg.initial_max_pairing_dist
        max_planar_dist = cfg.initial_max_planar_dist
        refinement_stage = False
        refinement_step = 0
        mean_dT = 0.0
        mean_dR = 0.0
        prev_T_ms = None
        solver_failed = False

        aligned = self.align(points, bin_ids, T_ms_bins)
        step = 0
        while True:
            # nearest map point for every query point
            dist, nn = point_map.kdtree.query(aligned, k=1, workers=-1)
            nn = np.minimum(nn, len(point_map) - 1)
            ref = point_map.points[nn]
            ref_normals = point_map.normals[nn]
            ref_scores = point_map.scores[nn]

            matched = dist < max_pair_dist
            if step >= cfg.first_num_steps:
                planar_dist = np.abs(np.sum((aligned - ref) * ref_normals, axis=1))
                matched &= planar_dist < max_planar_dist
            matched_ratio = matched.mean()
            use = matched & (ref_scores > 0.0)

            # one Gauss-Newton step
            problem = OptimizationProblem()
            problem.add_state_var(*state_vars)
            problem.add_cost_term(*prior_costs)
            problem.add_cost_term(PointToPlaneCostTerm(T_ms_bins, bin_ids[use], points[use], ref[use],
                                                       ref_normals[use], ref_scores[use]))
            solver = GaussNewtonSolver(problem, verbose=cfg.verbose, max_iterations=cfg.max_iterations)
            try:
                solver.optimize()
            except RuntimeError as err:
                print(f"  [odometry] solver failed: {err}")
                solver_failed = True

            aligned = self.align(points, bin_ids, T_ms_bins)

            # convergence: running average of the change of T_ms at the stamp
            T_ms = T_ms_stamp.evaluate().matrix()
            if prev_T_ms is not None:
                avg_tot = 1.0 if step == 1 else float(cfg.averaging_num_steps)
                diff = se3op.tran2vec(T_ms @ np.linalg.inv(prev_T_ms)).ravel()
                mean_dT += (np.linalg.norm(diff[:3]) - mean_dT) / avg_tot
                mean_dR += (np.linalg.norm(diff[3:]) - mean_dR) / avg_tot
            prev_T_ms = T_ms
            converged = mean_dT < cfg.trans_diff_thresh and mean_dR < cfg.rot_diff_thresh

            if refinement_stage:
                refinement_step += 1

            # switch to the refinement stage: tighter distance thresholds
            if not refinement_stage and step >= cfg.first_num_steps:
                if step >= max_iter - 1 or converged:
                    refinement_stage = True
                    max_iter = step + cfg.refined_max_iter
                    max_pair_dist = cfg.refined_max_pairing_dist
                    max_planar_dist = cfg.refined_max_planar_dist

            if (refinement_stage and step >= max_iter - 1) or \
                    (refinement_step > cfg.averaging_num_steps and converged) or solver_failed:
                break
            step += 1

        # health: RMS point-to-plane distance of the matched points after the last step
        planar_dist = np.sum((aligned[use] - ref[use]) * ref_normals[use], axis=1)
        icp_residual = float(np.sqrt(np.mean(planar_dist ** 2))) if use.any() else np.nan

        # ---- (4) outputs at the scan stamp ----
        T_r_m_stamp_eval = traj.get_pose_interpolator(Time(nsecs=int(stamp_ns)))
        jacs = Jacobians()
        T_r_m = T_r_m_stamp_eval.evaluate(np.eye(6), jacs)
        w_m_r_in_r = traj.get_velocity_interpolator(Time(nsecs=int(stamp_ns))).evaluate()

        covariance = Covariance(problem)
        J = np.hstack([jacs[v.key] if v.key in jacs else np.zeros((6, 6)) for v in state_vars])
        cov_T_r_m = J @ covariance.query(state_vars) @ J.T

        knots = [(t, state_vars[2 * i].value.matrix(), state_vars[2 * i + 1].value.ravel())
                 for i, t in enumerate(knot_times_ns)]
        if cfg.use_trajectory_estimation:
            col_T_sc_s, col_w_s = self.column_motion(knots, stamp_ns, T_r_m, col_times)
        else:  # rigid (vtr): every point is treated as measured at the end of the scan
            end_times = np.full(len(col_times), (frame_end_ns - stamp_ns) * 1e-9)
            col_T_sc_s, col_w_s = self.column_motion(knots, stamp_ns, T_r_m, end_times)

        success = (matched_ratio > cfg.min_matched_ratio) and not solver_failed
        if success:
            # the last knot becomes the prior of the next scan
            cov_end = traj.get_covariance(covariance, Time(nsecs=frame_end_ns))
            self.T_r_m_prior = Transformation(transformation=state_vars[-2].value)
            self.w_prior = state_vars[-1].value.copy()
            self.cov_prior = 0.5 * (cov_end + cov_end.T)
            self.time_prior_ns = frame_end_ns

        return dict(
            success=success,
            T_r_m=T_r_m.matrix(),
            w_m_r_in_r=w_m_r_in_r.ravel(),
            cov_T_r_m=cov_T_r_m,
            col_T_sc_s=col_T_sc_s,
            col_w_s=col_w_s,
            points_in_map=aligned,
            matched_ratio=float(matched_ratio),
            num_steps=step + 1,
            converged=bool(converged),   # last running averages below the thresholds
            mean_dT=mean_dT,
            mean_dR=mean_dR,
            icp_residual=icp_residual,
        )

    @staticmethod
    def align(points, bin_ids, T_ms_bins):
        """Query points -> map frame, each with the pose of its time bin."""
        aligned = np.empty_like(points)
        for k, T_ms_eval in enumerate(T_ms_bins):
            T_ms = T_ms_eval.evaluate().matrix()
            in_bin = bin_ids == k
            aligned[in_bin] = points[in_bin] @ T_ms[:3, :3].T + T_ms[:3, 3]
        return aligned

    def column_motion(self, knots, stamp_ns, T_r_m_stamp, col_times):
        """Per measurement column c, from the estimated trajectory:
            col_T_sc_s (W, 4, 4) : T_{s(t_c), s(stamp)}, sensor frame at the stamp -> at the column time
            col_w_s    (W, 6)    : body velocity in the sensor frame at the column time
        """
        T_s_r = self.T_s_r.matrix()
        T_r_s = np.linalg.inv(T_s_r)
        T_m_r_stamp = T_r_m_stamp.inverse().matrix()

        col_times_ns = int(stamp_ns) + np.rint(col_times * 1e9).astype(np.int64)
        T_rc_m, w_r = sample_trajectory(knots, col_times_ns)
        col_T_sc_s = T_s_r @ T_rc_m @ T_m_r_stamp @ T_r_s
        col_w_s = w_r @ self.T_s_r.adjoint().T  # Ad(T_s_r) @ w for every column
        return col_T_sc_s, col_w_s


def sample_trajectory(knots, times_ns):
    """Constant-velocity GP trajectory at many times at once; same formulas as pysteam's
    const_vel PoseInterpolator / VelocityInterpolator (and extrapolation outside the knots).
    Batched for speed: ~10x faster than one pysteam interpolator per column (~1.5 s per scan);
    tests/test_trajectory.py checks it against pysteam.
    knots: list of (time_ns, T_k0 (4, 4), w_0k_ink (6,)) sorted by time.
    Returns T_i0 (N, 4, 4) and w_0i_ini (N, 6)."""
    times_ns = np.asarray(times_ns, dtype=np.int64)
    T_out = np.zeros((len(times_ns), 4, 4))
    w_out = np.zeros((len(times_ns), 6))
    knot_times = np.array([k[0] for k in knots], dtype=np.int64)

    # before the first / after the last knot: extrapolate with that knot's velocity
    for side in ("before", "after"):
        t_k, T_k, w_k = knots[0] if side == "before" else knots[-1]
        outside = times_ns < knot_times[0] if side == "before" else times_ns >= knot_times[-1]
        if not outside.any():
            continue
        dt = (times_ns[outside] - t_k) * 1e-9
        T_out[outside] = se3op.vec2tran(dt[:, None, None] * w_k[None, :, None]) @ T_k
        w_out[outside] = w_k

    # between knots k and k+1
    segment = np.searchsorted(knot_times, times_ns, side="right") - 1
    for k in range(len(knots) - 1):
        (t1, T1, w1), (t2, T2, w2) = knots[k], knots[k + 1]
        inside = (segment == k) & (times_ns >= knot_times[0]) & (times_ns < knot_times[-1])
        if not inside.any():
            continue
        T = (t2 - t1) * 1e-9
        tau = (times_ns[inside] - t1) * 1e-9
        kappa = (t2 - times_ns[inside]) * 1e-9

        # 2x2 GP coefficients (helper.getQ / getQinv / getTran), one set per time
        Q_tau = np.stack([np.stack([tau ** 3 / 3, tau ** 2 / 2], -1), np.stack([tau ** 2 / 2, tau], -1)], 1)
        Qinv_T = np.array([[12 / T ** 3, -6 / T ** 2], [-6 / T ** 2, 4 / T]])
        Tran_kappa_T = np.stack([np.stack([np.ones_like(kappa), np.zeros_like(kappa)], -1),
                                 np.stack([kappa, np.ones_like(kappa)], -1)], 1)   # Tran(kappa)^T
        Tran_tau = np.stack([np.stack([np.ones_like(tau), tau], -1),
                             np.stack([np.zeros_like(tau), np.ones_like(tau)], -1)], 1)
        Tran_T = np.array([[1.0, T], [0.0, 1.0]])
        Omega = Q_tau @ Tran_kappa_T @ Qinv_T
        Lambda = Tran_tau - Omega @ Tran_T

        # knot-dependent terms, shared by all times in the segment
        xi_21 = se3op.tran2vec(T2 @ np.linalg.inv(T1)).ravel()
        gamma21 = (se3op.vec2jacinv(xi_21[:, None]) @ w2[:, None]).ravel()

        xi_i1 = Lambda[:, 0, 1, None] * w1 + Omega[:, 0, 0, None] * xi_21 + Omega[:, 0, 1, None] * gamma21
        xi_dot = Lambda[:, 1, 1, None] * w1 + Omega[:, 1, 0, None] * xi_21 + Omega[:, 1, 1, None] * gamma21
        T_out[inside] = se3op.vec2tran(xi_i1[:, :, None]) @ T1
        w_out[inside] = (se3op.vec2jac(xi_i1[:, :, None]) @ xi_dot[:, :, None])[:, :, 0]
    return T_out, w_out
