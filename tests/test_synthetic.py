"""
Known-motion test of the odometry and of the pair correspondences, with simulated scans
(tests/lidar_simulator.py: real Ouster beam model / timing / mounting, known continuous motion).

Runs the odometry on the same simulated scans with four settings:
    compensation on        : the default (continuous-time motion compensation)
    compensation off       : rigid, every point at the end of the scan (vtr's deployed config)
    on, tight convergence  : much smaller convergence thresholds -> does the ICP stop too early?
    on, forced failure     : one scan is forced to fail -> does the failure path recover?
and compares with the truth:
    drift         : estimated vs true motion since the first settled scan
                    (+ the constant offset of the map frame, from the uncompensated first scan)
    step          : scan-to-scan relative pose error
    velocity      : w_m_r_in_r vs the true body velocity at the stamp
    undistortion  : xyz_undist vs the true (noise-free) points in the sensor frame at the stamp
    pairs         : make_pairs correspondences vs correspondences computed with the true motion

Run:  python tests/test_synthetic.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

import lidar_simulator as sim
import make_pairs
import run_odometry
from bag_io import read_metadata, read_static_transform
from motion_distortion import mdist_scan
from odometry_icp import OdometryICP
from ouster_model import OusterModel
from point_map import PointMap
from preprocessing import preprocess

######## Parameters ########
NUM_SCANS = 40          # 4 s of driving
RANGE_NOISE = 0.01      # [m], along the beam
SETTLE = 5              # ignore the first scans while the velocity estimate settles
PAIR_GAP = 10           # pairs 1 s apart
FAIL_AT = 15            # scan forced to fail in the failure run
SEED = 0
T0_NS = 1_747_839_879_068_984_890   # realistic absolute stamps


def rot_deg(T):
    return np.rad2deg(np.linalg.norm(Rotation.from_matrix(T[:3, :3]).as_rotvec()))


def run_odometry_on(scans, model, T_s_r, cfg, fail_at=None):
    """Same loop as run_odometry.main, on simulated scans. Returns the results per scan (None for
    the first scan) and the time of the map frame (end of the first scan)."""
    point_map = PointMap(cfg.map_voxel_size, cfg.point_life_time)
    odometry = OdometryICP(cfg, T_s_r)
    results = [None]
    for k, scan in enumerate(scans):
        points, point_times_ns = preprocess(scan, model, cfg)
        if k == 0:
            point_map.update(odometry.first_scan(scan["stamp_ns"], points, point_times_ns, model.W)["points_in_map"])
            map_time_ns = odometry.time_prior_ns
            continue
        min_matched_ratio = cfg.min_matched_ratio
        if k == fail_at:
            cfg.min_matched_ratio = 1.01  # impossible -> this scan fails
        result = odometry.run(scan["stamp_ns"], points, point_times_ns, point_map, scan["col_times"])
        cfg.min_matched_ratio = min_matched_ratio
        if result["success"]:
            point_map.update(result["points_in_map"])
        else:
            result["col_T_sc_s"] = np.tile(np.eye(4), (model.W, 1, 1))  # as run_odometry saves it
        result["xyz_undist"] = mdist_scan(scan, model, result["col_T_sc_s"])
        results.append(result)
    return results, map_time_ns


def evaluate_odometry(scans, truths, results, map_time_ns, traj_T, traj_w):
    # The map frame is the robot frame at the end of the first scan, whose points are put into the
    # map without motion compensation (velocity unknown, as in vtr). That shifts the map frame by
    # about half a scan of motion: a constant offset, reported separately from the drift.
    T_r_M_map = traj_T[sim.pose_index((map_time_ns - T0_NS) * 1e-9)]
    T_true_settle = traj_T[sim.pose_index((scans[SETTLE]["stamp_ns"] - T0_NS) * 1e-9)] @ np.linalg.inv(T_r_M_map)
    E_offset = results[SETTLE]["T_r_m"] @ np.linalg.inv(T_true_settle)
    print(f"  map frame offset : {np.linalg.norm(E_offset[:3, 3]) * 100:.2f} cm / {rot_deg(E_offset):.3f} deg "
          f"(constant; from the uncompensated first scan)")
    rows = []
    prev = None
    for k in range(SETTLE, len(scans)):
        r = results[k]
        idx = sim.pose_index((scans[k]["stamp_ns"] - T0_NS) * 1e-9)
        T_true = traj_T[idx] @ np.linalg.inv(T_r_M_map)
        # drift since scan SETTLE: estimated vs true motion from scan SETTLE to scan k
        motion_est = r["T_r_m"] @ np.linalg.inv(results[SETTLE]["T_r_m"])
        motion_true = T_true @ np.linalg.inv(T_true_settle)
        E = motion_est @ np.linalg.inv(motion_true)
        w_err = r["w_m_r_in_r"] - traj_w[idx]
        step_t = step_r = np.nan
        if prev is not None:
            E_step = (r["T_r_m"] @ np.linalg.inv(prev[0])) @ np.linalg.inv(T_true @ np.linalg.inv(prev[1]))
            step_t, step_r = np.linalg.norm(E_step[:3, 3]), rot_deg(E_step)
        prev = (r["T_r_m"], T_true)
        valid = scans[k]["valid"]
        undist_err = np.linalg.norm(r["xyz_undist"][valid] - truths[k]["xyz_true"][valid], axis=1)
        raw_err = np.linalg.norm(scans[k]["xyz"][valid] - truths[k]["xyz_true"][valid], axis=1)
        rows.append([np.linalg.norm(E[:3, 3]), rot_deg(E), step_t, step_r, np.linalg.norm(w_err[:3]),
                     np.linalg.norm(w_err[3:]), np.median(undist_err), np.percentile(undist_err, 99),
                     np.median(raw_err), np.percentile(raw_err, 99), r["num_steps"], r["success"]])
    rows = np.array(rows, dtype=np.float64)
    ok = rows[:, 11] > 0
    path = sum(np.linalg.norm((traj_T[sim.pose_index((scans[k]["stamp_ns"] - T0_NS) * 1e-9)] @ np.linalg.inv(
        traj_T[sim.pose_index((scans[k - 1]["stamp_ns"] - T0_NS) * 1e-9)]))[:3, 3]) for k in range(SETTLE + 1, len(scans)))
    print(f"  drift            : {rows[-1, 0] * 100:.2f} cm / {rows[-1, 1]:.3f} deg over {path:.1f} m "
          f"(max {np.nanmax(rows[:, 0]) * 100:.2f} cm / {np.nanmax(rows[:, 1]):.3f} deg)")
    print(f"  step error       : mean {np.nanmean(rows[ok, 2]) * 1000:.1f} mm / {np.nanmean(rows[ok, 3]):.4f} deg   "
          f"(max {np.nanmax(rows[ok, 2]) * 1000:.1f} mm / {np.nanmax(rows[ok, 3]):.4f} deg)")
    print(f"  velocity error   : mean {np.mean(rows[ok, 4]):.3f} m/s, {np.mean(rows[ok, 5]):.3f} rad/s   "
          f"(max {np.max(rows[ok, 4]):.3f} m/s, {np.max(rows[ok, 5]):.3f} rad/s)")
    print(f"  undistortion err : median {np.mean(rows[ok, 6]) * 100:.2f} cm, 99% {np.mean(rows[ok, 7]) * 100:.2f} cm   "
          f"(raw scan: median {np.mean(rows[:, 8]) * 100:.1f} cm, 99% {np.mean(rows[:, 9]) * 100:.1f} cm)")
    print(f"  ICP steps        : {np.unique(rows[:, 10]).astype(int).tolist()},  failed scans: {int((~ok).sum())}")
    return rows


def frame_for_pairs(scan, xyz_undist, col_T_sc_s, T_s_m):
    return dict(valid=scan["valid"], range=scan["range"], xyz_undist=xyz_undist,
                col_T_sc_s=col_T_sc_s, T_s_m=T_s_m)


def evaluate_pairs(scans, truths, results, model, T_s_r):
    pcfg = make_pairs.cfg
    errors, ranges, false_matches, missed = [], [], [], []
    for a in range(SETTLE, len(scans) - PAIR_GAP, 3):
        b = a + PAIR_GAP
        if not (results[a]["success"] and results[b]["success"]):
            continue
        # pipeline: odometry -> ICP refinement -> correspondences
        fa = frame_for_pairs(scans[a], results[a]["xyz_undist"], results[a]["col_T_sc_s"], T_s_r @ results[a]["T_r_m"])
        fb = frame_for_pairs(scans[b], results[b]["xyz_undist"], results[b]["col_T_sc_s"], T_s_r @ results[b]["T_r_m"])
        T_AB_odom = fa["T_s_m"] @ np.linalg.inv(fb["T_s_m"])
        T_AB, _, _ = make_pairs.run_icp(make_pairs.icp_points(fb, pcfg), make_pairs.icp_points(fa, pcfg),
                                        T_AB_odom, pcfg.icp_scales)
        corr_est, _ = make_pairs.compute_correspondences(fa, fb, np.linalg.inv(T_AB), model, pcfg)

        # truth: same code with the true points, motion and poses
        ta = frame_for_pairs(scans[a], truths[a]["xyz_true"], truths[a]["col_T_sc_s"], truths[a]["T_s_M"])
        tb = frame_for_pairs(scans[b], truths[b]["xyz_true"], truths[b]["col_T_sc_s"], truths[b]["T_s_M"])
        T_AB_true = ta["T_s_m"] @ np.linalg.inv(tb["T_s_m"])
        corr_true, range_true = make_pairs.compute_correspondences(ta, tb, np.linalg.inv(T_AB_true), model, pcfg)

        est = np.isfinite(corr_est[..., 0])
        true = np.isfinite(corr_true[..., 0])
        both = est & true
        d_row = corr_est[both, 0] - corr_true[both, 0]
        d_col = (corr_est[both, 1] - corr_true[both, 1] + model.W / 2) % model.W - model.W / 2
        errors.append(np.hypot(d_row, d_col))
        ranges.append(range_true[both])
        false_matches.append((est & ~true).sum() / max(est.sum(), 1))
        missed.append((true & ~est).sum() / max(true.sum(), 1))
    err = np.concatenate(errors)
    rng = np.concatenate(ranges)
    print(f"  pair GT error    : median {np.median(err):.3f} px, 95% {np.percentile(err, 95):.3f} px, "
          f"99% {np.percentile(err, 99):.3f} px, >1px {100 * np.mean(err > 1):.2f}%   "
          f"| matches not in truth {100 * np.mean(false_matches):.2f}%, truth missed {100 * np.mean(missed):.2f}%  "
          f"({len(errors)} pairs)")
    for lo, hi in [(1, 3), (3, 10), (10, 200)]:
        m = (rng >= lo) & (rng < hi)
        print(f"      range {lo:3d}-{hi:3d} m : median {np.median(err[m]):.3f} px, >1px {100 * np.mean(err[m] > 1):.2f}%")
    return err, rng


def main():
    with AnyReader([Path(run_odometry.cfg.bag)], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
        model = OusterModel(read_metadata(reader))
        T_s_r = read_static_transform(reader, run_odometry.cfg.lidar_frame, run_odometry.cfg.robot_frame)

    rng = np.random.default_rng(SEED)
    traj_t, traj_T, traj_w = sim.make_trajectory((NUM_SCANS + 1) * model.scan_period)
    path_xy = np.linalg.inv(traj_T)[::100, :2, 3]
    scene, intensity = sim.build_world(rng, path_xy)
    scans, truths = [], []
    for k in range(NUM_SCANS):
        scan, truth = sim.simulate_scan(scene, intensity, model, T_s_r, traj_T, k * model.scan_period, T0_NS,
                                        rng, RANGE_NOISE)
        scans.append(scan)
        truths.append(truth)
    speed = -traj_w[:, 0]
    print(f"simulated {NUM_SCANS} scans: path {np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum():.1f} m, "
          f"speed {speed.min():.1f}-{speed.max():.1f} m/s, |yaw rate| <= {np.abs(traj_w[:, 5]).max():.2f} rad/s, "
          f"range noise {RANGE_NOISE * 100:.0f} cm")

    settings = {
        "compensation on": dict(),
        "compensation off": dict(use_trajectory_estimation=False),
        "on, tight convergence": dict(trans_diff_thresh=1e-4, rot_diff_thresh=np.deg2rad(1e-3)),
        "on, forced failure": dict(fail_at=FAIL_AT),
    }
    summary = {}
    for name, changes in settings.items():
        fail_at = changes.pop("fail_at", None)
        cfg = SimpleNamespace(**{**vars(run_odometry.cfg), **changes})
        print(f"\n=== {name} ===")
        results, map_time_ns = run_odometry_on(scans, model, T_s_r, cfg, fail_at=fail_at)
        rows = evaluate_odometry(scans, truths, results, map_time_ns, traj_T, traj_w)
        err, rng = evaluate_pairs(scans, truths, results, model, T_s_r)
        summary[name] = (rows, err, rng, results)

    # pass criteria for the default setting (beyond 3 m; closer, the GT is limited by the ~5 mm
    # relative pose error and the range noise)
    rows, err, rng, results = summary["compensation on"]
    assert rows[-1, 0] < 0.02 and rows[-1, 1] < 0.1, "drift too large"
    far = rng >= 3.0
    assert np.median(err[far]) < 0.25 and np.mean(err[far] > 1) < 0.01, "pair correspondences too inaccurate"
    # the forced failure is reported and the odometry recovers
    fail_results = summary["on, forced failure"][3]
    assert not fail_results[FAIL_AT]["success"] and all(r["success"] for r in fail_results[FAIL_AT + 1:])
    print("\nsynthetic test passed")


if __name__ == "__main__":
    main()
