"""
Step 1: run the lidar odometry over the bag and save every scan, motion-undistorted, to
    <out_dir>/frames/<frame_id>.npz      (frame_id = 000000, 000001, ... = row in trajectory.csv)

<out_dir> = outputs/<FOLDER_NAME>/odometry (bag and FOLDER_NAME are set in config.py). The run stops
if it already holds results: pick a new FOLDER_NAME (or delete the old results) so nothing is overwritten.

Saved per frame (H x W = 128 x 1024):
    frame_id       : int64, index of the frame (the first saved frame is 0)
    stamp_ns       : int64, scan header stamp (start of the scan); the frame every quantity refers to
    signal, reflec : (H, W) uint16 signal / reflectivity images, as published by the driver
    range          : (H, W) float32 range of the raw point [m], 0 = no return
    xyz_undist     : (H, W, 3) float32 points in the sensor frame at stamp_ns (mdist), NaN = no return
    col_times      : (W,) float64 firing time of each measurement column [s since stamp_ns]
    col_T_sc_s     : (W, 4, 4) T_{s(t_c), s(stamp)}: sensor frame at the stamp -> at column c's time
    col_w_s        : (W, 6) body velocity [rho; phi] in the sensor frame at column c's time
    T_r_m          : (4, 4) robot pose at stamp_ns (maps map-frame points into the robot frame)
    T_s_m          : (4, 4) T_s_r @ T_r_m (maps map-frame points into the sensor frame)
    w_m_r_in_r     : (6,) body velocity at stamp_ns in the robot frame (vtr convention)
    cov_T_r_m      : (6, 6) covariance of T_r_m
    T_rk_rkm1      : (4, 4) incremental pose from the previous saved frame (identity for the first)
    success, matched_ratio, num_steps, converged, icp_residual : odometry health
                     (icp_residual: RMS point-to-plane distance of the matched points [m])
Pixel (i, j) belongs to measurement column (j - pixel_shift_by_row[i]) mod W (see ouster_model.py).
To re-apply the motion distortion of a point p (sensor frame at the stamp): col_T_sc_s[c] @ p.

The map frame m is the robot frame at the end of the first processed scan. The first scan only
initializes the map and is not saved; the velocity estimate needs a few scans to settle.

Shared files: <out_dir>/meta.json (Ouster metadata, T_s_r, parameters) and trajectory.csv.

Run:  python run_odometry.py      (edit the parameters below)
"""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

import config
from core.bag_io import iter_scans, read_metadata, read_static_transform
from core.motion_distortion import mdist_scan
from core.odometry_icp import OdometryICP
from core.ouster_model import OusterModel
from core.point_map import PointMap
from core.preprocessing import preprocess

######## Parameters (edit by hand) ########
cfg = SimpleNamespace(
    bag=config.BAG,                 # set in config.py
    out_dir=config.ODOMETRY_DIR,    # outputs/<FOLDER_NAME>/odometry, set in config.py
    points_topic="/ouster/points",
    signal_topic="/ouster/signal_image",
    reflec_topic="/ouster/reflec_image",
    metadata_topic="/ouster/metadata",
    robot_frame="w200_0066_base_link",
    lidar_frame="w200_0066_os_lidar",

    # which part of the bag to process
    start_sec=0.0,        # offset from the bag start [s]
    duration_sec=None,    # process at most this many seconds (None = until the end)
    max_frames=None,      # stop after this many scans (None = no limit)

    # preprocessing (vtr ouster_warthog_default.yaml: preprocessing.conversion / filtering)
    body_filter_radius=0.8,
    body_filter_z_min=-0.2,
    body_filter_z_max=0.35,
    crop_range=40.0,
    frame_voxel_size=0.3,

    # sliding map (odometry.mapping)
    map_voxel_size=0.3,
    point_life_time=20,

    # odometry (odometry.icp)
    use_trajectory_estimation=True,     # continuous-time motion compensation (false in the vtr config)
    traj_num_extra_states=0,
    traj_qc_diag=[1.0, 0.1, 0.1, 0.1, 0.1, 1.0],
    num_time_bins=64,                   # points in a bin share one interpolated pose
    first_num_steps=2,
    initial_max_iter=4,
    initial_max_pairing_dist=1.5,
    initial_max_planar_dist=1.0,
    refined_max_iter=50,
    refined_max_pairing_dist=1.0,
    refined_max_planar_dist=0.3,
    averaging_num_steps=2,
    trans_diff_thresh=0.01,              # [m]
    rot_diff_thresh=0.1 * np.pi / 180,   # [rad]
    max_iterations=1,                    # Gauss-Newton iterations per ICP step
    min_matched_ratio=0.4,
    verbose=False,
    # covariance of the prior on the first scan: pose (6) then velocity (6). vtr uses 1e-5 for all
    # (robot at rest); the loose velocity part also allows starting while the robot is moving.
    init_prior_cov_diag=[1e-5] * 6 + [1.0] * 6,
)


def main(cfg):
    out_dir = Path(cfg.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already has results: choose another FOLDER_NAME in config.py or delete it")
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True)
    print(f"writing results to {out_dir}")

    with AnyReader([Path(cfg.bag)], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
        meta = read_metadata(reader, cfg.metadata_topic)
        model = OusterModel(meta)
        T_s_r = read_static_transform(reader, target_frame=cfg.lidar_frame, source_frame=cfg.robot_frame)
        print(f"Ouster {model.H}x{model.W}, T_s_r =\n{np.round(T_s_r, 4)}")

        params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cfg).items()}
        params["out_dir"] = str(out_dir)
        (out_dir / "meta.json").write_text(json.dumps(
            dict(ouster_metadata=meta, T_s_r=T_s_r.tolist(), params=params), indent=1))

        start_ns = reader.start_time + int(cfg.start_sec * 1e9)
        stop_ns = start_ns + int(cfg.duration_sec * 1e9) if cfg.duration_sec else None

        # expected number of scans, for the progress display (one scan per scan period)
        end_ns = min(stop_ns, reader.end_time) if stop_ns else reader.end_time
        num_expected = max(int((end_ns - start_ns) * 1e-9 / model.scan_period), 1)
        if cfg.max_frames is not None:
            num_expected = min(num_expected, cfg.max_frames)
        print(f"processing ~{num_expected} scans, bag time {cfg.start_sec:.1f}-{(end_ns - reader.start_time) * 1e-9:.1f} s")

        point_map = PointMap(cfg.map_voxel_size, cfg.point_life_time)
        odometry = OdometryICP(cfg, T_s_r)
        prev_T_r_m = None
        trajectory_rows = []
        run_start = time.time()

        for k, scan in enumerate(iter_scans(reader, model, cfg.points_topic, cfg.signal_topic,
                                            cfg.reflec_topic, start_ns, stop_ns)):
            stamp = scan["stamp_ns"]
            points, point_times_ns = preprocess(scan, model, cfg)

            if not odometry.initialized:
                # the first scan only initializes the map (its motion is unknown), it is not saved
                result = odometry.first_scan(stamp, points, point_times_ns, model.W)
                point_map.update(result["points_in_map"])
                print(f"[{k + 1:5d}/{num_expected}] first scan, map initialized with {len(point_map)} points")
                continue

            result = odometry.run(stamp, points, point_times_ns, point_map, scan["col_times"])

            if result["success"]:
                point_map.update(result["points_in_map"])

            T_r_m = result["T_r_m"]
            T_rk_rkm1 = np.eye(4) if prev_T_r_m is None else T_r_m @ np.linalg.inv(prev_T_r_m)
            prev_T_r_m = T_r_m
            col_T_sc_s = result["col_T_sc_s"] if result["success"] else np.tile(np.eye(4), (model.W, 1, 1))

            frame_id = len(trajectory_rows)  # = row in trajectory.csv
            np.savez_compressed(
                frames_dir / f"{frame_id:06d}.npz",
                frame_id=np.int64(frame_id),
                stamp_ns=np.int64(stamp),
                signal=scan["signal"],
                reflec=scan["reflec"],
                range=scan["range"],
                xyz_undist=mdist_scan(scan, model, col_T_sc_s).astype(np.float32),
                col_times=scan["col_times"],
                col_T_sc_s=col_T_sc_s,
                col_w_s=result["col_w_s"],
                T_r_m=T_r_m,
                T_s_m=T_s_r @ T_r_m,
                w_m_r_in_r=result["w_m_r_in_r"],
                cov_T_r_m=result["cov_T_r_m"],
                T_rk_rkm1=T_rk_rkm1,
                success=result["success"],
                matched_ratio=result["matched_ratio"],
                num_steps=result["num_steps"],
                converged=result["converged"],
                icp_residual=result["icp_residual"],
            )

            # robot position in the map frame and speed / yaw rate (w is the map's motion seen from the robot)
            r_m = np.linalg.inv(T_r_m)[:3, 3]
            w = result["w_m_r_in_r"]
            trajectory_rows.append([stamp, int(result["success"]), result["matched_ratio"], *r_m,
                                    np.linalg.norm(w[:3]), -w[5], result["num_steps"], int(result["converged"]),
                                    result["icp_residual"]])
            # progress: scans done, bag time, elapsed and estimated remaining time
            elapsed = time.time() - run_start
            remaining = elapsed / (k + 1) * max(num_expected - (k + 1), 0)
            status = "ok" if result["success"] else "FAILED"
            if not result["converged"]:
                status += " (not converged)"
            print(f"[{k + 1:5d}/{num_expected} {100 * (k + 1) / num_expected:3.0f}%]  "
                  f"bag t={(stamp - reader.start_time) * 1e-9:6.1f}s  "
                  f"elapsed {format_duration(elapsed)}  left ~{format_duration(remaining)}  |  "
                  f"{status}  matched={result['matched_ratio']:.2f}  res={result['icp_residual'] * 100:.1f}cm  |  "
                  f"pos=({r_m[0]:.1f}, {r_m[1]:.1f}, {r_m[2]:.1f})  speed={np.linalg.norm(w[:3]):.2f}m/s  "
                  f"yaw rate={-w[5]:+.2f}rad/s")

            if cfg.max_frames is not None and k + 1 >= cfg.max_frames:
                break

    # written by hand: going through a float array would round the int64 stamps
    row_format = "{:d},{:d},{:.3f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:d},{:d},{:.5f}\n"
    with open(out_dir / "trajectory.csv", "w") as f:
        f.write("# stamp_ns,success,matched_ratio,x_m,y_m,z_m,speed_mps,yaw_rate_radps,"
                "num_steps,converged,icp_residual_m\n")
        for row in trajectory_rows:
            f.write(row_format.format(*row))
    print(f"wrote {len(trajectory_rows)} frames to {frames_dir} in {format_duration(time.time() - run_start)}")


def format_duration(seconds):
    """e.g. 45 -> '45s', 125 -> '2m05s', 4000 -> '1h06m'."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


if __name__ == "__main__":
    main(cfg)
