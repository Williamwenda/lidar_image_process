"""
Simulated Ouster scans with a known, continuous robot motion (for testing the odometry).

  * A synthetic world (ground, walls, boxes, poles) is ray-cast with open3d.
  * The sensor uses the real beam model (altitude / azimuth angles, beam origin offset,
    pixel_shift_by_row) and the real mounting T_s_r. Every measurement column is fired at its own
    time, from the sensor pose at that time, so the scans are motion-distorted like real ones.
  * Scans are returned in the same format as bag_io.iter_scans, plus the ground truth.

Trajectory convention (same as the odometry): T_r(t)_M maps world points into the robot frame and
T_r(t + dt)_M = exp(dt * w^) @ T_r(t)_M with w = w_m_r_in_r (forward motion -> negative rho_x).
"""
import numpy as np
import open3d as o3d

import deps_path  # noqa: F401
from pylgmath import se3op

GROUND_Z = -0.3      # ground height in the world frame [m]
MAX_RANGE = 120.0    # [m]
TRAJ_DT = 1e-4       # trajectory integration step [s]


def make_trajectory(duration):
    """Robot driving with varying speed and yaw rate plus a little pitch / roll wobble.
    Returns times (N,), T_r_M (N, 4, 4) and w_m_r_in_r (N, 6)."""
    t = np.arange(0.0, duration, TRAJ_DT)
    speed = 1.5 + 0.4 * np.sin(2 * np.pi * t / 4.0)
    yaw_rate = 0.7 * np.sin(2 * np.pi * t / 5.0)
    roll_rate = 0.05 * np.sin(2 * np.pi * t / 1.1)
    pitch_rate = 0.05 * np.sin(2 * np.pi * t / 0.9)

    w = np.zeros((len(t), 6))
    w[:, 0] = -speed        # the world moves backwards in the robot frame
    w[:, 3] = -roll_rate
    w[:, 4] = -pitch_rate
    w[:, 5] = -yaw_rate

    T_r_M = np.zeros((len(t), 4, 4))
    T_r_M[0] = np.eye(4)
    steps = se3op.vec2tran(TRAJ_DT * w[:-1, :, None])
    for i in range(len(t) - 1):
        T_r_M[i + 1] = steps[i] @ T_r_M[i]
    return t, T_r_M, w


def pose_index(times_s):
    return np.rint(np.asarray(times_s) / TRAJ_DT).astype(np.int64)


def build_world(rng, path_xy, num_boxes=150, num_poles=40, clearance=2.5):
    """Ray-casting scene with ground, walls, random boxes and poles kept away from the path.
    Returns the scene and one random intensity per geometry (for the images)."""
    meshes = []
    ground = o3d.geometry.TriangleMesh.create_box(200.0, 200.0, 0.2)
    ground.translate([-100.0, -100.0, GROUND_Z - 0.2])
    meshes.append(ground)
    for x, y, sx, sy in [(-40, -40, 80, 0.5), (-40, 40, 80, 0.5), (-40, -40, 0.5, 80), (40, -40, 0.5, 80)]:
        wall = o3d.geometry.TriangleMesh.create_box(sx, sy, 8.0)
        wall.translate([x, y, GROUND_Z])
        meshes.append(wall)

    def far_from_path(xy):
        return np.min(np.linalg.norm(path_xy - xy, axis=1)) > clearance

    while len(meshes) < 5 + num_boxes:
        xy = rng.uniform(-35, 35, 2)
        size = rng.uniform([0.5, 0.5, 0.5], [4.0, 4.0, 6.0])
        if not far_from_path(xy) or not far_from_path(xy + size[:2]):
            continue
        box = o3d.geometry.TriangleMesh.create_box(*size)
        box.translate([xy[0], xy[1], GROUND_Z])
        box.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, 0, rng.uniform(0, np.pi)]),
                   center=[xy[0], xy[1], GROUND_Z])
        meshes.append(box)
    while len(meshes) < 5 + num_boxes + num_poles:
        xy = rng.uniform(-30, 30, 2)
        if not far_from_path(xy):
            continue
        pole = o3d.geometry.TriangleMesh.create_cylinder(radius=0.15, height=4.0)
        pole.translate([xy[0], xy[1], GROUND_Z + 2.0])
        meshes.append(pole)

    scene = o3d.t.geometry.RaycastingScene()
    for mesh in meshes:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    intensity = rng.integers(2000, 60000, len(meshes)).astype(np.uint16)
    return scene, intensity


def simulate_scan(scene, intensity, model, T_s_r, traj_T_r_M, stamp_s, t0_ns, rng, range_noise):
    """One motion-distorted scan starting at stamp_s [s since the trajectory start].
    Returns the scan dict (bag_io.iter_scans format) and the ground truth:
        xyz_true    : (H, W, 3) noise-free points in the sensor frame at the stamp, NaN = no hit
        col_T_sc_s  : (W, 4, 4) true T_{s(t_c), s(stamp)}
    """
    H, W = model.H, model.W
    rows = np.arange(H)[:, None]
    meas = np.arange(W)[None, :]                      # measurement column (firing order)
    col_times = np.arange(W) / W * model.scan_period  # [s since stamp]

    # beam origin and direction in the sensor frame for every (row, measurement column)
    encoder = 2 * np.pi * (1.0 - meas / W) * np.ones((H, 1))
    beam_azimuth = encoder - model.az[rows]
    altitude = model.alt[rows] * np.ones((1, W))
    origin_s = np.stack([model.n * np.cos(encoder), model.n * np.sin(encoder), np.zeros((H, W))], -1)
    dir_s = np.stack([np.cos(beam_azimuth) * np.cos(altitude), np.sin(beam_azimuth) * np.cos(altitude),
                      np.sin(altitude)], -1)

    # sensor pose of every column: T_M_s(t_c) = inv(T_s_r @ T_r_M(t_c))
    T_s_M_col = T_s_r @ traj_T_r_M[pose_index(stamp_s + col_times)]
    T_M_s_col = np.linalg.inv(T_s_M_col)
    origin_M = np.einsum("wij,hwj->hwi", T_M_s_col[:, :3, :3], origin_s) + T_M_s_col[None, :, :3, 3]
    dir_M = np.einsum("wij,hwj->hwi", T_M_s_col[:, :3, :3], dir_s)

    rays = o3d.core.Tensor(np.concatenate([origin_M, dir_M], -1).reshape(-1, 6).astype(np.float32))
    hits = scene.cast_rays(rays)
    distance = hits["t_hit"].numpy().astype(np.float64).reshape(H, W)
    geometry = hits["geometry_ids"].numpy().reshape(H, W)
    hit = np.isfinite(distance) & (distance < MAX_RANGE)

    # measured point in the sensor frame at its column time (range noise along the beam)
    noisy = distance + rng.normal(0.0, range_noise, (H, W))
    xyz_meas = origin_s + np.where(hit, noisy, np.nan)[..., None] * dir_s

    # ground truth: noise-free world point expressed in the sensor frame at the stamp
    T_s_M_stamp = T_s_r @ traj_T_r_M[pose_index(stamp_s)]
    X_M = origin_M + np.where(hit, distance, np.nan)[..., None] * dir_M
    xyz_true_meas = X_M @ T_s_M_stamp[:3, :3].T + T_s_M_stamp[:3, 3]

    # destagger: measurement column m of row i goes to image column (m + shift_i) mod W
    image_col = np.mod(meas + model.shift[rows], W)
    xyz = np.full((H, W, 3), np.nan)
    xyz_true = np.full((H, W, 3), np.nan)
    signal = np.zeros((H, W), dtype=np.uint16)
    row_idx = rows * np.ones((1, W), dtype=np.int64)
    xyz[row_idx, image_col] = xyz_meas
    xyz_true[row_idx, image_col] = xyz_true_meas
    signal[row_idx, image_col] = np.where(hit, intensity[np.where(hit, geometry, 0)], 0)

    valid = np.isfinite(xyz).all(-1)
    rng_img = np.zeros((H, W), dtype=np.float32)
    rng_img[valid] = np.linalg.norm(xyz[valid], axis=-1)
    scan = dict(stamp_ns=t0_ns + int(round(stamp_s * 1e9)), xyz=xyz, valid=valid, range=rng_img,
                col_times=col_times, signal=signal, reflec=signal.copy())

    T_M_s_stamp = np.linalg.inv(T_s_M_stamp)
    col_T_sc_s = T_s_M_col @ T_M_s_stamp
    return scan, dict(xyz_true=xyz_true, col_T_sc_s=col_T_sc_s, T_s_M=T_s_M_stamp)
