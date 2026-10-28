"""
Visualize the results of run_odometry.py:
  (1) the odometry trajectory           -> trajectory.png
  (2) the map rebuilt from the odometry -> map.ply, map_top.png (+ interactive 3-D window)
  (3) two frames aligned by the odometry -> pair_<A>_<B>.ply, pair_<A>_<B>_top.png (+ 3-D window)

The map frame m is the robot frame at the end of the first processed scan. A frame's undistorted
points (sensor frame at its stamp) go into the map frame with p_m = inv(T_s_m) @ p_s, and frame B
goes into frame A with p_A = T_AB @ p_B, T_AB = T_s_m(A) @ inv(T_s_m(B)).

Run:  python visualize_odometry.py      (edit the parameters below)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import config
from make_pairs import icp_points, load_frame, pose_diff, run_icp
from make_pairs import cfg as pairs_cfg

######## Parameters (edit by hand) ########
ODOMETRY_DIR = config.ODOMETRY_DIR   # folder set in config.py
OUT_DIR = ODOMETRY_DIR / "vis"
SHOW = True              # open interactive windows (False: only save the files)
MIN_RANGE = 3.0          # ignore closer points [m]: the robot's own body, and the nearby ground whose
                         # reflectivity reads ~3x too bright (leaves bright arcs along the path in the map)

# (2) map
MAP_EVERY = 5            # use every N-th frame
MAP_VOXEL = 0.1          # voxel size of the map [m]
MAP_MAX_RANGE = 40.0     # ignore points further from the sensor [m]
MAP_COLOR = "signal"     # "signal" (intensity image), "reflec" (reflectivity image) or "height"
MAP_CMAP = "turbo"       # matplotlib colour map, e.g. "turbo", "gray", "viridis"

# (3) two frames: indices into the sorted list of frames
FRAME_A = 0
FRAME_B = 100
REFINE_WITH_ICP = False  # refine the odometry alignment with ICP (as make_pairs.py does)
PAIR_MAX_RANGE = 40.0


def frame_files(odometry_dir):
    return sorted((Path(odometry_dir) / "frames").glob("*.npz"), key=lambda f: int(f.stem))


def transform(T, xyz):
    return xyz @ T[:3, :3].T + T[:3, 3]


def frame_points(frame, max_range, image="signal"):
    """Undistorted points (sensor frame at the stamp) between MIN_RANGE and max_range, and their
    value in the given image ("signal" or "reflec")."""
    keep = frame["valid"] & (frame["range"] > MIN_RANGE) & (frame["range"] < max_range)
    value = frame[image][keep].astype(np.float64)
    return frame["xyz_undist"][keep].astype(np.float64), value


# ---------------- (1) trajectory ----------------
def plot_trajectory(odometry_dir, out_dir):
    rows = np.loadtxt(Path(odometry_dir) / "trajectory.csv", delimiter=",", ndmin=2)
    t = (rows[:, 0] - rows[0, 0]) * 1e-9
    xyz = rows[:, 3:6]
    success = rows[:, 1] > 0
    length = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axes[0]
    path = ax.scatter(xyz[:, 0], xyz[:, 1], c=t, s=4, cmap="viridis")
    ax.plot(*xyz[0, :2], "go", ms=9, label="start")
    ax.plot(*xyz[-1, :2], "rs", ms=9, label="end")
    if (~success).any():
        ax.plot(xyz[~success, 0], xyz[~success, 1], "rx", ms=8, label="odometry failed")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"trajectory in the map frame: {len(rows)} frames, {length:.1f} m, {t[-1]:.1f} s")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.colorbar(path, ax=ax, label="time [s]")

    ax = axes[1]
    ax.plot(t, rows[:, 6], label="speed [m/s]")
    ax.plot(t, rows[:, 7], label="yaw rate [rad/s]")
    ax.plot(t, xyz[:, 2], label="z [m]")
    ax.set_xlabel("time [s]")
    ax.set_title("speed, yaw rate and height")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "trajectory.png", dpi=120)
    print(f"(1) trajectory: {len(rows)} frames, {length:.1f} m, failed {int((~success).sum())} "
          f"-> {out_dir / 'trajectory.png'}")
    return xyz


# ---------------- (2) map ----------------
def build_map(files):
    points, colors = [], []
    for f in files[::MAP_EVERY]:
        frame = load_frame(f)
        image = "reflec" if MAP_COLOR == "reflec" else "signal"
        xyz_s, value = frame_points(frame, MAP_MAX_RANGE, image)
        xyz_m = transform(np.linalg.inv(frame["T_s_m"]), xyz_s)
        # downsample each frame right away to keep the memory small (the image value rides along as a colour)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_m))
        pcd.colors = o3d.utility.Vector3dVector(np.repeat(value[:, None], 3, axis=1))
        pcd = pcd.voxel_down_sample(MAP_VOXEL)
        points.append(np.asarray(pcd.points))
        colors.append(np.asarray(pcd.colors)[:, 0])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(points)))
    pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.concatenate(colors)[:, None], 3, axis=1))
    pcd = pcd.voxel_down_sample(MAP_VOXEL)

    xyz = np.asarray(pcd.points)
    if MAP_COLOR == "height":
        value = xyz[:, 2]
    else:
        value = np.asarray(pcd.colors)[:, 0]
    lo, hi = np.percentile(value, [2, 98])
    pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap(MAP_CMAP)(np.clip((value - lo) / (hi - lo), 0, 1))[:, :3])
    return pcd


def trajectory_lineset(xyz):
    lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector(xyz),
                                 o3d.utility.Vector2iVector(np.stack([np.arange(len(xyz) - 1),
                                                                      np.arange(1, len(xyz))], 1)))
    lines.paint_uniform_color([1.0, 0.0, 0.0])
    return lines


def plot_top_view(ax, xyz, colors, title, max_points=400_000, seed=0):
    if len(xyz) > max_points:
        pick = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)
        xyz, colors = xyz[pick], colors[pick]
    order = np.argsort(xyz[:, 2])  # draw low points first
    ax.scatter(xyz[order, 0], xyz[order, 1], c=colors[order], s=0.2, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)


def show_map(files, trajectory_xyz, out_dir):
    pcd = build_map(files)
    o3d.io.write_point_cloud(str(out_dir / "map.ply"), pcd)

    fig, ax = plt.subplots(figsize=(12, 12))
    plot_top_view(ax, np.asarray(pcd.points), np.asarray(pcd.colors),
                  f"map (top view), every {MAP_EVERY}th frame, colour = {MAP_COLOR}")
    ax.plot(trajectory_xyz[:, 0], trajectory_xyz[:, 1], "r-", lw=1, label="trajectory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "map_top.png", dpi=150)
    print(f"(2) map: {len(pcd.points)} points from {len(files[::MAP_EVERY])} frames "
          f"-> {out_dir / 'map.ply'}, {out_dir / 'map_top.png'}")
    if SHOW:
        o3d.visualization.draw_geometries([pcd, trajectory_lineset(trajectory_xyz)], window_name="odometry map")


# ---------------- (3) two frames ----------------
def show_pair(files, out_dir):
    fa, fb = load_frame(files[FRAME_A]), load_frame(files[FRAME_B])
    xyz_a, _ = frame_points(fa, PAIR_MAX_RANGE)
    xyz_b, _ = frame_points(fb, PAIR_MAX_RANGE)

    T_AB = fa["T_s_m"] @ np.linalg.inv(fb["T_s_m"])
    dt = (fb["stamp_ns"] - fa["stamp_ns"]) * 1e-9
    print(f"(3) frames {FRAME_A} -> {FRAME_B} ({dt:.1f} s apart): odometry |t_AB| = "
          f"{np.linalg.norm(T_AB[:3, 3]):.2f} m")
    if REFINE_WITH_ICP:
        T_icp, fitness, rmse = run_icp(icp_points(fb, pairs_cfg), icp_points(fa, pairs_cfg), T_AB, pairs_cfg.icp_scales)
        d_trans, d_rot = pose_diff(T_icp, T_AB)
        print(f"    ICP refinement: fitness {fitness:.2f}, rmse {rmse:.3f} m, "
              f"moved {d_trans * 100:.1f} cm / {d_rot:.2f} deg from the odometry")
        T_AB = T_icp
    xyz_b_in_a = transform(T_AB, xyz_b)

    color_a, color_b = np.array([1.0, 0.55, 0.0]), np.array([0.0, 0.45, 1.0])
    pcd_a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_a))
    pcd_a.paint_uniform_color(color_a)
    pcd_b = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_b_in_a))
    pcd_b.paint_uniform_color(color_b)
    name = f"pair_{FRAME_A}_{FRAME_B}"
    o3d.io.write_point_cloud(str(out_dir / f"{name}.ply"), pcd_a + pcd_b)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax, xyz_b_shown, title in [(axes[0], xyz_b, "not aligned (both in their own sensor frame)"),
                                   (axes[1], xyz_b_in_a, "aligned with the odometry" +
                                    (" + ICP" if REFINE_WITH_ICP else ""))]:
        xyz = np.concatenate([xyz_a, xyz_b_shown])
        colors = np.concatenate([np.tile(color_a, (len(xyz_a), 1)), np.tile(color_b, (len(xyz_b_shown), 1))])
        plot_top_view(ax, xyz, colors, f"{title}\norange = frame {FRAME_A}, blue = frame {FRAME_B}")
        ax.set_xlim(-PAIR_MAX_RANGE / 2, PAIR_MAX_RANGE / 2)
        ax.set_ylim(-PAIR_MAX_RANGE / 2, PAIR_MAX_RANGE / 2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_top.png", dpi=120)
    print(f"    -> {out_dir / (name + '.ply')}, {out_dir / (name + '_top.png')}")
    if SHOW:
        o3d.visualization.draw_geometries([pcd_a, pcd_b], window_name=f"frames {FRAME_A} (orange) and {FRAME_B} (blue)")


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = frame_files(ODOMETRY_DIR)
    print(f"{len(files)} frames in {ODOMETRY_DIR}")

    trajectory_xyz = plot_trajectory(ODOMETRY_DIR, out_dir)
    show_map(files, trajectory_xyz, out_dir)
    show_pair(files, out_dir)
    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
