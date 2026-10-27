"""
Match two arbitrary odometry frames (make_pairs.match_frames) and visualize the correspondences.
Frames are given by their index in the sorted list of saved frames (= row in trajectory.csv), so
frames far apart or seen from very different headings can be matched; if the two frames are too
far apart, the match fails with a notice.

3-D view (PLOT_3D):
  The two point clouds, z up, with B shifted by B_OFFSET so the two sit side by side. B is either
  aligned to A with T_AB, or left in its own frame (ALIGN_B). Points with a correspondence are dark
  gray, points without one are light gray.
  -> Open3D window + <OUT_DIR>/match3d_<A>_<B>.ply / .png

2-D views (PLOT_2D), the ways commonly used for dense matching / optical flow:
  1. images A and B
  2. B warped into A through corr_AB (as in DKM / RoMa): looks like A where the matches are right
  3. overlay of A (yellow) and warped B (blue): aligned structure turns gray
  4. flow colour coding (as in optical flow, e.g. RAFT): hue = direction, brightness = magnitude
  5. sparse match lines between A and B (as in LoFTR / SuperGlue)
  6. coordinate colour transfer: A painted by pixel position, each B pixel painted with the colour
     of its match in A -> the same surface gets the same colour in both images
  -> <OUT_DIR>/match2d_<A>_<B>.png

Run:  python visualize_pairs.py      (edit the parameters below)
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

import config
import make_pairs

# cv2 points Qt to its own plugins, which breaks matplotlib's Qt backend
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import hsv_to_rgb  # noqa: E402

######## Parameters (edit by hand) ########
ODOMETRY_DIR = config.ODOMETRY_DIR   # folder set in config.py
OUT_DIR = config.PAIRS_DIR / "vis"
FRAME_A = 1000             # frame indices (rows of trajectory.csv); any two frames
FRAME_B = 1030
SHOW = True              # open windows (False: only save the files)
PLOT_3D = True
PLOT_2D = True
# matching parameters (ICP, thresholds, ...) are taken from make_pairs.cfg

# 3-D view
ALIGN_B = True                # True: B moved into A's frame with T_AB; False: B in its own frame
B_OFFSET = [0.0, 0.0, 15.0]   # display shift of cloud B [m]; [0, 0, 0] with ALIGN_B overlays the two clouds
MAX_RANGE_3D = 30.0           # show points up to this range [m]
MATCHED_GRAY = 0.25           # gray level of points with a correspondence (0 = black, 1 = white)
UNMATCHED_GRAY = 0.8          # gray level of points without one

# 2-D views
IMAGE = "signal"         # "signal" or "reflec"
ROW_SCALE = 3            # stretch the rows (images are 128 x 1024) for readability
NUM_LINES = 150          # number of match lines in panel 5


# ---------------- 3-D view ----------------
def oblique_projection(points, azim_deg=-60.0, elev_deg=25.0):
    """Orthographic view of (N, 3) points (z up), turned by azim around z and looking down by elev.
    Returns the screen x, screen y and the depth (larger = further away)."""
    a, e = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    R_z = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    R_x = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    q = points @ (R_x @ R_z).T
    return q[:, 0], q[:, 2], q[:, 1]


def plot_3d(fa, fb, pair, name):
    meta = json.loads((Path(ODOMETRY_DIR) / "meta.json").read_text())
    T_r_s = np.linalg.inv(np.array(meta["T_s_r"]))  # sensor -> robot frame (z up)

    # both clouds in a robot frame: A in its own, B in A's (ALIGN_B) or in its own; B then shifted
    T_b_to_display = T_r_s @ pair["T_AB"] if ALIGN_B else T_r_s
    pts_a = fa["xyz_undist"].astype(np.float64) @ T_r_s[:3, :3].T + T_r_s[:3, 3]
    pts_b = fb["xyz_undist"].astype(np.float64) @ T_b_to_display[:3, :3].T + T_b_to_display[:3, 3]
    pts_b = pts_b + np.asarray(B_OFFSET)

    matched_a = np.isfinite(pair["corr_AB"][..., 0])
    matched_b = np.isfinite(pair["corr_BA"][..., 0])
    show_a = fa["valid"] & (fa["range"] < MAX_RANGE_3D)
    show_b = fb["valid"] & (fb["range"] < MAX_RANGE_3D)

    points = np.concatenate([pts_a[show_a], pts_b[show_b]])
    matched = np.concatenate([matched_a[show_a], matched_b[show_b]])
    gray = np.where(matched, MATCHED_GRAY, UNMATCHED_GRAY)
    colors = np.repeat(gray[:, None], 3, axis=1)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(Path(OUT_DIR) / f"match3d_{name}.ply"), pcd)

    if SHOW:
        o3d.visualization.draw_geometries([pcd], window_name=f"frames {name}: dark = matched, light = not matched")


# ---------------- 2-D views ----------------
def to_gray(img16):
    """uint16 image -> float in [0, 1], contrast-stretched."""
    img = img16.astype(np.float64)
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1), 0, 1)


def stretch(img):
    """Repeat the rows so the 128-row images are easier to see."""
    return np.repeat(img, ROW_SCALE, axis=0)


def warp_to_a(img_b, corr_ab):
    """Sample image B at the match of every pixel of A (bilinear, wrapping in azimuth). NaN = no match."""
    has_match = np.isfinite(corr_ab[..., 0])
    map_row = np.where(has_match, corr_ab[..., 0], 0).astype(np.float32)
    map_col = np.where(has_match, corr_ab[..., 1], 0).astype(np.float32)
    warped = cv2.remap(img_b.astype(np.float32), map_col, map_row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return np.where(has_match, warped, np.nan)


def flow_ab(corr_ab):
    """Displacement (d_row, d_col) of every pixel of A, with the column difference wrapped."""
    H, W = corr_ab.shape[:2]
    rows, cols = np.mgrid[0:H, 0:W]
    d_row = corr_ab[..., 0] - rows
    d_col = (corr_ab[..., 1] - cols + W / 2) % W - W / 2
    return d_row, d_col


def flow_to_color(d_row, d_col, max_mag):
    """Optical-flow colour coding: hue = direction, brightness = magnitude (black = no match)."""
    angle = np.arctan2(d_row, d_col)                   # image coordinates: row down, col right
    mag = np.hypot(d_row, d_col)
    hsv = np.stack([(angle + np.pi) / (2 * np.pi), np.ones_like(mag), np.clip(mag / max_mag, 0, 1)], -1)
    rgb = hsv_to_rgb(np.nan_to_num(hsv))
    rgb[~np.isfinite(mag)] = 0
    return rgb


def flow_wheel(size=101):
    """Colour wheel legend for flow_to_color (max magnitude at the rim)."""
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    rgb = flow_to_color(y, x, 1.0)
    rgb[np.hypot(x, y) > 1] = 1
    return rgb


def position_colors(rows, cols, H, W):
    """Smooth colour for (fractional) pixel coordinates: hue = column (cyclic, like the azimuth),
    brightness = row."""
    hsv = np.stack([cols / W, np.full_like(cols, 0.9), 0.35 + 0.65 * (1 - rows / (H - 1))], -1)
    return hsv_to_rgb(np.clip(hsv, 0, 1))


def show_image(ax, img, title, cmap=None):
    ax.imshow(stretch(img), cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_2d(pair, name):
    img_a = pair[f"{IMAGE}_A"]
    img_b = pair[f"{IMAGE}_B"]
    corr_ab, corr_ba = pair["corr_AB"], pair["corr_BA"]
    H, W = img_a.shape
    if make_pairs.cfg.rotate180:  # show upright (upside-down mount), as in the saved pairs
        img_a, img_b = make_pairs.rotate180(img_a), make_pairs.rotate180(img_b)
        corr_ab = make_pairs.rotate180_corr(corr_ab, H, W)
        corr_ba = make_pairs.rotate180_corr(corr_ba, H, W)
    img_a, img_b = to_gray(img_a), to_gray(img_b)
    has_match_a = np.isfinite(corr_ab[..., 0])
    has_match_b = np.isfinite(corr_ba[..., 0])
    d_row, d_col = flow_ab(corr_ab)
    mag = np.hypot(d_row, d_col)
    print(f"2-D: flow median {np.nanmedian(mag):.1f} px, 95% {np.nanpercentile(mag, 95):.1f} px")

    fig, axes = plt.subplots(8, 1, figsize=(18, 30), gridspec_kw=dict(height_ratios=[1, 1, 1, 1, 1, 2.1, 1, 1]))
    fig.suptitle(f"frames {name.replace('_', ' and ')}: {100 * has_match_a.mean():.0f}% of A matched "
                 f"(black = no correspondence)", fontsize=14)

    # 1. the two images
    show_image(axes[0], img_a, f"1. image A ({IMAGE})", cmap="gray")
    show_image(axes[1], img_b, f"   image B ({IMAGE})", cmap="gray")

    # 2. B warped into A
    warped = warp_to_a(img_b, corr_ab)
    show_image(axes[2], np.nan_to_num(warped), "2. B warped into A through corr_AB (should look like A)", cmap="gray")

    # 3. overlay A (yellow = red + green) and warped B (blue)
    overlay = np.stack([img_a, img_a, np.nan_to_num(warped)], -1)
    show_image(axes[3], overlay, "3. overlay: A (yellow) + warped B (blue) -> aligned structure is gray")

    # 4. flow colour coding, with a colour wheel legend
    max_mag = np.nanpercentile(mag, 95)
    show_image(axes[4], flow_to_color(d_row, d_col, max_mag),
               f"4. flow A -> B: hue = direction, brightness = magnitude (rim of the wheel = {max_mag:.0f} px)")
    wheel_ax = axes[4].inset_axes([1.005, 0.0, 0.06, 1.0])
    wheel_ax.imshow(flow_wheel())
    wheel_ax.axis("off")

    # 5. sparse match lines between A (top) and B (bottom), spread evenly over the image
    gap = 20
    stacked = np.vstack([stretch(img_a), np.ones((gap, W)), stretch(img_b)])
    axes[5].imshow(stacked, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    rows, cols = np.nonzero(has_match_a)
    cell = (H * W / max(NUM_LINES, 1) / 8) ** 0.5                 # roughly one match per grid cell
    cell_id = (rows // max(cell / 8, 1)).astype(int) * 10_000 + (cols // max(cell, 1)).astype(int)
    _, first = np.unique(cell_id, return_index=True)
    rng = np.random.default_rng(0)
    pick = rng.choice(first, min(NUM_LINES, len(first)), replace=False)
    r1, c1 = corr_ab[rows[pick], cols[pick], 0], corr_ab[rows[pick], cols[pick], 1]
    colors = position_colors(rows[pick].astype(float), cols[pick].astype(float), H, W)
    y_a = (rows[pick] + 0.5) * ROW_SCALE
    y_b = (r1 + 0.5) * ROW_SCALE + H * ROW_SCALE + gap
    no_seam = np.abs(c1 - cols[pick]) < W / 2   # do not draw lines across the azimuth seam
    for i in np.nonzero(no_seam)[0]:
        axes[5].plot([cols[pick][i], c1[i]], [y_a[i], y_b[i]], "-", color=colors[i], lw=0.8)
    axes[5].scatter(cols[pick], y_a, s=6, c=colors)
    axes[5].scatter(c1, y_b, s=6, c=colors)
    axes[5].set_title(f"5. {no_seam.sum()} sampled matches: A (top) -> B (bottom)", fontsize=11)
    axes[5].set_xticks([])
    axes[5].set_yticks([])

    # 6. coordinate colour transfer: same colour = same surface point
    shade_a = 0.35 + 0.65 * img_a[..., None]
    shade_b = 0.35 + 0.65 * img_b[..., None]
    rows_a, cols_a = np.mgrid[0:H, 0:W].astype(float)
    color_a = position_colors(rows_a, cols_a, H, W) * shade_a
    color_a[~has_match_a] = img_a[~has_match_a, None] * 0.5
    color_b = position_colors(np.nan_to_num(corr_ba[..., 0]), np.nan_to_num(corr_ba[..., 1]), H, W) * shade_b
    color_b[~has_match_b] = img_b[~has_match_b, None] * 0.5
    show_image(axes[6], color_a, "6. A painted by pixel position (hue = column, brightness = row)")
    show_image(axes[7], color_b, "   B painted with the colour of its match in A (corr_BA) -> same colour = same point")

    fig.tight_layout(rect=[0, 0, 0.95, 0.98])
    out_path = Path(OUT_DIR) / f"match2d_{name}.png"
    fig.savefig(out_path, dpi=100)
    print(f"2-D views -> {out_path}")


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    files = make_pairs.list_frames(ODOMETRY_DIR)
    model = make_pairs.load_model(ODOMETRY_DIR)
    fa = make_pairs.load_frame(files[FRAME_A])
    fb = make_pairs.load_frame(files[FRAME_B])

    pair, message = make_pairs.match_frames(fa, fb, model, make_pairs.cfg)
    dt = (fb["stamp_ns"] - fa["stamp_ns"]) * 1e-9
    print(f"frames {FRAME_A} -> {FRAME_B} ({dt:.1f} s apart): {message}")
    if pair is None:
        return

    name = f"{FRAME_A}_{FRAME_B}"
    if PLOT_3D:
        plot_3d(fa, fb, pair, name)
    if PLOT_2D:
        plot_2d(pair, name)
    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
