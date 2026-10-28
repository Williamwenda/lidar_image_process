"""
Match two arbitrary odometry frames (make_pairs.match_frames) and inspect the correspondences.
Frames are given by their index in the sorted list of saved frames (= row in trajectory.csv), so
frames far apart or seen from very different headings can be matched; if the two frames are too
far apart, the match fails with a notice.

Correspondence inspector (2-D):
  Image A on top, image B below, shown upright (rotated 180 deg as in the saved pairs if
  make_pairs.cfg.rotate180). Pixels without a correspondence are tinted blue.
  * Click a pixel in A (or in B): a line is drawn to its sub-pixel match in the other image, and the
    panels below show the local patches around both points (+ = clicked pixel, x = sub-pixel match)
    and the numbers of the match: sub-pixel coordinates, flow, ranges, forward-backward (cycle)
    error, and the 3-D distance to the point measured by the other scan at the matched pixel.
  * For a pixel without a correspondence, the matching is recomputed for that pixel to tell why
    (too close, seam gap, outside the FOV, no return, occluded / range check, cycle check); where
    the point was predicted to land, it is drawn as a red o with a dashed line.
  * Keys: c = clear the lines, s = save the figure to <OUT_DIR>/inspect_<A>_<B>.png (nothing is
    saved otherwise).
  * Matches listed in PIXELS_A (and NUM_RANDOM random ones) are drawn at the start and printed.

3-D view (PLOT_3D):
  The two point clouds, z up, with B shifted by B_OFFSET so the two sit side by side. B is either
  aligned to A with T_AB, or left in its own frame (ALIGN_B). Points with a correspondence are dark
  gray, points without one are light gray (Open3D window).

Run:  python visualize_pairs.py      (edit the parameters below)
"""
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.patches import ConnectionPatch

import config
import make_pairs

######## Parameters (edit by hand) ########
ODOMETRY_DIR = config.ODOMETRY_DIR   # folder set in config.py
OUT_DIR = config.ODOMETRY_DIR / "vis"   # where the s key saves the inspector figure
FRAME_A = 530           # frame indices (rows of trajectory.csv); any two frames
FRAME_B = 532
PLOT_2D = True           # correspondence inspector
PLOT_3D = False          # Open3D window (blocks until closed, then the inspector opens)
# matching parameters (ICP, thresholds, ...) are taken from make_pairs.cfg

# inspector
IMAGE = "signal"         # "signal" or "reflec"
PIXELS_A = []            # (row, col) pixels of A to draw at the start, in the displayed (upright)
                         # image, e.g. [(64, 200), (90, 800)]
NUM_RANDOM = 5           # also draw this many random matched pixels of A
PATCH_ROWS = 8           # patch half-size [px]: rows ...
PATCH_COLS = 24          # ... and columns
FIG_SIZE = (18, 10)

# 3-D view
ALIGN_B = True                # True: B moved into A's frame with T_AB; False: B in its own frame
B_OFFSET = [0.0, 0.0, 15.0]   # display shift of cloud B [m]; [0, 0, 0] with ALIGN_B overlays the two clouds
MAX_RANGE_3D = 30.0           # show points up to this range [m]
MATCHED_GRAY = 0.25           # gray level of points with a correspondence (0 = black, 1 = white)
UNMATCHED_GRAY = 0.8          # gray level of points without one


# ---------------- 3-D view ----------------
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
    o3d.visualization.draw_geometries([pcd], window_name=f"frames {name}: dark = matched, light = not matched")


# ---------------- correspondence inspector ----------------
def to_gray(img16):
    """uint16 image -> float in [0, 1], contrast-stretched."""
    img = img16.astype(np.float64)
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1), 0, 1)


def display_rgb(gray, matched):
    """Gray image with the pixels that have no correspondence tinted blue."""
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    rgb[~matched] = 0.5 * rgb[~matched] + [0.0, 0.1, 0.35]
    return rgb


def wrap_col(d_col, W):
    """Column difference wrapped to [-W/2, W/2) (the azimuth is periodic)."""
    return (d_col + W / 2) % W - W / 2


def build_view(fa, fb, pair):
    """Everything the inspector looks up, per scan, in the displayed (upright) pixel layout, plus
    the raw (unrotated) frame and correspondences for recomputing a rejected match."""
    H, W = pair["corr_AB"].shape[:2]
    rot = make_pairs.cfg.rotate180
    image = (lambda img: make_pairs.rotate180(img)) if rot else (lambda img: img)
    corr = (lambda c: make_pairs.rotate180_corr(c, H, W)) if rot else (lambda c: c)
    T_AB = pair["T_AB"]
    view = {}
    for s, o, frame, T_other_s in (("A", "B", fa, np.linalg.inv(T_AB)), ("B", "A", fb, T_AB)):
        view[s] = dict(
            gray=to_gray(image(pair[f"{IMAGE}_{s}"])),
            range=image(pair[f"range_{s}"]),
            xyz=image(frame["xyz_undist"]),      # undistorted point, sensor frame at the stamp
            corr=corr(pair[f"corr_{s}{o}"]),     # sub-pixel (row, col) of the match in the other scan
            corr_range=image(pair[f"corr_range_{s}{o}"]),
            T_other_s=T_other_s,                 # this scan's stamp frame -> the other's
            frame=frame,                         # raw layout
        )
    return view, H, W


def flip(row, col, H, W):
    """Displayed <-> raw pixel coordinates (the 180 deg rotation is its own inverse)."""
    if not make_pairs.cfg.rotate180:
        return row, col
    return H - 1 - row, (W - 1 - col) % W


def explain(view, model, src, row, col, H, W):
    """Why pixel (row, col) of scan src has no correspondence. Returns (text lines, predicted
    (row, col) in the other scan in display coordinates, or None)."""
    dst = "B" if src == "A" else "A"
    s, d = view[src], view[dst]
    r_raw, c_raw = flip(row, col, H, W)
    reason, pred = make_pairs.explain_no_match(s["frame"], d["frame"], s["T_other_s"], r_raw, c_raw,
                                               model, make_pairs.cfg, names=(src, dst))
    lines = ["", f"NO CORRESPONDENCE in {dst}. Reason:"] + textwrap.wrap(reason, 64, initial_indent="  ",
                                                                         subsequent_indent="  ")
    if pred is None:
        return lines, None
    r_p, c_p = flip(pred[0], pred[1], H, W)
    lines += ["", f"predicted in {dst} (row, col) = ({r_p:.2f}, {c_p:.2f})   (red o)"]
    if -0.5 < r_p < H - 0.5:
        r_n, c_n = int(np.rint(r_p)), int(np.rint(c_p)) % W
        lines.append(f"range in {dst}   predicted {pred[2]:.2f} m,  measured {d['range'][r_n, c_n]:.2f} m at ({r_n}, {c_n})")
    else:
        lines.append(f"range in {dst}   predicted {pred[2]:.2f} m")
    return lines, (r_p, c_p)


def describe_match(view, model, src, row, col, H, W):
    """Numbers of the match of pixel (row, col) of scan src. Returns (match (row, col) or None,
    rejected prediction (row, col) or None, text)."""
    dst = "B" if src == "A" else "A"
    s, d = view[src], view[dst]
    lines = [f"{src} pixel   (row, col) = ({row}, {col})", f"range {src}       {s['range'][row, col]:.2f} m"]
    r_m, c_m = s["corr"][row, col]
    if not np.isfinite(r_m):
        more, pred = explain(view, model, src, row, col, H, W)
        return None, pred, "\n".join(lines + more)

    d_row, d_col = r_m - row, wrap_col(c_m - col, W)
    # nearest pixel of the match in the other scan, and the way back from there (cycle check)
    r_n, c_n = int(np.clip(np.rint(r_m), 0, H - 1)), int(np.rint(c_m)) % W
    r_back, c_back = d["corr"][r_n, c_n]
    cycle = np.hypot(r_back - row, wrap_col(c_back - col, W)) if np.isfinite(r_back) else np.nan
    # 3-D: this point moved into the other scan vs the point the other scan measured at that pixel
    T = s["T_other_s"]
    q = T[:3, :3] @ s["xyz"][row, col] + T[:3, 3]
    dist_3d = np.linalg.norm(q - d["xyz"][r_n, c_n]) if d["range"][r_n, c_n] > 0 else np.nan
    lines += [
        f"match in {dst}   (row, col) = ({r_m:.2f}, {c_m:.2f})",
        f"flow          (d_row, d_col) = ({d_row:+.2f}, {d_col:+.2f}),  |flow| = {np.hypot(d_row, d_col):.2f} px",
        f"range in {dst}   predicted {s['corr_range'][row, col]:.2f} m,  measured {d['range'][r_n, c_n]:.2f} m "
        f"at ({r_n}, {c_n})",
        f"cycle error   {src} -> {dst} -> {src}: {cycle:.2f} px",
        f"3-D distance  to {dst}'s point at ({r_n}, {c_n}): {100 * dist_3d:.1f} cm",
        "              (includes up to one pixel of spacing)",
    ]
    return (r_m, c_m), None, "\n".join(lines)


def patch(gray, row, col, H, W):
    """Window of PATCH_ROWS x PATCH_COLS half-size around integer (row, col), wrapping in azimuth
    (NaN above / below the image). Returns the patch and its extent in unwrapped image coordinates."""
    rows = np.arange(row - PATCH_ROWS, row + PATCH_ROWS + 1)
    cols = np.arange(col - PATCH_COLS, col + PATCH_COLS + 1)
    out = np.full((len(rows), len(cols)), np.nan)
    inside = (rows >= 0) & (rows < H)
    out[inside] = gray[rows[inside]][:, cols % W]
    extent = (cols[0] - 0.5, cols[-1] + 0.5, rows[-1] + 0.5, rows[0] - 0.5)
    return out, extent


class Inspector:
    def __init__(self, view, model, H, W, title, out_path):
        self.view, self.model, self.H, self.W, self.out_path = view, model, H, W, out_path
        self.colors = plt.get_cmap("tab10").colors
        self.artists = []
        self.num_drawn = 0

        self.fig = plt.figure(figsize=FIG_SIZE)
        grid = self.fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1], width_ratios=[1, 1, 1.4])
        self.ax = {"A": self.fig.add_subplot(grid[0, :]), "B": self.fig.add_subplot(grid[1, :])}
        self.ax_patch = {"A": self.fig.add_subplot(grid[2, 0]), "B": self.fig.add_subplot(grid[2, 1])}
        self.ax_text = self.fig.add_subplot(grid[2, 2])
        self.fig.suptitle(title, fontsize=12)
        for s in ("A", "B"):
            matched = np.isfinite(view[s]["corr"][..., 0])
            self.ax[s].imshow(display_rgb(view[s]["gray"], matched), interpolation="nearest", aspect="auto")
            self.ax[s].set_title(f"image {s} ({IMAGE}), {100 * matched.mean():.0f}% matched "
                                 f"(blue = no correspondence) - click a pixel", fontsize=10)
            self.ax[s].set_ylabel("row")
        self.ax["B"].set_xlabel("column")
        for s in ("A", "B"):
            self.ax_patch[s].set_title(f"patch {s}", fontsize=10)
            self.ax_patch[s].set_xticks([])
            self.ax_patch[s].set_yticks([])
        self.ax_text.axis("off")
        self.show_text("Click a pixel in image A or B.\nKeys: c = clear the lines, s = save the figure.")
        self.fig.tight_layout(rect=[0, 0, 1, 0.97])
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def show_text(self, text):
        self.ax_text.clear()
        self.ax_text.axis("off")
        self.ax_text.text(0, 1, text, va="top", ha="left", family="monospace", fontsize=9,
                          transform=self.ax_text.transAxes)

    def inspect(self, src, row, col):
        """Draw the match of pixel (row, col) of scan src and show its patches and numbers."""
        dst = "B" if src == "A" else "A"
        match, pred, text = describe_match(self.view, self.model, src, row, col, self.H, self.W)
        color = self.colors[self.num_drawn % len(self.colors)]
        self.num_drawn += 1
        self.show_text(text)
        print(f"\n{text}")

        self.artists += self.ax[src].plot(col, row, "+", color=color, ms=12, mew=2)
        if match is not None:
            self.draw_link(src, dst, row, col, match, color, "x", "-")
        elif pred is not None and -0.5 < pred[0] < self.H - 0.5:
            self.draw_link(src, dst, row, col, pred, "red", "o", "--")

        # patches: + = the clicked pixel, x = the sub-pixel match, in image coordinates
        self.draw_patch(src, row, col, (row, col), "+", color)
        target, marker, marker_color = (match, "x", color) if match is not None else (pred, "o", "red")
        if target is not None:
            r_t, c_t = target
            r_c, c_c = int(np.clip(np.rint(r_t), 0, self.H - 1)), int(np.rint(c_t))
            self.draw_patch(dst, r_c, c_c, (r_t, c_c + wrap_col(c_t - c_c, self.W)), marker, marker_color,
                            "" if match is not None else " (rejected prediction)")
        else:
            self.ax_patch[dst].clear()
            self.ax_patch[dst].set_title(f"patch {dst}: no match", fontsize=10)
            self.ax_patch[dst].set_xticks([])
            self.ax_patch[dst].set_yticks([])
        self.fig.canvas.draw_idle()

    def draw_link(self, src, dst, row, col, target, color, marker, style):
        """Marker at target in dst and a line from (row, col) in src, as for sparse feature matches;
        no line across the azimuth seam."""
        r_t, c_t = target
        self.artists += self.ax[dst].plot(c_t, r_t, marker, color=color, ms=10, mew=2, mfc="none")
        if abs(c_t - col) < self.W / 2:
            line = ConnectionPatch(xyA=(col, row), coordsA=self.ax[src].transData, xyB=(c_t, r_t),
                                   coordsB=self.ax[dst].transData, color=color, lw=1.2, ls=style)
            self.fig.add_artist(line)
            self.artists.append(line)

    def draw_patch(self, scan, row, col, marker_rc, marker, color, note=""):
        ax = self.ax_patch[scan]
        ax.clear()
        img, extent = patch(self.view[scan]["gray"], row, col, self.H, self.W)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=extent, interpolation="nearest", aspect="auto")
        ax.plot(marker_rc[1], marker_rc[0], marker, color=color, ms=14, mew=2, mfc="none")
        ax.set_title(f"patch {scan} around ({row}, {col % self.W}){note}", fontsize=10)

    def on_click(self, event):
        if self.fig.canvas.toolbar is not None and self.fig.canvas.toolbar.mode:
            return  # zooming / panning
        for s in ("A", "B"):
            if event.inaxes is self.ax[s] and event.xdata is not None:
                row = int(np.clip(np.rint(event.ydata), 0, self.H - 1))
                col = int(np.rint(event.xdata)) % self.W
                self.inspect(s, row, col)

    def on_key(self, event):
        if event.key == "c":
            for artist in self.artists:
                artist.remove()
            self.artists, self.num_drawn = [], 0
            self.fig.canvas.draw_idle()
        elif event.key == "s":
            self.save()

    def save(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.out_path, dpi=100)
        print(f"inspector figure -> {self.out_path}")


def plot_2d(fa, fb, pair, model, name, message):
    view, H, W = build_view(fa, fb, pair)
    title = f"frames {FRAME_A} (A) and {FRAME_B} (B): {message}"
    inspector = Inspector(view, model, H, W, title, Path(OUT_DIR) / f"inspect_{name}.png")

    pixels = [(int(r), int(c)) for r, c in PIXELS_A]
    if NUM_RANDOM > 0:
        rows, cols = np.nonzero(np.isfinite(view["A"]["corr"][..., 0]))
        pick = np.random.default_rng(0).choice(len(rows), min(NUM_RANDOM, len(rows)), replace=False)
        pixels += list(zip(rows[pick], cols[pick]))
    for row, col in pixels:
        inspector.inspect("A", row, col)
    return inspector


def main():
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
        inspector = plot_2d(fa, fb, pair, model, name, message)  # keep a reference: the callbacks live on it
        plt.show()
        del inspector


if __name__ == "__main__":
    main()
