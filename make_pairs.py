"""
Step 3: build image pairs with ground-truth pixel correspondences from the frames saved by
run_odometry.py.

Any two frames can be matched (match_frames / match_frame_ids), e.g. frames seen from very
different positions or headings: the odometry pose chain gives the initial guess. Frames are
identified by their index in the sorted list of saved frames (= row in trajectory.csv).

For a pair (A, B):
  1. Initial guess from the odometry: T_AB = T_s_m(A) @ inv(T_s_m(B))   (p_A = T_AB @ p_B,
     both sensor frames at their scan stamps).
  2. Refine T_AB with point-to-plane ICP between the undistorted clouds (rigid: the motion
     distortion is already removed). The match fails (with a notice) if ICP fits badly (too little
     overlap), moves far from the guess (unreliable), or too few pixels get a correspondence.
  3. For every valid pixel of A: undistorted point -> B's stamp frame -> re-apply B's motion
     distortion (re_mdist; the column, and so the transform, depends on where the point lands,
     hence a fixed-point iteration) -> project with the Ouster model -> sub-pixel (row, col) in
     B's raw image. Keep the match if it is inside B's FOV, B has a return there with a matching
     range (occlusion check), and the B -> A match comes back within cycle_tol px.

Reads outputs/<FOLDER_NAME>/odometry and writes to <out_dir> = outputs/<FOLDER_NAME>/pairs (config.py).

Saved per pair (<out_dir>/pairs/<frame A>_<frame B>.npz, H x W = 128 x 1024):
    signal_A, signal_B, reflec_A, reflec_B : (H, W) uint16 raw images
    range_A, range_B           : (H, W) float32 raw range [m], 0 = no return
    corr_AB                    : (H, W, 2) float32 (row, col) in B for every pixel of A, NaN = no match
    corr_BA                    : (H, W, 2) float32 (row, col) in A for every pixel of B, NaN = no match
    corr_range_AB, corr_range_BA : (H, W) float32 range of the matched point in the other scan [m]
                                 (the GT is less accurate at short range; use it to filter / weight)
    T_AB, T_AB_odom            : (4, 4) refined and odometry-only relative pose
    frame_A, frame_B, stamp_A, stamp_B, icp_fitness, icp_rmse, rotated180
Columns are periodic: col is in [0, W), and col in (W-1, W) lies between the last and first column.

Run:  python make_pairs.py      (edit the parameters below)
"""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

import config
from motion_distortion import re_mdist
from ouster_model import OusterModel

######## Parameters (edit by hand) ########
cfg = SimpleNamespace(
    odometry_dir=config.ODOMETRY_DIR,   # folder set in config.py
    out_dir=config.PAIRS_DIR,

    # which pairs: frame indices into the sorted list of saved frames (= rows of trajectory.csv)
    frame_pairs=None,     # list of (frame A, frame B) to match, e.g. [(40, 286), (20, 309)];
                          # None = sequential pairs, set by the four parameters below
    frame_gap=10,         # pair frame k with frame k + frame_gap (10 frames = 1 s apart)
    stride=5,             # start a new pair every `stride` frames
    first_frame=10,       # skip the first frames while the odometry velocity settles
    max_pairs=None,       # stop after this many pairs (None = no limit)

    # ICP refinement: list of (voxel [m], max correspondence dist [m], iterations), coarse -> fine
    icp_scales=[(0.5, 1.0, 30), (0.25, 0.5, 30), (0.1, 0.2, 50)],
    icp_min_range=1.5,
    icp_max_range=80.0,
    min_fitness=0.2,                 # fail if the ICP fitness is lower (almost no overlap). Pairs 15-20 m
                                     # apart are fine at ~0.3; the ICP-vs-odometry check below is the main guard
    max_trans_diff_to_odom=0.3,      # fail if ICP moved further than this from the odometry guess [m]
    max_rot_diff_to_odom_deg=3.0,    # ... or rotated more than this [deg]
    min_matches=2000,                # fail if fewer pixels of A get a correspondence

    # correspondence validity
    corr_min_range=1.0,   # no correspondences for closer points (robot body; ambiguous projection) [m]
    re_mdist_iters=5,     # fixed-point iterations for the column of the re-distorted point
    seam_margin=50,       # points projecting this close to the first / last measurement column are
                          # also iterated from the other end of the scan [columns]
    range_abs_tol=0.10,   # [m]
    range_rel_tol=0.02,   # fraction of range
    cycle_tol=1.0,        # max A -> B -> A error [px], < 0 disables

    # output
    rotate180=True,       # rotate images + correspondences (upside-down mount on the Warthog)
    vis_every=10,         # write an overlay image every N pairs (0 = off)
)


# ---------------- loading ----------------
def list_frames(odometry_dir):
    """Saved frame files sorted by stamp; a frame's index in this list is its frame id."""
    return sorted((Path(odometry_dir) / "frames").glob("*.npz"), key=lambda f: int(f.stem))


def load_model(odometry_dir):
    meta = json.loads((Path(odometry_dir) / "meta.json").read_text())
    return OusterModel(meta["ouster_metadata"])


def load_frame(path):
    with np.load(path) as data:
        frame = {key: data[key] for key in data.files}
    frame["valid"] = frame["range"] > 0
    return frame


# ---------------- ICP ----------------
def to_o3d(xyz, voxel):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    return pcd.voxel_down_sample(voxel) if voxel > 0 else pcd


def icp_points(frame, cfg):
    xyz = frame["xyz_undist"][frame["valid"]].astype(np.float64)
    rng = np.linalg.norm(xyz, axis=1)
    return xyz[(rng > cfg.icp_min_range) & (rng < cfg.icp_max_range)]


def run_icp(src_xyz, tgt_xyz, T_init, scales):
    """Multi-scale point-to-plane ICP. Returns T (src -> tgt), fitness, rmse at the finest scale."""
    T = T_init.copy()
    result = None
    for voxel, max_dist, iters in scales:
        src = to_o3d(src_xyz, voxel)
        tgt = to_o3d(tgt_xyz, voxel)
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3.0 * voxel, max_nn=30))
        loss = o3d.pipelines.registration.TukeyLoss(k=max_dist)
        result = o3d.pipelines.registration.registration_icp(
            src, tgt, max_dist, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters),
        )
        T = np.asarray(result.transformation)
    return T, result.fitness, result.inlier_rmse


# ---------------- correspondences ----------------
def predict_in_dst(q, dst, model, start_cols, num_iters):
    """Where dst's (moving) sensor saw the points q (dst sensor frame at dst's stamp).
    Re-applies dst's motion distortion; which column (and so which transform) applies depends on
    where the point lands, so iterate from start_cols (N,): re_mdist -> project -> column -> ...
    Returns the sub-pixel (row, col), the predicted range, and |measured - predicted| range at
    that pixel (inf if outside the FOV, dst has no return there, or the column did not converge,
    e.g. a point in the gap between the end and the start of dst's sweep)."""
    H, W = model.H, model.W
    # a point on a pixel border may flip between two rows, whose measurement columns differ by up
    # to the shift spread (a few ms apart); a jump across the seam (~W columns, a full scan) is not
    # convergence, so the column difference is not taken modulo W
    col_tol = np.ptp(model.shift) + 1
    c = start_cols
    converged = np.ones(len(q), dtype=bool)
    for _ in range(num_iters):
        row, col = model.project(re_mdist(q, c, dst["col_T_sc_s"]))
        c_new = model.meas_col(row, col)
        converged = np.abs(c_new - c) <= col_tol
        c = c_new
    p = re_mdist(q, c, dst["col_T_sc_s"])
    row, col = model.project(p)
    r_pred = np.linalg.norm(p, axis=1)

    row_int = np.clip(np.rint(row).astype(np.int64), 0, H - 1)
    col_int = np.mod(np.rint(col).astype(np.int64), W)
    has_return = (row > -0.5) & (row < H - 0.5) & dst["valid"][row_int, col_int] & converged
    range_residual = np.where(has_return, np.abs(dst["range"][row_int, col_int] - r_pred), np.inf)
    return row, col, r_pred, range_residual


def compute_correspondences(src, dst, T_dst_src, model, cfg):
    """For every valid pixel of src, the sub-pixel (row, col) in dst's raw image and the range of
    the point in dst. NaN where there is no valid match."""
    H, W = model.H, model.W
    corr = np.full((H, W, 2), np.nan, dtype=np.float32)
    corr_range = np.full((H, W), np.nan, dtype=np.float32)

    use = src["valid"] & (src["range"] > cfg.corr_min_range)
    xyz = src["xyz_undist"][use].astype(np.float64)
    q = xyz @ T_dst_src[:3, :3].T + T_dst_src[:3, 3]  # dst sensor frame at dst's stamp

    # start from the column q projects to without motion: within a few columns of the answer
    start = model.meas_col(*model.project(q))
    row, col, r_pred, residual = predict_in_dst(q, dst, model, start, cfg.re_mdist_iters)

    # Near the azimuth seam (first / last measurement columns) a point can be seen at the end or
    # at the start of dst's scan (the two are ~one scan of motion apart), so also iterate from the
    # other end and keep the candidate whose range agrees best with dst's measurement.
    near_first = start < cfg.seam_margin
    seam = near_first | (start >= W - cfg.seam_margin)
    idx = np.nonzero(seam)[0]
    other_end = np.where(near_first[idx], W - 1, 0)
    row2, col2, r_pred2, residual2 = predict_in_dst(q[idx], dst, model, other_end, cfg.re_mdist_iters)
    second = residual2 < residual[idx]
    sel = idx[second]
    row[sel], col[sel], r_pred[sel], residual[sel] = \
        row2[second], col2[second], r_pred2[second], residual2[second]

    # occlusion check: the range dst measured at that pixel must agree with the predicted range
    ok = residual < cfg.range_abs_tol + cfg.range_rel_tol * r_pred

    src_rows, src_cols = np.nonzero(use)
    corr[src_rows[ok], src_cols[ok], 0] = row[ok]
    corr[src_rows[ok], src_cols[ok], 1] = col[ok]
    corr_range[src_rows[ok], src_cols[ok]] = r_pred[ok]
    return corr, corr_range


def cycle_check(corr_ab, corr_ba, max_err):
    """Mask (H, W) of A->B matches whose B->A match comes back within max_err px."""
    H, W = corr_ab.shape[:2]
    keep = np.zeros((H, W), dtype=bool)
    rows, cols = np.nonzero(np.isfinite(corr_ab[..., 0]))
    row_b = np.clip(np.rint(corr_ab[rows, cols, 0]).astype(np.int64), 0, H - 1)
    col_b = np.mod(np.rint(corr_ab[rows, cols, 1]).astype(np.int64), W)
    back = corr_ba[row_b, col_b]
    d_col = np.abs((back[:, 1] - cols + W / 2) % W - W / 2)
    err = np.hypot(back[:, 0] - rows, d_col)
    keep[rows, cols] = err <= max_err  # NaN (no way back) -> False
    return keep


# ---------------- output helpers ----------------
def rotate180(img):
    return np.ascontiguousarray(img[::-1, ::-1])


def rotate180_corr(corr, H, W):
    out = rotate180(corr).copy()
    out[..., 0] = H - 1 - out[..., 0]
    out[..., 1] = np.mod(W - 1 - out[..., 1], W)
    return out


def to_u8(img16):
    return (img16 >> 8).astype(np.uint8)


def warp_to_src(img_dst, corr):
    """Sample the dst image at the correspondences of every src pixel (bilinear, wrapping in azimuth)."""
    has_match = np.isfinite(corr[..., 0])
    map_row = np.where(has_match, corr[..., 0], -1).astype(np.float32)
    map_col = np.where(has_match, corr[..., 1], -1).astype(np.float32)
    warped = cv2.remap(img_dst, map_col, map_row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    warped[~has_match] = 0
    return warped


def draw_overlay(sig_a, sig_b, corr_ab, row_scale=3):
    """Top: A (yellow) over B warped into A via corr_AB (blue) -> aligned structure is gray.
    Bottom: A over raw B for reference. Yellow-only regions have no valid correspondence."""
    H, W = sig_a.shape
    up = lambda x: cv2.resize(to_u8(x), (W, H * row_scale), interpolation=cv2.INTER_NEAREST)
    a, b, b_warped = up(sig_a), up(sig_b), up(warp_to_src(sig_b, corr_ab))
    bar = np.full((10, W, 3), 255, np.uint8)
    return np.vstack([np.dstack([b_warped, a, a]), bar, np.dstack([b, a, a])])


def pose_diff(T1, T2):
    """Translation [m] and rotation [deg] of T1 @ inv(T2)."""
    D = T1 @ np.linalg.inv(T2)
    return np.linalg.norm(D[:3, 3]), np.rad2deg(np.linalg.norm(Rotation.from_matrix(D[:3, :3]).as_rotvec()))


# ---------------- matching two frames ----------------
def match_frames(fa, fb, model, cfg):
    """Match two odometry frames (any two, not only consecutive ones).
    Returns (pair, message): pair is None if the match failed, and the message says why.
    The pair is in the frames' own pixel coordinates (save_pair applies the 180 deg rotation)."""
    if not (fa["success"] and fb["success"]):
        return None, "odometry failed on one of the frames"

    # (1) odometry guess from the pose chain
    T_AB_odom = fa["T_s_m"] @ np.linalg.inv(fb["T_s_m"])
    distance = np.linalg.norm(T_AB_odom[:3, 3])
    heading = np.rad2deg(np.linalg.norm(Rotation.from_matrix(T_AB_odom[:3, :3]).as_rotvec()))
    apart = f"{distance:.1f} m / {heading:.0f} deg apart"

    # (2) ICP refinement on the undistorted clouds
    T_AB, fitness, rmse = run_icp(icp_points(fb, cfg), icp_points(fa, cfg), T_AB_odom, cfg.icp_scales)
    d_trans, d_rot = pose_diff(T_AB, T_AB_odom)
    if fitness < cfg.min_fitness:
        return None, f"FAILED ({apart}): ICP fitness {fitness:.2f} < {cfg.min_fitness} - too little overlap"
    if d_trans > cfg.max_trans_diff_to_odom or d_rot > cfg.max_rot_diff_to_odom_deg:
        return None, (f"FAILED ({apart}): ICP moved {d_trans:.2f} m / {d_rot:.1f} deg from the odometry guess "
                      f"- odometry drift too large or ICP converged wrongly")

    # (3) correspondences in both directions + round-trip check
    corr_ab, range_ab = compute_correspondences(fa, fb, np.linalg.inv(T_AB), model, cfg)
    corr_ba, range_ba = compute_correspondences(fb, fa, T_AB, model, cfg)
    if cfg.cycle_tol >= 0:
        keep_ab = cycle_check(corr_ab, corr_ba, cfg.cycle_tol)
        keep_ba = cycle_check(corr_ba, corr_ab, cfg.cycle_tol)
        corr_ab[~keep_ab], range_ab[~keep_ab] = np.nan, np.nan
        corr_ba[~keep_ba], range_ba[~keep_ba] = np.nan, np.nan
    num_matches = int(np.isfinite(corr_ab[..., 0]).sum())
    if num_matches < cfg.min_matches:
        return None, f"FAILED ({apart}): only {num_matches} correspondences (< {cfg.min_matches})"

    pair = dict(signal_A=fa["signal"], signal_B=fb["signal"], reflec_A=fa["reflec"], reflec_B=fb["reflec"],
                range_A=fa["range"], range_B=fb["range"], corr_AB=corr_ab, corr_BA=corr_ba,
                corr_range_AB=range_ab, corr_range_BA=range_ba, T_AB=T_AB, T_AB_odom=T_AB_odom,
                stamp_A=fa["stamp_ns"], stamp_B=fb["stamp_ns"], icp_fitness=fitness, icp_rmse=rmse)
    message = (f"ok ({apart}): fitness {fitness:.2f}, ICP - odometry {d_trans * 100:.1f} cm / {d_rot:.2f} deg, "
               f"{num_matches} correspondences ({100 * num_matches / (model.H * model.W):.0f}% of A)")
    return pair, message


def match_frame_ids(frame_a, frame_b, cfg=cfg):
    """Match two frames given by their index (row in trajectory.csv). Returns (pair, message)."""
    files = list_frames(cfg.odometry_dir)
    model = load_model(cfg.odometry_dir)
    pair, message = match_frames(load_frame(files[frame_a]), load_frame(files[frame_b]), model, cfg)
    if pair is not None:
        pair.update(frame_A=frame_a, frame_B=frame_b)
    return pair, message


def save_pair(pair, out_dir, model, cfg, save_vis=False):
    """Save a pair as <out_dir>/pairs/<frame A>_<frame B>.npz (rotated 180 deg if cfg.rotate180)."""
    out_dir = Path(out_dir)
    name = f"{int(pair['frame_A']):06d}_{int(pair['frame_B']):06d}"
    data = dict(pair)
    if cfg.rotate180:
        for key in ["signal_A", "signal_B", "reflec_A", "reflec_B", "range_A", "range_B",
                    "corr_range_AB", "corr_range_BA"]:
            data[key] = rotate180(data[key])
        data["corr_AB"] = rotate180_corr(data["corr_AB"], model.H, model.W)
        data["corr_BA"] = rotate180_corr(data["corr_BA"], model.H, model.W)
    np.savez_compressed(out_dir / "pairs" / f"{name}.npz", **data, rotated180=cfg.rotate180)
    if save_vis:
        cv2.imwrite(str(out_dir / "vis" / f"{name}.png"),
                    draw_overlay(data["signal_A"], data["signal_B"], data["corr_AB"]))
    return name


# ---------------- main ----------------
def main(cfg):
    out_dir = Path(cfg.out_dir)
    if any((out_dir / "pairs").glob("*.npz")):
        raise FileExistsError(f"{out_dir / 'pairs'} already has pairs: delete them or change out_dir")
    (out_dir / "pairs").mkdir(parents=True, exist_ok=True)
    (out_dir / "vis").mkdir(exist_ok=True)
    model = load_model(cfg.odometry_dir)
    files = list_frames(cfg.odometry_dir)
    print(f"{len(files)} odometry frames in {cfg.odometry_dir}")

    if cfg.frame_pairs is not None:
        frame_pairs = list(cfg.frame_pairs)
    else:
        frame_pairs = [(i, i + cfg.frame_gap) for i in range(cfg.first_frame, len(files) - cfg.frame_gap, cfg.stride)]

    n_pairs = 0
    for frame_a, frame_b in frame_pairs:
        pair, message = match_frames(load_frame(files[frame_a]), load_frame(files[frame_b]), model, cfg)
        print(f"frames {frame_a:5d} -> {frame_b:5d}: {message}")
        if pair is None:
            continue
        pair.update(frame_A=frame_a, frame_B=frame_b)
        save_pair(pair, out_dir, model, cfg, save_vis=cfg.vis_every > 0 and n_pairs % cfg.vis_every == 0)
        n_pairs += 1
        if cfg.max_pairs is not None and n_pairs >= cfg.max_pairs:
            break
    print(f"wrote {n_pairs} of {len(frame_pairs)} pairs to {out_dir}")


if __name__ == "__main__":
    main(cfg)
