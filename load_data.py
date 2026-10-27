# simple script to look inside the saved npz files: odometry frames and image pairs
from pathlib import Path

import numpy as np

import config

######## Parameters (edit by hand) ########
FRAMES_DIR = config.ODOMETRY_DIR / "frames"   # from run_odometry.py (folder set in config.py)
PAIRS_DIR = config.PAIRS_DIR / "pairs"        # from make_pairs.py
NUM_FILES = 1                                       # how many files of each kind to print


def load_data(file_path):
    with np.load(file_path) as data:
        return {key: data[key] for key in data.files}


def describe(data):
    """Print every array: shape, dtype and value range (or the value for scalars)."""
    for key, arr in data.items():
        if arr.ndim == 0:
            print(f"  {key:14s} scalar  {str(arr.dtype):8s} = {arr.item()}")
            continue
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr.ravel()
        value_range = f"min={finite.min():.4g} max={finite.max():.4g}" if finite.size else "all NaN"
        print(f"  {key:14s} {str(arr.shape):16s} {str(arr.dtype):8s} {value_range}")


def describe_frame(data):
    valid = data["range"] > 0
    w = data["w_m_r_in_r"]
    print(f"\n  valid pixels: {valid.sum()} / {valid.size}")
    print(f"  speed {np.linalg.norm(w[:3]):.2f} m/s, yaw rate {-w[5]:+.3f} rad/s  (w_m_r_in_r = {np.round(w, 3)})")
    T_last = data["col_T_sc_s"][-1]
    print(f"  motion over the scan (last column): {np.linalg.norm(T_last[:3, 3]) * 100:.1f} cm")
    print(f"  T_r_m:\n{np.round(data['T_r_m'], 4)}")


def describe_pair(data):
    H, W = data["signal_A"].shape
    corr = data["corr_AB"]
    has_match = np.isfinite(corr[..., 0])
    print(f"\n  corr_AB: {has_match.sum()} / {H * W} pixels of A have a match in B ({100 * has_match.mean():.1f}%)")
    rows, cols = np.nonzero(has_match)
    d_row = corr[rows, cols, 0] - rows
    d_col = (corr[rows, cols, 1] - cols + W / 2) % W - W / 2  # columns wrap around 360 deg
    flow = np.hypot(d_row, d_col)
    print(f"  flow [px]: median={np.median(flow):.2f}  95th pct={np.percentile(flow, 95):.2f}")
    print("  examples  (row, col) in A -> (row, col) in B:")
    for k in np.linspace(0, rows.size - 1, 5).astype(int):
        r, c = rows[k], cols[k]
        print(f"    ({r:3d}, {c:4d}) -> ({corr[r, c, 0]:7.2f}, {corr[r, c, 1]:7.2f})   "
              f"range in B {data['corr_range_AB'][r, c]:.2f} m")
    print(f"  T_AB (p_A = T_AB @ p_B):\n{np.round(data['T_AB'], 4)}")
    print(f"  dt = {(data['stamp_B'] - data['stamp_A']) * 1e-6:.1f} ms")


def main():
    for kind, folder, extra in (("frame", FRAMES_DIR, describe_frame), ("pair", PAIRS_DIR, describe_pair)):
        files = sorted(Path(folder).glob("*.npz"))
        print(f"\n######## {len(files)} {kind} files in {folder}")
        for f in files[:NUM_FILES]:
            print(f"\n===== {f.name} =====")
            data = load_data(f)
            describe(data)
            extra(data)


if __name__ == "__main__":
    main()
