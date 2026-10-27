"""
Self-consistency of mdist -> re_mdist -> projection: pairing a saved frame with itself (T = I)
must map every valid pixel (beyond corr_min_range) onto itself. For comparison, the error of
projecting the undistorted points directly (no re_mdist) is printed as well.
Run:  python tests/test_correspondences.py      (needs frames from run_odometry.py)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config
import make_pairs
from core.ouster_model import OusterModel

ODOMETRY_DIR = config.ODOMETRY_DIR   # folder set in config.py
NUM_FRAMES = 5


def pixel_error(row, col, rows, cols, W):
    d_col = (col - cols + W / 2) % W - W / 2
    return np.hypot(row - rows, d_col)


def main(odometry_dir=ODOMETRY_DIR):
    meta = json.loads((Path(odometry_dir) / "meta.json").read_text())
    model = OusterModel(meta["ouster_metadata"])
    files = sorted((Path(odometry_dir) / "frames").glob("*.npz"), key=lambda f: int(f.stem))
    cfg = make_pairs.cfg

    for f in files[:: max(len(files) // NUM_FRAMES, 1)][:NUM_FRAMES]:
        frame = make_pairs.load_frame(f)
        use = frame["valid"] & (frame["range"] > cfg.corr_min_range)
        rows, cols = np.nonzero(use)

        corr, _ = make_pairs.compute_correspondences(frame, frame, np.eye(4), model, cfg)
        err = pixel_error(corr[rows, cols, 0], corr[rows, cols, 1], rows, cols, model.W)
        matched = np.isfinite(err).mean()

        naive_row, naive_col = model.project(frame["xyz_undist"][use].astype(np.float64))
        naive_err = pixel_error(naive_row, naive_col, rows, cols, model.W)

        yaw_rate = np.abs(frame["col_w_s"][:, 5]).max()
        print(f"{f.stem}: re_mdist: matched {100 * matched:.3f}%  max err {np.nanmax(err):.4f} px | "
              f"no re_mdist: median {np.median(naive_err):.2f} px, 99% {np.percentile(naive_err, 99):.2f} px "
              f"| max |yaw rate| {yaw_rate:.2f} rad/s")
        assert matched > 0.999 and np.nanmax(err) < 0.01
    print("self-correspondence test passed")


if __name__ == "__main__":
    main()
