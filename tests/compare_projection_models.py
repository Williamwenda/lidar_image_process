"""
Spherical projection (COIN-LIO, eq. 1) vs. the Ouster beam model (OusterModel.project).

Ground truth: the driver's organized, destaggered point cloud. The point at (row i, col j) IS pixel
(i, j) of the intensity image, so projecting it back must give exactly (i, j). Any error is the
error of the projection model. Models compared:
  1. spherical      : uniform elevation spacing over the FOV, no beam origin offset (eq. 1, r = 0)
  2. coin_lio       : as COIN-LIO implements it: measured beam altitudes (interpolated),
                      beam origin offset r, one constant azimuth offset for all beams
  3. coin_lio + dk  : coin_lio plus the per-row azimuth correction delta_k (= our model, one pass)
  4. ouster         : OusterModel.project (fixed-point iteration on the encoder angle)
The column convention u = W (1 - theta / 2 pi) is shared by all models (a constant column offset
is calibration, not the model).
Run:  python tests/compare_projection_models.py      (reads one scan from the bag)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from bag_io import iter_scans, read_metadata
from ouster_model import OusterModel

######## Parameters (edit by hand) ########
BAG = "/home/wenda/ASRL/vtr3/data/ouster_lidar/rosbag2_2025_05_21-11_02_59"
START_SEC = 30.0  # take the first scan after this offset from the bag start [s]
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "projection_models"
RANGE_BINS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0]  # [m], for the error-vs-range table


def azimuth_to_col(theta, W):
    return np.mod(W * (1.0 - theta / (2.0 * np.pi)), W)


def project_spherical(p, model):
    """COIN-LIO eq. (1) with r = 0: rows evenly spaced between the top and bottom beam."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    phi = np.arctan2(z, np.hypot(x, y))
    fov = model.alt[0] - model.alt[-1]
    row = (model.alt[0] - phi) / fov * (model.H - 1)
    return row, azimuth_to_col(np.arctan2(y, x), model.W)


def project_coin_lio(p, model, per_row_delta=False):
    """Measured beam altitudes, beam origin offset r = n along the point's own azimuth,
    one constant azimuth offset (zero here) unless per_row_delta."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    theta = np.arctan2(y, x)
    L = np.hypot(x, y) - model.n
    phi = np.arctan2(z, L)
    row = np.interp(-phi, -model.alt, np.arange(model.H, dtype=np.float64))
    if per_row_delta:
        theta = theta - model.delta[np.clip(np.rint(row).astype(np.int64), 0, model.H - 1)]
    return row, azimuth_to_col(theta, model.W)


def pixel_errors(row, col, rows, cols, W):
    d_row = row - rows
    d_col = (col - cols + W / 2) % W - W / 2  # columns wrap around 360 deg
    return d_row, d_col


def main():
    with AnyReader([Path(BAG)], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
        model = OusterModel(read_metadata(reader))
        start_ns = reader.start_time + int(START_SEC * 1e9)
        scan = next(iter_scans(reader, model, "/ouster/points", "/ouster/signal_image",
                               "/ouster/reflec_image", start_ns=start_ns))

    rows, cols = np.nonzero(scan["valid"])
    p = scan["xyz"][rows, cols]
    rng = scan["range"][rows, cols]
    print(f"scan {scan['stamp_ns']}: {rows.size} valid pixels, Ouster {model.H}x{model.W}, "
          f"beam origin offset n = {model.n * 1e3:.1f} mm")

    models = {
        "spherical": project_spherical(p, model),
        "coin_lio": project_coin_lio(p, model),
        "coin_lio + dk": project_coin_lio(p, model, per_row_delta=True),
        "ouster": model.project(p),
    }

    errors = {}
    print(f"\n{'model':14s} {'|d_row| med/p99/max':>24s} {'|d_col| med/p99/max':>24s} "
          f"{'wrong pixel':>12s} {'wrong meas col':>15s}")
    for name, (row, col) in models.items():
        d_row, d_col = pixel_errors(row, col, rows, cols, model.W)
        errors[name] = (d_row, d_col)
        # nearest pixel (what a pixel lookup / firing-time lookup would use)
        wrong_px = (np.rint(row) != rows) | (np.mod(np.rint(col), model.W) != cols)
        wrong_meas = model.meas_col(row, col) != model.meas_col_of_pixel[rows, cols]
        stats = lambda e: f"{np.median(np.abs(e)):6.3f}/{np.percentile(np.abs(e), 99):6.3f}/{np.abs(e).max():6.3f}"
        print(f"{name:14s} {stats(d_row):>24s} {stats(d_col):>24s} "
              f"{100 * wrong_px.mean():11.2f}% {100 * wrong_meas.mean():14.2f}%")

    # column error vs. range: the beam origin offset matters only close to the sensor
    print("\nmedian |d_col| [px] vs. range")
    edges = np.asarray(RANGE_BINS)
    print(f"{'model':14s} " + " ".join(f"{f'{a:g}-{b:g} m':>9s}" for a, b in zip(edges[:-1], edges[1:])))
    for name, (_, d_col) in errors.items():
        cells = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (rng >= a) & (rng < b)
            cells.append(f"{np.median(np.abs(d_col[m])):9.3f}" if m.any() else f"{'-':>9s}")
        print(f"{name:14s} " + " ".join(cells))

    # figure: signal image, then per model the row and column error maps
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1 + len(models), 2, figsize=(16, 2.2 * (1 + len(models))), squeeze=False)
    signal = scan["signal"].astype(np.float64)
    for ax in axes[0]:
        ax.imshow(signal, cmap="gray", vmax=np.percentile(signal, 99), aspect="auto")
        ax.set_title("signal image")
    for k, (name, (d_row, d_col)) in enumerate(errors.items(), start=1):
        for ax, e, label in ((axes[k, 0], d_row, "row"), (axes[k, 1], d_col, "col")):
            img = np.full((model.H, model.W), np.nan)
            img[rows, cols] = e
            lim = max(np.percentile(np.abs(e), 99), 1e-3)
            im = ax.imshow(img, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto")
            ax.set_title(f"{name}: {label} error [px]")
            fig.colorbar(im, ax=ax, fraction=0.02)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "error_maps.png", dpi=120)

    # row error of the uniform spherical model per beam: irregular beam spacing
    fig, ax = plt.subplots(figsize=(8, 3))
    d_row = errors["spherical"][0]
    per_beam = np.array([np.median(d_row[rows == i]) if (rows == i).any() else np.nan for i in range(model.H)])
    ax.plot(per_beam, ".-")
    ax.set_xlabel("beam (row)")
    ax.set_ylabel("median row error [px]")
    ax.set_title("spherical model with uniform elevation spacing")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spherical_row_error_per_beam.png", dpi=120)
    print(f"\nfigures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
