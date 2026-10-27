"""
Ouster sensor model: image layout and projection of 3-D points to (sub-pixel) image coordinates.

Image layout (verified on our bag, OS 128 x 1024 @ 10 Hz):
  * /ouster/points is organized and destaggered: point (row i, col j) is pixel (i, j) of
    /ouster/signal_image and /ouster/reflec_image.
  * The sensor fires one "measurement column" at a time. Pixel (i, j) of the destaggered image
    was measured in measurement column m = (j - pixel_shift_by_row[i]) mod W, so all pixels of a
    measurement column share one firing time, but an image column does not.
"""
import numpy as np


class OusterModel:
    def __init__(self, meta: dict):
        fmt = meta.get("lidar_data_format", meta.get("data_format"))
        beam = meta["beam_intrinsics"]
        self.H = int(fmt["pixels_per_column"])
        self.W = int(fmt["columns_per_frame"])
        self.shift = np.asarray(fmt["pixel_shift_by_row"], dtype=np.int64)
        self.alt = np.deg2rad(np.asarray(beam["beam_altitude_angles"], dtype=np.float64))  # descending
        self.az = np.deg2rad(np.asarray(beam["beam_azimuth_angles"], dtype=np.float64))
        self.n = float(beam["lidar_origin_to_beam_origin_mm"]) * 1e-3  # beam origin offset [m]

        # after destaggering, the beam azimuth of pixel (i, j) is 2*pi*(1 - j/W) + delta[i]
        self.delta = 2.0 * np.pi * self.shift / self.W - self.az

        # scan period from the lidar mode, e.g. "1024x10" -> 0.1 s
        mode = meta.get("config_params", {}).get("lidar_mode", f"{self.W}x10")
        self.scan_period = 1.0 / float(mode.split("x")[1])

        # measurement column of every pixel, (H, W)
        rows = np.arange(self.H)[:, None]
        cols = np.arange(self.W)[None, :]
        self.meas_col_of_pixel = np.mod(cols - self.shift[rows], self.W)

    def project(self, p):
        """(N, 3) points in the os_lidar frame -> fractional (row, col), each (N,).
        Inverts the Ouster beam model; re-projecting the driver's own points gives < 1e-3 px error."""
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        rows = np.arange(self.H, dtype=np.float64)
        encoder = np.arctan2(y, x)
        for _ in range(4):  # fixed-point iteration on the encoder angle (beam origin offset)
            dx = x - self.n * np.cos(encoder)
            dy = y - self.n * np.sin(encoder)
            theta = np.arctan2(dy, dx)
            phi = np.arctan2(z, np.hypot(dx, dy))
            # altitude angles are descending -> negate for np.interp; extrapolate linearly outside the FOV
            row = np.interp(-phi, -self.alt, rows)
            above = phi > self.alt[0]
            below = phi < self.alt[-1]
            row[above] = -(phi[above] - self.alt[0]) / (self.alt[0] - self.alt[1])
            row[below] = (self.H - 1) + (self.alt[-1] - phi[below]) / (self.alt[-2] - self.alt[-1])
            row_int = np.clip(np.rint(row).astype(np.int64), 0, self.H - 1)
            encoder = theta + self.az[row_int]
        col = np.mod(self.W * (1.0 - (theta - self.delta[row_int]) / (2.0 * np.pi)), self.W)
        return row, col

    def meas_col(self, row, col):
        """Nearest measurement column for fractional pixel coordinates."""
        row_int = np.clip(np.rint(row).astype(np.int64), 0, self.H - 1)
        col_int = np.rint(col).astype(np.int64)
        return np.mod(col_int - self.shift[row_int], self.W)
