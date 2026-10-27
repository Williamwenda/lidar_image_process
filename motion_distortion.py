"""
Remove (mdist) and re-apply (re_mdist) the motion distortion of an Ouster scan.

Every measurement column c is measured in the sensor frame at its firing time, s(t_c). The
odometry gives, per column, the transform T_sc_s = T_{s(t_c), s(stamp)}, which maps points from
the sensor frame at the scan stamp into the sensor frame at the column's firing time.

    mdist    : p_s  = inv(T_sc_s) @ p_sc   (measured point -> sensor frame at the stamp)
    re_mdist : p_sc = T_sc_s @ p_s         (sensor frame at the stamp -> frame the point is measured in)
"""
import numpy as np


def apply_per_point(T, xyz):
    """T (N, 4, 4), xyz (N, 3) -> (N, 3)."""
    return np.einsum("nij,nj->ni", T[:, :3, :3], xyz) + T[:, :3, 3]


def mdist(xyz, meas_cols, col_T_sc_s):
    """Remove motion distortion. xyz (N, 3) measured in column frames, meas_cols (N,) their columns."""
    col_T_s_sc = np.linalg.inv(col_T_sc_s)
    return apply_per_point(col_T_s_sc[meas_cols], xyz)


def re_mdist(xyz, meas_cols, col_T_sc_s):
    """Re-apply motion distortion. xyz (N, 3) in the sensor frame at the stamp."""
    return apply_per_point(col_T_sc_s[meas_cols], xyz)


def mdist_scan(scan, model, col_T_sc_s):
    """Undistort the full organized scan -> (H, W, 3) in the sensor frame at the stamp, NaN = no return."""
    valid = scan["valid"]
    xyz_undist = np.full(scan["xyz"].shape, np.nan)
    xyz_undist[valid] = mdist(scan["xyz"][valid], model.meas_col_of_pixel[valid], col_T_sc_s)
    return xyz_undist
