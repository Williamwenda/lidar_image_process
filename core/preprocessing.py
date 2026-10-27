"""
Select the points used by the odometry ICP (simplified version of vtr's ouster_converter +
preprocessing modules): drop the robot body, crop the range and voxel-downsample.

Differences to vtr: no planarity-based sampling of the query points (vtr only uses query normals
for that filter; the ICP cost uses the map normals).
"""
import numpy as np


def voxel_downsample_indices(xyz, voxel_size):
    """Index of one point per voxel: the point closest to the voxel centre."""
    ijk = np.floor(xyz / voxel_size).astype(np.int64)
    dist_to_center = np.linalg.norm(xyz - (ijk + 0.5) * voxel_size, axis=1)
    _, voxel_id = np.unique(ijk, axis=0, return_inverse=True)
    voxel_id = voxel_id.ravel()
    order = np.lexsort((dist_to_center, voxel_id))  # sort by voxel, then by distance
    first_in_voxel = np.ones(len(order), dtype=bool)
    first_in_voxel[1:] = voxel_id[order][1:] != voxel_id[order][:-1]
    return order[first_in_voxel]


def preprocess(scan, model, cfg):
    """Returns the ICP query points (N, 3) in the os_lidar frame and their absolute times (N,) [ns]."""
    valid = scan["valid"]
    xyz = scan["xyz"][valid]
    times = scan["col_times"][model.meas_col_of_pixel[valid]]  # [s since stamp]

    # robot body: same box as vtr's ouster_converter (filter_radius, filter_z_min/max), in the lidar frame
    radius_sq = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    on_body = (radius_sq <= cfg.body_filter_radius ** 2) & \
              (xyz[:, 2] >= cfg.body_filter_z_min) & (xyz[:, 2] <= cfg.body_filter_z_max)
    in_range = np.linalg.norm(xyz, axis=1) < cfg.crop_range
    keep = ~on_body & in_range
    xyz, times = xyz[keep], times[keep]

    pick = voxel_downsample_indices(xyz, cfg.frame_voxel_size)
    points = xyz[pick]
    point_times_ns = scan["stamp_ns"] + np.rint(times[pick] * 1e9).astype(np.int64)
    return points, point_times_ns
