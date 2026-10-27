"""
Sliding voxel map used as the ICP target (simplified version of vtr's odometry_map_maintenance_v2).

  * One point per voxel, stored in the odometry (map) frame.
  * Every voxel touched by a new scan gets its life time reset; all points then lose one life,
    and points that were not observed for `point_life_time` scans are removed.
  * Normals and planarity scores are recomputed for the touched points from their neighbours
    within 3 voxels: normal = smallest eigenvector, normal_score = (l1 - l0) / l2 (vtr's formula).
"""
import numpy as np
from scipy.spatial import cKDTree


class PointMap:
    def __init__(self, voxel_size, point_life_time, normal_knn=30):
        self.voxel_size = voxel_size
        self.point_life_time = point_life_time
        self.normal_knn = normal_knn

        self.points = np.zeros((0, 3))
        self.normals = np.zeros((0, 3))
        self.scores = np.zeros(0)
        self.life = np.zeros(0)
        self.keys = np.zeros(0, dtype=np.int64)
        self.kdtree = None

    def __len__(self):
        return len(self.points)

    def voxel_keys(self, xyz):
        """One int64 key per voxel (21 bits per axis)."""
        ijk = np.floor(xyz / self.voxel_size).astype(np.int64) + (1 << 20)
        return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]

    def update(self, points_in_map):
        # one new point per voxel
        new_keys, first = np.unique(self.voxel_keys(points_in_map), return_index=True)
        new_points = points_in_map[first]

        # voxels that already exist in the map keep their point, but get a fresh life time
        if len(self.keys) > 0:
            order = np.argsort(self.keys)
            sorted_keys = self.keys[order]
            pos = np.minimum(np.searchsorted(sorted_keys, new_keys), len(sorted_keys) - 1)
            exists = sorted_keys[pos] == new_keys
            old_idx = order[pos[exists]]
        else:
            exists = np.zeros(len(new_keys), dtype=bool)
            old_idx = np.zeros(0, dtype=np.int64)
        self.life[old_idx] = self.point_life_time

        # voxels that are new get the new point
        n_old = len(self.points)
        n_add = int((~exists).sum())
        self.points = np.vstack([self.points, new_points[~exists]])
        self.normals = np.vstack([self.normals, np.zeros((n_add, 3))])
        self.scores = np.concatenate([self.scores, np.zeros(n_add)])
        self.life = np.concatenate([self.life, np.full(n_add, float(self.point_life_time))])
        self.keys = np.concatenate([self.keys, new_keys[~exists]])

        touched = np.concatenate([old_idx, np.arange(n_old, n_old + n_add)])
        self.kdtree = cKDTree(self.points)
        self.update_normals(touched)

        # age all points, remove the ones that were not seen for too long
        self.life -= 1.0
        alive = self.life > 0.0
        self.points = self.points[alive]
        self.normals = self.normals[alive]
        self.scores = self.scores[alive]
        self.life = self.life[alive]
        self.keys = self.keys[alive]
        self.kdtree = cKDTree(self.points)

    def update_normals(self, idx):
        radius = 3.0 * self.voxel_size
        k = min(self.normal_knn, len(self.points))
        dist, nb = self.kdtree.query(self.points[idx], k=k, distance_upper_bound=radius, workers=-1)
        dist = dist.reshape(len(idx), k)
        nb = nb.reshape(len(idx), k)
        is_nb = np.isfinite(dist)  # missing neighbours have dist = inf
        count = is_nb.sum(axis=1)

        neighbours = self.points[np.where(is_nb, nb, 0)]           # (M, k, 3)
        weight = is_nb[..., None].astype(np.float64)
        mean = (neighbours * weight).sum(axis=1) / np.maximum(count, 1)[:, None]
        centered = (neighbours - mean[:, None, :]) * weight
        cov = np.einsum("mki,mkj->mij", centered, centered) / np.maximum(count, 1)[:, None, None]
        eigvals, eigvecs = np.linalg.eigh(cov)                       # ascending eigenvalues

        enough = count >= 4
        ok = idx[enough]
        self.normals[ok] = eigvecs[enough, :, 0]
        l0, l1, l2 = eigvals[enough, 0], eigvals[enough, 1], eigvals[enough, 2]
        self.scores[ok] = np.where(l2 > 0, (l1 - l0) / np.maximum(l2, 1e-12), 0.0)
