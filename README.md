# Lidar intensity image correspondences from lidar odometry

Builds pairs of Ouster lidar intensity images (`/ouster/signal_image`, `/ouster/reflec_image`) with
**ground-truth pixel correspondences**, using the fact that every image pixel is one 3-D lidar point.

The pipeline has two steps:

1. **Odometry** (`run_odometry.py`): a Python port of vtr3's lidar odometry
   (`vtr_lidar/src/modules/odometry/odometry_icp_module.cpp`). It runs scan-to-map point-to-plane
   ICP with a continuous-time, constant-velocity trajectory (pysteam), and removes the motion
   distortion of every scan. Each scan is saved with its undistorted points, its pose, and the
   sensor motion during the scan.
2. **Pairs** (`make_pairs.py`): for any two saved scans, refine their relative pose with ICP and
   compute, for every pixel of image A, its sub-pixel location in the **raw** image B.

The images themselves are never warped or undistorted. The correspondences point to where a surface
point really appears in the raw, motion-distorted image B.

## Quick start

```bash
cd accelerated_features/lidar_intensity_data_process

python3 run_odometry.py         # 1. odometry -> outputs/<FOLDER_NAME>/odometry/
python3 make_pairs.py           # 2. image pairs -> outputs/<FOLDER_NAME>/pairs/
python3 visualize_odometry.py   # trajectory, map, two aligned frames -> outputs/<FOLDER_NAME>/odometry/vis/
python3 visualize_pairs.py      # match any two frames (FRAME_A, FRAME_B), click pixels to inspect the matches
python3 load_data.py            # print what is inside the saved files
```

Every script has a **parameter block at the top**; there are no command-line arguments. For a first
try, set `duration_sec=30` in `run_odometry.py`. The whole bag (~4,100 scans) takes ~2.4 h at
~2.1 s per scan.

**Which bag, which folder: `config.py`.** Set `BAG` and `FOLDER_NAME` there; results go to
`outputs/<FOLDER_NAME>/` and every script reads that folder. `run_odometry.py` and `make_pairs.py`
stop instead of overwriting results already in it.

## Setup

- Python 3 with `numpy`, `scipy`, `open3d`, `rosbags`, `opencv-python`, `matplotlib`
  (no ROS installation needed: `rosbags` reads the bag directly).
- `pylgmath` and `pysteam` are git submodules in `deps/` and used from there. `core/deps_path.py` puts
  them on the Python path, so nothing has to be installed. Clone with
  `git clone --recurse-submodules <repo>`, or run `git submodule update --init` after a plain clone.
- The bag path is set in `config.py` (`BAG`).

## Code structure

| File | What it does |
|---|---|
| `config.py` | The bag (`BAG`) and the results folder (`FOLDER_NAME`), used by every script. |
| `run_odometry.py` | **Step 1 entry point.** Reads the bag, runs preprocessing → odometry → map update for every scan, saves the frames. |
| `make_pairs.py` | **Step 2 entry point.** Matches frame pairs: sequential (`frame_gap`, `stride`) or a list of any two frames (`frame_pairs`). `match_frames` / `match_frame_ids` match two arbitrary frames: odometry pose chain as the initial guess, ICP refinement, correspondences, and a notice when the match fails. |
| `visualize_odometry.py` | Trajectory plot, map rebuilt from the odometry, and two frames aligned by the odometry (Open3D windows + saved files). |
| `visualize_pairs.py` | Matches two arbitrary frames (`FRAME_A`, `FRAME_B`) and opens a correspondence inspector: images A and B one above the other; click a pixel in either to draw a line to its sub-pixel match, and see the local patches, flow, ranges, cycle error and 3-D error. Matches can also be listed by hand (`PIXELS_A`) or sampled (`NUM_RANDOM`). Optional 3-D view of the two point clouds (`PLOT_3D`). |
| `load_data.py` | Prints the content (keys, shapes, ranges) of saved frame and pair files. |
| `core/odometry_icp.py` | `OdometryICP`: the port of vtr's odometry (knots, priors, two-stage ICP loop, convergence, prior for the next scan). `sample_trajectory` evaluates the trajectory at the 1024 column times. |
| `core/cost_terms.py` | pysteam cost terms: batched point-to-plane (points grouped in time bins share one interpolated pose), and the joint 12x12 pose + velocity prior. |
| `core/point_map.py` | `PointMap`: sliding voxel map used as the ICP target (normals, planarity scores, point life time). |
| `core/preprocessing.py` | Selects the ICP query points: removes the robot body, crops the range, voxel-downsamples. |
| `core/motion_distortion.py` | `mdist` (remove motion distortion) and `re_mdist` (re-apply it). |
| `core/ouster_model.py` | `OusterModel`: image layout and the projection of 3-D points to sub-pixel image coordinates (inverse Ouster beam model). |
| `core/bag_io.py` | Reads scans (point cloud + signal + reflectivity image, matched by stamp), the Ouster metadata, and static transforms (`T_s_r`) from the bag. |
| `core/deps_path.py` | Adds `deps/pylgmath` and `deps/pysteam` to the Python path. |
| `tests/` | Tests (see below). |

How the modules connect:

```
run_odometry.py
  bag_io.iter_scans ─► preprocessing.preprocess ─► OdometryICP.run ─► PointMap.update
                                                       │  (cost_terms, pysteam)
                                                       └─► motion_distortion.mdist_scan ─► frames/<frame_id>.npz

make_pairs.py
  frames A, B ─► ICP refinement (open3d) ─► compute_correspondences
                                               (re_mdist + OusterModel.project) ─► pairs/<A>_<B>.npz
```

## Outputs

### `outputs/<FOLDER_NAME>/odometry/` (from `run_odometry.py`)

- `frames/<frame_id>.npz`: one file per scan, named 000000, 000001, ... (the frame id = row in
  `trajectory.csv`; the scan stamp is stored inside as `stamp_ns`). H x W = 128 x 1024.

  | Key | Content |
  |---|---|
  | `signal`, `reflec` | (H, W) uint16 raw images |
  | `range` | (H, W) raw range [m], 0 = no return |
  | `xyz_undist` | (H, W, 3) undistorted points, sensor frame at the stamp, NaN = no return |
  | `col_times` | (W,) firing time of each measurement column [s since the stamp] |
  | `col_T_sc_s` | (W, 4, 4) sensor frame at the stamp → at column c's firing time |
  | `col_w_s` | (W, 6) body velocity in the sensor frame at each column time |
  | `T_r_m`, `T_s_m` | (4, 4) robot / sensor pose at the stamp (map → robot / sensor) |
  | `w_m_r_in_r`, `cov_T_r_m` | body velocity at the stamp (robot frame), pose covariance |
  | `T_rk_rkm1` | incremental pose from the previous frame |
  | `success`, `matched_ratio`, `num_steps`, `converged`, `icp_residual` | odometry health |

- `trajectory.csv`: one row per frame (stamp, position, speed, yaw rate, health).
- `meta.json`: Ouster metadata, `T_s_r`, and the parameters used.
- `vis/`: figures and `.ply` files from `visualize_odometry.py`.

### `outputs/<FOLDER_NAME>/pairs/` (from `make_pairs.py`)

- `pairs/<A>_<B>.npz` (frame ids = rows of `trajectory.csv`): `signal_A/B`, `reflec_A/B`, `range_A/B`, and:
  - `corr_AB` (H, W, 2): (row, col) in B for every pixel of A, NaN = no match. `corr_BA` is the reverse.
  - `corr_range_AB/BA`: range of the matched point [m], useful to filter or weight matches.
  - `T_AB` (refined), `T_AB_odom`, stamps, ICP fitness / rmse.
- `vis/inspect_<A>_<B>.png` (inspector, `s` saves the current view) and `vis/match3d_<A>_<B>.ply`: from `visualize_pairs.py`.
- `vis/<A>_<B>.png`: overlay check. Top: A in yellow, B warped into A in blue, so aligned
  structure looks gray. Bottom: A over raw B.

Columns wrap around 360°: `col` is in [0, W), and a value in (W-1, W) lies between the last and the
first column. With `rotate180=True` (default), images and correspondences are rotated for the
upside-down mount on the Warthog.

## Conventions

- Barfoot / vtr notation. `T_b_a` maps points from frame a into frame b; `r` = robot (`base_link`),
  `s` = sensor (`os_lidar`), `m` = map. The map frame is the robot frame at the end of the first
  processed scan, and the first scan only initializes the map (it is not saved).
- `w_m_r_in_r` follows vtr: `T_r(t)_m = exp((t - t0) w^) T_r(t0)_m`, so driving forward gives a
  negative `rho_x`.
- **Image layout.** `/ouster/points` is organized and destaggered: point (i, j) is pixel (i, j) of
  the images. Pixel (i, j) was measured in measurement column `(j - pixel_shift_by_row[i]) mod W`,
  and all pixels of a measurement column share one firing time.
- **mdist / re_mdist.** `mdist`: `p_s = inv(col_T_sc_s[c]) @ p_measured`. `re_mdist`: `p_measured = col_T_sc_s[c] @ p_s`.
- **Correspondence A → B.** Undistorted A point → B's stamp frame (`inv(T_AB)`) → `re_mdist` with
  B's column motion → project with the Ouster model. The column depends on where the point lands,
  so this is iterated, starting from the column the point projects to without motion. Near the
  azimuth seam (`seam_margin` columns) it is also iterated from the other end of the scan, since a
  point can be seen at the start or at the end; points whose column does not converge (the gap
  between the end and the start of the sweep) are dropped. A match is kept if B has a return there with a matching range
  (occlusion check), the B → A match comes back within 1 px, and the point is beyond
  `corr_min_range`.

## Tests

```bash
python tests/test_cost_terms.py        # custom cost terms vs pysteam's own (~1e-13)
python tests/test_trajectory.py        # batched trajectory sampling vs pysteam interpolators
python tests/test_correspondences.py   # frame paired with itself maps every pixel onto itself (needs an odometry run)
python tests/test_synthetic.py         # known-motion test with simulated scans (~6 min)
```

`tests/lidar_simulator.py` ray-casts a synthetic world with the real Ouster beam model, timing and
mounting, while the robot follows a known path.

## Validation summary

| Check | Result |
|---|---|
| Simulated scans, known motion (1.1–1.9 m/s, up to 0.7 rad/s, 1 cm range noise) | scan-to-scan error 1.2 mm / 0.015°; drift 1.2 cm over 5 m; undistorted points 0.8 cm median error |
| Simulated pair correspondences vs truth | beyond 3 m: 0.13–0.18 px median, < 0.5 % above 1 px. 1–3 m: 0.68 px median |
| Real data, U-turn loop (revisit pairs 20–29 s apart, facing 140–174° apart) | odometry vs direct ICP: 1.6–2.3 cm, 0.05° |
| Motion compensation off (simulated / real) | drift 22 cm / 12 cm; pair error 1.2 px median in simulation |

## Known limitations

- **Close range.** Correspondences closer than ~3 m are less accurate (~0.7 px median), because
  of the relative-pose error and range noise. Filter with `corr_range_AB`, or raise
  `corr_min_range` in `make_pairs.py`.
- **Moving objects** (people, cars) are only partly removed, by the range check. An object sliding
  along its own surface cannot be detected from two frames.
- **The first scan** is put into the map without motion compensation (as in vtr), so the map frame
  has a constant offset of about half a scan of motion. Relative poses are not affected.
- **Simplified from vtr:** preprocessing and the map (no planarity-based sampling, no map cropping, no gyro terms).
