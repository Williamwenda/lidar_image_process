# which bag to process and where its results go; every script reads this file
from pathlib import Path

######## Edit by hand ########
# BAG = "/home/wenda/data/lidar_intensity/asrl_data/ouster_data/rosbag2_2025_05_21-11_02_59" # parking
BAG = "/home/wenda/data/lidar_intensity/asrl_data/ouster_data/rosbag2_2025_05_21-10_46_12"  # dome
FOLDER_NAME = "utias_dome"   # results go to outputs/<FOLDER_NAME>/

######## Derived paths ########
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs" / FOLDER_NAME
ODOMETRY_DIR = OUTPUTS_DIR / "odometry"   # from run_odometry.py
PAIRS_DIR = OUTPUTS_DIR / "pairs"         # from make_pairs.py
