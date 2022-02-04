import seaborn as sns
sns.set_theme()
import sys
sys.path.append('g2o_generator/GIS_Extraction')
sys.path.append('g2o_generator/robosim')

import generate_g2o as gg
import create_world as cw

# importing external files that are not generated in this simulator
filenamePoints = 'g2o_generator/GIS_Extraction/data/aarhus_features_v2.csv'
filenamePoly = 'g2o_generator/GIS_Extraction/data/aarhus_polygons_v2.csv'
landmarksFile = './g2o_generator/GIS_Extraction/landmarks/landmarks_w_types.json'
route_name = 'tester'

# Variables
FOV = 60 # Degrees | 45 is default
LM_RANGE = 20 # Meters | 15 is default
ODO_RANGE = 5 # Meters | 2 is default

set_xlim = [574714, 575168]
set_ylim = [6222368, 6222683]

# If corrupt dataset
corrupt_dataset = True

# Apparently it will be n_outliers + 1...
n_outliers = 100

# For analysis, select bearing only, landmark only OR pose-pose constraints only
landmark_only = False
bearing_only = False
pose_only = True

# Making robot path
show = cw.world(filenamePoints, filenamePoly, landmarksFile, route_name=route_name, save_path=True,\
                load_path=False, set_xlim=set_xlim, set_ylim=set_ylim)

# Create new odometry and ground truth file
odometry_file_gt, odometry_file_noise = show.make_robot_path()


# Load odometry file and ground truth route:
# odometry_file_gt = 'g2o_generator/robosim/data/robopath/route1_gt.json' # Ground truth route
# odometry_file_noise = 'g2o_generator/robosim/data/robopath/route1_noise.json' # Odom route

# Generating g2o file
genG2O = gg.g2o(odometry_file_gt, odometry_file_noise, filenamePoints, filenamePoly, landmarksFile,\
                landmark_only, bearing_only, pose_only, corrupt_dataset, LM_RANGE, ODO_RANGE, FOV, n_outliers)

# Corruption types: 'none' or 'random_grouped'
genG2O.generate_g2o(corruption_type="random_grouped", plot=True, plot_outliers=False, plot_constraints=True, plot_robot_heading=True)
