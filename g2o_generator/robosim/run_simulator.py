import sys
sys.path.append('g2o_generator/GIS_Extraction')

from lib.utility import *
from lib.helpers import *

import generate_g2o as gg
import create_world as cw

# importing external files that are not generated in this simulator
filenamePoints = 'g2o_generator/GIS_Extraction/data/aarhus_features_v2.csv'
filenamePoly = 'g2o_generator/GIS_Extraction/data/aarhus_polygons_v2.csv'
landmarksFile = './g2o_generator/GIS_Extraction/landmarks/landmarks_w_types.json'
route_name = 'brbr'

# Variables
FOV = 120 # Degrees
LM_RANGE = 20 # Meters
ODO_RANGE = 2 # Meters
set_xlim = [574714, 575168]
set_ylim = [6222368, 6222683]

# One landmark check
# set_xlim = [574689-200, 574689+150]
# set_ylim = [6222470-80, 6222470+80]

# set_xlim = [57500, 575168]
# set_ylim = [6222468, 6222683]


# Making robot path
show = cw.world(filenamePoints, filenamePoly, landmarksFile, route_name=route_name, save_path=True,\
                load_path=False, set_xlim=set_xlim, set_ylim=set_ylim)
odometry_file = show.make_robot_path()

# Generating g2o file
genG2O = gg.g2o(odometry_file, filenamePoints, filenamePoly, landmarksFile, LM_RANGE, ODO_RANGE, FOV)
genG2O.generate_g2o(corruption_type="none", plot=True, plot_constraints=False, plot_robot_heading=True)
