import time
import sys
sys.path.append('g2o_generator/GIS_Extraction')

from lib.utility import *
from lib.helpers import *

import generate_g2o as gg
import create_world as cw
import csv_reader as cr

# importing external files that are not generated in this simulator
filenamePoints = 'g2o_generator/GIS_Extraction/data/aarhus_features_v2.csv'
filenamePoly = 'g2o_generator/GIS_Extraction/data/aarhus_polygons_v2.csv'
landmarksFile = './g2o_generator/GIS_Extraction/landmarks/landmarks_w_types.json'
route_name = 'brbr'

# Variables
LM_RANGE = 10
ODO_RANGE = 1

# Making robot path
show = cw.world(filenamePoints, filenamePoly, landmarksFile, route_name=route_name, save_path=True, load_path=False)
odometry_file = show.make_robot_path()

# Generating g2o file
genG2O = gg.g2o(odometry_file, filenamePoints, filenamePoly, landmarksFile, LM_RANGE, ODO_RANGE)
genG2O.generate_g2o(plot=True, plot_constraints=False, plot_robot_heading=True)
