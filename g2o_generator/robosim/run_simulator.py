import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('g2o_generator/GIS_Extraction')

from lib.utility import *

import generate_g2o as gg
import create_world as cw
import csv_reader as cr


# importing external files that are not generated in this simulator
filenamePoints = 'g2o_generator/GIS_Extraction/data/aarhus_features_v2.csv'
filenamePoly = 'g2o_generator/GIS_Extraction/data/aarhus_polygons_v2.csv'
landmarksFile = './g2o_generator/GIS_Extraction/landmarks/landmarks_w_types.json'

# Variables
LM_RANGE = 10
ODO_RANGE = 1

# Using imported data
aarhus = cr.read_csv(filenamePoints, filenamePoly)
landmarks, _ = aarhus.read()

# Making robot path
show = cw.world(filenamePoints, filenamePoly, landmarksFile, save_path=True, load_path=False)
show.make_robot_path()

odometry_file = './g2o_generator/robosim/data/robopath/Aarhus_path1.json'

# Generating g2o file
genG2O = gg.g2o(odometry_file, filenamePoints, filenamePoly, landmarksFile, LM_RANGE, ODO_RANGE)
genG2O.generate_g2o(plot=True, plot_constraints=False, plot_robot_heading=True)





