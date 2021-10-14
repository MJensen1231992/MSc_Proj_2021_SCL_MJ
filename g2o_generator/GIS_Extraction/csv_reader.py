import csv
import os
import sys
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.function_base import linspace
import shapely.geometry as sg
import utm
from descartes import PolygonPatch
from utm.conversion import from_latlon

sys.path.append('g2o_generator/robosim')
from lib.utility import *

class read_csv():

    """
    When exporting from QGIS: 
        points: should be exported comma seperated, geometry: AS_XY, separator: COMMA, only select other_tags
        multipolygons: geometry: AS_WKT, separator: COMMA, select only "buildings"

    """

    def __init__(self, filename_points: str, filename_poly: str):
        self.filename_points = filename_points
        self.filename_poly = filename_poly
        pass

    def read(self, readPoints: bool = 1, readPoly: bool = 1):
        self.rowPoints = {}
        self.rowPoly = []

        self.features = ['"tree"', 'traffic_signals', 'bin', 'bench', 'fountain', 'statue']#, 'bump']
        # rowPoints will have format: [X,Y,OTHER_TAG]
        #                             [. .      .]
        #                             [. .      .]
        if readPoints:
            with open(self.filename_points, 'r') as f:
                csv_file = csv.reader(f)
                for id, row in enumerate(csv_file):
                    if id > 0:
                        row = row[0:3]

                        # Converting to UTM32
                        lon = float(row[0]); lat = float(row[1])
                        xutm, yutm, _, _ = from_latlon(lat, lon)
                        
                        # Looking for landmark features
                        for feature in self.features:
                            if feature in row[2]:
                                self.rowPoints.setdefault(feature, [])
                                self.rowPoints[feature].append(([xutm, yutm]))

        # Saveing landmarks as json                            
        if True:
            save_to_json(self.rowPoints,'./g2o_generator/GIS_Extraction/landmarks/landmarks_w_types.json', indent=4)

        # Polygons contains polygons and if they are buildings
        unwanted = ["MULTIPOLYGON","(",")"]
        if readPoly:
            with open(self.filename_poly, 'r') as f:
                csv_file = csv.reader(f)
                for id, rowPoly in enumerate(csv_file):
                    if id > 0:
                        ###
                        # Cleaning up in lists to only contain x,y points for polygons
                        for elem in unwanted:
                            rowPoly = [s.replace(elem,"") for s in rowPoly]

                        rowPoly = [s.replace(" ",",") for s in rowPoly]
                        rowPoly = rowPoly[0][1:]
                        rowPoly = [i for i in rowPoly.split(',')]
                        rowPoly = np.array(rowPoly, dtype=np.float64)

                        self.rowPoly.append(rowPoly)
                        ###
                
                self.rowPoly = np.array(self.rowPoly, dtype=object)

        return self.rowPoints, self.rowPoly

        
    def squeeze_polygons(self, polygon, plot: bool):
        
        poly_stack = []
        poly_area = []

        for poly in polygon:

            y = poly[0::2]; x = poly[1::2]
            x, y, _, _ = from_latlon(np.array(x), np.array(y))
            points = np.stack((x,y), axis=-1)
            self.polygon_final = sg.Polygon(np.squeeze(points))

            # Debug
            area = self.polygon_final.area
            
            poly_area.append(area)
            if area < 100000:
                poly_stack.append(self.polygon_final)

        # print('min area of poly: {}, max area: {}, \nmean area: {}'.format(min(poly_area),max(poly_area), np.mean(poly_area)))
        if plot:
            fig = plt.figure()
            for poly in poly_stack:
                ax = fig.add_subplot()
                ax.add_patch(PolygonPatch(poly.buffer(0)))
        else:
            return poly_stack

    def plot_map(self, show: bool = 0, save: bool = 0, filename: str = 'g2o_generator/GIS_Extraction/plots/GIS_map'):
        
        scalex, scaley = [], []

        self.squeeze_polygons(self.rowPoly, plot=True)

        landmarks = self.rowPoints.items()
        cmap = ListedColormap(["cyan", "darkblue", "magenta", "springgreen", "orange"])
        ax = plt.gca()
        colors = cmap(np.linspace(0, 1, len(landmarks)))

        for idx, (key, landmark) in enumerate(landmarks):
            fx, fy = [], []
            for pos in landmark:
                x, y = pos
                scalex.append(x), scaley.append(y)
                fx.append(x), fy.append(y)
            ax.scatter(fx, fy, zorder=2, s=10, color=colors[idx], label=key)

        # plt.xlim(min(scalex), max(scalex))
        # plt.ylim(min(scaley), max(scaley))
        plt.legend()

        if save:
            i = 0
            while os.path.exists('{}{:d}.png'.format(filename, i)):
                i += 1
            plt.savefig('{}{:d}.png'.format(filename, i), format='png')

        if show:
            plt.show()

    @staticmethod
    def to_utm(lat, lon):

        xutm, yutm = utm.from_latlon(lat, lon, 'U', 32)
        return xutm, yutm

def main():
    filenamePoints = 'g2o_generator/GIS_Extraction/data/aarhus_features_v2.csv'
    filenamePoly = 'g2o_generator/GIS_Extraction/data/aarhus_polygons_v2.csv'
    aarhus = read_csv(filenamePoints, filenamePoly)
    _, _ = aarhus.read()
    aarhus.plot_map(save=0, show=False)


if __name__ == "__main__":
    main()