import csv
import os
import sys
from matplotlib.colors import ListedColormap

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg
import shapely.ops as so
import utm
from descartes import PolygonPatch
from utm.conversion import from_latlon

sys.path.append('g2o_generator/robosim')
from lib.utility import *
from lib.helpers import *

class read_csv():

    """
    When exporting from QGIS: 
        points: should be exported comma seperated, geometry: AS_XY, separator: COMMA, only select other_tags
        multipolygons: geometry: AS_WKT, separator: COMMA, select only "buildings"

    """

    def __init__(self, filename_points: str, filename_poly: str):
        self.filename_points = filename_points
        self.filename_poly = filename_poly
        self.scalex, self.scaley = [], []
        pass

    def read(self, readPoints: bool=1, readPoly: bool=1):
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

        # Adding manual landmark:
        self.rowPoints['bin'].append(([574689, 6222470]))
        self.rowPoints['bench'].append(([574577, 6222430]))

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

        
    def squeeze_polygons(self, polygon):
        
        poly_stack = []
        poly_area = []

        for poly in polygon:

            y = poly[0::2]; x = poly[1::2]
            x, y, _, _ = from_latlon(np.array(x), np.array(y))
            for x_val, y_val in zip(x, y):
                self.scalex.append(x_val), self.scaley.append(y_val)
            points = np.stack((x,y), axis=-1)
            self.polygon_final = sg.Polygon(np.squeeze(points))

            area = self.polygon_final.area

            poly_area.append(area)
            if area < 100000:
                poly_stack.append(self.polygon_final)
        
        return so.cascaded_union(poly_stack)
    

    def plot_map(self, landmarks_input, show: bool):
        
        # l = []
        # world_running = True

        cascaded_poly = self.squeeze_polygons(self.rowPoly)

        _, axs = plt.subplots()
        axs.set_aspect('equal', 'datalim')

        # world = pygame.display.set_mode((800,600))
        for geom in cascaded_poly.geoms:
            # print(geom)

            x_casc, y_casc = geom.exterior.xy
        #     for x_casc, y_casc in zip(x_casc, y_casc):
        #         l.append((x_casc, y_casc))
        #     # print(l)
            

            x_casc, y_casc = geom.exterior.xy
            axs.fill(x_casc, y_casc, alpha=0.5, fc='b', ec='none')

        # while world_running:
            
        #     for event in pygame.event.get():
        #         # If user press 'X' world will close
        #         if event.type == pygame.QUIT:
        #             world_running = False

        #     world.fill((255,255,255))
            # pygame.draw.polygon(world, (40,255,40), l)
            
            # pygame.draw.polygon(world, (0,0,0), l)
            # pygame.display.update()

        self.plot_landmarks(landmarks=landmarks_input)

        if show:
            plt.show()

    def plot_landmarks(self, landmarks):

        landmarks = landmarks.items()
        cmap = ListedColormap(["cyan", "darkblue", "magenta", "springgreen", "orange"])
        ax = plt.gca()
        colors = cmap(np.linspace(0, 1, len(landmarks)))

        for idx, (key, landmark) in enumerate(landmarks):
            fx, fy = [], []
            for pos in landmark:
                x, y = pos
                self.scalex.append(x), self.scaley.append(y)
                fx.append(x), fy.append(y)
            ax.scatter(fx, fy, zorder=2, s=10, color=colors[idx], label=key)
        
        plt.legend()

    @staticmethod
    def to_utm(lat, lon):

        xutm, yutm = utm.from_latlon(lat, lon, 'U', 32)
        return xutm, yutm

def main():
    filenamePoints = 'g2o_generator/GIS_Extraction/data/aarhus_features_v2.csv'
    filenamePoly = 'g2o_generator/GIS_Extraction/data/aarhus_polygons_v2.csv'
    aarhus = read_csv(filenamePoints, filenamePoly)
    landmarks, _ = aarhus.read()
    aarhus.plot_map(landmarks, show=True)


if __name__ == "__main__":
    main()