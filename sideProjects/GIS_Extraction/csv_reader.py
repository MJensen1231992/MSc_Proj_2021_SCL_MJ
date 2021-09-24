import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg
from descartes import PolygonPatch



class read_csv():

    """
    When exporting from QGIS: 
        points: should be exported comma seperated, geometry: AS_XY, separator: COMMA
        multipolygons: geometry: AS_WKT, separator: COMMA

    """

    def __init__(self, filename_points: str, filename_poly: str):
        self.filename_points = filename_points
        self.filename_poly = filename_poly
        pass

    def read(self, readPoints: bool = 1, readPoly: bool = 1):
        self.rowPoints = []
        self.rowPoly = []
        # rowPoints will have format: [X,Y,ID]
        #                             [. . . ]
        #                             [. . . ]
        if readPoints:
            with open(self.filename_points, 'r') as f:
                csv_file = csv.reader(f)
                for id, row in enumerate(csv_file):
                    if id > 0:
                        row = row[0:3]
                        self.rowPoints.append(row)
            
            self.rowPoints = np.array(self.rowPoints, dtype=np.float64)
            # print(self.rowPoints)
        unwanted = ["MULTIPOLYGON","(",")","yes","1550051"]

        if readPoly:
            with open(self.filename_poly, 'r') as f:
                csv_file = csv.reader(f)
                for id, rowPoly in enumerate(csv_file):
                    if id > 0:
                        ###
                        # GRIM KODE JEG VED DET GODT :((
                        for elem in unwanted:
                            rowPoly = [s.replace(elem,"") for s in rowPoly]

                        rowPoly = [s.replace(" ",",") for s in rowPoly]
                        rowPoly = rowPoly[0]
                        rowPoly = rowPoly[1:]
                        rowPoly = [i for i in rowPoly.split(',')]
                        rowPoly = np.array(rowPoly, dtype=np.float64)

                        self.rowPoly.append(rowPoly)
                        ###
                
                self.rowPoly = np.array(self.rowPoly, dtype=object)
        return self.rowPoints, self.rowPoly

        
    def squeeze_polygons(self, polygon, plot: bool):
        

        poly_stack = []
        poly_area = []

        for id, poly in enumerate(polygon):
            x = poly[0::2]
            y = poly[1::2]
            points = np.stack((x,y), axis=-1)
            self.polygon_final = sg.Polygon(np.squeeze(points))

            # Debug
            area = self.polygon_final.area
            poly_area.append(area)
            if area < 1.0220626785217238e-05:
                poly_stack.append(self.polygon_final)

        #print('min area of poly: {}, max area: {}, \nmean area: {}'.format(min(poly_area),max(poly_area), np.mean(poly_area)))
        if plot:
            fig = plt.figure()
            for id, poly in enumerate(poly_stack):
                ax = fig.add_subplot()
                patch = PolygonPatch(poly_stack[id].buffer(0))
                ax.add_patch(patch)
        else:
            return poly_stack


    def export_landmarks(self, filename: str = 'sideProjects/GIS_Extraction/landmarks/landmarks_points.csv'):
        return np.savetxt(filename, self.rowPoints, delimiter=",")

    def plot_map(self, save: bool = 0, filename: str = 'sideProjects/GIS_Extraction/plots/GIS_map'):

        self.squeeze_polygons(self.rowPoly, plot=True)

        plt.scatter(self.rowPoints[:,0],self.rowPoints[:,1], zorder=2, s=10)
        frame = plt.gca()
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])

        plt.xlim([min(self.rowPoints[:,0]), max(self.rowPoints[:,0])])
        plt.ylim([min(self.rowPoints[:,1]), max(self.rowPoints[:,1])])
        
        if save:
            i = 0
            while os.path.exists('{}{:d}.png'.format(filename, i)):
                i += 1
            plt.savefig('{}{:d}.png'.format(filename, i), format='png')
        
        plt.show()

def main():
    filenamePoints = 'sideProjects/GIS_Extraction/data/aarhus_features.csv'
    filenamePoly = 'sideProjects/GIS_Extraction/data/aarhus_polygons.csv'
    aarhus = read_csv(filenamePoints, filenamePoly)
    _, rowpoints = aarhus.read()
    aarhus.plot_map(save=0)
    # aarhus.export_landmarks()

if __name__ == "__main__":
    main()