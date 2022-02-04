import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.npyio import save
from shapely import geometry
import utm
import pandas as pd
import json

import seaborn as sns
sns.set(rc={'figure.figsize':(12,6)})

# Description of data from Capra saved in CSV files
#   geo_odometry : It is the robot's filtered position in longitude-latitude-altitude and absolute EMU heading
#   odometry: Robot's wheel odometry in its local Cartesian system
#   odometry_fixed: Robot's filtered odometry in its local Cartesian system

class CSV_Reader:

    def __init__(self, geo_odometry: str, odometry: str, odometry_fixed: str, save_data: bool, if_plot: bool):
        self.geo_odometry = geo_odometry
        self.odometry = odometry
        self.odometry_fixed = odometry_fixed

        self.save_data = save_data
        self.if_plot = if_plot

    def import_csv(self, descriptor: str):
        
        self.descriptor = descriptor

        dict = {"geo": self.geo_odometry,
                "odo": self.odometry,
                "fixed": self.odometry_fixed,
                'all': [self.geo_odometry, self.odometry, self.odometry_fixed]}
        
        iter = 3 if descriptor == 'all' else 1

        if descriptor == 'all':
            dscptr = ['geo', 'odo', 'fixed']
    
        for i in range(iter):
            if self.descriptor == 'all':
                data = pd.read_csv(dict[dscptr[i]])
                descriptor = dscptr[i] 
            elif self.descriptor != 'all':
                data = pd.read_csv(dict[descriptor])

            if descriptor == 'geo':
                # GNSS
                df = pd.DataFrame(data, columns=['_start', '_stop', '_time', 'latitude', 'longitude'])
                lat = df['latitude']
                lon = df['longitude']

                x = []; y = []

                for lat, lon in zip(lat,lon):
                    
                    xUTM, yUTM = self.to_utm(lat, lon)
                    x.append(xUTM)
                    y.append(yUTM)
                
                df['latitude'] = x
                df['longitude'] = y
                df = df.rename(columns={'latitude': 'x', 'longitude': 'y'})
                
                gnss = (x, y)

            elif descriptor == 'odo':
                # Normal odometry
                df = pd.DataFrame(data, columns=['_start', '_stop', '_time', 'pose-orientation-yaw', 'pose-position-x', 'pose-position-y'])
                x = df['pose-position-x']
                y = df['pose-position-y']
                th = df['pose-orientation-yaw']
                # If "all" -> translating odometry to UTM32 format
                odo = (x,y,th) if self.descriptor == 'odo' else (x+gnss[0][0], y+gnss[1][0], th)

            elif descriptor == 'fixed':
                # Fixed odometry
                df = pd.DataFrame(data, columns=['_start', '_stop', '_time', 'pose-orientation-yaw', 'pose-position-x', 'pose-position-y'])
                x = df['pose-position-x']
                y = df['pose-position-y']
                th = df['pose-orientation-yaw']
                # If "all" -> translating odometry to UTM32 format
                fixed = (x,y,th) if self.descriptor == 'fixed' else (x+gnss[0][0], y+gnss[1][0], th)

            if self.save_data:
                dir = 'g2o_generator/capra/data/'
                results = df.to_json(orient='table')
                parsed = json.loads(results)
                with open(dir+descriptor+'.json', 'w') as f:
                    json.dump(parsed, f, indent=4)

        if self.descriptor == 'geo':
            odo = None
            fixed = None
        elif self.descriptor == 'odo':
            gnss = None
            fixed = None
        elif self.descriptor == 'fixed':
            gnss = None
            odo = None
        else:
            pass
        
        # dist = self.distance_traveled(np.array(fixed[0]), np.array(fixed[1]))
        # print(f"Distance traveled: {dist:.2f} m")

        if self.if_plot:
            self.plot_route(self.descriptor, gnss=gnss, odo=odo, fixed=fixed)

        if self.descriptor == 'all':
            return gnss, odo, fixed
        

    def plot_route(self, descriptor: str, odo: tuple=None, fixed: tuple=None, gnss: tuple=None):

        plt.figure()

        if descriptor == 'geo':
            plt.scatter(np.array(gnss[0])-gnss[0][0], np.array(gnss[1])-gnss[1][0], marker='.', color='green', label='GNSS')
        elif descriptor == 'odo':
            plt.plot(np.array(odo[0])-odo[0][0], np.array(odo[1])-odo[1][0], color='blue', linewidth=2, label='Odometry')
            circle1 = plt.Circle((-3.83, 1.78), 1, fill=False, label='Discontinuity', color='red', linewidth=4, zorder=5)
            plt.gca().add_patch(circle1)
            circle1 = plt.Circle((10.47, 6.57), 1, fill=False, color='red', linewidth=4, zorder=5)
            plt.gca().add_patch(circle1)
        elif descriptor == 'fixed':
            plt.plot(np.array(fixed[0]), np.array(fixed[1]), color='blue', marker='o', label='Fixed odometry')
        elif descriptor == 'all':
            plt.scatter(np.array(gnss[0]), np.array(gnss[1]), color='blue', marker='x', label='GNSS', alpha=0.3)
            plt.plot(np.array(odo[0]), np.array(odo[1]), color='red', marker='o', label='Odometry', alpha=0.5)
            # plt.quiver(np.array(fixed[0]), np.array(fixed[1]), np.cos(np.array(fixed[2])), np.sin(np.array(fixed[2])), angles='xy', color='red', scale=20)
            plt.plot(np.array(fixed[0]), np.array(fixed[1]), color='green', marker='o', label='Fixed odometry')

        plt.legend(fontsize=18, frameon=False)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('x (m)', fontsize="x-large")
        plt.ylabel('y (m)', fontsize="x-large")
        plt.tight_layout()

        plt.savefig('g2o_generator/1st_iter_odo.png')

        plt.show()


    @staticmethod
    def to_utm(lat, lon):
        xutm, yutm, _, _ = utm.from_latlon(lat, lon, force_zone_number=32, force_zone_letter='U')
        return xutm, yutm

    @staticmethod
    def distance_traveled(x, y):

        x = np.vstack(x)
        y = np.vstack(y)
        route = np.hstack((x,y))
        distance = sum([np.linalg.norm(pt1 - pt2) for pt1, pt2 in zip(route[:-1], route[1:])])

        return distance

def main():

    geo_odometry = "g2o_generator/capra/data/raw_data/geo_odometry.csv"
    odometry = "g2o_generator/capra/data/raw_data/odometry.csv"
    odometry_fixed = "g2o_generator/capra/data/raw_data/odometry_fixed.csv"

    capra = CSV_Reader(geo_odometry, odometry, odometry_fixed, save_data=False, if_plot=True)
    odometry_pdf = capra.import_csv('odo')

if __name__ == "__main__":
    main()