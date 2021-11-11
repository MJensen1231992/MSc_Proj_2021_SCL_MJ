import numpy as np
import shapely.geometry
import shapely.geometry as sg
import descartes
import matplotlib.pyplot as plt

from utm.conversion import from_latlon


def poly_intersection(robot_pose, landmark_xy, polygon):

    line = shapely.geometry.LineString([[robot_pose[0], robot_pose[1]], [landmark_xy[0], landmark_xy[1]]])

    polygons = (list(polygon.geoms))


    for i in range(len(polygons)):
        
        print(polygons[i])
        free = line.intersects(polygons[i])
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(*np.array(line).T, color='green', linewidth=3, solid_capstyle='round')
        # ax.add_patch(descartes.PolygonPatch(polygons[i], fc='blue', alpha=0.5))
        # ax.axis('equal')
        # plt.show()

        if free:
            continue
        else:
            # print("Polygon between robot and landmark!")
            return False
        
    return True        
