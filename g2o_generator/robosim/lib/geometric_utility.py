
def polygon_stacker(polygon):

    polygons = (list(polygon.geoms))

    poly_stack = []

    for i in range(len(polygons)):
        
        poly = polygons[i]
        poly_stack.append(poly)
        

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(*np.array(line).T, color='green', linewidth=3, solid_capstyle='round')
        # ax.add_patch(descartes.PolygonPatch(polygons[i], fc='blue', alpha=0.5))
        # ax.axis('equal')
        # plt.show()

    return poly_stack

def p_intersection(line, polygons):

    poly_stack = polygon_stacker(polygons)
    
    for p in poly_stack:
        
        if line.intersects(p.boundary):
            return False
        else:
            continue

    return True
