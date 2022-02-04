
def polygon_stacker(polygon):
    """
    Helper function for p_intersection. Stacks polygons.
    """    
    polygons = (list(polygon.geoms))

    poly_stack = []

    for i in range(len(polygons)):
        
        poly = polygons[i]
        poly_stack.append(poly)
        

    return poly_stack

def p_intersection(line, polygons):
    """Line of sight calculator, checking if polygons are in the way to landmarks.

    Args:
        line (linestring object): Line between robot and landmark
        polygons (polygons): Polygons (buildings in the robot env)

    Returns:
        Bool: True if clear line of sight, false if not clear
    """    
    poly_stack = polygon_stacker(polygons)
    
    for p in poly_stack:
        
        if line.intersects(p.boundary):
            return False
        else:
            continue

    return True
