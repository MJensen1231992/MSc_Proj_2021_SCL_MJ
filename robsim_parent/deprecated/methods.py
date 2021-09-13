
from primitives import Point, Line
from utility import dist_l2

from math import pi, log, atan2, degrees, radians, cos, sin
from random import choice

def fit_line(pts, angle_threshold: float, dist_threshold: float, T: int = None, p: float = 0.99, e: float = 0.5):
    assert len(pts) > 1

    # https://www.youtube.com/watch?v=9D5rrtCC_E0 (3:30)
    if T == None:
        T = int(log(1 - p) / log(1 - (1 - e)**2))

    line = None
    inliers = []
    outliers = []
    for t in range(T):
        a = choice(pts)
        b = choice([pt for pt in pts if pt != a])

        temp_line = Line(a, b)

        temp_inliers = [pt for pt in pts if temp_line.get_distance(pt)[0] <= dist_threshold]

        inlier_groups = []
        current_group = [temp_inliers[0]]
        temp_inliers = temp_inliers[1:]

        while len(temp_inliers) > 0:
            points = [(a, b) for a in temp_inliers for b in current_group]
            dists = [get_angular_difference(Point(0, 0), a, b) for a in temp_inliers for b in current_group]

            if degrees(min(dists)) < angle_threshold:
                a, b = points[dists.index(min(dists))]

                temp_inliers.remove(a)
                current_group.append(a)

            else:
                inlier_groups.append(current_group)
                current_group = [temp_inliers[0]]
                temp_inliers = temp_inliers[1:]

        inlier_groups.append(current_group)

        lengths = [len(group) for group in inlier_groups]
        temp_inliers = inlier_groups[lengths.index(max(lengths))]  

        temp_outliers = [pt for pt in pts if not pt in temp_inliers]

        if len(temp_inliers) > 1 and len(temp_inliers) > len(inliers):
            line = temp_line
            inliers = temp_inliers
            outliers = temp_outliers

    return line, inliers, outliers

def angle_diff(a, b):
    return min(abs(a - b), abs(a - b + 2*pi), abs(a - b - 2*pi))

def is_between(angle, a_from, a_to, d: float = 0):
    if a_from < a_to:
        return a_from - d <= angle <= a_to + d
    return a_to - d <= angle <= a_from + d

def get_angular_difference(origin: Point, a: Point, b: Point, t: str = 'radians'):
    v = [atan2(pt.y - origin.y, pt.x - origin.x) for pt in [a, b]]
    d = angle_diff(*v)

    if t == 'degrees':
        return degrees(d)
    return d

def get_corners(pts, angle_threshold: float, pt_threshold: int, dist_threshold: float, T: int = None, p: float = 0.995, e: float = 0.5):
    lines = []

    outliers = [pt for pt in pts]
    while len(outliers) >= pt_threshold:
        line, inliers, outliers = fit_line(outliers, 
            angle_threshold=angle_threshold,
            dist_threshold=dist_threshold, 
            T=T, 
            p=p, 
            e=e)

        if len(inliers) < pt_threshold:
            break

        #line = Line.from_points(inliers)

        lines.append((line, inliers))

    corners = []

    for i, (a, inliers_a) in enumerate(lines):
        for (b, inliers_b) in lines[i + 1:]:
            r = a.get_intersection(b)
            if r == None:
                continue

            p, t, u = r

            dists_a = [get_angular_difference(Point(0,0), p, inlier, t='degrees') for inlier in inliers_a]
            dists_b = [get_angular_difference(Point(0,0), p, inlier, t='degrees') for inlier in inliers_b]

            if max(min(dists_a), min(dists_b)) > angle_threshold / 0.95:
                continue

            corners.append(p)

    line_pts = [pt for line, pts in lines for pt in pts]

    for pt_a, pt_b, pt_c in zip(pts[:-2], pts[1:-1], pts[2:]):
        if pt_b not in line_pts:
            continue

        angle_a, angle_b, angle_c = atan2(pt_a.y, pt_a.x), atan2(pt_b.y, pt_b.x), atan2(pt_c.y, pt_c.x)
        dist_a, dist_b, dist_c = dist_l2(Point(0,0), pt_a), dist_l2(Point(0,0), pt_b), dist_l2(Point(0,0), pt_c)

        if abs(angle_a - angle_b) < radians(1.25 / 0.95) and abs(angle_c - angle_b) < radians(2 * 1.25 / 0.95) and pt_a in line_pts:
            line_a = Line(Point(0, 0), Point(cos(angle_c), sin(angle_c)))
            pred_a = Line(pt_a, pt_b).get_intersection(line_a)[0]
            d_pred_a = dist_l2(Point(0,0), pred_a)

            if dist_c > d_pred_a / 0.9:
                corners.append(pt_b)
                continue
              
        if abs(angle_c - angle_b) < radians(1.25 / 0.95) and abs(angle_a - angle_b) < radians(2 * 1.25 / 0.95) and pt_c in line_pts:  
            line_c = Line(Point(0, 0), Point(cos(angle_a), sin(angle_a)))
            pred_c = Line(pt_c, pt_b).get_intersection(line_c)[0]
            d_pred_c = dist_l2(Point(0,0), pred_c)

            if dist_a > d_pred_c / 0.9:
                corners.append(pt_b)
                continue
    
    return lines, corners

# Corner detection demo
# while True:
#     pts = choice(data)['scan']

#     max_range = 6

#     pts = [Point(pt[0], pt[1]) for pt in pts if dist_l2(Point(0, 0), pt) < max_range]

#     dists = [dist_l2(Point(0, 0), pt) for pt in pts]

#     lines, corners = get_corners(pts, angle_threshold=2, pt_threshold=4, dist_threshold=0.02)

#     plt.plot((0,), (0,), 'o', color='r')

#     for pt in pts:
#         plt.plot((0, pt.x), (0, pt.y), '--', color='black')

#     plt.plot([pt.x for pt in pts], [pt.y for pt in pts], 'x')

#     for (line, inliers) in lines:
#         plt.axline((line.a.x, line.a.y), (line.b.x, line.b.y))
#         plt.plot([pt.x for pt in inliers], [pt.y for pt in inliers], 'o')

#     plt.plot([pt.x for pt in corners], [pt.y for pt in corners], 'x', color='r')

#     ax = plt.gca()
#     ax.set_aspect(1)

#     plt.show()

# Line detection demo
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a = 0.1
    b = 2
    N = 20

    xs = [100 * i / N for i in range(N)]
    ys = [a*x + b + random() for x in xs]

    pts = [Point(x, y) for x, y in zip(xs, ys)]

    line, inliers, outliers = fit_line(pts, threshold=1)

    plt.axline((line.a.x, line.a.y), (line.b.x, line.b.y))
    plt.plot([pt.x for pt in inliers], [pt.y for pt in inliers], 'o')
    plt.plot([pt.x for pt in outliers], [pt.y for pt in outliers], 'x')

    plt.show()