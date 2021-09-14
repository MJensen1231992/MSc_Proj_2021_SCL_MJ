from math import sqrt, cos, sin, atan2

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __eq__(self, pt):
        if pt == None: return False
        return (self.x == pt.x) and (self.y == pt.y)

    def __ne__(self, pt):
        if pt == None: return True
        return (not isinstance(pt, Point)) or (self.x != pt.x) or (self.y != pt.y)

    def __iter__(self):
        return iter([self.x, self.y])

    def norm(self):
        return sqrt(self.x * self.x + self.y * self.y)

    def normalize(self):
        n = self.norm()
        return Point(self.x / n, self.y / n)

    def __add__(self, pt):
        return Point(self.x + pt.x, self.y + pt.y)

    def __sub__(self, pt):
        return Point(self.x - pt.x, self.y - pt.y)

    def __mul__(self, k):
        return Point(self.x * k, self.y * k)

    def __truediv__(self, k):
        return Point(self.x / k, self.y / k)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def as_polar(self):
        d = sqrt(self.x**2 + self.y**2)
        a = atan2(self.y, self.x)

        return a, d

    @staticmethod
    def from_polar(a, d):
        x = cos(a) * d
        y = sin(a) * d

        return Point(x, y)

    # Returns a relative point in respect to another pose.
    def relative(self, pose):
        x = self.x - pose.x
        y = self.y - pose.y

        x, y = x * cos(-pose.theta) - y * sin(-pose.theta), \
               x * sin(-pose.theta) + y * cos(-pose.theta)

        return Point(x, y)

    # Returns the global coordinate in respect to another coordinate.
    def absolute(self, pose):
        x, y = self.x * cos(pose.theta) - self.y * sin(pose.theta) + pose.x, \
               self.x * sin(pose.theta) + self.y * cos(pose.theta) + pose.y

        return Point(x, y)

class Pose:
    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta

    def __str__(self):
        return f'({self.x:.1f}, {self.y:.1f}, {self.theta:.1f})'

    def __iter__(self):
        return iter([self.x, self.y, self.theta])

    # Returns a relative coordinate in respect to another coordinate. (basically self - other including rotation)
    def relative(self, other):
        x = self.x - other.x
        y = self.y - other.y

        theta = self.theta - other.theta
        x, y = x * cos(-other.theta) - y * sin(-other.theta), \
               x * sin(-other.theta) + y * cos(-other.theta)

        return Pose(x, y, theta)

    # Returns the global coordinate in respect to another coordinate. (basically self + other including rotation)
    def absolute(self, other):
        theta = other.theta + self.theta

        x, y = self.x * cos(other.theta) - self.y * sin(other.theta), \
               self.x * sin(other.theta) + self.y * cos(other.theta)

        x = x + other.x
        y = y + other.y

        return Pose(x, y, theta)

    def __eq__(self, other):
        if other == None:
            return False

        return (self.x == other.x) and (self.y == other.y) and (self.theta == other.theta)

    def __ne__(self, other):
        return not (self == other)

    def __add__(self, other):
        return Pose(self.x + other.x, self.y + other.y, self.theta + other.theta)

    def __sub__(self, other):
        return Pose(self.x - other.x, self.y - other.y, self.theta - other.theta)

    def __mul__(self, k):
        return Pose(self.x * k, self.y * k, self.theta * k)

    def __truediv__(self, k):
        return Pose(self.x / k, self.y / k, self.theta / k)

    def set_x(self, val):
        self.x = val
        return self

    def set_y(self, val):
        self.y = val
        return self

    def set_theta(self, val):
        self.theta = val
        return self

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_theta(self):
        return self.theta
    
    def to_json(self):
        return [self.x, self.y] + [t for t in [self.theta] if t != 0.0]

    @staticmethod
    def from_json(obj):   
        return Pose(x=obj[0], y=obj[1], theta=(obj[2:] + [0])[0])
    