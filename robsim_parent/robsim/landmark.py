from random import random

from robsim.primitives import Point

class FeatureType:
    def __init__(self, 
            name: str,
            source: str = '', 
            description: str = '', 
            detection_range: float = None, 
            recall: float = 1.0, 
            visibility: float = 1.0, 
            error: float = 0.0, 
            color: str = '#000000'):

        self.name = name
        self.source = source
        self.description = description
        self.detection_range = detection_range
        self.visibility = visibility
        self.recall = recall
        self.error = error
        self.color = color

class Landmark(Point):
    def __init__(self, x, y, name):
        super(Landmark, self).__init__(x, y)

        if not name in landmark_types:
            raise(Exception('Fisk'))

        self.type = landmark_types[name]

landmark_types = {
    'gd_lf': FeatureType(
        name = 'gd_lf',
        source = 'GeoDanmark',
        description = 'Light Fixture', 
        detection_range = 10.0, 
        recall = 0.646,
        visibility=0.922,
        error = 0.48,
        color = '#e0de38')
}