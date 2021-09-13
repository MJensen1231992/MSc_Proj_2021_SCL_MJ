

class CSV:

    feature_list = ['osm_id','highway','']

    def __init__(self, filename: str):
        self.filename = filename
        pass

    def read(self):
        with open(self.filename, 'r') as f:
            arr = []
            crash = []


