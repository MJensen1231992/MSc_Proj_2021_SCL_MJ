import numpy as np
from math import atan2

from primitives import Point, Pose


class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    @staticmethod
    def read_G2O(filename: str):
        
        """ Reading and loading a g2o file containing both pose-pose constraints and pose-landmark constraints """

        nodes = {}
        edges = []

        with open(filename, 'r') as f:
            for line in f:
                inline = line.split()

                if inline[0] == 'VERTEX_XY':
                    id = int(inline[1])
                    landmark = np.array(inline[2:], dtype=np.float64)
                    nodes[id] = Point(landmark[0], landmark[1])
                    
                elif inline[0] == 'VERTEX_SE2':
                    id = int(inline[1])
                    pose = np.array(inline[2:], dtype=np.float64)
                    nodes[id] = Pose(pose[0], pose[1], pose[2])
                
                elif inline[0] == 'EDGE_SE2_XY':
                    from_node_id = int(inline[1])
                    to_node_id = int(inline[2])
                    meas = np.array(inline[3:5], dtype=np.float64)
                    elems = np.array(inline[5:8], dtype=np.float64)
                    information_matrix = np.array([[elems[0], elems[1]],
                                                   [elems[1], elems[2]]])

                    edges.append(('Landmark',from_node_id, to_node_id, Point(meas[0], meas[1]), information_matrix))

                elif inline[0] == 'EDGE_SE2':
                    from_node_id = int(inline[1])
                    to_node_id = int(inline[2])
                    meas = np.array(inline[3:6], dtype=np.float64)
                    elems = np.array(inline[6:], dtype=np.float64)
                    information_matrix = np.array([[elems[0], elems[1], elems[2]],
                                                   [elems[1], elems[3], elems[4]],
                                                   [elems[2], elems[4], elems[5]]])
                    
                    edges.append(('Pose',from_node_id, to_node_id, Pose(meas[0], meas[1], meas[2]), information_matrix))
                
                else:
                    print('No datatype defined')

        print('Number of nodes: {} \nNumber of edges: {}'.format(len(nodes), len(edges)))

        return Graph(nodes, edges)

    def to_matrix(self, pose: Pose):
        """ Takes a vector and returns the homogeneous transformation Se(2) """

        return np.array([[np.cos(pose[2]), -np.sin(pose[2]), pose[0]], 
                         [np.sin(pose[2]),  np.cos(pose[2]), pose[1]], 
                         [0., 0., 1.]], dtype=np.float64)


    def to_vector(self, mat):
        """ Takes a tranformation and returns vector converted to Se(2) instance """
        
        th = atan2(mat[1, 0], mat[0, 0])

        return np.array([mat[0, 2], mat[1, 2]], th)


    def _get_poses_landmarks(self, graph):
        
        poses, landmarks = [], []

        for id in graph.nodes:
            
            if type(graph.nodes[id]) == Pose:
                poses.append(graph.nodes[id])
            
            elif type(graph.nodes[id]) == Point:
                landmarks.append(graph.nodes[id])
        

        return poses, landmarks

    def 

    
    
if __name__ == '__main__' :

    file = 'GraphSlam/data/sim_pose_landmark_data.g2o'

    read_file = Graph.read_G2O(file)


    poses, landmarks = [], []

    for id in read_file.nodes:
                
        if type(read_file.nodes[id]) == Pose:
            poses.append(read_file.nodes[id])
        
        elif type(read_file.nodes[id]) == Point:
            landmarks.append(read_file.nodes[id])

    print((poses[5]))
    print(landmarks[5])


