import numpy as np

from math import atan2
from numpy.linalg import inv


class Graph:
    def __init__(self, nodes, edges, state, lkt):
        self.nodes = nodes
        self.edges = edges
        self.state = state
        self.lkt = lkt

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
                    landmark = np.array(inline[2:5], dtype=np.float64)
                    nodes[id] = landmark
                    
                elif inline[0] == 'VERTEX_SE2':
                    id = int(inline[1])
                    pose = np.array(inline[2:4], dtype=np.float64)
                    nodes[id] = pose
                
                elif inline[0] == 'EDGE_SE2_XY':
                    from_node_id = int(inline[1])
                    to_node_id = int(inline[2])
                    meas = np.array(inline[3:5], dtype=np.float64)
                    elems = np.array(inline[5:8], dtype=np.float64)
                    information_matrix = np.array([[elems[0], elems[1]],
                                                   [elems[1], elems[2]]])

                    edges.append(('Landmark',from_node_id, to_node_id, meas, information_matrix))

                elif inline[0] == 'EDGE_SE2':
                    from_node_id = int(inline[1])
                    to_node_id = int(inline[2])
                    meas = np.array(inline[3:6], dtype=np.float64)
                    elems = np.array(inline[6:], dtype=np.float64)
                    information_matrix = np.array([[elems[0], elems[1], elems[2]],
                                                   [elems[1], elems[3], elems[4]],
                                                   [elems[2], elems[4], elems[5]]])
                    
                    edges.append(('Pose',from_node_id, to_node_id, meas, information_matrix))
                
                else:
                    raise TypeError('No data')

        # Computing a lookup table and a state vector
        lkt, state = {}, []
        offset = 0
        for id in nodes:
            lkt.update({id: offset})
            offset = offset + len(nodes[id])
            state.append(nodes[id])
       

        print('Number of nodes: {} \nNumber of edges: {}'.format(len(nodes), len(edges)))

        return Graph(nodes, edges, state, lkt)

    def to_matrix(self, pose):
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
            
            if len(graph.nodes[id]) == 3:
                poses.append(graph.nodes[id])
            
            elif len(graph.nodes[id]) == 2:
                landmarks.append(graph.nodes[id])
        

        return poses, landmarks

    def global_error(self, graph):
        """ Computing the global error of the graph (pose-pose constraints and pose-landmark constraints) 
        http://ais.informatik.uni-freiburg.de/teaching/ws09/robotics2/pdfs/rob2-06-least-squares-slam.pdf"""

        err = 0
        
        # Pose-pose error
        for edge in graph.edges:
            if edge[0] == 'Pose':
                
                fromNodeId = graph.lkt[edge[1]]
                toNodeId = graph.lkt[edge[2]]
                xi = graph.state[fromNodeId:fromNodeId]
                xj = graph.state[toNodeId:toNodeId]
                zij = edge[3]




        return 0

def main():
    file = 'GraphSlam/data/sim_pose_landmark_data.g2o'

    read_file = Graph.read_G2O(file)

    # for edge in read_file.edges:
    #     if edge[0] == 'Pose':
    iter = 0
    for edge in read_file.edges:
            if edge[0] == 'Pose':
                
                
                fromNodeId = read_file.lkt[edge[1]]
                toNodeId = read_file.lkt[edge[2]]
                xi = read_file.state[fromNodeId:fromNodeId+3]
                xj = read_file.state[toNodeId:toNodeId+3]
                print(xi)
                zij = edge[3]
                # Xi_inv = inv(Graph.to_matrix(xi))
                # Xj = Graph.to_vector(xj)


    
if __name__ == '__main__' :
    main()

