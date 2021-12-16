from math import inf
import numpy as np
from itertools import combinations
from lib.least_squares import LeastSquares as LS
from lib.triangulation import Triangulation as TRI
from collections import namedtuple, defaultdict
from helper import from_uppertri_to_full, wrap2pi

def load_g2o_graph(filename, noBearing=True):#, firstMeas=True):
    
    print(f"Loading file: {filename[15:-20]}")

    Edge = namedtuple(
        'Edge', ['Type', 'nodeFrom', 'nodeTo', 'poseMeasurement', 'information'] # g2o format of files.
    )
    edges = []
    nodes = {}
    nodeTypes = {}
    lm_status = {}
    initial_b_guess = False
    initial_qualified_guess = True

 
    with open(filename, 'r') as file:
        for line in file:
            data = line.split() # splits the columns
            
            if data[0] == 'VERTEX_SE2':
                
                nodeType = 'VSE2'
                nodeId = int(data[1])
                pose = np.array(data[2:5],dtype=np.float64)
                nodes[nodeId] = pose
                nodeTypes[nodeId] = nodeType

            elif data[0] == 'VERTEX_XY':
                if filename[15:-20] == 'ground_truth':
                    nodeType = 'VXY'
                    nodeId = int(data[1])
                    landmark = np.array(data[2:4],dtype=np.float64)  
                    nodes[nodeId] = landmark
                    nodeTypes[nodeId] = nodeType
                else:
                    continue

            elif data[0] == 'VERTEX_GPS':
                continue
                nodeType = 'VGPS'
                nodeId = int(data[1])
                gps_point = np.array(data[2:4],dtype=np.float64)
                nodes[nodeId] = gps_point
                nodeTypes[nodeId] = nodeType

            elif data[0] == 'EDGE_SE2':
                #print('hej P')
                
                Type = 'P' # pose type
                nodeFrom = int(data[1])
                nodeTo = int(data[2])
                poseMeasurement = np.array(data[3:6], dtype=np.float64)
                upperTriangle = np.array(data[6:12], dtype=np.float64)
                # information = np.array(
                #     [[upperTriangle[0], upperTriangle[1], upperTriangle[4]],
                #      [upperTriangle[1], upperTriangle[2], upperTriangle[5]],
                #      [upperTriangle[4], upperTriangle[5], upperTriangle[3]]]) #Toro
                information = from_uppertri_to_full(upperTriangle,3)
                #print(f"shape pose{np.shape(information)}")
               #print(f"shape pose posemeas{np.shape(poseMeasurement)}")
                edge = Edge(Type, nodeFrom, nodeTo, poseMeasurement, information)
                edges.append(edge)
            

            elif data[0] == 'EDGE_SE2_XY':
                
                Type = 'L' #landmark type
                nodeFrom = int(data[1])
                nodeTo = int(data[2])
                
                if noBearing: #If we dont have any bearing to landmark
                    poseMeasurement = np.array(data[3:5],dtype=np.float64)
                    upperTriangle = np.array(data[5:8],dtype=np.float64) #data[5:8] if benchmark else 9:12
                    information = from_uppertri_to_full(upperTriangle,2)
                    
                else: #If we need landmark bearing
                    poseMeasurement = np.array(data[3:6],dtype=np.float64)
                    upperTriangle = np.array(data[6:12],dtype=np.float64)
                    information = from_uppertri_to_full(upperTriangle,3)

                edge = Edge(Type, nodeFrom, nodeTo, poseMeasurement, information)
                edges.append(edge)
                

            elif data[0] == 'EDGE_SE2_GPS':
                continue
                Type = 'G' #landmark type
                nodeFrom = int(data[1])
                nodeTo = int(data[2])                
                poseMeasurement = np.array(data[3:5],dtype=np.float64)
                upperTriangle = np.array(data[5:8],dtype=np.float64)
                information = from_uppertri_to_full(upperTriangle,2)

                edge = Edge(Type, nodeFrom,nodeTo, poseMeasurement, information)
                edges.append(edge)

            elif data[0] == 'EDGE_SE2_BEARING' or 'EDGE_BEARING_SE2_XY':
                #print(f"data {data}")
                Type = 'B' #landmark type
                nodeFrom = int(data[1])
                nodeTo = int(data[2])                
                poseMeasurement = float(data[3])#,dtype=np.float64)
                information = float(data[4])#,dtype=np.float64)
                
                edge = Edge(Type, nodeFrom, nodeTo, poseMeasurement, information)
                edges.append(edge)

                if initial_b_guess:
                    initial_bearing_guess(nodes, nodeFrom, nodeTo, nodeTypes, poseMeasurement, lm_status)
            else: 
                print("Error, edge or vertex not defined")
        
    lut = {}
    x = []
    offset = 0
    for nodeId in nodes:
        lut.update({nodeId: offset})
        offset = offset + len(nodes[nodeId])
        x.append(nodes[nodeId])
    x = np.concatenate(x, axis=0)

    if initial_qualified_guess:
        qualified_guess(edges, lut, x, nodes, nodeTypes, least_squares=True, triangulation=False, epsilon=50)

    lut = {}
    x = []
    offset = 0
    for nodeId in nodes:
        lut.update({nodeId: offset})
        offset = offset + len(nodes[nodeId])
        x.append(nodes[nodeId])
    x = np.concatenate(x, axis=0)

    # collect nodes, edges and lookup in graph structure
    from run_slam import Graph
    
    graph = Graph(x, nodes, edges, lut, nodeTypes)
    
    print('Loaded graph with {} nodes and {} edges'.format(
        len(graph.nodes), len(graph.edges)))
    print('\n')

    return graph


def qualified_guess(edges, lut, x, nodes, nodeTypes, least_squares: bool, triangulation: bool, epsilon: float):
    
    if least_squares:
        triangulation = False
    if triangulation:
        least_squares = False

    ls = LS()
    tri = TRI()
    mem = defaultdict(list)
    count = 0

    for e in edges:
        if e.Type == 'B':
            
            _nodePose = e.nodeFrom
            _nodeLm = e.nodeTo
        
            fromIdx = lut[_nodePose]
            x_b = x[fromIdx:fromIdx+3] # Robot pose
            z_ij = e.poseMeasurement # Bearing measurement

            if count > 0 and check_parallel_lines(_x_b, _z_ij, x_b, z_ij, epsilon=epsilon):
                continue
            else:
                _meas = [x_b[0], x_b[1], x_b[2], z_ij]
                mem[_nodeLm].append(_meas)

            _x_b = x_b # Updating old value
            _z_ij = z_ij # Updating old value
            count += 1


    for ID, meas in mem.items():

        m = np.vstack(meas)
        Xr = m[:,0:3]
        z_list = list(m[:,3])

        if len(z_list) > 0:
            if least_squares:
                Xl = ls.least_squares_klines(Xr, z_list)  # Computing least squares best guess
            elif triangulation:
                Xl = tri.triangulation(Xr, z_list) # Computing triangulation best guess

            landmark = np.array([Xl[0,0], Xl[1,0]], dtype=np.float64)

            nodeType = 'VXY'
            nodeId = ID
            nodes[nodeId] = landmark
            nodeTypes[nodeId] = nodeType

        del m


def initial_bearing_guess(nodes, nodeFrom, nodeTo, nodeTypes, poseMeasurement, lm_status):

    x_b = nodes[nodeFrom]
    z_ij = poseMeasurement

    lm_status.update(dict([(nodeTo, False)]))
  
    for id, status in lm_status.items():
        if id == nodeTo and status == False:
            

            lambdadistx = 5
            lambdadisty = 5
            xguess = x_b[0]+lambdadistx*np.cos(wrap2pi(x_b[2]+z_ij))
            yguess = x_b[1]+lambdadisty*np.sin(wrap2pi(x_b[2]+z_ij))

            nodeType = 'VXY'
            nodeId = nodeTo
            landmark = np.array([xguess,yguess],dtype=np.float64)  
            nodes[nodeId] = landmark
            nodeTypes[nodeId] = nodeType
            lm_status[nodeTo] = True
        
def check_parallel_lines(p1, z1, p2, z2, epsilon: float=1) -> bool:

    n1 = wrap2pi(z1+p1[2])
    n2 = wrap2pi(z2+p2[2])

    diff = np.rad2deg(abs(wrap2pi(n1 - n2)))

    parallel = diff < epsilon

    return parallel