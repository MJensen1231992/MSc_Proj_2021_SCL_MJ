from math import inf
import numpy as np
from collections import namedtuple
from helper import from_uppertri_to_full, wrap2pi

def load_g2o_graph(filename, noBearing=True):#, firstMeas=True):
    
    Edge = namedtuple(
        'Edge', ['Type', 'nodeFrom', 'nodeTo', 'poseMeasurement', 'information'] # g2o format of files.
    )
    edges = []
    nodes = {}
    nodeTypes = {}
    lm_status = {}

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

                nodeType = 'VXY'
                nodeId = int(data[1])
                landmark = np.array(data[2:4],dtype=np.float64)  
                nodes[nodeId] = landmark
                nodeTypes[nodeId] = nodeType

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
                poseMeasurement = float(data[5])#,dtype=np.float64)
                information = float(data[11])#,dtype=np.float64)
                
                edge = Edge(Type, nodeFrom,nodeTo, poseMeasurement, information)
                edges.append(edge)

                x_b = nodes[nodeFrom]
                z_ij = poseMeasurement
                # print(f"Robot pose from(loader):\n{x_b}\nBearing(loader)\n{z_ij}\n")

            
                # print(f"nodeFrom{nodeFrom}")

                # if firstMeas:
                
                #     firstMeas = False

                lm_status.update(dict([(nodeTo, False)]))
                #print(lm_status)
                for id, status in lm_status.items():
                    if id == nodeTo and status == False:
                        #calcguess
                        lambdadistx = 5
                        lambdadisty = 5
                        xguess = x_b[0]+lambdadistx*np.cos(wrap2pi(x_b[2]+z_ij))
                        yguess = x_b[1]+lambdadisty*np.sin(wrap2pi(x_b[2]+z_ij))

                        nodeType = 'VXY'
                        nodeId = nodeTo
                        landmark = np.array([xguess,yguess],dtype=np.float64)  
                        nodes[nodeId] = landmark
                        nodeTypes[nodeId] = nodeType
                        lm_status[nodeTo]=True
                    
                #if lm_status:

                    else:
                        continue
                
                print(f"landmark dict: \n{lm_status}\n")
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
    
    # collect nodes, edges and lookup in graph structure
    from run_slam import Graph
    
    graph = Graph(x, nodes, edges, lut, nodeTypes)
    
    print('Loaded graph with {} nodes and {} edges'.format(
        len(graph.nodes), len(graph.edges)))
    
    return graph

