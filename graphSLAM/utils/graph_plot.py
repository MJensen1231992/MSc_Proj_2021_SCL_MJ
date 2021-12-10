
import numpy as np
import matplotlib.pyplot as plt
from helper import *

def graph_plot(graph, animate = False, poseEdgesPlot = True, landmarkEdgesPlot = False, gpsEdgesPlot = False):

    #init plt figure
    plt.figure(1)
    plt.clf()

    from run_slam import get_poses_landmarks
    poses, landmarks, gps = get_poses_landmarks(graph)
    
    #plot poses and landmarks if exits
    if len(poses) > 0:
        poses = np.stack(poses, axis=0) # axis = 0 turns into integers/slices and not tuple
        plt.plot(poses[:,0], poses[:,1], 'bo', markersize=4)
        #plt.quiver(poses[:,0], poses[:,1], np.cos(poses[:,2]),np.sin(poses[:,2]), angles= 'xy',scale=0.5)
    
    if len(landmarks) > 0:
        landmarks = np.stack(landmarks, axis=0)
        plt.plot(landmarks[:,0], landmarks[:,1], 'r*', markersize=12)
    
    if len(gps) > 0:
        gps = np.stack(gps, axis=0)
        plt.plot(gps[:,0], gps[:,1], "gd")

    poseEdgesFrom = []
    poseEdgesTo = []

    landmarkEdgesFrom = []
    landmarkEdgesTo = []

    bearingEdgesFrom = []
    bearingEdgesTo = []
    bearingZ = []

    gpsEdgesFrom = []
    gpsEdgesTo = []

    for edge in graph.edges:
        fromIdx = graph.lut[edge.nodeFrom]
        toIdx = graph.lut[edge.nodeTo]
        
        
        if edge.Type == 'P':
            poseEdgesFrom.append(graph.x[fromIdx:fromIdx+3])
            poseEdgesTo.append(graph.x[toIdx:toIdx+3])

        elif edge.Type == 'L': #Tager pose-landmark vertices derfor (3,2)
            landmarkEdgesFrom.append(graph.x[fromIdx:fromIdx+3])
            landmarkEdgesTo.append(graph.x[toIdx:toIdx+2])

        
        elif edge.Type == 'B': #Tager pose-landmark vertices derfor (3,2)
            bearingEdgesFrom.append(graph.x[fromIdx:fromIdx+3])
            bearingEdgesTo.append(graph.x[toIdx:toIdx+2])
            bearingZ.append(edge.poseMeasurement)
            
        
        elif edge.Type == 'G':
            gpsEdgesFrom.append(graph.x[fromIdx:fromIdx+3])
            gpsEdgesTo.append(graph.x[toIdx:toIdx+2]) 
            

    if len(poses) > 0:
        poseEdgesFrom = np.stack(poseEdgesFrom, axis = 0)
        poseEdgesTo = np.stack(poseEdgesTo, axis = 0)

        poseZip = zip(poseEdgesFrom, poseEdgesTo)
        poseEdges = np.vstack(poseZip)

        poseEdgesX = poseEdges[:,0]
        poseEdgesY = poseEdges[:,1]

        #Split into (x,y) pairs
        poseEdgeX_corr = np.vstack([poseEdgesX[0::2], poseEdgesX[1::2]])
        poseEdgeY_corr = np.vstack([poseEdgesY[0::2], poseEdgesY[1::2]])

        if poseEdgesPlot == True:
            plt.plot(poseEdgeX_corr,poseEdgeY_corr,'r--')#,label = 'poseEdges')
    
    if len(landmarks) > 0 and edge.Type == 'L':

        landmarkEdgesFrom = np.stack(landmarkEdgesFrom, axis = 0)
        landmarkEdgesTo = np.stack(landmarkEdgesTo, axis = 0)
    
        # Zip landmark_from(x,y) with corresponding landmark_to(x,y)
        landmarkZip = zip(landmarkEdgesFrom[:,0:2], landmarkEdgesTo)
        landmarkEdges = np.vstack(landmarkZip)
        
        landmarkEdgesX = landmarkEdges[:,0]
        landmarkEdgesY = landmarkEdges[:,1]

        # # Use every 2nd x and y coordinate so correct correlation
        landmarkEdgeX_corr = np.vstack([landmarkEdgesX[0::2], landmarkEdgesX[1::2]])
        landmarkEdgeY_corr = np.vstack([landmarkEdgesY[0::2], landmarkEdgesY[1::2]])

        if landmarkEdgesPlot == True:
            plt.plot(landmarkEdgeX_corr,landmarkEdgeY_corr,'k--', label = 'landEdges')

    if len(landmarks) > 0 and edge.Type == 'B':
        
        
        bearingEdgesFrom = np.stack(bearingEdgesFrom, axis = 0)
        bearingEdgesTo = np.stack(bearingEdgesTo, axis = 0)
        
        # Zip landmark_from(x,y) with corresponding landmark_to(x,y)
        bearingZip = zip(bearingEdgesFrom[:,0:2], bearingEdgesTo)
        bearingEdges = np.vstack(bearingZip)
        
        bearingEdgesX = bearingEdges[:,0]
        bearingEdgesY = bearingEdges[:,1]

        # # Use every 2nd x and y coordinate so correct correlation
        bearingEdgeX_corr = np.vstack([bearingEdgesX[0::2], bearingEdgesX[1::2]])
        bearingEdgeY_corr = np.vstack([bearingEdgesY[0::2], bearingEdgesY[1::2]])
        
        bearingZ = np.reshape(bearingZ,(1,len(bearingZ)))
        
        localBearing = bearingZ+bearingEdgesFrom[:,2]
        
        if landmarkEdgesPlot == True:
            plt.plot()#bearingEdgeX_corr,bearingEdgeY_corr)#,'k--')#, label = 'bearingEdges')
            plt.quiver(bearingEdgesFrom[:,0] , bearingEdgesFrom[:,1], np.cos(localBearing),np.sin(localBearing), angles='xy',scale=0.5,alpha=0.2)

    if len(gps) > 0:

        gpsEdgesFrom = np.stack(gpsEdgesFrom, axis = 0)
        gpsEdgesTo = np.stack(gpsEdgesTo, axis = 0)
    
        # Zip gps_from(x,y) with corresponding gps_to(x,y)
        gpsZip = zip(gpsEdgesFrom[:,0:2], gpsEdgesTo)
        gpsEdges = np.vstack(gpsZip)
        
        gpsEdgesX = gpsEdges[:,0]
        gpsEdgesY = gpsEdges[:,1]

        # # Use every 2nd x and y coordinate so correct correlation
        gpsEdgeX_corr = np.vstack([gpsEdgesX[0::2], gpsEdgesX[1::2]])
        gpsEdgeY_corr = np.vstack([gpsEdgesY[0::2], gpsEdgesY[1::2]])
    
        if gpsEdgesPlot == True:
            plt.plot(gpsEdgeX_corr,gpsEdgeY_corr,'k--', label = 'gpsEdges')

    if animate == True:
        plt.draw()
        plt.pause(1)
    else:
        plt.axis('equal')
        plt.show()

    return

def plot_ground_together_noise(ground_graph, noise_graph):
    #plt.figure(2)
    gposes, _, _ = get_poses_landmarks(ground_graph)
    nposes, _, _ = get_poses_landmarks(noise_graph)
    
    if len(gposes) > 0:
        gposes = np.stack(gposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        plt.plot(gposes[:,0], gposes[:,1], 'b',label='Ground truth')
        


    if len(nposes) > 0:
        nposes = np.stack(nposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        plt.plot(nposes[:,0], nposes[:,1], 'g',label='Noisy route')
        #plt.quiver(nposes[:,0], nposes[:,1], np.cos(nposes[:,2]),np.sin(nposes[:,2]), angles= 'xy')
        
        
    
        plt.axis('equal')
    plt.legend()
    plt.show()

def plot_errors(pose_error,land_error,gps_error):

    e_pose = np.abs(np.vstack((pose_error)))
    #e_land = np.abs(np.vstack((land_error)))
    #e_gps = np.abs(np.vstack((gps_error)))
   # print(f"e_pose : {e_pose} , e_land: {e_land}")

    fig1, axs1 = plt.subplots(1,3)
    axs1[0].plot(e_pose[:,0],label ='X')
    axs1[1].plot(e_pose[:,1],label = 'Y')
    axs1[2].plot(e_pose[:,2],label = 'Theta')
    
    # plt.plot(e_pose[:,0])
    # plt.plot(e_pose[:,1])
    # plt.plot(e_pose[:,2])
    

    # fig2, axs2 = plt.subplots(1,2)
    # axs2[0].plot(e_land[0::2])
    # axs2[1].plot(e_land[1::2])

    # fig3, axs3 = plt.subplots(1,2)
    # axs3[0].plot(e_gps[0::2])
    # axs3[1].plot(e_gps[1::2])
    
    plt.show()