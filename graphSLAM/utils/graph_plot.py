import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils.helper import get_poses_landmarks
sns.set()

# plt.style.use('seaborn-white')

def graph_plot(graph, figid:int, Label:str, animate = False, poseEdgesPlot = True, landmarkEdgesPlot = False, gpsEdgesPlot = False):
    
    # print(FOV)
    #init plt figure
    fig , ax = plt.subplots(figid)
    plt.clf()
    colorlist = colors()
    
    from run_slam import get_poses_landmarks
    poses, landmarks, lm_ID, gps = get_poses_landmarks(graph)
    
    #plot poses and landmarks if exits
    if len(poses) > 0:
        poses = np.stack(poses, axis=0) # axis = 0 turns into integers/slices and not tuple
        plt.plot(poses[:,0], poses[:,1], color='indigo', marker='o', markersize=5,label = 'Robot pose')
        #plt.quiver(poses[:,0], poses[:,1], np.cos(poses[:,2]),np.sin(poses[:,2]), angles= 'xy',scale=0.5)
    
    if len(landmarks) > 0:
        landmarks = np.stack(landmarks, axis=0)
        plt.scatter(landmarks[:,0], landmarks[:,1], marker='*', color='red', s=150, zorder=10)
        for lx, ly, ID in zip(landmarks[:,0], landmarks[:,1], lm_ID):
            plt.annotate(str(ID), xy=(lx, ly), color='lime', zorder=11)
    
    if len(gps) > 0:
        gps = np.stack(gps, axis=0)
        plt.plot(gps[:,0], gps[:,1], "gd", label = 'GPS point')

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
            plt.plot(poseEdgeX_corr,poseEdgeY_corr,'r--')#,label = labels)
            plt.plot([],[], 'r--', label='poseEdges')

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
            plt.plot(landmarkEdgeX_corr,landmarkEdgeY_corr,'k--')#, label = 'landEdges')
            plt.plot([],[],'k--',label = 'landEdges')

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
            plt.plot(bearingEdgeX_corr,bearingEdgeY_corr,'k--')#, label = 'bearingEdges')
            plt.plot([],[],'k--', label = 'bearingEdges')
            plt.quiver(bearingEdgesFrom[:,0] , bearingEdgesFrom[:,1], np.cos(localBearing),np.sin(localBearing), label='Bearing', angles='xy',scale=0.5,alpha=0.2)

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
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.legend(frameon=False,loc='lower center', ncol=5)
        #plt.gca().add_artist(first_leg)
        # plt.legend(LAMBDAH, loc = 'upper left')
    return

def plot_ground_together_noise(ground_graph, noise_graph, figid:int,):
    plt.figure(figid)
    gposes, _, _, _ = get_poses_landmarks(ground_graph)
    nposes, _, _, _ = get_poses_landmarks(noise_graph)
    
    if len(gposes) > 0:
        gposes = np.stack(gposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        plt.plot(gposes[:,0], gposes[:,1], 'b',label='Ground truth')
        


    if len(nposes) > 0:
        nposes = np.stack(nposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        plt.plot(nposes[:,0], nposes[:,1], 'g',label='Noisy route')
        
        plt.axis('equal')
    plt.legend()
    #plt.show()

def poses_per_landmark(graph):

    pose_land_dict = {}
    poses = []
    landmarks = []

    for edge in graph.edges:
        if edge.Type == 'B':

            poses.append(edge.nodeFrom)
            landmarks.append(edge.nodeTo)
            
            # pose_land_dict.update(dict([(str(pose),land)]))
    
    # a = Counter(landmarks)
    # print(a)
    values, counts = np.unique(landmarks, return_counts=True)
    # print(values, counts)
    # print(len(values))
    # data = np.vstack(zip(values,counts))
    plt.hist(landmarks, bins = len(values),  alpha=0.5, histtype='stepfilled',color ='steelblue')
    
    plt.xticks(np.arange(0, len(values), 5))
    plt.show()
    return


def plot_errors(e_full,pose_error,bearing_error,land_error,gps_error):

    e_pose = np.vstack((pose_error))

    if len(pose_error) >0:
        f1, (ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.plot(e_pose[:,0], color='g', marker="o", label ='x err')
        ax1.legend(loc="upper right")
        ax2.plot(e_pose[:,1], label ='y err')
        ax2.legend(loc="upper right")
        ax3.plot(e_pose[:,2], label ='$\\theta$')
        ax3.legend(loc="upper right")
    
    if len(bearing_error)>0:
        _, ax4 = plt.subplots()
        ax4.plot(bearing_error, label = 'bearing error')
        ax4.legend()
    
    if len(land_error)>1:
        e_land = np.abs(np.vstack((land_error))) 
        _, (ax5,ax6) = plt.subplots(1,2)
        ax5.plot(e_land[:,0], label ='x land error')
        ax5.legend()
        ax6.plot(e_land[:,1], label ='x land error')
        ax6.legend()

    if len(gps_error)>1:
        e_gps = np.abs(np.vstack((gps_error)))
        _, (ax7,ax8) = plt.subplots(1,2)
        ax7.plot(e_gps[:,0], label ='x gps error')
        ax7.legend()
        ax8.plot(e_gps[:,1], label ='x gps error')
        ax8.legend()

    _, ax9 = plt.subplots()
    ax9.plot(e_full, label = 'full error')
    ax9.legend()
    
    
    plt.show()

def colors():
    CB91_Blue = "#2CBDFE"
    CB91_Green = "#47DBCD"
    CB91_Pink = "#F3A0F2"
    CB91_Purple = "#9D2EC5"
    CB91_Violet = "#661D98"
    CB91_Amber = "#F5B14C"
    color_list = [
        CB91_Blue,
        CB91_Pink,
        CB91_Green,
        CB91_Amber,
        CB91_Purple,
        CB91_Violet,
    ]

    return color_list
