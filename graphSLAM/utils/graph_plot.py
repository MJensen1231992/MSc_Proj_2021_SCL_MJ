import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.helper import get_poses_landmarks, vec2trans, trans2vec
from sklearn.metrics import mean_squared_error
from utils.error import *
import matplotlib.cm as cm
sns.set()

# plt.style.use('seaborn-white')

def graph_plot(graph, figid:int, Label:str, animate = False, poseEdgesPlot = True, landmarkEdgesPlot = False, gpsEdgesPlot = False):
    
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
        # plt.plot(landmarks[:,0], landmarks[:,1], 'r*', markersize=12, label = 'Landmark')
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
            plt.plot(bearingEdgeX_corr,bearingEdgeY_corr,'k--')
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

    return

def plot_ground_together_noise(n_graph, g_graph, pre_graph, figid: int):

    _ , ax1 = plt.subplots()
    gposes, _, _, _ = get_poses_landmarks(g_graph)
    nposes, _, _, _ = get_poses_landmarks(n_graph)
    pposes, _, _, _ = get_poses_landmarks(pre_graph)
    
    if len(gposes) > 0:
        gposes = np.stack(gposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        plt.plot(gposes[:,0], gposes[:,1], 'b--',label='Ground truth')
        

    if len(nposes) > 0:
        nposes = np.stack(nposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        plt.plot(nposes[:,0], nposes[:,1], 'g',label='Noisy route')
    
    if len(pposes) > 0:
        pposes = np.stack(pposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        plt.plot(pposes[:,0], pposes[:,1], 'r',label='Before opt route')
        
    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend(['Ground truth','Noisy route', 'Pre'])
    
    

def poses_per_landmark(graph):

    poses = []
    landmarks = []

    for edge in graph.edges:
        if edge.Type == 'B':

            poses.append(edge.nodeFrom)
            landmarks.append(edge.nodeTo)
    
    l_values, _ = np.unique(landmarks, return_counts=True)
    p_values, _ = np.unique(poses, return_counts=True)
    ax1 = sns.histplot(landmarks, bins= len(l_values), kde=True)
    plt.title('Poses per landmark')
    plt.show()

    # ax2 = sns.histplot(poses, bins= len(p_values), kde=True)
    # plt.title('land per pose')
    # plt.show()

    return

def landmark_ba(n_graph, pre_graph, g_graph):

    _, nlandmarks, lm_ID, _ = get_poses_landmarks(n_graph)
    _, prelandmarks, lm_ID, _ = get_poses_landmarks(pre_graph)
    _, glandmarks, lm_ID, _ = get_poses_landmarks(g_graph)
    
    if len(nlandmarks) > 0:
        nlandmarks = np.stack(nlandmarks, axis=0)
        # plt.plot(landmarks[:,0], landmarks[:,1], 'r*', markersize=12, label = 'Landmark')
        plt.scatter(nlandmarks[:,0], nlandmarks[:,1], marker='*', color='lime', s=150, zorder=10)
        for lx, ly, ID in zip(nlandmarks[:,0], nlandmarks[:,1], lm_ID):
            plt.annotate(str(ID), xy=(lx, ly), color='lime', zorder=11)
        
        if len(prelandmarks) > 0:
            prelandmarks = np.stack(prelandmarks, axis=0)
            # plt.plot(landmarks[:,0], landmarks[:,1], 'r*', markersize=12, label = 'Landmark')
            plt.scatter(prelandmarks[:,0], prelandmarks[:,1], marker='*', color='red', s=150, zorder=10)
            for lx, ly, ID in zip(prelandmarks[:,0], prelandmarks[:,1], lm_ID):
                plt.annotate(str(ID), xy=(lx, ly), color='r', zorder=11)

        if len(glandmarks) > 0:
            glandmarks = np.stack(glandmarks, axis=0)
            # plt.plot(landmarks[:,0], landmarks[:,1], 'r*', markersize=12, label = 'Landmark')
            plt.scatter(glandmarks[:,0], glandmarks[:,1], marker='*', color='b', s=150, zorder=10)
            for lx, ly, ID in zip(glandmarks[:,0],glandmarks[:,1], lm_ID):
                plt.annotate(str(ID), xy=(lx, ly), color='b', zorder=11)

        plt.legend(['n','p','g'])
        
    return

def ATE(n_graph, g_graph):
    
    g_poses, _ , _ , _ = get_poses_landmarks(g_graph)
    n_poses, _ , _ , _ = get_poses_landmarks(n_graph)
    # n_poses = np.stack(n_poses,axis=0)
    # g_poses = np.stack(g_poses,axis=0)
    
    E = []

    for g, n in zip(g_poses,n_poses):
       
        G = vec2trans(g)
        N = vec2trans(n)
        
        error = np.linalg.inv(G) @ N
        error = trans2vec(error)
        # E.append(error)
        # E = np.stack(E,axis=0)
        ATE = np.sqrt((error ** 2))
        E.append(ATE)

    # print(f"error\n{E}\n")
    
    
    
    
    E = np.stack(E,axis=0)
    ATE_rmse = E.mean()
    print(ATE_rmse)
    plt.plot(E[:,0])
    plt.show()

    return



def statistics_plot(graph):
    pose, land, _,_ = get_poses_landmarks(graph)
    pose = np.stack(pose,axis=0)
   

    # ax1.hist(landmarks, bins = len(values),  alpha=0.5, histtype='stepfilled',color ='steelblue', density=True)
    df = pd.DataFrame(pose, columns = ['x', 'y', 'theta'])
    
    print(df)

    ax2 = sns.histplot(data = pose[:,0], kde=True)
    plt.title('ax2')
    plt.show()
    ax3 = sns.histplot(data = pose[:,1], kde=True)
    plt.title('ax3')
    plt.show()
    ax3 = sns.histplot(data = pose[:,2], kde=True)
    plt.title('ax4')
    plt.show()
    ax5 = sns.kdeplot(data = df, x= "x", y = "y",fill=True)
    
    plt.title('ax5')
    plt.show()
    ax6= sns.kdeplot(data = pose[:,:2], fill = True)#, kde=True)
    plt.legend(loc = 'lower center', labels=["x","y"])
    plt.title('ax6')
    plt.show()
    # plt.plot(x, mean_1, 'b-', label='mean_1')
    # plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)

    return
    

def plot_errors(e_full,pose_error,bearing_error,land_error,gps_error):

    e_pose = np.vstack((pose_error))

    if len(pose_error) >0:
        f1, (ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.plot(e_pose[:,0], color='g', marker="o", label ='x err')
        ax1.legend(loc="upper right")
        ax2.plot(e_pose[:,1], color='b', marker="o", label ='y err')
        ax2.legend(loc="upper right")
        ax3.plot(e_pose[:,2], color='r', marker="o", label ='$\\theta$')
        ax3.legend(loc="upper right")
    
    if len(bearing_error)>0:
        ax4 = sns.relplot(data = bearing_error, kind="line", ci = 100)
        #ax4.plot(bearing_error, label = 'bearing error')
        #ax4.legend()
    
    if len(land_error)>1:
        e_land = np.abs(np.vstack((land_error))) 
        _, (ax5,ax6) = plt.subplots(1,2)
        ax5.plot(e_land[:,0], label ='x land error')
        ax5.legend()
        ax6.plot(e_land[:,1], label ='y land error')
        ax6.legend()

    if len(gps_error)>1:
        e_gps = np.abs(np.vstack((gps_error)))
        _, (ax7,ax8) = plt.subplots(1,2)
        ax7.plot(e_gps[:,0], label ='x gps error')
        ax7.legend()
        ax8.plot(e_gps[:,1], label ='y gps error')
        ax8.legend()

    _, ax9 = plt.subplots()
    ax9.plot(e_full, label = 'full error')
    ax9.legend()
 
    plt.show()

    return

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

def test_plot(n_graph, g_graph):

    gposes, _, _, _ = get_poses_landmarks(g_graph)
    nposes, _, _, _ = get_poses_landmarks(n_graph)
    nposes = np.stack(nposes, axis=0)
    gposes = np.stack(gposes, axis=0)

    save_as_txt(nposes,gposes)
    
   
    return

def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

def save_as_txt(array1, array2):

    array1_txt = open("noise.txt", "w")
    for row in array1:
        np.savetxt(array1_txt, row)

    array1_txt.close()
    array2_txt = open("ground.txt", "w")
    for row in array2:
        np.savetxt(array2_txt, row)
    array2_txt.close()
    
    
