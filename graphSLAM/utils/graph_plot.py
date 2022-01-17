import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import diff
import seaborn as sns
import pandas as pd
from utils.helper import get_poses_landmarks, vec2trans, trans2vec, load_from_json
from utils.error import *
import sys
sys.path.append('g2o_generator/GIS_Extraction')

import csv_reader as GIS

sns.set()

def graph_plot(graph, animate = False, poseEdgesPlot = True, landmarkEdgesPlot = False, gpsEdgesPlot = False):
    
    fig , ax = plt.subplots()
    plt.clf()
    colorlist = colors()
    
    from run_slam import get_poses_landmarks
    poses, landmarks, lm_ID, gps = get_poses_landmarks(graph)
    
    #plot poses and landmarks if exits
    if len(poses) > 0:
        poses = np.stack(poses, axis=0) # axis = 0 turns into integers/slices and not tuple
        plt.plot(poses[:,0], poses[:,1], color='b', alpha = 0.9, marker='o', markersize=5,label = 'Robot pose')
        #plt.quiver(poses[:,0], poses[:,1], np.cos(poses[:,2]),np.sin(poses[:,2]), angles= 'xy',scale=0.5)
    
    if len(landmarks) > 0:
        landmarks = np.stack(landmarks, axis=0)
        # plt.plot(landmarks[:,0], landmarks[:,1], 'r*', markersize=12, label = 'Landmark')
        plt.scatter(landmarks[:,0], landmarks[:,1], label = 'Landmark', marker='*', color='r', s=150, zorder=10, alpha = 0.8)
        for lx, ly, ID in zip(landmarks[:,0], landmarks[:,1], lm_ID):
            plt.annotate(str(ID), xy=(lx, ly), color='r', zorder=11)
    
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
            plt.plot(poseEdgeX_corr,poseEdgeY_corr,'k--', alpha = 0.8)
            plt.plot([],[], 'k--', label='poseEdges')

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
        plt.axis('equal')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.legend(frameon=False,loc='lower center', ncol=5)
        plt.draw()
        plt.pause(1)
    else:
        plt.axis('equal')
        plt.xlabel(f'x [m]')
        plt.ylabel('y [m]')
        
        plt.legend(frameon=False,loc='lower center', ncol=5)
        
    return

def plot_ground_together_noise(n_graph, g_graph, pre_graph, lm_plot: bool=False):
    
    # from utils.slam_iterate import FOV, PHI, LAMBDAH
    fig , (ax1,ax2) = plt.subplots(1, 2)
    gposes, glandmarks, lm_ID, _ = get_poses_landmarks(g_graph)
    nposes, nlandmarks, lm_ID, _ = get_poses_landmarks(n_graph)
    pposes, prelandmarks, lm_ID, _ = get_poses_landmarks(pre_graph)
    
    if len(gposes) > 0:
        gposes = np.stack(gposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        ax1.plot(gposes[:,0], gposes[:,1], 'b--')#,label='Ground truth')
        ax2.plot(gposes[:,0], gposes[:,1], 'b--')#,label='Ground truth')
        

    if len(nposes) > 0:
        nposes = np.stack(nposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        ax2.plot(nposes[:,0], nposes[:,1], 'g')#,label='Noisy route')

    if len(pposes) > 0:
        pposes = np.stack(pposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        ax1.plot(pposes[:,0], pposes[:,1], 'r')#,label='Before Optimization route')

    if lm_plot == True:
        
        if len(nlandmarks) > 0:
            nlandmarks = np.stack(nlandmarks, axis=0)
            # plt.plot(landmarks[:,0], landmarks[:,1], 'r*', markersize=12, label = 'Landmark')
            ax2.scatter(nlandmarks[:,0], nlandmarks[:,1], marker='*', color='lime', s=150, zorder=10)
            for lx, ly, ID in zip(nlandmarks[:,0], nlandmarks[:,1], lm_ID):
                ax2.annotate(str(ID), xy=(lx, ly), color='g',alpha = 0.8, zorder=11)
            
        if len(prelandmarks) > 0:
            prelandmarks = np.stack(prelandmarks, axis=0)
            # plt.plot(landmarks[:,0], landmarks[:,1], 'r*', markersize=12, label = 'Landmark')
            ax1.scatter(prelandmarks[:,0], prelandmarks[:,1], marker='*', color='red', s=150, zorder=14)
            for lx, ly, ID in zip(prelandmarks[:,0], prelandmarks[:,1], lm_ID):
                ax1.annotate(str(ID), xy=(lx, ly), color='r', zorder=15)

        if len(glandmarks) > 0:
            glandmarks = np.stack(glandmarks, axis=0)
            # plt.plot(landmarks[:,0], landmarks[:,1], 'r*', markersize=12, label = 'Landmark')
            ax1.scatter(glandmarks[:,0], glandmarks[:,1], marker='*', color='b', s=150, zorder=12)
            ax2.scatter(glandmarks[:,0], glandmarks[:,1], marker='*', color='b', s=150, zorder=12)

            for lx, ly, ID in zip(glandmarks[:,0],glandmarks[:,1], lm_ID):
                ax1.annotate(str(ID), xy=(lx, ly), color='b',alpha = 0.8,zorder=13)
                ax2.annotate(str(ID), xy=(lx, ly), color='b',alpha = 0.8,zorder=13)

    plt.suptitle('Pre and Post Optimization')
    ax1.axis('equal')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax2.axis('equal')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax1.legend(['Ground truth', 'Wheel Odometry'], frameon = False)
    ax2.legend(['Ground truth','Optimized'], frameon = False)
    
    plt.show()
    return
    

def poses_per_landmark(graph, deg: bool=False):

    poses = []
    landmarks = []
    bearing = []
    
    for edge in graph.edges:
        if edge.Type == 'B':

            poses.append(edge.nodeFrom)
            landmarks.append(edge.nodeTo)
            bearing.append(edge.poseMeasurement)
           
    total = np.stack(zip(landmarks,bearing),axis=0)
    
    l_values, _ = np.unique(landmarks, return_counts=True)
    p_values, _ = np.unique(poses, return_counts=True)

    sns.histplot(poses, bins= len(p_values), kde=True)
    plt.xlabel('Pose ID')
    plt.title('Landmarks acquired per pose')
    plt.show()

    
    if deg:
        ax3 = sns.histplot(np.rad2deg(total[:,1]), bins=len(l_values))
        ax3.set_xlabel('Bearing [deg]')

    else:
        ax3 = sns.histplot((total[:,1]), bins=len(l_values))
        ax3.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4)) # denom needs to be written here and in multipleformatter input param for denom
        ax3.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 16))
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        ax3.set_xlabel('Bearing [rad]')

    ax3.legend(['Bearing count'])
    ax3.set_title('Bearings to landmarks')

    plt.show()
    ax4 = sns.histplot(total[:,0], bins=len(l_values))
    ax4.set_title('Amount of times each unique landmark has been observed')
    ax4.set_xlabel('Landmark ID')
    ax4.legend(['Landmark count'])

    plt.show()

    return

def landmark_ba(n_graph, g_graph, pre_graph):

    _, nlandmarks, nlm_ID, _ = get_poses_landmarks(n_graph)
    _, glandmarks, glm_ID, _ = get_poses_landmarks(g_graph)
    _, prelandmarks, prelm_ID, _ = get_poses_landmarks(pre_graph)
    
    
    fig, axes = plt.subplots(1,2)

    if len(nlandmarks) > 0:
        nlandmarks = np.stack(nlandmarks, axis=0)
        axes[1].scatter(nlandmarks[:,0], nlandmarks[:,1], marker='*', color='g',  s=150, zorder=11)
        for lx, ly, ID in zip(nlandmarks[:,0], nlandmarks[:,1], nlm_ID):
            axes[1].annotate(str(ID), xy=(lx, ly), color='g',alpha = 0.8, zorder=11)
    if len(glandmarks) > 0:
        glandmarks = np.stack(glandmarks, axis=0)
        axes[0].scatter(glandmarks[:,0], glandmarks[:,1], marker='*', color='b', alpha=0.8, s=150, zorder=10)
        axes[1].scatter(glandmarks[:,0], glandmarks[:,1], marker='*', color='b', alpha=0.8, s=150, zorder=10)
        for lx, ly, ID in zip(glandmarks[:,0], glandmarks[:,1], glm_ID):
            axes[0].annotate(str(ID), xy=(lx, ly), color='b', alpha = 0.8, zorder=10)
            axes[1].annotate(str(ID), xy=(lx, ly), color='b', alpha = 0.8, zorder=10)

    if len(prelandmarks) > 0:
        prelandmarks = np.stack(prelandmarks, axis=0)
        axes[0].scatter(prelandmarks[:,0], prelandmarks[:,1], marker='*', color='red', s=150, zorder=11)
        for lx, ly, ID in zip(prelandmarks[:,0], prelandmarks[:,1], prelm_ID):
            axes[0].annotate(str(ID), xy=(lx, ly), color='r', zorder=11)

    
    plt.suptitle('Landmark position before and after Optimization')
    axes[0].set_xlabel('x [m]')
    axes[1].set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    axes[1].set_ylabel('y [m]')
    axes[0].axis('equal')
    axes[1].axis('equal')
    axes[0].legend(['Pre landmarks','Ground truth landmarks'])
    axes[1].legend(['Noisy landmarks','Ground truth landmarks'])
    
    diff_gn = [elem for elem in glm_ID if elem not in nlm_ID]
    print(len(diff_gn))
    print(len(glm_ID))
    print(len(nlm_ID))
    corr_glandmarks = np.delete(glandmarks,diff_gn, axis=0)
    print(len(corr_glandmarks))

    # corr_prelandmarks = np.delete(prelandmarks,diff_gn, axis=0)
    
    diff_ng_abs = np.around(abs(nlandmarks-corr_glandmarks),2)
    diff_ng = np.around((nlandmarks-corr_glandmarks),2)

    diff_pg_abs = np.around(abs(prelandmarks-corr_glandmarks),2)
    diff_pg = np.around((prelandmarks-corr_glandmarks),2)

    x = np.expand_dims(diff_ng[:,0],axis = 1)
    y = np.expand_dims(diff_ng[:,1],axis = 1)
    ng_sum = round(np.sum(diff_ng),2)
    pg_sum = round(np.sum(diff_pg),2)

    fig1, (ax1,ax2) = plt.subplots(1,2)    
    
    ax1.scatter(diff_ng[:,0], diff_ng[:,1], color = 'b')
    for lx, ly, ID in zip(x, y, nlm_ID):
           ax1.annotate(str(ID), xy=(lx, ly), color='b', alpha = 0.8)
    
    ax2.scatter(diff_pg[:,0], diff_pg[:,1], color = 'r')
    ax1.set_title('error between noise and ground')
    ax2.set_title('error between pre and ground')
    ax1.set_xlabel('x [m]')
    ax2.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax2.set_ylabel('y [m]')
    leg = ax1.legend(['sum '+str(ng_sum)], handlelength=0, handletextpad=0, fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)
    leg = ax2.legend(['sum '+str(pg_sum)], handlelength=0, handletextpad=0, fancybox=True)
    for item in leg.legendHandles:
        item.set_visible(False)

    plt.show()
    
    row_sum_ng = [sum(x) for x in diff_ng]
    row_ng_lm = np.stack(zip(nlm_ID,row_sum_ng))
    
    df = pd.DataFrame(data = row_ng_lm, columns = ['Landmark ID', 'Sum of x,y error'])
    # # sns.heatmap(x)len()
    # # sns.histplot(diff_ng)#, bins = len(nlandmarks))
    sns.barplot(data = df, x= 'Landmark ID', y = 'Sum of x,y error')
    plt.show()
    return

def A_traj_error(n_graph, g_graph):
    
    g_poses, _ , _ , _ = get_poses_landmarks(g_graph)
    n_poses, _ , _ , _ = get_poses_landmarks(n_graph)
    
    E = []

    for g, n in zip(g_poses,n_poses):
       
        G = vec2trans(g)
        N = vec2trans(n)
        
        error = np.linalg.inv(G) @ N
        error = trans2vec(error)

        ATE = np.sqrt((error ** 2))
        E.append(ATE)
    
    E = np.stack(E,axis=0)
    ATE_rmse = E.mean()
    print(ATE_rmse)
 
    plt.plot(E[:,0])
    plt.show()
    plt.plot(E[:,1])
    plt.show()
    plt.plot(E[:,2])
    plt.show()
    sns.histplot(E[:,0])
    plt.show()
    sns.kdeplot(E[:,0])
    plt.show()

    return


def statistics_plot(graph):

    pose, land, _,_ = get_poses_landmarks(graph)
    pose = np.stack(pose,axis=0)

    # ax1.hist(landmarks, bins = len(values),  alpha=0.5, histtype='stepfilled',color ='steelblue', density=True)
    df = pd.DataFrame(pose, columns = ['x', 'y', 'theta'])
    
    fig, axes = plt.subplots(1,3)

    sns.histplot(ax=axes[0], data = pose[:,0], kde=True)
    axes[0].set_xlabel('X')
    
    sns.histplot(ax=axes[1], data = pose[:,1], kde=True)
    axes[1].set_xlabel('Y')
    
    sns.histplot(ax=axes[2], data = pose[:,2], kde=True)
    axes[2].set_xlabel('Heading')
    plt.suptitle('Count of pose coordinates in map')
    plt.show()

    ax5 = sns.kdeplot(data = df, x= "x", y = "y", fill=True, levels=5,
                      cmap='viridis', shade=True, shade_lowest=False, cbar=True)
    
    plt.title('2D Distribution of poses in map')
    plt.show()

    ax6= sns.kdeplot(data = pose[:,:2], fill = True)#, kde=True)
    plt.legend(loc = 'upper right', labels=["x","y"])
    plt.title('1D Distribution of x, y poses in map')
    plt.show()
    # plt.plot(x, mean_1, 'b-', label='mean_1')
    # plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)

    return
    
    plt.xticks(np.arange(0, len(values), 5))
    plt.show()
    return

def plot_errors(e_full,pose_error,bearing_error,land_error,gps_error):

    e_pose = np.vstack((pose_error))

    if len(pose_error) >0:
        f1, (ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.plot(e_pose[1:,0], color='g', marker="o", label ='x err')
        ax1.legend(loc="upper right")
        ax2.plot(e_pose[1:,1], color='b', marker="o", label ='y err')
        ax2.legend(loc="upper right")
        ax3.plot(e_pose[1:,2], color='r', marker="o", label ='$\\theta$')
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

def color_error_plot(n_graph,g_graph):

    gposes, _, _, _ = get_poses_landmarks(g_graph)
    nposes, _, _, _ = get_poses_landmarks(n_graph)
    nposes = np.stack(nposes, axis=0)
    gposes = np.stack(gposes, axis=0)

    _, ax = plt.subplots()

    e = np.sum(np.abs((gposes[:,:2]-nposes[:,:2])),axis=1)
    print(e)
    x = nposes[:,0]
    y = nposes[:,1]
    xg = gposes[:,0]
    yg = gposes[:,1]

    bob = ax.scatter(x,y,c=e, cmap="viridis", alpha=0.8)
    cbar = plt.colorbar(bob)
    cbar.set_label('Absolute x, y error', rotation = 270)
    cbar.ax.get_yaxis().labelpad = 15

    ax.plot(x,y, 'b-')
    ax.plot(xg,yg, 'k--')
    ax.legend(['Noise','Ground'])
    plt.title('Error plotted onto noisy route')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.show()

    
    return
    
    
def error_plot(g_graph, n_graph):

    gposes, _, _, _ = get_poses_landmarks(g_graph)
    nposes, _, _, _ = get_poses_landmarks(n_graph)
    nposes = np.stack(nposes, axis=0)
    gposes = np.stack(gposes, axis=0)
    diff =[]

    _, (ax1, ax2, ax3) = plt.subplots(3,1)
    diff = gposes-nposes
    for i in range(len(diff[:,2])):

        diff[i,2]= wrap2pi((diff[i,2]))

    ax1.plot(gposes[:,0], color = 'b')
    ax2.plot(gposes[:,1], color = 'b')
    ax3.plot(gposes[:,2], color = 'b')

    ax1.plot(nposes[:,0], color = 'y')
    ax2.plot(nposes[:,1], color = 'y')
    ax3.plot(nposes[:,2], color = 'y')

    ax1.plot(np.abs(diff[:,0]), color = 'r')
    ax2.plot(np.abs(diff[:,1]), color = 'r')
    ax3.plot(np.abs(diff[:,2]), color = 'r')
    ax1.legend(['Ground truth','Noisy','Abs error'], frameon=False)
    ax1.set_ylabel('x [m]')
    # ax2.legend(['Ground truth','Noisy','Error(diff)'])
    ax2.set_ylabel('y [m]')
    # ax3.legend(['Ground truth','Noisy','Error(diff)'])
    ax3.set_ylabel('$\\theta$ [rad]')
    
    plt.suptitle('Pose error between Noise and Ground truth')
    plt.show()



def plot_map(n_graph, g_graph):
    odometry_file = 'g2o_generator/robosim/data/robopath/tester.json'
    odometry = load_from_json(odometry_file)

    temp_x = np.asfarray(odometry[0]); temp_y = np.asfarray(odometry[1]); temp_th = np.asfarray(odometry[2])

    nposes, _, _, _ = get_poses_landmarks(n_graph)
    gposes, _, _, _ = get_poses_landmarks(g_graph)


    _, axs = plt.subplots()
    axs.set_aspect('equal', 'datalim')

    axs.set_xlim([574730, 575168])
    axs.set_ylim([6222350, 6222750])

    if len(nposes) > 0:
        nposes = np.stack(nposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        axs.plot(nposes[:,0]+temp_x[0], nposes[:,1]+temp_y[0], 'b-')

    if len(gposes) > 0:
        gposes = np.stack(gposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        axs.plot(gposes[:,0]+temp_x[0], gposes[:,1]+temp_y[0], 'g-')
    
    filenamePoints = 'g2o_generator/GIS_Extraction/data/aarhus_features_v2.csv'
    filenamePoly = 'g2o_generator/GIS_Extraction/data/aarhus_polygons_v2.csv'
    # landmarks = './g2o_generator/GIS_Extraction/landmarks/landmarks_w_types.json'

    aarhus = GIS.read_csv(filenamePoints, filenamePoly)
    _, rowPoly = aarhus.read()
    cascaded_poly = aarhus.squeeze_polygons(rowPoly)

    
    for geom in cascaded_poly.geoms:
        x_casc, y_casc = geom.exterior.xy
        axs.fill(x_casc, y_casc, alpha=0.5, fc='b', ec='none')

    # aarhus.plot_landmarks(landmarks)
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

def multiple_formatter(denominator=4, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter
