import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FuncFormatter
from mpl_toolkits import mplot3d
from utils.helper import get_poses_landmarks, vec2trans, trans2vec, load_from_json
from utils.error import *

import sys

sys.path.append('g2o_generator/GIS_Extraction')

import csv_reader as GIS

sns.set(rc={'figure.figsize':(12, 6)} )

def graph_plot(graph, pregraph, animate = False, poseEdgesPlot = True, landmarkEdgesPlot = False, gpsEdgesPlot = False, ontop = False):
    
    fig , ax = plt.subplots()
    plt.clf()
    colorlist = colors()
    
    from run_slam import get_poses_landmarks
    poses, landmarks, lm_ID, gps = get_poses_landmarks(graph)
    pre_poses, pre_landmarks, pre_lm_ID, _ = get_poses_landmarks(pregraph)
    
    #plot poses and landmarks if exits
    if len(poses) > 0:
        poses = np.stack(poses, axis=0) # axis = 0 turns into integers/slices and not tuple
        plt.plot(poses[:,0], poses[:,1], color='royalblue', marker='o', markersize=4,label = 'Post Optimization',zorder = 2)
        #plt.quiver(poses[:,0], poses[:,1], np.cos(poses[:,2]),np.sin(poses[:,2]), angles= 'xy',scale=0.5)
    
    if ontop:
        if len(pre_poses):
            pre_poses = np.stack(pre_poses, axis=0) # axis = 0 turns into integers/slices and not tuple
            plt.plot(pre_poses[:,0], pre_poses[:,1], color= 'skyblue', alpha = 0.35, marker='o', markersize=4, label = 'Pre Optimization',zorder = 1)

        
    
    if len(landmarks) > 0:
        landmarks = np.stack(landmarks, axis=0)
        # plt.plot(landmarks[:,0], landmarks[:,1], 'r*', markersize=12, label = 'Landmark')
        plt.scatter(landmarks[:,0], landmarks[:,1], label = 'Landmark', marker='*', color='firebrick', s=120, zorder=3, alpha = 0.7)
        # for lx, ly, ID in zip(landmarks[:,0], landmarks[:,1], lm_ID):
        #     plt.annotate(str(ID), xy=(lx, ly), color='r', zorder=2, alpha = 0.5)
    
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
            plt.plot(poseEdgeX_corr,poseEdgeY_corr,'k--', alpha = 0.8,zorder=4)
            plt.plot([],[], 'k--', label='Pose Edges')

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
            plt.plot([],[],'k--',label = 'Land edges')

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
            plt.plot([],[],'k--', label = 'Bearing edges')
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
        plt.xlabel('x (m)', fontsize = 14)
        plt.ylabel('y (m)', fontsize = 14)
        plt.legend(frameon=False,loc='lower center', ncol=5)
        plt.draw()
        plt.pause(1)

    else:

        plt.axis('equal')
        plt.xlabel('x (m)', fontsize="x-large")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel('y (m)', fontsize="x-large")
        plt.legend(frameon=False,loc='lower center', ncol=5, fontsize = 14)

    
    plt.tight_layout()
    plt.savefig("results/Owndata/Fullroute.png")
    return

def plot_ground_together_noise(n_graph, g_graph, pre_graph, lm_plot: bool=False):
    
    # from utils.slam_iterate import FOV, PHI, LAMBDAH
    fig , (ax1,ax2) = plt.subplots(1, 2)
    gposes, glandmarks, lm_ID, _ = get_poses_landmarks(g_graph)
    nposes, nlandmarks, lm_ID, _ = get_poses_landmarks(n_graph)
    pposes, prelandmarks, lm_ID, _ = get_poses_landmarks(pre_graph)
    
    if len(gposes) > 0:
        gposes = np.stack(gposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        ax1.plot(gposes[:,0], gposes[:,1], 'forestgreen', linestyle='dashed')#,label='Ground truth')
        ax2.plot(gposes[:,0], gposes[:,1], 'forestgreen',linestyle='dashed')#,label='Ground truth')
        

    if len(nposes) > 0:
        nposes = np.stack(nposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        ax2.plot(nposes[:,0], nposes[:,1], 'royalblue')#,label='Noisy route')

    if len(pposes) > 0:
        pposes = np.stack(pposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        ax1.plot(pposes[:,0], pposes[:,1], 'firebrick')#,label='Before Optimization route')

    if lm_plot == True:
        
        if len(nlandmarks) > 0:
            nlandmarks = np.stack(nlandmarks, axis=0)
            # plt.plot(landmarks[:,0], landmarks[:,1], 'r*', markersize=12, label = 'Landmark')
            ax2.scatter(nlandmarks[:,0], nlandmarks[:,1], marker='*', color='royalblue', s=120, zorder=10)
            for lx, ly, ID in zip(nlandmarks[:,0], nlandmarks[:,1], lm_ID):
                ax2.annotate(str(ID), xy=(lx, ly), color='royalblue',alpha = 0.8, zorder=11)
            
        if len(prelandmarks) > 0:
            prelandmarks = np.stack(prelandmarks, axis=0)
            # plt.plot(landmarks[:,0], landmarks[:,1], 'r*', markersize=12, label = 'Landmark')
            ax1.scatter(prelandmarks[:,0], prelandmarks[:,1], marker='*', color='firebrick', s=120, zorder=14)
            for lx, ly, ID in zip(prelandmarks[:,0], prelandmarks[:,1], lm_ID):
                ax1.annotate(str(ID), xy=(lx, ly), color='firebrick', zorder=15)

        if len(glandmarks) > 0:
            glandmarks = np.stack(glandmarks, axis=0)
            # plt.plot(landmarks[:,0], landmarks[:,1], 'r*', markersize=12, label = 'Landmark')
            ax1.scatter(glandmarks[:,0], glandmarks[:,1], marker='*', color='royalblue', s=120, zorder=12)
            ax2.scatter(glandmarks[:,0], glandmarks[:,1], marker='*', color='royalblue', s=120, zorder=12)

            for lx, ly, ID in zip(glandmarks[:,0],glandmarks[:,1], lm_ID):
                ax1.annotate(str(ID), xy=(lx, ly), color='royalblue',alpha = 0.8,zorder=13)
                ax2.annotate(str(ID), xy=(lx, ly), color='royalblue',alpha = 0.8,zorder=13)

    # plt.suptitle('Before and after optimization')
    ax1.axis('equal')
    ax1.set_xlabel('x (m)', fontsize="x-large")
    ax1.set_ylabel('y (m)', fontsize="x-large")
    # ax1.set_title('Before Optimization')
    ax2.axis('equal')
    ax2.set_xlabel('x (m)', fontsize="x-large")
    ax2.set_ylabel('y (m)', fontsize="x-large")

    # ax2.set_title('After Optimization')

    ax1.legend(['Ground truth', 'Odometry'], frameon = False, loc = 'upper left', fontsize = 14)
    ax2.legend(['Ground truth','Odometry'], frameon = False, loc = 'upper left',fontsize = 14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    plt.savefig("results/Owndata/ground_Together.png")
    plt.show()
    return
    

def poses_per_landmark(graph, deg: bool=False, pre: bool=False):

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

    # sns.histplot(poses, bins= len(p_values), kde=True)
    # plt.xlabel('Pose ID')
    # plt.title('Landmarks acquired per pose')
    # plt.show()

    
    if deg:
        ax3 = sns.histplot(np.rad2deg(total[:,1]), bins=len(l_values))
        ax3.set_xlabel('Bearing (deg)')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        

    else:
        ax3 = sns.histplot((total[:,1]), bins=len(l_values), color = 'royalblue')
        ax3.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4)) # denom needs to be written here and in multipleformatter input param for denom
        ax3.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 16))
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        ax3.set_xlabel('Bearing (rad)')
        

    ax3.legend(['Bearing count'],frameon=False)
    # ax3.set_title('Bearing angles to landmarks')
    if pre:
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        plt.savefig("results/Owndata/bearingcountpre.png")
    else:
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        plt.savefig("results/Owndata/bearingcountpost.png")

    plt.show()


    ax4 = sns.histplot(total[:,0], bins=len(l_values), color = 'royalblue')
    # ax4.set_title('Observations per unique landmark')
    ax4.set_xlabel('Landmark ID')
    ax4.legend(['Landmark count'],frameon=False)
    if pre:
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig("results/Owndata/Landmarkcountpre.png")
    else:
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig("results/Owndata/Landmarkcountpost.png")

        

    plt.show()

    return

def landmark_ba(n_graph, g_graph, pre_graph):

    _, nlandmarks, nlm_ID, _ = get_poses_landmarks(n_graph)
    _, glandmarks, glm_ID, _ = get_poses_landmarks(g_graph)
    _, prelandmarks, prelm_ID, _ = get_poses_landmarks(pre_graph)
    
    
    fig, axes = plt.subplots(1,2, figsize = (14,6))

    if len(nlandmarks) > 0:
        nlandmarks = np.stack(nlandmarks, axis=0)
        axes[1].scatter(nlandmarks[:,0], nlandmarks[:,1], marker='*', color='royalblue',  s=120, zorder=11)
        for lx, ly, ID in zip(nlandmarks[:,0], nlandmarks[:,1], nlm_ID):
            axes[1].annotate(str(ID), xy=(lx, ly), color='royalblue',alpha = 0.8, zorder=11)

    if len(prelandmarks) > 0:
        prelandmarks = np.stack(prelandmarks, axis=0)
        axes[0].scatter(prelandmarks[:,0], prelandmarks[:,1], marker='*', color='firebrick', s=120, zorder=11)
        for lx, ly, ID in zip(prelandmarks[:,0], prelandmarks[:,1], prelm_ID):
            axes[0].annotate(str(ID), xy=(lx, ly), color='firebrick', zorder=11)

    if len(glandmarks) > 0:
        glandmarks = np.stack(glandmarks, axis=0)
        axes[0].scatter(glandmarks[:,0], glandmarks[:,1], marker='*', color='forestgreen', alpha=0.8, s=120, zorder=10)
        axes[1].scatter(glandmarks[:,0], glandmarks[:,1], marker='*', color='forestgreen', alpha=0.8, s=120, zorder=10)
        for lx, ly, ID in zip(glandmarks[:,0], glandmarks[:,1], glm_ID):
            axes[0].annotate(str(ID), xy=(lx, ly), color='forestgreen', alpha = 0.8, zorder=10)
            axes[1].annotate(str(ID), xy=(lx, ly), color='forestgreen', alpha = 0.8, zorder=10)



    
    # plt.suptitle('Landmark position before and after Optimization')
    axes[0].set_xlabel('x (m)', fontsize="x-large")
    axes[1].set_xlabel('x (m)', fontsize="x-large")
    axes[0].set_ylabel('y (m)', fontsize="x-large")
    axes[1].set_ylabel('y (m)', fontsize="x-large")
    axes[0].axis('equal')
    axes[1].axis('equal')
    axes[0].legend(['Pre Optimization','Ground truth landmarks'], frameon=False, loc = 'upper left', fontsize=14)
    axes[1].legend(['Post Optimization','Ground truth landmarks'],frameon=False, loc = 'upper left', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("results/Owndata/landmarkpositionsbeforeafter.png")

    diff_gn = [elem for elem in glm_ID if elem not in nlm_ID]
    diff_gp = [elem for elem in prelm_ID if elem not in nlm_ID]
    corr_glandmarks = np.delete(glandmarks,diff_gn, axis=0)
    corr_prelandmarks = np.delete(prelandmarks,diff_gp, axis=0)
    
    print(len(corr_glandmarks))
    print(len(corr_glandmarks))
    diff_ng = np.around((nlandmarks-corr_glandmarks),2)
    diff_pg = np.around((corr_prelandmarks-corr_glandmarks),2)

    xn = np.expand_dims(diff_ng[:,0],axis = 1)
    yn = np.expand_dims(diff_ng[:,1],axis = 1)
    xp = np.expand_dims(diff_pg[:,0],axis = 1)
    yp = np.expand_dims(diff_pg[:,1],axis = 1)
    xn_sum = round(np.mean(np.abs(xn)),2)
    yn_sum = round(np.mean(np.abs(yn)),2)
    xp_sum = round(np.mean(np.abs(xp)),2)
    yp_sum = round(np.mean(np.abs(yp)),2)

    # print(f"landmark{xn_sum}, {yn_sum}, {xp_sum}, {yp_sum}")
    ng_sum = round(np.mean(np.abs(diff_ng)),2)
    pg_sum = round(np.mean(np.abs(diff_pg)),2)


    fig1, ax1 = plt.subplots()    
    
    ax1.scatter(diff_ng[:,0], diff_ng[:,1], color = 'royalblue', label = ' Post landmark error', marker =  '*', s=120)
    # for lx, ly, ID in zip(x, y, nlm_ID):
        #    ax1.annotate(str(ID), xy=(lx, ly), color='b', alpha = 0.8)
    
    ax1.scatter(diff_pg[:,0], diff_pg[:,1], color = 'firebrick', label = ' Pre landmark error', marker = '*', s=120)
    # for lx, ly, ID in zip(x, y, prelm_ID):
        #    ax2.annotate(str(ID), xy=(lx, ly), color='r', alpha = 0.8)
    # ax1.set_title('Odometry Post and Ground error')
    # ax2.set_title('Odometry Pre and Ground error')
    ax1.set_xlabel('x (m)', fontsize="x-large")
    # ax2.set_xlabel('x (m)', fontsize = 14)
    ax1.set_ylabel('y (m)', fontsize="x-large")
    ax1.legend(frameon=False,fontsize = 14)
    print((xn_sum,yn_sum,xp_sum,yp_sum))
    np.savetxt("results/Owndata/ng_pg_sum.txt", (ng_sum,pg_sum), fmt="%s")
    np.savetxt("results/Owndata/xy_landmark_sum_post_pre.txt", (xn_sum,yn_sum,xp_sum,yp_sum), fmt="%s")
    print(f"Noisy landmark sum:\n{ng_sum}\n Pre landmark sum:\n{pg_sum}")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("results/Owndata/landmarkerror_pre_post.png")
    plt.axis('equal')
    plt.show()
    print(nlm_ID)
    row_sum_ng = [abs(sum(x)) for x in diff_ng]
    row_ng_lm = np.stack(zip(nlm_ID,row_sum_ng))
    
    df = pd.DataFrame(data = row_ng_lm, columns = ['Landmark ID', 'Sum of x,y error (m)'])
    # # sns.heatmap(x)len()
    # # sns.histplot(diff_ng)#, bins = len(nlandmarks))
    snsb = sns.barplot(data = df, x= 'Landmark ID', y = 'Sum of x,y error (m)', color = 'royalblue')
    snsb.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("results/Owndata/landmarkerrorsum.png")

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
    # print(ATE_rmse)
 
    _, (ax1, ax2, ax3) = plt.subplots(3,1)

    ax1.plot(E[:,0])
    ax1.set_ylabel('x (m)')

    ax2.plot(E[:,1])
    ax2.set_ylabel('y (m)')

    ax3.plot(E[:,2])
    ax3.set_ylabel('$\\theta$ [rad]')

    # plt.suptitle('Absolute Trajectory error')
    plt.legend(frameon=False)
    plt.show()
    # sns.histplot(E[:,0])
    # plt.show()
    # sns.kdeplot(E[:,0])
    # plt.show()

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

def plot_errors(e_full, pose_error, bearing_error,land_error,gps_error):

    e_pose = np.vstack((pose_error))
    if len(pose_error) >0:
        f1, (ax1,ax2,ax3) = plt.subplots(1,3,sharex = True)
        ax1.plot(e_pose[1:,0], color='forestgreen', marker="o", label ='x')
        ax1.legend(loc="upper right", fontsize = 14,frameon=False)
        ax1.set_ylabel('Absolute error (m)', fontsize="x-large")
        ax2.plot(e_pose[1:,1], color='royalblue', marker="o", label ='y')
        ax2.set_ylabel('Absolute error (m)', fontsize="x-large")
        ax2.legend(loc="upper right", fontsize = 14,frameon=False)
        ax3.plot(e_pose[1:,2], color='firebrick', marker="o", label ='$\\theta$')
        ax3.set_ylabel('Absolute error (rad)', fontsize="x-large")
        ax3.legend(loc="upper right", fontsize = 14,frameon=False)
        ax2.set_xlabel('Iteration', fontsize="x-large")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig("results/Owndata/pose_error_split.png")


   
    
    if len(bearing_error)>0:

        _, ax4 = plt.subplots()
        ax4.plot(bearing_error, color='royalblue', marker="o", label ='Bearing error')
        ax4.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig("results/Owndata/bearing_error.png")

        # ax4 = sns.relplot(data = bearing_error, kind="line", ci = 100, label = 'yeeeeet')
        # ax4.set_title('Bearing error')
        
        #ax4.plot(bearing_error, label = 'bearing error')
        
        
    
    if len(land_error)>0:

        e_land = (np.vstack((land_error)))
        _, ax5 = plt.subplots()

        ax5.plot(e_land[:,0], color='forestgreen', label ='Landmark x error', marker="o")
        ax5.legend()
        ax5.plot(e_land[:,1], color='royalblue', label ='Landmark y error', marker="o")
        ax5.set_yscale('log')
        ax5.set_ylabel('Absolute error (m)')
        ax5.set_xlabel('Iteration')
        ax5.set_xticks(range(len(e_land)))
        ax5.legend(fontsize=14,frameon=False)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig("results/Owndata/land_error_log.png")

    if len(gps_error)>1:
        e_gps = np.abs(np.vstack((gps_error)))
        _, (ax7,ax8) = plt.subplots(1,2)
        ax7.plot(e_gps[:,0], color='forestgreen', label ='x gps error')
        ax7.legend()
        ax8.plot(e_gps[:,1], color='royalblue', label ='y gps error')
        ax8.legend()

    _, ax9 = plt.subplots()
    ax9.plot(e_full, color='royalblue', marker="o")#,label = 'Full error')
    ax9.set_xlabel('Iteration', fontsize="x-large")
    ax9.set_ylabel('Absolute error', fontsize="x-large")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("results/Owndata/pose_error_full.png")


    # ax9.legend(loc="upper right", fontsize = 12)
 
    plt.show()

    return

def color_error_plot(n_graph,g_graph):

    gposes, _, _, _ = get_poses_landmarks(g_graph)
    nposes, _, _, _ = get_poses_landmarks(n_graph)
    nposes = np.stack(nposes, axis=0)
    gposes = np.stack(gposes, axis=0)

    _, ax = plt.subplots()

    # e = np.sum(np.abs((gposes[:,:2]-nposes[:,:2])),axis=1)

    e = np.sum(np.abs((gposes[:,:2]-nposes[:,:2])),axis=1)

    # print(e)
    x = nposes[:,0]
    y = nposes[:,1]
    xg = gposes[:,0]
    yg = gposes[:,1]

    bob = ax.scatter(x,y,c=e, cmap="viridis", alpha=0.8)
    cbar = plt.colorbar(bob)
    cbar.set_label('Absolute x, y error', rotation = 270, fontsize="x-large")
    cbar.ax.get_yaxis().labelpad = 15

    # ax.plot(x,y, 'royalblue')
    # ax.plot(xg,yg, 'forestgreen',linestyle = 'dashed')
    # ax.legend(['Odometry','Ground'], fontsize = 14)
    # plt.title('Odometry error', fontsize = 16)
    plt.xlabel('x (m)', fontsize="x-large")
    plt.ylabel('y (m)', fontsize="x-large")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("results/Owndata/error_color_route.png")

    plt.show()

    # print(f"e is  {e}")
    error_direct = np.sum(np.abs((gposes-nposes)),axis=1)
    # np.savetxt("results/Owndata/error_pose_split_check.txt", error_direct, fmt="%s")
    plt.plot(error_direct, label = 'Error')
    plt.legend(frameon=False, fontsize = 14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("results/Owndata/error_basic.png")

    # plt.title('error bob 1')
    plt.show()
    # plt.plot(e[:1])
    # plt.title('error bob 2')
    # plt.show()
    
    return
    
    
def error_plot(n_graph,g_graph,pre_graph):

    gposes, _, _, _ = get_poses_landmarks(g_graph)
    nposes, _, _, _ = get_poses_landmarks(n_graph)
    pre_poses, _, _, _ = get_poses_landmarks(pre_graph)


    nposes = np.stack(nposes, axis=0)
    gposes = np.stack(gposes, axis=0)
    pre_poses = np.stack(pre_poses,axis=0)
    diffng =[]
   

    fig , (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(14,12))
    diffng = gposes-nposes
    
    for i in range(len(diffng[:,2])):

        diffng[i,2]= wrap2pi((diffng[i,2]))
        # nposes[i,2] = wrap2pi((nposes[i,2]))

    ax1.plot(gposes[:,0], color = 'forestgreen',linestyle = 'dashed')
    ax2.plot(gposes[:,1], color = 'forestgreen',linestyle = 'dashed')
    ax3.plot(gposes[:,2], color = 'forestgreen',linestyle = 'dashed')

    ax1.plot(nposes[:,0], color = 'royalblue')
    ax2.plot(nposes[:,1], color = 'royalblue')
    ax3.plot(nposes[:,2], color = 'royalblue')

    ax1.plot(pre_poses[:,0], color = 'firebrick')
    ax2.plot(pre_poses[:,1], color = 'firebrick')
    ax3.plot(pre_poses[:,2], color = 'firebrick')

    ax1.plot(np.abs(diffng[:,0]), color = 'yellow')
    ax2.plot(np.abs(diffng[:,1]), color = 'yellow')
    ax3.plot(np.abs(diffng[:,2]), color = 'yellow')
    ax1.legend(['Ground truth','Post Optimization', 'Pre Optimization','Abs error'], frameon=False, fontsize = 14)
    ax1.set_ylabel('x (m)', fontsize="x-large")
    # ax2.legend(['Ground truth','Noisy','Error(diff)'])
    ax2.set_ylabel('y (m)', fontsize="x-large")
    # ax3.legend(['Ground truth','Noisy','Error(diff)'])
    ax3.set_ylabel('$\\theta$ (rad)', fontsize="x-large")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("results/Owndata/full_diff_errorplot.png")

    # plt.suptitle('Pose error between Odometry and Ground truth')
    
    
    plt.show()

def dcs_arrayplot(dcs_array):
    _, ax1 = plt.subplots()
    # ax1 = sns.kdeplot(dcs_array, ax = ax1)
    
    sns.histplot(dcs_array, color = 'royalblue')
    plt.tight_layout()
    plt.show()



def plot_map(n_graph, g_graph, post:bool = True):
    odometry_file = 'g2o_generator/robosim/data/robopath/fullroute120.json'
    odometry = load_from_json(odometry_file)

    temp_x = np.asfarray(odometry[0]); temp_y = np.asfarray(odometry[1]); temp_th = np.asfarray(odometry[2])

    nposes, _, _, _ = get_poses_landmarks(n_graph)
    gposes, _, _, _ = get_poses_landmarks(g_graph)


    _, axs = plt.subplots()
    

    axs.set_xlim([574750, 575100])
    axs.set_ylim([6222350, 6222700])

    axs.set_aspect('equal', 'datalim')
    if len(nposes) > 0:
        nposes = np.stack(nposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        if post:
            axs.plot(nposes[:,0]+temp_x[0], nposes[:,1]+temp_y[0], 'royalblue',label = 'Odometry route')
        else:
            axs.plot(nposes[:,0]+temp_x[0], nposes[:,1]+temp_y[0], 'firebrick',label = 'Odometry route')

    if len(gposes) > 0:
        gposes = np.stack(gposes, axis=0) # axis = 0 turns into integers/slices and not tuple
        axs.plot(gposes[:,0]+temp_x[0], gposes[:,1]+temp_y[0], 'forestgreen', linestyle = 'dashed', label = 'Ground truth')
    axs.set_xlabel('x (m)', fontsize="x-large")
    axs.set_ylabel('y (m)', fontsize="x-large")
    
    axs.legend(fontsize = 14, frameon = False)

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
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    plt.savefig("results/Owndata/map_plot.png")

    plt.show()




def colors():

    CB91_Blue = "#2CBDFE"
    CB91_Green = "#47DBCD"
    CB91_Pink = "#F3A0F2"
    CB91_Purple = "#9D2EC5"
    CB91_Violet = "#661D98"
    CB91_Amber = "#F5B14C"
    ROBO_Green = "006800ff"
    color_list = [
        CB91_Blue,
        CB91_Pink,
        CB91_Green,
        CB91_Amber,
        CB91_Purple,
        CB91_Violet,
        ROBO_Green,
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


def color_error_plot3d(n_graph,g_graph):

    gposes, _, _, _ = get_poses_landmarks(g_graph)
    nposes, _, _, _ = get_poses_landmarks(n_graph)
    nposes = np.stack(nposes, axis=0)
    gposes = np.stack(gposes, axis=0)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = plt.axes(projection='3d')

    # e = np.sum(np.abs((gposes[:,:2]-nposes[:,:2])),axis=1)

    e = np.sum(np.abs((gposes[:,:2]-nposes[:,:2])),axis=1)

    # print(e)
    x = nposes[:,0]
    y = nposes[:,1]
    xg = gposes[:,0]
    yg = gposes[:,1]

    zline = np.linspace(0, 10, len(gposes))

    # bob = ax.scatter3D(x,y,c=e, cmap="viridis", alpha=0.8)
    # cbar = plt.colorbar(bob)
    # cbar.set_label('Absolute x, y error', rotation = 270, fontsize="x-large")
    # cbar.ax.get_yaxis().labelpad = 15

    k = ax.scatter3D(x,y,zline,c=e,cmap='viridis', alpha=0.5)
    cbar = plt.colorbar(k)
    cbar.set_label('Absolute x, y error', rotation = 270, fontsize="x-large")
    cbar.ax.get_yaxis().labelpad = 15

    # ax.plot(x,y, 'royalblue')
    # ax.plot(xg,yg, 'forestgreen',linestyle = 'dashed')

    # ax.legend(frameon=False, fontsize = 14)
    # # plt.title('Odometry error', fontsize = 16)
    plt.xlabel('x (m)', fontsize="x-large")
    plt.ylabel('y (m)', fontsize="x-large")
    # ax.set_zlabel('y (m)', fontsize="x-large")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig("results/Owndata/color_ploterror3D.png")

    plt.show()
