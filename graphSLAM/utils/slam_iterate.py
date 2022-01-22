import warnings
import numpy as np
from utils.helper import *
from utils.error import *
from utils.graph_plot import *
from utils.g2o_loader import load_g2o_graph
from tqdm import trange
import imageio
import os
warnings.filterwarnings('ignore')

NOISE_FILENAME = '20211230-132841'
LAMBDAH = 1e-1
PHI = 0.2
FOV = 120 # Degrees
LM_RANGE = 20 # Meters
ODO_RANGE = 2 #


def graph_slam_run_algorithm(graph, numIter, g_graph, pre_noise):

    tol = 1e-4# If error difference is smaller than tolerance it breaks.
    norm_dX_all = []
    err_opt_f = []
    err_diff = []
    diff = []
    e_pose = []
    e_bear = []
    e_land = []
    e_gps = []
    
    filenames = []
    frames = []

    # graph_plot(graph,pre_noise,landmarkEdgesPlot=False)
    
    # plt.show()

    lambdaH = LAMBDAH
    for i in trange(numIter, position=0, leave=True, desc='Running SLAM algorithm'):
        
        if i>0:
            old_x = graph.x
        
        error_before, err_pose , err_bearing , err_land, err_gps = compute_global_error(graph)
        # print(f"error bearing iteration\n:{err_bearing}\n")
        # print(f"error pose iteration\n:{err_pose}\n")
        # print(f"error full iteration\n:{error_before}\n")

        err_opt_f.append(error_before)
        e_pose.append(err_pose)
        e_bear.append(err_bearing)
        # e_land.append(err_land.reshape(1,2))
        # e_gps.append(err_gps)

        from utils.linearize import linearize_solve
        
        dX, _, _, dcs_array = linearize_solve(graph,lambdaH=lambdaH)


        graph.x += dX

        if i>0:
            err_diff = err_opt_f[i-1]-err_opt_f[i] # E - error(x)
            print(f"error diff : {err_diff}")
            if err_diff < 0:
                graph.x = old_x
                lambdaH *= 2
            else:
                lambdaH /= 10


        # graph_plot(graph, pre_noise)
        # plt.title(f'iteration: {i}')
        
        # filename = f'slamiter{i}.png'
        # filenames.append(filename)

        # plt.savefig('./graphSLAM/utils/figs/'+ filename)
        # plt.close()

        # with imageio.get_writer('hej.gif', mode='I') as writer:
            # for filename in filenames:
            #     image = imageio.imread('./graphSLAM/utils/figs/'+ filename)
            #     # image = imageio.imread(filename)
                
            #     writer.append_data(image)
            # frames.append(image)
        


        diff = np.append(diff, err_diff)

        norm_dX = np.linalg.norm(dX)
        norm_dX_all.append(norm_dX)

        print(f"|dx| for step {i} : {norm_dX}\n")

        if i >=1 and np.abs(norm_dX_all[i]-norm_dX_all[i-1]) < tol: 
            break
    

    # imageio.mimsave('./graphSLAM/utils/figs/'+NOISE_FILENAME+'.gif', frames, format='GIF', fps=2)
    # # Remove files
    # for filename in set(filenames):
    #     os.remove('./graphSLAM/utils/figs/'+ filename)
    
    np.savetxt("results/Owndata/error_full_array.txt", err_opt_f, fmt="%s")
    np.savetxt("results/Owndata/rel_change_dx.txt", norm_dX_all, fmt="%s")
    np.savetxt("results/Owndata/pose_error_split.txt", e_pose, fmt="%s")
    np.savetxt("results/Owndata/params.txt", (LAMBDAH,PHI,FOV), fmt="%s")


    # dcs_arrayplot(dcs_array)

    graph_plot(graph, pre_noise, ontop=True)
    plot_ground_together_noise(graph, g_graph, pre_noise, lm_plot=False)
    plot_map(graph,g_graph)
    poses_per_landmark(graph, pre = False)
    landmark_ba(graph,g_graph,pre_noise)
    color_error_plot(graph, g_graph)
    error_plot(graph, g_graph,pre_noise)
    # A_traj_error(graph, g_graph)
    # print(f"slamiter{e_land}")
    plot_errors(err_opt_f, e_pose, e_bear, e_land, e_gps)
   
    return norm_dX_all