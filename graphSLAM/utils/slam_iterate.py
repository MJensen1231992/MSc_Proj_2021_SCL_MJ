import warnings
import numpy as np
from utils.helper import *
from utils.error import *
from utils.graph_plot import *
from utils.g2o_loader import load_g2o_graph
from tqdm import trange
import os
import imageio
warnings.filterwarnings('ignore')

NOISE_FILENAME = '20211230-132841'
LAMBDAH = 1
PHI = 0.2
FOV = 120 # Degrees
LM_RANGE = 20 # Meters
ODO_RANGE = 2 #


def graph_slam_run_algorithm(graph, numIter, g_graph, pre_noise):

    tol = 1e-10

    norm_dX_all = []
    err_opt_f = []
    err_diff = []
    diff = []
    e_pose = []
    e_bear = []
    e_land = []
    e_gps = []
    e_direct = []
    iter = []
    filenames = []
    frames = []

    lambdaH = LAMBDAH
    for i in trange(numIter, position=0, leave=True, desc='Running SLAM algorithm'):
        
        if i>0:
            old_x = graph.x
        
        error_before, err_pose , err_bearing , err_land, _ = compute_global_error(graph)
 

        err_opt_f.append(error_before)
        e_pose.append(err_pose)
        e_bear.append(err_bearing)
        e_dir = error_direct_calc(graph, g_graph)
        e_direct.append(e_dir)

        from utils.linearize import linearize_solve
        dX, _, _, _ = linearize_solve(graph,lambdaH=lambdaH)


        graph.x += dX

        if i>0:
            err_diff = err_opt_f[i-1]-err_opt_f[i] # E - error(x)
            print(f"error diff : {err_diff}")
            if err_diff < 0:
                graph.x = old_x
                lambdaH *= 2
            else:
                lambdaH /= 6


        
        graph_plot(graph, pre_noise)
        plt.title(f'iteration: {i}')
        
        filename = f'slamiter{i}.png'
        filenames.append(filename)

        plt.savefig('./graphSLAM/utils/figs/'+ filename)
        plt.close()

        with imageio.get_writer('hej.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread('./graphSLAM/utils/figs/'+ filename)
                writer.append_data(image)
            frames.append(image)
        


        diff = np.append(diff, err_diff)

        norm_dX = np.linalg.norm(dX)
        norm_dX_all.append(norm_dX)
        iter.append(i)
        print(f"|dx| for step {i} : {norm_dX}\n")
        
        if i >=1 and np.abs(norm_dX_all[i]-norm_dX_all[i-1]) < tol:
            
            break
    

    imageio.mimsave('./graphSLAM/utils/figs/'+NOISE_FILENAME+'.mp4', frames, format='GIF', fps=2)
   
    for filename in set(filenames):
        os.remove('./graphSLAM/utils/figs/'+ filename)
    os.remove('hej.gif')
        
    np.savetxt("results/Owndata/params.txt", (LAMBDAH,PHI,FOV,iter), fmt="%s")
    np.savetxt("results/Owndata/error_split_mean.txt", e_direct, fmt="%s")


    graph_plot(graph, pre_noise, ontop=True)
    plot_ground_together_noise(graph, g_graph, pre_noise, lm_plot=False)
    plot_map(graph,g_graph)
    
    color_error_plot(graph, g_graph)
    error_plot(graph, g_graph,pre_noise)
    plot_errors(err_opt_f, e_pose, e_bear, e_land, e_gps)
    landmark_ba(graph,g_graph,pre_noise)
    color_error_plot3d(graph, g_graph)
    
    return norm_dX_all