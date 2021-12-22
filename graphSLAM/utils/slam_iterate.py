import warnings
import numpy as np

from utils.helper import *
from utils.error import *
from utils.graph_plot import *
from utils.g2o_loader import load_g2o_graph
from tqdm import trange
warnings.filterwarnings('ignore')


LAMBDAH = 1e-10
PHI = 1
FOV = 120 # Degrees
LM_RANGE = 15 # Meters
ODO_RANGE = 2

# def solve_system(H,b,lambdaH: float = 1.0):

    
#     H_damp = (H+lambdaH*np.eye(H.shape[0]))
#     H_sparse = csr_matrix(H_damp)
#     sparse_dxstar = spsolve(H_sparse,-b)
#     dX = sparse_dxstar.squeeze()#

#     return dX, H_sparse

def graph_slam_run_algorithm(graph, numIter, g_graph, pre_noise):

    tol = 1e-10# If error difference is smaller than tolerance it breaks.
    norm_dX_all = []
    err_opt_f = []
    err_diff = []
    diff = []
    e_pose = []
    e_bear = []
    e_land = []
    e_gps = []
    
    graph_plot(graph,figid=2, Label = '',landmarkEdgesPlot=False)
    plt.title('Before optimization')
    plt.show()

    lambdaH = LAMBDAH
    for i in trange(numIter, position=0, leave=True, desc='Running SLAM algorithm'):
        
        # if i>0:
        old_x = graph.x
        
        error_before, err_pose , err_bearing , err_land, err_gps  = compute_global_error(graph)
        
        err_opt_f.append(error_before)
        e_pose.append(err_pose)
        e_bear.append(err_bearing)
        # e_land.append(err_land)
        # e_gps.append(err_gps)
        from utils.linearize import linearize_solve
        print(f"lambdaH slamiterate{lambdaH}")
        dX, _, _, _ = linearize_solve(graph,lambdaH=lambdaH)

        graph.x += dX

        if i>1:

            err_diff = err_opt_f[i-1]-err_opt_f[i] # E - error(x)
            print(f"error diff : {err_diff}")
            if err_diff < 0:
                graph.x = old_x
            #     #print(f"OLD X RECOVERED\n")
                lambdaH *= 5
            #     #print("Error is larger than previous iteration")
            #     #print(f"lambda is: {lambdaH}")
            else:
                lambdaH /= 5
            #     #print(f"lambda is: {lambdaH}")
        print(f"lambda after slam iterate{lambdaH}")
        diff = np.append(diff,err_diff)

        norm_dX = np.linalg.norm(dX)

        print(f"|dx| for step {i} : {norm_dX}\n")
        norm_dX_all.append(norm_dX)

        if i >=1 and np.abs(norm_dX_all[i]-norm_dX_all[i-1]) < tol: 
            break
    
    
    # graph_plot(graph,figid=3,Label = '')
    # plt.title('After optimization\n Damping coeff={}, DCS={}, FOV={}'.format(LAMBDAH,PHI,FOV),loc = 'center')
    # graph_plot(pre_noise,figid=4,Label = '')
    # plt.title('Before optimization')
    # plot_ground_together_noise(g_graph, graph,figid=5)
    # plt.title('After optimization\n Damping coeff={}, DCS={}, FOV={}'.format(LAMBDAH,PHI,FOV),loc = 'center')
    # plot_ground_together_noise(g_graph, pre_noise,figid=6)
    # plt.title('Before optimization')
    # plt.show()
    # landmark_ba(graph,pre_noise,g_graph)
    # ATE(graph, g_graph)
    
    
    plot_errors(err_opt_f,e_pose,e_bear,e_land,e_gps)
   
    return norm_dX_all