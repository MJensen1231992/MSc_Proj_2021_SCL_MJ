import warnings
import numpy as np
import copy
from utils.linearize_solve import *
from utils.helper import *
from utils.error import *
from utils.graph_plot import *
from utils.g2o_loader import load_g2o_graph
from tqdm import trange
warnings.filterwarnings('ignore')

LAMBDAH = 1
PHI = 1
FOV = 120 # Degrees
LM_RANGE = 20 # Meters
ODO_RANGE = 2

class Graph: 
    def __init__(self, x, nodes, edges,lut, nodeTypes, withoutBearing=False):
        self.x = x
        self.nodes = nodes
        self.edges = edges
        self.lut = lut
        self.withoutBearing = withoutBearing
        self.nodeTypes = nodeTypes


class G2O:
    def __init__(self, filename, gt: bool):
        
        self.graph = load_g2o_graph(filename, gt, noBearing=False)
        self.error = compute_global_error(self.graph)


def graph_slam_run_algorithm(graph, numIter):

    tol = 0 # If error difference is smaller than tolerance it breaks.
    norm_dX_all = []
    err_opt_f = []
    err_diff = []
    diff = []
    e_pose = []
    e_land = []
    e_gps = []
    e_bear = []

    graph_plot(graph, figid=1, Label='', landmarkEdgesPlot=True)
    plt.title('Before optimization')
    plt.show()

    lambdaH = LAMBDAH
    for i in trange(numIter, position=0, leave=True, desc='Running SLAM algorithm'):

        old_x = graph.x
        error_before, err_pose, err_bearing, err_land , err_gps = compute_global_error(graph)

        err_opt_f.append(error_before)
        e_pose.append(err_pose)
        e_bear.append(err_bearing)
        # e_land.append(err_land)
        # e_gps.append(err_gps)

        dX, _, _, _ = linearize_solve(graph, lambdaH=lambdaH)
        graph.x += dX

        if i>0:
            err_diff = err_opt_f[i-1]-err_opt_f[i] # E - error(x)

            # Error x and y
            errx = e_pose[i-1][0] - e_pose[i][0] 
            erry = e_pose[i-1][1] - e_pose[i][1]

            print(f"error diff : {err_diff}\n")
            if err_diff < 0 or errx < 0 or erry < 0:
                graph.x = old_x
                lambdaH *= 2

            else:
                lambdaH /= 2


        diff = np.append(diff, err_diff)
        
        norm_dX = np.linalg.norm(dX)

        print(f"|dx| for step {i} : {norm_dX}\n")
        norm_dX_all.append(norm_dX)

        if i >=1 and np.abs(norm_dX_all[i]-norm_dX_all[i-1]) < tol:
            break
    
    

    graph_plot(graph,figid=3,Label = '')#),landmarkEdgesPlot=True)
    plt.title('After optimization\n Damping coeff={}, DCS={}, FOV={}'.format(LAMBDAH,PHI,FOV),loc = 'center')
    graph_plot(pre_graph,figid=4,Label = '')#,landmarkEdgesPlot=True)
    plt.title('Before optimization')
    plot_ground_together_noise(g_graph, graph,figid=5)
    plt.title('After optimization\n Damping coeff={}, DCS={}, FOV={}'.format(LAMBDAH,PHI,FOV),loc = 'center')
    plot_ground_together_noise(g_graph, pre_graph,figid=6)
    plt.title('Before optimization')
    # plt.legend('post')#
    plt.show()
    # RMSE(graph.x, g_graph.x) 
    
    plot_errors(err_opt_f, e_pose, e_bear, e_land, e_gps)
    return norm_dX_all


if __name__ == '__main__' :
   
    # noise = G2O('graphSLAM/data/giusensonoise.g2o')
    noise = G2O('graphSLAM/data/internet.g2o', gt=False)
    n_graph = noise.graph
    # # ground = G2O('graphSLAM/data/giusensoground.g2o')
    ground = G2O('graphSLAM/data/internet_gt.g2o', gt=True)
    g_graph = ground.graph

    pre_graph = copy.deepcopy(n_graph)

    plot_ground_together_noise(g_graph, n_graph, figid=4)

    dx = graph_slam_run_algorithm(n_graph,20)
    

