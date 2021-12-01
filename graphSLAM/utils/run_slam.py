import warnings
import numpy as np
from linearize_solve import *
from helper import *
from error import *
from graph_plot import *
from g2o_loader import load_g2o_graph
from tqdm import trange
warnings.filterwarnings('ignore')

class Graph: 
    def __init__(self, x, nodes, edges,lut, nodeTypes, withoutBearing=False):
        self.x = x
        self.nodes = nodes
        self.edges = edges
        self.lut = lut
        self.withoutBearing = withoutBearing
        self.nodeTypes = nodeTypes
        #self.lambdaH = lambdaH


class G2O:
    def __init__(self, filename):
        
        self.graph = load_g2o_graph(filename,noBearing=False)
        self.error = compute_global_error(self.graph)
        


def graph_slam_run_algorithm(graph, numIter):

    tol = 1e-10 # If error difference is smaller than tolerance it breaks.
    norm_dX_all = []
    err_opt_f = []
    err_diff = []
    diff = []
    e_pose = []
    e_land = []
    e_gps = []
    
    graph_plot(graph)
    lambdaH = 1e-10
    for i in trange(numIter, position=0, leave=True, desc='Running SLAM algorithm'):
        
        #if i>0:
        old_x = graph.x
        #print(f"OLD X TOP OF FOR LOOP:\n{old_x}\n")
        
        error_before, err_pose , err_land , err_gps = compute_global_error(graph)
        #print(f"global error before:\n{error_before}\n")
        err_opt_f.append(error_before)
        e_pose.append(err_pose)
        e_land.append(err_land)
        e_gps.append(err_gps)

        dX, _, _, _ = linearize_solve(graph,lambdaH=lambdaH)
        graph.x += dX

        if i>0:
            err_diff = err_opt_f[i-1]-err_opt_f[i] # E - error(x)
            #print(f"error diff : {err_diff}")
            if err_diff < 0:
                graph.x = old_x
                print(f"OLD X RECOVERED\n")
                lambdaH *= 2
                print("Error is larger than previous iteration")
                print(f"lambda is: {lambdaH}")
            else:
                lambdaH /= 2
                print(f"lambda is: {lambdaH}")
            
        diff = np.append(diff,err_diff)
        
        
        norm_dX = np.linalg.norm(dX)

        print(f"|dx| for step {i} : {norm_dX}\n")
        norm_dX_all.append(norm_dX)

        if i >=1 and np.abs(norm_dX_all[i]-norm_dX_all[i-1]) < tol: 
            break

    graph_plot(graph)
    
    print(f"error diff array:\n{diff}\n")
    plot_ground_together_noise(g_graph, graph)
    #rmse_post = RMSE(graph.x, g_graph.x)
    #print(f"print RMSE Post: {rmse_post}")
    plt.plot(err_opt_f)
    plt.show()
    plot_errors(e_pose,e_land,e_gps)
    #globalerrorpost,_,_,_=compute_global_error(graph)
    #print(f"Global error after SLAM:\n{globalerrorpost}\n")
    return norm_dX_all


if __name__ == '__main__' :
   

    noise = G2O('graphSLAM/data/noise_20211130-214904.g2o')
    n_graph = noise.graph
    ground = G2O('graphSLAM/data/ground_truth_20211130-214904.g2o')
    g_graph = ground.graph
    plot_ground_together_noise(g_graph,n_graph)
    #Rmse_pre =RMSE(n_graph.x, g_graph.x)
    #print(f"print RMSE PRE: {Rmse_pre}")
    e,_,_,_ = noise.error
    print(e)
    #_, Hs ,_,_= linearize_solve(n_graph)
    # plt.spy(Hs)
    # plt.show()
    #error_before, _ , _ , _ = compute_global_error(n_graph)
    

    dx = graph_slam_run_algorithm(n_graph,10)