import warnings
import copy
import sys


from utils.linearize import *
from utils.helper import *
from utils.error import *
from utils.graph_plot import *
from utils.g2o_loader import load_g2o_graph
from utils.slam_iterate import *

sys.path.append
warnings.filterwarnings('ignore')


class Graph: 
    def __init__(self, x, nodes, edges,lut, nodeTypes, descriptor):
        self.x = x
        self.nodes = nodes
        self.edges = edges
        self.lut = lut
        self.nodeTypes = nodeTypes
        self.descriptor = descriptor 

class G2O:
    def __init__(self, filename, gt):
        
        self.graph = load_g2o_graph(filename,gt,descriptor=False)
        self.error = compute_global_error(self.graph)

if __name__ == '__main__' :

    noise = G2O('graphSLAM/data/noise_20220125-204649.g2o', gt=False)#120
    # noise = G2O('graphSLAM/data/noise_20220125-235259.g2o', gt=False)#30
    # noise = G2O('graphSLAM/data/noise_20220125-235701.g2o', gt=False)#60
    # noise = G2O('graphSLAM/data/noise_20220125-235959g2o', gt=False)#90
    # noise = G2O('graphSLAM/data/noise_20220126-000155.g2o', gt=False)#150
    # noise = G2O('graphSLAM/data/noise_20220126-000335.g2o', gt=False)#180
    n_graph = noise.graph
    
    
    ground = G2O('graphSLAM/data/ground_truth_20220125-204649.g2o', gt=True)#120
    # ground = G2O('graphSLAM/data/ground_truth_20220125-235259.g2o', gt=True)#30
    # ground = G2O('graphSLAM/data/ground_truth_20220125-235701.g2o', gt=True)#60
    # ground = G2O('graphSLAM/data/ground_truth_20220125-235959.g2o', gt=True)#90
    # ground = G2O('graphSLAM/data/ground_truth_20220125-000155.g2o', gt=True)#150
    # ground = G2O('graphSLAM/data/ground_truth_20220125-000335.g2o', gt=True)#180
  
    g_graph = ground.graph

    pre_graph = copy.deepcopy(n_graph)
    # plot_ground_together_noise(n_graph, g_graph, pre_graph)
    # plot_map(n_graph, g_graph,post = False)
    # poses_per_landmark(pre_graph, pre = True)

    # landmark_ba(n_graph,g_graph,pre_graph)
    # graph_plot(n_graph,pre_graph,ontop=False)
    # plt.show()
    # color_error_plot3d(n_graph,g_graph)
    dx = graph_slam_run_algorithm(n_graph, 100, g_graph, pre_graph)






 #route 1
    # noise = G2O('graphSLAM/data/noise_20220120-210013.g2o', gt=False)
    # noise = G2O('graphSLAM/data/noise_20220124-190221.g2o', gt=False)#120fov
    # noise = G2O('graphSLAM/data/noise_20220124-193659.g2o', gt=False)#60fov
    # noise = G2O('graphSLAM/data/noise_20220124-195759.g2o', gt=False)#180fov
    # noise = G2O('graphSLAM/data/noise_20220124-201820.g2o', gt=False)#30fov
    # noise = G2O('graphSLAM/data/noise_20220124-202740.g2o', gt=False)#90fov
    # noise = G2O('graphSLAM/data/noise_20220124-204614.g2o', gt=False)#150fov

    #  ground = G2O('graphSLAM/data/ground_truth_20220120-210013.g2o', gt=True)
    # ground = G2O('graphSLAM/data/ground_truth_20220124-190221.g2o', gt=True)#120fov
    # ground = G2O('graphSLAM/data/ground_truth_20220124-193659.g2o', gt=True)#60fov
    # ground = G2O('graphSLAM/data/ground_truth_20220124-195759.g2o', gt=True)#180fov
    # ground = G2O('graphSLAM/data/ground_truth_20220124-201820.g2o', gt=True)#30fov
    # ground = G2O('graphSLAM/data/ground_truth_20220124-202740.g2o', gt=True)#90fov
    # ground = G2O('graphSLAM/data/ground_truth_20220124-204614.g2o', gt=True)#150fov


#route 2 big


#  ground = G2O('graphSLAM/data/ground_truth_20220125-204649.g2o', gt=True)#120
    # noise = G2O('graphSLAM/data/noise_20220125-204649.g2o', gt=False)#120
