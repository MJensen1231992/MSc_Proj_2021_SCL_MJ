import numpy as np
from utility import Graph
import os
import primitives
import inspect

# print(elems[0:])
#print(os.getcwd())
file = 'GraphSlam/data/sim_pose_landmark_data.g2o'

read_file = Graph.read_G2O(file)


poses, landmarks = [], []

for id in read_file.nodes:
            
    if type(read_file.nodes[id]) == primitives.Pose:
        poses.append(read_file.nodes[id])
    
    elif type(read_file.nodes[id]) == primitives.Point:
        landmarks.append(read_file.nodes[id])

print(poses[5])
print(landmarks[5])
inspect.getmembers(primitives, lambda a:not(inspect.isroutine(Pose)))

# tp = type(read_file.nodes[1])



# if tp == primitives.Point:
#     print('its a point!')
# else:
#     print('Its not a point')