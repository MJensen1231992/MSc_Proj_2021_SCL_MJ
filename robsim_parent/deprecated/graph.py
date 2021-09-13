from primitives import Pose
from SLAM.landmark import Landmark

import numpy as np

class Graph:
    def __init__(self, initial_pose: Pose = Pose(0, 0, 0)):
        self.landmarks = []
        self.poses = [initial_pose]

        self.landmark_constraints = []
        self.odometry_constraints = []

    def add_odometry(self, odometry_constraint: Pose):
        new_pose = self.poses[-1]

        new_pose = odometry_constraint.absolute(new_pose) # The actual calculation. The rest are fun too, though.
        #new_pose = new_pose.absolute(odometry_constraint)
        #new_pose = odometry_constraint.relative(new_pose)
        #new_pose = new_pose.relative(odometry_constraint)

        self.poses.append(new_pose)
        self.odometry_constraints.append(odometry_constraint)

    def add_observations(self, observations: list, matching_threshold: float = 1.5):

        dists = np.array([[landmark.distance_point(observation) for observation in observations] for landmark in self.landmarks])
        #print(dists)

        matches = []
        while dists.size != 0 and dists.min() < matching_threshold:
            i = np.argmin(dists)
            
            i_landmark = i // len(observations)
            i_observation = i % len(observations)

            #print(f'Found match: ({i_observation}, {i_landmark})')

            matches.append((observations[i_observation], self.landmarks[i_landmark]))

            dists[i_landmark,:] = matching_threshold
            dists[:,i_observation] = matching_threshold

            #print(dists)

        for observation, landmark in matches:
            landmark.add_pose(observation)

        unmatched = [observation for observation in observations if observation not in [match[0] for match in matches]]

        self.landmarks = [landmark for landmark in self.landmarks if len(landmark.poses) > 1]

        for observation in unmatched:
            landmark = Landmark()
            landmark.add_pose(observation)

            self.landmarks.append(landmark)
