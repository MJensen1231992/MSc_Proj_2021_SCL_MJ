import json
import numpy as np

odometry_dist_std = 0.01
odometry_angle_std = 0.01
theta_std = 0.001
angle_std = 0.001
distance_std = 0.001

def scan_noise(scan, robot_pose, angle_std, distance_std):
    noisy_scan = []
    for i, point in enumerate(scan):
        dist = np.sqrt((robot_pose[0] - point[0])**2 + (robot_pose[1] - point[1])**2)
        
        noise_dist = np.random.normal(0, distance_std * dist)
        noise_angle = np.random.normal(0, angle_std)

        dist = dist + noise_dist
        angle = np.arctan((robot_pose[1] - point[1]) / (robot_pose[0] - point[0])) + noise_angle

        scan_x = dist * np.cos(angle) - robot_pose[0]
        scan_y = dist * np.sin(angle) - robot_pose[1]

        noisy_scan.append([scan_x, scan_y])
    return noisy_scan

def odometry_noise(odometry, last_pose, odometry_dist_std, odometry_angle_std, theta_std):
    dist = np.sqrt((last_pose[0] - odometry[0])**2 + (last_pose[1] - odometry[1])**2)

    noise_dist = np.random.normal(0, odometry_dist_std * dist)
    noise_angle = np.random.normal(0, odometry_angle_std)

    dist = dist + noise_dist
    angle = np.arctan((last_pose[1] - odometry[1]) / (last_pose[0] - odometry[0])) + noise_angle

    x = dist * np.cos(angle) - last_pose[0]
    y = dist * np.sin(angle) - last_pose[1]
    theta = odometry[2] + np.random.normal(0, theta_std)

    return [x, y, theta]

def add_noise(this_measurement, last_noisy_measurement, odometry_dist_std: float = 0, odometry_angle_std: float = 0, theta_std: float = 0, angle_std: float = 0, distance_std: float = 0):
    noisy_measurements = {}

    noisy_measurements['scan'] = scan_noise(this_measurement['scan'], this_measurement['origin'], angle_std, distance_std)

    noisy_measurements['odometry'] = odometry_noise(this_measurement['odometry'], last_noisy_measurement['origin'], odometry_dist_std, odometry_angle_std, theta_std)#[odometry + noise for odometry, noise in zip(this_measurement['odometry'], odometry_noise)]

    noisy_measurements['origin'] = [x+dx for x, dx in zip(last_noisy_measurement['origin'], noisy_measurements['odometry'])]

    return noisy_measurements

with open('lidar_data.json') as f:
    file = json.load(f)

path = [{'groundtruth': val} for val in file]
for i in range(len(path[1:])):
    path[i + 1]['groundtruth']['odometry'] = [path[i + 1]['groundtruth']['origin'][j] - path[i]['groundtruth']['origin'][j] for j in range(3)]

for i, ms in enumerate(path):
    path[i]['noisy'] = {}
    if i == 0:
        path[i]['noisy']['origin'] = ms['groundtruth']['origin']
        path[i]['noisy']['scan'] = scan_noise(ms['groundtruth']['scan'], ms['groundtruth']['origin'], angle_std, distance_std)
    else:
        path[i]['noisy'] = add_noise(path[i]['groundtruth'], path[i-1]['noisy'], odometry_dist_std=odometry_dist_std, odometry_angle_std=odometry_angle_std, theta_std=theta_std, angle_std=angle_std, distance_std=distance_std)

with open('noisy_lidar_data.json', 'w') as f:
    json.dump(path, f, indent=4)