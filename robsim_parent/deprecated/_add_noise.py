import numpy as np

def scan_noise(scan, robot_pose, angle_std, distance_std):
    noisy_scan = []
    for i, point in enumerate(scan):
        dist = np.sqrt((robot_pose[0] - point[0])**2 + (robot_pose[1] - point[1])**2)
        
        noise_dist = np.random.normal(0, distance_std * dist)
        noise_theta = np.random.normal(0, angle_std)

        dist = dist + noise_dist
        theta = np.arctan((robot_pose[1] - point[1]) / (robot_pose[0] - point[0])) + noise_theta

        scan_x = dist * np.cos(theta) - robot_pose[0]
        scan_y = dist * np.sin(theta) - robot_pose[1]

        noisy_scan.append([scan_x, scan_y])
    return noisy_scan

def add_noise(this_measurement, last_noisy_measurement, x_std: float = 0, y_std: float = 0, theta_std: float = 0, angle_std: float = 0, distance_std: float = 0):
    noisy_measurements = {}

    odometry_noise = [np.random.normal(0, std) for std in [x_std, y_std, theta_std]]

    noisy_measurements['scan'] = scan_noise(this_measurement['scan'], this_measurement['origin'], angle_std, distance_std)

    noisy_measurements['odometry'] = [odometry + noise for odometry, noise in zip(this_measurement['odometry'], odometry_noise)]

    noisy_measurements['origin'] = [x+dx for x, dx in zip(last_noisy_measurement['origin'], noisy_measurements['odometry'])]

    return noisy_measurements

if __name__ == '__main__':
    import json

    x_std = 0.01
    y_std = 0.01
    theta_std = 0.001
    angle_std = 0.001
    distance_std = 0.001

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
            path[i]['noisy'] = add_noise(path[i]['groundtruth'], path[i-1]['noisy'], x_std=x_std, y_std=y_std, theta_std=theta_std, angle_std=angle_std, distance_std=distance_std)

    with open('noisy_lidar_data.json', 'w') as f:
        json.dump(path, f, indent=4)
