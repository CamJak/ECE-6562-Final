import numpy as np
import matplotlib.pyplot as plt

from Classes.Target import *
from Classes.Radar import *
from Classes.Camera import *

# Plot function to display area with targets and radar
def plot_sim(boundary_width, boundary_height, target_arr, radar, radar_tracker_arr):
    plt.clf()

    radar_pos = radar.get_position()
    plt.scatter(radar_pos[0], radar_pos[1], color='blue', marker='^', s=150, label='Radar')

    target_x = [t.get_position()[0] for t in target_arr]
    target_y = [t.get_position()[1] for t in target_arr]
    plt.scatter(target_x, target_y, color='red', marker='o', s=50, label='Targets')

    tracker_x = [t.x[0] for t in radar_tracker_arr]
    tracker_y = [t.x[2] for t in radar_tracker_arr]
    plt.scatter(tracker_x, tracker_y, color='orange', marker='+', s=50, label='Tracks')

    plt.xlim(-(boundary_width/2)-10, (boundary_width/2)+10)
    plt.ylim(-(boundary_height/2)-10, (boundary_height/2)+10)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

    plt.title('Radar and Target Simulation Setup')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')

    plt.show()

# Generate a specified number of targets with random position and velocity
def gen_targets(num_targets, boundary_width, boundary_height, id_counter, radar, camera, dt):
    target_arr = []
    radar_tracker_arr = []
    camera_tracker_arr = []
    for i in range(num_targets):
        rand_x_pos = np.random.uniform(-(boundary_width/2), (boundary_width/2))
        rand_y_pos = np.random.uniform(-(boundary_height/2), (boundary_height/2))

        rand_x_vel = np.random.uniform(-3, 3)
        rand_y_vel = np.random.uniform(-3, 3)

        new_state_vector = np.array([rand_x_pos, rand_x_vel, rand_y_pos, rand_y_vel]).T

        new_target = Target(new_state_vector, id_counter)
        new_rad_tracker = RadarTracker(id_counter, radar.get_position(), radar.sigma_range, radar.sigma_azimuth)
        new_cam_tracker = CameraTracker(id_counter, dt, camera.sigma_azimuth)

        id_counter += 1

        target_arr.append(new_target)
        radar_tracker_arr.append(new_rad_tracker)
    return target_arr, radar_tracker_arr, id_counter