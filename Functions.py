import numpy as np
import matplotlib.pyplot as plt

from Classes.Target import *
from Classes.Radar import *
from Classes.Camera import *

# Plot function to display area with targets and radar
def plot_sim(boundary_width, boundary_height, target_arr, 
             radar, radar_tracker_arr, camera, camera_tracker_arr, fused_tracks):
    plt.clf()

    radar_pos = radar.get_position()
    plt.scatter(radar_pos[0], radar_pos[1], color='blue', marker='^', s=150, label='Radar')

    cam_pos = camera.get_position()
    plt.scatter(cam_pos[0], cam_pos[1], color='purple', marker='s', s=20, label='Camera (IR/EO)')

    target_x = [t.get_position()[0] for t in target_arr]
    target_y = [t.get_position()[1] for t in target_arr]
    plt.scatter(target_x, target_y, color='red', marker='o', s=50, label='Targets')

    tracker_x = [t.x[0] for t in radar_tracker_arr]
    tracker_y = [t.x[2] for t in radar_tracker_arr]
    plt.scatter(tracker_x, tracker_y, color='orange', marker='+', s=50, label='Radar Tracks')

    # Plot track lines for EO tracks
    max_line_length = max(boundary_width, boundary_height)
    for c_tracker in camera_tracker_arr:
        if c_tracker.is_initialized:
            # The current tracked angle theta is the first element of the state vector
            theta_est = c_tracker.x[0]
            
            # Calculate the end point of the line originating from the camera
            x_end = cam_pos[0] + max_line_length * np.cos(theta_est)
            y_end = cam_pos[1] + max_line_length * np.sin(theta_est)
            
            # Plot the line from camera position to the projected end point
            plt.plot([cam_pos[0], x_end], [cam_pos[1], y_end], 
                     color='purple', linestyle='--', linewidth=1.5, alpha=0.7,
                     label='Camera LOS' if 'Camera LOS' not in plt.gca().get_legend_handles_labels()[1] else "")
            
    fused_x = [t[0] for t in fused_tracks]
    fused_y = [t[2] for t in fused_tracks]
    plt.scatter(fused_x, fused_y, color='green', marker='+', s=50, label='Fused Tracks')

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
        new_rad_tracker = RadarTracker(id_counter, radar.get_position(), radar.sigma_range, radar.sigma_azimuth, dt)
        new_cam_tracker = CameraTracker(id_counter, dt, camera.sigma_azimuth)

        id_counter += 1

        target_arr.append(new_target)
        radar_tracker_arr.append(new_rad_tracker)
        camera_tracker_arr.append(new_cam_tracker)
    return target_arr, radar_tracker_arr, camera_tracker_arr, id_counter

# Function for computing the jacobian matrix
def compute_jacobian(x_pred, cam_pos):
    dx = x_pred[0] - cam_pos[0]
    dy = x_pred[2] - cam_pos[1]
    vx = x_pred[1]
    vy = x_pred[3]

    v = np.sqrt(vx**2 + vy**2) + 1e-6
    re = np.sqrt(dx**2 + dy**2) + 1e-6
    phi = np.arctan2(vy, vx) - np.arctan2(dy, dx)

    G = np.zeros((2, 4))

    # Equations 37 - 44 [1]
    G[0, 0] = -dy / (re**2)
    G[0, 1] = 0.0 
    G[0, 2] = dx / (re**2)
    G[0, 3] = 0.0
    G[1, 0] = (v / (re**3)) * (-np.sin(phi) * dx + np.cos(phi) * dy)
    G[1, 1] = (1.0 / (v * re)) * (np.sin(phi) * vx - np.cos(phi) * vy)
    G[1, 2] = (v / (re**3)) * (-np.sin(phi) * dx - np.cos(phi) * dy)
    G[1, 3] = (1.0 / (v * re)) * (np.sin(phi) * vy + np.cos(phi) * vx)

    return G

# Function to process the updating of radar tracks
def process_radar_tracks(num_targets, target_arr, radar_tracker_arr, radar):
    for i in range(num_targets):
            # Check if target still exists
            target_exists = False
            target_obj = None

            for target in target_arr:
                if radar_tracker_arr[i].target_id == target.id:
                    target_exists = True
                    target_obj = target
                    break

            if target_exists:
                # If it still exists, predict and update
                raw_z = radar.get_noisy_measurements(target_obj)
                radar_tracker_arr[i].predict()
                radar_tracker_arr[i].update(raw_z, target_obj.id)
            else:
                # If the target no longer exists, pick a new untracked id
                tracked_ids = set([t.target_id for t in radar_tracker_arr])
                for target in target_arr:
                    if target.id not in tracked_ids:
                        raw_z = radar.get_noisy_measurements(target)
                        radar_tracker_arr[i].update(raw_z, target.id)
                        break

# Function for handling update of camera tracks
def process_camera_tracks(num_targets, target_arr, camera_tracker_arr, camera):
    for i in range(num_targets):
        # Check if target still exists
            target_exists = False
            target_obj = None

            for target in target_arr:
                if camera_tracker_arr[i].target_id == target.id:
                    target_exists = True
                    target_obj = target
                    break

            if target_exists:
                # If it still exists, predict and update
                raw_z = camera.get_noisy_measurements(target_obj)
                camera_tracker_arr[i].predict()
                camera_tracker_arr[i].update(raw_z, target_obj.id)
            else:
                # If the target no longer exists, pick a new untracked id
                tracked_ids = set([t.target_id for t in camera_tracker_arr])
                for target in target_arr:
                    if target.id not in tracked_ids:
                        raw_z = camera.get_noisy_measurements(target)
                        camera_tracker_arr[i].update(raw_z, target.id)
                        break