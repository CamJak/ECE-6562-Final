import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from Classes.Target import *
from Classes.Radar import *
from Classes.Camera import *
from Classes.FusionManager import *

# Plot function to display area with targets and radar
def plot_sim(boundary_width, boundary_height, target_arr, 
             radar, radar_tracker_arr, camera, camera_tracker_arr, fused_tracks):
    plt.clf()
    ax = plt.gca()

    radar_pos = radar.get_position()
    plt.scatter(radar_pos[0], radar_pos[1], color='blue', marker='^', s=150, label='Radar')

    cam_pos = camera.get_position()

    # Shade the camera's blackout (glare) sector, if configured
    max_line_length = max(boundary_width, boundary_height)
    if camera.blackout_size > 0:
        start_deg = np.degrees(camera.blackout_start)
        end_deg = np.degrees(camera.blackout_start + camera.blackout_size)
        blackout_wedge = Wedge((cam_pos[0], cam_pos[1]), max_line_length, start_deg, end_deg,
                                facecolor='gray', edgecolor='none', alpha=0.25,
                                label='Camera Blackout Zone', zorder=0)
        ax.add_patch(blackout_wedge)

    plt.scatter(cam_pos[0], cam_pos[1], color='purple', marker='s', s=20, label='Camera (IR/EO)')

    target_x = [t.get_position()[0] for t in target_arr]
    target_y = [t.get_position()[1] for t in target_arr]
    plt.scatter(target_x, target_y, color='red', marker='o', s=50, label='Targets')

    tracker_x = [t.x[0] for t in radar_tracker_arr]
    tracker_y = [t.x[2] for t in radar_tracker_arr]
    plt.scatter(tracker_x, tracker_y, color='orange', marker='+', s=50, label='Radar Tracks')

    # Plot track lines for EO tracks
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
            
    fused_x = [t[1][0] for t in fused_tracks]
    fused_y = [t[1][2] for t in fused_tracks]
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

# Compute squared position error, predicted position variance (trace of the
# x,y sub-block of P), and NEES (Normalized Estimation Error Squared) for a
# track. NEES checks filter *consistency* -- whether the reported uncertainty
# P actually matches the real error, not just its magnitude. For a 2D
# position measurement, a well-tuned/consistent filter has E[NEES] ~= 2.
def _position_metrics(true_pos, est_x, est_y, P_full):
    dx = true_pos[0] - est_x
    dy = true_pos[1] - est_y
    sq_err = dx**2 + dy**2

    P_pos = P_full[np.ix_([0, 2], [0, 2])]
    trace_val = np.trace(P_pos)

    e = np.array([dx, dy])
    try:
        nees_val = e @ np.linalg.inv(P_pos) @ e
    except np.linalg.LinAlgError:
        nees_val = np.nan

    return sq_err, trace_val, nees_val

# Function to process the updating of radar tracks
def process_radar_tracks(num_targets, target_arr, radar_tracker_arr, radar):
    radar_step_errors = {}
    radar_step_trace = {}
    radar_step_nees = {}

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
                    target_obj = target
                    break
        
        # Calculate error and uncertainty metrics
        tracker = radar_tracker_arr[i]
        if tracker.is_initialized and target_obj is not None:
            true_pos = target_obj.get_position()
            sq_err, trace_val, nees_val = _position_metrics(true_pos, tracker.x[0], tracker.x[2], tracker.P)
            radar_step_errors[tracker.target_id] = sq_err
            radar_step_trace[tracker.target_id] = trace_val
            radar_step_nees[tracker.target_id] = nees_val

    return radar_step_errors, radar_step_trace, radar_step_nees
            

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

# Function to fuse given set of tracks using HT2TF
def fuse_tracks(num_targets, radar_tracker_arr, camera_tracker_arr, camera, P_re_dict, FC, target_arr):
    active_P_re = {}
    fused_tracks = []
    fused_step_errors = {}
    fused_step_trace = {}
    fused_step_nees = {}

    for i in range(num_targets):
        curr_target_id = radar_tracker_arr[i].target_id
        radar_track = radar_tracker_arr[i]
        camera_track = None

        for track in camera_tracker_arr:
            if track.target_id == curr_target_id:
                camera_track = track
                break

        if curr_target_id not in P_re_dict:
            P_re_dict[curr_target_id] = np.zeros((4, 2))

        P_re_old = P_re_dict[curr_target_id]

        A_k = compute_jacobian(radar_track.x_pred, camera.get_position())

        # Equation 46 [1]
        Q_re = radar_track.Q @ A_k.T

        # Equation 53 [1]
        P_re_pred = radar_track.F @ P_re_old @ camera_track.F.T + Q_re
        I_r = np.eye(4)
        I_e = np.eye(2)
        term_r = I_r - radar_track.K @ radar_track.H
        term_e = I_e - camera_track.K @ camera_track.H
        P_re_updated = term_r @ P_re_pred @ term_e.T

        active_P_re[curr_target_id] = P_re_updated

        x_f, P_f = FC.fuse(radar_track, camera_track, P_re_updated)

        if x_f is not None:
            fused_tracks.append((curr_target_id, x_f))

            # Calculate error and uncertainty metrics
            matching_target = next((t for t in target_arr if t.id == curr_target_id), None)
            if matching_target is not None:
                true_pos = matching_target.get_position()
                sq_err, trace_val, nees_val = _position_metrics(true_pos, x_f[0], x_f[2], P_f)
                fused_step_errors[curr_target_id] = sq_err
                fused_step_trace[curr_target_id] = trace_val
                fused_step_nees[curr_target_id] = nees_val

    return fused_tracks, fused_step_errors, fused_step_trace, fused_step_nees

# Function that runs single iteration of the basic simulation
def sim(NUM_TARGETS, BOUNDARY_WIDTH, BOUNDARY_HEIGHT, GUI_ON, 
        dt, sim_steps, radar_position, radar_sigma_range, radar_sigma_azimuth, 
        camera_position, camera_sigma_azimuth, blackout_start=0.0, blackout_size=0.0):
    
    # Create radar and camera at origin
    radar = Radar(np.array(radar_position), radar_sigma_range, radar_sigma_azimuth)
    camera = Camera(np.array(camera_position), camera_sigma_azimuth, blackout_start, blackout_size)

    # Create fusion manager and cross-covariance
    FC = FusionManager(camera.get_position())
    P_re_dict = {}

    # Create set of random targets
    ID_COUNTER = 0
    target_arr, radar_tracker_arr, camera_tracker_arr, ID_COUNTER = gen_targets(NUM_TARGETS, BOUNDARY_WIDTH, BOUNDARY_HEIGHT, ID_COUNTER, radar, camera, dt)

    if GUI_ON:
        # Init plot
        plt.ion()
        plt.figure(figsize=(8, 8))
        plot_sim(BOUNDARY_WIDTH, BOUNDARY_HEIGHT, target_arr, 
                 radar, radar_tracker_arr, camera, camera_tracker_arr, [])

    # Create empty error/uncertainty arrays
    radar_errors = np.zeros(sim_steps)
    fused_errors = np.zeros(sim_steps)
    radar_trace = np.zeros(sim_steps)
    fused_trace = np.zeros(sim_steps)
    radar_nees = np.zeros(sim_steps)
    fused_nees = np.zeros(sim_steps)

    ## Sim loop
    for step in range(sim_steps):
        
        # Update all targets
        for i in range(NUM_TARGETS):
            # Update current target's position
            target_arr[i].update(dt)
            curr_target_pos = target_arr[i].get_position()

            # If target is outside the boundary, kill it and birth a new target
            # Only do this for displaying purposes, no point for calculations.
            if GUI_ON:
                if (abs(curr_target_pos[0]) > BOUNDARY_WIDTH / 2) or (abs(curr_target_pos[1]) > BOUNDARY_HEIGHT / 2):
                    target_arr[i].re_init(BOUNDARY_WIDTH, BOUNDARY_HEIGHT, ID_COUNTER)
                    ID_COUNTER += 1

        # Update all trackers
        radar_step_errors, radar_step_trace, radar_step_nees = process_radar_tracks(NUM_TARGETS, target_arr, radar_tracker_arr, radar)
        process_camera_tracks(NUM_TARGETS, target_arr, camera_tracker_arr, camera)

        # Fuse tracks
        fused_tracks, fused_step_errors, fused_step_trace, fused_step_nees = fuse_tracks(NUM_TARGETS, radar_tracker_arr, camera_tracker_arr, 
                                                      camera, P_re_dict, FC, target_arr)

        # Compile and average errors
        step_radar_list = []
        step_fused_list = []
        step_radar_trace_list = []
        step_fused_trace_list = []
        step_radar_nees_list = []
        step_fused_nees_list = []
        
        for t_id, r_err in radar_step_errors.items():
            step_radar_list.append(r_err)
            step_radar_trace_list.append(radar_step_trace[t_id])
            step_radar_nees_list.append(radar_step_nees[t_id])
            if t_id in fused_step_errors:
                step_fused_list.append(fused_step_errors[t_id])
                step_fused_trace_list.append(fused_step_trace[t_id])
                step_fused_nees_list.append(fused_step_nees[t_id])
            else:
                # No fused estimate yet (e.g. camera track not initialized) -- fall back to radar-only
                step_fused_list.append(r_err)
                step_fused_trace_list.append(radar_step_trace[t_id])
                step_fused_nees_list.append(radar_step_nees[t_id])

        if len(step_radar_list) > 0:
            radar_errors[step] = np.mean(step_radar_list)
            fused_errors[step] = np.mean(step_fused_list)
            radar_trace[step] = np.mean(step_radar_trace_list)
            fused_trace[step] = np.mean(step_fused_trace_list)
            radar_nees[step] = np.mean(step_radar_nees_list)
            fused_nees[step] = np.mean(step_fused_nees_list)

        if GUI_ON:
            # Update plot
            plot_sim(BOUNDARY_WIDTH, BOUNDARY_HEIGHT, 
                     target_arr, radar, radar_tracker_arr, 
                     camera, camera_tracker_arr, fused_tracks)
            plt.pause(0.001)

    if GUI_ON:
        plt.ioff()
        plt.close()

    return radar_errors, fused_errors, radar_trace, fused_trace, radar_nees, fused_nees