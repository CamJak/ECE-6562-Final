import numpy as np
import matplotlib.pyplot as plt

from Classes.Target import *
from Classes.Radar import *
from Classes.Camera import *
from Classes.FusionManager import *
from Functions import *

## Sim loop
if __name__ == "__main__":
    ## Set configuration variables
    NUM_TARGETS = 5
    BOUNDARY_WIDTH = 200
    BOUNDARY_HEIGHT = 200
    GUI_ON = True

    dt = 0.1
    sim_steps = 500

    # Statistics configurations
    radar_position = [0, 0]
    radar_sigma_range = 10
    radar_sigma_azimuth = 1

    camera_position = [55, 30]
    camera_sigma_azimuth = 0.2

    # Create radar and camera at origin
    radar = Radar(np.array(radar_position), radar_sigma_range, radar_sigma_azimuth)
    camera = Camera(np.array(camera_position), camera_sigma_azimuth)

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

    ## Sim loop
    for step in range(sim_steps):
        
        # Update all targets
        for i in range(NUM_TARGETS):
            # Update current target's position
            target_arr[i].update(dt)
            curr_target_pos = target_arr[i].get_position()

            # If target is outside the boundary, kill it and birth a new target
            if (abs(curr_target_pos[0]) > BOUNDARY_WIDTH / 2) or (abs(curr_target_pos[1]) > BOUNDARY_HEIGHT / 2):
                target_arr[i].re_init(BOUNDARY_WIDTH, BOUNDARY_HEIGHT, ID_COUNTER)
                ID_COUNTER += 1

        # Update all trackers
        process_radar_tracks(NUM_TARGETS, target_arr, radar_tracker_arr, radar)
        process_camera_tracks(NUM_TARGETS, target_arr, camera_tracker_arr, camera)

        active_P_re = {}
        fused_tracks = []

        # Fuse tracks
        for i in range(NUM_TARGETS):
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
                fused_tracks.append(x_f)

        if GUI_ON:
            # Update plot
            plot_sim(BOUNDARY_WIDTH, BOUNDARY_HEIGHT, 
                     target_arr, radar, radar_tracker_arr, 
                     camera, camera_tracker_arr, fused_tracks)
            plt.pause(0.01)

    if GUI_ON:
        plt.ioff()
        plt.close()