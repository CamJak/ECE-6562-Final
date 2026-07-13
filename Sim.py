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

        # Fuse tracks
        fused_tracks = fuse_tracks(NUM_TARGETS, radar_tracker_arr, camera_tracker_arr, camera, P_re_dict, FC)

        if GUI_ON:
            # Update plot
            plot_sim(BOUNDARY_WIDTH, BOUNDARY_HEIGHT, 
                     target_arr, radar, radar_tracker_arr, 
                     camera, camera_tracker_arr, fused_tracks)
            plt.pause(0.01)

    if GUI_ON:
        plt.ioff()
        plt.close()