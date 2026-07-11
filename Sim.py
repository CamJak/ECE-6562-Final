import numpy as np
import matplotlib.pyplot as plt

from Classes.Target import *
from Classes.Radar import *
from Classes.Camera import *
from Functions import *

## Sim loop
if __name__ == "__main__":
    # Set configuration variables
    NUM_TARGETS = 5
    BOUNDARY_WIDTH = 200
    BOUNDARY_HEIGHT = 200
    GUI_ON = True

    # Sim config
    dt = 0.1
    sim_steps = 500

    # Create radar and camera at origin
    radar = Radar(np.array([0, 0]))
    camera = Camera(np.array([0, 0]))

    # Create set of random targets
    ID_COUNTER = 0
    target_arr, radar_tracker_arr, camera_tracker_arr, ID_COUNTER = gen_targets(NUM_TARGETS, BOUNDARY_WIDTH, BOUNDARY_HEIGHT, ID_COUNTER, radar, camera, dt)

    if GUI_ON:
        # Init plot
        plt.ion()
        plt.figure(figsize=(8, 8))
        plot_sim(BOUNDARY_WIDTH, BOUNDARY_HEIGHT, target_arr, radar, radar_tracker_arr, camera, camera_tracker_arr)

    # Sim loop
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

        # # Calculate metrics
        # if tracker.is_initialized:
        #     true_pos = target_arr[0].get_position()
        #     est_pos = (tracker.x[0], tracker.x[2])
        #     squared_error = (true_pos[0] - est_pos[0])**2 + (true_pos[1] - est_pos[1])**2
        #     mse_history.append(squared_error)
        #     steps_history.append(step)

        if GUI_ON:
            # Update plot
            plot_sim(BOUNDARY_WIDTH, BOUNDARY_HEIGHT, target_arr, radar, radar_tracker_arr, camera, camera_tracker_arr)
            plt.pause(0.01)

    if GUI_ON:
        plt.ioff()
        plt.close()

    # print(f"Number of targets created in sim: {ID_COUNTER}")
    # print(f"Average Tracking Position MSE: {np.mean(mse_history):.4f} m^2")

    # plt.figure(figsize=(10, 5))
    # plt.plot(steps_history, mse_history, color='purple', linewidth=1.5, label='Squared Position Error')
    # plt.axhline(np.mean(mse_history), color='red', linestyle='--', linewidth=1.5, label=f'Mean SE ({np.mean(mse_history):.2f})')

    # for s_idx, b_step in enumerate(birth_steps):
    #     # We only add a label to the first line so it doesn't clutter the legend
    #     lbl = "New Target Created" if s_idx == 0 else ""
    #     plt.axvline(x=b_step, color='gray', linestyle=':', linewidth=1.2, alpha=0.8, label=lbl)

    # plt.title('Tracker Position Error Across Simulation Steps')
    # plt.xlabel('Simulation Step')
    # plt.ylabel('Squared Error ($m^2$)')
    # plt.grid(True, linestyle=':', alpha=0.6)
    # plt.legend()
    # plt.show()