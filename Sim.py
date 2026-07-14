import numpy as np
import matplotlib.pyplot as plt

from Functions import *

## Set configuration variables
NUM_TARGETS = 1
GUI_ON = False
BOUNDARY_WIDTH = 200
BOUNDARY_HEIGHT = 200

dt = 0.1
sim_steps = 250
num_runs = 300

# Statistics configurations
radar_position = [0, 0]
radar_sigma_range = 10
radar_sigma_azimuth = 1

camera_position = [55, 30]
camera_sigma_azimuth = 0.2

# Sensor Blackout Config
blackout_start_theta = np.deg2rad(145)
blackout_size_theta = np.deg2rad(180)

# Experiment debug
DEBUG = [0, 1, 0]

## Sim loop
if __name__ == "__main__":
    if DEBUG[0]:
        ## Control Simulation
        all_radar_sq_err = np.zeros((num_runs, sim_steps))
        all_fused_sq_err = np.zeros((num_runs, sim_steps))

        for run in range(num_runs):
            radar_errors, fused_errors = sim(NUM_TARGETS, BOUNDARY_WIDTH, BOUNDARY_HEIGHT, GUI_ON, dt, 
                                            sim_steps, radar_position, radar_sigma_range, radar_sigma_azimuth, 
                                            camera_position, camera_sigma_azimuth)
            all_radar_sq_err[run, :] = radar_errors
            all_fused_sq_err[run, :] = fused_errors

        # Compute RMSE
        radar_rmse = np.sqrt(np.mean(all_radar_sq_err, axis=0))
        fused_rmse = np.sqrt(np.mean(all_fused_sq_err, axis=0))
        
        # Plot RMSE over time
        time_axis = np.arange(sim_steps) * dt
        plt.figure(figsize=(10, 6))
        
        plt.plot(time_axis, radar_rmse, label="Local Radar Tracker (Baseline)", color="orange", linewidth=2)
        plt.plot(time_axis, fused_rmse, label="Fused Tracker (LMMSE)", color="blue", linewidth=2)
        
        plt.title(f"Position RMSE Averaged Over {num_runs} Monte Carlo Trials")
        plt.xlabel("Time (seconds)")
        plt.ylabel("RMSE (meters)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.show()

    if DEBUG[1]:
        ## Sensor Blackout Simulation
        all_radar_sq_err = np.zeros((num_runs, sim_steps))
        all_fused_sq_err = np.zeros((num_runs, sim_steps))

        for run in range(num_runs):
            radar_errors, fused_errors = sim(NUM_TARGETS, BOUNDARY_WIDTH, BOUNDARY_HEIGHT, GUI_ON, dt, 
                                            sim_steps, radar_position, radar_sigma_range, radar_sigma_azimuth, 
                                            camera_position, camera_sigma_azimuth, blackout_start_theta, blackout_size_theta)
            all_radar_sq_err[run, :] = radar_errors
            all_fused_sq_err[run, :] = fused_errors

        # Compute RMSE
        radar_rmse = np.sqrt(np.mean(all_radar_sq_err, axis=0))
        fused_rmse = np.sqrt(np.mean(all_fused_sq_err, axis=0))
        
        # Plot RMSE over time
        time_axis = np.arange(sim_steps) * dt
        plt.figure(figsize=(10, 6))
        
        plt.plot(time_axis, radar_rmse, label="Local Radar Tracker (Baseline)", color="orange", linewidth=2)
        plt.plot(time_axis, fused_rmse, label="Fused Tracker (LMMSE)", color="blue", linewidth=2)
        
        plt.title(f"Position RMSE Averaged Over {num_runs} Monte Carlo Trials w/ Camera Blackout")
        plt.xlabel("Time (seconds)")
        plt.ylabel("RMSE (meters)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.show()