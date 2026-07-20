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
DEBUG = [1, 1]

# NEES degrees of freedom
NEES_DOF = 2


def run_monte_carlo(blackout_start=0.0, blackout_size=0.0):
    """Run num_runs Monte Carlo trials and collect per-step error/uncertainty arrays."""
    all_radar_sq_err = np.zeros((num_runs, sim_steps))
    all_fused_sq_err = np.zeros((num_runs, sim_steps))
    all_radar_trace = np.zeros((num_runs, sim_steps))
    all_fused_trace = np.zeros((num_runs, sim_steps))
    all_radar_nees = np.zeros((num_runs, sim_steps))
    all_fused_nees = np.zeros((num_runs, sim_steps))
 
    for run in range(num_runs):
        radar_errors, fused_errors, radar_trace, fused_trace, radar_nees, fused_nees = sim(
            NUM_TARGETS, BOUNDARY_WIDTH, BOUNDARY_HEIGHT, GUI_ON, dt,
            sim_steps, radar_position, radar_sigma_range, radar_sigma_azimuth,
            camera_position, camera_sigma_azimuth, blackout_start, blackout_size)
        all_radar_sq_err[run, :] = radar_errors
        all_fused_sq_err[run, :] = fused_errors
        all_radar_trace[run, :] = radar_trace
        all_fused_trace[run, :] = fused_trace
        all_radar_nees[run, :] = radar_nees
        all_fused_nees[run, :] = fused_nees
 
    return {
        "radar_rmse": np.sqrt(np.mean(all_radar_sq_err, axis=0)),
        "fused_rmse": np.sqrt(np.mean(all_fused_sq_err, axis=0)),
        "radar_predicted_rmse": np.sqrt(np.mean(all_radar_trace, axis=0)),
        "fused_predicted_rmse": np.sqrt(np.mean(all_fused_trace, axis=0)),
        "radar_nees": np.mean(all_radar_nees, axis=0),
        "fused_nees": np.mean(all_fused_nees, axis=0),
    }


def plot_rmse(results, title):
    time_axis = np.arange(sim_steps) * dt
    plt.figure(figsize=(10, 6))
 
    plt.plot(time_axis, results["radar_rmse"], label="Local Radar Tracker (Baseline)", color="orange", linewidth=2)
    plt.plot(time_axis, results["fused_rmse"], label="Fused Tracker (LMMSE)", color="blue", linewidth=2)
 
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("RMSE (meters)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()
 
 
def plot_rmse_with_uncertainty(results, title):
    """Actual RMSE vs. the RMSE the filter *predicts* it should have (sqrt(trace(P_pos))).
    When the dotted (predicted) line tracks the solid (actual) line, the filter's
    reported covariance is a trustworthy indicator of its real-world error."""
    time_axis = np.arange(sim_steps) * dt
    plt.figure(figsize=(10, 6))
 
    plt.plot(time_axis, results["radar_rmse"], label="Radar (Actual RMSE)", color="orange", linewidth=2)
    plt.plot(time_axis, results["radar_predicted_rmse"], label="Radar (Predicted, sqrt(tr P))",
             color="orange", linestyle=":", linewidth=1.5)
    plt.plot(time_axis, results["fused_rmse"], label="Fused (Actual RMSE)", color="blue", linewidth=2)
    plt.plot(time_axis, results["fused_predicted_rmse"], label="Fused (Predicted, sqrt(tr P))",
             color="blue", linestyle=":", linewidth=1.5)
 
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Position RMSE (meters)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()
 
 
def plot_nees(results, title):
    """Filter consistency check. If avg NEES sits within the bounds, the filter's
    covariance P is neither over- nor under-confident relative to its true error."""
    time_axis = np.arange(sim_steps) * dt
 
    # Normal approximation to the mean of `num_runs` iid chi-square(NEES_DOF)
    # samples: mean ~= N(NEES_DOF, 2*NEES_DOF/num_runs) for large num_runs.
    nees_std = np.sqrt(2 * NEES_DOF / num_runs)
    lower_bound = NEES_DOF - 1.96 * nees_std
    upper_bound = NEES_DOF + 1.96 * nees_std
 
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, results["radar_nees"], label="Radar NEES", color="orange", linewidth=2)
    plt.plot(time_axis, results["fused_nees"], label="Fused NEES", color="blue", linewidth=2)
    plt.axhline(NEES_DOF, color="black", linestyle="--", linewidth=1, label=f"Ideal (dof={NEES_DOF})")
    plt.axhspan(lower_bound, upper_bound, color="gray", alpha=0.2, label="95% consistency bounds")
 
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Average NEES")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()


## Sim loop
if __name__ == "__main__":
    if DEBUG[0]:
        ## Control Simulation
        results = run_monte_carlo()
 
        plot_rmse(results, f"Position RMSE Averaged Over {num_runs} Monte Carlo Trials")
        plot_rmse_with_uncertainty(results, f"Actual vs. Predicted RMSE Over {num_runs} Monte Carlo Trials")
        plot_nees(results, f"Filter Consistency (NEES) Averaged Over {num_runs} Monte Carlo Trials")

    if DEBUG[1]:
        ## Sensor Blackout Simulation
        results = run_monte_carlo(blackout_start_theta, blackout_size_theta)
 
        plot_rmse(results, f"Position RMSE Averaged Over {num_runs} Monte Carlo Trials w/ Camera Blackout")
        plot_rmse_with_uncertainty(results, f"Actual vs. Predicted RMSE Over {num_runs} Monte Carlo Trials w/ Camera Blackout")
        plot_nees(results, f"Filter Consistency (NEES) Averaged Over {num_runs} Monte Carlo Trials w/ Camera Blackout")