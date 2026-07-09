import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Target class to represent a moving object
class Target:
    def __init__(self, state_vector, id):
        self.state_vector = state_vector
        self.id = id

    def update(self, dt):
        sigma_q = 0.5
        x_old = self.state_vector
        F = np.array([[1, dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 1]])
        dt2 = dt**2
        dt3 = dt**3
        Q = np.array([
            [dt3/3,   dt2/2,    0,           0],
            [dt2/2,   dt,       0,           0],
            [0,       0,        dt3/3,   dt2/2],
            [0,       0,        dt2/2,      dt]
        ]) * (sigma_q**2)
        mean = np.zeros(4)
        w = np.random.multivariate_normal(mean, Q)
        x_new = F @ x_old + w
        self.state_vector = x_new

    def re_init(self, boundary_width, boundary_height, id):
        rand_x_pos = np.random.uniform(-(boundary_width/2), (boundary_width/2))
        rand_y_pos = np.random.uniform(-(boundary_height/2), (boundary_height/2))

        rand_x_vel = np.random.uniform(-3, 3)
        rand_y_vel = np.random.uniform(-3, 3)

        self.state_vector = np.array([rand_x_pos, rand_x_vel, rand_y_pos, rand_y_vel]).T
        self.id = id

    def get_position(self):
        return (self.state_vector[0], self.state_vector[2])
    
    def get_velocity(self):
        return (self.state_vector[1], self.state_vector[3])
    
    def get_speed(self):
        return np.linalg.norm(self.get_velocity())
    
    def get_direction(self):
        return self.get_velocity() / self.get_speed()
    
    def get_ID(self):
        return self.id

# Radar class to represent a radar system
class Radar:
    def __init__(self, position, sigma_range=10.0, sigma_azimuth_deg=1.0):
        self.position = position
        self.sigma_range = sigma_range
        self.sigma_azimuth = np.radians(sigma_azimuth_deg)

    def get_position(self):
        return self.position
    
    def get_range(self, target):
        return np.linalg.norm(target.get_position() - self.position)

    def get_azimuth(self, target):
        return np.arctan2(target.get_position()[1] - self.position[1], target.get_position()[0] - self.position[0])
    
    def get_noisy_measurements(self, target):
        true_range = self.get_range(target)
        true_azimuth = self.get_azimuth(target)
        noisy_range = true_range + np.random.normal(0, self.sigma_range)
        noisy_azimuth = true_azimuth + np.random.normal(0, self.sigma_azimuth)
        return (noisy_range, noisy_azimuth)
    
# Tracker class to handle Kalman math for radar
class RadarTracker:
    def __init__(self, target_id, radar_position, sigma_range, sigma_azimuth_rad, q_bar=0.5):
        self.target_id = target_id
        self.sensor_pos = radar_position
        self.sigma_r = sigma_range
        self.sigma_theta = sigma_azimuth_rad
        self.q_bar = q_bar
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        self.reset()

    def reset(self):
        self.x = np.array([0.0, 0.0, 0.0, 0.0])
        self.P = np.eye(4) * 100
        self.is_initialized = False

    def init_track(self, raw_measurement):
        r, theta = raw_measurement

        b1 = np.exp(-0.5 * (self.sigma_theta**2))

        x_pos = (r * np.cos(theta) / b1 + self.sensor_pos[0])
        y_pos = (r * np.sin(theta) / b1 + self.sensor_pos[1])

        self.x = np.array([x_pos, 0.0, y_pos, 0.0])
        self.P = np.diag([self.sigma_r**2, 10.0, self.sigma_r**2, 10.0])
        self.is_initialized = True

    def predict(self, dt):
        F = np.array([[1, dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 1]])
        
        dt2 = dt**2
        dt3 = dt**3

        Q = np.array([
            [dt3/3,   dt2/2,    0,           0],
            [dt2/2,   dt,       0,           0],
            [0,       0,        dt3/3,   dt2/2],
            [0,       0,        dt2/2,      dt]
        ]) * (self.q_bar**2)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, raw_measurement, id):
        if id != self.target_id:
            self.target_id = id
            self.reset()

        if not self.is_initialized:
            self.init_track(raw_measurement)
            return
        
        r, theta = raw_measurement

        b1 = np.exp(-0.5 * (self.sigma_theta**2))
        b2 = np.exp(-2 * (self.sigma_theta**2))

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        cos_2t = np.cos(2 * theta)
        sin_2t = np.sin(2 * theta)
        r2_sigma = r**2 + self.sigma_r**2

        x_c = r * cos_t / b1
        y_c = r * sin_t / b1

        Zc = np.array([
            x_c + self.sensor_pos[0],
            y_c + self.sensor_pos[1]
        ])

        # Citation: "Unbiased Converted Measurements for Tracking" 
        # Mo Longbin, Song Xiaoquan, Zhou Yiyu, Sun Zhong Kang and Y. Bar-Shalom, "Unbiased converted measurements for tracking," in IEEE Transactions on Aerospace and Electronic Systems, vol. 34, no. 3, pp. 1023-1027, July 1998, doi: 10.1109/7.705921.
        R11 = 0.5 * r2_sigma * (1 + b2 * cos_2t) - (x_c**2)
        R22 = 0.5 * r2_sigma * (1 - b2 * cos_2t) - (y_c**2)
        R12 = 0.5 * r2_sigma * b2 * sin_2t - (x_c * y_c)

        Rc = np.array([
            [R11, R12],
            [R12, R22]
        ])

        y = Zc - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + Rc
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

# Camera class to represent a EO sensor
class Camera:
    def __init__(self, position, sigma_azimuth_deg=0.2):
        self.position = position
        self.sigma_azimuth = np.radians(sigma_azimuth_deg)

    def get_position(self):
        return self.position

    def get_azimuth(self, target):
        return np.arctan2(target.get_position()[1] - self.position[1], target.get_position()[0] - self.position[0])
    
    def get_noisy_measurements(self, target):
        true_azimuth = self.get_azimuth(target)
        noisy_azimuth = true_azimuth + np.random.normal(0, self.sigma_azimuth)
        return noisy_azimuth

    
# Plot function to display area with targets and radar
# TODO: update this tracker portion, this was just added for testing
def plot_sim(target_arr, radar, radar_tracker_arr):
    plt.clf()

    radar_pos = radar.get_position()
    plt.scatter(radar_pos[0], radar_pos[1], color='blue', marker='^', s=150, label='Radar')

    target_x = [t.get_position()[0] for t in target_arr]
    target_y = [t.get_position()[1] for t in target_arr]
    plt.scatter(target_x, target_y, color='red', marker='o', s=50, label='Targets')

    tracker_x = [t.x[0] for t in radar_tracker_arr]
    tracker_y = [t.x[2] for t in radar_tracker_arr]
    plt.scatter(tracker_x, tracker_y, color='orange', marker='+', s=50, label='Tracks')

    plt.xlim(-(BOUNDARY_WIDTH/2)-10, (BOUNDARY_WIDTH/2)+10)
    plt.ylim(-(BOUNDARY_HEIGHT/2)-10, (BOUNDARY_HEIGHT/2)+10)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

    plt.title('Radar and Target Simulation Setup')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')

    plt.show()

# Generate a specified number of targets with random position and velocity
def gen_targets(num_targets, boundary_width, boundary_height, id_counter, radar):
    target_arr = []
    radar_tracker_arr = []
    for i in range(num_targets):
        rand_x_pos = np.random.uniform(-(boundary_width/2), (boundary_width/2))
        rand_y_pos = np.random.uniform(-(boundary_height/2), (boundary_height/2))

        rand_x_vel = np.random.uniform(-3, 3)
        rand_y_vel = np.random.uniform(-3, 3)

        new_state_vector = np.array([rand_x_pos, rand_x_vel, rand_y_pos, rand_y_vel]).T

        new_target = Target(new_state_vector, id_counter)
        new_tracker = RadarTracker(id_counter, radar.get_position(), radar.sigma_range, radar.sigma_azimuth)
        id_counter += 1
        target_arr.append(new_target)
        radar_tracker_arr.append(new_tracker)
    return target_arr, radar_tracker_arr, id_counter


# TODO: Kinda cheating by using target IDs with the trackers, add some kind of target detection if there is time later on.
## Sim loop
if __name__ == "__main__":
    # Set configuration variables
    NUM_TARGETS = 5
    BOUNDARY_WIDTH = 200
    BOUNDARY_HEIGHT = 200
    GUI_ON = True

    # Create radar at origin
    radar = Radar(np.array([0, 0]))

    # Create set of random targets
    ID_COUNTER = 0
    target_arr, radar_tracker_arr, ID_COUNTER = gen_targets(NUM_TARGETS, BOUNDARY_WIDTH, BOUNDARY_HEIGHT, ID_COUNTER, radar)

    if GUI_ON:
        # Init plot
        plt.ion()
        plt.figure(figsize=(8, 8))
        plot_sim(target_arr, radar, radar_tracker_arr)

    # Sim loop
    dt = 0.1
    sim_steps = 500
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
        for i in range(NUM_TARGETS):
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
                radar_tracker_arr[i].predict(dt)
                radar_tracker_arr[i].update(raw_z, target_obj.id)
            else:
                # If the target no longer exists, pick a new untracked id
                tracked_ids = set([t.target_id for t in radar_tracker_arr])
                for target in target_arr:
                    if target.id not in tracked_ids:
                        raw_z = radar.get_noisy_measurements(target)
                        radar_tracker_arr[i].update(raw_z, target.id)
                        break

        # # Calculate metrics
        # if tracker.is_initialized:
        #     true_pos = target_arr[0].get_position()
        #     est_pos = (tracker.x[0], tracker.x[2])
        #     squared_error = (true_pos[0] - est_pos[0])**2 + (true_pos[1] - est_pos[1])**2
        #     mse_history.append(squared_error)
        #     steps_history.append(step)

        if GUI_ON:
            # Update plot
            plot_sim(target_arr, radar, radar_tracker_arr)
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