import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Target class to represent a moving object
class Target:
    def __init__(self, state_vector, id):
        self.state_vector = state_vector
        self.id = id

    def update(self, dt):
        sigma = 0.5
        x_old = self.state_vector
        F = np.array([[1, dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 1]])
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        Q = np.array([
            [0.25*dt4, 0.33*dt3,        0,        0],
            [0.33*dt3,      dt2,        0,        0],
            [       0,        0, 0.25*dt4, 0.33*dt3],
            [       0,        0, 0.33*dt3,      dt2]
        ]) * (sigma**2)
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
    
# Plot function to display area with targets and radar
def plot_sim(target_arr, radar):
    plt.clf()

    radar_pos = radar.get_position()
    plt.scatter(radar_pos[0], radar_pos[1], color='blue', marker='^', s=150, label='Radar')

    target_x = [t.get_position()[0] for t in target_arr]
    target_y = [t.get_position()[1] for t in target_arr]
    plt.scatter(target_x, target_y, color='red', marker='o', s=50, label='Targets')

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
def gen_targets(num_targets, boundary_width, boundary_height, id_counter):
    target_arr = []
    for i in range(num_targets):
        rand_x_pos = np.random.uniform(-(boundary_width/2), (boundary_width/2))
        rand_y_pos = np.random.uniform(-(boundary_height/2), (boundary_height/2))

        rand_x_vel = np.random.uniform(-3, 3)
        rand_y_vel = np.random.uniform(-3, 3)

        new_state_vector = np.array([rand_x_pos, rand_x_vel, rand_y_pos, rand_y_vel]).T

        new_target = Target(new_state_vector, id_counter)
        id_counter += 1
        target_arr.append(new_target)
    return target_arr, id_counter


## Sim loop
if __name__ == "__main__":
    # Set configuration variables
    NUM_TARGETS = 5
    BOUNDARY_WIDTH = 200
    BOUNDARY_HEIGHT = 200
    GUI_ON = False

    # Create set of random targets
    ID_COUNTER = 0
    target_arr, ID_COUNTER = gen_targets(NUM_TARGETS, BOUNDARY_WIDTH, BOUNDARY_HEIGHT, ID_COUNTER)

    # Create radar at origin
    radar = Radar(np.array([0, 0]))

    if GUI_ON:
        # Init plot
        plt.ion()
        plt.figure(figsize=(8, 8))
        plot_sim(target_arr, radar)

    # Test loop
    dt = 0.1
    sim_steps = 1000
    for step in range(sim_steps):
        
        # Update all targets
        for i in range(NUM_TARGETS):
            target_arr[i].update(dt)
            curr_target_pos = target_arr[i].get_position()
            if (abs(curr_target_pos[0]) > BOUNDARY_WIDTH / 2) or (abs(curr_target_pos[1]) > BOUNDARY_HEIGHT / 2):
                target_arr[i].re_init(BOUNDARY_WIDTH, BOUNDARY_HEIGHT, ID_COUNTER)
                ID_COUNTER += 1

        if GUI_ON:
            # Update plot
            plot_sim(target_arr, radar)
            plt.pause(0.01)

    print(f"Number of targets created in sim: {ID_COUNTER}")