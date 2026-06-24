import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Target class to represent a moving object
class Target:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity

    def update(self, dt):
        self.position += self.velocity * dt

    def get_position(self):
        return self.position
    
    def get_velocity(self):
        return self.velocity
    
    def get_speed(self):
        return np.linalg.norm(self.velocity)
    
    def get_direction(self):
        return self.velocity / self.get_speed()

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
    plt.figure(figsize=(8, 8))

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
def gen_targets(num_targets, boundary_width, boundary_height):
    target_arr = []
    for i in range(num_targets):
        rand_x_pos = np.random.uniform(-(boundary_width/2), (boundary_width/2))
        rand_y_pos = np.random.uniform(-(boundary_height/2), (boundary_height/2))
        rand_pos = np.array([rand_x_pos, rand_y_pos])

        rand_x_vel = np.random.uniform()

        new_target = Target(rand_pos, np.array([0, 0]))
        target_arr.append(new_target)
    return target_arr


## Sim loop
if __name__ == "__main__":
    # Set configuration variables
    NUM_TARGETS = 5
    BOUNDARY_WIDTH = 200
    BOUNDARY_HEIGHT = 200

    # Create set of random targets
    target_arr = gen_targets(NUM_TARGETS, BOUNDARY_WIDTH, BOUNDARY_HEIGHT)

    # Create radar at origin
    radar = Radar(np.array([0, 0]))

    # Plot targets for testing
    plot_sim(target_arr, radar)