import numpy as np

# Camera class to represent a EO sensor
# TODO: Work on this, this is just temp code
class Camera:
    def __init__(self, position, sigma_azimuth_deg=0.2):
        self.position = np.array(position)
        self.sigma_azimuth = np.radians(sigma_azimuth_deg)

    def get_position(self):
        return self.position

    def get_azimuth(self, target):
        # Citation: "Track-to-Track fusion with cross-covariances from radar and IR/EO sensor" [1]
        # K. Yang, Y. Bar-Shalom and P. Willett, "Track-to-Track fusion with cross-covariances from radar and IR/EO sensor," 2019 22th International Conference on Information Fusion (FUSION), Ottawa, ON, Canada, 2019, pp. 1-5, doi: 10.23919/FUSION43075.2019.9011439.
        target_pos = target.get_position()
        dx = target_pos[0] - self.position[0]
        dy = target_pos[1] - self.position[1]
        return np.arctan2(dy, dx)
    
    def get_noisy_measurements(self, target):
        true_azimuth = self.get_azimuth(target)
        return true_azimuth + np.random.normal(0, self.sigma_azimuth)
    
# Tracker class to handle Kalman math for camera
# TODO: Work on this, this is just temp code (why does camera tracker not need pos?)
class CameraTracker:
    def __init__(self, target_id, dt, sigma_azimuth_rad, q_noise=0.01):
        self.target_id = target_id
        self.dt = dt
        self.sigma_theta = sigma_azimuth_rad
        self.q = q_noise
        
        # Citation [1]
        self.F = np.array([[1, self.dt],
                           [0, 1]])
        
        self.H = np.array([[1, 0]])
        
        dt2 = self.dt**2
        self.Q = np.array([[dt2/4, dt/2],
                           [dt/2,  1]]) * self.q
        
        self.R = np.array([[self.sigma_theta**2]])
        
        self.reset()

    def reset(self):
        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 10.0
        self.is_initialized = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        if not self.is_initialized:
            self.x = np.array([z, 0.0])
            self.is_initialized = True
            return

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
