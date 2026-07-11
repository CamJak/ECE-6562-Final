import numpy as np

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
    def __init__(self, target_id, radar_position, sigma_range, sigma_azimuth_rad, dt, q_bar=0.5):
        self.target_id = target_id
        self.sensor_pos = radar_position
        self.sigma_r = sigma_range
        self.sigma_theta = sigma_azimuth_rad
        self.dt = dt
        self.q_bar = q_bar

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

        # Citation [1]
        self.F = np.array([[1, self.dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, self.dt],
                      [0, 0, 0, 1]])
        
        dt2 = self.dt**2
        dt3 = self.dt**3

        self.Q = np.array([
            [dt3/3,   dt2/2,    0,           0],
            [dt2/2,   self.dt,       0,           0],
            [0,       0,        dt3/3,   dt2/2],
            [0,       0,        dt2/2,      self.dt]
        ]) * (self.q_bar**2)

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

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, raw_measurement, target_id):
        if target_id != self.target_id:
            self.target_id = target_id
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

        # Citation: "Unbiased Converted Measurements for Tracking" [2]
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