import numpy as np

from Classes.Target import *
from Classes.Radar import *
from Classes.Camera import *

# Class to handle fusion of estimates, core of HT2TF
class FusionManager:
    def __init__(self, camera_position):
        self.cam_pos = camera_position

    def fuse(self, radar_track, camera_track, P_re=None):
        # Fail if either track is uninitialized
        if not radar_track.is_initialized and camera_track.is_initialized:
            return None, None
        if radar_track.is_initialized and not camera_track.is_initialized:
            return radar_track.x, radar_track.P
        if not radar_track.is_initialized and not camera_track.is_initialized:
            return None, None
        
        x_r = radar_track.x
        P_r = radar_track.P

        x_e = camera_track.x
        P_e = camera_track.P

        # Relate radar position data to camera reference frame
        dx = x_r[0] - self.cam_pos[0]
        dy = x_r[2] - self.cam_pos[1]
        vx = x_r[1]
        vy = x_r[3]

        r2 = dx**2 + dy**2 + 1e-6 # Prevent division by zero

        # Project radar position data onto camera reference frame
        v_theta = dx * vy - dy * vx
        theta_pred = np.arctan2(dy, dx)
        theta_dot_pred = v_theta / r2

        h_x = np.array([theta_pred, theta_dot_pred])

        ## Construct the Jacobian
        # Equations 15-17 [1]
        v = np.sqrt(vx**2 + vy**2) + 1e-6
        re = np.sqrt(dx**2 + dy**2) + 1e-6
        phi = np.arctan2(vy, vx) - np.arctan2(dy, dx)

        G = np.zeros((2, 4))

        # Equations 37 - 44 [1]
        G[0, 0] = -dy / (re**2)
        G[0, 1] = 0.0 
        G[0, 2] = dx / (re**2)
        G[0, 3] = 0.0
        G[1, 0] = (v / (re**3)) * (-np.sin(phi) * dx + np.cos(phi) * dy)
        G[1, 1] = (1.0 / (v * re)) * (np.sin(phi) * vx - np.cos(phi) * vy)
        G[1, 2] = (v / (re**3)) * (-np.sin(phi) * dx - np.cos(phi) * dy)
        G[1, 3] = (1.0 / (v * re)) * (np.sin(phi) * vy + np.cos(phi) * vx)

        if P_re is None:
            P_re = np.zeros((4,2))
        P_er = P_re.T

        # Equations 30 and 31 [1]
        P_xz = P_r @ G.T - P_re
        P_zz = P_e - G @ P_re - P_er @ G.T + G @ P_r @ G.T

        P_zz += np.eye(2) * 1e-6

        W = P_xz @ np.linalg.inv(P_zz)

        # Calculate Innovation
        y_innov = x_e - h_x
        y_innov[0] = (y_innov[0] + np.pi) % (2 * np.pi) - np.pi # Normalize angle between -pi and pi

        # Create fused estimate
        x_f = x_r + W @ y_innov
        P_f = P_r - W @ P_xz.T

        return x_f, P_f