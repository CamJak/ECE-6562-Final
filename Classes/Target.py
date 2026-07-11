import numpy as np

# Target class to represent a moving object
class Target:
    def __init__(self, state_vector, id):
        self.state_vector = state_vector
        self.id = id

    def update(self, dt):
        # Citation [1]
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