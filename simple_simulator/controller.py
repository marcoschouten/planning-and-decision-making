import numpy as np
from scipy.spatial.transform import Rotation
import quadrotor

class PDcontroller:
    def __init__(self):
        # hover control gains
        self.Kp = np.diag([10, 10, 200])
        self.Kd = np.diag([10, 10, 3])
        # angular control gains
        self.Kp_t = np.diag([250, 250, 30])
        self.Kd_t = np.diag([30, 30, 7.55])
        # get quadrotor model info.
        quad_model = quadrotor.Quadrotor()
        self.mass = quad_model.mass
        self.g = quad_model.g
        self.arm_length = quad_model.arm_length
        self.rotor_speed_min = 0
        self.rotor_speed_max = 2500
        self.k_thrust = quad_model.k_thrust
        self.k_drag = quad_model.k_drag
        self.to_TM = quad_model.to_TM
        I = np.array([[1.43e-5, 0, 0],
                      [0, 1.43e-5, 0],
                      [0, 0, 2.89e-5]])  # inertial tensor in m^2 kg
        self.inertia = I
        # self.invI = np.linalg.inv(I)

    def control(self, flat_output, state):
        '''
        :param desired state: pos, vel, acc, yaw, yaw_dot
        :param current state: pos, vel, euler, omega
        :return:
        '''
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # page 29, position controller
        error_pos = flat_output.get('x') - state.get('x').reshape(3, 1)
        error_vel = flat_output.get('x_dot') - state.get('v').reshape(3, 1)
        rdd_des = flat_output.get('x_ddt')
        rdd_cmd = rdd_des + self.Kd @ error_vel + self.Kp @ error_pos
        u1 = self.mass * (self.g + rdd_cmd[2]).reshape(-1, 1)
        
        # page 30, attitude controller
        psi_des = flat_output.get('yaw')
        phi_cmd = (rdd_cmd[0] * np.sin(psi_des) - rdd_cmd[1] * np.cos(psi_des)) / self.g
        theta_cmd = (rdd_cmd[0] * np.cos(psi_des) + rdd_cmd[1] * np.sin(psi_des)) / self.g
        quat = state['q']
        rotation = Rotation.from_quat(quat)
        angle = np.array(rotation.as_rotvec()).reshape(3, 1)  # euler angles
        omega = np.array(state['w']).reshape(3, 1)  # anglar velocity
        psid_des = flat_output.get('yaw_dot')
        ang_ddt = self.Kd_t @ (np.array([[0], [0], [psid_des]]) - omega) + \
             self.Kp_t @ (np.array([[phi_cmd[0]], [theta_cmd[0]], [psi_des]]) - angle)
        u2 = self.inertia @ ang_ddt

        u = np.vstack((u1, u2))

        F = (np.linalg.inv(self.to_TM) @ u).astype(float)

        for i in range(4):
            if F[i] < 0:
                F[i] = 0
                cmd_motor_speeds[i] = self.rotor_speed_min
            cmd_motor_speeds[i] = np.sqrt(F[i] / self.k_thrust)
            if cmd_motor_speeds[i] > self.rotor_speed_max:
                cmd_motor_speeds[i] = self.rotor_speed_max

        cmd_thrust = u1
        cmd_moment[0] = u2[0]
        cmd_moment[1] = u2[1]
        cmd_moment[2] = u2[2]

        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment,
                         'error_pos': error_pos}
        return control_input


    