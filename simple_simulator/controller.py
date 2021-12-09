import numpy as np
from scipy.spatial.transform import Rotation
import quadrotor
import cvxpy as cp
import trajectory


class Controller:
    def __init__(self):
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
        self.invI = np.linalg.inv(I)

    def generate_control_input(self, u):
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        # cmd_q = np.zeros((4,))

        F = (np.linalg.inv(self.to_TM) @ u).astype(float)

        for i in range(4):
            if F[i] < 0:
                F[i] = 0
                cmd_motor_speeds[i] = self.rotor_speed_min
            cmd_motor_speeds[i] = np.sqrt(F[i] / self.k_thrust)
            if cmd_motor_speeds[i] > self.rotor_speed_max:
                cmd_motor_speeds[i] = self.rotor_speed_max

        cmd_thrust = u[0]
        cmd_moment[0] = u[1]
        cmd_moment[1] = u[2]
        cmd_moment[2] = u[3]

        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment}
        return control_input


class PDcontroller(Controller):
    def __init__(self):
        Controller.__init__(self)
        # hover control gains
        self.Kp = np.diag([10, 10, 200])
        self.Kd = np.diag([10, 10, 3])
        # angular control gains
        self.Kp_t = np.diag([250, 250, 30])
        self.Kd_t = np.diag([30, 30, 7.55])

    def control(self, cur_time, obs_state, input_traj):
        '''
        :param desired state: pos, vel, acc, yaw, yaw_dot
        :param current state: pos, vel, euler, omega
        :return:
        '''
        des_state, _ = trajectory.generate_trajec(
            input_traj, cur_time)  # Change the trajectory
        print(input_traj)

        # position controller
        error_pos = des_state.get('x') - obs_state.get('x').reshape(3, 1)
        error_vel = des_state.get('x_dot') - obs_state.get('v').reshape(3, 1)
        rdd_des = des_state.get('x_ddt')
        rdd_cmd = rdd_des + self.Kd @ error_vel + self.Kp @ error_pos
        u1 = self.mass * (self.g + rdd_cmd[2]).reshape(-1, 1)

        # attitude controller
        psi_des = des_state.get('yaw')
        phi_cmd = (rdd_cmd[0] * np.sin(psi_des) -
                   rdd_cmd[1] * np.cos(psi_des)) / self.g
        theta_cmd = (rdd_cmd[0] * np.cos(psi_des) +
                     rdd_cmd[1] * np.sin(psi_des)) / self.g
        quat = obs_state['q']
        rotation = Rotation.from_quat(quat)
        angle = np.array(rotation.as_rotvec()).reshape(3, 1)  # euler angles
        omega = np.array(obs_state['w']).reshape(3, 1)  # anglar velocity
        psid_des = des_state.get('yaw_dot')
        ang_ddt = self.Kd_t @ (np.array([[0], [0], [psid_des]]) - omega) + \
            self.Kp_t @ (np.array([[phi_cmd[0]],
                         [theta_cmd[0]], [psi_des]]) - angle)
        u2 = self.inertia @ ang_ddt

        u = np.vstack((u1, u2))
        control_input = self.generate_control_input(u)
        return control_input, des_state, error_pos


'''
Classical MPC without terminal cost
MPC is linearized around hovering condition.
We also assume the heading angle does not change. (roll = pitch = yaw = 0)
This only works for nonzero roll and pitch reference.
'''
class Linear_MPC(Controller):
    def __init__(self):
        Controller.__init__(self)
        self.dt = 0.01
        # Linearized state space model
        # Lec09, P74, Wei Pan
        self.A_c = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ])
        self.B_c = np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [1/self.mass, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 69930.06993007, 0, 0],
                             [0, 0, 69930.06993007, 0],
                             [0, 0, 0, 34602.07612457], ])
        self.C_c = np.array([[0],
                             [0],
                             [0],
                             [0],
                             [0],
                             [-self.g],
                             [0],
                             [0],
                             [0],
                             [0],
                             [0],
                             [0]])
        # Euler discretization
        self.A = np.eye(12) + self.A_c * self.dt
        self.B = self.B_c * self.dt
        self.C = self.C_c

    def control(self, cur_time, obs_state, input_traj):
        x_init = np.block(
            [obs_state.get('x'),  obs_state.get('v'), np.zeros(6)])
        N = 20  # the number of predicted steps
        x = cp.Variable((12, N + 1))
        u = cp.Variable((4, N))
        cost = 0
        constr = []
        sub_Q = np.array([[100, 0, 0, 0, 0, 0],
                          [0, 100, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])
        Q = np.block([[sub_Q, np.zeros((6, 6))],
                      [np.zeros((6, 6)), np.zeros((6, 6))]])
        R = np.eye(4)
        des_state, _ = trajectory.generate_trajec(
            input_traj, cur_time)  # Change the trajectory
        error_pos = des_state.get(
                    'x') - obs_state.get('x').reshape(3, 1)
        
        mpc_time = cur_time

        for k in range(N):
            des_state_ahead, _ = trajectory.generate_trajec(
                input_traj, mpc_time)  # Change the trajectory                
            mpc_time += k * self.dt
            x_ref_k = np.block([[des_state_ahead.get('x')],
                                [des_state_ahead.get('x_dot')],
                                [np.zeros((6, 1))]]).flatten()
            
            cost += cp.quad_form(x[:, k+1] - x_ref_k, Q)
            # cost += cp.quad_form(u[:, k], R)
            constr.append(x[:, k + 1] == self.A @ x[:, k] +
                          self.B @ u[:, k] + self.C.flatten())
            constr.append(u[0, 0] <= 2.5 * self.mass * self.g)
            # constr.append(cp.norm(x[5:8, k], 'inf') <= 5)

        constr.append(x[:, 0] == x_init)
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.OSQP)

        u = u[:, 0].value
        print("control input:")
        print(u)
        control_input = self.generate_control_input(u)
        return control_input, des_state, error_pos

