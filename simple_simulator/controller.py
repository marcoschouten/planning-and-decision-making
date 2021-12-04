import numpy as np
from scipy.spatial.transform import Rotation
import quadrotor
import cvxpy as cp
import trajectory


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

    def control(self, cur_time, state, input_traj):
        '''
        :param desired state: pos, vel, acc, yaw, yaw_dot
        :param current state: pos, vel, euler, omega
        :return:
        '''
        cmd_state, _ = trajectory.generate_trajec(input_traj, cur_time) # Change the trajectory
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # page 29, position controller
        error_pos = cmd_state.get('x') - state.get('x').reshape(3, 1)
        error_vel = cmd_state.get('x_dot') - state.get('v').reshape(3, 1)
        rdd_des = cmd_state.get('x_ddt')
        rdd_cmd = rdd_des + self.Kd @ error_vel + self.Kp @ error_pos
        u1 = self.mass * (self.g + rdd_cmd[2]).reshape(-1, 1)
        
        # page 30, attitude controller
        psi_des = cmd_state.get('yaw')
        phi_cmd = (rdd_cmd[0] * np.sin(psi_des) - rdd_cmd[1] * np.cos(psi_des)) / self.g
        theta_cmd = (rdd_cmd[0] * np.cos(psi_des) + rdd_cmd[1] * np.sin(psi_des)) / self.g
        quat = state['q']
        rotation = Rotation.from_quat(quat)
        angle = np.array(rotation.as_rotvec()).reshape(3, 1)  # euler angles
        omega = np.array(state['w']).reshape(3, 1)  # anglar velocity
        psid_des = cmd_state.get('yaw_dot')
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
        return control_input, cmd_state


class MPC():
    def __init__(self):
        self.dt = 0.01

    def control(self, cur_time, state, input_traj):
        """
        Build discrete linearization model
        x(t+1) = A @ x(t) + B @ u(t) + C
        """
        A = np.array([[1, 0, -0.004905, 0.01, 0, -0.000001635],
                      [0, 1, 0, 0, 0.01, 0],
                      [0, 0, 1, 0, 0, 0.01],
                      [0, 0, -0.0981, 1, 0, -0.0004905],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
        B = np.array([[0, -0.0002858],
                      [0.001667, 0],
                      [0, 3.497],
                      [0, -0.1143],
                      [0.3333, 0],
                      [0, 699.3]])
        C = np.array([0, -0.0004905, 0, 0, -0.0981, 0])
        if state[2] > 1.5:
            state[2] = 1.5
        elif state[2] < -1.5:
            state[2] =-1.5
        x_0 = np.array([state[0], state[1], state[2], state[3], state[4], state[5]])
        T = 20  # the number of predicted steps
        x = cp.Variable((6, T + 1))
        u = cp.Variable((2, T))
        cost = 0
        constr = []
        """
        It could be solved as a Convex optimization problem
        Given a target position [yt, zt] = [0,1], [y, z] = [x[0], x[1]]
        minimize ||[y, z]-[yt, zt]||
        Moreover, minimize ||[y, z]-[yt, zt]|| + ||[Fz, Mx]|| to save energy 
        """
        for t in range(T):
            cost += cp.sum_squares(x[0:2, t + 1]-np.array([0,0.5]))
            constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C,
                       cp.norm(x[2,t], 'inf') <= 1,
                       u[0, t] <= 0.75,  # limit of motor speed
                       cp.norm(u[1, t], 'inf') <= 0.072]
        # sums problem objectives and concatenates constraints.
        constr += [x[:, 0] == x_0]
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.ECOS)
        action = np.random.randn(2)
        action[0] = u[0, 0].value
        action[1] = u[1, 0].value
        print("u1 u2:",action)
        return action
    