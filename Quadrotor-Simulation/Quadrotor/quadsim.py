import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
from collections import deque

from TrajGen.trajutils import DesiredState
from .quadrotor import Quadrotor
from .VO import ComputeVelocityObstacle


class QuadSim:
    # control frequency was set at 200
    def __init__(self,controller,des_state,Tmax, rev_des_state,
                 pos = None, rev_pos=None, attitude = [0,0,0],
                 animation_frequency = 50,
                 control_frequency = 300):

        self.t = 0
        self.Tmax = Tmax
        self.dt = 1/control_frequency
        self.animation_rate = 1/animation_frequency
        self.control_iterations = int(control_frequency / animation_frequency)

        self.des_state = des_state
        self.rev_des_state = rev_des_state
        self.controller = controller
        if pos is None: pos = des_state(0).pos
        if rev_pos is None: rev_pos = rev_des_state(0).pos
        self.Quadrotor = Quadrotor(pos, attitude)
        self.rev_Quadrotor = Quadrotor(rev_pos, attitude)

        self.pos_history = deque(maxlen=100)
        self.rev_pos_history = deque(maxlen=100)

    def Step(self):
        des_state = self.des_state(self.t)
        rev_des_state = self.rev_des_state(self.t)
        state = self.Quadrotor.get_state()
        rev_state = self.rev_Quadrotor.get_state()


        ### add Velocity Obstaclesa
        # des_pos_VO = des_state.pos + des_state.pos *0.1
        des_pos_VO = ComputeVelocityObstacle(state, des_state, rev_state, self.dt)  # function in different file
        des_state = DesiredState(des_pos_VO, des_state.vel, des_state.acc, des_state.jerk, des_state.yaw,
                               des_state.yawdot)
        #



        if(self.t >= self.Tmax):
            U, M = self.controller.run_hover(state, des_state,self.dt)
            rev_U, rev_M = self.controller.run_hover(rev_state, rev_des_state, self.dt)
        else:
            U, M = self.controller.run(state, des_state)
            rev_U, rev_M = self.controller.run(rev_state, rev_des_state)
        self.Quadrotor.update(self.dt, U, M)
        self.rev_Quadrotor.update(self.dt, rev_U, rev_M)
        self.t += self.dt

    def control_loop(self):
        for _ in range(self.control_iterations):
            self.Step()
        return  self.Quadrotor.world_frame(), self.rev_Quadrotor.world_frame()

    def run(self,ax = None,save = False):
        self.init_plot(ax)
        while self.t < self.Tmax + 2:
            frame1, frame2 = self.control_loop()
            self.update_plot(frame1, frame2)
            plt.pause(self.animation_rate)

    def init_plot(self,ax = None):
        if ax is None:
            fig = plt.figure()
            ax = Axes3D.Axes3D(fig)
            ax.set_xlim((0,2))
            ax.set_ylim((0,2))
            ax.set_zlim((0,2))
        # for drone 1
        ax.plot([], [], [], '-', c='red',zorder = 10)
        ax.plot([], [], [], '-', c='red',zorder = 10)
        ax.plot([], [], [], '-', c='red', marker='o', markevery=2,zorder = 10)
        ax.plot([], [], [], '.', c='red', markersize=2,zorder = 10)
        # for drone 2
        ax.plot([], [], [], '-', c='red', zorder=10)
        ax.plot([], [], [], '-', c='blue', zorder=10)
        ax.plot([], [], [], '-', c='green', marker='o', markevery=2, zorder=10)
        ax.plot([], [], [], '.', c='green', markersize=2, zorder=10)

        # self.lines = ax.get_lines()[-4:]
        self.lines = ax.get_lines()[-8:]

    def update_plot(self, frame, frame2):

        # for quadrotor 1
        lines_data = [frame[:,[0,2]], frame[:,[1,3]], frame[:,[4,5]]]

        for line, line_data in zip(self.lines[:3], lines_data):
            x, y, z = line_data
            line.set_data(x, y)
            line.set_3d_properties(z)

        self.pos_history.append(frame[:,4])
        history = np.array(self.pos_history)


        self.lines[-1].set_data(history[:,0], history[:,1])
        self.lines[-1].set_3d_properties(history[:,-1])


        # for quadrotor 2
        lines_data2 = [frame2[:, [0, 2]], frame2[:, [1, 3]], frame2[:, [4, 5]]]

        for line, line_data in zip(self.lines[4:7], lines_data2):
            x, y, z = line_data
            line.set_data(x, y)
            line.set_3d_properties(z)

        self.rev_pos_history.append(frame2[:, 4])
        rev_history = np.array(self.rev_pos_history)

        self.lines[3].set_data(rev_history[:, 0], rev_history[:, 1])
        self.lines[3].set_3d_properties(rev_history[:, -1])
