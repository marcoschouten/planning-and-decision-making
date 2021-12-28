from .quadsim import *
from numpy import linalg as LA

class QuadSim_plan_traj_visual(QuadSim):
    def __init__(self, controller, des_state, Tmax,
                 pos=None, attitude=[0, 0, 0],
                 animation_frequency=50,
                 control_frequency=200):
        super().__init__(controller, des_state, Tmax, pos=pos, attitude=attitude,
                         animation_frequency=animation_frequency, control_frequency=control_frequency)
        self.planned_pos_history = [] # make the tail eliminate
        
    def init_plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = Axes3D.Axes3D(fig)
            ax.set_xlim((0, 2))
            ax.set_ylim((0, 2))
            ax.set_zlim((0, 2))
        ax.plot([], [], [], '-', c='red', zorder=10) # quad body
        ax.plot([], [], [], '-', c='blue', zorder=10) # quad body
        ax.plot([], [], [], '-', c='green', marker='o', markevery=2, zorder=10) # quad body
        ax.plot([], [], [], '-', c='red', markersize=3, zorder=10) # planned traj
        ax.plot([], [], [], '.', c='green', markersize=3, zorder=10) # real traj
        self.lines = ax.get_lines()[-5:]

    def update_plot(self, frame):
        lines_data = [frame[:, [0, 2]], frame[:, [1, 3]], frame[:, [4, 5]]]
        for line, line_data in zip(self.lines[:3], lines_data):
            x, y, z = line_data
            line.set_data(x, y)
            line.set_3d_properties(z)

        self.pos_history.append(frame[:, 4])
        history = np.array(self.pos_history)
        self.lines[-1].set_data(history[:, 0], history[:, 1])
        self.lines[-1].set_3d_properties(history[:, -1])
        
        des_state = self.des_state(self.t)
        self.planned_pos_history.append(des_state.pos)
        print("pos: ", des_state.pos, "vel: ", des_state.vel, "acc: ", des_state.acc)
        history = np.array(self.planned_pos_history)
        self.lines[-2].set_data(history[:, 0], history[:, 1])
        self.lines[-2].set_3d_properties(history[:, -1])
        
        if LA.norm(self.pos_history[-1] - des_state.pos) > 5:
            raise ValueError("Out of control!")
        
    def run(self, ax=None, save=False):
        self.init_plot(ax)
        while self.t < self.Tmax + 20:
            frame = self.control_loop()
            self.update_plot(frame)
            plt.pause(self.animation_rate)