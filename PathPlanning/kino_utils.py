import numpy as np
from .rrtutils import *
from collections import namedtuple
from numpy import linalg as LA

DesiredState = namedtuple('DesiredState', 'pos vel acc jerk yaw yawdot')

def polyder(t, k=0, order=6):
    '''
    (10th) order polynomial: t**0 + t**1 + ... + t**9
    k: take k derivative
    '''
    terms = np.zeros(order)
    coeffs = np.polyder([1]*order, k)[::-1]
    pows = t**np.arange(0, order-k, 1)
    terms[k:] = coeffs*pows
    return terms

class linearized_qued_model:
    def __init__(self):
        self.mass = 0.18  # kg
        self.g = 9.81  # m/s^2
        self.arm_length = 0.086  # meter
        L = self.arm_length
        self.A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, self.g, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, -self.g, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ])
        self.B = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [1/self.mass, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, L*4000, 0, 0],
                           [0, 0, L*4310.34, 0],
                           [0, 0, 0, L*2675.23], ])
        self.R = np.diag([1, 1, 1, 1])


class Node_with_traj(Node):
    def __init__(self, coords):
        super().__init__(coords)
        self.vel = 0
        self.trajectories = {}
        self.T = 1

class Trajectory_segment:
    def __init__(self, coeff, cost, T):
        self.coeff = coeff  # list of 1d coefficients
        self.cost = cost
        self.T = T

    def get_pos(self, t):
        pos = []
        for i in range(len(self.coeff)):
            pos_val = (self.coeff[i] @ polyder(t, order=6))
            pos.append(pos_val)
        return pos


class trajGenerator:
    def __init__(self, trajectory_segments):
        self.yaw = 0
        self.heading = np.zeros(2)
        self.trajectory_segments = trajectory_segments

    def get_Tmax(self):
        Tmax = 0
        for i in range(len(self.trajectory_segments)-1, 0, -1):
            trajectory_segment = self.trajectory_segments[i]
            Tmax += trajectory_segment.T
        return Tmax
    
    def get_des_state(self, t):
        for i in range(len(self.trajectory_segments)-1, -1, -1):
            trajectory_segment = self.trajectory_segments[i]
            if t > trajectory_segment.T:
                t -= trajectory_segment.T
                continue
            coeff = np.array(trajectory_segment.coeff)
            break
        
        # print(coeff)
        pos = coeff @ polyder(t, order=6)
        vel = coeff @ polyder(t, 1, order=6)
        accl = coeff @ polyder(t, 2, order=6)
        jerk = coeff @ polyder(t, 3, order=6)
        # set yaw in the direction of velocity
        yaw, yawdot = self.get_yaw(vel[:2])
        # print(pos)
        return DesiredState(pos, vel, accl, jerk, yaw, yawdot)


    def get_yaw(self, vel):
        curr_heading = vel/LA.norm(vel)
        prev_heading = self.heading
        cosine = max(-1, min(np.dot(prev_heading, curr_heading), 1))
        dyaw = np.arccos(cosine)
        norm_v = np.cross(prev_heading, curr_heading)
        self.yaw += np.sign(norm_v)*dyaw

        if self.yaw > np.pi:
            self.yaw -= 2*np.pi
        if self.yaw < -np.pi:
            self.yaw += 2*np.pi

        self.heading = curr_heading
        yawdot = max(-30, min(dyaw/0.005, 30))
        return self.yaw, yawdot
