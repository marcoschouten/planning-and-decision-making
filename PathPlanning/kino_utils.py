import numpy as np
from .rrtutils import *
from collections import namedtuple
from numpy import linalg as LA

DesiredState = namedtuple('DesiredState', 'pos vel acc jerk yaw yawdot')
poly_order = 10  # 9th order polynomial


def polyder(t, k=0, order=poly_order):
    '''
    (10th) order polynomial: t**0 + t**1 + ... + t**9
    k: take k derivative
    '''
    terms = np.zeros(order)
    coeffs = np.polyder([1]*order, k)[::-1]
    pows = t**np.arange(0, order-k, 1)
    terms[k:] = coeffs*pows
    return terms


def Hessian(T, order=poly_order, opt=4):
    T = [float(T)]
    n = len(T)
    Q = np.zeros((order*n, order*n))
    for k in range(n):
        m = np.arange(0, opt, 1)
        for i in range(order):
            for j in range(order):
                if i >= opt and j >= opt:
                    pow = i+j-2*opt+1
                    Q[order*k+i, order*k+j] = 2 * \
                        np.prod((i-m)*(j-m))*T[k]**pow/pow
    return Q


class Node_with_traj(Node):
    def __init__(self, coords):
        super().__init__(coords)
        self.vel = 0
        self.acc = 0
        self.jerk = 0
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
            pos_val = (self.coeff[i] @ polyder(t, order=poly_order))
            pos.append(pos_val)
        return pos

    def get_des_state_seg(self, t):
        coeff = np.array(self.coeff)
        pos = coeff @ polyder(t, order=poly_order)
        vel = coeff @ polyder(t, 1, order=poly_order)
        acc = coeff @ polyder(t, 2, order=poly_order)
        jerk = coeff @ polyder(t, 3, order=poly_order)
        return pos, vel, acc, jerk


class trajGenerator:
    def __init__(self, trajectory_segments):
        self.yaw = 0
        self.heading = np.zeros(2)
        self.trajectory_segments = trajectory_segments

    def get_Tmax(self):
        Tmax = 0
        for seg in self.trajectory_segments:
            Tmax += seg.T
        return Tmax

    def get_des_state(self, t):
        # determine which trajectory we should use
        for i in range(len(self.trajectory_segments)-1, -1, -1):
            trajectory_segment = self.trajectory_segments[i]
            if t > trajectory_segment.T:
                t -= trajectory_segment.T
                continue
            break

        pos, vel, acc, jerk = trajectory_segment.get_des_state_seg(t)
        # set yaw in the direction of velocity
        yaw, yawdot = self.get_yaw(vel[:2])
        # print(pos)
        return DesiredState(pos, vel, acc, jerk, yaw, yawdot)

    def get_yaw(self, vel):
        if LA.norm(vel) < 1e-3:
            curr_heading = self.heading
        else:  
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
