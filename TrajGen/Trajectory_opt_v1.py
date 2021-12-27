# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 18:34:31 2021

@author: ASUS
"""

import numpy as np
from numpy import linalg as LA
from scipy import optimize
from .trajutils import *
from scipy import optimize

# from cvxopt import matrix, solvers


class trajOpt:
    def __init__(self, waypoints, Map, max_vel=1.5, gamma=1e4):
        self.waypoints = waypoints
        self.max_vel = max_vel
        self.mean_vel = 1.0
        len, dim = waypoints.shape
        self.dim = dim
        self.len = len
        self.Map=Map
        self.gamma=gamma
        self.time_list = np.zeros(self.len)
        # self.optimize()
        self.yaw = 0
        self.heading = np.zeros(2)
        self.optimize()
        # self.time_allocation()
        self.min_snap_traj()

    def time_allocation(self):  # Assign time interval to each segments
        # waypoints are the vertices of the path
        # weight represents the weight of yaw angle displacement in the overall distance metric in configuration space
        # mean_vel is a very rough estimation of velocity, which is used to calculate the overall time
        position = self.waypoints[:, :3]
        # print('position',position)
        # displacement verctors between each point
        displacement_vec = position[1:]-position[:-1]
        # distance verctors between each point
        displacement = np.linalg.norm(displacement_vec, axis=1, keepdims=True)
        # print('distances',distances)
        T = np.sum(displacement)/self.mean_vel  # Overall time
        t = T*displacement/np.sum(displacement)  # Time spent on each segment
        # print('t',t)
        time_list = np.zeros([self.len])
        for i in range(1, self.waypoints.shape[0]):
            time_list[i] = time_list[i-1]+t[i-1]
        self.time_list = time_list
        
    def get_cost(self, T):
        cost = self.cost_cal(T)
        cost = cost + self.gamma*np.sum(T)
        # print('cost',cost)
        return cost
    
    def cost_cal(self,T):
        self.time_list[1:] = np.cumsum(T)
        self.min_snap_traj()
        return self.cost
    
    def optimize(self):
        position = self.waypoints[:, :3]
        displacement_vec = position[1:]-position[:-1]
        displacement = np.linalg.norm(displacement_vec, axis=1)

        Tmin = displacement/self.max_vel
        T = optimize.minimize(self.get_cost, Tmin, method="COBYLA", constraints=(
            {'type': 'ineq', 'fun': lambda T: T-Tmin}))['x']

        self.time_list[1:] = np.cumsum(T)


    def min_snap_traj(self):
        time_list = self.time_list.squeeze()
        x = self.waypoints[:, 0].squeeze()
        y = self.waypoints[:, 1].squeeze()
        z = self.waypoints[:, 2].squeeze()
        # P=matrix(np.zeros([(self.time_list.shape[0]-1)*8]))
        # Optimization for X direction
        Qx=get_Q(self.time_list,x)
        # Aeqx,Beqx=get_eq_const(time_list,x)
        # solx=solvers.qp(Qx, P, A= Aeqx, b=Beqx)
        # print('x is',solx['status'])
        px_c, _, _ = cls_form(time_list, x)
        # Optimization for Y direction
        Qy=get_Q(time_list,y)
        # Aeqy,Beqy=get_eq_const(time_list,y)
        # soly=solvers.qp(Qy, P, A= Aeqy, b=Beqy)
        # print('y is',soly['status'])
        py_c, _, _ = cls_form(time_list, y)
        # Optimization for Z direction
        Qz=get_Q(time_list,z)
        # Aeqz,Beqz=get_eq_const(time_list,z)
        # solz=solvers.qp(Qz, P, A= Aeqz, b=Beqz)
        # print('z is',solz['status'])
        pz_c, _, _ = cls_form(time_list, z)
        self.px_c = px_c
        self.py_c = py_c
        self.pz_c = pz_c
        self.cost=np.trace(px_c.T@Qx@px_c+py_c.T@Qy@py_c+pz_c.T@Qz@pz_c)
        # print('cost',self.cost+self.gamma*self.time_list[-1])
        self.colli_check()

        
    def colli_check(self):
        T =0
        colli_num=0
        collision=False
        while T< self.time_list[-1]:
            pos=self.get_des_state(T).pos
            if self.Map.idx.count((*pos,)) != 0:
                idx=np.where(T >= self.time_list)[0][-1]+colli_num
                # print(idx)
                new_waypoint=(self.waypoints[idx]+self.waypoints[idx+1])/2
                # print(new_waypoint)
                self.waypoints=np.insert(self.waypoints,idx+1,new_waypoint,axis=0)
                # print(self.waypoints)
                collision=True
                colli_num+=1
                T=self.time_list[idx+1]
            T+=0.025
        if collision:
            self.len = self.waypoints.shape[0]
            self.yaw = 0
            self.heading = np.zeros(2)
            self.optimize()
            # self.time_allocation()
            self.min_snap_traj()

    def get_des_state(self, t):
        # print(t)
        time_list = self.time_list.squeeze()
        for i in range(1, time_list.shape[0]):
            if t < time_list[i] or t == time_list[i]:
                x = np.array([1, t, t**2, t**3, t**4, t**5, t **
                             6, t**7]).dot(self.px_c[8*i-8:8*i])
                x_dot = np.array(
                    [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5, 7*t**6]).dot(self.px_c[8*i-8:8*i])
                x_ddt = np.array([0, 0, 2, 3*2*t, 4*3*t**2, 5*4*t **
                                 3, 6*5*t**4, 7*6*t**5]).dot(self.px_c[8*i-8:8*i])
                x_jerk = np.array([0, 0, 0, 3*2, 4*3*2*t, 5*4*3*t**2,
                                  6*5*4*t**3, 7*6*5*t**4]).dot(self.px_c[8*i-8:8*i])
                y = np.array([1, t, t**2, t**3, t**4, t**5, t **
                             6, t**7]).dot(self.py_c[8*i-8:8*i])
                y_dot = np.array(
                    [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5, 7*t**6]).dot(self.py_c[8*i-8:8*i])
                y_ddt = np.array([0, 0, 2, 3*2*t, 4*3*t**2, 5*4*t **
                                 3, 6*5*t**4, 7*6*t**5]).dot(self.py_c[8*i-8:8*i])
                y_jerk = np.array([0, 0, 0, 3*2, 4*3*2*t, 5*4*3*t**2,
                                  6*5*4*t**3, 7*6*5*t**4]).dot(self.py_c[8*i-8:8*i])
                z = np.array([1, t, t**2, t**3, t**4, t**5, t **
                             6, t**7]).dot(self.pz_c[8*i-8:8*i])
                z_dot = np.array(
                    [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5, 7*t**6]).dot(self.pz_c[8*i-8:8*i])
                z_ddt = np.array([0, 0, 2, 3*2*t, 4*3*t**2, 5*4*t **
                                 3, 6*5*t**4, 7*6*t**5]).dot(self.pz_c[8*i-8:8*i])
                z_jerk = np.array([0, 0, 0, 3*2, 4*3*2*t, 5*4*3*t**2,
                                  6*5*4*t**3, 7*6*5*t**4]).dot(self.pz_c[8*i-8:8*i])
                pos = np.array([x, y, z]).squeeze()
                pos_dot = np.array([x_dot, y_dot, z_dot]).squeeze()
                pos_ddt = np.array([x_ddt, y_ddt, z_ddt]).squeeze()
                yaw, yawdot = self.get_yaw(pos_dot[:2])
                jerk = np.array([x_jerk, y_jerk, z_jerk]).squeeze()
                # print(DesiredState(pos, pos_dot, pos_ddt, jerk, yaw, yawdot))
                return DesiredState(pos, pos_dot, pos_ddt, jerk, yaw, yawdot)
        pos = self.waypoints[-1]
        pos_dot = np.array([0, 0, 0]).squeeze()
        pos_ddt = np.array([0, 0, 0]).squeeze()
        yaw, yawdot = self.get_yaw(pos_dot[:2])
        jerk = np.array([0, 0, 0]).squeeze()
        # print(DesiredState(pos, pos_dot, pos_ddt, jerk, yaw, yawdot))
        return DesiredState(pos, pos_dot, pos_ddt, jerk, yaw, yawdot)

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

    def get_seg_num(self, t):
        if t > self.time_list[-1]:
            t = self.time_list[-1] - 0.001
        i = np.where(t >= self.time_list)[0][-1]
        return i



def cls_form(time_list, x):
    seg_num = x.shape[0]-1
    Aeq = np.zeros([seg_num*8, seg_num*8])
    Beq = np.zeros([seg_num*8, 1])
    for i in range(seg_num):
        if i == 0:
            Aeq[i*6, i*8:i*8+8] = np.array([0, 1, 2*time_list[i], 3*time_list[i]**2, 4*time_list[i]
                                           ** 3, 5*time_list[i]**4, 6*time_list[i]**5, 7*time_list[i]**6])  # Derivative 1 at t
            Aeq[1+i*6, i*8:i*8+8] = np.array([0, 0, 2, 3*2*time_list[i]**1, 4*3*time_list[i]**2,
                                             5*4*time_list[i]**3, 6*5*time_list[i]**4, 7*6*time_list[i]**5])  # Derivative 2 at t
            Aeq[2+i*6, i*8:i*8+8] = np.array([0, 0, 0, 3*2*1, 4*3*2*time_list[i]**1, 5*4*3 *
                                             time_list[i]**2, 6*5*4*time_list[i]**3, 7*6*5*time_list[i]**4])  # Derivative 3 at t
            Aeq[3+i*6, i*8:i*8+8] = -np.array([0, 1, 2*time_list[i+1], 3*time_list[i+1]**2, 4*time_list[i+1]
                                              ** 3, 5*time_list[i+1]**4, 6*time_list[i+1]**5, 7*time_list[i+1]**6])  # Derivative 1 at t+1
            Aeq[4+i*6, i*8:i*8+8] = -np.array([0, 0, 2, 3*2*time_list[i+1]**1, 4*3*time_list[i+1]**2,
                                              5*4*time_list[i+1]**3, 6*5*time_list[i+1]**4, 7*6*time_list[i+1]**5])  # Derivative 2 at t+1
            Aeq[5+i*6, i*8:i*8+8] = -np.array([0, 0, 0, 3*2*1, 4*3*2*time_list[i+1]**1, 5*4*3*time_list[i+1]
                                              ** 2, 6*5*4*time_list[i+1]**3, 7*6*5*time_list[i+1]**4])  # Derivative 3 at t+1
            Aeq[6+i*6, i*8:i*8+8] = -np.array([0, 0, 0, 0, 4*3*2*1, 5*4*3*2*time_list[i+1] **
                                              1, 6*5*4*3*time_list[i+1]**2, 7*6*5*4*time_list[i+1]**3])  # Derivative 4 at t+1
            Aeq[7+i*6, i*8:i*8+8] = -np.array([0, 0, 0, 0, 0, 5*4*3*2*1, 6*5*4*3 *
                                              2*time_list[i+1], 7*6*5*4*3*time_list[i+1]**2])  # Derivative 5 at t+1
            # Derivative 6 at t+1
            Aeq[8+i*6, i*8:i*8+8] = - \
                np.array([0, 0, 0, 0, 0, 0, 6*5*4*3*2*1,
                         np.math.factorial(7)*time_list[i+1]])
        elif i != seg_num-1:
            Aeq[-3+i*6, i*8:i*8+8] = np.array([0, 1, 2*time_list[i], 3*time_list[i]**2, 4*time_list[i]
                                              ** 3, 5*time_list[i]**4, 6*time_list[i]**5, 7*time_list[i]**6])  # Derivative 1 at t
            Aeq[-2+i*6, i*8:i*8+8] = np.array([0, 0, 2, 3*2*time_list[i]**1, 4*3*time_list[i]**2,
                                              5*4*time_list[i]**3, 6*5*time_list[i]**4, 7*6*time_list[i]**5])  # Derivative 2 at t
            Aeq[-1+i*6, i*8:i*8+8] = np.array([0, 0, 0, 3*2*1, 4*3*2*time_list[i]**1, 5*4*3 *
                                              time_list[i]**2, 6*5*4*time_list[i]**3, 7*6*5*time_list[i]**4])  # Derivative 3 at t
            Aeq[i*6, i*8:i*8+8] = np.array([0, 0, 0, 0, 4*3*2*1, 5*4*3*2*time_list[i] **
                                           1, 6*5*4*3*time_list[i]**2, 7*6*5*4*time_list[i]**3])  # Derivative 4 at t
            Aeq[1+i*6, i*8:i*8+8] = np.array([0, 0, 0, 0, 0, 5*4*3*2*1, 6*5*4 *
                                             3*2*time_list[i], 7*6*5*4*3*time_list[i]**2])  # Derivative 5 at t
            # Derivative 6 at t
            Aeq[2+i*6, i*8:i*8+8] = np.array(
                [0, 0, 0, 0, 0, 0, 6*5*4*3*2*1, np.math.factorial(7)*time_list[i]])
            Aeq[3+i*6, i*8:i*8+8] = -np.array([0, 1, 2*time_list[i+1], 3*time_list[i+1]**2, 4*time_list[i+1]
                                              ** 3, 5*time_list[i+1]**4, 6*time_list[i+1]**5, 7*time_list[i+1]**6])  # Derivative 1 at t+1
            Aeq[4+i*6, i*8:i*8+8] = -np.array([0, 0, 2, 3*2*time_list[i+1]**1, 4*3*time_list[i+1]**2,
                                              5*4*time_list[i+1]**3, 6*5*time_list[i+1]**4, 7*6*time_list[i+1]**5])  # Derivative 2 at t+1
            Aeq[5+i*6, i*8:i*8+8] = -np.array([0, 0, 0, 3*2*1, 4*3*2*time_list[i+1]**1, 5*4*3*time_list[i+1]
                                              ** 2, 6*5*4*time_list[i+1]**3, 7*6*5*time_list[i+1]**4])  # Derivative 3 at t+1
            Aeq[6+i*6, i*8:i*8+8] = -np.array([0, 0, 0, 0, 4*3*2*1, 5*4*3*2*time_list[i+1] **
                                              1, 6*5*4*3*time_list[i+1]**2, 7*6*5*4*time_list[i+1]**3])  # Derivative 4 at t+1
            Aeq[7+i*6, i*8:i*8+8] = -np.array([0, 0, 0, 0, 0, 5*4*3*2*1, 6*5*4*3 *
                                              2*time_list[i+1], 7*6*5*4*3*time_list[i+1]**2])  # Derivative 5 at t+1
            # Derivative 6 at t+1
            Aeq[8+i*6, i*8:i*8+8] = - \
                np.array([0, 0, 0, 0, 0, 0, 6*5*4*3*2*1,
                         np.math.factorial(7)*time_list[i+1]])
        else:
            Aeq[-3+i*6, i*8:i*8+8] = np.array([0, 1, 2*time_list[i], 3*time_list[i]**2, 4*time_list[i]
                                              ** 3, 5*time_list[i]**4, 6*time_list[i]**5, 7*time_list[i]**6])  # Derivative 1 at t
            Aeq[-2+i*6, i*8:i*8+8] = np.array([0, 0, 2, 3*2*time_list[i]**1, 4*3*time_list[i]**2,
                                              5*4*time_list[i]**3, 6*5*time_list[i]**4, 7*6*time_list[i]**5])  # Derivative 2 at t
            Aeq[-1+i*6, i*8:i*8+8] = np.array([0, 0, 0, 3*2*1, 4*3*2*time_list[i]**1, 5*4*3 *
                                              time_list[i]**2, 6*5*4*time_list[i]**3, 7*6*5*time_list[i]**4])  # Derivative 3 at t
            Aeq[i*6, i*8:i*8+8] = np.array([0, 0, 0, 0, 4*3*2*1, 5*4*3*2*time_list[i] **
                                           1, 6*5*4*3*time_list[i]**2, 7*6*5*4*time_list[i]**3])  # Derivative 4 at t
            Aeq[1+i*6, i*8:i*8+8] = np.array([0, 0, 0, 0, 0, 5*4*3*2*1, 6*5*4 *
                                             3*2*time_list[i], 7*6*5*4*3*time_list[i]**2])  # Derivative 5 at t
            # Derivative 6 at t
            Aeq[2+i*6, i*8:i*8+8] = np.array(
                [0, 0, 0, 0, 0, 0, 6*5*4*3*2*1, np.math.factorial(7)*time_list[i]])
            Aeq[3+i*6, i*8:i*8+8] = -np.array([0, 1, 2*time_list[i+1], 3*time_list[i+1]**2, 4*time_list[i+1]
                                              ** 3, 5*time_list[i+1]**4, 6*time_list[i+1]**5, 7*time_list[i+1]**6])  # Derivative 1 at t+1
            Aeq[4+i*6, i*8:i*8+8] = -np.array([0, 0, 2, 3*2*time_list[i+1]**1, 4*3*time_list[i+1]**2,
                                              5*4*time_list[i+1]**3, 6*5*time_list[i+1]**4, 7*6*time_list[i+1]**5])  # Derivative 2 at t+1
            Aeq[5+i*6, i*8:i*8+8] = -np.array([0, 0, 0, 3*2*1, 4*3*2*time_list[i+1]**1, 5*4*3*time_list[i+1]
                                              ** 2, 6*5*4*time_list[i+1]**3, 7*6*5*time_list[i+1]**4])  # Derivative 3 at t+1
        Aeq[seg_num*6+i*2, i*8:i*8+8] = np.array([1, time_list[i], time_list[i]**2, time_list[i] **
                                                 3, time_list[i]**4, time_list[i]**5, time_list[i]**6, time_list[i]**7])  # Position at t
        Aeq[1+seg_num*6+i*2, i*8:i*8+8] = np.array([1, time_list[i+1], time_list[i+1]**2, time_list[i+1] **
                                                   3, time_list[i+1]**4, time_list[i+1]**5, time_list[i+1]**6, time_list[i+1]**7])  # Position at t+1
        Beq[seg_num*6+i*2] = x[i]
        Beq[1+seg_num*6+i*2] = x[i+1]
    p = np.linalg.solve(Aeq, Beq)
    return p, Aeq, Beq
def get_Q(time_list,x):
    seg_num=x.shape[0]-1
    Q=np.zeros([seg_num*8,seg_num*8])
    for i in range(seg_num):
        Q[i*8:i*8+8,i*8:i*8+8]=get_subq(time_list[i],time_list[i+1])
    return Q
def get_subq(T1,T2):
    subq=np.zeros([8,8]) 
    para_vec=np.array([[np.math.factorial(4)],[np.math.factorial(5)/np.math.factorial(1)],
                       [np.math.factorial(6)/np.math.factorial(2)],[np.math.factorial(7)/np.math.factorial(3)]])
    para_mat=np.dot(para_vec,para_vec.T)
    T2_mat=np.array([[[T2],[T2**2/2],[T2**3/3],[T2**4/4]],
                     [[T2**2/2],[T2**3/3],[T2**4/4],[T2**5/5]],
                     [[T2**3/3],[T2**4/4],[T2**5/5],[T2**6/6]],
                     [[T2**4/4],[T2**5/5],[T2**6/6],[T2**7/7]]]).squeeze()
    T1_mat=np.array([[[T1],[T1**2/2],[T1**3/3],[T1**4/4]],
                     [[T1**2/2],[T1**3/3],[T1**4/4],[T1**5/5]],
                     [[T1**3/3],[T1**4/4],[T1**5/5],[T1**6/6]],
                     [[T1**4/4],[T1**5/5],[T1**6/6],[T1**7/7]]]).squeeze()
    subq[4:,4:]=para_mat*(T2_mat-T1_mat)
    return subq