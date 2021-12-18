# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 22:09:45 2021

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 18:34:31 2021

@author: ASUS
"""

import numpy as np
from numpy import linalg as LA
from .trajutils import *
from PathPlanning.sampleutils import *
class dwa_planner:
    def __init__(self,des_state,state,Map,goal,waypoints,dt=0.005,state_dot=np.zeros(13)):
        self.destate = des_state
        self.state=state
        self.Map=Map
        self.max_vel=3.0
        self.max_acc=5.0
        self.sim_time=0.4
        self.samp_num=90
        self.segnum=0
        self.state_dot=state_dot
        self.waypoints=waypoints
        self.dt=dt
        self.goal=goal*0.02
        self.yaw = 0
        self.heading= np.zeros(2)
    def update(self,des_state,state,segnum,state_dot):
        self.destate = des_state
        self.state=state
        self.segnum=segnum
        self.state_dot=state_dot
    def plan(self):
        pos=self.destate.pos
        vel=self.destate.vel
        collision,_=self.collision_distance(pos,vel)
        if collision:##################################
            pos, pos_dot, pos_ddt, jerk=self.sample_vel()
            yaw,yawdot=self.get_yaw(pos_dot[:2])
            print('pos',pos)
            print('yaw',yaw)
            print('yawdot',yawdot)
            # raise NotImplementedError()
            return DesiredState(pos, pos_dot, pos_ddt, jerk, 0, 0)
        self.heading=vel[:2]/LA.norm(vel[:2])
        return self.destate
    def collision_distance(self,pos,vel):#Calculate the distance before collision
        t=np.linspace(0,self.sim_time,40)
        for T in t:
            sim_pos=pos+vel*T
            if self.Map.idx.count((*sim_pos,)) != 0:
                distance=LA.norm(vel)*T
                if LA.norm(vel)<np.sqrt(2*distance*self.max_acc):
                    return False,distance
                else:
                    return True,distance
        return False,np.inf
    def sample_vel(self):
        i=0
        sampled_vel=np.zeros([self.samp_num,3])
        clearance=np.zeros([self.samp_num,])  
        while i<self.samp_num:
            # print('radius',self.max_acc*self.dt)
            sampled_point=SphereSampler(self.state.vel, self.max_acc*self.dt, num = 1)
            # print('first sample',sampled_point)
            if LA.norm(sampled_point)<self.max_vel or LA.norm(sampled_point)==self.max_vel:
                # print(' second sample',sampled_point)
                collision,distance=self.collision_distance(self.state.pos,sampled_point)
                if not collision:
                    # print('     third sample',sampled_point)
                    sampled_vel[i]=sampled_point
                    clearance[i]=distance
                    i+=1
        # print('sampled_vel',sampled_vel)
        # raise NotImplementedError()
        pos_dot=self.vel_selection(sampled_vel,clearance)
        pos=self.state.pos+pos_dot*self.dt
        pos_ddt=(pos_dot-self.state.vel)/self.dt
        jerk=(pos_ddt-self.state_dot[3:6])/self.dt
        return pos, pos_dot, pos_ddt, jerk
        
    def vel_selection(self,sampled_vel,clearance):
        pos=self.state.pos
        segnum=self.segnum
        goal_heading=(self.goal-pos)/LA.norm(self.goal-pos)
        way_heading=(self.waypoints[segnum+1]-pos)/LA.norm(self.waypoints[segnum+1]-pos)
        # raise NotImplementedError()
        ang_cur_goal=np.ones(clearance.shape)
        ang_cur_way=np.ones(clearance.shape)
        for i in range(sampled_vel.shape[0]):
            curr_heading = sampled_vel[i]/LA.norm(sampled_vel[i])
            cosine_cur_goal= max(-1,min(np.dot(goal_heading, curr_heading),1))
            ang_cur_goal[i]= np.arccos(cosine_cur_goal)
            cosine_cur_way= max(-1,min(np.dot(way_heading, curr_heading),1))
            ang_cur_way[i]= np.arccos(cosine_cur_way)
            if ang_cur_way[i] > np.pi: ang_cur_way[i] -= 2*np.pi
            if ang_cur_way[i] < -np.pi: ang_cur_way[i] += 2*np.pi
            if ang_cur_goal[i] > np.pi: ang_cur_goal[i] -= 2*np.pi
            if ang_cur_goal[i] < -np.pi: ang_cur_goal[i] += 2*np.pi
        cost=1/clearance+1.5*ang_cur_goal+4*ang_cur_way+1/LA.norm(sampled_vel,axis=1)
        # print('clearance',clearance.shape)
        # print('ang goal',ang_cur_goal.shape)
        # print('ang way',ang_cur_way.shape)
        # print(cost.shape)
        # raise NotImplementedError()
        pos_dot=sampled_vel[np.argmin(cost)]
        return pos_dot
    def get_yaw(self,vel):
        curr_heading = vel/LA.norm(vel)
        prev_yaw= self.state.rot[2]
        prev_heading = np.array([np.cos(prev_yaw),np.sin(prev_yaw)])
        # prev_heading=self.heading
        cosine = max(-1,min(np.dot(prev_heading, curr_heading),1))
        dyaw = np.arccos(cosine)
        norm_v = np.cross(prev_heading,curr_heading)
        self.yaw += np.sign(norm_v)*dyaw
        
        if self.yaw > np.pi: self.yaw -= 2*np.pi
        if self.yaw < -np.pi: self.yaw += 2*np.pi
        self.heading = curr_heading
        yawdot = max(-30,min(dyaw/0.005,30))
        # print('yaw',self.yaw)
        return self.yaw,yawdot

