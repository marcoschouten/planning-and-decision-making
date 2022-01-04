import numpy as np
from numpy import linalg as LA
from .trajutils import *
from PathPlanning.sampleutils import *


class dwa_planner:
    def __init__(self, des_state, state, Map, goal, waypoints, dt=0.005, state_dot=np.zeros(13)):
        self.destate = des_state
        self.state = state
        self.Map = Map
        self.max_vel = 5.0
        self.max_acc = 20.0
        self.sim_time = 0.5
        self.samp_num = 150
        self.segnum = 0
        self.state_dot = state_dot
        self.waypoints = waypoints
        self.dt = dt
        self.goal = goal*0.02
        self.yaw = 0
        self.heading = np.zeros(2)
        self.generateline()
    def generateline(self):
        lines=np.ones([self.waypoints.shape[0]-1,2,3])
        for i in range(1,self.waypoints.shape[0]):
            lines[i-1,0]=self.waypoints[i-1]
            lines[i-1,1]=self.waypoints[i]
        self.lines=lines

    def update(self, des_state, state, segnum, state_dot):
        self.destate = des_state
        self.state = state
        # self.segnum = segnum
        self.state_dot = state_dot

    def plan(self):
        pos = self.destate.pos
        vel = self.destate.vel
        collision, _ = self.collision_distance(pos, vel)
        if collision:
            pos, pos_dot, pos_ddt, jerk = self.sample_vel()
            return DesiredState(pos, pos_dot, pos_ddt, jerk, 0, 0)
        self.heading = vel[:2]/LA.norm(vel[:2])
        return self.destate
        # pos, pos_dot, pos_ddt, jerk = self.sample_vel()
        # yaw, yawdot = self.get_yaw(pos_dot[:2])
        # # raise NotImplementedError()
        # return DesiredState(pos, pos_dot, pos_ddt, jerk, 0, 0)

    # Calculate the distance before collision
    def collision_distance(self, pos, vel):
        t = np.linspace(0, self.sim_time, 30)
        for T in t:
            sim_pos = (pos+vel*T)/0.02
            if self.Map.idx.count((*sim_pos,)) != 0:
                # brake distance
                distance = LA.norm(vel)*T
                if LA.norm(vel) < np.sqrt(2*distance*self.max_acc):
                    return False, distance
                else:
                    return True, distance
        return False, np.inf

    def sample_vel(self):
        i = 0
        sampled_vel = np.zeros([self.samp_num, 3])
        clearance = np.zeros([self.samp_num, ])
        while i < self.samp_num:
            sampled_point = SphereSampler(self.state.vel, self.max_acc*self.dt, num=1)
            print('first sample',LA.norm(sampled_point))
            if LA.norm(sampled_point) < self.max_vel or LA.norm(sampled_point) == self.max_vel:
                print('     second sample',sampled_point)
                collision, distance = self.collision_distance(
                    self.state.pos, sampled_point)
                # print(distance)
                if not collision:
                    print('         third sample',sampled_point)
                    sampled_vel[i] = sampled_point
                    clearance[i] = distance
                    i += 1
        # print('sampled_vel',sampled_vel)
        # raise NotImplementedError()
        pos_dot = self.vel_selection(sampled_vel, clearance)
        pos = self.state.pos+pos_dot*self.dt
        pos_ddt = (pos_dot-self.state.vel)/self.dt
        jerk = (pos_ddt-self.state_dot[3:6])/self.dt
        return pos, pos_dot, pos_ddt, jerk

    def vel_selection(self, sampled_vel, clearance):
        pos = self.state.pos
        # segnum = self.segnum
        # print('position',pos)
        goal_heading = (self.goal-pos)/LA.norm(self.goal-pos)
        _,segnumber=compute_pldis(pos,self.lines)
        # print('segnumber',segnumber)
        # way_heading = (self.waypoints[segnumber+1]-pos) / \
        #     LA.norm(self.waypoints[segnumber+1]-pos)
        # print(self.waypoints[segnumber+1])
        # print(pos)

        ang_cur_goal = np.ones(clearance.shape)
        deviation = np.ones(clearance.shape)
        waypoint_distance = np.ones(clearance.shape)
        # ang_cur_way = np.ones(clearance.shape)
        for i in range(sampled_vel.shape[0]):
            curr_heading = sampled_vel[i]/LA.norm(sampled_vel[i])

            cosine_cur_goal = max(-1,
                                  min(np.dot(goal_heading, curr_heading), 1))
            ang_cur_goal[i] = np.arccos(cosine_cur_goal)
            # cosine_cur_way = max(-1, min(np.dot(way_heading, curr_heading), 1))
            # ang_cur_way[i] = np.arccos(cosine_cur_way)
            sim_pos=pos+sampled_vel[i]*self.dt
            # print('lines[segnum]',np.array([lines[segnum]]).shape[0])
            deviation[i],_=compute_pldis(sim_pos,np.array([self.lines[segnumber]]))
            waypoint_distance[i]=LA.norm(self.lines[segnumber,1]-sim_pos)
            # if ang_cur_way[i] > np.pi:
            #     ang_cur_way[i] -= 2*np.pi
            # if ang_cur_way[i] < -np.pi:
            #     ang_cur_way[i] += 2*np.pi
            # if ang_cur_goal[i] > np.pi:
            #     ang_cur_goal[i] -= 2*np.pi
            # if ang_cur_goal[i] < -np.pi:
            #     ang_cur_goal[i] += 2*np.pi
        # print("clearance",clearance)
        # print("ang_cur_way",ang_cur_way)
        # print('ang_cur_goal',ang_cur_goal)
        deviation=deviation-np.mean(deviation)
        waypoint_distance=waypoint_distance-np.mean(waypoint_distance)
        # print('deviation',np.std(6e6*deviation**2))
        # print('waypoint_distance',np.std(2e3*waypoint_distance))
        # print('mean deviation',np.mean(1e3*deviation))
        # print('ang_cur_way',np.std(10*ang_cur_way))
        # print('clearance',np.std(1e7/clearance))
        # print('ang_cur_goal',np.std(2*ang_cur_goal))
        # print('velocity',np.std(1.2/LA.norm(sampled_vel, axis=1)))
        print('clearance',clearance)
        cost = 1e7/clearance+2.5*ang_cur_goal+6e6*deviation**2+1.0/LA.norm(sampled_vel, axis=1)+2e3*waypoint_distance#+10*ang_cur_way
        # print('ang goal',ang_cur_goal.shape)
        # print('ang way',ang_cur_way.shape)
        # print(cost.shape)
        # raise NotImplementedError()
        pos_dot = sampled_vel[np.argmin(cost)]
        return pos_dot

    def get_yaw(self, vel):
        curr_heading = vel/LA.norm(vel)
        prev_yaw = self.state.rot[2]
        prev_heading = np.array([np.cos(prev_yaw), np.sin(prev_yaw)])
        # prev_heading=self.heading
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
        # print('yaw',self.yaw)
        return self.yaw, yawdot
def compute_pldis(point,lines):
    # print('lines',lines)
    assert(lines.shape[1]==2 and lines.shape[2]==3)
    distances=np.ones([lines.shape[0]])
    for i in range(lines.shape[0]):
        ap=point-lines[i,0]
        ab=lines[i,1]-lines[i,0]
        r=np.sum(ap*ab)/(LA.norm(ab)**2)
        # print('r',r)
        if r >1 or r==1:
            distances[i]=LA.norm(point-lines[i,1])
        elif r<0 or r==0:
            distances[i]=LA.norm(ap)
        else:
            distances[i]=np.sqrt(LA.norm(ap)**2-(np.sum(ap*ab)/(LA.norm(ab)))**2)
    # print('distances',distances)
    nearest_line=np.argmin(distances)
    return distances,nearest_line