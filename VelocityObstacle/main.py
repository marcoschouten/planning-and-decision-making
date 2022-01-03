import sys


from RVO import  VO_update, compute_V_des_marco
from vis import visualize_traj_dynamic
import numpy as np


#------------------------------
#define workspace model
ws_model = dict()
#robot radius
ws_model['robot_radius'] = 0.2
#circular obstacles, format [x,y,rad]
# no obstacles
ws_model['circular_obstacles'] = []
# with obstacles
# ws_model['circular_obstacles'] = [[-0.3, 2.5, 0.3], [1.5, 2.5, 0.3], [3.3, 2.5, 0.3], [5.1, 2.5, 0.3]]
#rectangular boundary, format [x,y,width/2,heigth/2]
ws_model['boundary'] = [] 

#------------------------------
#initialization for robot 
# position of [x,y]
X = np.array([[2,0] ,[2,5]]).astype(float)

# velocity of [vx,vy]
V = np.array([[0,0] for i in range(len(X))]).astype(float)
# maximal velocity norm
V_max = np.array([1.0 for i in range(len(X))]).astype(float)
# goal of [x,y]
goal = np.array([[2,5],[2,0]]).astype(float)

#------------------------------
#simulation setup
# total simulation time (s)
total_time = 10
# simulation step
step = 0.2

#------------------------------
#simulation starts
t = 0
exit = True


picNum = 0

while t*step < total_time and exit:
    # compute desired vel to goal
    V_des = compute_V_des_marco(X, goal, V_max)


    # compute the optimal vel to avoid collision
    # velocity of 1st robot
    V[0] = VO_update(X, V_des, V, ws_model)

    # velocity of 2nd robot
    V[1] = V_des[1]

    # update position
    for i in range(len(X)):
        if (np.linalg.norm(goal[0] - X[0]) > 0.5):
            X[i] += V[i]*step
        else:
            print("completion Time {}".format(t*step ))
            exit = False
        # X[i][0] += V[i][0]*step # horizontal component
        # X[i][1] += V[i][1]*step # vertical component
    #----------------------------------------
    # visualization
    # if t%10 == 0:
    # visualize_traj_dynamic(ws_model, X, V, goal, time=t*step, name='data/snap%s.png'%str(np.round(t*step,1)))
    visualize_traj_dynamic(ws_model, X, V, goal, time=t * step, name='data/snap%s.png' % str(picNum).zfill(3))

    picNum += 1
    #visualize_traj_dynamic(ws_model, X, V, goal, time=t*step, name='data/snap%s.png'%str(t/10))
    t += 1
    
