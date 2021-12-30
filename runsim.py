import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

from PathPlanning import RRTStar, Map, KinoRRTStar
from PathPlanning.prm_star import PRMStar
from TrajGen import trajOpt, Helix_waypoints, Circle_waypoints, trajGenerator, Bs_trajOpt
from Quadrotor import QuadSim, QuadSim_plan_traj_visual
from PathPlanning.maputils import *
import controller
np.random.seed(8)

# create a figure
scale_factor = 0.02
fig = plt.figure()
ax = Axes3D.Axes3D(fig)
ax.set_xlim((0, 100*scale_factor))
ax.set_ylim((0, 100*scale_factor))
ax.set_zlim((0, 100*scale_factor))



# create map with obstacles 
# 3D boxes   lx, ly, lz, hx, hy, hz
obstacles = [[-5, 25, 0, 20, 35, 60],
             [30, 25, 0, 55, 35, 100],
             [45, 35, 0, 55, 60, 60],
             [45, 75, 0, 55, 85, 100],
             [70, 50, 0, 80, 80, 100]]

bounds = np.array([0, 100]) # limits on map dimensions
mapobs = Map(obstacles, bounds, dim=3) # create map with obstacles
mapobs.plotobs(ax, scale=scale_factor) # plot obstacles
inflated_obs=map_inflate(obstacles,inf_dis=2)
mapobs_inf = Map(inflated_obs, bounds, dim=3)
# plan a path from start to goal
start = np.array([80, 20, 10])
goal = np.array([30, 80, 50])

solution = "prm"

# Informed RRT* and Minimized Snap
if solution == "rrt":
    rrt = RRTStar(start=start, goal=goal,
              Map=mapobs_inf, max_iter=500,
              goal_sample_rate=0.1)
    waypoints, min_cost = rrt.plan()
    waypoints = scale_factor * waypoints  # scale the waypoints
    traj = Bs_trajOpt(waypoints, mapobs, max_vel=1.5, gamma=1e6) # Generate trajectory
    Tmax = traj.time_list[-1]
    des_state = traj.get_des_state
    sim = QuadSim_plan_traj_visual(controller, des_state, Tmax) # Init simulation
    rrt.draw_path(ax, waypoints) # plot the waypoints
    sim.run(ax) # run simulation

# PRM* and Minimize Snap
if solution == "prm":
    prm = PRMStar(start=start, goal=goal, Map=mapobs_inf)
    waypoints, min_cost = prm.plan()
    waypoints = scale_factor * waypoints  # scale the waypoints
    traj = Bs_trajOpt(waypoints, mapobs, max_vel=1.5, gamma=1e6) # Generate trajectory
    Tmax = traj.time_list[-1]
    des_state = traj.get_des_state
    sim = QuadSim_plan_traj_visual(controller, des_state, Tmax) # Init simulation
    # prm.draw_graph(ax, scale_factor) # plot ramdom road map
    prm.draw_path(ax, waypoints) # plot the waypoints
    sim.run(ax) # run simulation

# Kinodynamics RRT*
if solution == "kino_rrt":
    kino_rrt = KinoRRTStar(start=start, goal=goal,
                Map=mapobs, max_iter=100)
    traj, waypoints = kino_rrt.plan()
    print("waypoints: ", waypoints)
    Tmax = traj.get_Tmax()
    des_state = traj.get_des_state
    sim = QuadSim_plan_traj_visual(controller, des_state, Tmax) # Init simulation
    kino_rrt.draw_path(ax, waypoints)
    sim.run(ax) # run simulation






