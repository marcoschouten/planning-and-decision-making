import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
from PathPlanning import RRTStar, Map, KinoRRTStar
from PathPlanning.prm_star import PRMStar
from TrajGen import trajOpt, Helix_waypoints, Circle_waypoints, trajGenerator, Bs_trajOpt
from Quadrotor import QuadSim, QuadSim_plan_traj_visual
from PathPlanning.maputils import *
import controller
import random_obstacle_map
import time
np.random.seed(8)

solution = "prm"

# create a figure
scale_factor = 0.02
if solution == "kino_rrt":
    scale_factor = 1
fig = plt.figure()
ax = Axes3D.Axes3D(fig)
ax.set_xlim((0, 100*scale_factor))
ax.set_ylim((0, 100*scale_factor))
ax.set_zlim((0, 100*scale_factor))

# create map with obstacles 
bounds = np.array([0, 100]) # limits on map dimensions
# # 3D boxes   lx, ly, lz, hx, hy, hz
# obstacles = [[-5, 25, 0, 20, 35, 60],
#              [30, 25, 0, 55, 35, 100],
#              [45, 35, 0, 55, 60, 60],
#              [45, 75, 0, 55, 85, 100],
#              [70, 50, 0, 80, 80, 100]]
# mapobs = Map(obstacles, bounds, dim=3) # create map with obstacles
mapobs, obstacles = random_obstacle_map.generate_map(bounds)
mapobs.plotobs(ax, scale=scale_factor) # plot obstacles
inflated_obs=map_inflate(obstacles,inf_dis=2)
mapobs_inf = Map(inflated_obs, bounds, dim=3)
# plan a path from start to goal
start = np.array([80, 20, 10])
goal = np.array([30, 80, 50])

# Informed RRT* and Minimized Snap
if solution == "rrt":
    # plan waypoints
    start = time.time()
    rrt = RRTStar(start=start, goal=goal,
              Map=mapobs_inf, max_iter=500,
              goal_sample_rate=0.1)
    waypoints, min_cost = rrt.plan()
    waypoints = scale_factor * waypoints  # scale the waypoints
    time_for_waypoints = np.round_(time.time() - start, decimals=2, out=None)
    rrt.draw_path(ax, waypoints) # plot the waypoints
    # generate collision free trajectory
    start = time.time()
    try:
        print("Obstacle avoidance using flying corridor")
        traj = Bs_trajOpt(waypoints, mapobs, max_vel=1.0, gamma=1e6) # minisnap & flying corridor
    except:
        print("Obstacle avoidance using point inseration")
        traj = trajOpt(waypoints, mapobs, max_vel=1.0, gamma=1e6) # minisnap &  point inseration
    Tmax = traj.time_list[-1]
    des_state = traj.get_des_state
    time_for_trajectory = np.round_(time.time() - start, decimals=2, out=None)
    # run simulation
    sim = QuadSim_plan_traj_visual(controller, des_state, Tmax)
    sim.run(ax)

# PRM* and Minimize Snap
if solution == "prm":
    # plan waypoints
    start = time.time()
    prm = PRMStar(start=start, goal=goal, Map=mapobs_inf)
    waypoints, min_cost = prm.plan()
    waypoints = scale_factor * waypoints  # scale the waypoints
    time_for_waypoints = np.round_(time.time() - start, decimals=2, out=None)
    # prm.draw_graph(ax, scale_factor) # plot ramdom road map
    prm.draw_path(ax, waypoints) # plot the waypoints
    # generate collision free trajectory
    start = time.time()
    try:
        print("Obstacle avoidance using flying corridor")
        traj = Bs_trajOpt(waypoints, mapobs, max_vel=1.0, gamma=1e6) # obstacle avoidance using flying corridor
    except:
        print("Obstacle avoidance using point inseration")
        traj = trajOpt(waypoints, mapobs, max_vel=1.0, gamma=1e6) # obstacle avoidance using point inseration
    Tmax = traj.time_list[-1]
    des_state = traj.get_des_state
    time_for_trajectory = np.round_(time.time() - start, decimals=2, out=None)
    # run simulation
    sim = QuadSim_plan_traj_visual(controller, des_state, Tmax) # Init simulation
    sim.run(ax) # run simulation

# Kinodynamics RRT*
if solution == "kino_rrt":
    # generate collision free trajectory
    start = time.time()
    kino_rrt = KinoRRTStar(start=start, goal=goal,
                Map=mapobs, max_iter=30)
    traj, waypoints = kino_rrt.plan()
    print("waypoints: ", waypoints)
    kino_rrt.draw_path(ax, waypoints) # draw lines connecting waypoints
    Tmax = traj.get_Tmax()
    des_state = traj.get_des_state
    # run simulation
    sim = QuadSim_plan_traj_visual(controller, des_state, Tmax) # Init simulation
    sim.run(ax) # run simulation






