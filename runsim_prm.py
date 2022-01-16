import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
from PathPlanning import RRTStar, Map, KinoRRTStar
from PathPlanning.prm_star import PRMStar
from TrajGen import Helix_waypoints, Circle_waypoints, Waypoint_insertion_trajOpt, Corridor_bounding_trajOpt
from Quadrotor import QuadSim, QuadSim_plan_traj_visual
from PathPlanning.maputils import *
import controller
import random_obstacle_map
import time

solution = "prm"

# create a figure
scale_factor = 0.02

fig = plt.figure()
ax = Axes3D.Axes3D(fig)
ax.set_xlim((0, 100*scale_factor))
ax.set_ylim((0, 100*scale_factor))
ax.set_zlim((0, 100*scale_factor))

# create map with obstacles 
bounds = np.array([0, 100]) # limits on map dimensions

# plan a path from start to goal
start = np.array([80, 20, 10])
goal = np.array([30, 80, 10])

mapobs, obstacles = random_obstacle_map.generate_map(bounds=bounds, density=2.5, height=50, start=start, goal=goal, path_resolution=0.5)
mapobs.plotobs(ax, scale=scale_factor) # plot obstacles
inflated_obs=map_inflate(obstacles,inf_dis=2)
mapobs_inf = Map(inflated_obs, bounds, dim=3)

# Informed RRT* and Minimized Snap
if solution == "rrt":
    # plan waypoints
    start_time = time.time()
    rrt = RRTStar(start=start, goal=goal,
              Map=mapobs_inf, max_iter=500,
              goal_sample_rate=0.1)
    waypoints, min_cost = rrt.plan()
    waypoints = scale_factor * waypoints  # scale the waypoints
    time_for_waypoints = np.round_(time.time() - start_time, decimals=2, out=None)
    rrt.draw_path(ax, waypoints) # plot the waypoints
    # generate collision free trajectory
    start_time = time.time()
    try:
        print("Obstacle avoidance using flying corridor")
        traj = Corridor_bounding_trajOpt(waypoints, mapobs, max_vel=1.0, gamma=1e6) # minisnap & flying corridor
    except:
        print("Obstacle avoidance using point inseration")
        traj = Waypoint_insertion_trajOpt(waypoints, mapobs, max_vel=1.0, gamma=1e6) # minisnap &  point inseration
    Tmax = traj.time_list[-1]
    des_state = traj.get_des_state
    time_for_trajectory = np.round_(time.time() - start_time, decimals=2, out=None)
    print(time_for_trajectory)
    # run simulation
    sim = QuadSim_plan_traj_visual(controller, des_state, Tmax)
    sim.run(ax)

# PRM* and Minimize Snap
if solution == "prm":
    # plan waypoints
    start_time = time.time()
    prm = PRMStar(start=start, goal=goal, Map=mapobs_inf, num_sample=500)
    waypoints, min_cost = prm.plan()
    waypoints = scale_factor * waypoints  # scale the waypoints
    # time_for_waypoints = np.round_(time.time() - start, decimals=2, out=None)
    # prm.draw_graph(ax, scale_factor) # plot ramdom road map
    prm.draw_path(ax, waypoints) # plot the waypoints
    # generate collision free trajectory
    start_time = time.time()
    try:
        print("Obstacle avoidance using flying corridor")
        traj = Corridor_bounding_trajOpt(waypoints, mapobs, max_vel=1.0, gamma=1e6) # obstacle avoidance using flying corridor
    except:
        print("Obstacle avoidance using point inseration")
        traj = Waypoint_insertion_trajOpt(waypoints, mapobs, max_vel=1.0, gamma=1e6) # obstacle avoidance using point inseration
    Tmax = traj.time_list[-1]
    des_state = traj.get_des_state
    time_for_trajectory = np.round_(time.time() - start_time, decimals=2, out=None)
    print("Time for planning ", time_for_trajectory)
    # run simulation
    sim = QuadSim_plan_traj_visual(controller, des_state, Tmax) # Init simulation
    sim.run(ax) # run simulation







