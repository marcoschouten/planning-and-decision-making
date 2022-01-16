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

scale_factor = 0.02

start = np.array([80, 20, 10]) * scale_factor
goal = np.array([30, 80, 10]) * scale_factor

# create a figure
fig = plt.figure()
ax = Axes3D.Axes3D(fig)
ax.set_xlim((0, 100*scale_factor))
ax.set_ylim((0, 100*scale_factor))
ax.set_zlim((0, 100*scale_factor))

# create map with obstacles 
bounds = np.array([0, 100]) * scale_factor # limits on map dimensions
bounds = bounds.astype(np.int)
# # 3D boxes   lx, ly, lz, hx, hy, hz
# obstacles = [[-5, 25, 0, 20, 35, 60],
#              [30, 25, 0, 55, 35, 100],
#              [45, 35, 0, 55, 60, 60],
#              [45, 75, 0, 55, 85, 100],
#              [70, 50, 0, 80, 80, 100]]

# for i in range(len(obstacles)):
#     for j in range(len(obstacles[i])):
#         obstacles[i][j] *= scale_factor

# mapobs = Map(obstacles, bounds, dim=3, path_resolution=0.5*scale_factor) # create map with obstacles
# mapobs, obstacles = random_obstacle_map.generate_map(bounds)

mapobs, obstacles = random_obstacle_map.generate_map(bounds=bounds, density=2.5, height=50, start=start, goal=goal, path_resolution=0.5*scale_factor)
mapobs.plotobs(ax, scale=1) # plot obstacles

# Kinodynamics RRT*
# generate collision free trajectory
start_time = time.time()
kino_rrt = KinoRRTStar(start=start, goal=goal,
            Map=mapobs, scale_factor=scale_factor, max_iter=50)
traj, waypoints = kino_rrt.plan()
print("waypoints: ", waypoints)
kino_rrt.draw_path(ax, waypoints) # draw lines connecting waypoints
Tmax = traj.get_Tmax()
des_state = traj.get_des_state
time_for_trajectory = np.round_(time.time() - start_time, decimals=2, out=None)

print("time_for_run: ", Tmax)
print("time_for_planning: ", time_for_trajectory)
# run simulation
sim = QuadSim_plan_traj_visual(controller, des_state, Tmax) # Init simulation
sim.run(ax) # run simulation






