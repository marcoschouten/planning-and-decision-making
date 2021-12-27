import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

from PathPlanning import RRTStar, Map, KinoRRTStar
from PathPlanning.prm_star import PRMStar
from TrajGen import trajGenerator, Helix_waypoints, Circle_waypoints
from Quadrotor import QuadSim, QuadSim_plan_traj_visual
import controller
np.random.seed(8)

# create a figure
fig = plt.figure()
ax = Axes3D.Axes3D(fig)
ax.set_xlim((0, 100))
ax.set_ylim((0, 100))
ax.set_zlim((0, 100))
scale_factor = 1


# create map with obstacles 
# 3D boxes   lx, ly, lz, hx, hy, hz
obstacles = [[-5, 25, 0, 20, 35, 60],
             [30, 25, 0, 55, 35, 100],
             [45, 35, 0, 55, 60, 60],
             [45, 75, 0, 55, 85, 100],
             [-5, 65, 0, 30, 70, 100],
             [70, 50, 0, 80, 80, 100]]
obstacles = [[-5, 25, 0, 20, 35, 60],
             [30, 25, 0, 55, 35, 100],
             [45, 35, 0, 55, 60, 60]]
bounds = np.array([0, 100]) # limits on map dimensions
mapobs = Map(obstacles, bounds, dim=3) # create map with obstacles
mapobs.plotobs(ax, scale=scale_factor) # plot obstacles


# plan a path from start to goal
start = np.array([80, 20, 10])
goal = np.array([30, 80, 50])

solution = "kino_rrt"

# Informed RRT* and Minimized Snap
if solution == "rrt":
    rrt = RRTStar(start=start, goal=goal,
              Map=mapobs, max_iter=500,
              goal_sample_rate=0.1)
    waypoints, min_cost = rrt.plan()
    waypoints = scale_factor * waypoints  # scale the waypoints
    traj = trajGenerator(waypoints, max_vel=10, gamma=1e6) # Generate trajectory
    Tmax = traj.TS[-1]
    des_state = traj.get_des_state
    sim = QuadSim_plan_traj_visual(controller, des_state, Tmax) # Init simulation
    rrt.draw_path(ax, waypoints) # plot the waypoints
    sim.run(ax) # run simulation

# PRM* and Minimize Snap
if solution == "prm":
    prm = PRMStar(start=start, goal=goal, Map=mapobs)
    waypoints, min_cost = prm.plan()
    waypoints = scale_factor * waypoints  # scale the waypoints
    traj = trajGenerator(waypoints, max_vel=10, gamma=1e6) # Generate trajectory
    Tmax = traj.TS[-1]
    des_state = traj.get_des_state
    sim = QuadSim_plan_traj_visual(controller, des_state, Tmax) # Init simulation
    # prm.draw_graph(ax, scale_factor) # plot ramdom road map
    prm.draw_path(ax, waypoints) # plot the waypoints
    sim.run(ax) # run simulation

# Kinodynamics RRT*
if solution == "kino_rrt":
    kino_rrt = KinoRRTStar(start=start, goal=goal,
                Map=mapobs, max_iter=200)
    traj, waypoints = kino_rrt.plan()
    print("waypoints: ", waypoints)
    Tmax = traj.get_Tmax()
    des_state = traj.get_des_state
    sim = QuadSim_plan_traj_visual(controller, des_state, Tmax) # Init simulation
    kino_rrt.draw_path(ax, waypoints)
    sim.run(ax) # run simulation






