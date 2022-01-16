## Quadrotor Planning (RO47005 Project)

### Simulation environment
This project is built based on this [simulation environment](https://github.com/Bharath2/Quadrotor-Simulation).  The already built functionalities in this environment are:

- Informed-RRT* path planner.
- Minisnap trajectory generation (solved by closed-form solution) without collision avoidance .
- A non-linear controller for path following.
- A quadrotor simulator.

### Our contribution
The **newly added** functionalities by us are:

- PRM* path planner.
- Collision free minisnap trajectory optimization using two methods, namely corridor bounding method and waypoint insertion method.
- Kinodynamics RRT* using polynomial as steering function.
- Velocity Obstacles (currently under construction).
- Dynamic window approach (not used due to poor performance).
- Random generated map for simulation.

### Dependencies
numpy, matplotlib, rtree-linux, scipy, CVXPY, CVXOPT

### Usage
#### K-PRM* with Minisnap and Corridor Bounding Method
```
python runsim_prm.py
```
<div align=center>
<img width="400" height="350" src="https://github.com/MarcoSchouten/Planning_Project/blob/main/imgs/k_prm.gif"/>
</div>

#### Kinodynamics RRT*
```
python runsim_kinorrt.py
```
<div align=center>
<img width="400" height="350" src="https://github.com/MarcoSchouten/Planning_Project/blob/main/imgs/kino_rrt.gif"/>
</div>

#### Velocity Obstacle (still under construction)
```
python Quadrotor-Simulation/runsim.py
```
<div align=center>
<img width="400" height="350" src="https://github.com/MarcoSchouten/Planning_Project/blob/main/imgs/vo.gif"/>
</div>

### Future work
- Trajectory optimization of kinodynamics RRT*
- Using B-spline for collision avoidance
- Finer adjustment for velocity obstacle 
- Try to implement model predictive control (MPC)
