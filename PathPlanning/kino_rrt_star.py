from numpy import sqrt

from Quadrotor.params import H
from .kino_utils import *
from .rrt import *
import time
import cvxpy as cp
import numpy.linalg as LA
from .sampleutils import InformedSampler


class KinoRRTStar(RRT):
    def __init__(self, start, goal, Map, scale_factor,
                 max_extend_length=10.0,
                 path_resolution=0.5,
                 goal_sample_rate=0.1,
                 max_iter=100):
        super().__init__(start, goal, Map, max_extend_length,
                         path_resolution, goal_sample_rate, max_iter)
        self.final_nodes = []
        self.scale_factor = scale_factor
        self.start = Node_with_traj(start)
        self.goal = Node_with_traj(goal)
        self.start.cost = 0
        for i in [self.start, self.goal]:
            i.vel = np.zeros(3)
            i.acc = np.zeros(3)
            i.jerk = np.zeros(3)

    def plan(self):
        """Plans the path from start to goal while avoiding obstacles"""
        self.start.cost = 0
        self.tree.add(self.start)
        for i in range(self.max_iter):
            self.start_time = time.time()
            # Generate a random node (rnd_node)
            new_node = self.get_random_node()
            # Get nearest node
            nearest_node = self.tree.nearest(new_node)
            if new_node is not self.goal:
                # Attach velocity, acceleration to random node
                self.init_new_node(nearest_node, new_node)
            # If path between new_node and nearest node is not in collision
            if not self.collision(nearest_node, new_node):
                # choose parent + rewiring
                self.add(new_node)
        # Return traj if it exists
        if not self.goal.parent:
            raise ValueError('No trajectory found!')
        else:
            traj = trajGenerator(self.final_traj())
            path = self.final_path()
        return traj, path

    def get_random_node(self):
        """
        Sample random node inside bounds or sample goal point.
        Each node is attached with a sampled velocity.
        """
        # if self.goal.parent:
        #     rnd = self.sample(bounds=np.array([-15, 15])) + self.goal.p
        #     while self.map.collision(rnd, rnd): #in case the sampled point is in obstacle
        #         rnd = self.sample(bounds=np.array([-15, 15])) + self.goal.p
        #     return Node_with_traj(rnd)
        if np.random.rand() > self.goal_sample_rate:
            # rnd = self.sample(bounds=self.map.bounds)
            rnd = self.get_random_pos(bounds=self.map.bounds)
            # in case the sampled point is in obstacle
            while self.map.collision(rnd, rnd):
                rnd = self.get_random_pos(bounds=self.map.bounds)
            return Node_with_traj(rnd)
        else:
            return self.goal

    def init_new_node(self, nearest_node, new_node):
        '''Method 1, by optimizeing'''
        # trajectory_segment = self.steer(nearest_node, new_node, constr_option = "no_constr")
        # # Update velocity and acceleration of new_node
        # _, vel, acc, jerk = trajectory_segment.get_des_state_seg(trajectory_segment.T)
        # new_node.vel = vel
        # new_node.acc = acc
        # new_node.jerk = jerk
        '''Method 2, by sampling'''
        new_node.vel = self.get_random_vel(nearest_node, new_node)
        new_node.acc = self.get_random_acc(nearest_node, new_node)
        new_node.jerk = self.get_random_jerk(nearest_node, new_node)
        '''Method 3, combined method (sampling + optimization'''
        # new_node.vel = self.get_random_vel(nearest_node, new_node)
        # new_node.acc = self.get_random_acc(nearest_node, new_node)
        # trajectory_segment = self.steer(nearest_node, new_node, constr_option = "partially_constr")
        # # Update velocity and acceleration of new_node
        # _, vel, acc, jerk = trajectory_segment.get_des_state_seg(trajectory_segment.T)
        # new_node.jerk = jerk

    def sample(self, bounds):
        # Sample random point inside boundaries
        lower, upper = bounds
        # Return a 3d array
        return lower + np.random.rand(3)*(upper - lower)

    def get_random_pos(self, bounds):
        # Sample random point inside boundaries
        lower, upper = bounds
        diff = upper - lower
        scale = np.array([diff, diff, diff*0.6])
        point = lower + np.random.rand(3)*scale
        # Return a 3d array
        return point
    
    def get_random_vel(self, from_node, to_node):
        '''
        Sample velocity close to the direction from from_node to to_node
        '''
        pos1 = from_node.p
        pos2 = to_node.p
        direc_pos = (pos2 - pos1) / LA.norm(pos2 - pos1)
        vel_bounds = np.array([-10, 10]) * self.scale_factor
        vel = self.sample(bounds=vel_bounds)
        direc_vel = vel / LA.norm(vel)
        while ((direc_vel.T @ direc_pos) < 0.5):
            vel = self.sample(bounds=vel_bounds)
            direc_vel = vel / LA.norm(vel)
        return vel

    def get_random_acc(self, from_node, to_node):
        acc_bounds = np.array([-10, 10]) * self.scale_factor
        acc = self.sample(bounds=acc_bounds)
        return acc

    def get_random_jerk(self, from_node, to_node):
        jerk_bounds = np.array([-10, 10]) * self.scale_factor
        jerk = self.sample(bounds=jerk_bounds)
        return jerk

    def collision(self, nearest_node, new_node):
        '''If collide, return True. Otherwise, return False'''
        traj_segment = self.steer(nearest_node, new_node)
        for t in np.linspace(0, traj_segment.T, 1000):
            pos = traj_segment.get_pos(t)
            if self.map.idx.count((*pos,)) != 0:
                if new_node == self.goal:
                    print("collision happend: ", nearest_node, new_node)
                return True
        return False

    def steer(self, from_node, to_node, T_max=1.5, constr_option="all_constr"):
        if str(from_node.p) in to_node.trajectories:
            return to_node.trajectories[str(from_node.p)]
        T = (self.dist(from_node, to_node) /
             self.dist(self.start, self.goal)) * T_max
        coeff = []
        cost_val = 0
        for idx in range(3):
            # Refer: https://www.cvxpy.org/examples/basic/quadratic_program.html
            # Optimization variables: 9th order polynomial, 10 variables, lowest order first
            c = cp.Variable(poly_order)
            # Cost function: minimize snap**2
            cost = 0
            Q = Hessian(T) * 1e-20
            cost += cp.quad_form(c, Q)
            # Constraints:
            constraints = []
            if constr_option == "partially_constr":
                A = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, T**1, T**2, T**3, T**4, T **
                                  5, T**6, T**7, T**8, T**9],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
                              [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4,
                               6*T**5, 7*T**6, 8*T**7, 9*T**8],
                              [0, 0, 2, 6*T, 12*T**2, 20*T**3, 30*T**4, 42*T**5, 56*T**6, 72*T**7]])
                b = np.array([from_node.p[idx],
                              to_node.p[idx],
                              from_node.vel[idx],
                              from_node.acc[idx],
                              from_node.jerk[idx],
                              to_node.vel[idx],
                              to_node.acc[idx]])
                constraints.append(A @ c == b)  # boundary conditions
                G = np.array(
                    [[0, 0, 0, 6, 24*T**1, 60*T**2, 120*T**3, 210*T**4, 336*T**5, 504*T**6]])
                h = np.array([5])
                constraints.append(G @ c <= h)  # to_node vel, acc
                constraints.append(G @ c >= -h)
            elif constr_option == "no_constr":
                A = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, T**1, T**2, T**3, T**4, T**5, T**6, T**7, T**8, T**9],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 6, 0, 0, 0, 0, 0, 0]])
                b = np.array([from_node.p[idx],
                              to_node.p[idx],
                              from_node.vel[idx],
                              from_node.acc[idx],
                              from_node.jerk[idx]])
                constraints.append(A @ c == b)  # boundary conditions
                # G = np.array([[0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4, 6*T**5, 7*T**6, 8*T**7, 9*T**8],
                #               [0, 0, 2, 6*T, 12*T**2, 20*T**3, 30*T**4, 42*T**5, 56*T**6, 72*T**7]])
                # h = np.array([5,
                #               1])*self.scale_factor
                # constraints.append(G @ c <= h)  # to_node vel, acc
                # constraints.append(G @ c >= -h)
            elif constr_option == "all_constr":
                A = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, T**1, T**2, T**3, T**4, T **
                                  5, T**6, T**7, T**8, T**9],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4,
                               6*T**5, 7*T**6, 8*T**7, 9*T**8],
                              [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 6*T, 12*T**2, 20*T**3, 30 *
                               T**4, 42*T**5, 56*T**6, 72*T**7],
                              [0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 6, 24*T**1, 60*T**2, 120*T**3, 210*T**4, 336*T**5, 504*T**6]])
                b = np.array([from_node.p[idx],
                              to_node.p[idx],
                              from_node.vel[idx],
                              to_node.vel[idx],
                              from_node.acc[idx],
                              to_node.acc[idx],
                              from_node.jerk[idx],
                              to_node.jerk[idx]])
                constraints.append(A @ c == b)  # boundary conditions
            # Solves the problem
            problem = cp.Problem(cp.Minimize(cost), constraints)
            try:
                problem.solve()
            except:
                coeff = [np.zeros(poly_order) for i in range(3)]
                cost = np.inf
                trajectory_segment = Trajectory_segment(coeff, cost_val, T)
                to_node.trajectories[str(from_node.p)] = trajectory_segment
                return trajectory_segment
            coeff_1d = c.value
            if coeff_1d is None:
                coeff = [np.zeros(poly_order) for i in range(3)]
                cost = np.inf
                trajectory_segment = Trajectory_segment(coeff, cost, T)
                to_node.trajectories[str(from_node.p)] = trajectory_segment
                return trajectory_segment
            coeff.append(coeff_1d.flatten())
            cost_val += problem.value
        trajectory_segment = Trajectory_segment(coeff, cost_val, T)
        # Store the trajectory segment inside to_node
        to_node.trajectories[str(from_node.p)] = trajectory_segment
        return trajectory_segment

    def add(self, new_node):
        near_nodes = self.near_nodes(new_node)
        # if new_node == self.goal:
        #     print("first", list(near_nodes))
        # Connect the new node to the best parent in near_inds
        self.choose_parent(near_nodes, new_node)
        if new_node.parent == None:
            return
        # add the new_node to tree
        self.tree.add(new_node)
        print("new node added:", new_node)
        self.rewire(new_node, near_nodes)
        # # check if it is in close proximity to the goal
        # if self.dist(new_node, self.goal) <= self.max_extend_length:
        #     # Connection between node and goal needs to be collision free
        #     if not self.collision(self.goal, new_node):
        #         # add to final nodes if in goal region
        #         self.final_nodes.append(new_node)
        # # set best final node and min_cost
        # self.choose_parent(self.final_nodes, self.goal)

    def choose_parent(self, parents, new_node):
        """
        Set node.parent to the lowest resulting cost parent in parents and
        node.cost to the corresponding minimal cost
        """
        # Go through all near nodes and evaluate them as potential parent nodes
        for parent in parents:
            # checking whether a connection would result in a collision
            if not self.collision(parent, new_node):
                # evaluating the cost of the new_node if it had that near node as a parent
                cost = self.new_cost(parent, new_node)
                # picking the parent resulting in the lowest cost and updating the cost of the new_node to the minimum cost.
                if cost < new_node.cost:
                    new_node.parent = parent
                    new_node.cost = cost

    def rewire(self, new_node, near_nodes):
        """Rewire near nodes to new_node if this will result in a lower cost"""
        # Go through all near nodes and check whether rewiring them to the new_node is useful
        for node in near_nodes:
            self.choose_parent([new_node], node)
        self.propagate_cost_to_leaves(new_node)

    def near_nodes(self, node):
        """Find the nodes in close proximity to given node"""
        return self.tree.k_nearest(node, 5)

    def new_cost(self, from_node, to_node):
        """to_node's new cost if from_node were the parent"""
        traj_segment = to_node.trajectories[str(from_node.p)]
        return from_node.cost + traj_segment.cost

    def propagate_cost_to_leaves(self, parent_node):
        """Recursively update the cost of the nodes"""
        for node in self.tree.all():
            if node.parent == parent_node:
                node.cost = self.new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def final_traj(self):
        trajectory_segments = []
        node = self.goal
        if (node.p == node.parent.p).all():
            node = node.parent
        while node.parent:
            seg = node.trajectories[str(node.parent.p)]
            trajectory_segments.append(seg)
            # print("----------------------------------------")
            # print("seg stored in: ", node)
            # print("coeff: ", seg.coeff)
            # print("seg starts from (pos vel acc jerk): ",
            #       seg.get_des_state_seg(0))
            # print("seg ends at: ", seg.get_des_state_seg(seg.T))
            # print("----------------------------------------")
            node = node.parent
        # raise ValueError("planned finished")
        return trajectory_segments

    def draw_path(self, ax, path):
        '''draw the path if available'''
        if path is None:
            print("path not available")
        else:
            ax.plot(*np.array(path).T, '-',
                    color='b', zorder=5)
