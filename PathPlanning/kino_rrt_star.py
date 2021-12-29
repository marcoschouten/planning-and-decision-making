from numpy import sqrt
from .kino_utils import *
from .rrt import *
import time
import cvxpy as cp
import numpy.linalg as LA

class KinoRRTStar(RRT):
    def __init__(self, start, goal, Map,
                 max_extend_length=10.0,
                 path_resolution=0.5,
                 goal_sample_rate=0.05,
                 max_iter=100):
        super().__init__(start, goal, Map, max_extend_length,
                         path_resolution, goal_sample_rate, max_iter)
        self.final_nodes = []
        self.start = Node_with_traj(start)
        self.goal = Node_with_traj(goal)
        for i in [self.start, self.goal]:
            i.vel = np.zeros(3)
            i.acc = np.zeros(3)

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
            # Discard new_node if it is too close to nearest_node
            if self.dist(nearest_node, new_node) < 1e-3:
                continue
            # Attach velocity to random node
            new_node.vel = self.get_random_vel(nearest_node, new_node)
            # If path between new_node and nearest node is not in collision
            if (not self.collision(nearest_node, new_node)):
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
        if np.random.rand() > self.goal_sample_rate:
            rnd = self.sample()
            while self.map.collision(rnd, rnd): #in case the sampled point is in obstacle
                rnd = self.sample()
            return Node_with_traj(rnd)
        else:
            return self.goal

    def sample(self, bounds=np.array([0,100])):
        # Sample random point inside boundaries
        lower, upper = bounds
        # Return a 3d array
        return lower + np.random.rand(3)*(upper - lower)
    
    def get_random_vel(self, from_node, to_node):
        '''
        Sample velocity close to the direction from from_node to to_node
        '''
        pos1 = from_node.p
        pos2 = to_node.p
        direc_pos = (pos2 - pos1) / LA.norm(pos2 - pos1)
        # vel_bounds = np.array([-5, 5])
        # vel = self.sample(bounds=vel_bounds)
        # direc_vel = vel / LA.norm(vel)
        # while ((direc_vel.T @ direc_pos) < 0.8):
        #     vel = self.sample(bounds=vel_bounds)
        #     direc_vel = vel / LA.norm(vel)
        if from_node.vel.all() == 0:
            direc_vel1 = direc_pos
        else:
            direc_vel1 = from_node.vel / LA.norm(from_node.vel)
        # TODO vel of goal point
        vel_bounds = np.array([-5, 5])
        vel = self.sample(bounds=vel_bounds)
        direc_vel2 = vel / LA.norm(vel)
        while ((direc_vel2.T @ direc_vel1) < 0.6 and (direc_vel2.T @ direc_pos) < 0.6):
            vel = self.sample(bounds=vel_bounds)
            direc_vel2 = vel / LA.norm(vel)
        return vel

    def collision(self, nearest_node, new_node):
        '''If collide, return True. Otherwise, return False'''
        traj_segment = self.steer(nearest_node, new_node)
        for t in np.linspace(0, traj_segment.T, 1000):
            pos = traj_segment.get_pos(t)
            if self.map.idx.count((*pos,)) != 0:
                print("collision happend: ", nearest_node, new_node)
                return True
        return False
    
    def steer(self, from_node, to_node, T_max=8):
        if str(from_node.p) in to_node.trajectories:
            return to_node.trajectories[str(from_node.p)]
        T = (self.dist(from_node, to_node) /
             self.dist(self.start, self.goal)) * T_max
        # if T < 1e-3:
        #     coeff = [np.zeros(6) for i in range(3)]
        #     cost = np.inf
        #     return Trajectory_segment(coeff, cost, T)
        # direc_vel1 = from_node.vel / LA.norm(from_node.vel)
        # direc_vel2 = to_node.vel / LA.norm(to_node.vel)
        # if ((direc_vel1.T @ direc_vel2) < 0.2):
        #     coeff = [np.zeros(6) for i in range(3)]
        #     cost = np.inf
        #     return Trajectory_segment(coeff, cost, T)
        coeff = []
        cost_val = 0
        for idx in range(3):
            # Refer: https://www.cvxpy.org/examples/basic/quadratic_program.html
            # Optimization variables: 5th order polynomial, 6 variables, higher order first
            c = cp.Variable(6)
            # Cost function: integrate snap**2 over period T
            Q = np.array([[8*T**3, 2.5*T**2, 0, 0, 0, 0],
                          [2.5*T**2, T, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]])
            cost = cp.quad_form(c, Q)
            # cost += np.array([5*T**4, 4*T**3, 3*T**2, 2*T, 0, 0]) @ c 
            # Constraints: boundary conditions (only position)
            A = np.array([[0, 0, 0, 0, 0, 1],
                          [T**5, T**4, T**3, T**2, T, 1],
                          [0, 0, 0, 0, 1, 0],
                          [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0]])
            b = np.array([from_node.p[idx],
                          to_node.p[idx],
                          from_node.vel[idx],
                          to_node.vel[idx]])
            constraints = [A @ c == b]
            # Solves the problem
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve()
            coeff_1d = c.value
            # Filp the sequence as required by polyder function
            coeff.append(np.flip(coeff_1d.flatten()))
            cost_val += problem.value
        # Store the trajectory segment inside to_node
        to_node.trajectories[str(from_node.p)] = Trajectory_segment(coeff, cost_val, T)
        return to_node.trajectories[str(from_node.p)]
    
    def add(self, new_node):
        near_nodes = self.near_nodes(new_node)
        # Connect the new node to the best parent in near_inds
        self.choose_parent(near_nodes, new_node)
        # add the new_node to tree
        self.tree.add(new_node)
        print("new node added:", new_node)
        # print("goal parent: ", self.goal.parent)
        # Rewire the nodes in the proximity of new_node if it improves their costs
        # self.rewire(new_node, near_nodes)
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
        traj_segment = self.steer(from_node, to_node)
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
            print("----------------------------------------")
            print("seg stored in: ", node)
            print("coeff: ", seg.coeff)
            print("seg starts from: ", seg.get_pos(0))
            print("seg ends at: ", seg.get_pos(seg.T))
            print("----------------------------------------")
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
    
    