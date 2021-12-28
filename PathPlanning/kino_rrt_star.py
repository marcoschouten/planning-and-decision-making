from numpy import sqrt
from .kino_utils import *
from .rrt import *
import time

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
        # for i in [self.start, self.goal]:
        #     i.vel = np.zeros(3)
        #     i.acc = np.zeros(3)

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

    def add(self, new_node):
        near_nodes = self.near_nodes(new_node)
        # Connect the new node to the best parent in near_inds
        self.choose_parent(near_nodes, new_node)
        # add the new_node to tree
        self.tree.add(new_node)
        print("new node added:", new_node)
        # # Rewire the nodes in the proximity of new_node if it improves their costs
        # self.rewire(new_node, near_nodes)
        # # check if it is in close proximity to the goal
        # if self.dist(new_node, self.goal) <= self.max_extend_length:
        #     # Connection between node and goal needs to be collision free
        #     if not self.collision(self.goal, new_node):
        #         # add to final nodes if in goal region
        #         self.final_nodes.append(new_node)
        # # set best final node and min_cost
        # self.choose_parent(self.final_nodes, self.goal)

    def get_random_node(self):
        """
        Sample random node inside bounds or sample goal point.
        Each node is attached with a sampled velocity.
        """
        if np.random.rand() > self.goal_sample_rate:
            rnd = self.sample()
            # while self.map.collision(rnd, rnd): #in case the sampled point is in obstacle
            #     rnd = self.sample()
            return Node_with_traj(rnd)
        else:
            return self.goal

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

    def steer(self, from_node, to_node, T_max=10):
        if str(from_node.p) in to_node.trajectories:
            return to_node.trajectories[str(from_node.p)]
        T = (self.dist(from_node, to_node) /
             self.dist(self.start, self.goal)) * T_max
        if T < 1e-3:
            coeff = [np.zeros(6) for i in range(3)]
            cost = np.inf
            return Trajectory_segment(coeff, cost, T)
        from_node.vel, to_node.vel = self.sample_vel(from_node, to_node)
        coeff = []
        cost = 0
        for idx in range(3):
            # naively solve 5th order polyder
            A = np.array([[0, 0, 0, 0, 0, 1],
                          [T**5, T**4, T**3, T**2, T, 1],
                          [0, 0, 0, 0, 1, 0],
                          [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
                          [0, 0, 0, 2, 0, 0],
                          [20*T**3, 12*T**2, 6*T, 2, 0, 0]])
            b = np.array([from_node.p[idx],
                          to_node.p[idx],
                          from_node.vel[idx],
                          to_node.vel[idx],
                          from_node.acc[idx],
                          to_node.acc[idx]]).reshape(-1, 1)

            coeff_1d = np.linalg.solve(A, b)
            coeff.append(np.flip(coeff_1d.flatten()))
            # cost: integrate snap**2 over period T
            c4 = coeff_1d[1]
            c5 = coeff_1d[0]
            cost += (8*c5**2*T**3 + 5*c4*c5*T**2 + c4**2*T)
        # Store the trajectory segment inside to_node
        to_node.trajectories[str(from_node.p)] = Trajectory_segment(coeff, cost, T)
        return to_node.trajectories[str(from_node.p)]

    def sample_vel(self, from_node, to_node):
        """Sample velcity in a cone"""
        pos1 = from_node.p
        pos2 = to_node.p
        direc_pos = (pos2 - pos1) / np.sqrt(np.sum((pos2 - pos1)**2))
        vel_list = []
        for i in range(2):
            vel = self.sample(bounds=np.array([-5,5]))
            direc_vel = vel / np.sqrt(np.sum((vel)**2))
            while ((direc_vel.T @ direc_pos) < 0.8):
                vel = self.sample(bounds=np.array([-5,5]))
                direc_vel = vel / np.sqrt(np.sum((vel)**2))
            vel_list.append(vel)
        return vel_list
    
    def sample(self, bounds=np.array([0,100])):
        # Sample random point inside boundaries
        lower, upper = bounds
        # Return a 3d array
        return lower + np.random.rand(3)*(upper - lower)
    
    def collision(self, nearest_node, new_node):
        '''If collide, return True. Otherwise, return False'''
        traj_segment = self.steer(nearest_node, new_node)
        for t in np.linspace(0, traj_segment.T, 1000):
            pos = traj_segment.get_pos(t)
            if self.map.idx.count((*pos,)) != 0:
                print("collision happend: ", nearest_node, new_node)
                return True
        return False

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
            print("\n----------------------------------------")
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