import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
import networkx as nx
import copy
from .rrtutils import *
from matplotlib.collections import LineCollection


class PRMStar():
    def __init__(self, start, goal, Map,
                 num_sample=200):
        self.start = Node(start)
        self.goal = Node(goal)
        self.map = Map
        self.num_sample = num_sample
        self.dim = start.shape[0]
        self.graph = nx.Graph()
        self.tree = Rtree(self.dim)

    def build_graph(self):
        '''build graph for multi-quary'''
        # Add nodes by sampling
        for i in range(self.num_sample):
            new_node = Node(self.sample())
            if (self.map.collision(new_node.p, new_node.p) != True):
                self.graph.add_node(new_node)
                self.tree.add(new_node)
        # Connect near neighbors
        for node in self.tree.all():
            self.connect_neighbors(node, self.graph, self.tree)

    def plan(self):
        self.build_graph()
        # TODO add deep copy, note copy.deepcopy does not work for RTree
        # Add start and goal to the built graph
        temp_graph = self.graph
        temp_tree = self.tree
        for node in [self.start, self.goal]:
            temp_graph.add_node(node)
            temp_tree.add(node)
        for node in [self.start, self.goal]:
            self.connect_neighbors(node, temp_graph, temp_tree)
        node_path = nx.shortest_path(
            temp_graph, source=self.start, target=self.goal)
        length = nx.shortest_path_length(
            temp_graph, source=self.start, target=self.goal)
        # Construct path from nodes
        final_path = []
        for node in node_path:
            final_path.append(node.p)
        return np.array(final_path), length

    
    def connect_neighbors(self, node, graph, tree):
        neighbors = self.find_neighbors(node, tree)
        for neighbor in neighbors:
            if (self.map.collision(node.p, neighbor.p) != True):
                weight = self.dist(node, neighbor)
                graph.add_edge(node, neighbor, weight=weight)
        return neighbors

    def find_neighbors(self, node, tree):
        # num_near_neighbor = 5
        nnode = tree.len
        num_near_neighbor = int(5*np.log(nnode))
        neighbors = tree.k_nearest(node, num_near_neighbor)
        return neighbors
        
    def dist(self, from_node, to_node):
        # euler distance (2-norm)
        return np.linalg.norm(from_node.p - to_node.p)

    def sample(self):
        # Sample random point inside boundaries
        lower, upper = self.map.bounds
        return lower + np.random.rand(self.dim)*(upper - lower)

    def draw_path(self, ax, path):
        '''draw the path if available'''
        if path is None:
            print("path not available")
        else:
            # ax.plot(*np.array(path).T, '-',
            #         color=(0.9, 0.2, 0.5, 0.8), zorder=5)
            ax.plot(*np.array(path).T, '-',
                    color='b', zorder=5)
    
    def draw_graph(self, ax, scale_factor):
        '''draw the path if available'''
        self.build_graph()
        xs = []
        ys = []
        zs = []
        for node in list(self.graph.nodes):
            xs.append(scale_factor*node.p[0])
            ys.append(scale_factor*node.p[1])
            zs.append(scale_factor*node.p[2])
        ax.scatter(xs, ys, zs, marker='o', s=1)

        for edge in list(self.graph.edges):
            x1 = scale_factor*edge[0].p[0]
            x2 = scale_factor*edge[1].p[0]
            y1 = scale_factor*edge[0].p[1]
            y2 = scale_factor*edge[1].p[1]
            z1 = scale_factor*edge[0].p[2]
            z2 = scale_factor*edge[1].p[2]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='b', linewidth=0.1)
        

