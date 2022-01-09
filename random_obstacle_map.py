import numpy as np
from rtree import index
from matplotlib.pyplot import Rectangle
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

class Map:
  def __init__(self, obstacle_list, bounds, path_resolution = 0.5, dim = 3):
    '''initialise map with given properties'''
    self.dim = dim 
    self.idx = self.get_tree(obstacle_list, dim)
    self.len = len(obstacle_list)
    self.path_res = path_resolution
    self.obstacles = obstacle_list
    self.bounds = bounds

  @staticmethod
  def get_tree(obstacle_list, dim):
    '''initialise map with given obstacle_list'''
    p = index.Property()
    p.dimension = dim
    ls = [(i,(*obj,),None) for i, obj in enumerate(obstacle_list)]
    return index.Index(ls, properties=p)

  def add(self, obstacle):
    '''add new obstacle to the list'''
    self.idx.insert(self.len, obstacle)
    self.obstacles.append(obstacle)
    self.len += 1

  def collision(self,start,end):
    '''find if the ray between start and end collides with obstacles'''
    dist = np.linalg.norm(start-end)
    n = int(dist/self.path_res)
    points = np.linspace(start,end,n)
    for p in points:
      if self.idx.count((*p,)) != 0 :
          return True
    return False

  def inbounds(self,p):
    '''Check if p lies inside map bounds'''
    lower,upper = self.bounds
    return (lower <= p).all() and (p <= upper).all()

  def plotobs(self,ax,scale = 1):
    '''plot all obstacles'''
    obstacles = scale*np.array(self.obstacles)
    if self.dim == 2:
        for box in obstacles:
            l = box[2] - box[0]
            w = box[3] - box[1]
            box_plt = Rectangle((box[0], box[1]),l,w,color='k',zorder = 1)
            ax.add_patch(box_plt)
    elif self.dim == 3:
        for box in obstacles:
            X, Y, Z = cuboid_data(box)
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color=(0.1, 0.15, 0.3, 0.2), zorder = 1)
    else: print('can not plot for given dimensions')


#to plot obstacle surfaces
def cuboid_data(box):
    l = box[3] - box[0]
    w = box[4] - box[1]
    h = box[5] - box[2]
    x = [[0, l, l, 0, 0],
         [0, l, l, 0, 0],
         [0, l, l, 0, 0],
         [0, l, l, 0, 0]]
    y = [[0, 0, w, w, 0],
         [0, 0, w, w, 0],
         [0, 0, 0, 0, 0],
         [w, w, w, w, w]]
    z = [[0, 0, 0, 0, 0],
         [h, h, h, h, h],
         [0, 0, h, h, 0],
         [0, 0, h, h, 0]]
    return box[0] + np.array(x), box[1] + np.array(y), box[2] + np.array(z)


# Generate random obstacle parameters
def random_grid_3D(bounds, density, height):
  x_size = bounds[1]
  y_size = bounds[1]
  z_size = bounds[1]

  # Generate random grid with discrete 0/1 altitude using normal distribtion
  mean_E = 0
  sigma = 1
  k_sigma = density
  E = np.random.normal(mean_E, sigma, size=(x_size+1, y_size+1))
  h = height

  # Set the decision threshold
  sigma_obstacle = k_sigma * sigma
  E = E > sigma_obstacle
  E = E.astype(np.float)

  # Generate random altitude to blocks
  h_min = 10 # minimal obstacles altitude
  E_temp = E
  for i in range(x_size):
      for j in range(y_size):
          #k = range(i - 1 - round(np.random.beta(0.5, 0.5)), i + 1 + round(np.random.beta(0.5, 0.5)), 1)
          #l = range(j - 1 - round(np.random.beta(0.5, 0.5)), j + 1 + round(np.random.beta(0.5, 0.5)), 1)

          if E_temp[j,i]==1:
              hh = round(np.random.normal(0.7*h, 0.5*h))
              if hh < h_min:
                  hh = h_min
              elif hh > z_size:
                  hh = z_size
              E[j,i] = hh
  return E

def generate_map(bounds, density, height, start, goal):
  # Create the obstacles on the map
  obstacles_ = random_grid_3D(bounds,density,height)
  obstacles = []
  for i in range(bounds[1]):
    for j in range(bounds[1]):
      if obstacles_[i,j] > 0:
        ss = round(np.random.normal(3, 1)) # Define the parameter to randomize the obstacle size
        obstacles.append([i, j, 0, i+ss, j+ss, obstacles_[i,j]])
        
  # create map with selected obstacles
  obstacles = start_goal_mapcheck(start,goal,obstacles)
  mapobs = Map(obstacles, bounds, dim = 3)
  print('Generate %d obstacles on the random map.'%len(obstacles))
  return mapobs, obstacles


# Check if the start point and the goal point on the map  
def start_goal_mapcheck(start,goal,obstacles):
  for i in range(len(obstacles)):
    if (start[0] >= obstacles[i][0] and start[0] <= obstacles[i][3] and start[1] >= obstacles[i][1] \
    and start[1] <= obstacles[i][4] and start[2] >= obstacles[i][2] and start[2] <= obstacles[i][5]) \
    or (goal[0] >= obstacles[i][0] and goal[0] <= obstacles[i][3] and goal[1] >= obstacles[i][1] \
    and goal[1] <= obstacles[i][4] and goal[2] >= obstacles[i][2] and goal[2] <= obstacles[i][5]):
      obstacles.pop(i)
      print("The start point and goal point collides with obstacles!")
  return obstacles


def main():
  # limits on map dimensions
  bounds = np.array([0,100])

  # Define the density value of the map
  density = 2.5

  # Define the height parameter of the obstacles on the map
  height = 0.5 * bounds[1]

  # Define the start point and goal point
  start_point = np.array([0,0,5])
  goal_point = np.array([65,65,65])

  # create map with selected obstacles
  mapobs,obstacles = generate_map(bounds, density, height, start_point, goal_point)
  # Visualize the obstacle map 
  fig = plt.figure()
  ax = Axes3D.Axes3D(fig)
  ax.set_xlim((bounds[0],bounds[1]))
  ax.set_ylim((bounds[0],bounds[1]))
  ax.set_zlim((bounds[0],bounds[1]))

  mapobs.plotobs(ax, scale = 1)
  plt.show()

'''Call the main function'''
if __name__ == "__main__":
  main()

  
