import numpy as np
from rtree import index
from matplotlib.pyplot import Rectangle
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

class Map:
  # Self initiation
  def __init__(self, obstacle_list, bounds, path_resolution = 0.5, dim = 3):
    self.dim = dim 
    self.idx = self.get_tree(obstacle_list,dim)
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

def random_grid_3D(bounds, density):
  x_size = bounds[1]
  y_size = bounds[1]
  z_size = bounds[1]

  # Generate random grid with discrete 0/1 altitude using normal distribtion
  mean_E = 0
  sigma = 1
  k_sigma = density
  E = np.random.normal(mean_E, sigma, size=(x_size+1,y_size+1))
  h = 60

  # Set the decision threshold
  sigma_obstacle = k_sigma * sigma
  E = E > sigma_obstacle
  E = E.astype(np.float)
  
  # Generate random altitude to blocks
  h_min = 10 # minimal obstacles altitude

  E_temp = E

  for i in range(x_size):
      for j in range(y_size):
          k = range(i - 1 - round(np.random.beta(0.5, 0.5)), i + 1 + round(np.random.beta(0.5, 0.5)), 1)
          l = range(j - 1 - round(np.random.beta(0.5, 0.5)), j + 1 + round(np.random.beta(0.5, 0.5)), 1)

          if min(k) > 0 and min(l) > 0 and max(k) <= x_size and max(l) <= y_size and E_temp[j,i]==1:
              hh = round(np.random.normal(0.5*h, 0.5*h))

              if hh < h_min:
                  hh = h_min
              elif hh > z_size:
                  hh = z_size
              for m in k:
                  E[l,m] = hh
  return E


def main():
  # limits on map dimensions
  bounds = np.array([0,100])

  # Define the density value of the map
  density = 2.5

  # Create the obstacles on the map
  obstacles_ = random_grid_3D(bounds,density)
  obstacles = []
  for i in range(100):
    for j in range(100):
      if obstacles_[i,j] > 0:
        obstacles.append([i, j, 0, i-1, j+1, obstacles_[i,j]])

  # create map with selected obstacles
  mapobs = Map(obstacles, bounds, dim = 3)

  fig = plt.figure()
  ax = Axes3D.Axes3D(fig)
  ax.set_xlim((0,2))
  ax.set_ylim((0,2))
  ax.set_zlim((0,2))

  mapobs.plotobs(ax, scale = 0.02)
  plt.show()

if __name__ == "__main__":
  main()
