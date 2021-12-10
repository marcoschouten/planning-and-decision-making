from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def grid_3D(sizeE, d_grid, h, P_init, P_end, n_low):
    x_size = sizeE[0]
    y_size = sizeE[1]
    z_size = sizeE[2]

    z_grid = np.linspace(0, z_size, (int)(z_size/d_grid)+1)

    # Generate random grid with discrete 0/1 altitude using normal distribtion
    mean_E = 0
    sigma = 1
    k_sigma = 2.2
    E = np.random.normal(mean_E, sigma, size=(x_size+1,y_size+1))
    sigma_obstacle = k_sigma * sigma
    E = E > sigma_obstacle
    E = E.astype(np.float)

    # Generate random altitude to blocks
    h_min = 5 # minimal obstacles altitude

    # Assign the temporary matrix
    E_temp = E


    for i in range(x_size):
        for j in range(y_size):
            k = range(i - 1 - round(np.random.beta(0.5, 0.5)), i + 1 + round(np.random.beta(0.5, 0.5)), 1)
            l = range(j - 1 - round(np.random.beta(0.5, 0.5)), j + 1 + round(np.random.beta(0.5, 0.5)), 1)

            if min(k) > 0 and min(l) > 0 and max(k) <= x_size and max(l) <= y_size and E_temp[j,i]==1:
                hh = round(np.random.normal(0.75*h, 0.75*h))

                if hh < h_min:
                    hh = h_min
                elif hh > z_size:
                    hh = z_size
                for m in k:
                    E[l,m] = hh

    return E

sizeE = [80, 80, 10]
d_grid = 1
h = 15
P_init = [5, 7, 12]
P_end = [66, 68, 6]
n_low = 3
a= grid_3D(sizeE, d_grid, h, P_init, P_end, n_low)
np.set_printoptions(threshold=np.inf)

x_grid = range(0, sizeE[1], d_grid)
y_grid = range(0, sizeE[0], d_grid)
X,Y = np.meshgrid(x_grid, y_grid)
fig = plt.figure(figsize = (14,9))
ax = plt.axes(projection='3d')

ax.plot_surface(X,Y, a[:80,:80])
plt.show()
