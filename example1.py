import numpy as np
import math
import gudhi as gd
from random import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from copy import deepcopy as dc

def scatter_3D(X, Y, Z, colorarray = None, color="green", title="3D Scatter plot"):
    fig = plt.figure()
    ax = Axes3D(fig)
    if colorarray is None:
        ax.scatter3D(X, Y, Z, color=color)
    else:
        ax.scatter3D(X, Y, Z, c = colorarray)
    lowlim = min(*X, *Y, *Z)
    uplim = max(*X, *Y, *Z)
    ax.set_xlim3d(lowlim, uplim)
    ax.set_ylim3d(lowlim, uplim)
    ax.set_zlim3d(lowlim, uplim)
    ax.set_title(title)
    # plt.show()

F = {}

def f(x, mu, n):
    if (x, mu, n) in F:
        return F[(x, mu, n)]
    elif n == 0:
        return x
    y = f(x, mu, n-1)
    F[(x, mu, n)] = mu + y - (y*y)
    return F[(x, mu, n)]

def generate_point_cloud(num_series, series_length, data_dim, noise=0):
    assert(series_length >= data_dim)
    pt_cloud = []
    noisy_pt_cloud = []
    mu_values = np.linspace(start=0.75 + (math.sqrt(768)/32), stop=0, num = num_series, endpoint=False)
    for mu in mu_values:
        init_max = 0.5 + (0.5 * math.sqrt(1 + (4 * mu)))
        init_min = -math.sqrt(mu)
        init_range = init_max - init_min
        x_init = (random() * init_range) + init_min
        series = [f(x_init, mu, n) for n in range(1, series_length+1)]
        if noise != 0:
            noisy_series = [x + np.random.uniform(0, noise) for x in series]
        s = np.array(series)
        if any(s > init_max) or any(s < init_min):
            # debug printing to check when variable goes out of domain
            print("mu = %d" % mu)
            print("x_init = %d" % x_init)
            print("series = %s" % str(series))
            exit()
        for j in range(series_length-data_dim+1):
            pt_cloud.append(series[j:j+data_dim])
            if noise != 0:
                noisy_pt_cloud.append(noisy_series[j:j+data_dim])
    if noise != 0:
        return (pt_cloud, noisy_pt_cloud)
    else:
        return pt_cloud

def make_persistence_diagram(point_cloud, maximal_radius=1, max_dim=1):
    rips_cmplx = gd.RipsComplex(points=point_cloud, max_edge_length=maximal_radius) 
    simplex_tree = rips_cmplx.create_simplex_tree(max_dimension = max_dim)
    pers = simplex_tree.persistence()
    ax = gd.plot_persistence_diagram(pers, legend = True)
    ax.set_aspect("equal")

num_series = 100
data_dim = 3
series_length = data_dim
point_cloud, noisy_point_cloud = generate_point_cloud(num_series, series_length, data_dim, noise = 1)

# print(point_cloud)

points = np.array(point_cloud)
noisy_points = np.array(noisy_point_cloud)
X = points[:,0]
Y = points[:,1]
Z = points[:,2]
# scatter_3D(X, Y, Z, title="Reconstructed Phase Space")
X = noisy_points[:,0]
Y = noisy_points[:,1]
Z = noisy_points[:,2]
scatter_3D(X, Y, Z, color="red", title="Reconstructed Phase Space with Noise")

# make_persistence_diagram(point_cloud, maximal_radius=2, max_dim=data_dim)
make_persistence_diagram(noisy_point_cloud, maximal_radius=4, max_dim=data_dim)

plt.show()
