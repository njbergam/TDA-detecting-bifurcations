import numpy as np
import math
import gudhi as gd
from gudhi.representations import Landscape
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
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(title)
    # plt.show()

def split_point_cloud(pt_cloud):
    pt_cloud = np.array(pt_cloud)
    s = np.shape(pt_cloud)
    # print("Shape: ", s)
    return [pt_cloud[:,i] for i in range(s[1])]

F = {}

def f(x, mu, n):
    if (x, mu, n) in F:
        return F[(x, mu, n)]
    elif n == 0:
        return x
    y = f(x, mu, n-1)
    F[(x, mu, n)] = mu + y - (y*y)
    return F[(x, mu, n)]

def generate_point_cloud(num_series, series_length, data_dim, noise=0, fixed_mu=0):
    assert(series_length >= data_dim)
    assert(fixed_mu >= 0)
    pt_cloud = []
    noisy_pt_cloud = []
    mu_values = []
    if fixed_mu == 0:
        mu_values = np.linspace(start=0.75 + (math.sqrt(768)/32), stop=0, num = num_series, endpoint=False)
    else:
        mu_values = [fixed_mu for i in range(num_series)]
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

def draw_persistence_diagram(point_cloud, maximal_radius=1, max_dim=1, plot=True):
    """
    makes the persistence diagram given the point cloud, the maximum radius and the maximum dimension
    """
    rips_cmplx = gd.RipsComplex(points=point_cloud, max_edge_length=maximal_radius) 
    simplex_tree = rips_cmplx.create_simplex_tree(max_dimension = max_dim)
    pers = simplex_tree.persistence()
    dim_wise_pers_diag = {}
    for point in pers:
        if point[0] not in dim_wise_pers_diag:
            dim_wise_pers_diag[point[0]] = []
        if not math.isinf(point[1][1]):
            dim_wise_pers_diag[point[0]].append(point[1])
    if plot:
        ax = gd.plot_persistence_diagram(pers, legend = True)
        ax.set_aspect("equal")
    return dim_wise_pers_diag

def draw_persistence_landscape(pers_diagram, dim, resolution=0, num_landscapes=0):
    pers_diagram = np.array(pers_diagram[dim])
    if num_landscapes == 0:
        num_landscapes = len(pers_diagram)
    if resolution == 0:
        resolution = 5 * num_landscapes
    print("resolution = %d" % resolution)
    print("resolution = %d" % num_landscapes)
    landscape_obj = Landscape(num_landscapes=num_landscapes, resolution=resolution)
    landscape = landscape_obj(pers_diagram)
    plt.figure()
    plt.plot(landscape, color="b")
    return landscape

def compute_landscape_norm(pers_landscape, p):
    """
    Computes the L1-norm of a ZERO dimensional persistence diagram
    """
    pass # Persistence landscape doesn't seem too useful for analysing 0-dim persistence.

MU_MAX = 0.75 + (math.sqrt(768)/32)

def get_fixed_param_persistence(num_points, mu, noise=0.25, plot_cloud=True):
    """
    1) Generates a dataset from the dynamical system such that
        a) main parameter = mu
        b) dataset has size num_points
    2) Plots the persistence diagram, and optionally the point cloud
    3) Returns the persistence diagram as a dictionary such that
        a) keys are the dimensions being considered
        b) for a given key D, the value would be the list of all points
           in the D-dimensional persistence diagram.
    """
    assert(type(mu) == float)
    assert(mu > 0 and mu <= MU_MAX)

    num_series = num_points
    data_dim = 3
    series_length = data_dim
    point_cloud, noisy_point_cloud = generate_point_cloud(num_series, series_length, data_dim, noise = noise, fixed_mu = mu)

    if noise != 0:
        point_cloud = noisy_point_cloud

    if plot_cloud:
        X, Y, Z = split_point_cloud(point_cloud)
        scatter_3D(X, Y, Z)

    dim_v_persistence = draw_persistence_diagram(noisy_point_cloud, maximal_radius=4, max_dim=1, plot=True)

    return dim_v_persistence

# def compute_separation_norm(pers_diag, thresh):
#     pers = [pers]
#     pass
# def detect_bifurcation():
#     pass

get_fixed_param_persistence(200, MU_MAX)

plt.show()