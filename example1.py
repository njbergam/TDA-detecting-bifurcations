import numpy as np
import math
import gudhi as gd
from gudhi.representations import Landscape
from random import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from copy import deepcopy as dc

def scatter_3D(X, Y, Z, colorarray = None, color="green", title="3D Scatter plot"):
    """
    Helper function to create a 3D scatter plot with the default settings for axis limits and labels
    """
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
    """
    Splits a list of n-dimensional points (the point cloud) into a list of n lists, where the k-th list contains the k-th coordinates of all the points.

    EXAMPLES
    ========
    >>> split_point_cloud([[1,2,3,4], [-2,-4,-5,-9],[10,3,9,-6]])
    [[1,-2,10], [2,-4,3], [3, -5, 9], [4,-9,-6]]
    """
    pt_cloud = np.array(pt_cloud)
    s = np.shape(pt_cloud)
    # print("Shape: ", s)
    return [pt_cloud[:,i] for i in range(s[1])]

F = {}
MU_MAX = 0.75 + (math.sqrt(768)/32)

def f(x, mu, n):
    """
    The governing function of the dynamical system. In our case, the function is:
    `F_mu: R -> R`
    such that for all `x in R`,
    `F_mu(x) = mu + x(1-x)`

    ARGUMENTS
    =========
    `x` (float): the input to evaluate the function at
    `mu` (float): the parameter value for the dynamical system
    `n` (int): the number of times `x` has to be iterated under the function `F_mu`
    """
    if (x, mu, n) in F:
        return F[(x, mu, n)]
    elif n == 0:
        return x
    y = f(x, mu, n-1)
    F[(x, mu, n)] = mu + y - (y*y)
    return F[(x, mu, n)]

def generate_point_cloud(num_series, series_length, data_dim, noise=0, fixed_mu=0, starting_point_in_range=True):
    """
    Generates a point cloud based on the given parameters. The current method to do so is:
    (1) Start with a random point X
    (2) Store a K-length trajectory, T of X (the first K points that X visits)
    (3) Append all D-length continuous subsequences of T to a list: this list of D-dimensional points is the point cloud
    (4) Repeat steps (1) -> (2) -> (3) multiple times (say N), and merge all the point clouds thus obtained.

    ARGUMENTS
    =========
    `num_series` (int): the number `N` in the above description
    `series_length` (int): the number `K` in the above description
    `data_dim` (int): the number `D` in the above description
    `noise` (float): variance of the Gaussian noise to be added to the point cloud. If `0` is given, no noise will be added.
    `fixed_mu` (float): Execute step (4) above with system parameter = `fixed_mu` for every repetition. If `0` is given, evenly distributed values of the system parameter would be used for each repetition of (1) -> (2) -> (3)
    `starting_point_in_range` (bool): If `True`, the point `X` is chosen in a reasonably nice range so that it doesn't blow up following the dynamical system. If `False`, the point `X` is chosen in a more arbitrary range, where it could potentially blow up.
    """
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
        # x_init = 0
        series = []
        if starting_point_in_range:
            init_max = 0.5 + (0.5 * math.sqrt(1 + (4 * mu)))
            init_min = -math.sqrt(mu)
            init_range = init_max - init_min
            x_init = (random() * init_range) + init_min
            series = [f(x_init, mu, n) for n in range(1, series_length+1)]
        else:
            x_init = (random() * 4) - 2 # start with a point in [-2, 2)
            reset_count = 0
            x = f(x_init, mu, 1)
            while len(series) < series_length:
                if x < -5 or x > 5: # if it goes outside [-3, 3], RESET!
                    x = (random() * 4) - 2
                    reset_count += 1
                series.append(x)
                x = f(x, mu, 1)
            if reset_count >= 1:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!! TOO MANY RANDOM POINTS !!!!!!!!!!!!!!!!!!!!!!!!!!")
        if noise != 0:
            noisy_series = [x + np.random.uniform(0, noise) for x in series]
        # s = np.array(series)
        # if any(s > init_max) or any(s < init_min):
        #     print("mu = %d" % mu)
        #     print("x_init = %d" % x_init)
        #     print("series = %s" % str(series))
        #     exit()
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
    Returns and optionally plots the persistence diagram of the given point cloud using the Rips Complex.
    
    ARGUMENTS
    =========
    `point_cloud` (list): the point cloud to undergo persistence analysis
    `maximal_radius` (float): the maximum radius, epsilon, to be used for constructing the Rips Complex
    `max_dim` (int): the maximum dimension of persistence to be computed
    `plot` (bool): if `True`, the persistence diagram is plotted

    RETURN
    ======
    Returns the persistence diagram as a dictionary such that
        a) keys are the dimensions being considered
        b) for a given key `D`, the value would be the list of all points
           in the `D`-dimensional persistence diagram.
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
    """
    Plots the persistence landscape of the given persistence diagram. Unverified.
    """
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
    Computes the L1-norm of a ZERO dimensional persistence diagram. Not implemented yet.
    """
    pass # Persistence landscape doesn't seem too useful for analysing 0-dim persistence.

def run_fixed_param_persis_on_random_data(num_points, mu, noise=0.25, plot_cloud=True, plot_pers_diag=True, restrict_init_point=True):
    """
    1) Generates a dataset from the dynamical system such that
        a) main parameter = `mu`
        b) dataset has size `num_points`
    2) Optionally plots the persistence diagram and the point cloud
    3) Returns the persistence diagram.

    ARGUMENTS
    =========
    `num_points` (int): the number of points in the dataset to be generated
    `mu` (float): the value of the system parameter to be used
    `noise` (float): variance of the noise to be added
    `plot_cloud` (bool): if `True`, the point cloud is plotted
    `plot_pers_diag` (bool): if `True`, the persistence diagram is plotted
    `restrict_init_point` (bool): this value is set to the parameter `starting_point_in_range` when calling the `generate_point_cloud` function

    RETURN
    ======
    Returns the persistence diagram as a dictionary such that
        a) keys are the dimensions being considered
        b) for a given key `D`, the value would be the list of all points
           in the `D`-dimensional persistence diagram.
    """
    assert(mu <= MU_MAX)
    if restrict_init_point:
        assert(mu > 0)

    num_series = num_points
    data_dim = 3
    series_length = data_dim
    point_cloud, noisy_point_cloud = generate_point_cloud(num_series, series_length, data_dim, noise = noise, fixed_mu = mu, starting_point_in_range=restrict_init_point)

    if noise != 0:
        point_cloud = noisy_point_cloud

    if plot_cloud:
        X, Y, Z = split_point_cloud(point_cloud)
        scatter_3D(X, Y, Z)

    dim_v_persistence = draw_persistence_diagram(noisy_point_cloud, maximal_radius=4, max_dim=1, plot=plot_pers_diag)

    return dim_v_persistence

def compute_separation_norm(pers_diag):
    """
    Computes the *separation norm* of the given zero dimensional persistence diagram.
    The separation norm of a zero-dimensional persistence diagram is defined as the maximum
    distance between any two consecutive points. Since in a 0-dim persis. diag., all points
    are on the y-axis, they are naturally ordered. So 'consecutive' means consecutive under
    this natural order.

    ARGUMENT
    ========
    `pers_diag` (dict): to be given in the same format as the return type of the function `run_fixed_param_persis_on_random_data`. This function uses only the 0-dim diagram from that.

    EXAMPLES
    ========
    >>> compute_separation_norm({0:[(0,0.5), (0,1), (0,2), (0,5), (0,6)]})
    3
    """
    pers = [point[1] for point in pers_diag[0]] # collecting y coordinates of points in 0-dim persistence diagram
    pers = sorted(pers)
    max_sep = max([pers[i+1]-pers[i] for i in range(len(pers)-1)])
    return max_sep

def sample_norms_for_fixed_mu(mu, sample_size, restrict_init_point=True):
    """
    Samples and returns the separation norm of multiple persistence diagram with the same fixed system parameter `mu`.

    ARGUMENTS
    =========
    `mu` (float): the value of the system parameter to be used
    `sample_size` (int): the number of samples of persistence diagrams to be used
    `restrict_init_point` (bool): this value is set to the argument of the same name in the function `run_fixed_param_persis_on_random_data`

    RETURN
    ======
    A list of the sampled separation norms
    """
    norms = [0 for i in range(sample_size)]
    for i in range(sample_size):
        pers_diag = run_fixed_param_persis_on_random_data(100, mu, noise=0.25, plot_cloud=False, plot_pers_diag=False, restrict_init_point=restrict_init_point)
        norms[i] = compute_separation_norm(pers_diag)
    return norms

def plot_bifurcation_metric(num_mu_values, sample_size_for_each_mu, plot_max_norm=True, plot_avg_norm=True):
    """
    Plots the bifurcation metric for various values of the system parameter. The bifurcation
    metric is defined as the average (or alternatively, maximum) of the sampled separation norms obtained for a given value of mu (from the function `sample_norms_for_fixed_mu`).

    ARGUMENTS
    =========
    `num_mu_values` (int): number of values of the system parameter for which the bifurcation metric is to be plotted.
    `sample_size_for_each_mu` (int): the number of samples of separation norms to be used for each mu
    `plot_max_norm` (bool): if `True` the bifurcation metric (using maximum definition) is plotted against the system parameter
    `plot_avg_norm` (bool): if `True` the bifurcation metric (using average definition) is plotted against the system parameter
    """
    mu_values = np.linspace(MU_MAX, 0, num_mu_values, endpoint=False)
    max_norm = []
    avg_norm = []

    for mu in mu_values:
        norms_of_multiple_experiments = sample_norms_for_fixed_mu(mu, sample_size_for_each_mu, restrict_init_point=True)
        max_norm.append(max(norms_of_multiple_experiments))
        avg_norm.append(sum(norms_of_multiple_experiments)/len(norms_of_multiple_experiments))

    if plot_max_norm:
        plt.plot(mu_values, max_norm, color="r", label=("Maximum of Separation Norm of %d pers diagrams" % sample_size_for_each_mu))
    if plot_avg_norm:
        plt.plot(mu_values, avg_norm, color="b", label=("Mean of Separation Norm of %d pers diagrams" % sample_size_for_each_mu))
    plt.xlabel("System Parameter: mu")
    plt.ylabel("Bifurcation Metric")
    plt.legend()

    ymin = min(*avg_norm, *max_norm)
    ymax = max(*avg_norm, *max_norm)
    for mu in mu_values:
        plt.plot([mu, mu], [ymin, ymax], '--', color="grey")
        # plt.text(mu, -0.1, str(mu), fontsize=5)

# run_fixed_param_persis_on_random_data(400, 1, restrict_init_point=True)

plot_bifurcation_metric(20, 50)

plt.show()