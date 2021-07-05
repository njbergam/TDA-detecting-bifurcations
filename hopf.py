import numpy as np
import math
import gudhi as gd
from random import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from copy import deepcopy as dc


#Define the Hopf bifurcation where x is the "vector" (in a list form)
#i.e. x[0] = x_n and x[1]=y_n and n is the index
def f_mu(vec,mu):

	#given parameters for the system (From the note)
	a= (-1)*.02
	b=0.1
	c=0.1
	d = 0.2

	x = vec[0]
	y = vec[1]

	var1 = 1+d*mu+a*(x**2+y**2)
	var2 = b+c*(x**2+y**2)
	x_next = var1*(x*math.cos(var2)-y*math.sin(var2))
	y_next = var1*(x*math.sin(var2)+y*math.cos(var2))

	return [x_next,y_next]

#generate a point cloud for a fixed mu_valuestart for a given initial condition
#In this case I've set initial value to be (0,0) as a default since (0,0) is an equilibrium point
#I used the points of the system to be the point cloud here. Another method I could use is generate a point cloud
#by taking 
def generate_pt_cloud(N,mu,init_value=[0,0]):
	pt_cloud = []
	#vec(x)_i is expressed as pt_cloud[i]
	for i in range(N):
		#initial value for the series
		if i == 0:
			pt_cloud.append(init_value)
		else:
			pt_cloud.append(f_mu(pt_cloud[i-1],mu))

	return pt_cloud


#generate a point cloud by taking one variable, and embedding it to the given dimension
#ex) from (x,y) take x's and embed them to two dimension so that we will be considering
#   (x_1,x_2), (x_2,x_3),...,(x_(N-1),x_N) as a point cloud
#spec
def generate_pt_cloud_one_var(N,mu,init_value=[0,0], dim = 2):
	pts = generate_pt_cloud(N,mu,init_value)
	x_pts = []
	pt_cloud = []

	for i in range(len(pts)):
		x_pts.append(pts[i][0])

	for i in range(len(pts)-(dim-1)):
		pt = []
		for j in range(dim):
			pt.append(x_pts[i+j])
		pt_cloud.append(pt)

	return pt_cloud



#Testing: the generate_pt_cloud function works as I intend to (it does :)
# pt_cloud = generate_pt_cloud(1000,0.2,[0.1,0])
# x = []
# y = []
# for i in range(len(pt_cloud)):
# 	x.append(pt_cloud[i][0])
# 	y.append(pt_cloud[i][1])
# plt.plot(x,y)
# plt.show()


#Test case for generate_pt_cloud_one_var
# pt_cloud = generate_pt_cloud_one_var(1000,0.5,[0.1,0],3)
# print(pt_cloud)
# x = []
# y = []
# z = []
# for i in range(len(pt_cloud)):
# 	x.append(pt_cloud[i][0])
# 	y.append(pt_cloud[i][1])
# 	z.append(pt_cloud[i][2])

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(x,y,z, cmap='Greens')
# plt.show()

def generate_persistence_diag(pt_cloud, max_radius = 1, max_dim=2):
	rips_complex = gd.RipsComplex(points = pt_cloud, max_edge_length = max_radius)
	simplex_tree = rips_complex.create_simplex_tree(max_dimension = max_dim)
	pers = simplex_tree.persistence()
	ax = gd.plot_persistence_diagram(pers)
	return ax

#Test case for the function "generate_persistence_diag" -> seems to be working
pt_cloud = generate_pt_cloud_one_var(1000,0.1,[0.5,0],2)
print(pt_cloud)
generate_persistence_diag(pt_cloud,0.5,2)
plt.show()

# mu_values = np.linspace(start = 0.1, stop = 0.8, num = 14)
# subplot = {}
# for mu in mu_values:
# 	pt_cloud = generate_pt_cloud(100,mu,[1,0])
# 	plot = generate_persistence_diag(pt_cloud)
# 	fig, sub = plt.subplots(14)
# 	subplot[mu] = sub

# for mu in mu_values:
# 	subplot[mu].plot()


# def persistence_diag_varying_param(mu, mu_max, iterations, initial_pt = [0,0]
# 	, pt_cloud_size=500, max_dim=2):
# 	pers_dictionary = {}
# 	mu_values = []
# 	mu_values = np.linspace(start = mu, stop = mu_max, num = iterations)
# 	for mu in mu_values:
# 		pt_cloud = generate_pt_cloud(pt_cloud_size, mu, initial_pt)
# 		simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
# 		pers = simplex_tree.persistence()
# 		per_dictionary{mu} = gd.plot_persistence_diagram(pers)

# 	for mu in mu_values:


