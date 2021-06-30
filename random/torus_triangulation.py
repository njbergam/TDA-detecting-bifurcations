from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from math import *

COLORS = ["red", "green", "blue", "yellow", "purple", "black","orange","pink","indigo"]

def get_flat_vertices(centroid, radius):
    a = [centroid[0] - radius*sqrt(3), centroid[1], centroid[2]]
    b = [centroid[0], centroid[1], centroid[2]]
    c = [centroid[0] + radius*sqrt(3), centroid[1], centroid[2]]
    d = [centroid[0] + 2*radius*sqrt(3), centroid[1], centroid[2]]
    return [a,b,c,d]

def get_equitri_vertices(centroid, radius):
    a = [centroid[0], centroid[1], centroid[2] + radius]
    b = [centroid[0] + radius*cos(pi/6), centroid[1], centroid[2] - radius*sin(pi/6)]
    c = [centroid[0] - radius*cos(pi/6), centroid[1], centroid[2] - radius*sin(pi/6)]
    return [a,b,c]

def plot_3ply_rect(start_center, radius, length, inclination, ax_obj):
    end_center = [start_center[0], start_center[1] + (length*cos(inclination)), start_center[2] + (length*sin(inclination))]
    a, c, b, a_ = get_flat_vertices(start_center, radius)
    d, f, e, d_  = get_flat_vertices(end_center, radius)

    # vertices = [a, d, f, c]
    vertices = [a_,b,e,d_]
    # print(vertices)
    face1 = Poly3DCollection([vertices], alpha=0.5, color=COLORS.pop())
    vertices = [b,c,f,e]
    # print(vertices)
    face2 = Poly3DCollection([vertices], alpha=0.5, color=COLORS.pop())
    vertices = [c,a,d,f]
    # print(vertices)
    face3 = Poly3DCollection([vertices], alpha=0.5, color=COLORS.pop())

    ax_obj.add_collection3d(face1)
    ax_obj.add_collection3d(face2)
    ax_obj.add_collection3d(face3)

    return (a, c, b, a_, d, f, e, d_)

def plot_prism(start_center, radius, length, inclination, ax_obj):
    end_center = [start_center[0], start_center[1] + (length*cos(inclination)), start_center[2] + (length*sin(inclination))]
    a, b, c = get_equitri_vertices(start_center, radius)
    d, e, f = get_equitri_vertices(end_center, radius)

    # vertices = [a, d, f, c]
    vertices = [a,b,e,d]
    # print(vertices)
    face1 = Poly3DCollection([vertices], alpha=0.5, color=COLORS.pop())
    vertices = [b,c,f,e]
    # print(vertices)
    face2 = Poly3DCollection([vertices], alpha=0.5, color=COLORS.pop())
    vertices = [c,a,d,f]
    # print(vertices)
    face3 = Poly3DCollection([vertices], alpha=0.5, color=COLORS.pop())

    ax_obj.add_collection3d(face1)
    ax_obj.add_collection3d(face2)
    ax_obj.add_collection3d(face3)

    return (a, b, c, d, e, f)

fig = plt.figure()
ax = Axes3D(fig)

def annotate(text, pose):
    ax.text(*pose, text, size=20, color="red")


################ FIRST PRISM (BASE) ################

vertices = plot_3ply_rect([0, 0, 0], 1/sqrt(3), 1, 0, ax)

annotate("G", vertices[4])
annotate("H", vertices[5])
annotate("I", vertices[6])
annotate("G", vertices[7])

################ SECOND PRISM (LEFT) ################

vertices = plot_3ply_rect([0, -1, 0], 1/sqrt(3), 1, 0, ax)

annotate("A", vertices[0])
annotate("C", vertices[1])
annotate("B", vertices[2])
annotate("A", vertices[3])
annotate("D", vertices[4])
annotate("F", vertices[5])
annotate("E", vertices[6])
annotate("D", vertices[7])

################ THIRD PRISM (RIGHT) ################

vertices = plot_3ply_rect([0, 1, 0], 1/sqrt(3), 1, 0, ax)

annotate("A", vertices[4])
annotate("C", vertices[5])
annotate("B", vertices[6])
annotate("A", vertices[7])

ax.set_xlim(-1,2)
ax.set_ylim(-1,2)
ax.set_zlim(-1,2)

plt.show()
