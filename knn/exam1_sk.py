#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from sklearn.neighbors import KDTree

np.random.seed(0)
points = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
tree = KDTree(points)
point = [6, 2]
# kNN
print np.linalg.norm(point - points[5])
dists, indices = tree.query([point], k=3)
print(dists, indices)

# query radius
indices = tree.query_radius([point], r=2.3)
print(indices)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.add_patch(Circle(point, 2.3, color='r', fill=False))
X, Y = [p[0] for p in points], [p[1] for p in points]
plt.scatter(X, Y)
plt.scatter([point[0]], [point[1]], c='r')
plt.show()
