# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 19:06:30 2016

@author: nkim30
"""
import numpy as np
import matplotlib.pyplot as plt

mean = [10, 10]
cov = [[1, 0], [0, 1]]  # diagonal covariance
#Diagonal covariance means that points are oriented along x or y-axis:
x1, y1 = np.random.multivariate_normal(mean, cov, 1000).T
plt.plot(x1, y1, 'x')

mean = [-10, -10]
cov = [[1, 0], [0, 1]]
x2, y2 = np.random.multivariate_normal(mean, cov, 1000).T
plt.plot(x2, y2, 'x')

mean = [10, -10]
cov = [[1, 0], [0, 1]]  # diagonal covariance
#Diagonal covariance means that points are oriented along x or y-axis:
x3, y3 = np.random.multivariate_normal(mean, cov, 1000).T
plt.plot(x3, y3, 'o')

mean = [-10, 10]
cov = [[1, 0], [0, 1]]
x4, y4 = np.random.multivariate_normal(mean, cov, 1000).T
plt.plot(x4, y4, 'o')

#plt.axis('equal')
#plt.scatter(x, y)
plt.show()
