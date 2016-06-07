"""
Plotting a scatter of points
==============================

A simple example showing how to plot a scatter of points with matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

n = 1024
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X)

fig = plt.figure()
ax = fig.add_axes([0.025, 0.025, 0.95, 0.95])
ax.scatter(X, Y, s=75, c=T, alpha=.5)

ax.set_xlim(-1.5, 1.5)
ax.set_xticks(())
ax.set_ylim(-1.5, 1.5)
ax.set_yticks(())

plt.show()
