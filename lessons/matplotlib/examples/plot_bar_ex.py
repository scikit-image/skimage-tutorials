"""
Bar plots
==========

An example of bar plots with matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

n = 12
X = np.arange(n)
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

fig = plt.figure()
ax = fig.add_axes([0.025, 0.025, 0.95, 0.95])
ax.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
ax.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x, y in zip(X, Y1):
    ax.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')

for x, y in zip(X, Y2):
    ax.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va='top')

ax.set_xlim(-.5, n)
ax.set_xticks(())
ax.set_ylim(-1.25, 1.25)
ax.set_yticks(())

plt.show()
