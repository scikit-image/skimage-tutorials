"""
Exercise 8
==========

Exercise 8 with matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5), dpi=80)

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C = np.cos(X)
S = np.sin(X)

ax.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
ax.plot(X, S, color="red", linewidth=2.5, linestyle="-",  label="sine")

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

ax.set_xlim(X.min() * 1.1, X.max() * 1.1)
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
              [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

ax.set_ylim(C.min() * 1.1, C.max() * 1.1)
ax.set_yticks([-1, +1],
              [r'$-1$', r'$+1$'])

ax.legend(loc='upper left')

plt.show()
