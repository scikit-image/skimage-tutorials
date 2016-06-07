```python
%matplotlib inline
import matplotlib
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'nearest'
```

# RANSAC

From WikiPedia:

> Random sample consensus (RANSAC) is an iterative method to estimate
> parameters of a mathematical model from a set of observed data which
> contains outliers. Therefore, it also can be interpreted as an
> outlier detection method.

[Gallery example](http://scikit-image.org/docs/dev/auto_examples/plot_matching.html)



```python
import numpy as np
from matplotlib import pyplot as plt

from skimage.measure import ransac, LineModelND
```

Let's set up some random data points:



```python
np.random.seed(seed=1)

# generate coordinates of line
x = np.arange(-200, 200)
y = 0.2 * x + 20
data = np.column_stack([x, y])

# add faulty data
faulty = np.array(30 * [(180., -100)])
faulty += 5 * np.random.normal(size=faulty.shape)
data[:faulty.shape[0]] = faulty

# add gaussian noise to coordinates
noise = np.random.normal(size=data.shape)
data += 0.5 * noise
data[::2] += 5 * noise[::2]
data[::4] += 20 * noise[::4]

plt.plot(data[:, 0], data[:, 1], '.');
```

Now, fit a line to the data.  We start with our model:

$$\mathbf{y} = m \mathbf{x} + c$$

Or, in matrix notation:

$$\mathbf{y} = \left[ \begin{array}{c} \vdots \\ \mathbf{x} \\ \vdots \end{array}
                     \ \begin{array}{c} \vdots \\ \mathbf{1} \\ \vdots \end{array} \right]
                     \left[ \begin{array}{c} m \\ c \end{array} \right]
                     = X \mathbf{p}$$

TODO: the above is not rendered correctly. p is column vector.

Since we have an over-determined system, we use least squares to solve:


```python
x = data[:, 0]
y = data[:, 1]

X = np.column_stack((x, np.ones_like(x)))

p, _, _, _ = np.linalg.lstsq(X, y)
p
```

With those parameters in hand, let's plot the resulting line:


```python
m, c = p
plt.plot(x, y, 'b.')

xx = np.arange(-250, 250)
plt.plot(xx, m * xx + c, 'r-');
```

Scikit-image provides an N-dimensional LineModel object that encapsulates the above:


```python
model = LineModelND()
model.estimate(data)
model.params
```

Instead of ``m`` and ``c``, it parameterizes the line by ``origin``
and ``direction`` --- much safer when dealing with vertical lines,
e.g.!


```python
origin, direction = model.params
plt.plot(x, y, 'b.')
plt.plot(xx, model.predict_y(xx), 'r-');
```

Now, we robustly fit the line using inlier data selecte with the RANSAC algorithm:

```python
model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                               residual_threshold=10, max_trials=1000)
outliers = (inliers == False)

yy = model_robust.predict_y(xx)

fig, ax = plt.subplots()

ax.plot(data[inliers, 0], data[inliers, 1], '.b', alpha=0.6, label='Inlier data')
ax.plot(data[outliers, 0], data[outliers, 1], '.r', alpha=0.6, label='Outlier data')
ax.plot(xx, yy, '-b', label='Robust line model')

plt.legend(loc='lower left')
plt.show()
```

## Going interplanetary

The sun is one of the most spherical objects in our solar system.
According to an [article in Scientific American](http://www.scientificamerican.com/gallery/well-rounded-sun-stays-nearly-spherical-even-when-it-freaks-out/):

> Earth's closest star is one of the roundest objects humans have
> measured. If you shrank the sun down to beach ball size, the
> difference between its north-south and the east-west diameters would
> be thinner than the width of a human hair, says Jeffery Kuhn, a
> physicist and solar researcher at the University of Hawaii at
> Manoa. "Not only is it very round, but it's too round," he adds. The
> sun is more spherical and more invariable than theories predict.

If the sun is spherical, we should be able to fit a circle to a 2D
slice of it!  Let's load an example image:


```python
from skimage import io

image = io.imread('../../images/superprom_prev.jpg')

f, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image);
```

In this specific image, we got a bit more than we bargained for in the
form of magnificently large solar flares.  Let's see if some *canny
edge detection* will help isolate the sun's boundaries.

```python
from skimage import feature, color

f, ax = plt.subplots(figsize=(10, 10))

edges = feature.canny(color.rgb2gray(image), sigma=2)
ax.imshow(edges, cmap='gray')
```

The edges look good, but there's a lot going on inside the sun.  We
use RANSAC to fit a robust circle model.

```python
from skimage.measure import CircleModel

points = np.array(np.nonzero(edges)).T

model_robust, inliers = ransac(points, CircleModel, min_samples=3,
                               residual_threshold=2, max_trials=1000)
```

The parameters of the circle are center x, y and radius:

```python
model_robust.params
```

Let's visualize the results, drawing a circle on the sun, and also
highlighting inlier vs outlier edge pixels:

```python
from skimage import draw

cy, cx, r = model_robust.params

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 8))

ax0.imshow(image)
ax1.imshow(image)

ax1.plot(points[inliers, 1], points[inliers, 0], 'b.', markersize=1)
ax1.plot(points[~inliers, 1], points[~inliers, 0], 'g.', markersize=1)
ax1.axis('image')

circle = plt.Circle((cx, cy), radius=r, facecolor='none', linewidth=2)
ax0.add_patch(circle);
```

The circular fit is, indeed, excellent, and rejects all the inner
squiggly edges generated by solar turbulence!

Note a general principle here: algorithms that aggregate across an
entire path are often robust against noise.  Here, we have *high
uncertainty* in the solar edge, but also know that only the solar edge
pixels contribute coherently to the full circular path around the
solar edge.

## Exercises

Your small start-up, CardShark, run from your garage over nights and
evenings, takes photos of credit cards and turns them into machine
readable information.

The first step is to identify where in a photo the credit card is
located.

1. Load the photo `../../images/credit_card.jpg`
2. Using RANSAC and LineModelND shown above, find the first most
   prominent edge of the card
3. Remove the datapoints belonging to the most prominent edge, and
   repeat the process to find the second, third, and fourth



```python
f, ax = plt.subplots()

image = io.imread('../../images/credit_card.jpg')
ax.imshow(image);
```

----
### Warning: SPOILER ALERT.  Solution follows!
----

```python
f, ax = plt.subplots(figsize=(10, 10))

edges = feature.canny(color.rgb2gray(image), sigma=3)
edge_pts = np.array(np.nonzero(edges), dtype=float).T
edge_pts_xy = edge_pts[:, ::-1]

for i in range(4):
    model_robust, inliers = ransac(edge_pts_xy, LineModelND, min_samples=2,
                                   residual_threshold=1, max_trials=1000)

    x = np.arange(800)
    plt.plot(x, model_robust.predict_y(x))

    edge_pts_xy = edge_pts_xy[~inliers]

plt.imshow(edges);
```
