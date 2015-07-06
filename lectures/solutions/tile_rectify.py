from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from skimage import transform


from skimage.transform import estimate_transform

source = np.array([(129, 72),
                   (302, 76),
                   (90, 185),
                   (326, 193)])

target = np.array([[0, 0],
                   [400, 0],
                   [0, 400],
                   [400, 400]])

tf = estimate_transform('projective', source, target)
H = tf.params   # in older versions of skimage, this should be
                # H = tf._matrix

print(H)

# H = np.array([[  3.04026872e+00,   1.04929628e+00,  -4.67743998e+02],
#               [ -1.44134582e-01,   6.23382067e+00,  -4.30241727e+02],
#               [  2.63620673e-05,   4.17694527e-03,   1.00000000e+00]])

def rectify(xy):
    x = xy[:, 0]
    y = xy[:, 1]

    # You must fill in your code here.
    #
    # Handy functions are:
    #
    # - np.dot (matrix multiplication)
    # - np.ones_like (make an array of ones the same shape as another array)
    # - np.column_stack
    # - A.T -- type .T after a matrix to transpose it
    # - x.reshape -- reshapes the array x

    # We need to provide the backward mapping
    HH = np.linalg.inv(H)

    homogeneous_coordinates = np.column_stack([x, y, np.ones_like(x)])
    xyz = np.dot(HH, homogeneous_coordinates.T)

    # We want one coordinate per row
    xyz = xyz.T

    # Turn z into a column vector
    z = xyz[:, 2]
    z = z.reshape([len(z), 1])

    xyz = xyz / z

    return xyz[:, :2]

image = plt.imread('../../images/chapel_floor.png')
out = transform.warp(image, rectify, output_shape=(400, 400))

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
ax0.imshow(image)
ax1.imshow(out)

plt.show()
