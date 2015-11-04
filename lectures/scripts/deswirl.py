r"""
=====
Swirl
=====

Image swirling is a non-linear image deformation that creates a whirlpool
effect.

Image warping
`````````````
When applying a geometric transformation on an image, we typically make use of
a reverse mapping, i.e., for each pixel in the output image, we compute its
corresponding position in the input.  The reason is that, if we were to do it
the other way around (map each input pixel to its new output position), some
pixels in the output may be left empty.  On the other hand, each output
coordinate has exactly one corresponding location in (or outside) the input
image, and even if that position is non-integer, we may use interpolation to
compute the corresponding image value.

Performing a reverse mapping
````````````````````````````
To perform a geometric warp in ``skimage``, you simply need to provide the
reverse mapping to the ``skimage.transform.warp`` function.  E.g., consider the
case where we would like to shift an image 50 pixels to the left.  The reverse
mapping for such a shift would be::

    def shift_left(xy):
        xy[:, 0] += 50
        return xy

The corresponding call to warp is::

    from skimage.transform import warp
    warp(image, shift_left)

The swirl transformation
````````````````````````
Consider the coordinate :math:`(x, y)` in the output image.  The reverse
mapping for the swirl transformation first computes, relative to a center
:math:`(x_0, y_0)`, its polar coordinates,

.. math::

    \theta = \arctan(y/x)

    \rho = \sqrt{(x - x_0)^2 + (y - y_0)^2},

and then transforms them according to

.. math::

    r = \ln(2) \, \mathtt{radius} / 5

    \phi = \mathtt{rotation}

    s = \mathtt{strength}

    \theta' = \phi + s \, e^{-\rho / r + \theta}

where ``strength`` is a parameter for the amount of swirl, ``radius`` indicates
the swirl extent in pixels, and ``rotation`` adds a rotation angle.  The
transformation of ``radius`` into :math:`r` is to ensure that the
transformation decays to :math:`\approx 1/1000^{\mathsf{th}}` within the
specified radius.
"""

from __future__ import division

from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

import numpy as np
from scipy import ndimage
from skimage import io, transform


def _swirl_mapping(xy, center, rotation, strength, radius):
    """Compute the coordinate mapping for a swirl transformation.

    """
    x, y = xy.T
    x0, y0 = center
    rho = np.sqrt((x - x0)**2 + (y - y0)**2)

    # Ensure that the transformation decays to approximately 1/1000-th
    # within the specified radius.
    radius = radius / 5 * np.log(2)

    theta = rotation + strength * \
            np.exp(-rho / radius) + \
            np.arctan2(y - y0, x - x0)

    xy[..., 0] = x0 + rho * np.cos(theta)
    xy[..., 1] = y0 + rho * np.sin(theta)

    return xy

def swirl(image, center=None, strength=1, radius=100, rotation=0):
    """Perform a swirl transformation.

    Parameters
    ----------
    image : ndarray
        Input image.
    center : (x,y) tuple or (2,) ndarray
        Center coordinate of transformation.
    strength : float
        The amount of swirling applied.
    radius : float
        The extent of the swirl in pixels.  The effect dies out
        rapidly beyond `radius`.
    rotation : float
        Additional rotation applied to the image.

    Returns
    -------
    swirled : ndarray
        Swirled version of the input.

    """

    if center is None:
        center = np.array(image.shape)[:2] / 2

    warp_args = {'center': center,
                 'rotation': rotation,
                 'strength': strength,
                 'radius': radius}

    return transform.warp(image, _swirl_mapping, map_args=warp_args)


# Read the input image, and compute its center
mona = io.imread('../../images/mona_lisa.jpg')
h, w, d = mona.shape
center = np.array([w/2, h/2])

# Construct three outputs: input image, swirled and deswirled
f, (ax0, ax1, ax2) = plt.subplots(1, 3)
plt.subplots_adjust(bottom=0.5)

# Swirl the input image with fixed parameters
mona_swirled = swirl(mona, center=center, rotation=0, strength=10, radius=100)

source = ax0.imshow(mona, interpolation='nearest')
ax0.set_title('Click to move\nthe red dot\n(the transform center)')
ax0.set_xlabel('Original Mona Lisa')

swirled = ax1.imshow(mona_swirled, interpolation='nearest')
ax1.set_xlabel('Swirled Mona Lisa')

deswirled = ax2.imshow(mona_swirled, interpolation='nearest')
ax2.set_xlabel('Restored using\nyour choice of\nparameters')

# Plot a dot to indicate the center-point of the reverse transform
center += [10, -5]
center_dot, = ax0.plot(center[0], center[1], 'ro')
ax0.axis('image')

def update(event=None):
    """This function will be executed each time the interactive sliders are
    changed or when clicking the input image to adjust the center-point.  It
    reads the new parameters, and performs the deswirl accordingly.

    Note that the swirl is always performed using a fixed center, strength and
    radius, so that you can investigate the sensitivity of the inverse
    transform with regards to the parameters.

    """
    # Mouse click detected on input image -- set center position
    if hasattr(event, 'inaxes') and event.inaxes is ax0:
        center[:] = [event.xdata, event.ydata]

    # Perform deswirl and update the output image
    out_deswirl = swirl(mona_swirled,
                        center=center, rotation=-np.deg2rad(rotation.val),
                        strength=-strength.val, radius=radius.val)

    deswirled.set_data(out_deswirl)

    # Re-position the center dot according to the clicked position
    center_dot.set_xdata(center[0])
    center_dot.set_ydata(center[1])

    plt.draw()

# Set up the parameter sliders
ax_rotation = plt.axes([0.25, 0.15, 0.65, 0.03])
rotation = Slider(ax_rotation, 'Rotation', 0, 360, valinit=0)
ax_strength = plt.axes([0.25, 0.25, 0.65, 0.03])
strength = Slider(ax_strength, 'Strength', -50, 50, valinit=10+10)
ax_radius = plt.axes([0.25, 0.35, 0.65, 0.03])
radius = Slider(ax_radius, 'Radius', 0, 250, valinit=100-20)

# Trigger an update whenever the parameters change
rotation.on_changed(update)
strength.on_changed(update)
radius.on_changed(update)

# Also trigger an update whenever the mouse is clicked on the input image
# (setting the center point)
f.canvas.mpl_connect('button_press_event', update)

# Do a single update when we start the program
update(None)

plt.show()
