from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import grey_dilation

from skimage import img_as_float
from skimage import color
from skimage import exposure
from skimage.util.dtype import dtype_limits


__all__ = ['imshow_all', 'imshow_with_histogram', 'mean_filter_demo',
           'mean_filter_interactive_demo', 'plot_cdf', 'plot_histogram']


# Gray-scale images should actually be gray!
plt.rcParams['image.cmap'] = 'gray'


#--------------------------------------------------------------------------
#  Custom `imshow` functions
#--------------------------------------------------------------------------

def imshow_rgb_shifted(rgb_image, shift=100, ax=None):
    """Plot each RGB layer with an x, y shift."""
    if ax is None:
        ax = plt.gca()

    height, width, n_channels = rgb_image.shape
    x = y = 0
    for i_channel, channel in enumerate(iter_channels(rgb_image)):
        image = np.zeros((height, width, n_channels), dtype=channel.dtype)

        image[:, :, i_channel] = channel
        ax.imshow(image, extent=[x, x+width, y, y+height], alpha=0.7)
        x += shift
        y += shift
    # `imshow` fits the extents of the last image shown, so we need to rescale.
    ax.autoscale()
    ax.set_axis_off()


def imshow_all(*images, **kwargs):
    """ Plot a series of images side-by-side.

    Convert all images to float so that images have a common intensity range.

    Parameters
    ----------
    limits : str
        Control the intensity limits. By default, 'image' is used set the
        min/max intensities to the min/max of all images. Setting `limits` to
        'dtype' can also be used if you want to preserve the image exposure.
    titles : list of str
        Titles for subplots. If the length of titles is less than the number
        of images, empty strings are appended.
    kwargs : dict
        Additional keyword-arguments passed to `imshow`.
    """
    images = [img_as_float(img) for img in images]

    titles = kwargs.pop('titles', [])
    if len(titles) != len(images):
        titles = list(titles) + [''] * (len(images) - len(titles))

    limits = kwargs.pop('limits', 'image')
    if limits == 'image':
        kwargs.setdefault('vmin', min(img.min() for img in images))
        kwargs.setdefault('vmax', max(img.max() for img in images))
    elif limits == 'dtype':
        vmin, vmax = dtype_limits(images[0])
        kwargs.setdefault('vmin', vmin)
        kwargs.setdefault('vmax', vmax)

    nrows, ncols = kwargs.get('shape', (1, len(images)))

    size = nrows * kwargs.pop('size', 5)
    width = size * len(images)
    if nrows > 1:
        width /= nrows * 1.33
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, size))
    for ax, img, label in zip(axes.ravel(), images, titles):
        ax.imshow(img, **kwargs)
        ax.set_title(label)


def imshow_with_histogram(image, **kwargs):
    """ Plot an image side-by-side with its histogram.

    - Plot the image next to the histogram
    - Plot each RGB channel separately (if input is color)
    - Automatically flatten channels
    - Select reasonable bins based on the image's dtype

    See `plot_histogram` for information on how the histogram is plotted.
    """
    width, height = plt.rcParams['figure.figsize']
    fig, (ax_image, ax_hist) = plt.subplots(ncols=2, figsize=(2*width, height))

    kwargs.setdefault('cmap', plt.cm.gray)
    ax_image.imshow(image, **kwargs)
    plot_histogram(image, ax=ax_hist)

    # pretty it up
    ax_image.set_axis_off()
    match_axes_height(ax_image, ax_hist)
    return ax_image, ax_hist


#--------------------------------------------------------------------------
#  Helper functions
#--------------------------------------------------------------------------


def match_axes_height(ax_src, ax_dst):
    """ Match the axes height of two axes objects.

    The height of `ax_dst` is synced to that of `ax_src`.
    """
    # HACK: plot geometry isn't set until the plot is drawn
    plt.draw()
    dst = ax_dst.get_position()
    src = ax_src.get_position()
    ax_dst.set_position([dst.xmin, src.ymin, dst.width, src.height])


def plot_cdf(image, ax=None):
    img_cdf, bins = exposure.cumulative_distribution(image)
    ax.plot(bins, img_cdf, 'r')
    ax.set_ylabel("Fraction of pixels below intensity")


def plot_histogram(image, ax=None, **kwargs):
    """ Plot the histogram of an image (gray-scale or RGB) on `ax`.

    Calculate histogram using `skimage.exposure.histogram` and plot as filled
    line. If an image has a 3rd dimension, assume it's RGB and plot each
    channel separately.
    """
    ax = ax if ax is not None else plt.gca()

    if image.ndim == 2:
        _plot_histogram(ax, image, color='black', **kwargs)
    elif image.ndim == 3:
        # `channel` is the red, green, or blue channel of the image.
        for channel, channel_color in zip(iter_channels(image), 'rgb'):
            _plot_histogram(ax, channel, color=channel_color, **kwargs)


def _plot_histogram(ax, image, alpha=0.3, **kwargs):
    # Use skimage's histogram function which has nice defaults for
    # integer and float images.
    hist, bin_centers = exposure.histogram(image)
    ax.fill_between(bin_centers, hist, alpha=alpha, **kwargs)
    ax.set_xlabel('intensity')
    ax.set_ylabel('# pixels')


def iter_channels(color_image):
    """Yield color channels of an image."""
    # Roll array-axis so that we iterate over the color channels of an image.
    for channel in np.rollaxis(color_image, -1):
        yield channel


#--------------------------------------------------------------------------
#  Convolution Demo
#--------------------------------------------------------------------------

def mean_filter_demo(image, vmax=1):
    mean_factor = 1.0 / 9.0  # This assumes a 3x3 kernel.
    iter_kernel_and_subimage = iter_kernel(image)

    image_cache = []

    def mean_filter_step(i_step):
        while i_step >= len(image_cache):
            filtered = image if i_step == 0 else image_cache[-1][1]
            filtered = filtered.copy()

            (i, j), mask, subimage = iter_kernel_and_subimage.next()
            filter_overlay = color.label2rgb(mask, image, bg_label=0,
                                             colors=('yellow', 'red'))
            filtered[i, j] = np.sum(mean_factor * subimage)
            image_cache.append((filter_overlay, filtered))

        imshow_all(*image_cache[i_step], vmax=vmax)
        plt.show()
    return mean_filter_step


def mean_filter_interactive_demo(image):
    from IPython.html import widgets
    mean_filter_step = mean_filter_demo(image)
    step_slider = widgets.IntSliderWidget(min=0, max=image.size-1, value=0)
    widgets.interact(mean_filter_step, i_step=step_slider)


def iter_kernel(image, size=1):
    """ Yield position, kernel mask, and image for each pixel in the image.

    The kernel mask has a 2 at the center pixel and 1 around it. The actual
    width of the kernel is 2*size + 1.
    """
    width = 2*size + 1
    for (i, j), pixel in iter_pixels(image):
        mask = np.zeros(image.shape, dtype='int16')
        mask[i, j] = 1
        mask = grey_dilation(mask, size=width)
        mask[i, j] = 2
        subimage = image[bounded_slice((i, j), image.shape[:2], size=size)]
        yield (i, j), mask, subimage


def iter_pixels(image):
    """ Yield pixel position (row, column) and pixel intensity. """
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            yield (i, j), image[i, j]


def bounded_slice(center, xy_max, size=1, i_min=0):
    slices = []
    for i, i_max in zip(center, xy_max):
        slices.append(slice(max(i - size, i_min), min(i + size + 1, i_max)))
    return slices
