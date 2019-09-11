import matplotlib.pyplot as plt
import numpy as np

from ipywidgets import interact
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import exposure, io, measure


def show_plane(axis, plane, cmap="gray", title=None):
    """Shows a specific plane within 3D data.
    """
    axis.imshow(plane, cmap=cmap)
    axis.set_xticks([])
    axis.set_yticks([])

    if title:
        axis.set_title(title)

    return None


def slice_in_3d(axis, shape, plane):
    """Draws a cube in a 3D plot.

    Parameters
    ----------
    axis : matplotlib.Axes
        A matplotlib axis to be drawn.
    shape : tuple or array (1, 3)
        Shape of the input data.
    plane : int
        Number of the plane to be drawn.

    Notes
    -----
    Originally from:
    https://stackoverflow.com/questions/44881885/python-draw-parallelepiped
    """
    Z = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 1],
                  [1, 1, 1],
                  [0, 1, 1]])

    Z = Z * shape

    r = [-1, 1]

    X, Y = np.meshgrid(r, r)

    # plotting vertices
    axis.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

    # list of sides' polygons of figure
    verts = [[Z[0], Z[1], Z[2], Z[3]],
             [Z[4], Z[5], Z[6], Z[7]],
             [Z[0], Z[1], Z[5], Z[4]],
             [Z[2], Z[3], Z[7], Z[6]],
             [Z[1], Z[2], Z[6], Z[5]],
             [Z[4], Z[7], Z[3], Z[0]],
             [Z[2], Z[3], Z[7], Z[6]]]

    # plotting sides
    axis.add_collection3d(
        Poly3DCollection(verts,
                         facecolors=(0, 1, 1, 0.25),
                         linewidths=1,
                         edgecolors='darkblue')
    )

    verts = np.array([[[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 1],
                       [0, 1, 0]]])
    verts = verts * shape
    verts += [plane, 0, 0]

    axis.add_collection3d(
        Poly3DCollection(verts,
                         facecolors='magenta',
                         linewidths=1,
                         edgecolors='black')
    )

    axis.set_xlabel('plane')
    axis.set_ylabel('col')
    axis.set_zlabel('row')

    # auto-scale plot axes
    scaling = np.array([getattr(axis, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    axis.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]] * 3)

    return None


def slice_explorer(data, cmap='gray'):
    """Allows to explore 2D slices in 3D data.

    Parameters
    ----------
    data : array (M, N, P)
        3D interest image.
    cmap : str (optional)
        A string referring to one of matplotlib's colormaps.
    """
    data_len = len(data)

    @interact(plane=(0, data_len-1), continuous_update=False)
    def display_slice(plane=data_len/2):
        fig, axis = plt.subplots(figsize=(20, 7))
        axis_3d = fig.add_subplot(133, projection='3d')
        show_plane(axis, data[plane], title='Plane {}'.format(plane), cmap=cmap)
        slice_in_3d(axis=axis_3d, shape=data.shape, plane=plane)
        plt.show()

    return display_slice


def display(data, cmap="gray", step=2):
    _, axes = plt.subplots(nrows=5, ncols=6, figsize=(16, 14))

    # getting data min and max to plot limits
    vmin, vmax = data.min(), data.max()

    for axis, image in zip(axes.flatten(), data[::step]):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_xticks([])
        axis.set_yticks([])

    return None


def plot_hist(axis, data, title=None):
    """Helper function for plotting histograms.
    """
    axis.hist(data.ravel(), bins=256)
    axis.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    if title:
        axis.set_title(title)

    return None


def plot_3d_surface(data, labels, region=3, spacing=(1.0, 1.0, 1.0)):
    """Generates a 3D surface plot for the specified region.

    Parameters
    ----------
    data : array (M, N, P)
        3D interest image.
    labels : array (M, N, P)
        Labels corresponding to data, obtained by measure.label.
    region : int, optional
        The region of interest to be plotted.
    spacing : array (1, 3)
        Spacing information, set distances between pixels.

    Notes
    -----
    The volume is visualized using the mesh vertexes and faces.
    """
    properties = measure.regionprops(labels, intensity_image=data)
    # skimage.measure.marching_cubes expects ordering (row, col, plane).
    # We need to transpose the data:
    volume = (labels == properties[region].label).transpose(1, 2, 0)

    verts_px, faces_px, _, _ = measure.marching_cubes_lewiner(volume, level=0, spacing=(1.0, 1.0, 1.0))
    surface_area_pixels = measure.mesh_surface_area(verts_px, faces_px)

    verts_actual, faces_actual, _, _ = measure.marching_cubes_lewiner(volume, level=0, spacing=tuple(spacing))
    surface_area_actual = measure.mesh_surface_area(verts_actual, faces_actual)

    print('Surface area\n')
    print(' * Total pixels: {:0.2f}'.format(surface_area_pixels))
    print(' * Actual: {:0.2f}'.format(surface_area_actual))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts_px[faces_px])
    mesh.set_edgecolor('black')
    ax.add_collection3d(mesh)

    ax.set_xlabel('col')
    ax.set_ylabel('row')
    ax.set_zlabel('plane')

    min_pln, min_row, min_col, max_pln, max_row, max_col = properties[region].bbox

    ax.set_xlim(min_row, max_row)
    ax.set_ylim(min_col, max_col)
    ax.set_zlim(min_pln, max_pln)

    plt.tight_layout()
    plt.show()

    return None


def results_from_part_1():

    data = io.imread("images/cells.tif")

    vmin, vmax = np.percentile(data, q=(0.5, 99.5))
    rescaled = exposure.rescale_intensity(
        data,
        in_range=(vmin, vmax),
        out_range=np.float32
    )

    equalized = exposure.equalize_hist(data)

    return data, rescaled, equalized
