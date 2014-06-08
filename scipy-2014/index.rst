.. default-role:: literal

=================================
scikit-image tutorial, SciPy 2014
=================================


Software requirements:

- Python (>= 2.5)
- Numpy (>= 1.6)
- SciPy (>= 0.10)
- Matplotlib (>= 1.0, for examples only)
- Cython (>= 0.17, for development only)

Optional dependencies:

- Qt + PyQt4/PySide (for skimage.viewer and skivi)
- FreeImage (i/o plugin)
- PyAmg (for faster random-walker segmentation)
- Pillow/PIL (for some I/O functionality)
- Astropy (for FITS I/O)


Introduction
============

In our data-rich world, images represent a significant subset of all
measurements made. Examples include DNA microarrays, microscopy slides,
astronomical observations, satellite maps, robotic vision capture, synthetic
aperture radar images, and higher-dimensional images such as 3-D magnetic
resonance or computed tomography imaging. scikit-image (package name:
`skimage`) provides image processing tools in Python that are easy to use,
free of charge and restrictions, and able to address all the challenges posed
by such a diverse field of analysis. (Adapted from skimage-paper)


Image processing in Python
--------------------------

There are many image processing packages in the Python ecosystem. As
a scientific Python user, the Numpy array is our bread-and-butter, and it's
quite natural for scikit-image, and other packages, to represent images as
Numpy arrays. That said, not everyone who processes images in Python uses Numpy
arrays.


Non-numpy-array-based packages
..............................

`PIL`
  PIL (Python Imaging Library) was one of the first Python packages for
  image processing in Python. It's still a popular package for basic image
  I/O and manipulation.

`Pillow`
  Pillow is a fork of PIL that fixes many bugs in the original package, which
  is relatively unmaintained.

`OpenCV`
  OpenCV is a C++ image-processing library, but provides bindings for use
  in Python. It's focused on computer-vision, which is, to simplify things a
  bit, is a combination of image processing and machine learning.

`SimpleCV`
  SimpleCV adds an easier-to-use interface on top of OpenCV (plus more?)


Image-processing packages based on numpy arrays
...............................................

`scipy.ndimage`
  Image processing algorithms for n-dimensional arrays (e.g. 2D = "image",
  3D = volumes/data-cubes, ...)

`mahotas`
  Mahotas is an image processing library that embraces numpy arrays and uses
  it whenever possible. When algorithms need to drop down to lower-level
  code, those algorithms are implemented in C++.

`scikit-image`
  scikit-image is an image processing library that also embraces numpy arrays
  and uses it whenever possible. But, unlike `mahotas`, lower-level code is
  implemented in Cython.


Image-processing applications
.............................

Ilastic
  Image classification and segmentation

CellProfiler
  Cell image analysis software


The scikits and their relationship with scipy
.............................................

The goal of scikit-image (and the scikits, in general) is to extend the
functionality of scipy. (Smaller, more focused projects tend to evolve more
rapidly than larger ones.) It tries not to duplicate any functionality, and
only does so if it can improve upon that functionality. (Copied from my answer
to a StackOverflow question about scikit-image.)


The scikit-image subpackages
............................

(Descriptions taken from scikit-image docs)

`color`
    Color space conversion.
`data`
    Test images and example data.
`draw`
    Drawing primitives (lines, text, etc.) that operate on NumPy arrays.
`exposure`
    Image intensity adjustment, e.g., histogram equalization, etc.
`feature`
    Feature detection and extraction, e.g., texture analysis corners, etc.
`filter`
    Sharpening, edge finding, rank filters, thresholding, etc.
`graph`
    Graph-theoretic operations, e.g., shortest paths.
`io`
    Reading, saving, and displaying images and video.
`measure`
    Measurement of image properties, e.g., similarity and contours.
`morphology`
    Morphological operations, e.g., opening or skeletonization.
`novice`
    Simplified interface for teaching purposes.
`restoration`
    Restoration algorithms, e.g., deconvolution algorithms, denoising, etc.
`segmentation`
    Partitioning an image into multiple regions.
`transform`
    Geometric and other transforms, e.g., rotation or the Radon transform.
`util`
    Generic utilities.
`viewer`
    A simple graphical user interface for visualizing results and exploring
    parameters.

* Gallery

* Documentation


Image processing 101: The basics (45 mins)
==========================================


I/O
---

* images and different formats
* i/o backends
* image collections
* videos


data types
----------

* the basic data types
* dtypes and skimage functions

  - Under/overflow with uint8 images

* converting between data types
* basic array manipulation

  - slicing = cropping, resizing
  - simple math (esp, subtraction)


color
-----

* array-manipulation and colors
* color spaces
* Exercise: M&M detection


exposure
--------

* histogram
* contrast stretching
  - clip and display with `matplotlib`
  - rescale_intensity
* equalization
* Otsu thresholding


convolution and denoising
-------------------------

* Gaussian filter
* median filter
* rank-order filters?
* bilateral filtering


Edge detection (20 mins)
========================

* Sobel filtering
* Canny filter
* Exercise: Using an edge-preserving denoising filter instead of Gaussian in
  Canny filter


Feature detection
=================

* template matching
  - peak_local_max
* corners
* hough_line
* hough_circle
* contours
* texture
  - Gray-level co-occurence matrix (GLCM)
  - Gabor filter

Segmentation (30 mins)
======================

* rank filters
* watershed
* segmentation methods and visualization
* exercise


Morphology and measurement (40 mins)
====================================

* morphological operations for cleaning up segmented and binary images.
  - dilation, erosion, open, close
* labeled images
* measurement of region properties


Registration example (30 mins)
==============================

* corners (`skimage.feature`)
* detectors and descriptors
* geometric transforms of images (`skimage.transform`)
* texture
* detecting lines and circles using the hough transform
