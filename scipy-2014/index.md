# Image analysis in Python with SciPy and scikit-image

From telescopes to satellite cameras to electron microscopes, scientists are
producing more images than they can manually inspect. This tutorial will
introduce automated image analysis using the "images as numpy arrays"
abstraction, run through various fundamental image analysis operations
(filters, morphology, segmentation), and finally complete one or two more
advanced real-world examples.

Image analysis is central to a boggling number of scientific endeavors. Google
needs it for their self-driving cars and to match satellite imagery and mapping
data. Neuroscientists need it to understand the brain. NASA needs it to [map
asteroids](http://www.bbc.co.uk/news/technology-26528516) and save the human
race. It is, however, a relatively underdeveloped area of scientific computing.
Attendees will leave this tutorial confident of their ability to extract
information from their images in Python.

# Prerequisites

All of the below packages, including the non-Python ones, can be found in the
[Anaconda](https://store.continuum.io/cshop/anaconda/) Python distribution,
which can (and should) be obtained for free. (Though some may need
`conda install`ing.)

## Required packages

- scikit-image (0.10 or higher)

Required for scikit-image:

- Python (>=2.5 required, 2.7 recommended)
- numpy (>=1.6 required, 1.7 recommended)
- scipy (>=0.10 required, 0.13 recommended)

Required for image viewing and other examples:

- matplotlib (>=1.0 required, 1.2 recommended)

Required for skimage.viewer and skivi interactive examples

- Qt
- PyQt4/PySide

Required for development:

- cython (>=0.16 required, 0.19 recommended)

Recommended for IO:

- FreeImage
- Pillow/PIL

Recommended:

PyAmg (Fast random-walker segmentation)

## Example images

scikit-image ships with some example images in `skimage.data`. For this
tutorial, however, we will make use of additional images that you can download
here:

LINK

# Introduction: images are numpy arrays

A grayscale image is just a 2D array:

```python
import numpy as np
r = np.random.rand(500, 500)
from matplotlib import pyplot as plt, cm
plt.imshow(r, cmap=cm.gray, interpolation='nearest')
```

Trying that with a real picture:

```python
from skimage import data
coins = data.coins()
print(type(coins))
print(coins.dtype)
print(coins.shape)
plt.imshow(coins, cmap=cm.gray, interpolation='nearest')
```

A color image is a 3D array, where the last dimension has size 3 and represents
the red, green, and blue channels:

```python
lena = data.lena()
print(lena.shape)
plt.imshow(lena, interpolation='nearest')
```

These are _just numpy arrays_. making a red square is easy using just array
slicing and manipulation:

```python
lena[100:200, 100:200, :] = [255, 0, 0] # [red, green, blue]
plt.imshow(lena)
```

As we will see, this opens up many lines of analysis for free.

## Exercise: draw an H

Define a function that takes as input an RGB image and a pair of coordinates
(row, column), and returns the image (optionally a copy) with green letter H
overlaid at those coordinates. The coordinates should point to the top-left
corner of the H.

The arms and strut of the H should have a width of 2 pixels, and the H itself
should have a height of 12 pixels and width of 10 pixels.

```python
def draw_h(image, coords, in_place=True):
    pass # code goes here
```

Test your function like so:

```python
lena_h = draw_h(lena, (50, -60), in_place=False)
plt.imshow(lena)
```

## Bonus points: RGB intensity plot

Plot the intensity of each channel of the image along some row.

# Image analysis fundamentals 0: colors, exposure, and contrast (Tony)

# Image analysis fundamentals 1: filters and convolution (Tony)

Explain filters (slides or drawing on board): simple difference filter. Sobel
filter. Gradient magnitude. Since filters produce floating point values, take
opportunity to introduce data types used by scikit-image.

# Image analysis fundamentals 2: feature detection (Tony)

- Canny filter, Corner detection, Hough transforms

# Image analysis fundamentals 3: morphological operations (Juan)

- Demonstrate erosion, dilation, opening, closing on tiny example images

Morphology is the study of shapes. In image processing, some simple operations
can get you a long way. The first thing to learn is *erosion* and *dilation*.
In erosion, we look at a pixel's local neighborhood and replace the value of
that pixel with the minimum value of that neighborhood. In dilation, we instead
choose the maximum.

```python
import numpy as np
image = np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]], dtype=np.uint8)
from skimage import morphology
```

The documentation for scikit-image's morphology module is
[here](http://scikit-image.org/docs/0.10.x/api/skimage.morphology.html).

Importantly, we must use a *structuring element*, which defines the local
neighborhood of each pixel. To get every neighbor (up, down, left, right, and
diagonals), use `morphology.square`; to avoid diagonals, use
`morphology.diamond`:

```python
sq = morphology.square(width=3)
dia = morphology.diamond(radius=1)
```

The central value of the structuring element represents the pixel being
considered, and the surrounding values are the neighbors: a 1 value means that
pixel counts as a neighbor, while a 0 value does not. So:

```python
morphology.erosion(image, sq)
```

and

```python
morphology.dilation(image, sq)
```

and

```python
morphology.dilation(image, dia)
```

Erosion and dilation can be combined into two slightly more sophisticated
operations, *opening* and *closing*. Here's an example:

```python
image = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0, 0, 1, 0],
                  [0, 1, 1, 1, 0, 0, 1, 0],
                  [0, 1, 1, 1, 0, 0, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]], np.uint8)
```

What happens when run an erosion followed by a dilation of this image?

What about the reverse?

Exercise: use morphological operations to remove speckle from an image of a
galaxy.

# Image analysis fundamentals 4: segmentation (Juan)

# Advanced examples 0: measuring fluorescence intensity on chromosomes (Juan)

# Advanced examples 1: measuring line profile intensity (Juan)

# Advanced examples 2: interactive analysis with the viewer (Tony)

- Make sure to mention the new overlay/data output function in 0.10!

# Advanced examples 3: BYO!
