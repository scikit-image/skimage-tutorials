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

Segmentation is the division of an image into "meaningful" regions. If you've
seen The Terminator, you've seen image segmentation:

![Terminator vision](images/terminator-vision.png)

In `scikit-image`, you can find segmentation functions in the `segmentation`
package (oddly enough), with one exception: the `watershed` function is in
`morphology`, because it's a bit of both. We'll use two algorithms, SLIC and
watershed, and just discuss the rest, and applications of each.

There are two kinds of segmentation: *contrast-based* and *boundary-based*. The
first is used when the regions of the image you are trying to divide have
different characteristics, such as a red flower on a green background. The
second is used when you want to segment an image in which borders between
objects are prominent, but objects themselves are not very distinct. For
example, a pile of oranges.

## Image types: contrast

SLIC is a segmentation algorithm of the first kind: it's clustering pixels in
both space and color. (Simple Linear Iterative Clustering.) Therefore, regions
of space that are similar in color will end up in the same segment.

Let's try to segment this image:

![Spice, by Clyde
Robinson](https://farm6.staticflickr.com/5252/5519129659_02be0e5011_o_d.jpg)

(Photo by Flickr user Clyde Robinson, used under CC-BY 2.0 license.)

The SLIC function takes two parameters: the desired number of segments, and the
"compactness", which is the relative weighting of the space and color
dimensions. The higher the compactness, the more "square" the returned
segments.

(I will probably make the section below interactive.)

```python
import skdemo
from skimage import io, segmentation as seg, color
url = 'https://farm6.staticflickr.com/5252/5519129659_02be0e5011_o_d.jpg'
image = io.imread(url)
labels = seg.slic(image, n_segments=18, compactness=10)
label_image = color.label2rgb(labels, image, kind='avg')
skdemo.imshow_all(image, label_image)
```

Notice that some spices are broken up into "light" and "dark" parts. We have
multiple parameters to control this:

- `enforce_connectivity`: Do some post-processing so that small regions get
  merged to adjacent big regions.

```python
labels = seg.slic(image, n_segments=18, compactness=10,
                  enforce_connectivity=True)
label_image = color.label2rgb(labels, image, kind='avg')
skdemo.imshow_all(image, label_image)
```

Yikes! It looks like a little too much merging went on! This is because of the
intertwining of the labels. One way to avoid this is to blur the image before
segmentation. Because this is so useful, a Gaussian blur is included in SLIC:
just pass in the `sigma` parameter:

```python
labels = seg.slic(image, n_segments=18, compactness=10,
                  sigma=2, enforce_connectivity=True)
label_image = color.label2rgb(labels, image, kind='avg')
skdemo.imshow_all(image, label_image)
```

Getting there! But it looks like some regions are merged together. We can
alleviate this by increasing the number of segments:

```python
labels = seg.slic(image, n_segments=24, compactness=10,
                  sigma=2, enforce_connectivity=True)
label_image = color.label2rgb(labels, image, kind='avg')
skdemo.imshow_all(image, label_image)
```

That's looking pretty good! Some regions are still too squiggly though... Let's
try jacking up the compactness:


```python
labels = seg.slic(image, n_segments=24, compactness=40,
                  sigma=2, enforce_connectivity=True)
label_image = color.label2rgb(labels, image, kind='avg')
skdemo.imshow_all(image, label_image)
```

**Exercise**: Try segmenting the following image:

```python
url2 = 'https://farm4.staticflickr.com/3557/3326786046_8647d993db_b_d.jpg'
image = io.imread(url2)
```

Note: this image is more challenging to segment because the color regions are
different from one part of the image to the other. Try the `slic_zero`
parameter in combination with different values for `n_segments`.


## Image types: boundary images

Often, the contrast between regions is not sufficient to distinguish them, but
there is a clear boundary between the two. Using an edge detector on these
images, followed by a *watershed*, often gives very good segmentation. For
example, look at the output of the Sobel filter on the coins image:

```python
from skimage import data, filter as filters
from matplotlib import pyplot as plt, cm
coins = data.coins()
edges = filters.sobel(coins)
plt.imshow(edges, cmap=cm.cubehelix)
```

The *watershed algorithm* finds the regions between these edges. It does so by
envisioning the pixel intensity as height on a topographic map. It then
"floods" the map from the bottom up, starting from seed points. These flood
areas are called "watershed basins" and when they meet, they form the image
segmentation.

Let's look at a one-dimensional example:

```python
from skimage.morphology import watershed
image = np.array([[1, 0, 1, 2, 1, 3, 2, 0, 2, 4, 1, 0]])
seeds = nd.label(image == 0)[0]
plt.plot(image[0])
plt.plot(watershed(image, seeds))
```
(I'm intending to plot the lines in different colors for different basins,
marking the seeds.)

Let's find some seeds for `coins`:

```python
from scipy import ndimage as nd
threshold = 0.4
distance_from_edge = nd.distance_transform_edt(edges < threshold)
from skimage import features
peaks = features.peak_local_max(distance_from_edge)
seeds, num_seeds = nd.label(peaks)
ws = watershed(edges, seeds)
from skimage import color
plt.imshow(color.label2rgb(ws, coins))
```

**Exercise**: We can see that watershed gives a very good segmentation, but
some coins are missing. Why? Can you suggest better seed points for the
watershed operation?

# Advanced examples 0: measuring fluorescence intensity on chromosomes (Juan)

# Advanced examples 1: measuring line profile intensity (Juan)

# Advanced examples 2: interactive analysis with the viewer (Tony)

- Make sure to mention the new overlay/data output function in 0.10!

# Advanced examples 3: BYO!
