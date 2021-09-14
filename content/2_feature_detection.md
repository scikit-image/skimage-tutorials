---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} python
from __future__ import division, print_function
%matplotlib inline
```

+++ {"slideshow": {"slide_type": "slide"}}

# Feature detection

+++ {"slideshow": {"slide_type": "notes"}}

Feature detection is often the end result of image processing. We'll detect some basic features (edges, points, and circles) in this section, but there are a wealth of feature detectors that are available.

+++ {"slideshow": {"slide_type": "slide"}}

## Edge detection

+++ {"slideshow": {"slide_type": "notes"}}

Before we start, let's set the default colormap to grayscale and turn off pixel interpolation.

```{code-cell} python
---
slideshow:
  slide_type: skip
---
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'none'
```

+++ {"slideshow": {"slide_type": "notes"}}

We've already discussed edge filtering, using the Sobel filter, in the last section.

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
import skdemo
from skimage import data
# Rename module so we don't shadow the builtin function
import skimage.filter as filters

image = data.camera()
pixelated = image[::10, ::10]
gradient = filters.sobel(pixelated)
skdemo.imshow_all(pixelated, gradient)
```

+++ {"slideshow": {"slide_type": "notes"}}

With the Sobel filter, however, we get back a grayscale image, which essentially tells us the likelihood that a pixel is on the edge of an object.

We can apply a bit more logic to *detect* an edge; i.e. we can use that filtered image to make a *decision* whether or not a pixel is on an edge. The simplest way to do that is with thresholding:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
skdemo.imshow_all(gradient, gradient > 0.4)
```

+++ {"slideshow": {"slide_type": "notes"}}

That approach doesn't do a great job. It's noisy and produces thick edges. Furthermore, it doesn't use our *knowledge* of how edges work: They should be thin and tend to be connected along the direction of the edge.

+++ {"slideshow": {"slide_type": "fragment"}}

### Canny edge detector

+++ {"slideshow": {"slide_type": "fragment"}}

The Canny edge detector combines the Sobel filter with a few other steps to give a binary edge image. The steps are as follows:
* Gaussian filter
* Sobel filter
* Non-maximal suppression
* Hysteresis thresholding

+++ {"slideshow": {"slide_type": "notes"}}

Let's go through these steps one at a time.

+++ {"slideshow": {"slide_type": "slide"}}

### Step 1: Gaussian filter

+++ {"slideshow": {"slide_type": "notes"}}

As discussed earlier, gradients tend to enhance noise. To combat this effect, we first smooth the image using a gradient filter:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
from skimage import img_as_float

sigma = 1  # Standard-deviation of Gaussian; larger smooths more.
pixelated_float = img_as_float(pixelated)
pixelated_float = pixelated
smooth = filters.gaussian_filter(pixelated_float, sigma)
skdemo.imshow_all(pixelated_float, smooth)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Step 2: Sobel filter

+++ {"slideshow": {"slide_type": "notes"}}

Next, we apply our edge filter:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
gradient_magnitude = filters.sobel(smooth)
skdemo.imshow_all(smooth, gradient_magnitude)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Step 3: Non-maximal suppression

+++ {"slideshow": {"slide_type": "fragment"}}

Goal: Suppress gradients that aren't on an edge

+++ {"slideshow": {"slide_type": "notes"}}

Ideally, an edge is thin: In some sense, an edge is infinitely thin, but images are discrete so we want edges to be a single pixel wide. To accomplish this, we thin the image using "non-maximal suppression". This takes the edge-filtered image and thins the response *across* the edge direction; i.e. in the direction of the maximum gradient:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
zoomed_grad = gradient_magnitude[15:25, 5:15]
maximal_mask = np.zeros_like(zoomed_grad)
# This mask is made up for demo purposes
maximal_mask[range(10), (7, 6, 5, 4, 3, 2, 2, 2, 3, 3)] = 1
grad_along_edge = maximal_mask * zoomed_grad
skdemo.imshow_all(zoomed_grad, grad_along_edge, limits='dtype')
```

+++ {"slideshow": {"slide_type": "notes"}}

Obviously, this is a faked version of non-maximal suppression: Pixels are *manually* masked here. The actual algorithm detects the direction of edges, and keeps a pixel only if it has a locally maximal gradient-magnitude in the direction *normal to the edge direction*. It doesn't mask pixels *along* the edge direction since an adjacent edge pixel will be of comparable magnitude.

The result of the filter is that an edge is only possible if there are no better edges near it.

+++ {"slideshow": {"slide_type": "slide"}}

### Step 4: Hysteresis thresholding

+++ {"slideshow": {"slide_type": "fragment"}}

Goal: Prefer pixels that are connected to edges

+++ {"slideshow": {"slide_type": "notes"}}

The final step is the actual decision-making process. Here, we have two parameters: The low threshold and the high threshold. The high threshold sets the gradient value that you *know* is definitely an edge. The low threshold sets the gradient value that could be an edge, but only if it's connected to a pixel that we know is an edge.

These two thresholds are displayed below:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
from skimage import color

low_threshold = 0.2
high_threshold = 0.3
label_image = np.zeros_like(pixelated)
# This uses `gradient_magnitude` which has NOT gone through non-maximal-suppression.
label_image[gradient_magnitude > low_threshold] = 1
label_image[gradient_magnitude > high_threshold] = 2
demo_image = color.label2rgb(label_image, gradient_magnitude,
                             bg_label=0, colors=('yellow', 'red'))
plt.imshow(demo_image)
```

+++ {"slideshow": {"slide_type": "notes"}}

The **red points** here are above `high_threshold` and are seed points for edges. The **yellow points** are edges if connected (possibly by other yellow points) to seed points; i.e. isolated groups of yellow points will not be detected as edges.

Note that the demo above is on the edge image *before* non-maximal suppression, but in reality, this would be done on the image *after* non-maximal suppression. There isn't currently an easy way to get at the intermediate result.

+++ {"slideshow": {"slide_type": "slide"}}

### The Canny edge detector

+++ {"slideshow": {"slide_type": "notes"}}

Now we're ready to look at the actual result:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
from IPython.html import widgets
from skimage import data

image = data.coins()

def canny_demo(**kwargs):
    edges = filters.canny(image, **kwargs)
    plt.imshow(edges)
    plt.show()
# As written, the following doesn't actually interact with the
# `canny_demo` function. Figure out what you need to add.
widgets.interact(canny_demo);  # <-- add keyword arguments for `canny`
```

+++ {"slideshow": {"slide_type": "notes"}}

Play around with the demo above. Make sure to add any keyword arguments to `interact` that might be necessary. (Note that keyword arguments passed to `interact` are passed to `canny_demo` and forwarded to `filter.canny`. So you should be looking at the docstring for `filter.canny` or the discussion above to figure out what to add.)

+++ {"slideshow": {"slide_type": "fragment"}}

Can you describe how the low threshold makes a decision about a potential edge, as compared to the high threshold?

+++ {"slideshow": {"slide_type": "slide"}}

## Aside: Feature detection in research

+++ {"slideshow": {"slide_type": "notes"}}

When taking image data for an experiment, the end goal is often to detect some sort of feature. Here are a few examples from some of my own research.

+++ {"slideshow": {"slide_type": "slide"}}

### Feature detection: case study

+++ {"slideshow": {"slide_type": "notes"}}

I got started with image processing when I needed to detect the position of a device I built to study swimming in viscous environments (low-Reynolds numbers):

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
from IPython.display import Image, YouTubeVideo

YouTubeVideo('1Pjlj9Pymx8')
```

+++ {"slideshow": {"slide_type": "slide"}}

### Particle detection

+++ {"slideshow": {"slide_type": "notes"}}

For my post-doc, I ended up investigating the collection of fog from the environment and built the apparatus displayed below:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
from IPython.display import Image, YouTubeVideo

Image(filename='images/fog_tunnel.png')
```

+++ {"slideshow": {"slide_type": "notes"}}

The resulting experiments looked something like this:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
YouTubeVideo('14qlyhnyjT8')
```

+++ {"slideshow": {"slide_type": "notes"}}

The water droplets in the video can be detected using some of the features in `scikit-image`. In particular, `peak_local_max` from the `feature` package is useful here. (We'll play with this function in the Hough transform section.) There's a bit more work to get subpixel accuracy, but that function can get you most of the way there:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
Image(filename='images/particle_detection.png')
```

+++ {"slideshow": {"slide_type": "slide"}}

### Circle detection

+++ {"slideshow": {"slide_type": "notes"}}

If we look at the previous video over a longer period of time (time-lapsed), then we can see droplets accumulating on the surface:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
YouTubeVideo('_qeOggvx5Rc')
```

+++ {"slideshow": {"slide_type": "notes"}}

To allow easier measurement, the microscope objective was moved to a top-down view on the substrate in the video below:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
YouTubeVideo('8utP9Ju_rdc')
```

+++ {"slideshow": {"slide_type": "notes"}}

At the time, I was interested in figuring out how droplet sizes evolved. To accomplish that, we could open up each frame in some image-analysis software and manually measure each drop size. That becomes pretty tediuous, pretty quickly. Using feature-detection techniques, we can get a good result with significantly less effort.

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
Image(filename='images/circle_detection.png')
```

+++ {"slideshow": {"slide_type": "notes"}}

In case you're wondering about the differences between the flat (lower) and textured (upper) surfaces pictured above, the circle measurements were used to describe the difference in growth, which is summarized below:

```{code-cell} python
---
slideshow:
  slide_type: notes
---
Image(filename='images/power_law_growth_regimes.png')
```

+++ {"slideshow": {"slide_type": "slide"}}

## Hough transforms

+++ {"slideshow": {"slide_type": "notes"}}

Hough transforms are a general class of operations that make up a step in feature detection. Just like we saw with edge detection, Hough transforms produce a result that we can use to detect a feature. (The distinction between the "filter" that we used for edge detection and the "transform" that we use here is a bit arbitrary.)

+++ {"slideshow": {"slide_type": "fragment"}}

### Circle detection

+++ {"slideshow": {"slide_type": "notes"}}

To explore the Hough transform, let's take the *circular* Hough transform as our example. Let's grab an image with some circles:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
image = data.coins()[0:95, 180:370]
plt.imshow(image);
```

+++ {"slideshow": {"slide_type": "notes"}}

We can use the Canny edge filter to get a pretty good representation of these circles:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
edges = filters.canny(image, sigma=3, low_threshold=10, high_threshold=60)
plt.imshow(edges);
```

+++ {"slideshow": {"slide_type": "notes"}}

While it looks like we've extracted circles, this doesn't give us much if what we want to do is *measure* these circles. For example, what are the size and position of the circles in the above image? The edge image doesn't really tell us much about that.

We'll use the Hough transform to extract circle positions and radii:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
from skimage.transform import hough_circle
 
hough_radii = np.arange(15, 30, 2)
hough_response = hough_circle(edges, hough_radii)
```

+++ {"slideshow": {"slide_type": "notes"}}

Here, the circular Hough transform actually uses the edge image from before. We also have to define the radii that we want to search for in our image.

So... what's the actual result that we get back?

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
print edges.shape,
print hough_response.shape
```

+++ {"slideshow": {"slide_type": "notes"}}

We can see that the last two dimensions of the response are exactly the same as the original image, so the response is image-like. Then what does the first dimension correspond to?

As always, you can get a better feel for the result by plotting it:

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
# Use max value to intelligently rescale the data for plotting.
h_max = hough_response.max()

def hough_responses_demo(i):
    # Use `plt.title` to add a meaningful title for each index.
    plt.imshow(hough_response[i, :, :], vmax=h_max*0.5)
    plt.show()
widgets.interact(hough_responses_demo, i=(0, len(hough_response)-1));
```

+++ {"slideshow": {"slide_type": "notes"}}

Playing around with the slider should give you a pretty good feel for the result.

This Hough transform simply counts the pixels in a thin (as opposed to filled) circular mask. Since the input is an edge image, the response is strongest when the center of the circular mask lies at the center of a circle with the same radius.

+++

---

+++ {"slideshow": {"slide_type": "fragment"}}

## <span style="color:cornflowerblue">Exercise:</span>

+++ {"slideshow": {"slide_type": "fragment"}}

Use the response from the Hough transform to **detect the position and radius of each coin**.

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter

centers = []
likelihood = []
# Your code here
```

+++ {"slideshow": {"slide_type": "notes"}}

The same concept described in this section can be applied to line detection, ellipse detection, and various other features of interest.

---

+++ {"slideshow": {"slide_type": "slide"}}

## Further reading

+++ {"slideshow": {"slide_type": "fragment"}}

### Interest point detection

+++ {"slideshow": {"slide_type": "notes"}}

We've only skimmed the surface of what might be classified as "feature detection". One major area that we haven't covered is called [interest point detection](http://en.wikipedia.org/wiki/Interest_point_detection). Here, we don't even need to know what we're looking for, we just identify small patches (centered on a pixel) that are "distinct". (The definition of "distinct" varies by algorithm; e.g., the Harris corner detector defines interest points as corners.) These distinct points can then be used to, e.g., compare with distinct points in other images.

One common use of interest point detection is for image registration, in which we align (i.e. "register") images based on interest points. Here's an example of the [CENSURE feature detector from the gallery](http://scikit-image.org/docs/dev/auto_examples/plot_censure.html):

```{code-cell} python
---
slideshow:
  slide_type: fragment
---
Image(filename='images/censure_example.png')
```

+++ {"slideshow": {"slide_type": "fragment"}}

* [Probabilistic Hough transform](http://scikit-image.org/docs/dev/auto_examples/plot_line_hough_transform.html)
* [Circular and elliptical Hough transforms](http://scikit-image.org/docs/dev/auto_examples/plot_circular_elliptical_hough_transform.html)
* [Template matching](http://scikit-image.org/docs/dev/auto_examples/plot_template.html)
* [Histogram of Oriented Gradients](http://scikit-image.org/docs/dev/auto_examples/plot_hog.html)
* [BRIEF](http://scikit-image.org/docs/dev/auto_examples/plot_brief.html), [CENSURE](http://scikit-image.org/docs/dev/auto_examples/plot_censure.html), and [ORB](http://scikit-image.org/docs/dev/auto_examples/plot_orb.html) feature detectors/descriptors
* [Robust matching using RANSAC](http://scikit-image.org/docs/dev/auto_examples/plot_matching.html)

+++

---

<div style="height: 400px;"></div>

```{code-cell} python
%reload_ext load_style
%load_style ../themes/tutorial.css
```
