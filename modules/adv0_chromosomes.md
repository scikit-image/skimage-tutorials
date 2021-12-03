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

# Measuring chromatin fluorescence

Goal: we want to quantify the amount of a particular protein (red fluorescence) localized on the centromeres (green) versus the rest of the chromosome (blue).

<img src="../images/chromosomes.jpg" width="550px"/>

The main challenge here is the uneven illumination, which makes isolating the chromosomes a struggle.

```{code-cell} python
import numpy as np
from matplotlib import cm, pyplot as plt
import skdemo
plt.rcParams['image.cmap'] = 'cubehelix'
plt.rcParams['image.interpolation'] = 'none'
```

```{code-cell} python
from skimage import io
image = io.imread('../images/chromosomes.tif')
skdemo.imshow_with_histogram(image);
```

Let's separate the channels so we can work on each individually.

```{code-cell} python
protein, centromeres, chromosomes = image.transpose((2, 0, 1))
```

Getting the centromeres is easy because the signal is so clean:

```{code-cell} python
from skimage.filters import threshold_otsu
centromeres_binary = centromeres > threshold_otsu(centromeres)
skdemo.imshow_all(centromeres, centromeres_binary)
```

But getting the chromosomes is not so easy:

```{code-cell} python
chromosomes_binary = chromosomes > threshold_otsu(chromosomes)
skdemo.imshow_all(chromosomes, chromosomes_binary, cmap='gray')
```

Let's try using an adaptive threshold:

```{code-cell} python
from skimage.filters import threshold_local
chromosomes_adapt = threshold_local(chromosomes, block_size=51)
# Question: how did I choose this block size?
skdemo.imshow_all(chromosomes, chromosomes_adapt)
```

Not only is the uneven illumination a problem, but there seem to be some artifacts due to the illumination pattern!

**Exercise:** Can you think of a way to fix this?

(Hint: in addition to everything you've learned so far, check out [`skimage.morphology.remove_small_objects`](http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.remove_small_objects))

Now that we have the centromeres and the chromosomes, it's time to do the science: get the distribution of intensities in the red channel using both centromere and chromosome locations.

```python
# Replace "None" below with the right expressions!
centromere_intensities = None
chromosome_intensities = None
all_intensities = np.concatenate((centromere_intensities,
                                  chromosome_intensities))
minint = np.min(all_intensities)
maxint = np.max(all_intensities)
bins = np.linspace(minint, maxint, 100)
plt.hist(centromere_intensities, bins=bins, color='blue',
         alpha=0.5, label='centromeres')
plt.hist(chromosome_intensities, bins=bins, color='orange',
         alpha=0.5, label='chromosomes')
plt.legend(loc='upper right')
plt.show()
```
