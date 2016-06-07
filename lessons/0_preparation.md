# Preparation

## Format

The tutorial consists of lecture segments, followed by hands-on
exercises.  We strongly encourage you to bring a laptop with all the
required packages installed in order to participate fully.

## Software required

- Python

  If you are new to Python, please install the
  [Anaconda distribution](https://www.continuum.io/downloads) for
  **Python version 3** (available on OSX, Linux and Windows).
  Everyone else, feel free to use your favorite distribution, but
  please ensure the requirements below are met:

  - `numpy` >= 1.10
  - `scipy` >= 0.15
  - `matplotlib` >= 1.5
  - `skimage` >= 0.12
  - `sklearn` >= 0.17

  In the next section below, we provide a test script to confirm the
  version numbers on your system.

- Jupyter

  The lecture material includes Jupyter notebooks.  Please follow the
  [Jupyter installation instructions](http://jupyter.readthedocs.io/en/latest/install.html),
  and ensure you have version 4 or later:

  ```bash
  $ jupyter --version
  4.1.0
  ```

## Test your setup

Please run the following commands inside of Python:

```
import numpy as np
import scipy as sp
import matplotlib as mpl
import skimage
import sklearn

for module in (np, sp, mpl, skimage, sklearn):
    print(module.__name__, module.__version__)
```

On my computer, I see (but your version numbers may differ):

```
numpy 1.11.0
scipy 0.17.0
matplotlib 1.5.1
skimage 0.12.3
sklearn 0.17.1
```

**If you do not have a working setup, please contact the instructors.
We have a limited number of hosted online accounts available for
attendees.**

## Download lecture material

There are two ways of downloading the lecture materials:

1. Get the [ZIP file from GitHub](https://github.com/scikit-image/skimage-tutorials/archive/master.zip)
2. Clone the repository at
    [https://github.com/scikit-image/skimage-tutorials](https://github.com/scikit-image/skimage-tutorials)
