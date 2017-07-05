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

  - `numpy` >= 1.12
  - `scipy` >= 0.19
  - `matplotlib` >= 2.0
  - `skimage` >= 0.13
  - `sklearn` >= 0.18
  
  Please see "Test your setup" below.

- Jupyter

  The lecture material includes Jupyter notebooks.  Please follow the
  [Jupyter installation instructions](http://jupyter.readthedocs.io/en/latest/install.html),
  and ensure you have version 4 or later:

  ```bash
  $ jupyter --version
  4.1.0
  ```

## Download lecture material

There are two ways of downloading the lecture materials:

1. Get the [ZIP file from GitHub](https://github.com/scikit-image/skimage-tutorials/archive/master.zip)
2. Clone the repository at
    [https://github.com/scikit-image/skimage-tutorials](https://github.com/scikit-image/skimage-tutorials)

## Test your setup

Please switch into the repository you downloaded in the previous step, and run `check_setup.py`
to validate your installation.

On my computer, I see (but your version numbers may differ):

```
[✓] scikit-image  0.13.0
[✓] scipy         0.19.0
[✓] matplotlib    2.0.0
[✓] notebook      5.0.0
[✓] scikit-learn  0.18.1
```

**If you do not have a working setup, please contact the instructors.**


