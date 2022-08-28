# Preparation

## Format

The tutorial consists of lecture segments, followed by hands-on
exercises.  We strongly encourage you to bring a laptop with all the
required packages installed in order to participate fully.

## Software required

- Python

  If you are new to Python, please install the
  [Anaconda distribution](https://www.anaconda.com/distribution/) for
  **Python version 3** (available on OSX, Linux, and Windows).
  Everyone else, feel free to use your favorite distribution, but
  please ensure the requirements below are met:

  - `numpy` >= 1.21
  - `scipy` >= 1.7
  - `matplotlib` >= 3.5
  - `scikit-image` >= 0.19
  - `scikit-learn` >= 1.0
  - `notebook` >= 6.4 (or `jupyterlab` >= 3.3) 

  Please see "Test your setup" below.

- Jupyter

  The lecture material includes Jupyter notebooks.  Please follow the
  [Jupyter installation instructions](http://jupyter.readthedocs.io/en/latest/install.html),
  and ensure you have version 4 or later:

  ```bash
  $ jupyter --version
  4.4.0
  ```

  Also activate Jupyter Widgets:

  ```
  pip install -q ipywidgets
  jupyter nbextension enable --py --sys-prefix widgetsnbextension
  ```

## Download lecture material

1. [Install Git](https://git-scm.com/downloads)
2. Clone the repository at
   [https://github.com/scikit-image/skimage-tutorials](https://github.com/scikit-image/skimage-tutorials)

We may make editorial corrections to the material until the day before
the workshop, so please execute `git pull` to update.

## Test your setup

Please switch into the repository you downloaded in the previous step,
and run `check_setup.py` to validate your installation.

On my computer, I see (but your version numbers may differ):

```
[✓] scikit-image  0.15.0
[✓] numpy         1.14.5
[✓] scipy         1.1.0
[✓] matplotlib    2.2.2
[✓] notebook      5.4.0
[✓] scikit-learn  0.19.1
```

**If you do not have a working setup, please contact the instructors.**
