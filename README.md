# scikit-image tutorials

A collection of tutorials for the [scikit-image](http://skimage.org) package.

Launch the tutorial notebooks directly with MyBinder now:

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/scikit-image/skimage-tutorials/main?filepath=index.ipynb)

Or you can setup and run on your local machine:
1. [Follow the preparation instructions](preparation.md)
2. Start the notebook server *from the same directory as this README*
   with `jupyter notebook`

Refer to [the gallery](http://scikit-image.org/docs/dev/auto_examples/) as
well as [scikit-image demos](https://github.com/scikit-image/skimage-demos)
for more examples.

## Usage

These usage guidelines are based on goodwill. They are not a legal contract.

The scikit-image team requests that you follow these guidelines if you use
these materials in downstream projects.

All materials in this repository are available free of restriction
under the Creative Commons CC0 1.0 Universal Public Domain Dedication
(see LICENSE.txt).

However, we ask that you actively acknowledge and give
attribution to this repo and to the authors if you reproduce them or create any
derivative works.  Specifically:

 * Keep the CC0 Public Domain Dedication intact when reusing large
   portions of the material (such as an entire lecture), so that
   others may benefit from the same license you did.

 * Do not represent yourself as the author of re-used material.

For more information on these guidelines, which are sometimes known as
CC0 (+BY), see [this blog post](http://www.dancohen.org/2013/11/26/cc0-by/) by
Dan Cohen.

## Contributing

If you make any modifications to these tutorials that you think would benefit
the community at large, please
[create a pull request](http://scikit-image.org/docs/dev/contribute.html)!

The tutorials live at
https://github.com/scikit-image/skimage-tutorials


## Contributor notes

- Notebooks are stored in `modules`; see [modules/00_images_are_arrays.md](modules/00_images_are_arrays.md)
  for an example.
- They use the [myst](https://myst-nb.readthedocs.io/en/latest/)
  notebook format
- Cells can be tagged with:
  `remove-input` : Get rid of the input but display the output
  `remove-output` : Show the input but not the output
  `raises-exception` : This cell is expected to fail execution, so
  don't halt the book build because of it.

To build the book, run `make`. Results appear in `book/_build`.
Notebooks can be edited in your favorite text editor or in Jupyter (as
long as Jupytext is installed).
