import fabric.api as fabric
import os


NBCONVERT = 'ipython nbconvert {}'

BUILD_SLIDES =  NBCONVERT.format('--to slides {}')

LECTURES = '../lectures'

NOTEBOOKS = ['color_and_exposure.ipynb',
             'image_filters.ipynb',
             'feature_detection.ipynb']

NOTEBOOKS = [os.path.join(LECTURES, nb) for nb in NOTEBOOKS]

@fabric.task
def build_slides(exclude=None):
    """ Build slides of all default notebooks. """
    exclude = exclude or []
    filtered_notebooks = (nb for nb in NOTEBOOKS if nb not in exclude)
    for nb in filtered_notebooks:
        fabric.local(BUILD_SLIDES.format(nb))


@fabric.task
def slideshow(nb=0):
    """ Build slides of all default notebooks and start slide-show. """
    try:
        nb = NOTEBOOKS[int(nb)]
    except ValueError:
        pass
    build_slides(exclude=[nb])
    serve = BUILD_SLIDES.format('--post serve {}')
    fabric.local(serve.format(nb))
