import os
import io
from contextlib import contextmanager

import fabric.api as fabric
from IPython.nbformat import current


NBCONVERT = 'ipython nbconvert {}'

BUILD_SLIDES =  NBCONVERT.format('--to slides {}')

NOTEBOOKS = ['0_color_and_exposure.ipynb',
             '1_image_filters.ipynb',
             '2_feature_detection.ipynb']

BACKUP_SUFFIX = '.bak'


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


@fabric.task
def clear_notebooks():
    """ Clear output and prompt numbers from default notebooks. """
    for nb in NOTEBOOKS:
        clear_notebook(nb)


def clear_notebook(filename):
    # Adapted from http://stackoverflow.com/a/19761645/260303
    notebook = load_notebook(filename)
    clear_all_nb_cells(notebook)
    save_notebook(notebook, filename)


def load_notebook(filename):
    with io.open(filename, 'r', encoding='utf-8') as f:
        notebook = current.reads(f.read(), format='ipynb')
    return notebook


def clear_all_nb_cells(notebook):
    """ Strip out all of the output and prompt_number sections. """
    for worksheet in notebook['worksheets']:
        for cell in worksheet['cells']:
           cell.outputs = []
           if 'prompt_number' in cell:
                del cell['prompt_number']


def save_notebook(notebook, filename):
    with safe_open(filename, 'w', encoding='utf-8') as f:
        current.write(notebook, f, format='ipynb')


@contextmanager
def safe_open(filename, *args, **kwargs):
    """ Open file context and save backup.

    If an error is raised in the context, replace output with backup.
    Otherwise, the backup is removed.
    """
    backup = filename + BACKUP_SUFFIX
    os.rename(filename, backup)
    try:
        with io.open(filename, *args, **kwargs) as f:
            yield f
    except:
        os.rename(backup, filename)
    else:
        os.remove(backup)
