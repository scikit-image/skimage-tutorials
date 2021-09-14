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

# Interaction with other libraries

+++

## Keras

- It's a very romantic notion to think that we can come up with the best features
  to model our world.  That notion has now been dispelled.
- Most *object detection/labeling/segmentation/classification* tasks now have
  neural network equivalent algorithms that perform on-par with or better than
  hand-crafted methods.
- One library that gives Python users particularly easy access to deep learning is Keras: https://github.com/fchollet/keras/tree/master/examples (it works with both Theano and TensorFlow).
- **At SciPy2017:** "Fully Convolutional Networks for Image Segmentation", Daniil Pakhomov, SciPy2017 (Friday 2:30pm)
  - Particularly interesting, because such networks can be applied to images of any size
  - ... and because Daniil is a scikit-image contributor ;)

+++

### Configurations

From http://www.asimovinstitute.org/neural-network-zoo/:

<img src="neuralnetworks.png" style="width: 80%"/>

E.g., see how to fine tune a model on top of InceptionV3:

<img src="inception_v3_architecture.png"/>

- https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes


- https://github.com/fchollet/keras/tree/master/examples
- https://keras.io/scikit-learn-api/


- In the Keras docs, you may read about `image_data_format`.  By default, this is `channels-last`, which is
compatible with scikit-image's storage of `(row, cols, ch)`.

```{code-cell} python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

import matplotlib.pyplot as plt
%matplotlib inline

## Generate dummy data
#X_train = np.random.random((1000, 2))
#y_train = np.random.randint(2, size=(1000, 1))
#X_test = np.random.random((100, 2))
#y_test = np.random.randint(2, size=(100, 1))

## Generate dummy data with some structure

from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_classification(n_features=2, n_samples=2000, n_redundant=0, n_informative=1,
                                    n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(X_test, y_test, batch_size=128)

print('\n\nAccuracy:', score[1]);
```

```{code-cell} python
from sklearn.ensemble import RandomForestClassifier
```

```{code-cell} python
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
```

```{code-cell} python
f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))

mask = (y_train == 0)
ax0.plot(X_train[mask, 0], X_train[mask, 1], 'b.')
ax0.plot(X_train[~mask, 0], X_train[~mask, 1], 'r.')
ax0.set_title('True Labels')

y_nn = model.predict_classes(X_test).flatten()
mask = (y_nn == 0)
ax1.plot(X_test[mask, 0], X_test[mask, 1], 'b.')
ax1.plot(X_test[~mask, 0], X_test[~mask, 1], 'r.')
ax1.set_title('Labels by neural net')

y_rf = rf.predict(X_test)
mask = (y_rf == 0)
ax2.plot(X_test[mask, 0], X_test[mask, 1], 'b.')
ax2.plot(X_test[~mask, 0], X_test[~mask, 1], 'r.');
ax2.set_title('Labels by random forest')
```

```{code-cell} python
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
net = InceptionV3()
```

```{code-cell} python
from skimage import transform

def inception_predict(image):
    # Rescale image to 299x299, as required by InceptionV3
    image_prep = transform.resize(image, (299, 299, 3), mode='reflect')
    
    # Scale image values to [-1, 1], as required by InceptionV3
    image_prep = (img_as_float(image_prep) - 0.5) * 2
    
    predictions = decode_predictions(
        net.predict(image_prep[None, ...])
    )
    
    plt.imshow(image, cmap='gray')
    
    for pred in predictions[0]:
        (n, klass, prob) = pred
        print(f'{klass:>15} ({prob:.3f})')
```

```{code-cell} python
from skimage import data, img_as_float
inception_predict(data.chelsea())
```

```{code-cell} python
inception_predict(data.camera())
```

```{code-cell} python
inception_predict(data.coffee())
```

You can fine-tune Inception to classify your own classes, as described at

https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes

+++

## SciPy: LowLevelCallable

https://ilovesymposia.com/2017/03/12/scipys-new-lowlevelcallable-is-a-game-changer/

```{code-cell} python
import numpy as np
image = np.random.random((512, 512))

footprint = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=bool)
```

```{code-cell} python
from scipy import ndimage as ndi
%timeit ndi.grey_erosion(image, footprint=footprint)
```

```{code-cell} python
%timeit ndi.generic_filter(image, np.min, footprint=footprint)
```

```{code-cell} python
f'Slowdown is {825 / 2.85} times'
```

```{code-cell} python
%load_ext Cython
```

```{code-cell} python
%%cython --name=test9

from libc.stdint cimport intptr_t
from numpy.math cimport INFINITY

cdef api int erosion_kernel(double* input_arr_1d, intptr_t filter_size,
                            double* return_value, void* user_data):
    
    cdef:
        double[:] input_arr
        ssize_t i
        
    return_value[0] = INFINITY
    
    for i in range(filter_size):
        if input_arr_1d[i] < return_value[0]:
            return_value[0] = input_arr_1d[i]
    
    return 1
```

```{code-cell} python
from scipy import LowLevelCallable, ndimage
import sys

def erosion_fast(image, footprint):
    out = ndimage.generic_filter(
            image,
            LowLevelCallable.from_cython(sys.modules['test9'], name='erosion_kernel'),
            footprint=footprint
    )
    return out
```

```{code-cell} python
np.sum(
    np.abs(
        erosion_fast(image, footprint=footprint)
        - ndi.generic_filter(image, np.min, footprint=footprint)
    )
)
```

```{code-cell} python
%timeit erosion_fast(image, footprint=footprint)
```

```{code-cell} python
!pip install numba
```

```{code-cell} python
# Taken from Juan Nunez-Iglesias's blog post:
# https://ilovesymposia.com/2017/03/12/scipys-new-lowlevelcallable-is-a-game-changer/

import numba
from numba import cfunc, carray
from numba.types import intc, CPointer, float64, intp, voidptr
from scipy import LowLevelCallable

def jit_filter_function(filter_function):
    jitted_function = numba.jit(filter_function, nopython=True)
    
    @cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
    def wrapped(values_ptr, len_values, result, data):
        values = carray(values_ptr, (len_values,), dtype=float64)
        result[0] = jitted_function(values)
        return 1
    
    return LowLevelCallable(wrapped.ctypes)
```

```{code-cell} python
@jit_filter_function
def fmin(values):
    result = np.inf
    for v in values:
        if v < result:
            result = v
    return result
```

```{code-cell} python
%timeit ndi.generic_filter(image, fmin, footprint=footprint)
```

## Parallel and batch processing

+++

[Joblib](https://pythonhosted.org/joblib/) (developed by scikit-learn) is used for:


1. transparent disk-caching of the output values and lazy re-evaluation (memoize pattern)
2. easy simple parallel computing
3. logging and tracing of the execution

```{code-cell} python
from sklearn.externals import joblib

from joblib import Memory
mem = Memory(cachedir='/tmp/joblib')
```

```{code-cell} python
from skimage import segmentation

@mem.cache
def cached_slic(image):
    return segmentation.slic(image)
```

```{code-cell} python
from skimage import io
large_image = io.imread('../images/Bells-Beach.jpg')
```

```{code-cell} python
%time segmentation.slic(large_image)
```

```{code-cell} python
%time cached_slic(large_image)
```

```{code-cell} python
%time cached_slic(large_image)
```

[Dask](https://dask.pydata.org) is a parallel computing library.  It has two components:

- Dynamic task scheduling optimized for computation. This is similar to Airflow, Luigi, Celery, or Make, but optimized for interactive computational workloads.
- “Big Data” collections like parallel arrays, dataframes, and lists that extend common interfaces like NumPy, Pandas, or Python iterators to larger-than-memory or distributed environments. These parallel collections run on top of the dynamic task schedulers.
- See Matt Rocklin's [blogpost](http://matthewrocklin.com/blog/work/2017/01/17/dask-images) for a more detailed example
