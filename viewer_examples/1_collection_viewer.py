from skimage import data
from skimage.viewer import CollectionViewer
from skimage.transform import pyramid_gaussian


img = data.lena()
img_collection = tuple(pyramid_gaussian(img))

view = CollectionViewer(img_collection)
view.show()
