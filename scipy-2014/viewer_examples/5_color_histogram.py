from skimage import io
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.color_histogram import ColorHistogram


image = io.imread('mms_scipy2013.png')
viewer = ImageViewer(image)
viewer += ColorHistogram(max_pct=0.99)
viewer.show()
