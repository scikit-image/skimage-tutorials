from skimage import data
from skimage.feature import canny
from skimage.viewer import ImageViewer
from skimage.viewer.widgets import Slider
from skimage.viewer.plugins.overlayplugin import OverlayPlugin


plugin = OverlayPlugin(image_filter=canny)
plugin += Slider('sigma', 0, 5)
plugin += Slider('low threshold', 0, 255, value_type='int')
plugin += Slider('high threshold', 0, 255, value_type='int')

viewer = ImageViewer(data.camera())
viewer += plugin
viewer.show()
