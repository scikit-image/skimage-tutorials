from skimage import data
from skimage.feature import canny
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.overlayplugin import OverlayPlugin
from skimage.viewer.widgets import Slider


class CannyPlugin(OverlayPlugin):

    def __init__(self, *args, **kwargs):
        super(CannyPlugin, self).__init__(image_filter=canny, **kwargs)

    def attach(self, image_viewer):
        # add widgets
        self.add_widget(Slider('sigma', 0, 5))
        self.add_widget(Slider('low threshold', 0, 255, value_type='int'))
        self.add_widget(Slider('high threshold', 0, 255, value_type='int'))

        super(CannyPlugin, self).attach(image_viewer)


image = data.camera()
viewer = ImageViewer(image)
viewer += CannyPlugin()
viewer.show()
