import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.viewer import ImageViewer
from skimage.viewer.widgets import Slider
from skimage.viewer.plugins.base import Plugin


image = io.imread('../../images/clock_motion.png')
M, N = image.shape

## Should pad, but doesn't make much difference in this case
MM, NN = 2 * M + 1, 2 * N + 1

def hann(image):
    wy = np.hanning(image.shape[0])[:, None]
    wx = np.hanning(image.shape[1])


## Apply Hann window to prevent ringing
wy = np.hanning(M)[:, None]
wx = np.hanning(N)

f = np.zeros((MM, NN))
f[:M, :N] = wy * wx * image

F = np.fft.fft2(f)

v, u = np.ogrid[:MM, :NN]
v -= (MM - 1) // 2
u -= (NN - 1) // 2


def apply_inverse_filter(image, T, a, b, K=5, clip=500):
    uavb = u * a + v * b
    H = T * np.sinc(uavb) * np.exp(-1j * np.pi * uavb)
    H = np.fft.fftshift(H)

    HH = 1./H
    HH[np.abs(HH) > K] = K

    gg = np.abs(np.fft.ifft2(F * HH))
    gg = gg[:M, :N]
    gg = np.clip(gg, 0, clip)
    gg -= gg.min()
    gg /= gg.max()

    return gg

viewer = ImageViewer(image)

plugin = Plugin(image_filter=apply_inverse_filter)
plugin += Slider('T', 0, 1, value=0.5, value_type='float', update_on='release')
plugin += Slider('a', -0.1, 0.1, value=0, value_type='float', update_on='release')
plugin += Slider('b', -0.1, 0.1, value=0, value_type='float', update_on='release')
plugin += Slider('K', 0, 100, value=15, value_type='float', update_on='release')
plugin += Slider('clip', 0, 1000, value=750, value_type='float', update_on='release')
viewer += plugin
viewer.show()
