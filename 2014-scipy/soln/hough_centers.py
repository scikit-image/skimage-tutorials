from skimage.feature import peak_local_max

centers = []
likelihood = []

for h in hough_response:
    peaks = peak_local_max(h)
    centers.extend(peaks)
    likelihood.extend(h[peaks[:, 0], peaks[:, 1]])

for i in np.argsort(likelihood)[-3:]:
    row, column = centers[i]
    plt.plot(column, row, 'ro')
    plt.imshow(image)
