from scipy import interpolate

all_coords = np.indices((M, N)).reshape(2, -1).T
all_vals = interpolate.griddata(coords, samples, all_coords, method='cubic')

restored_image = all_vals.reshape((M, N))

plt.figure(figsize=(8, 8))
plt.imshow(restored_image, cmap='gray');

