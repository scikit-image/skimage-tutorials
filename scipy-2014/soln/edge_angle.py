def angle_image(image):
    image = img_as_float(image)
    dy = convolve(image, dy_kernel)
    dx = convolve(image, dx_kernel)

    angle = np.arctan2(dy, dx)
    return np.degrees(angle)
