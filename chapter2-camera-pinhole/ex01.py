import numpy as np

# Image size from the file: img.shape = (height, width, channels)
height, width = 4272, 2848

# Principal point assumed at the image center
cx = width / 2
cy = height / 2

# Focal length in pixels (a guess for now, we will learn to estimate it later)
f = 1000.0

K = np.array([[f,   0.0, cx],
              [0.0, f,   cy],
              [0.0, 0.0, 1.0]])

print(K)