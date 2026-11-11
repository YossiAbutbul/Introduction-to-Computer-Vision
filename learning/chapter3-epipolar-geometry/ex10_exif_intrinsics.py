# ex10_exif_intrinsics.py
# Estimate the intrinsic matrix K from the EXIF metadata of a photo.
# Two methods:
#   1. 35mm-equivalent focal length (if present in EXIF)
#   2. Real focal length [mm] + sensor width [mm] from the camera's spec sheet

import os
import numpy as np
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))

# Image file (renamed without special characters)
IMAGE_NAME = "koscielec_view_2.jpg"
img_path = os.path.join(script_dir, IMAGE_NAME)

# Sensor width of the camera, used only by method 2.
# Sony ILCE-6000 (a6000), APS-C sensor: about 23.5 mm. Check the spec sheet.
SENSOR_WIDTH_MM = 23.5

# ---------------------------------------------------------------------------
# 1. Read EXIF
# ---------------------------------------------------------------------------
img = Image.open(img_path)
exif = img.getexif()

make = exif.get(271)            # tag 271 = Make
model = exif.get(272)           # tag 272 = Model

exif_ifd = exif.get_ifd(0x8769) # "Exif" sub-block holds the focal lengths
f_mm = exif_ifd.get(37386)      # FocalLength [mm]
f_35 = exif_ifd.get(41989)      # FocalLengthIn35mmFilm [mm]

w, h = img.size
long_side = max(w, h)

print("camera:", make, model)
print("image size (w, h):", w, h)
print("focal length [mm]:", f_mm)
print("35mm-equivalent focal length [mm]:", f_35)

# ---------------------------------------------------------------------------
# 2. Focal length in pixels
# ---------------------------------------------------------------------------
if f_35:
    # Method 1: 36 mm = long side of a full-frame sensor
    f_px = float(f_35) / 36.0 * long_side
    print("method: 35mm-equivalent focal length")
else:
    # Method 2: pixels per mm of sensor = long_side / SENSOR_WIDTH_MM
    if f_mm is None:
        raise ValueError("No focal length in EXIF")
    f_px = float(f_mm) / SENSOR_WIDTH_MM * long_side
    print("method: focal length [mm] + sensor width")

# Assumption: the file is the full sensor image, only resized (not cropped).
# If it was cropped, f_px should be computed from the full sensor width in pixels.

# ---------------------------------------------------------------------------
# 3. Build K (principal point assumed at the image center, fx = fy)
# ---------------------------------------------------------------------------
K = np.array([[f_px, 0.0,  w / 2],
              [0.0,  f_px, h / 2],
              [0.0,  0.0,  1.0]])

print("f_px:", round(f_px, 1))
print("K =\n", np.round(K, 1))