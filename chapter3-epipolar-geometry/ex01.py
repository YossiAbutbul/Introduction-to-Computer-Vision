import os
import cv2

# Build the image path relative to this script file, not the terminal's folder
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "IMG_1320.jpg")

# Load the image in grayscale (SIFT works on intensity values)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Stop early with a clear message if the file was not loaded
if img is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")

# Create a SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and compute a descriptor for each one
keypoints, descriptors = sift.detectAndCompute(img, None)

print(len(keypoints))       # how many keypoints were found
print(descriptors.shape)    # (number of keypoints, descriptor length)
print(keypoints[0].pt)      # (u, v) position of the first keypoint


# Draw the keypoints on the image and save it, to see where SIFT found them
img_color = cv2.imread(img_path)
vis = cv2.drawKeypoints(img_color, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(os.path.join(script_dir, "keypoints_sift.jpg"), vis)