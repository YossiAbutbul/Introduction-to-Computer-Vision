import numpy as np

point = np.array([8, 12])            # normal point
point_h = np.append(point, 1)       # homogeneous: [3 5 1]
print(point_h)

back = point_h[:2] / point_h[2]     # divide by last element
print(back)                         # [3. 5.]