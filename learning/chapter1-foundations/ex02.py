import numpy as np

# Matrix M: 2 rows, 3 columns
M = np.array([[5, 7, 2],
              [9, 1, 4]])

# Vector with 3 elements
v = np.array([1, 0, 0])

# Matrix-vector multiplication
result = M @ v   # the @ operator performs matrix multiplication

print(result)    # expected: [5 9]