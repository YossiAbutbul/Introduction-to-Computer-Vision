import cv2
import numpy as np

img = cv2.imread("IMG_1320.jpg")   
print(img.shape)     
print(img[0, 0])
print(img[100, 200])     
print(img[0:2, 0:3])     

cv2.imshow("תמונה", img)             
cv2.waitKey(0)                       
cv2.destroyAllWindows()