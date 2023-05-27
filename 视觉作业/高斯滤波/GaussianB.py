import numpy as np
from matplotlib import pyplot as plt
import cv2
# 图像的低通滤波 (高斯滤波器核)
img = cv2.imread("origin.jpg", flags=1)
GaussBlur1 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) 
GaussBlur1 = GaussBlur1 / 16 # 高斯滤波器核
imgGaussBlur1  = cv2.filter2D(img, -1, GaussBlur1, borderType=cv2.BORDER_REFLECT)
plt.figure(figsize=(7, 7))
plt.subplot(121), plt.axis('off'), plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(122), plt.axis('off'), plt.title("GaussBlur")
plt.imshow(cv2.cvtColor(imgGaussBlur1, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()
