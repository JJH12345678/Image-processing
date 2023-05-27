import cv2
import numpy as np
import matplotlib.pyplot as plt
originimg=cv2.imread("origin.jpg")
img=cv2.cvtColor(originimg,cv2.COLOR_BGR2GRAY)
alpha=0.04
ksize=3
threshold=0.01
Ix=cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize)
Iy=cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize)
height, width = img.shape[:2]
# 对M矩阵进行高斯滤波
IxIx = cv2.GaussianBlur(Ix*Ix, (ksize, ksize), 0)
IyIy = cv2.GaussianBlur(Iy*Iy, (ksize, ksize), 0)
IxIy = cv2.GaussianBlur(Ix*Iy, (ksize, ksize), 0)

# 计算角点响应函数R
det_M = IxIx * IyIy - IxIy * IxIy
trace_M = IxIx + IyIy
R = det_M - alpha * trace_M * trace_M


# 阈值处理，得到角点
corners = np.zeros_like(R, dtype=np.uint8)
corners[R > threshold * R.max()] = 255
plt.subplot(1, 2, 1), plt.imshow(originimg, cmap='gray')
plt.title('Original Image'), plt.axis('off')
plt.subplot(1, 2, 2), plt.imshow(corners, cmap='gray')
plt.title('corners'), plt.axis('off')
plt.show()
cv2.imwrite("output.jpg",corners)
