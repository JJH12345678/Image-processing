# 图像锐化：Sobel 算子
import numpy as np
from matplotlib import pyplot as plt
import cv2
#  直方图均衡555
img = cv2.imread("origin.jpg")  # 读取图像
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 使用函数 filter2D 实现 Sobel 算子
kernSobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # SobelX kernel
kernSobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # SobelY kernel
imgSobelX = cv2.filter2D(img, -1, kernSobelX, borderType=cv2.BORDER_REFLECT)
imgSobelY = cv2.filter2D(img, -1, kernSobelY, borderType=cv2.BORDER_REFLECT)

plt.figure(figsize=(7, 7))
plt.subplot(131), plt.axis('off'), plt.title("Original")
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(132), plt.axis('off'), plt.title("SobelX")
plt.imshow(imgSobelX, cmap='gray', vmin=0, vmax=255)
plt.subplot(133), plt.axis('off'), plt.title("SobelY")
plt.imshow(imgSobelY, cmap='gray', vmin=0, vmax=255)
plt.tight_layout() # 自动调整图形的布局
plt.show()
