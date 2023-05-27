import numpy as np
from matplotlib import pyplot as plt
import cv2
# 图像的非线性灰度变换: 幂律变换 (伽马变换)

img = cv2.imread("origin.jpg")  # 读取图像
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 转换为灰度图像
gamma1=0.5  # gamma1 值
gamma2=1.5  # gamma2 值
normImg = lambda x: 255. * (x-x.min()) / (x.max()-x.min()+1e-6)  # 归一化为 [0,255]

plt.figure(figsize=(9,6))
imgGamma1 = np.power(img, gamma1)#使 用NumPy库的power()函数对原始图像进行幂律变换
imgGamma1 = np.uint8(normImg(imgGamma1))#转换为8位无符号整数类型
imgGamma2 = np.power(img, gamma2)#使用NumPy库的power()函数对原始图像进行幂律变换
imgGamma2 = np.uint8(normImg(imgGamma2))#转换为8位无符号整数类型
plt.subplot(131), plt.axis('off')
plt.imshow(img,  cmap='gray', vmin=0, vmax=255)#展示原图
plt.subplot(132), plt.axis('off')
plt.imshow(imgGamma1,  cmap='gray', vmin=0, vmax=255)#展示gamma=0.5时的图
plt.title(f"$\gamma={gamma1}$")
plt.subplot(133), plt.axis('off')
plt.imshow(imgGamma2,  cmap='gray', vmin=0, vmax=255)#展示gamma=1.5时的图
plt.title(f"$\gamma={gamma2}$")
plt.show()
