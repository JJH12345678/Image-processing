import numpy as np
from matplotlib import pyplot as plt
import cv2
#  直方图均衡
img = cv2.imread("origin.jpg")  # 读取为灰度图像
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
histImg, bins = np.histogram(img.flatten(), 256)  # 计算原始图像直方图
cdf = histImg.cumsum()  # 计算累积分布函数 CDF
cdf = cdf * 255 / cdf[-1]  # 累计函数 CDF 归一化: [0,1]->[0,255]
imgEqu = np.interp(img.flatten(), bins[:256], cdf)  # 线性插值，计算新的灰度值
imgEqu = imgEqu.reshape(img.shape)  # 将压平的图像数组重新变成二维数组

fig = plt.figure(figsize=(7,7))
plt.subplot(121), plt.title("Origin image")
plt.imshow(img)  # 原始图像
plt.subplot(122),plt.title("Hist-equalized image")
plt.imshow(imgEqu)  # 转换后图像
plt.show()
