import numpy as np
import matplotlib.pyplot as plt
import cv2
# 读取图像
img = cv2.imread('origin.jpg')
# 转换为数组
img_array = np.array(img)
# 计算方差
variance = np.var(img_array)
print("方差为：", variance)

# 计算三阶矩
skewness = np.mean((img_array - np.mean(img_array)) ** 3) / np.std(img_array) ** 3
print("偏度为：", skewness)

# 计算四阶矩
kurtosis = np.mean((img_array - np.mean(img_array)) ** 4) / np.std(img_array) ** 4
print("峰度为：", kurtosis)