import cv2
import numpy as np
import matplotlib.pyplot as plt
originimg=cv2.imread("origin.jpg")
img=cv2.cvtColor(originimg,cv2.COLOR_BGR2GRAY)
#计算直方图
hist, bins = np.histogram(img.ravel(), 256, [0, 256])
#计算像素总数
total_pixels = img.shape[0] * img.shape[1]#高度乘以宽度
#初始化最大方差和最佳阈值
max_variance = 0
best_threshold = 0
#遍历所有可能的阈值
for threshold in range(256):
    foreground_pixels = np.sum(hist[:threshold])
    background_pixels=total_pixels-foreground_pixels
    foreground_pixels_mean=np.sum(np.arange(threshold) * hist[:threshold]) / foreground_pixels
    background_pixels_mean=np.sum(np.arange(threshold, 256) * hist[threshold:]) / background_pixels
    variance=foreground_pixels*background_pixels*(foreground_pixels_mean-background_pixels_mean)**2
    if variance>max_variance:
        max_variance=variance
        best_threshold=threshold
print(best_threshold)
thresholded_img = (img > best_threshold).astype(np.uint8) * 255
plt.subplot(1, 2, 1), plt.imshow(originimg, cmap='gray')
plt.title('Original Image'), plt.axis('off')
plt.subplot(1, 2, 2), plt.imshow(thresholded_img, cmap='gray')
plt.title('Thresholded Image'), plt.axis('off')
plt.show()
cv2.imwrite("output.jpg",thresholded_img)
