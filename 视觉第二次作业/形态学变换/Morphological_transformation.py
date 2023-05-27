import cv2
import numpy as np
import matplotlib.pyplot as plt
# 读取图像并转为灰度图像
originimg=cv2.imread("origin.jpg")
img=cv2.cvtColor(originimg,cv2.COLOR_BGR2GRAY)

# 二值化处理
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 定义结构元素,使用矩形结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 腐蚀操作
erosion = cv2.erode(thresh, kernel, iterations=1)

# 膨胀操作
dilation = cv2.dilate(thresh, kernel, iterations=1)

# 开运算操作
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# 闭运算操作
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 显示结果
cv2.imwrite('Binary Image.jpg', thresh)
cv2.imwrite('Erosion.jpg', erosion)
cv2.imwrite('Dilation.jpg', dilation)
cv2.imwrite('Opening.jpg', opening)
cv2.imwrite('Closing.jpg', closing)
cv2.waitKey(0)
plt.subplot(3, 3, 1), plt.imshow(originimg, cmap='gray')
plt.title('Original Image'), plt.axis('off')
plt.subplot(3, 3, 2), plt.imshow(thresh, cmap='gray')
plt.title('Binary Image'), plt.axis('off')
plt.subplot(3, 3, 3), plt.imshow(erosion, cmap='gray')
plt.title('Erosion'), plt.axis('off')
plt.subplot(3, 3, 4), plt.imshow(dilation, cmap='gray')
plt.title('Dilation'), plt.axis('off')
plt.subplot(3, 3, 5), plt.imshow(opening, cmap='gray')
plt.title('Opening'), plt.axis('off')
plt.subplot(3, 3, 6), plt.imshow(closing, cmap='gray')
plt.title('Closing'), plt.axis('off')
plt.show()
