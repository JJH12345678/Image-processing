import numpy as np
from matplotlib import pyplot as plt
import cv2
#  直方图均衡
img = cv2.imread("origin.jpg") 
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 读取并转换为灰度图像
imgMedianBlur = cv2.medianBlur(img, 3)#利用cv函数medianBlur实现中值滤波
plt.figure(figsize=(7, 7))
plt.subplot(121), plt.axis('off'), plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(122), plt.axis('off'), plt.title("cv2.medianBlur(size=3)")
plt.imshow(cv2.cvtColor(imgMedianBlur, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()
