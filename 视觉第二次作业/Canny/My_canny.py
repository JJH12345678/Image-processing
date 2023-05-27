import cv2
import numpy as np
import matplotlib.pyplot as plt
originimg=cv2.imread("origin.jpg")
img=cv2.cvtColor(originimg,cv2.COLOR_BGR2GRAY)
#高斯模糊
blurred=cv2.GaussianBlur(img,(3,3),0)
#计算梯度
dx=cv2.Sobel(blurred,cv2.CV_64F,1,0)
dy=cv2.Sobel(blurred,cv2.CV_64F,0,1)
mag=np.sqrt(dx**2+dy**2)
#计算阈值
v=np.median(mag)#计算梯度幅值的中位数
lower=int((1.0-0.33)*v)
upper=int((1.0+0.33)*v)

edges=cv2.Canny(blurred,lower,upper)
plt.subplot(1,2,1),plt.imshow(originimg,cmap='gray')
plt.title('Original Image'),plt.axis('off')
plt.subplot(1,2,2),plt.imshow(edges,cmap='gray')
plt.title('Edge Image'),plt.axis('off')
plt.show()
cv2.imwrite("output.jpg",edges)
