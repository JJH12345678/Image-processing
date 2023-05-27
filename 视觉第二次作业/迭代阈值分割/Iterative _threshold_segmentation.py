import cv2
import numpy as np
from matplotlib import pyplot as plt
# 基于直方图的迭代阈值算法
T=127#初始阈值T
originimg=cv2.imread("origin.jpg")

img=cv2.cvtColor(originimg,cv2.COLOR_BGR2GRAY)
count=0
while True:
    img1=img>T#高于阈值的部分
    img2=img<=T#低于阈值的部分
    mean1=np.mean(img[img1])
    mean2=np.mean(img[img2])
    T_new=(mean1+mean2)/2
    if abs(T_new-T)<=0.6:
        print("迭代结束")
        break
    T=T_new
    count+=1
    print(f"迭代次数:{count},阈值:{T}")
img[img1] = 255
img[img2] = 0
cv2.imwrite("output.jpg",img)
plt.figure(figsize=(7,7))
plt.subplot(121),plt.title("image")
plt.imshow(originimg)
plt.subplot(122),plt.title("processed image")
plt.imshow(img,cmap='gray')
plt.show()
