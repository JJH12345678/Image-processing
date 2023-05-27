import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('simple_origin.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 计算平均像素值
mean = np.mean(gray)

# 计算自相关函数
acf = np.zeros(gray.shape)# 创建一个与灰度图像大小相同的数组，用于存储计算得到的自相关函数
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        s = 0
        for k in range(i, gray.shape[0]):
            for l in range(j, gray.shape[1]):
                s += (gray[k, l] - mean) * (gray[k - i, l - j] - mean)#计算自相关函数的值
        acf[i, j] = s / ((gray.shape[0] - i) * (gray.shape[1] - j))#归一化
plt.subplot(1,2,1),plt.imshow(img,cmap='gray')
plt.title('Original Image'),plt.axis('off')
plt.subplot(1,2,2),plt.imshow(acf,cmap='gray')
plt.title('Autocorrelation'),plt.axis('off')
plt.show()
cv2.imwrite('output.jpg',acf)
cv2.imshow('output.jpg',acf)
cv2.waitKey(0)
cv2.destroyAllWindows()