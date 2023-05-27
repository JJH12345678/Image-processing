import numpy as np
from matplotlib import pyplot as plt
import cv2
    # 8.23：频率域高通滤波器
def gaussHighPassFilter(shape, radius=32):  # 高斯高通滤波器
    # 高斯滤波器：# Gauss = 1/(2*pi*s2) * exp(-(x**2+y**2)/(2*s2))
    u, v = np.mgrid[-1:1:2.0/shape[0], -1:1:2.0/shape[1]]
    D = np.sqrt(u**2 + v**2)
    D0 = radius / shape[0]
    kernel = 1 - np.exp(- (D ** 2) / (2 *D0**2))
    return kernel

def gaussLowPassFilter(shape, radius=32):  # 高斯高通滤波器
    # 高斯滤波器：# Gauss = 1/(2*pi*s2) * exp(-(x**2+y**2)/(2*s2))
    u, v = np.mgrid[-1:1:2.0/shape[0], -1:1:2.0/shape[1]]
    D = np.sqrt(u**2 + v**2)
    D0 = radius / shape[0]
    kernel = np.exp(- (D ** 2) / (2 *D0**2))
    return kernel

img = cv2.imread("origin.jpg")  # 读取为灰度图像
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
GHPF = gaussHighPassFilter(img.shape, 32) #计算高斯高通滤波器
hpFilter=np.fft.ifftshift(GHPF)
fimg=np.fft.fft2(img)
fimg=np.fft.fftshift(fimg)
fimg=fimg*hpFilter
fimg = np.fft.ifftshift(fimg)
img_filtered1 = np.fft.ifft2(fimg).real

GLPF= gaussLowPassFilter(img.shape, ) #计算高斯低通滤波器
lpFilter=np.fft.ifftshift(GLPF)
fimg=np.fft.fft2(img)
fimg=np.fft.fftshift(fimg)
fimg=fimg*lpFilter
fimg = np.fft.ifftshift(fimg)
img_filtered2 = np.fft.ifft2(fimg).real

fig = plt.figure()
plt.subplot(221)
plt.imshow(img_filtered1, 'gray')
plt.title('gaussHighPassFilter') 
plt.subplot(222)
plt.imshow(hpFilter, 'gray')
plt.title('GHPF')
plt.subplot(223)
plt.imshow(img_filtered2, 'gray')
plt.title('gaussLowPassFilter') 
plt.subplot(224)
plt.imshow(lpFilter, 'gray')
plt.title('GLPF')
plt.show()
