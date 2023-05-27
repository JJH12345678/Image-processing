import cv2
import numpy as np

# 读取原始图片
img = cv2.imread('origin.jpg')

# 获取原始图片的宽度和高度
height, width = img.shape[:2]

# 取较小的值作为新的边长
new_size = min(width, height)

# 创建一个新的空白图像，使用白色填充
square_img = np.zeros((new_size, new_size, 3), np.uint8)
square_img.fill(255)
resized_img = cv2.resize(img, (new_size, new_size))
# square_img[y_offset:y_offset+new_size, x_offset:x_offset+new_size] = resized_img

# 显示结果图像
cv2.imwrite('square_origin.jpg', resized_img)