import numpy as np

def autocorrelation(image):
    # 计算灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算平均像素值
    mean = np.mean(gray)

    # 计算自相关函数
    acf = np.zeros(gray.shape)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            s = 0
            for k in range(i, gray.shape[0]):
                for l in range(j, gray.shape[1]):
                    s += (gray[k, l] - mean) * (gray[k - i, l - j] - mean)
            acf[i, j] = s / ((gray.shape[0] - i) * (gray.shape[1] - j))

    return acf