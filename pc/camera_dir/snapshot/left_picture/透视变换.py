# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2021/5/17
import numpy as np
import cv2 as cv

img = cv.imread('left_0.PNG', 1)
rows, cols, channels = img.shape
print(rows, cols)
p1 = np.float32([[110, 8], [12, 462], [588, 3], [634, 460]])
p2 = np.float32([[0, 0], [0, 1.4*340], [1.5*340, 0], [1.5*340, 1.4*340]])
# p2 = np.float32([[1.5*340, 0], [1.5*340, 1.4*340], [0, 0], [0, 1.4*340]])
# 获取透视变换矩阵
M = cv.getPerspectiveTransform(p1, p2)
# 执行透视变换
dst = cv.warpPerspective(img, M, (cols, rows))
cv.imshow('original', img)
cv.imshow('result', dst)
cv.waitKey(0)
cv.destroyAllWindows()
