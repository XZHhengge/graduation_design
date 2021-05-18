# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2021/5/17
import cv2
import numpy as np
from pc.camera_dir import camera_configs2 as camera_configs
from config import PIX_TIME_OF_REAl

def main():
    cap1 = cv2.VideoCapture(1)
    cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    while True:
        # 获取每一帧
        ret1, frame1 = cap1.read()
        img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
        hsv1 = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2HSV)
        rows, cols, channels = hsv1.shape
        p1 = np.float32([[120, 25], [2, 402], [545, 26], [635, 417]])
        p2 = np.float32([[0, 0], [0, 1.4 * PIX_TIME_OF_REAl], [1.5 * PIX_TIME_OF_REAl, 0], [1.5 * PIX_TIME_OF_REAl, 1.4 * PIX_TIME_OF_REAl]])
        # 获取透视变换矩阵
        M = cv2.getPerspectiveTransform(p1, p2)
        # 执行透视变换
        dst = cv2.warpPerspective(hsv1, M, (cols, rows))
        cv2.imshow('toushizhuanahun', hsv1)
        cv2.imshow('toushizhuanahun2', dst)

        k = cv2.waitKey(1)  # & 0xFF
        if k == ord('q'):
            break

if __name__ == '__main__':
    main()