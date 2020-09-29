# -*- coding: utf-8 -*-
from cv2 import cv2
import numpy as np

left_camera_matrix = np.array([[6.695034657444062e+02,0,0],
                               [-0.301765471107652,6.696120193096965e+02,0],
                               [3.222569596407729e+02,2.405742697975227e+02,1]]).T
left_distortion = np.array([0.008689952270353,0.074950145118212, 0.002124567617296,-3.399113106617445e-04, -0.190794480921519])

right_camera_matrix = np.array([[6.708696269687142e+02,0,0],
                                [-0.535421394081546,6.710711662105660e+02,0],
                                [3.160139538751914e+02,2.592722271450259e+02,1]]).T
right_distortion = np.array([1.118926374732241e-04,0.094341272683564, 0.002326806301775,5.207706578368006e-06, -0.185173732174287])

R = np.array([
    [0.999997157369108,9.932198546542005e-04,-0.002167664185938],
    [-9.824922947778085e-04,0.999987293878664,0.004944369536966],
    [0.002172547489327,-0.004942225768588,0.999985427114745]]).T

# print(R)

T = np.array([-59.916182589368400,0.083854900327739,0.053597301167885])  # 平移关系向量

size = (640, 480)  # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

