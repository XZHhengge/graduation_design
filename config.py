# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2020/10/10

'''pygame 配置'''
import numpy as np
MAP_WIDTH = 750  # 根据真实地图 5:1，真实地图单位cm
MAP_HEIGHT = 700

CAR_SIZE = (27 * 5, 15 * 5)

'''摄像头配置'''
# 红色标记和相机坐标
CAMERA_POS_OF_MAP = (75, 0)
MARK_POS_OF_MAP = (150, 140)
# 定蓝色的HSV阈值
blue_lower = np.array([80, 175, 100])
blue_upper = np.array([130, 255, 255])

# 黄色
yellow_lower = np.array([20, 132, 86])
yellow_upper = np.array([31, 255, 255])

# 黄和蓝色
# lower = np.array([20, 132, 152])
# upper = np.array([33, 255, 255])
# 黑色
# lower_black = np.array([0, 0, 0])
# upper_black = np.array([180, 255, 30])

# i = 0
# 红色
red_lower = np.array([156, 180, 0])
red_upper = np.array([255, 255, 255])
