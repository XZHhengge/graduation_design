# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2020/10/10

'''pygame 配置'''
import numpy as np

MAP_WIDTH = 150*5  # 根据真实地图 5:1，真实地图单位cm
MAP_HEIGHT = 140*5  # 相机在这一边

CAR_SIZE = (27 * 5, 15 * 5)
# 在线画画 http://www.pixvi.net/piline/drawer.php
'''配置pygame背景图'''
PYGAME_BACKGROUND_FILE_PATH = '/home/perfectman/PycharmProjects/graduation_design/pc/pygame_dir/paint1.jpg'

# 下面记录地图上的标记坐标(基于pygame的背景图）
'''
    -------------------------------------
    - (0,0)                       (xx,0)-
    -                                   -
    -                                   -
    -                                   -
    -                                   -
    -                                   -
    - (0,xx)                            -
    -------------------------------------
'''
POSITION_A = (0, MAP_HEIGHT)
POSITION_B = (MAP_WIDTH, MAP_HEIGHT)
POSITION_C = (MAP_WIDTH, MAP_HEIGHT*3/5)
POSITION_D = (0, MAP_HEIGHT*3/5)
POSITION_E = (0, 0)
POSITION_F = (MAP_WIDTH, 0)

NODE_LIST = [
    POSITION_A, POSITION_B, POSITION_C, POSITION_D, POSITION_E, POSITION_F,
]
# 有边的节点，有向图 [(POSITION_A, POSITION_B)] 代表 A -> B
EDGE_LIST = [
    (POSITION_A, POSITION_B), (POSITION_B, POSITION_C), (POSITION_C, POSITION_D), (POSITION_D, POSITION_E),
    (POSITION_E, POSITION_F)
]

'''摄像头配置'''
# 红色标记和相机在真实地图上的坐标
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


# 红色
red_lower = np.array([156, 180, 0])
red_upper = np.array([255, 255, 255])
