# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2020/10/10

'''配置文件'''
import numpy as np
import os

VIR_SIZE_TIMES_OF_REALITY_SIZE = 5  # 虚拟的地图是真实地图 比例
REALITY_SIZE_OF_MAP = (150, 140)  # 宽 × 高
REALITY_SIZE_OF_CAR = (27, 15)  # 宽 × 高

PIX_TIME_OF_REAl = 340  # 像素中大小是真实的多少倍，比如真实地图1.5*1.4,像素为640*480,1.4*340-=476，最大能显示倍数为340，这值个用于在像素中计算真实距离

MAP_WIDTH = REALITY_SIZE_OF_MAP[0] * VIR_SIZE_TIMES_OF_REALITY_SIZE  # 根据真实地图 5:1，真实地图单位cm
MAP_HEIGHT = REALITY_SIZE_OF_MAP[1] * VIR_SIZE_TIMES_OF_REALITY_SIZE  # 相机在这一边

CAR_SIZE = (REALITY_SIZE_OF_CAR[0] * VIR_SIZE_TIMES_OF_REALITY_SIZE,
            REALITY_SIZE_OF_CAR[1] * VIR_SIZE_TIMES_OF_REALITY_SIZE)


# 俯拍摄像头的高度/mm
CAMERA_HIGH = 1460
# 俯拍摄像头与地图的距离
CAMERA_LENGTH = 600


# 收集deep信息时间与次数
COLLECT_TIME, COLLECT_TIMES = 20, 40

# 在线画画 http://www.pixvi.net/piline/drawer.php
'''配置pygame背景图'''
PYGAME_BACKGROUND_FILE_PATH = './new_map_2.png'
if not os.path.exists(PYGAME_BACKGROUND_FILE_PATH[0:-4] + '_resize' + '.jpg') and os.path.exists(
        PYGAME_BACKGROUND_FILE_PATH):
    import cv2

    im1 = cv2.imread(PYGAME_BACKGROUND_FILE_PATH)
    im2 = cv2.resize(im1, (MAP_WIDTH, MAP_HEIGHT), )  # 横，竖
    left = PYGAME_BACKGROUND_FILE_PATH.rfind('/') + 1
    right = PYGAME_BACKGROUND_FILE_PATH.rfind('.')
    cv2.imwrite(PYGAME_BACKGROUND_FILE_PATH[left:right] + '_resize.jpg', im2)

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
POSITION_C = (MAP_WIDTH, MAP_HEIGHT * 3 / VIR_SIZE_TIMES_OF_REALITY_SIZE)
POSITION_D = (0, MAP_HEIGHT * 3 / VIR_SIZE_TIMES_OF_REALITY_SIZE)
POSITION_E = (0, 0)
POSITION_F = (MAP_WIDTH, 0)
#  初始化地图
NODE_DICT = {
    'A': POSITION_A, 'B': POSITION_B, 'C': POSITION_C, 'D': POSITION_D, 'E': POSITION_E, 'F': POSITION_F,
}
# 有边的节点，有向图 [(POSITION_A, POSITION_B)] 代表 A -> B
EDGE_LIST = [
    (POSITION_A, POSITION_B), (POSITION_B, POSITION_C), (POSITION_C, POSITION_D), (POSITION_D, POSITION_E),
    (POSITION_E, POSITION_F)
]

'''摄像头配置'''
# 红色标记和相机在真实地图上的坐标
# CAMERA_POS_OF_MAP = (75, 0)
CAMERA_POS_OF_MAP = (75, -100)
MARK_POS_OF_MAP = (150, 140)
# 定蓝色的HSV阈值
blue_lower = np.array([80, 175, 100])
blue_upper = np.array([130, 255, 255])

# 黄色
yellow_lower = np.array([20, 132, 86])
yellow_upper = np.array([31, 255, 255])

# 绿色
green_lower = np.array([44, 46, 28])
green_upper = np.array([79, 255, 255])
# 黑色
# lower_black = np.array([0, 0, 0])
# upper_black = np.array([180, 255, 30])

# 地图颜色 LH, LS, LV
#          UH, US, UV
map_lower = np.array([0, 0, 90])
map_upper = np.array([255, 255, 255])

# 红色
red_lower = np.array([156, 180, 0])
red_upper = np.array([255, 255, 255])


# 地图最好阈值分割
# cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
# cv2.threshold(img, 90, 255, cv2.THRESH_BINARY_INV)

# 小车坐标

class CarVar:
    CAR_X, CAR_Y = 0, 0
    HIGH = 0
    # 修正值
    CHANGE_VALUE = ''

class Map:
    SOURCE, TARGET = 0, 0
    Pic = None
    Flag = False


# tcp连接
class Tcp:
    CONN = None


