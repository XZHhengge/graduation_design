# -*- coding:utf-8 -*-
import threading
import time
from typing import List
from itertools import combinations
import RPi.GPIO as GPIO
import math
import numpy as np
from numpy import array, int32
import cv2
from config import Road, Car, DeviateThreshold
# from drive_car import ThreadMoveCar
from test import ThreadMoveCar
# 高斯滤波核大小
blur_ksize = 3
# Canny边缘检测高低阈值
canny_lth = 200
canny_hth = 300

# 霍夫变换参数
rho = 1
theta = np.pi / 180
threshold = 100
min_line_len = 40
max_line_gap = 20

FLAG = True


# def process3(img):
#     '''
#     识别矩形进行判断直线、转弯
#     :param img:
#     :return:
#     '''
#     l_h, l_s, l_v = 0, 0, 90
#     u_h, u_s, u_v = 255, 255, 255
#
#     lower_b = np.array([l_h, l_s, l_v])
#     upper_b = np.array([u_h, u_s, u_v])
#
#     # img = cv2.imread('foo.jpg')
#     # img = cv2.resize(img, (640, 480))
#
#     def roi_mask(img, corner_points):
#         #
#         mask = np.zeros_like(img)
#         cv2.fillPoly(mask, corner_points, 255)
#         masked_img = cv2.bitwise_and(img, mask)
#         return masked_img
#
#     def get_point_line_distance(point, line):
#         point_x = point[0]
#         point_y = point[1]
#         line_s_x = line[0][0]
#         line_s_y = line[0][1]
#         line_e_x = line[1][0]
#         line_e_y = line[1][1]
#         # 若直线与y轴平行，则距离为点的x坐标与直线上任意一点的x坐标差值的绝对值
#         if line_e_x - line_s_x == 0:
#             return math.fabs(point_x - line_s_x)
#         # 若直线与x轴平行，则距离为点的y坐标与直线上任意一点的y坐标差值的绝对值
#         if line_e_y - line_s_y == 0:
#             return math.fabs(point_y - line_s_y)
#         # 斜率
#         k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
#         # 截距
#         b = line_s_y - k * line_s_x
#         # 带入公式得到距离dis
#         dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
#         return dis
#     # hsv选取
#     # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     # img = cv2.cvtColor(img, cv2.COLOR_GRAY2)
#     mask = cv2.inRange(img, lower_b, upper_b)
#     # mask2 = cv2.inRange(img, lower_b, upper_b)
#     # blur_gray = cv2.Canny(mask, canny_lth, canny_hth)
#
#     # blur_gray = cv2.GaussianBlur(blur_gray, (blur_ksize, blur_ksize), 0)
#     # cv2.imshow('mask1', blur_gray)
#
#     # 开运算
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素
#     opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算
#
#     # ROI切割
#     # points = np.array([[0, 370], [0, 150], [640, 150], [640, 370]], np.int32)
#     points = np.array([[0, 370], [0, 250], [640, 250], [640, 370]], np.int32)
#     roi_edges = roi_mask(opening, [points])
#     # cv2.imshow('xx', roi_edges)
#     # 取最大轮廓
#     image, contours, hierarchy = cv2.findContours(
#         roi_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
#     cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
#     # cv2.imshow('aa', img)
#     # 获取重心
#     cnt = contours[0]
#     M = cv2.moments(cnt)
#     # 计算矩形最大面积
#     area = cv2.contourArea(cnt)
#     print('矩形面积', area)
#     # print('process_img', Road.IS_STRAIGHT)
#     # 画屏幕中线
#     rows, cols = img.shape[:2]
#     cv2.line(img, (int(cols / 2), 0), (int(cols / 2), rows), (0, 255, 0), 2)
#     if M:
#         #  获取矩形重心
#         cx = int(M['m10'] / M['m00'])
#         cy = int(M['m01'] / M['m00'])
#         # print(cx, cy)
#         global FLAG
#         if FLAG:
#             print('move_car start')
#             Road.IS_STRAIGHT = True
#             FLAG = False
#
#         if get_point_line_distance(point=[cx, cy], line=[(int(cols / 2), 0), (int(cols / 2), rows)]) > DeviateThreshold:
#             if cx > int(cols / 2):
#                 # print('左轮速度:{}, 右轮速度:{}'.format(Car.left_forward_v, Car.right_forward_v))
#                 print('偏左，左加速')
#                 if Car.left_forward_v > 30:
#                     if Car.right_forward_v > Car.default_right_forward_v:
#                         Car.right_forward_v -= 1
#                 else:
#                     Car.left_forward_v += 1
#             elif cx < int(cols / 2):
#                 # print('左轮速度:{}, 右轮速度:{}'.format(Car.left_forward_v, Car.right_forward_v))
#                 print('偏右, 右加速')
#                 if Car.right_forward_v > 30:
#                     if Car.left_forward_v > Car.default_left_forward_v:
#                         Car.left_forward_v -= 1
#                 else:
#                     Car.right_forward_v += 1
#     else:
#         print('无重心')
#     # cv2.line(img, (rows, int(cols / 2)), (0, int(cols / 2)), (255, 255, 0), 2)
#     # 画矩形
#     # x, y, w, h = cv2.boundingRect(cnt)
#     # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     #
#     # rect = cv2.minAreaRect(cnt)  # 最小外接矩形
#     # box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
#     # cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
#     # cv2.imshow('img', img)
#
#     # cv2.imshow('img2', img2)
#     # 拟合直线
#     # rows, cols = img.shape[:2]
#     #
#     # out = cv2.fitLine(contours[0], cv2.DIST_L1, 0, 0.01, 0.01)
#     # [vx, vy, x, y] = out
#     # left = int((-x * vy / vx) + y)
#     # righty = int(((cols - x) * vy / vx) + y)
#     # img = cv2.line(img, (cols - 1, righty), (0, left), (0, 255, 0), 2)
#     # cv2.imshow('xxaa', img)
#     # cv2.fit
#     # cv2.waitKey(0)


def move_car():
    global FLAG
    MV = MoveCar()
    MV.go_straight()
    FLAG = False


def process_img2(img):
    '''
    plan A :识别线条，利用分簇算法判断直线，转弯，效果不是理想
    :param img:
    :return:
    '''
    # 1. 灰度化、滤波和Canny

    # print(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.Canny(img, canny_lth, canny_hth)

    blur_gray = cv2.GaussianBlur(blur_gray, (blur_ksize, blur_ksize), 0)
    # cv2.imshow("eeaa", edges)
    # cv2.imshow("edges", edges)
    # 2. 2. 标记四个坐标点用于ROI截取
    points = np.array([[0, 370], [0, 240], [130, 150], [470, 150], [630, 240], [630, 370]], np.int32)
    # points = np.array([[0, 120], [0, 240], [640, 240], [640, 120]], np.int32) # half
    roi_edges = roi_mask(blur_gray, [points])
    # 3. 霍夫直线提取
    drawing, lines = hough_lines(roi_edges, rho, theta, threshold, min_line_len, max_line_gap)
    # print(len(lines))
    # print(lines[0])
    # 4. 车道拟合计算
    # draw_lanes2(drawing, lines)     # 画出直线检测结果
    # 5. 最终将结果合在原图上
    # result = cv2.addWeighted(img, 0.9, drawing, 0.2, 0)
    # cv2.waitKey(0)
    # lines2 = cv2.HoughLinesP(drawing, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
    # print(len(lines2), "second")
    return drawing


def roi_mask(img, corner_points):
    #
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, corner_points, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # 统计概率霍夫直线变换
    lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)

    # 新建一副空白画布
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is None:
        Road.IS_STRAIGHT = False
        print('no lines')
    else:
        lines = lines.tolist()
        draw_lanes2(drawing, lines, threshold=0.1)  # 画出直线检测结果
    return drawing, lines


def draw_lanes2(img, lines, threshold):
    # a. 划分左右车道和水平车道,不能单独地按照平均值划分，
    left_lines, right_lines, verdict = [], [], []
    for line in lines:
        for x1, y1, x2, y2 in line:  # line = [ x1, y1, x2, y2 ]
            k = (y2 - y1) / (x2 - x1)  # 斜率
            if k < -threshold:
                left_lines.append(line)
            elif k > +threshold:
                right_lines.append(line)
            else:
                verdict.append(line)

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return


    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)
    clean_lines(verdict, 0.5)
    # show_lines_data(right_lines, label='right_org')
    # show_lines_data(left_lines, label='left_org')
    # show_lines_data(verdict, label='verdict_org')
    # draw_lines(img, right_lines, dir=1)  # 黄色
    # draw_lines(img, left_lines, dir=2)  # 红色
    left_length, right_length = 0, 0
    if verdict:
        # draw_lines(img, verdict, dir=3)  # 绿色
        cluster_num = count_line_cluster(lines=verdict, r=30)
        print(cluster_num, "count_line_cluster")
        if cluster_num < 4:
            time.sleep(1)
            # 还要考虑十字路口的情况
            Road.IS_STRAIGHT = False
            Road.IS_CROSS = True
            #  查看队列中的
            print('转弯')
        elif cluster_num == 4:
            #  判断是左右转还是前进,优先前进
            print('十字路口')

    left_results = average_lines(left_lines)
    right_results = average_lines(right_lines)
    if right_results:
        right_length = get_line_length(right_results)
        # draw_lines(img, right_results, dir=2)  # 红色
    if left_results:
        left_length = get_line_length(left_results)
        # draw_lines(img, left_results, dir=1)  # 黄色

    # draw_lines(img, verdict, dir=3)  # 绿色
    # if left_results and right_results:
    #     print(abs(left_length - right_length), '差')
    global FLAG

    if left_length and right_length:  # 有线
        if FLAG:
            print('move_car start')
            t = threading.Thread(target=move_car)
            t.start()
        if right_length - left_length > 100:  # left + / right - 左大偏左
            print('偏左')
            if Car.left_forward_v > 20:
                Car.right_forward_v -= 1
            else:
                Car.left_forward_v += 1
        if left_length - right_length > 100:  # right - / left +  右大偏右
            print('偏右')
            if Car.right_forward_v > 15:
                # print()
                Car.left_forward_v -= 1
            else:
                Car.right_forward_v += 1
    else:
        print('no straight')
        # Road.IS_STRAIGHT = False


def get_line_length(line):
    '''
    线与线之间的距离
    :param line:
    :return:
    '''
    return np.sqrt((line[0] - line[2]) ** 2 + (line[1] - line[3]) ** 2)





def count_line_cluster(lines, r: int) -> int:
    '''
    分簇算法
    :param lines:
    :param r:
    :return:
    '''
    copy_line = sorted(lines, key=lambda x: x[0][0])
    count_list = [copy_line[0]]
    # copy_line
    for j in range(len(copy_line) - 1):
        for i in range(len(count_list)):
            if get_two_line_short_length(copy_line[j + 1][0], count_list[i][0]) > r:
                if len(count_list) == i + 1:
                    count_list.append(copy_line[j + 1])
                    break
            else:
                if copy_line[j + 1][0][2] > count_list[i][0][2]:
                    count_list[i] = copy_line[j + 1]
                    break
                else:
                    break
    if len(count_list) not in Road.LINE_CLUSTER:
        Road.LINE_CLUSTER[len(count_list)] = 1
    else:
        Road.LINE_CLUSTER[len(count_list)] += 1
    return len(count_list)


def clean_lines(lines, threshold):
    '''
    清理线段
    :param lines:
    :param threshold:
    :return:
    '''
    for index, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < 100:
                # print(line)
                lines.pop(index)

    # 迭代计算斜率均值，排除掉与差值差异较大的数据
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)  # 选出最大值的下标
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


def get_two_line_short_length(point1, point2):
    x1, y1, n = get_x_y(point1)
    x2, y2, m = get_x_y(point2)
    ar = np.zeros((n, m))  # 11*11矩阵
    for i in range(n):  # 欧氏距离
        ar[i, :] = np.sqrt((x2 - x1[i]) ** 2 + (y2 - y1[i]) ** 2)
    return ar.min()  # 取最小的


def get_x_y(point):
    g = point[1] - point[3]  # y1 - y2
    h = point[0] - point[2]  # x1 - x2 #  这个始终为负(point始终在左边)
    if g < 0 or h < 0:
        t = array([-i / 10 for i in range(0, 11)])  # 斜率为负数, 这里越长，精度越高
    else:
        t = array([i / 10 for i in range(0, 11)])  # 斜率不存在
    x = array(point[0]) + h * t  # 线段横坐标平均取11个点
    y = array(point[1]) + g * t  # 线段纵坐标平均取11个点
    return x, y, len(t)


def average_lines(lines: list) -> list:
    '''
    线段归一化
    :param lines:
    :return:
    '''
    if len(lines) == 1:
        if isinstance(lines[0][0], int):
            return lines[0]
        else:
            return lines[0][0]
    else:
        num = len(lines)
        if num:
            array = np.array(lines)
            return np.mean(array, axis=0).astype(int)[0].tolist()


if __name__ == '__main__':
    # file = '/home/perfectman/PycharmProjects/graduation_design/raspberryPi/straight.jpg'
    # file = '/home/perfectman/PycharmProjects/graduation_design/raspberryPi/crossroads.jpg'
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    try:
        while ret:
            ret, frame = cap.read()
            process3(frame)
    except KeyboardInterrupt:
        GPIO.cleanup()
        # print('--')
    # cv2.imshow("process1", process_img(img))
