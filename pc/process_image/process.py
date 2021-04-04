# -*- coding:utf-8 -*-
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import cv2

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


def process_img(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 1)
    edges = cv2.Canny(blur_gray, canny_lth, canny_hth)


def process_img2(img):
    # 1. 灰度化、滤波和Canny

    # print(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blur_gray = cv2.Canny(img, canny_lth, canny_hth)

    blur_gray = cv2.GaussianBlur(blur_gray, (blur_ksize, blur_ksize), 0)
    # cv2.imshow("eeaa", edges)
    # cv2.imshow("edges", edges)
    # 2. 2. 标记坐标点用于ROI截取
    # points = np.array([[0, 0], [0, 700], [750, 0], [750, 700]], np.int32)  # 全屏
    points = np.array([[0, 370], [0, 150], [600, 150], [600, 370]], np.int32)
    # points = np.array([[0, 500], [0, 100], [630, 500], [630, 100]], np.int32)
    # points = np.array([[0, 120], [0, 240], [640, 240], [640, 120]], np.int32) # half
    roi_edges = roi_mask(blur_gray, [points])
    # 3. 霍夫直线提取
    drawing, lines = hough_lines(roi_edges, rho, theta, threshold, min_line_len, max_line_gap)
    # print(len(lines), '线条数量')
    # 4. 车道拟合计算
    if lines:
        draw_lanes2(drawing, lines, threshold=0.1)  # 画出直线检测结果
    else:
        print('no lines')
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
        pass
    else:
        lines = lines.tolist()
        draw_lanes2(drawing, lines, threshold=0.1)  # 画出直线检测结果
    return drawing, lines


def show_lines_data(lines: List, label: str):
    if len(lines) == 1:
        for x1, y1, x2, y2 in lines:
            plt.suptitle(label)
            plt.plot([x1, x2], [y1, y2])
    else:
        for line in lines:
            for x1, y1, x2, y2 in line:
                plt.suptitle(label)
                plt.plot([x1, x2], [y1, y2])
    plt.show()
    # fig = plt.figure(1)
    # ax  =fig.gca()


def draw_lanes2(img, lines, threshold):
    # a. 划分左右车道和水平车道,不能单独地按照平均值划分，
    left_lines, right_lines, verdict, unkown = [], [], [], []
    for line in lines:
        for x1, y1, x2, y2 in line:  # line = [ x1, y1, x2, y2 ]
            if (x2 - x1) == 0 or (y2 - y1) == 0:
                unkown.append(line)
            else:
                k = (y2 - y1) / (x2 - x1)  # 斜率
                if k < -threshold:
                    left_lines.append(line)
                elif k > +threshold:
                    right_lines.append(line)
                else:
                    verdict.append(line)

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        print('无倾斜线条')

    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)
    clean_lines(verdict, 0.5)
    # show_lines_data(right_lines, label='right_org')
    # show_lines_data(left_lines, label='left_org')
    # show_lines_data(verdict, label='verdict_org')
    # draw_lines(img, right_lines, dir=1)  # 黄色
    # draw_lines(img, left_lines, dir=2)  # 红色

    if verdict:
        draw_lines(img, verdict, dir=3)  # 绿色
        count_line_cluster(lines=verdict, r=30)
    if unkown:
        draw_lines(img, unkown, dir=4)
    left_results = average_lines(left_lines)
    right_results = average_lines(right_lines)
    if right_results:
        right_length = get_line_length(right_results)
        draw_lines(img, right_results, dir=2)  # 红色
    if left_results:
        left_length = get_line_length(left_results)
        draw_lines(img, left_results, dir=1)  # 黄色

    # draw_lines(img, verdict, dir=3)  # 绿色

    if left_results and right_results:
        print(abs(left_length - right_length), '差')
    # show_lines_data([left_results], label='right')
    # show_lines_data([right_results], label='left')
    # show_lines_data([verdict], label='verdict')


def get_line_length(line):
    return np.sqrt((line[0] - line[2]) ** 2 + (line[1] - line[3]) ** 2)


def draw_lines(img, lines, dir: int):
    if dir == 1:
        color = [105, 255, 255]  # 黄色
    elif dir == 2:
        color = [101, 30, 255]  # 红色
    elif dir == 3:
        color = [101, 198, 39]  # 绿色
    elif dir == 4:
        color = [110, 110, 255]
    # print(lines)
    if len(lines) > 0:
        if not isinstance(lines[0], int):
            for line in lines:
                # if line.all():
                coords = line[0]
                cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), color, 5)
        else:
            cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, 5)


def count_line_cluster(lines, r: int) -> int:
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
    print(len(count_list), '簇数')
    return len(count_list)


def average_lines(lines: list) -> list:
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


def clean_lines(lines, threshold):
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


def least_squares_fit(point_list, ymin, ymax):
    '''
    左右车道可以用最小二乘法拟合， 平行车道不行
    :param point_list:
    :param ymin:
    :param ymax:
    :return:
    '''
    # 最小二乘法拟合
    # print()
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    # plt.plot(y, x, 'ro')
    # plt.show()
    # polyfit第三个参数为拟合多项式的阶数，所以1代表线性
    fit = np.polyfit(x, y, 1)
    fit_fn = np.poly1d(fit)  # 获取拟合的结果

    xmin = int(fit_fn(min(x)))
    xmax = int(fit_fn(max(x)))

    return [(xmin, ymin), (xmax, ymax)]


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


if __name__ == '__main__':
    # file = '/home/perfectman/PycharmProjects/graduation_design/raspberryPi/straight.jpg'
    # file = '/home/perfectman/PycharmProjects/graduation_design/raspberryPi/crossroads.jpg'
    file = '/home/perfectman/PycharmProjects/graduation_design/raspberryPi/foo.jpg'
    img = cv2.imread(file)

    img = cv2.resize(img, (640, 480))
    # process_img2(img=img)
    cv2.imshow("org", img)
    cv2.imshow("process2", process_img2(img))
    # cv2.imshow("process1", process_img(img))
    cv2.waitKey(0)
