# -*- coding:utf-8 -*-
from typing import List
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
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


def process_img2(img):
    # 1. 灰度化、滤波和Canny

    # print(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.Canny(img, canny_lth, canny_hth)

    blur_gray = cv2.GaussianBlur(blur_gray, (blur_ksize, blur_ksize), 0)
    # cv2.imshow("eeaa", edges)
    # cv2.imshow("edges", edges)
    # 2. 2. 标记四个坐标点用于ROI截取
    # points = np.array([[0, 0], [0, 700], [750, 0], [750, 700]], np.int32)  # 全屏
    points = np.array([[0, 370], [0, 240], [130, 150], [470, 150], [630, 240], [630, 370]], np.int32)
    # points = np.array([[0, 500], [0, 100], [630, 500], [630, 100]], np.int32)
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
    if len(lines) != 0:
        # print(lines)
        draw_lanes2(drawing, lines, threshold=0.05)  # 画出直线检测结果
    # show_lines_data(lines)
    return drawing, lines


def cross(p1: tuple, p2: tuple, p3: tuple):
    '''
    跨立实验，若相交：如果两线段相交，则两线段必然相互跨立对方.
    若A1A2跨立B1B2，则矢量( A1 - B1 ) 和(A2-B1)位于矢量(B2-B1)的两侧，
    即(A1-B1) × (B2-B1) * (A2-B1) × (B2-B1)<0
    :param p1:  点1
    :param p2:  点2
    :param p3:  点3
    :return:
    '''
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    return x1 * y2 - x2 * y1


def count_verdict(lines, threshold, ver_threshold):
    '''
    跨立实验，求出直线相交的数量, 参考 https://www.cnblogs.com/g0feng/archive/2012/05/18/2508293.html https://blog.csdn.net/s0rose/article/details/78831570
    :param lines: [array([  0, 313, 162, 168]), array([543, 247, 627, 333]), array([184, 171, 409, 164])]
    :param threshold:
    :return:
    '''
    verdict_num = 0
    for c in combinations(lines, 2):
        # print(c[0], c[1])
        # x1, y1, x2, y2 = c[0][0][0], c[0][0][1], c[0][0][2], c[0][0][3]  # point1
        # x3, y3, x4, y4 = c[1][0][0], c[1][0][1], c[1][0][2], c[1][0][3]  # point2
        # if len(lines) >= 2:
        x1, y1, x2, y2 = c[0][0], c[0][1], c[0][2], c[0][3]  # point1
        x3, y3, x4, y4 = c[1][0], c[1][1], c[1][2], c[1][3]  # point2
        # 快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
        if (max(x1, x2) >= min(x3, x4)  # 矩形1最右端大于矩形2最左端
                and max(x3, x4) >= min(x1, x2)  # 矩形2最右端大于矩形最左端
                and max(y1, y2) >= min(y3, y4)  # 矩形1最高端大于矩形最低端
                and max(y3, y4) >= min(y1, y2)):  # 矩形2最高端大于矩形最低端
            if cross(p1=(x1, y1), p2=(x2, y2), p3=(x3, y3)) * cross(p1=(x1, y1), p2=(x2, y2),
                                                                    p3=(x4, y4)) <= 0 and cross(p1=(x3, y3),
                                                                                                p2=(x4, y4),
                                                                                                p3=(x1, y1)) * cross(
                p1=(x3, y3), p2=(x4, y4), p3=(x2, y2)) <= 0:
                # pass
                # print(x1, y1, x2, y2, x3, y3, x4, y4)
                verdict_num += 1
    print('{}条相交'.format(verdict_num))
    # else:
    # print("")
    # if abs((x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1)) < threshold:  # gongxaing
    #     if (x1 == x3) and ((y3 - y1) * (y3 - y2) <= threshold or (y4 - y1) * (y4 - y2) <= threshold):
    #         k1 = (y2 - y1) / (x2 - x1)
    #         k2 = (y3 - y4) / (x3 - x4)
    #         print(k1 * k2)
    #         if 1 + ver_threshold > abs(k1 * k2) > 1-ver_threshold:
    #             print("垂直")
    #             print(x1, y1, x2, y2, x3, y3, x4, y4)
    #         # print("相交yes")
    # else:
    #     m = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    #     n = (x2 - x1) * (y4 - y1) - (x4 - x1) * (y2 - y1)
    #     p = (x4 - x3) * (y1 - y3) - (x1 - x3) * (y4 - y3)
    #     q = (x4 - x3) * (y2 - y3) - (x2 - x3) * (y4 - y3)
    #     if m * n <= threshold and p * q <= threshold:
    #         print(m*n, threshold)
    #         k1 = (y2 - y1) / (x2 - x1)
    #         k2 = (y3 - y4) / (x3 - x4)
    #         print(k1 * k2)
    #         if 1 + ver_threshold > abs(k1 * k2) > 1-ver_threshold:
    #             print("垂直")
    #             print(x1, y1, x2, y2, x3, y3, x4, y4)
    #         # print(x1, y1, x2, y2, x3, y3, x4, y4)
    #         # print("相交yes")
    # # if abs(c[0])
    # # print(point[0], point[1], point[2], point[3])
    # # if c < threshold and c > -threshold:
    # #     print(c)


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
    left_lines, right_lines, verdict = [], [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)  # 斜率
            if k < -threshold:
                left_lines.append(line)
            elif k > +threshold:
                right_lines.append(line)
            else:
                verdict.append(line)

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return

    # draw_lines(img, left_lines)
    # draw_lines(img, right_lines)
    # print(len(verdict))

    # print(right_lines)
    # print(left_lines)
    # print("垂直个数", len(verdict))
    # draw_lines(img, right_lines)
    # b. 清理异常数据
    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)
    clean_lines(verdict, 0.1)

    # show_lines_data(right_lines, label='right_org')
    # show_lines_data(left_lines, label='left_org')
    # show_lines_data(verdict, label='verdict_org')
    draw_lines(img, right_lines, dir=1)  # 黄色
    draw_lines(img, left_lines, dir=2)  # 红色
    draw_lines(img, verdict, dir=3)  # 绿色
    # print('左边有{}'.format(len(left_lines)))
    # print('右边有{}'.format(len(right_lines)))
    # print('水平有{}'.format(len(verdict)))
    # c. 得到左右车道线点的集合，
    # print(left_lines)
    # left_array = np.array(left_lines)
    # print(left_array.mean(axis=0))
    #  得到起始点，通过计算起始点之间的距离，得到
    # left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    # # left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    # right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    verdict_points = [(x1, y1) for line in verdict for x1, y1, x2, y2 in line]
    count = count_horizontal(verdict_points, r=50)
    print(count)
    # right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    # left_results = least_squares_fit(left_points, 325, img.shape[0])
    # # show_lines_data(left_results, label="left_result")
    # right_results = least_squares_fit(right_points, 325, img.shape[0])
    # print(left_results, "lest")
    # print(right_results, "right")
    # 注意这里点的顺序
    # vtxs = np.array([[left_results[1], left_results[0], right_results[0], right_results[1]]])
    # d. 填充车道区域
    # cv2.fillPoly(img, vtxs, (0, 255, 0))

    # 或者只画车道线
    # cv2.line(img, left_results[0], left_results[1], (105, 255, 255), 5)
    # cv2.line(img, right_results[0], right_results[1], (101, 30, 255), 5)
    # left_results = average_lines(left_lines)
    # # print(left_results)
    # # cv2.line(img, (left_results[0], left_results[1]), (left_results[2], left_results[3]), (255, 255, 255), 5)
    # right_results = average_lines(right_lines)
    # verdict = average_lines(verdict)
    # # new_lines = []
    # # new_lines.append(left_results)
    # # new_lines.append(right_results)
    # # new_lines.append(verdict)
    # # print(new_lines)
    # # count_verdict(new_lines, threshold=0.1, ver_threshold=0.3)
    # # print(left_results, right_results, verdict, 'xxxxxxxxxxxxxx')
    # draw_lines(img, left_results, dir=1)  # 黄色
    # draw_lines(img, right_results, dir=2)  # 红色
    # draw_lines(img, verdict, dir=3)  # 绿色
    # show_lines_data([left_results], label='right')
    # show_lines_data([right_results], label='left')
    # show_lines_data([verdict], label='verdict')


def draw_lines(img, lines, dir: int):
    if dir == 1:
        color = [105, 255, 255]  # 黄色
    elif dir == 2:
        color = [101, 30, 255]  # 红色
    elif dir == 3:
        color = [101, 198, 39]  # 绿色
    # print(lines)
    if len(lines) > 0:
        if not isinstance(lines[0], np.int64):
            for line in lines:
                if line.all():
                    coords = line[0]
                    cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), color, 5)
        else:
            cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, 5)


def count_horizontal(data: list, r: int):
    count = 0
    while data:
        a = data[0]
        del data[0]
        for index, b in enumerate(data):
            if np.sqrt(np.sum(np.array(a) - np.array(b)) ** 2) < r:
                del data[index]
        count += 1
        if len(data) == 1: break
    return count


def average_lines(lines: list) -> list:
    if len(lines) == 1:
        return lines[0]
    else:
        num = len(lines)
        if num:
            array = np.array(lines)
            return np.mean(array, axis=0).astype(int)[0]


def clean_lines(lines, threshold):
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


if __name__ == '__main__':
    # file = '/home/perfectman/PycharmProjects/graduation_design/raspberryPi/straight.jpg'
    # file = '/home/perfectman/PycharmProjects/graduation_design/raspberryPi/crossroads.jpg'
    file = '/home/perfectman/PycharmProjects/graduation_design/raspberryPi/turn_left.jpg'
    img = cv2.imread(file)

    img = cv2.resize(img, (640, 480))
    # process_img2(img=img)
    cv2.imshow("org", img)
    cv2.imshow("process2", process_img2(img))
    # cv2.imshow("process1", process_img(img))
    cv2.waitKey(0)
