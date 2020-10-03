# -*- coding:utf-8 -*-
from pc.process_image.process1 import process_img
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
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.Canny(gray, canny_lth, canny_hth)

    blur_gray = cv2.GaussianBlur(blur_gray, (blur_ksize, blur_ksize), 0)
    # cv2.imshow("eeaa", edges)
    # cv2.imshow("edges", edges)
    # 2. 2. 标记四个坐标点用于ROI截取
    points = np.array([[0, 370], [0, 240], [130, 125], [470, 125], [640, 240], [640, 370]], np.int32)
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
    draw_lanes2(drawing, lines, threshold=0.05)  # 画出直线检测结果

    return drawing, lines


def count_verdict(lines, threshold, ver_threshold):
    '''
    两两直线之间计算斜率，求出约等于直角的个数, 参考 https://blog.csdn.net/qq_36135484/article/details/79450998
    :param lines:
    :param threshold:
    :return:
    '''
    for c in combinations(lines, 2):
        # print(c[0], c[1])
        x1, y1, x2, y2 = c[0][0][0], c[0][0][1], c[0][0][2], c[0][0][3]
        x3, y3, x4, y4 = c[1][0][0], c[1][0][1], c[1][0][2], c[1][0][3]
        if abs((x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1)) < threshold:  # gongxaing
            if (x1 == x3) and ((y3 - y1) * (y3 - y2) <= threshold or (y4 - y1) * (y4 - y2) <= threshold):
                k1 = (y2 - y1) / (x2 - x1)
                k2 = (y3 - y4) / (x3 - x4)
                print(k1 * k2)
                if 1 + ver_threshold > abs(k1 * k2) > 1-ver_threshold:
                    print("垂直")
                    print(x1, y1, x2, y2, x3, y3, x4, y4)
                # print("相交yes")
        else:
            m = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            n = (x2 - x1) * (y4 - y1) - (x4 - x1) * (y2 - y1)
            p = (x4 - x3) * (y1 - y3) - (x1 - x3) * (y4 - y3)
            q = (x4 - x3) * (y2 - y3) - (x2 - x3) * (y4 - y3)
            if m * n <= threshold and p * q <= threshold:
                print(m*n, threshold)
                k1 = (y2 - y1) / (x2 - x1)
                k2 = (y3 - y4) / (x3 - x4)
                print(k1 * k2)
                if 1 + ver_threshold > abs(k1 * k2) > 1-ver_threshold:
                    print("垂直")
                    print(x1, y1, x2, y2, x3, y3, x4, y4)
                # print(x1, y1, x2, y2, x3, y3, x4, y4)
                # print("相交yes")
        # if abs(c[0])
        # print(point[0], point[1], point[2], point[3])
        # if c < threshold and c > -threshold:
        #     print(c)


def draw_lanes2(img, lines, threshold):
    # a. 划分左右车道和水平车道
    left_lines, right_lines, verdict = [], [], []
    count_verdict(lines, threshold=0.1, ver_threshold=0.3000000000000000)
    # count_verdict(lines, threshold=0.1, ver_threshold=0.3000000000000000)
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

    draw_lines(img, right_lines, dir=1)  # 黄色
    draw_lines(img, left_lines, dir=2)  # 红色
    draw_lines(img, verdict, dir=3)  # 绿色
    # draw_lines(img, right_lines)
    # b. 清理异常数据
    # clean_lines(left_lines, 0.1)
    # clean_lines(right_lines, 0.1)

    # c. 得到左右车道线点的集合，拟合直线
    # left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    # print(left_points)
    # left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    # print(left_points)
    # right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    # right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    # left_results = least_squares_fit(left_points, 325, img.shape[0])
    # right_results = least_squares_fit(right_points, 325, img.shape[0])

    # 注意这里点的顺序
    # vtxs = np.array([[left_results[1], left_results[0], right_results[0], right_results[1]]])
    # d. 填充车道区域
    # cv2.fillPoly(img, vtxs, (0, 255, 0))

    # 或者只画车道线
    # cv2.line(img, left_results[0], left_results[1], (0, 255, 0), thickness)
    # cv2.line(img, right_results[0], right_results[1], (0, 255, 0), thickness)


def draw_lines(img, lines, dir: int):
    try:
        if dir == 1:
            color = [105, 255, 255]  # 黄色
        elif dir == 2:
            color = [101, 30, 255]  # 红色
        elif dir == 3:
            color = [101, 198, 39]  # 绿色
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), color, 3)
        # cv2.imshow("line", img)
    except:
        pass


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
    # 最小二乘法拟合
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]

    # polyfit第三个参数为拟合多项式的阶数，所以1代表线性
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)  # 获取拟合的结果

    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))

    return [(xmin, ymin), (xmax, ymax)]


if __name__ == '__main__':
    file = '/home/perfectman/PycharmProjects/graduation_design/raspberryPi/straight.jpg'
    # file = '/home/perfectman/PycharmProjects/graduation_design/raspberryPi/crossroads.jpg'
    # file = '/home/perfectman/PycharmProjects/graduation_design/raspberryPi/turn_left.jpg'
    img = cv2.imread(file)

    img = cv2.resize(img, (640, 480))
    # process_img2(img=img)
    cv2.imshow("org", img)
    cv2.imshow("process2", process_img2(img))
    # cv2.imshow("process1", process_img(img))
    cv2.waitKey(0)
