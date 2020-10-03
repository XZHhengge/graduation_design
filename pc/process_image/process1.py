# -*- coding:utf-8 -*-
import cv2
import numpy as np

"""
不同情况下的ROI区域
左转：
[0, 240]  [180, 0], [350,0]    [640, 240]


"""
# 高斯滤波核大小
blur_ksize = 5
# Canny边缘检测高低阈值
canny_lth = 50
canny_hth = 150

# 霍夫变换参数
rho = 1
theta = np.pi / 180
threshold = 15
min_line_len = 40
max_line_gap = 20


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 3)
    except:
        pass


def process_img(original_image):
    # processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)  # 灰度化
    processed_img = cv2.Canny(original_image, threshold1=200, threshold2=300)  # 边缘特征
    # cv2.imshow("ssaa", processed_img)
    processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)  # 高斯模糊
    vertices = np.array([[0, 370], [0, 240], [130, 125], [470, 125], [640, 240], [640, 370]], np.int32)
    processed_img = roi(processed_img, [vertices])  # 不规则ROI区域截取

    #                       edges
    lines = cv2.HoughLinesP(image=processed_img, rho=1, theta=np.pi / 180, threshold=100, minLineLength=30, maxLineGap=30)  # 霍夫直线检测
    draw_lines(processed_img, lines)  # 划线
    return processed_img


def process_img2(img):
    # 1. 灰度化、滤波和Canny
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 1)
    edges = cv2.Canny(blur_gray, canny_lth, canny_hth)
    # cv2.imshow("eeaa", edges)
    # cv2.imshow("edges", edges)
    # 2. 2. 标记四个坐标点用于ROI截取
    points = np.array([[0, 370], [0, 240], [190, 0], [360, 0], [640, 240], [640, 370]], np.int32)
    # points = np.array([[0, 0], [0, 480], [480, 640], [640, 0]], np.int32) # 全屏
    roi_edges = roi_mask(edges, [points])
    # 3. 霍夫直线提取
    drawing, lines = hough_lines(roi_edges, rho, theta, threshold, min_line_len, max_line_gap)  # 一条直线四个坐标
    print(len(lines))
    print(lines[0])
    # 4. 车道拟合计算
    # draw_lanes2(drawing, lines)  # 画出直线检测结果
    # 5. 最终将结果合在原图上
    # result = cv2.addWeighted(img, 0.9, drawing, 0.2, 0)
    # cv2.imshow("drawing", roi_edges)
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
    # draw_lines(drawing, lines)     # 画出直线检测结果

    return drawing, lines


def draw_lanes2(img, lines, color=[255, 0, 0], thickness=8):
    # a. 划分左右车道
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)  # 斜率
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return

    # b. 清理异常数据
    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)

    # c. 得到左右车道线点的集合，拟合直线
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    left_results = least_squares_fit(left_points, 325, img.shape[0])
    right_results = least_squares_fit(right_points, 325, img.shape[0])

    # 注意这里点的顺序
    vtxs = np.array([[left_results[1], left_results[0], right_results[0], right_results[1]]])
    # d. 填充车道区域
    # cv2.fillPoly(img, vtxs, (0, 255, 0))

    # 或者只画车道线
    cv2.line(img, left_results[0], left_results[1], (0, 255, 0), thickness)
    cv2.line(img, right_results[0], right_results[1], (0, 255, 0), thickness)


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
