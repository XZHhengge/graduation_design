# -*- coding: utf-8 -*-
import math
import time
import cv2
import numpy as np
from pc.camera_dir import camera_configs2 as camera_configs
from config import blue_upper, yellow_upper, red_lower, blue_lower, red_upper, yellow_lower, MARK_POS_OF_MAP, \
    CAMERA_POS_OF_MAP, CarVar, Map, PIX_TIME_OF_REAl, CAMERA_HIGH, COLLECT_TIME, COLLECT_TIMES, CAMERA_LENGTH

# 消除差异
FIRST = []
THREED_SAVE_LIST = []

AVERAGE_NOW_BLUE_DEEP_LIST = []


# 通过设定好红色mark的位置，从而获取到小车的位置


def get_correct_value(values: list, threshold):
    """
    误差消除, 迭代计算斜率均值，排除掉与差值差异较大的数据
    :param values:
    :return:
    """
    # 求众数
    # global FIRST
    slope = [(y / x) for x, y in values]
    while len(values) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]

        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            values.pop(idx)
        else:

            # return (np.mean())
            return values[0]


def get_2point_distance(point1, point2, shape: int):
    if shape == 1:
        return abs(point1 - point2)
    elif shape == 2:
        return int(math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))


def get_car_of_mark_x_distance(car_point, mark_point):
    return abs(car_point[0] - mark_point[0])


def get_coordinate(mark_pos_of_camera: tuple, power_pos_of_camera: tuple,
                   threeD, car_center):
    '''
        一般位置的摆放如下，单位都是cm
    ---------------------------------MARK
    -                                   -
    -                        CAR        -
    -                       / |         -
    -                      /  |         -
    -                     /   |         -
    -                    /    |         -
    -                   /     |         -
    ---------------CAMERA-----------------

    :param mark_pos_of_map: 标记物在真实地图上的位置
    :param camera_pos_of_map: 相机在真实地图上的位置
    :param power_pos_of_camera: 电池在相机上的位置
    :param mark_pos_of_camera: 标记物在相机上的位置
    :param threeD:   使用MATLAB矫正和opencv得到的深度矩阵
    '''
    #  计算深度
    car_deepth = threeD[power_pos_of_camera[1]][power_pos_of_camera[0]][-1]
    mark_deepth = threeD[mark_pos_of_camera[1]][mark_pos_of_camera[0]][-1]
    if car_center != 'inf' and mark_deepth != 'inf':
        if car_deepth > 0 and mark_deepth > 0:
            car_deepth /= 10.0
            car_deepth -= CAMERA_POS_OF_MAP[1]  # 减去相机的位置
            # mark_deepth /= 10.0  # 毫米转化为厘米
            # if abs(mark_deepth - math.sqrt(mark_pos_of_map[0]**2 + mark_pos_of_map[1] ** 2)) > 50:
            #     print("标记与相机测量误差太大")
            # else:
            # 计算小车与标记的横向距离,两点之间的距离在除以5，这个5是通过实际测量和图像点计算出来的
            # x_length = math.sqrt((power_pos_of_camera[0] - mark_pos_of_camera[0]) ** 2
            #                      + ((power_pos_of_camera[1] - mark_pos_of_camera[1]) ** 2)) / 5.0
            x_length2 = get_2point_distance(point1=power_pos_of_camera, point2=mark_pos_of_camera, shape=2)
            # print(x_length2)
            # x_length = get_car_of_mark_x_distance(power_pos_of_camera, mark_pos_of_camera)
            if power_pos_of_camera[0] > mark_pos_of_camera[0]:
                pass
                # x_length2 /= 2.0
            #     print(x_length, 'x轴距离')
            else:
                x_length2 /= 3.0
            # print(x_length2, 'x_length')

            #     print(x_length, '待定')
            # 在通过标记的坐标得到小车的横向坐标
            # x = mark_pos_of_map[0] - x_length
            x = MARK_POS_OF_MAP[0] - x_length2
            # print(x, '小车的x轴')
            # # print("小车与标记的距离为{}, 横向坐标为{}".format(x_length, x))
            # # 再用勾股定理得到y
            y = math.sqrt(car_deepth ** 2 - (x - CAMERA_POS_OF_MAP[0]) ** 2)
            y -= 200
            y *= 10
            print(x, 'x', y, 'y')
            # global FIRST
            # global CAR_X, CAR_Y

            if len(FIRST) == 5:
                CarVar.CAR_X, CarVar.CAR_Y = get_correct_value(FIRST, threshold=0.1)  # 5个一组计算坐标
                FIRST.clear()
            else:
                # print(x, y/
                # print('color_track,lines 89', x, y)
                if 0 < x <= 160 and 0 < y <= 1600:  # 获得小车坐标
                    FIRST.append([x, y / 10.0])


def threeD_get_coordinate(mark_pos_of_camera: tuple, power_pos_of_camera: tuple,
                          threeD, car_center) -> tuple:
    '''
    三维空间定位
    :param mark_pos_of_map: 标记物在真实地图上的位置
    :param camera_pos_of_map: 相机在真实地图上的位置
    :param power_pos_of_camera: 电池在相机上的位置
    :param mark_pos_of_camera: 标记物在相机上的位置
    :param threeD:   使用MATLAB矫正和opencv得到的深度矩阵
    '''
    #  计算深度
    car_deepth = threeD[power_pos_of_camera[1]][power_pos_of_camera[0]][-1]
    mark_deepth = threeD[mark_pos_of_camera[1]][mark_pos_of_camera[0]][-1]
    if car_center != 'inf' and mark_deepth != 'inf':
        if car_deepth > 0 and mark_deepth > 0:
            # car_deepth /= 10.0
            mark_deepth /= 10.0  # 毫米转化为厘米
            pix_x_length = get_car_of_mark_x_distance(car_point=power_pos_of_camera,
                                                      mark_point=mark_pos_of_camera) / PIX_TIME_OF_REAl
            print(pix_x_length, '两个像素点的x轴的距离')
            print(get_2point_distance(power_pos_of_camera, mark_pos_of_camera, shape=2) / PIX_TIME_OF_REAl)

    # pass


def toushibianhaun(img):
    rows, cols, channels = img.shape
    p1 = np.float32([[120, 25], [2, 402], [545, 26], [635, 417]])
    p2 = np.float32([[0, 0], [0, 1.4 * PIX_TIME_OF_REAl], [1.5 * PIX_TIME_OF_REAl, 0],
                     [1.5 * PIX_TIME_OF_REAl, 1.4 * PIX_TIME_OF_REAl]])
    # 获取透视变换矩阵
    M = cv2.getPerspectiveTransform(p1, p2)
    # 执行透视变换
    dst = cv2.warpPerspective(img, M, (cols, rows))
    # hsv2 = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2HSV)
    cv2.imshow('toushizhuanahun', dst)
    return dst


def rapi_tacker(cnts1, cnts3, threeD):
    '''
    水平追踪小车
    :return:
    '''
    car_center = None
    if len(cnts1) > 0:
        # if len(cnts2) > 0:
        c1 = max(cnts1, key=cv2.contourArea)  # 最大蓝色
        # c2 = max(cnts2, key=cv2.contourArea)  # 最大黄色
        # ((x1, y1), radius1) = cv2.minEnclosingCircle(c1)  # 外接圆
        # ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
        M1 = cv2.moments(c1)  # 计算轮廓的矩
        # M2 = cv2.moments(c2)
        # cv2.imshow("fame11", img1_rectified)
        # M1 = get_color_center(img1_rectified, blue_lower, blue_upper)
        if M1["m00"]:
            car_center = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))  # 电池
            # car_center[0] += 10
            print(car_center, '蓝色坐标')
            # if M2["m00"]:
            # center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))  # 车轮
            # length = get_2point_distance(center1, center2, 2)
            # if length < 400:
            # car_center = (int((center1[0] + center2[0]) / 2), int((center1[1] + center2[1]) / 2))  # 取两点之间的
            # car_center = (int(center1[0]), int(center1[1]))  # 取两点之间的
            # print(car_center)
        # else:
        #     print("目标丢失")
        #     CarVar.CAR_X, CarVar.CAR_Y = 0, 0
        # print("距离", length)
        # print("中心", car_center)
        # else:
        #     CarVar.CAR_X, CarVar.CAR_Y = 0, 0
        #     print("没有黄色目标")
        else:
            CarVar.CAR_X, CarVar.CAR_Y = 0, 0
            print("没有蓝色目标")

    # if Map.Flag:
    #     cv2.imshow('car lines', Map.Pic)
    # cv2.waitKey(1)
    if len(cnts3) > 0:
        c3 = max(cnts3, key=cv2.contourArea)
        M3 = cv2.moments(c3)
        if M3["m00"]:
            red_center = (int(M3["m10"] / M3["m00"]), int(M3["m01"] / M3["m00"]))
            print(red_center, '红色坐标')
            # if car_center:
            #         print(threeD.shape)
            #         print(threeD)
            if car_center:
                # print(car_center)
                # 获取平面小车坐标
                get_coordinate(red_center, car_center, threeD=threeD,
                               car_center=car_center)

                # 获取空间坐标
                # threeD_get_coordinate(red_center, car_center, threeD, car_center)
                # print(CAR_X, CAR_Y)
        else:
            print("红色丢失")


# def check_one_clor_high(cnts, threeD):
#     c = max(cnts, key=cv2.contourArea)


def check_two_clor_high(cnts1, cnts2, cnts3, threeD):
    '''
    高角度俯拍定位小车
    :cnts1:蓝色电池，用深度图来计算小车的高度
    :cnts2:透视转换后的黄色车轮，用来计算小车的x轴
    :cnts3:黄色车轮

    :return:
    '''
    if len(cnts1) > 0:
        # if len(cnts2) > 0:
        c1 = max(cnts1, key=cv2.contourArea)  # 最大蓝色
        # c2 = max(cnts2, key=cv2.contourArea)  # 最大黄色
        # c3 = max(cnts3, key=cv2.contourArea)  # 最大黄色
        # ((x1, y1), radius1) = cv2.minEnclosingCircle(c1)  # 外接圆
        # ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
        M1 = cv2.moments(c1)  # 计算轮廓的矩
        # M2 = cv2.moments(c2)
        # M3 = cv2.moments(c3)
        # cv2.imshow("fame11", img1_rectified)
        # M1 = get_color_center(img1_rectified, blue_lower, blue_upper)
        if M1["m00"]:
            car_center = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))  # 电池
            # car_center[0] += 10
            # print(car_center, '蓝色坐标')
            if len(THREED_SAVE_LIST) == COLLECT_TIMES:
                org_blue_deep = 0
                i = 0
                for threed in THREED_SAVE_LIST:
                    deep = threed[car_center[1]][car_center[0]][-1]
                    if deep > 0:
                        i += 1
                        org_blue_deep += deep
                org_blue_deep /= i
                now_blue_deep = threeD[car_center[1]][car_center[0]][-1]
                # print('高度差距为', org_blue_deep - now_blue_deep)

                # 这里利用三角形内部的比例计算出小车的高度
                # 取十次进行计算
                if len(AVERAGE_NOW_BLUE_DEEP_LIST) == 12:
                    AVERAGE_NOW_BLUE_DEEP_LIST.remove(max(AVERAGE_NOW_BLUE_DEEP_LIST))
                    AVERAGE_NOW_BLUE_DEEP_LIST.remove(min(AVERAGE_NOW_BLUE_DEEP_LIST))
                    sum = 0
                    for value in AVERAGE_NOW_BLUE_DEEP_LIST:
                        sum += value
                    average_now_blue_deep = sum / 10.0
                    # print(average_now_blue_deep, '平均后的高度')

                    # 利用比例
                    # CarVar.HIGH = (((org_blue_deep - average_now_blue_deep) * CAMERA_HIGH) / org_blue_deep) - 100
                    # CarVar.HIGH = (((org_blue_deep - average_now_blue_deep) * CAMERA_HIGH) / org_blue_deep) * 2
                    CarVar.HIGH = (((org_blue_deep - average_now_blue_deep) * CAMERA_HIGH) / org_blue_deep)
                    print('修正前', CarVar.HIGH)
                    if CarVar.CHANGE_VALUE:
                        print('开始修正')
                        if CarVar.CHANGE_VALUE[0] == '-':
                            CarVar.HIGH -= int(CarVar.CHANGE_VALUE[1:])
                        else:
                            CarVar.HIGH += int(CarVar.CHANGE_VALUE[1:])
                    print(CarVar.HIGH, '修正后')
                    # 减去误差值
                    AVERAGE_NOW_BLUE_DEEP_LIST.clear()

                    if len(cnts2) > 0:
                        c2 = max(cnts2, key=cv2.contourArea)
                        M2 = cv2.moments(c2)
                        if M2["m00"]:
                            yellow_ord_center = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))
                            # print(yellow_ord_center, '黄色透视转换坐标')
                            x = (yellow_ord_center[0] / PIX_TIME_OF_REAl) * 100
                            # print(x, '小车的x轴坐标')
                            # 利用勾股定理计算小车的y值
                            y = (math.sqrt(average_now_blue_deep**2 - CAMERA_HIGH **2) - CAMERA_LENGTH) - 100
                            # print(y, '小车的y值')
                            if len(FIRST) == 5:
                                CarVar.CAR_X, CarVar.CAR_Y = get_correct_value(FIRST, threshold=0.1)  # 5个一组计算坐标
                                FIRST.clear()
                            else:
                                # print(x, y/
                                # print('color_track,lines 89', x, y)
                                if 0 < x <= 160 and 0 < y <= 1600:  # 获得小车坐标
                                    FIRST.append([x, y / 10.0])

                    else:
                        print('木有黄色物体')

                else:
                    if now_blue_deep > 0:
                        AVERAGE_NOW_BLUE_DEEP_LIST.append(now_blue_deep)

    else:
        CarVar.CAR_X, CarVar.CAR_Y = 0, 0
        print("没有蓝色目标")

    # if Map.Flag:
    #     cv2.imshow('car lines', Map.Pic)
    # cv2.waitKey(1)





import time


# def save_threeD(threeD):
#     '''
#     保存三维深度图十次,10后视觉稳定
#     :param threeD:
#     :return:
#     '''
#     # time.sleep(10)
#     print('start to save')
#     global THREED_SAVE_LIST
#     if len(THREED_SAVE_LIST) == 10:
#         pass
#     else:
#         THREED_SAVE_LIST.append(threeD)


def main():
    np.seterr(invalid='ignore')
    # pts = deque(maxlen=16)
    cap1 = cv2.VideoCapture(1)
    cap2 = cv2.VideoCapture(2)
    cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # ret = cap.set(3, 640)  # X
    # ret = cap.set(4, 480)  # Y

    cv2.namedWindow("config", cv2.WINDOW_NORMAL)
    cv2.moveWindow("left", 600, 0)
    cv2.moveWindow("right", 0, 0)

    # SGBM设置调节的参数
    cv2.createTrackbar("num", "config", 5, 60, lambda x: None)
    cv2.createTrackbar("blockSize", "config", 10, 15, lambda x: None)
    cv2.createTrackbar("SpeckleWindowSize", "config", 0, 200, lambda x: None)
    cv2.createTrackbar("SpeckleRange", "config", 1, 2, lambda x: None)
    cv2.createTrackbar("UniquenessRatio", "config", 0, 15, lambda x: None)
    cv2.createTrackbar("MinDisparity", "config", 0, 255, lambda x: None)
    cv2.createTrackbar("PreFilterCap", "config", 1, 63, lambda x: None)  # 注意调节的时候这个值必须是奇数
    cv2.createTrackbar("disp12MaxDiff", "config", 1, 255, lambda x: None)

    # BM
    # cv2.createTrackbar("num", "config", 0, 60, lambda x: None)
    # cv2.createTrackbar("blockSize", "config", 94, 255, lambda x: None)
    # cv2.createTrackbar("SpeckleWindowSize", "config", 1, 10, lambda x: None)
    # cv2.createTrackbar("SpeckleRange", "config", 1, 255, lambda x: None)
    # cv2.createTrackbar("UniquenessRatio", "config", 1, 255, lambda x: None)
    # cv2.createTrackbar("TextureThreshold", "config", 1, 255, lambda x: None)
    # cv2.createTrackbar("UniquenessRatio", "config", 1, 255, lambda x: None)
    # cv2.createTrackbar("MinDisparity", "config", 0, 255, lambda x: None)
    # cv2.createTrackbar("PreFilterCap", "config", 1, 62, lambda x: None)  # 注意调节的时候这个值必须是奇数
    # cv2.createTrackbar("MaxDiff", "config", 1, 400, lambda x: None)
    start_time = time.time()
    while True:
        # 获取每一帧
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        # 根据更正map对图片进行重构
        img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
        # cv2.imshow("megre", np.hstack((img1_rectified, img2_rectified)))
        # picmerge2(img1_rectified, img2_rectified)
        # stitcher = cv2.createStitcher(False)
        # (status, stitched) = stitcher.stitch([img1_rectified, img2_rectified])

        # out = np.hstack((img1_rectified, img2_rectified))

        cv2.imshow("left", img1_rectified)

        # 透视变换
        img1 = toushibianhaun(img1_rectified)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        # 透视变换
        # toushibianhaun(img1_rectified)

        # if status == 0:
        # cv2.imwrite(args["output"], stitched)
        # stitched = picmerge(img1_rectified, img2_rectified)
        # cv2.imshow("Stitched", stitched)

        # cv2.imshow("merge", picmerge(img1_rectified, img2_rectified))
        # 换到 HSV
        hsv1 = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2HSV)

        imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
        # 根据阈值构建掩模
        mask1 = cv2.inRange(hsv1, blue_lower, blue_upper)  # 蓝色
        mask2 = cv2.inRange(hsv1, yellow_lower, yellow_upper)  # 黄色
        mask3 = cv2.inRange(hsv1, red_lower, red_upper)  # 红色

        # 透视转化后的黄色
        change_yellow_mask = cv2.inRange(img1, yellow_lower, yellow_upper)
        change_yellow_cnts2 = cv2.findContours(change_yellow_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

        # 寻找透视转换后的黄色轮廓
        # inRange()：介于lower / upper之间的为白色，其余黑色
        # mask = cv2.inRange(hsv, lower_black, upper_black)
        # 对原图像和掩模位运算,只保留原图中的蓝色部分
        # res1 = cv2.bitwise_and(img1_rectified, img1_rectified, mask=mask1)
        # res2 = cv2.bitwise_and(frame2, frame2, mask=mask2)

        # mask1 = cv2.erode(mask1, None, iterations=2)  # 腐蚀
        # mask2 = cv2.erode(mask2, None, iterations=2)  # 腐蚀
        # mask1 = cv2.dilate(mask1, None, iterations=2)  # 膨胀
        # mask2 = cv2.dilate(mask2, None, iterations=2)  # 膨胀
        # mask = cv2.findContours(mask.copy())
        cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]  # 寻找蓝色轮廓
        cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]  # 寻找黄色轮廓
        cnts3 = cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]  # 寻找红色轮廓

        num = cv2.getTrackbarPos("num", "config")  # 即最大视差值与最小视差值之差, 窗口大小必须是16的整数倍，int 型
        blockSize = cv2.getTrackbarPos("blockSize",
                                       "config")  # 匹配的块大小。它必须是> = 1的奇数。通常情况下，它应该在3--11的范围内。这里设置为大于11也可以，但必须为奇数。
        SpeckleWindowSize = cv2.getTrackbarPos("SpeckleWindowSize", "config")  # 19
        SpeckleRange = cv2.getTrackbarPos("SpeckleRange", "config")
        UniquenessRatio = cv2.getTrackbarPos("UniquenessRatio", "config")
        MinDisparity = cv2.getTrackbarPos("MinDisparity", "config")  # 0
        PreFilterCap = cv2.getTrackbarPos("PreFilterCap", "config")  #
        Disp12MaxDiff = cv2.getTrackbarPos("disp12MaxDiff", "config")  #
        if blockSize == 0:
            blockSize += 1
        if blockSize % 2 == 0:
            blockSize += 1
        if blockSize < 5:
            blockSize = 3

        # SGBM
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * num,
            blockSize=blockSize,
            P1=8 * 3 * blockSize ** 2,
            P2=32 * 3 * blockSize ** 2,
            # uniquenessRatio=5,
            # mode=cv2.STEREO_SGBM_MODE_HH,
            mode=cv2.STEREO_SGBM_MODE_HH4
            # mode=cv2.STEREO_SGBM_MODE_SGBM,
            # mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        # BM
        # stereo = cv2.StereoBM_create(
        #     numDisparities=16 * num,
        #     blockSize=blockSize,
        # )
        stereo.setPreFilterCap(PreFilterCap)
        stereo.setMinDisparity(MinDisparity)
        # stereo.setTextureThreshold(TextureThreshold)
        stereo.setUniquenessRatio(UniquenessRatio)
        stereo.setSpeckleWindowSize(SpeckleWindowSize)
        stereo.setSpeckleRange(SpeckleRange)
        stereo.setDisp12MaxDiff(Disp12MaxDiff)

        disparity = stereo.compute(imgL, imgR)
        dsp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow('dsp', dsp)

        # threeD = cv2.
        threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., camera_configs.Q)

        if time.time() - start_time > COLLECT_TIME:
            if len(THREED_SAVE_LIST) != COLLECT_TIMES:
                THREED_SAVE_LIST.append(threeD)
                print('保存成功, 数目为', len(THREED_SAVE_LIST))

        # 追踪小车
        # rapi_tacker(cnts1, cnts3, threeD)

        # 俯视角定位小车和测量小车高度
        check_two_clor_high(cnts1, change_yellow_cnts2, cnts2, threeD)

        k = cv2.waitKey(1)  # & 0xFF
        if k == ord('q'):
            break
    # 关闭窗口
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
