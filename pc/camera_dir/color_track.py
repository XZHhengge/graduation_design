# -*- coding: utf-8 -*-
import math
import time
import cv2
import numpy as np
from pc.camera_dir import camera_configs2 as camera_configs
from config import blue_upper, yellow_upper, red_lower, blue_lower, red_upper, yellow_lower, MARK_POS_OF_MAP, CAMERA_POS_OF_MAP, CarVar, Map

# 消除差异
FIRST = []

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


def get_coordinate(mark_pos_ofcamera: tuple, power_pos_ofcamera: tuple,
                   camera_pos_ofmap: tuple, threeD, mark_pos_ofmap, car_center) -> tuple:
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

    :param mark_pos_ofmap: 标记物在真实地图上的位置
    :param camera_pos_ofmap: 相机在真实地图上的位置
    :param power_pos_ofcamera: 电池在相机上的位置
    :param mark_pos_ofcamera: 标记物在相机上的位置
    :param threeD:   使用MATLAB矫正和opencv得到的深度矩阵
    '''
    #  计算深度
    car_deepth = threeD[power_pos_ofcamera[1]][power_pos_ofcamera[0]][-1]
    mark_deepth = threeD[mark_pos_ofcamera[1]][mark_pos_ofcamera[0]][-1]
    if car_center != 'inf' and mark_deepth != 'inf':
        if car_deepth > 0 and mark_deepth > 0:
            # car_deepth /= 10.0
            mark_deepth /= 10.0  # 毫米转化为厘米
            # if abs(mark_deepth - math.sqrt(mark_pos_ofmap[0]**2 + mark_pos_ofmap[1] ** 2)) > 50:
            #     print("标记与相机测量误差太大")
            # else:
            # 计算小车与标记的横向距离,两点之间的距离在除以5，这个5是通过实际测量和图像点计算出来的
            # x_length = math.sqrt((power_pos_ofcamera[0] - mark_pos_ofcamera[0]) ** 2
            #                      + ((power_pos_ofcamera[1] - mark_pos_ofcamera[1]) ** 2)) / 5.0
            x_length = get_2point_distance(point1=power_pos_ofcamera, point2=mark_pos_ofcamera, shape=2) / 5.0
            # 在通过标记的坐标得到小车的横向坐标
            x = mark_pos_ofmap[0] - x_length
            # print("小车与标记的距离为{}, 横向坐标为{}".format(x_length, x))
            # 再用勾股定理得到y
            y = math.sqrt(car_deepth ** 2 - (x - camera_pos_ofmap[0]) ** 2)
            # global FIRST
            # global CAR_X, CAR_Y
            if len(FIRST) == 5:
                CarVar.CAR_X, CarVar.CAR_Y = get_correct_value(FIRST, threshold=0.1)  # 20个一组计算坐标
                FIRST.clear()
            else:
                # print(x, y/
                # print('color_track,lines 89', x, y)
                if 0 < x <= 150 and 0 < y <= 1400:  # 获得小车坐标
                    FIRST.append([x, y/10.0])


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

        # print(status)

        # if status == 0:
        # cv2.imwrite(args["output"], stitched)
        # stitched = picmerge(img1_rectified, img2_rectified)
        # cv2.imshow("Stitched", stitched)

        # cv2.imshow("merge", picmerge(img1_rectified, img2_rectified))
        # 换到 HSV
        hsv1 = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2HSV)
        # hsv2 = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2HSV)

        imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
        # 根据阈值构建掩模
        mask1 = cv2.inRange(hsv1, blue_lower, blue_upper)  # 蓝色
        # mask2 = cv2.inRange(hsv1, yellow_lower, yellow_upper)  # 黄色
        mask3 = cv2.inRange(hsv1, red_lower, red_upper)  # 红色

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
        # cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]  # 寻找黄色轮廓
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

        # threeD = cv2.
        threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., camera_configs.Q)
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
                center1 = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))  # 电池
                # if M2["m00"]:
                    # center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))  # 车轮
                    # length = get_2point_distance(center1, center2, 2)
                    # if length < 400:
                # car_center = (int((center1[0] + center2[0]) / 2), int((center1[1] + center2[1]) / 2))  # 取两点之间的
                car_center = (int(center1[0]), int(center1[1]))  # 取两点之间的
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


        cv2.imshow('frame1', dsp)
        if Map.Flag:
            cv2.imshow('car lines', Map.Pic)
        # cv2.waitKey(1)
        if len(cnts3) > 0:
            c3 = max(cnts3, key=cv2.contourArea)
            M3 = cv2.moments(c3)
            if M3["m00"]:
                red_center = (int(M3["m10"] / M3["m00"]), int(M3["m01"] / M3["m00"]))
                # if car_center:
                #         print(threeD.shape)
                #         print(threeD)
                if car_center:
                    # 获取小车坐标
                    get_coordinate(red_center, car_center, camera_pos_ofmap=CAMERA_POS_OF_MAP, threeD=threeD,
                                   mark_pos_ofmap=MARK_POS_OF_MAP, car_center=car_center)
                    # print(CAR_X, CAR_Y)
                # print("红色坐标为", threeD[red_center[1]][red_center[0]])
            else:
                print("红色丢失")
        # if car_center:
        #     if threeD[car_center[1]][car_center[0]][-1] > 0:
        #         print("小车坐标", threeD[car_center[1]][car_center[0]])

        # cv2.imshow('image', image)
        # print(cv2.)
        # cv2
        # cv2.moveWindow('frame1', x=0, y=0)  # 原地
        # cv2.imshow('mask1', mask1)
        # cv2.imshow('mask2', mask2)
        # cv2.imshow('mask3', mask3)
        # cv2.moveWindow('mask', x=frame1.shape[1], y=0)  # 右边
        # cv2.imshow('res', res1)
        # cv2.moveWindow('res', y=frame1.shape[0], x=0)  # 下边

        k = cv2.waitKey(1)  # & 0xFF
        if k == ord('q'):
            break
    # 关闭窗口
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
