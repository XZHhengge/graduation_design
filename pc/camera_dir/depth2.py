# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2020/9/21
'''
使用opencv的SGBM进行深度检测 算法解释
https://blog.csdn.net/zfjBIT/article/details/91436530?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param
'''
import numpy as np
import cv2
# from pc import camera_configs
from pc.camera_dir import camera_configs2 as camera_configs

camera1 = cv2.VideoCapture(1)
camera2 = cv2.VideoCapture(2)
camera1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
camera2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# cv2.namedWindow("left")
# cv2.namedWindow("right")
cv2.namedWindow("depth")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 600, 0)
# 创建设置的bar
cv2.namedWindow("config", cv2.WINDOW_NORMAL)
cv2.moveWindow("left", 600, 0)
cv2.moveWindow("right", 0, 0)

# 设置调节的参数
cv2.createTrackbar("num", "config", 5, 60, lambda x: None)
cv2.createTrackbar("blockSize", "config", 1, 15, lambda x: None)
cv2.createTrackbar("SpeckleWindowSize", "config", 0, 200, lambda x: None)
cv2.createTrackbar("SpeckleRange", "config", 1, 2, lambda x: None)
cv2.createTrackbar("UniquenessRatio", "config", 5, 15, lambda x: None)
cv2.createTrackbar("MinDisparity", "config", 0, 255, lambda x: None)
cv2.createTrackbar("PreFilterCap", "config", 1, 63, lambda x: None)  # 注意调节的时候这个值必须是奇数
cv2.createTrackbar("disp12MaxDiff", "config", 1, 255, lambda x: None)


# 添加点击事件，打印当前点的距离
def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        print(threeD[y][x])


#
cv2.setMouseCallback("depth", callbackFunc, None)

# 初始化计算FPS需要用到参数 注意千万不要用opencv自带fps的函数，那个函数得到的是摄像头最大的FPS
# frame_rate_calc = 1
# freq = cv2.getTickFrequency()
# font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()

    if not ret1 or not ret2:
        print("camera is not connected!")
        break

    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

    # 将图片置为灰度图，为StereoSGBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    #
    # print(imgL)
    # print(imgR)


    # 合并两张图
    out = np.hstack((img1_rectified, img2_rectified))

    cv2.imshow("epipolar lines", out)
    # trackbar用来调节不同的参数查看效果
    num = cv2.getTrackbarPos("num", "config")  # 即最大视差值与最小视差值之差, 窗口大小必须是16的整数倍，int 型
    blockSize = cv2.getTrackbarPos("blockSize","config")  # 匹配的块大小。它必须是> = 1的奇数。通常情况下，它应该在3--11的范围内。这里设置为大于11也可以，但必须为奇数。
    SpeckleWindowSize = cv2.getTrackbarPos("SpeckleWindowSize", "config")  # 19
    SpeckleRange = cv2.getTrackbarPos("SpeckleRange", "config")
    UniquenessRatio = cv2.getTrackbarPos("UniquenessRatio", "config")
    MinDisparity = cv2.getTrackbarPos("MinDisparity", "config")  # 0
    PreFilterCap = cv2.getTrackbarPos("PreFilterCap", "config")  #
    Disp12MaxDiff = cv2.getTrackbarPos("disp12MaxDiff", "config")  #

    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 5:
        blockSize = 3

    # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=0,
    #     numDisparities=240,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    #     blockSize=3,
    #     P1=8 * 3 * 3 ** 2,
    #     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    #     P2=32 * 3 * 3 ** 2,
    #     disp12MaxDiff=1,
    #     uniquenessRatio=15,
    #     speckleWindowSize=0,
    #     speckleRange=2,
    #     preFilterCap=63,
    #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    # )
    # print(num, blockSize)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * num,
        blockSize=blockSize,
        P1=8 * 3 * blockSize ** 2,
        P2=32 * 3 * blockSize ** 2,
        # mode=cv2.STEREO_SGBM_MODE_HH,
        # mode=cv2.STEREO_SGBM_MODE_HH4
        # mode=cv2.STEREO_SGBM_MODE_SGBM,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    # 8\*number_of_image_channels\*SADWindowSize\*SADWindowSize
    # 32\*number_of_image_channels\*SADWindowSize\*SADWindowSize
    # stereo.setROI1(camera_configs.validPixROI1)
    # stereo.setROI2(camera_configs.validPixROI2)
    stereo.setPreFilterCap(PreFilterCap)
    stereo.setMinDisparity(MinDisparity)
    # stereo.setTextureThreshold(TextureThreshold)
    stereo.setUniquenessRatio(UniquenessRatio)
    stereo.setSpeckleWindowSize(SpeckleWindowSize)
    stereo.setSpeckleRange(SpeckleRange)
    stereo.setDisp12MaxDiff(Disp12MaxDiff)

    disparity = stereo.compute(imgL, imgR)

    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., camera_configs.Q)

    # 将深度图转为伪色图，这一步对深度测量没有关系，只是好看而已
    fakeColorDepth = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    cv2.imshow("depth", disp)
    cv2.imshow("fakeColor", fakeColorDepth)  # 输出深度图的伪色图，这个图没有用只是好看

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite("./snapshot/BM_left.jpg", imgL)
        cv2.imwrite("./snapshot/BM_right.jpg", imgR)
        cv2.imwrite("./snapshot/BM_depth.jpg", disp)

camera1.release()
camera2.release()
cv2.destroyAllWindows()
