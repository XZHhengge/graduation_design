# -*- coding:utf-8 -*-
# coding:utf-8
import cv2
import numpy as np
from pc.camera_dir import camera_configs2 as camera_configs
 

def nothing(args):
    pass

camera1 = cv2.VideoCapture(1)
camera2 = cv2.VideoCapture(2)
camera1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
camera2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# img = cv2.imread(r"C:\Users\Administrator\Desktop\frame.png")
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.namedWindow('tracks')
cv2.createTrackbar("LH", "tracks", 0, 255, nothing)
cv2.createTrackbar("LS", "tracks", 0, 255, nothing)
cv2.createTrackbar("LV", "tracks", 0, 255, nothing)

cv2.createTrackbar("UH", "tracks", 255, 255, nothing)
cv2.createTrackbar("US", "tracks", 255, 255, nothing)
cv2.createTrackbar("UV", "tracks", 255, 255, nothing)

# switch = "0:OFF \n1:ON"
# cv2.createTrackbar(switch,"tracks",0,1,nothing)


while 1:

    ret, frame1 = camera1.read()
    ret, frame2 = camera2.read()
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    # img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
    # cv2.imshow("left", img1_rectified)
    # cv2.imshow("right", img2_rectified)
    img_hsv = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("LH", "tracks")
    l_s = cv2.getTrackbarPos("LS", "tracks")
    l_v = cv2.getTrackbarPos("LV", "tracks")
    u_h = cv2.getTrackbarPos("UH", "tracks")
    u_s = cv2.getTrackbarPos("US", "tracks")
    u_v = cv2.getTrackbarPos("UV", "tracks")

    lower_b = np.array([l_h, l_s, l_v])
    upper_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(img_hsv, lower_b, upper_b)
    res = cv2.add(img1_rectified, img1_rectified, mask=mask)

    # cv2.imshow("img", img)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    k = cv2.waitKey(1)
    if k == 27:
        break

    # print(r,g,b)
    # if s==0:
    # img[:]=0
    # else:
    # img[:]=

cv2.destroyAllWindows()

# 蓝色:
# lower = np.array([95, 200, 140])
# upper = np.array([255, 255, 255])

# 黄色
# lower = np.array([22, 138, 145])
# upper = np.array([76, 255, 255])
