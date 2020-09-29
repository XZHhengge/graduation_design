# *_*coding:utf-8 *_*
import cv2
import time
from pc.camera_dir import camera_configs2 as camera_configs

AUTO = False  # 自动拍照，或手动按s键拍照
INTERVAL = 2  # 自动拍照间隔

cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 400, 0)
left_camera = cv2.VideoCapture(1)
right_camera = cv2.VideoCapture(2)
left_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
right_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# cv2.stereoCalibrate()
counter = 0
utc = time.time()
pattern = (12, 8)  # 棋盘格尺寸
left_folder = "./snapshot/left_picture/"  # 拍照文件目录
right_folder = "./snapshot/right_picture/"  # 拍照文件目录


def shot(pos, frame):
    global counter
    if pos == 'left':
        path = left_folder + pos + "_" + str(counter) + ".PNG"
    else:
        path = right_folder + pos + "_" + str(counter) + ".PNG"

    cv2.imwrite(path, frame)
    print("snapshot saved into: " + path)


while True:
    ret, left_frame = left_camera.read()
    ret, right_frame = right_camera.read()

    img1_rectified = cv2.remap(left_frame, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR,
                               cv2.BORDER_CONSTANT)
    img2_rectified = cv2.remap(right_frame, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR,
                               cv2.BORDER_CONSTANT)


    now = time.time()
    # if AUTO and now - utc >= INTERVAL:
    #     shot("left", left_frame)
    #     shot("right", right_frame)
    #     counter += 1
    #     utc = now

    key = cv2.waitKey(1)
    # time.sleep(10)
    # key = input('in')
    if key == ord("q"):
        break
    # elif key == ord("s"):
    elif key == ord("s"):
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1
    cv2.imshow("left", left_frame)
    cv2.imshow("right", right_frame)

    cv2.imshow("revise_left", img1_rectified)
    cv2.imshow("revise_right", img2_rectified)
left_camera.release()
right_camera.release()
cv2.destroyWindow("left")
cv2.destroyWindow("right")
