import RPi.GPIO as GPIO
import math
import time
import stream_client
import threading
import config
from config import Car, Map
import process_img
import cv2
from path import findShortPath
from drive_car import ThreadMoveCar

# 设置 GPIO 模式为 BCM
GPIO.setmode(GPIO.BOARD)
#
# 定义 GPIO 引脚
GPIO_TRIGGER = 18
GPIO_ECHO = 16
#
# 设置 GPIO 的工作方式 (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

# FLAG判断是否已经开启行车线程
FLAG = False

Distance_list = []


def get_position():
    '''
    计算小车当前的位置
    '''
    min_length, pos = 9999999, 0
    while 1:
        if config.Map.PATH:  # 主机是否下发路线
            global FLAG
            if config.Map.WIM:
                wim = config.Map.WIM
                node_dict = config.Map.NODE_DICT
                path = config.Map.PATH
                '''
                (448, 254)
                {'A': (0, 700), 'B': (750, 700), 'C': (750, 420.0), 'D': (0, 420.0), 'E': (0, 0), 'F': (750, 0)}
                [(0, 700), (750, 700), (750, 420.0), (0, 420.0), (0, 0), (750, 0)]
                '''
                if math.sqrt(abs((wim[0] - path[-1][0])) ** 2 + abs((wim[1] - path[-1][1]) ** 2)) < 100:
                    Car.STOP = True
                    # 查看是否在终点附近
                    print('到终点了, 车停')
                for i in range(len(path) - 1):
                    get_length = get_distance_from_point_to_line(wim, path[i], path[i + 1])
                    if get_length < min_length:
                        min_length, pos = get_length, i
                node_list = list(node_dict)
                node_values = list(node_dict.values())
                print('小车在{}到{}直线上'.format(node_list[node_values.index(path[pos])],
                                           node_list[node_values.index(path[pos + 1])]))
                # findShortPath(node_list[node_values.index(path[pos])], config.Map.TARGET)
            if not FLAG:
                print('car to start')
                t3 = ThreadMoveCar('move_car')
                t3.start()
                #     # print('main_thread_open_camera')
                #     # t = threading.Thread(target=open_camera)  # 打开小车的摄像头，开始行车，并接收位置信息
                #     # t.start()
                FLAG = True
            # time.sleep(2)


def open_camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    while ret:
        process_img.process3(frame)


def get_distance_from_point_to_line(point: tuple, line_of_point1: tuple, line_of_point2: tuple):
    '''
    计算点到直线的距离
    '''
    A = line_of_point2[1] - line_of_point1[1]
    B = line_of_point1[0] - line_of_point2[0]
    C = (line_of_point1[1] - line_of_point2[1]) * line_of_point1[0] + \
        (line_of_point2[0] - line_of_point1[0]) * line_of_point1[1]
    return abs(A * point[0] + B * point[1] + C) / (math.sqrt(A ** 2 + B ** 2))


def distance():
    '''
    超声波测距，用来测量前面的障碍物，进行调头/停下来
    '''

    min_length, pos = 9999999, 0
    while 1:
        # 发送高电平信号到 Trig 引脚
        GPIO.output(GPIO_TRIGGER, True)

        # 持续 10 us
        # time.sleep(0.00001)
        GPIO.output(GPIO_TRIGGER, False)
        start_time = time.time()
        stop_time = time.time()

        # 记录发送超声波的时刻1
        while GPIO.input(GPIO_ECHO) == 0:
            start_time = time.time()

        # 记录接收到返回超声波的时刻2
        while GPIO.input(GPIO_ECHO) == 1:
            stop_time = time.time()

        # 计算超声波的往返时间 = 时刻2 - 时刻1
        time_elapsed = stop_time - start_time
        # 声波的速度为 343m/s， 转化为 34300cm/s。
        distance = (time_elapsed * 34300) / 2
        print(distance)
        # print(Distance_list)
        if int(distance) < Car.THRESHOLD_DISTANCE_OF_OBSTACLE:
            Car.TURN_CORNER = True
            # time.sleep(1)
        # if len(Distance_list) == 2:
        #
        #     if int(sum(Distance_list) / 2) < Car.THRESHOLD_DISTANCE_OF_OBSTACLE:  # 遇到障碍物，调头/停下来，重新规划路线
        #         print(int(sum(Distance_list) / 2))
        #         print('turn corner')
        #         Car.TURN_CORNER = True
        #     Distance_list.clear()
        # else:
        #     Distance_list.append(distance)
        # if Map.DYNAMIC_PATH:

        # Car.TURN_AROUND = True
        #

        # wim = config.Map.WIM
        # node_dict = config.Map.NODE_DICT
        # path = config.Map.PATH
        # '''
        # (448, 254) ： wim
        # {'A': (0, 700), 'B': (750, 700), 'C': (750, 420.0), 'D': (0, 420.0), 'E': (0, 0), 'F': (750, 0)} : NODE_DICT
        # [(0, 700), (750, 700), (750, 420.0), (0, 420.0), (0, 0), (750, 0)] # PATH
        # '''
        # # if math.sqrt(abs((wim[0] - path[-1][0])) ** 2 + abs((wim[1] - path[-1][1]) ** 2)) < 100:
        # #     # 查看是否在终点附近
        # #     print('到终点了, 车停')
        # for i in range(len(path) - 1):
        #     get_length = get_distance_from_point_to_line(wim, path[i], path[i + 1])
        #     if get_length < min_length:
        #         min_length, pos = get_length, i
        # node_list = list(node_dict)
        # node_values = list(node_dict.values())
        # print('小车在{}到{}直线上'.format(node_list[node_values.index(path[pos])],
        #                            node_list[node_values.index(path[pos + 1])]))
        # print('删除边')  # 通过小车的位置，删除该边
        # config.Map.G.remove_edges_from([(path[pos], path[
        #     pos + 1])])  # [((0,700),(750, 700),length)] ex # F.remove_edges_from([(11,12), (13,14)])
        # path = findShortPath(node_list[node_values.index(path[pos])], config.Map.TARGET)  # 重新规划路线
        # if not path:  # 没有路
        #     config.Map.STOP = True

        time.sleep(0.5)


if __name__ == '__main__':
    t1 = threading.Thread(target=stream_client.send_video)  # tcp_client连接
    t1.start()
    # t1.join()
    t2 = threading.Thread(target=get_position)  # 初始路线规划
    t2.start()

    # 手动停止行车线程，使用ctrl+c停止，如果先中断主线程，行车速度会跑满，失控
    x = input('input')
    if isinstance(x, str):
        Car.STOP = True
        import sys

        sys.exit(0)
