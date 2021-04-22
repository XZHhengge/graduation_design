# -*- coding:utf-8 -*-
import time
import RPi.GPIO as GPIO
from config import Car, Road, Map
import threading
#
# right_forward = 11  # 右轮前进
# right_back = 12  # 右轮后退
# left_forward = 13  # 左轮前进
# left_back = 15  # 左轮后退

FLAG = True


class MoveCar:
    '''
    小车行驶
    '''
    def __init__(self):
        self.GPIO = GPIO
        self.GPIO.setwarnings(False)
        right_forward = 11  # 右轮前进
        right_back = 12  # 右轮后退
        left_forward = 13  # 左轮前进
        left_back = 15  # 左轮后退
        self.GPIO.setmode(GPIO.BOARD)
        self.GPIO.setup(right_forward, GPIO.OUT)  # right straight
        self.GPIO.setup(left_forward, GPIO.OUT)  # left straight
        self.GPIO.setup(right_back, GPIO.OUT)  # right back
        self.GPIO.setup(left_back, GPIO.OUT)
        self.right_forward = GPIO.PWM(right_forward, 50)  # channel=13 frequency=50Hz
        self.left_forward = GPIO.PWM(left_forward, 50)  # channel=12 frequency=50Hz
        self.right_back = GPIO.PWM(right_back, 50)
        self.left_back = GPIO.PWM(left_back, 50)

    def reset(self):
        self.right_forward.stop()
        self.right_back.stop()
        self.left_forward.stop()
        self.left_back.stop()
        self.right_forward.start(0)
        self.right_back.start(0)
        self.left_forward.start(0)
        self.left_back.start(0)

    # '''right_forward = 11  # 右轮前进
    # right_back = 12  # 右轮后退
    # left_forward = 13  # 左轮前进
    # left_back = 15  # 左轮后退
    # '''
    '''
        left_forward_v = 14  #
    left_back_v = 0
    right_forward_v = 6
    right_back_v = 0
    '''

    def go_straight(self):
        self.reset()
        # self.right_forward.ChangeDutyCycle(6)

        while 1:
            # print('go_loop')
            # print('process_img', Road.IS_STRAIGHT)
            # if Road.IS_STRAIGHT:
            try:
                # print('go!!!!!!!!!!!!!')
                # print('左轮速度:{}, 右轮速度:{}'.format(Car.left_forward_v, Car.right_forward_v))
                self.right_forward.ChangeDutyCycle(Car.right_forward_v)
                self.left_forward.ChangeDutyCycle(Car.left_forward_v)
                # Road.IS_STRAIGHT = False
                if Car.STOP:
                    self.reset()
                    # exit()
                # time.sleep(0.5)
            except Exception:
                self.reset()
                self.clean()

    def go_straight2(self):
        self.reset()
        # self.right_forward.ChangeDutyCycle(6)
        # while 1:
        #     while Road.IS_STRAIGHT:

        while 1:
            self.right_forward.ChangeDutyCycle(Car.right_forward_v)
            self.left_forward.ChangeDutyCycle(Car.left_forward_v)
            if Car.TURN_CORNER:  # 查看是否转弯且路径存在
                if Map.DYNAMIC_PATH:
                    action = Map.DYNAMIC_PATH.pop()
                    # del Map.DYNAMIC_PATH[0]
                    if action == 'a':  # 左转
                        self.turn_left()
                        Car.TURN_CORNER = False
                        self.go_straight2()
                    elif action == 'd':  # 右转
                        self.turn_right()
                        Car.TURN_CORNER = False
                        self.go_straight2()
            # if not Map.DYNAMIC_PATH:
            #     self.reset()
            if Car.STOP:
                self.reset()
            #     exit()
            # else:
            #     self.go_straight2()
    #     # self.clean()
    #     # while Road.IS_CROSS:
    #     #     print('corss')
    #     #     # 从DYNAMIC_PATH中获取左转，右转
    #     #     if Map.DYNAMIC_PATH:
    #     #         if Map.DYNAMIC_PATH[0] == 'a':
    #     #             self.turn_left()
    #     #         elif Map.DYNAMIC_PATH[0] == 'd':
    #     #             self.turn_right()
    #     #     else:
    #     #         # 手动加入测试
    #     #         self.turn_right()
    #     # self.clean()
    #     # self.left_forward.ChangeDutyCycle(14)
    #     # time.sleep(0.1)
    def go_straight3(self):
        self.reset()
        self.right_forward.ChangeDutyCycle(15)
        self.left_forward.ChangeDutyCycle(28)

    def stop(self):
        self.reset()
        self.right_back.ChangeDutyCycle(25)
        self.left_back.ChangeDutyCycle(25)
        self.right_back.stop()
        self.left_back.stop()

    def turn_left(self):
        self.reset()
        # self.left_forward.stop()
        # self.left_forward.start()
        self.right_forward.ChangeDutyCycle(50)
        self.left_back.ChangeDutyCycle(15)
        # self.left_forward.ChangeDutyCycle(15)
        # self.right_forward.stop()
        # self.right_forward.ChangeDutyCycle(30)
        time.sleep(1.25)
        self.reset()
        # self.go_straight()
        # time.sleep(0.5)

    def turn_right(self):
        self.reset()
        # while 1:

        # self.clean()
        # self.left_forward.stop()
        self.left_forward.ChangeDutyCycle(60)
        self.right_back.ChangeDutyCycle(30)
        # self.right_forward.ChangeDutyCycle(15)
        # # self.right_forward.stop()
        time.sleep(1.5)
        # # self.right_forward.ChangeDutyCycle(9)
        self.reset()
        # self.go_straight()
        # time.sleep(0.5)

    def backward(self):
        self.reset()
        self.right_back.ChangeDutyCycle(40)
        self.left_back.ChangeDutyCycle(40)

    def turn_around(self):
        self.reset()
        # self.left_forward.ChangeDutyCycle(8)
        self.right_forward.ChangeDutyCycle(35)
        self.left_back.ChangeDutyCycle(35)
        # self.right_forward.stop()
        # self.right_forward.ChangeDutyCycle(30)
        time.sleep(1)
        self.reset()
        # self.go_straight()

    def clean(self):
        self.GPIO.cleanup()

    def test(self):
        self.left_forward.ChangeDutyCycle(45)


class ThreadMoveCar(threading.Thread):  # 继承父类threading.Thread
    # def __init__(self, threadID, name, counter):
    def __init__(self, name):
        threading.Thread.__init__(self)
        # self.threadID = threadID
        self.name = name
        self.car = MoveCar()
        # self.counter = counter

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        # print('Map.DYNAMIC_PATH', Map.DYNAMIC_PATH)
        self.car.go_straight2()


if __name__ == '__main__':
    MC = MoveCar()
    # MC.left_back_v
    try:
        while 1:
            x = input('in')
            if x == 'w':
                MC.go_straight3()
            elif x == 'a':
                MC.turn_left()
            elif x == 'd':
                MC.turn_right()
            elif x == 's':
                MC.stop()
            elif x == 'ss':
                MC.backward()
            elif x == 'au':
                try:
                    while 1:
                        MC.go_straight()
                        time.sleep(2)
                        MC.backward()
                        time.sleep(2)
                except KeyboardInterrupt:
                    pass
            elif x == 't':
                MC.turn_around()
            elif x == 'test':
                MC.test()

    except KeyboardInterrupt:
        GPIO.cleanup()
        # pass
    # p.stop()
    # finally:
    #     GPIO.cleanup()
