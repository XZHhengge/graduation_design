# -*- coding:utf-8 -*-
import time
import RPi.GPIO as GPIO
from config import Car, Road, Map
import threading
right_forward = 11  # 右轮前进
right_back = 12  # 右轮后退
left_forward = 13  # 左轮前进
left_back = 15  # 左轮后退

FLAG = True

V = 20



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
        global V
        while 1:
        # if Road.IS_STRAIGHT:
        #     if V >
            try:
                # print('go!!!!!!!!!!!!!')
                # print('左轮速度:{}, 右轮速度:{}'.format(Car.left_forward_v, Car.right_forward_v))
                print('左轮速度:{}, 右轮速度:{}'.format(V, V))
                # self.right_forward.ChangeDutyCycle(Car.right_forward_v)
                self.right_forward.ChangeDutyCycle(V)
                # self.left_forward.ChangeDutyCycle(Car.left_forward_v)
                self.left_forward.ChangeDutyCycle(V)
            except Exception:
                self.reset()
                self.clean()
                # break

            # while Road.IS_CROSS:
            #     print('cross')
            #     # 从DYNAMIC_PATH中获取左转，右转
            #     try:
            #         if Map.DYNAMIC_PATH:
            #             if Map.DYNAMIC_PATH[0] == 'a':
            #                 self.turn_left()
            #             elif Map.DYNAMIC_PATH[0] == 'd':
            #                 self.turn_right()
            #         else:
            #             # 手动加入测试
            #             print('handle cross')
            #             try:
            #                 while 1:
            #                     self.left_forward.ChangeDutyCycle(45)
            #                     time.sleep(1)
            #                     # self.clean()
            #                     break
            #             except KeyboardInterrupt:
            #                 self.reset()
            #                 self.clean()
            #         print('return')
            #         Road.IS_CROSS = False
            #         Road.IS_STRAIGHT = True
            #     except KeyboardInterrupt:
            #         self.reset()
            #         self.clean()
            # self.reset()
            # self.clean()

        # self.left_forward.ChangeDutyCycle(14)
        # time.sleep(0.1)

    def go_straight2(self):
        self.reset()
        # self.right_forward.ChangeDutyCycle(6)
        # while 1:
        #     while Road.IS_STRAIGHT:
        self.right_forward.ChangeDutyCycle(20)
        self.left_forward.ChangeDutyCycle(20)
        # self.clean()
        # while Road.IS_CROSS:
        #     print('corss')
        #     # 从DYNAMIC_PATH中获取左转，右转
        #     if Map.DYNAMIC_PATH:
        #         if Map.DYNAMIC_PATH[0] == 'a':
        #             self.turn_left()
        #         elif Map.DYNAMIC_PATH[0] == 'd':
        #             self.turn_right()
        #     else:
        #         # 手动加入测试
        #         self.turn_right()
        # self.clean()
        # self.left_forward.ChangeDutyCycle(14)
        # time.sleep(0.1)

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
        self.right_forward.ChangeDutyCycle(63)
        # self.left_forward.ChangeDutyCycle(15)
        # self.right_forward.stop()
        # self.right_forward.ChangeDutyCycle(30)
        time.sleep(1.5)
        self.reset()
        # self.go_straight()
        # time.sleep(0.5)

    def turn_right(self):
        self.reset()
        # while 1:

        # self.clean()
        # self.left_forward.stop()
        self.left_forward.ChangeDutyCycle(70)
        # self.right_forward.ChangeDutyCycle(15)
        # # self.right_forward.stop()
        time.sleep(1.5)
        # # self.right_forward.ChangeDutyCycle(9)
        self.reset()
        # self.go_straight()
        # time.sleep(0.5)

    def backward(self):
        self.reset()
        self.right_back.ChangeDutyCycle(11)
        self.left_back.ChangeDutyCycle(25)

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
        self.car.go_straight()


def upv():
    for i in range(30):
        global V
        V += 1
        time.sleep(0.5)


if __name__ == '__main__':

    # 创建新线程
    # thread1 = led_blue(1, "light_blue_on_on", 1)
    # try:
    #     thread2 = ThreadMoveCar('xxx')
    #     t3 = threading.Thread(target=upv)
    #     # 开启线程
    #     # thread1.start()
    #
    #     thread2.start()
    #     # thread2.setDaemon(True)
    #     t3.start()
    #     t3.join()
    #     # time.sleep(30)
    # except KeyboardInterrupt:
    # # time.sleep(2)
    #     GPIO.cleanup()
    # pass
    # p.stop()
    # finally:
    #     GPIO.cleanup()
    pass