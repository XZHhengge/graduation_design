# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2020/10/7

# 导入
import pygame
from pygame.color import THECOLORS
import globalVar

car_x = 0
car_y = 0
car_speed_x = 5
car_speed_y = 5

MAP_WIDTH = 750  # 根据真实地图 5:1，真实地图单位cm
MAP_HEIGHT = 700


# CAR_SIZE = (27*5, 15*5)


# 初始化
def creat():
    pygame.init()

    # 创建一个窗口
    screen = pygame.display.set_mode([MAP_WIDTH, MAP_HEIGHT])  # 宽，高

    # 用白色填充屏幕
    screen.fill(THECOLORS['white'])

    background = pygame.image.load('/home/perfectman/PycharmProjects/graduation_design/pc/pygame_dir/paint1_resize.jpg')
    screen.blit(background, (0, 0))
    # 加载小车的图片，更新图像, 小车图片也是按照5：1
    pngFileName = '/home/perfectman/PycharmProjects/graduation_design/pc/pygame_dir/car2.png'
    car = pygame.image.load(pngFileName)

    # 获取小车的边边
    carRect = car.get_rect()

    # 翻转
    pygame.display.flip()

    speed = [0.5, 0.5]
    # 主循环
    mRunning = True
    while mRunning:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                mRunning = False
        # 时间延迟
        # pygame.time.delay(20)
        # # 覆盖痕迹
        # pygame.draw.rect(screen, THECOLORS['white'], [car_x, car_y, 100, 100], 0)
        # carRect = carRect.move(speed)
        # 小鸟的位置
        # car_x = car_x + car_speed_x
        # car_y = car_y + car_speed_y
        # 左右边缘
        CAR_X, CAR_Y = globalVar.GloVar.CAR_X, globalVar.GloVar.CAR_Y
        if CAR_X and CAR_Y:
            # print(CAR_X, CAR_Y)
            CAR_X *= 5
            CAR_Y = -((CAR_Y*5) - MAP_HEIGHT)
            update(CAR_X, CAR_Y)
            # print(int(CAR_X), int(CAR_Y), 'pygame')
        # print(color_track.CAR_Y, color_track.CAR_Y)
        if carRect.left < 0 or carRect.right > MAP_WIDTH:
            speed[0] = -speed[0]
        # 上下边缘
        if carRect.top < 0 or carRect.bottom > MAP_HEIGHT:
            speed[1] = -speed[1]
        # if car_x > 750 or car_x < 0:
        #     car_speed_x = -car_speed_x
        # if car_y > 740 or car_y < 0:
        #     car_speed_y = -car_speed_y
        screen.blit(car, [CAR_X, CAR_Y])
        pygame.display.flip()
    pygame.quit()


s = []


def update(x, y):
    global s
    if s:
        if s[0] == (x, y):
            pass
        else:
            print(x, y)
            s[0] = (x, y)
    else:
        print(x, y)
        s.append((x, y))


if __name__ == '__main__':
    creat()
    # start()
    # t = threading.Thread(target=creat)
    # t.start()
    # while 1:
    #     print(CAR_X, CAR_Y)
