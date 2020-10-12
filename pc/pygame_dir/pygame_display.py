# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2020/10/7

import pygame
from pygame.color import THECOLORS
# import globalVar
from config import MAP_WIDTH, MAP_HEIGHT, CAR_SIZE, PYGAME_BACKGROUND_FILE_PATH, CarVar
# from pc.camera_dir.color_track import CAR_X, CAR_Y
car_x = 0
car_y = 0
car_speed_x = 5
car_speed_y = 5


# 初始化
def creat():
    pygame.init()

    # 创建一个窗口
    screen = pygame.display.set_mode([MAP_WIDTH, MAP_HEIGHT])  # 宽，高

    # 用白色填充屏幕
    screen.fill(THECOLORS['white'])

    background = pygame.image.load(PYGAME_BACKGROUND_FILE_PATH[0:-4] + '_resize' + '.jpg')
    screen.blit(background, (0, 0))
    # 加载小车的图片，更新图像, 小车图片也是按照5：1
    # pngFileName = '/home/perfectman/PycharmProjects/graduation_design/pc/pygame_dir/car2.png'
    pngFileName = './car2.png'
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
        CAR_X, CAR_Y = CarVar.CAR_X, CarVar.CAR_Y
        # CAR_X, CAR_Y = globalVar.GloVar.CAR_X, globalVar.GloVar.CAR_Y
        if CAR_X and CAR_Y:
            # print(CAR_X, CAR_Y, 'pygame_display')
            CAR_X = 5 * CAR_X - CAR_SIZE[0] / 2  # 减除半个车距
            CAR_Y = -((CAR_Y * 5) - MAP_HEIGHT + CAR_SIZE[1] / 2)
            # update(CAR_X, CAR_Y)
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
        # pygame.time.delay(20)
        pygame.draw.rect(screen, THECOLORS['white'], [CAR_X, CAR_Y, CAR_SIZE[0], CAR_SIZE[1]], 0)
        screen.blit(car, [CAR_X, CAR_Y])
        pygame.display.flip()
    pygame.quit()


if __name__ == '__main__':
    pass
    # creat(CAR_X, CAR_Y)
    # start()
    # t = threading.Thread(target=creat)
    # t.start()
    # while 1:
    #     print(CAR_X, CAR_Y)
