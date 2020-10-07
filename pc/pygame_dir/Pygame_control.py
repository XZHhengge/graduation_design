# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2020/10/7
car_x = 0
car_y = 0
car_speed_x = 5
car_speed_y = 5

MAP_WIDTH = 750  # 根据真实地图 5:1，真实地图单位cm
MAP_HEIGHT = 700
# CAR_SIZE = (27*5, 15*5)
# 导入
import pygame
from pygame.color import THECOLORS

# 初始化
pygame.init()

# 创建一个窗口
screen = pygame.display.set_mode([MAP_WIDTH, MAP_HEIGHT])  # 宽，高

# 用白色填充屏幕
screen.fill(THECOLORS['white'])

background = pygame.image.load('paint1_resize.jpg')
screen.blit(background, (0,0))
# 加载小车的图片，更新图像
pngFileName = 'car2.png'
car = pygame.image.load(pngFileName)
screen.blit(car, [car_x, car_y])

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
    carRect = carRect.move(speed)
    # 小鸟的位置
    car_x = car_x + car_speed_x
    car_y = car_y + car_speed_y
    # 左右边缘
    if carRect.left < 0 or carRect.right > MAP_WIDTH:
        speed[0] = -speed[0]
    # 上下边缘
    if carRect.top < 0 or carRect.bottom > MAP_HEIGHT:
        speed[1] = -speed[1]
    # if car_x > 750 or car_x < 0:
    #     car_speed_x = -car_speed_x
    # if car_y > 740 or car_y < 0:
    #     car_speed_y = -car_speed_y
    screen.blit(car, carRect)
    pygame.display.flip()
pygame.quit()
