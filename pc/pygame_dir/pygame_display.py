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

COLOR_INACTIVE = pygame.Color('lightskyblue3')
COLOR_ACTIVE = pygame.Color('dodgerblue2')
# FONT = pygame.font.Font(None, 32)

# 初始化
def creat():
    pygame.init()

    # 创建一个窗口
    screen = pygame.display.set_mode([MAP_WIDTH, MAP_HEIGHT+100])  # 宽，高
    # 用白色填充屏幕
    font = pygame.font.Font(None, 32)
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    color = color_inactive
    active = False
    text = ''
    text2 = ''
    done = False
    # 文本输入框
    input_box1 = pygame.Rect(50, MAP_HEIGHT+50, 32, 32)
    input_box2 = pygame.Rect(250, MAP_HEIGHT+50, 32, 32)

    # 加载小车的图片，更新图像, 小车图片也是按照5：1
    # pngFileName = '/home/perfectman/PycharmProjects/graduation_design/pc/pygame_dir/car2.png'
    pngFileName = './car2.png'
    car = pygame.image.load(pngFileName)

    # 获取小车的边边
    carRect = car.get_rect()

    # 翻转

    speed = [0.5, 0.5]
    # 主循环
    # mRunning = True
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                # If the user clicked on the input_box rect.
                if input_box1.collidepoint(event.pos[0], event.pos[1]):
                    # Toggle the active variable.
                    active = not active
                else:
                    active = False
                # Change the current color of the input box.
                color = color_active if active else color_inactive
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        print(text)
                        text = ''
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode
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
        if CAR_X and CAR_Y:
            CAR_X = 5 * CAR_X - CAR_SIZE[0] / 2  # 减除半个车距
            CAR_Y = -((CAR_Y * 5) - MAP_HEIGHT + CAR_SIZE[1] / 2)
            screen.blit(car, [CAR_X, CAR_Y])
            # print(int(CAR_X), int(CAR_Y), 'pygame')
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
        txt_surface = font.render(text, True, color)
        txt_surface2 = font.render(text2, True, color)
        screen.fill(THECOLORS['white'])
        background = pygame.image.load(PYGAME_BACKGROUND_FILE_PATH[0:-4] + '_resize' + '.jpg')
        screen.blit(background, (0, 0))
        width = max(100, txt_surface.get_width()+10)
        input_box1.w = width
        # Blit the text.
        screen.blit(txt_surface, (input_box1.x+5, input_box1.y+5))
        screen.blit(txt_surface2, (input_box2.x+5, input_box2.y+5))
        # Blit the input_box rect.
        pygame.draw.rect(screen, color, input_box1, 2)
        pygame.draw.rect(screen, color, input_box2, 2)
        # pygame.draw.rect(screen, THECOLORS['white'], [CAR_X, CAR_Y, CAR_SIZE[0], CAR_SIZE[1]], 0)
        pygame.display.flip()
    pygame.quit()





if __name__ == '__main__':
    creat()
    # creat(CAR_X, CAR_Y)
    # start()
    # t = threading.Thread(target=creat)
    # t.start()
    # while 1:
    #     print(CAR_X, CAR_Y)
