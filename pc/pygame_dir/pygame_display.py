# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2020/10/7

import pygame
from pygame.color import THECOLORS
from config import MAP_WIDTH, MAP_HEIGHT, CAR_SIZE, PYGAME_BACKGROUND_FILE_PATH, CarVar, Tcp, \
    VIR_SIZE_TIMES_OF_REALITY_SIZE
from pygame.locals import *

left = 20
right = 20


# 初始化
def creat():
    pygame.init()

    # 创建一个窗口
    screen = pygame.display.set_mode([MAP_WIDTH, MAP_HEIGHT+100])  # 宽，高
    # 用白色填充屏幕
    screen.fill(THECOLORS['white'])  # 背景颜色
    background = pygame.image.load(PYGAME_BACKGROUND_FILE_PATH[0:-4] + '_resize' + '.jpg')  # 导入背景图
    screen.blit(background, (0, 0))
    font = pygame.font.Font(None, 32)

    color_inactive1, color_inactive2 = pygame.Color('lightskyblue3'), pygame.Color('lightskyblue3')
    color_active1, color_active2 = pygame.Color('dodgerblue2'), pygame.Color('dodgerblue2')
    color1 = color_inactive1
    color2 = color_inactive2
    button_color_inactive = (255, 0, 0)
    button_color_active = (0, 255, 0)
    button_color = button_color_inactive
    active = False
    active2 = False
    active3 = False
    text = ''
    text2 = ''
    done = False
    # 文本输入框
    input_box1 = pygame.Rect(MAP_WIDTH/(VIR_SIZE_TIMES_OF_REALITY_SIZE+2), MAP_HEIGHT+50, 32, 32)
    input_box2 = pygame.Rect(MAP_WIDTH*3/(VIR_SIZE_TIMES_OF_REALITY_SIZE+2), MAP_HEIGHT+50, 32, 32)
    button = pygame.Rect(MAP_WIDTH*5/(VIR_SIZE_TIMES_OF_REALITY_SIZE+2), MAP_HEIGHT+50, 44, 30)
    # 加载小车的图片，更新图像, 小车图片也是按照5：1
    # pngFileName = '/home/perfectman/PycharmProjects/graduation_design/pc/pygame_dir/car2.png'


    pngFileName = './car2.png'
    car = pygame.image.load(pngFileName)

    # 获取小车的边边
    carRect = car.get_rect()

    # 翻转

    # 主循环
    # mRunning = True
    pygame.key.set_repeat(500, 5)
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                # If the user clicked on the input_box rect.
                if input_box1.collidepoint(event.pos[0], event.pos[1]):
                    # Toggle the active variable.
                    # active = not active
                    active = not active
                    active2 = not active2 if active2 else active2
                elif input_box2.collidepoint(event.pos[0], event.pos[1]):
                    active2 = not active2
                    active = not active if active else active
                elif button.collidepoint(event.pos[0], event.pos[1]):
                    active3 = not active3
                    button_color = button_color_active
                    if Tcp.CONN:
                        Tcp.CONN.send(bytes(str(len(text+text2)).ljust(20), encoding='utf-8'))
                        Tcp.CONN.send(bytes(text+text2, encoding='utf-8'))
                        print(text, text2)
                        print('click the go')
                else:
                    active, active2, active3 = False, False, False
                # Change the current color of the input box.
                color1 = color_active1 if active else color_inactive1
                color2 = color_active2 if active2 else color_inactive2
                button_color = button_color_active if active3 else button_color_inactive
            if event.type == pygame.KEYDOWN:
                if active:
                    print('input1')
                    if event.key == pygame.K_RETURN:
                        print(text)
                        text = ''
                    elif event.key == pygame.K_BACKSPACE:
                        print('k-back-1')
                        text = text[:-1]
                    else:
                        text += event.unicode
                elif active2:
                    print('input2')
                    if event.key == pygame.K_BACKSPACE:
                        print('k-back-2')
                        text2 = text2[:-1]
                    else:
                        text2 += event.unicode
                global left, right
                if event.key == pygame.K_INSERT:
                    left += 1
                    if Tcp.CONN and 100 > left > 10:
                        Tcp.CONN.send(bytes(str(len('left'+str(left))).ljust(20), encoding='utf-8'))
                        Tcp.CONN.send(bytes('left'+str(left), encoding='utf-8'))
                    print('left up')
                elif event.key == pygame.K_DELETE:
                    left -= 1
                    if Tcp.CONN and 100 > left > 10:
                        Tcp.CONN.send(bytes(str(len('left'+str(left))).ljust(20), encoding='utf-8'))
                        Tcp.CONN.send(bytes('left'+str(left), encoding='utf-8'))
                    print('left down')
                elif event.key == pygame.K_PAGEUP:
                    right += 1
                    if Tcp.CONN and 100 > right > 10:
                        Tcp.CONN.send(bytes(str(len('right'+str(right))).ljust(20), encoding='utf-8'))
                        Tcp.CONN.send(bytes('right'+str(right), encoding='utf-8'))
                    print('right up')
                elif event.key == pygame.K_PAGEDOWN:
                    right -= 1
                    if Tcp.CONN and 100 > right > 10:
                        Tcp.CONN.send(bytes(str(len('right'+str(right))).ljust(20), encoding='utf-8'))
                        Tcp.CONN.send(bytes('right'+str(right), encoding='utf-8'))
                    print('right down')
                elif event.key == pygame.K_SPACE:
                    if Tcp.CONN:
                        Tcp.CONN.send(bytes(str(len('space1')).ljust(20), encoding='utf-8'))
                        Tcp.CONN.send(bytes(str('space1'), encoding='utf-8'))
                    print('SPACE')
                elif event.key == pygame.K_UP:
                    if Tcp.CONN:
                        Tcp.CONN.send(bytes(str(len('upup1')).ljust(20), encoding='utf-8'))
                        Tcp.CONN.send(bytes(str('upup1'), encoding='utf-8'))
                    print('up')
                elif event.key == pygame.K_LEFT:
                    if Tcp.CONN:
                        Tcp.CONN.send(bytes(str(len('turnl1')).ljust(20), encoding='utf-8'))
                        Tcp.CONN.send(bytes(str('turnl1'), encoding='utf-8'))
                elif event.key == pygame.K_RIGHT:
                    if Tcp.CONN:
                        Tcp.CONN.send(bytes(str(len('turnr1')).ljust(20), encoding='utf-8'))
                        Tcp.CONN.send(bytes(str('turnr1'), encoding='utf-8'))
                elif event.key == pygame.K_DOWN:
                    right = 20
                    left = 20
        # pressed_keys = pygame.key.get_pressed()
        #
        # if pressed_keys[K_INSERT]:
        #     left = +2
        #     print('pre left up')
        #     if Tcp.CONN:
        #         Tcp.CONN.send(bytes(str(len(str(left))).ljust(20), encoding='utf-8'))
        #         Tcp.CONN.send(bytes(str(left), encoding='utf-8'))
        # if pressed_keys[K_DELETE]:
        #     right = + 2
        #     if Tcp.CONN:
        #         Tcp.CONN.send(bytes(str(len(str(right))).ljust(20), encoding='utf-8'))
        #         Tcp.CONN.send(bytes(str(right), encoding='utf-8'))
        #     print('pre left down')
        # # 前进、后退
        # if pressed_keys[K_PAGEUP]:
        #     movement_direction = +1.
        #     print('pre right up')
        # if pressed_keys[K_PAGEDOWN]:
        #     movement_direction = -1.
        #     print('pre right down')
        # 时间延迟
        # pygame.time.delay(20)
        # # 覆盖痕迹
        # pygame.draw.rect(screen, THECOLORS['white'], [car_x, car_y, 100, 100], 0)
        # carRect = carRect.move(speed)
        # 小鸟的位置
        # car_x = car_x + car_speed_x
        # car_y = car_y + car_speed_y
        # 左右边缘
        screen.fill(THECOLORS['white'])  # 背景颜色
        screen.blit(background, (0, 0))
        CAR_X, CAR_Y = CarVar.CAR_X, CarVar.CAR_Y
        if CAR_X and CAR_Y:
            screen.blit(background, (0, 0))
            CAR_X = VIR_SIZE_TIMES_OF_REALITY_SIZE * CAR_X - CAR_SIZE[0] / 2  # 减除半个车距
            CAR_Y = -((CAR_Y * VIR_SIZE_TIMES_OF_REALITY_SIZE) - MAP_HEIGHT + CAR_SIZE[1] / 2)
            screen.blit(car, [CAR_X, CAR_Y])
            # print(int(CAR_X), int(CAR_Y), 'pygame')
        # if car_x > 750 or car_x < 0:
        #     car_speed_x = -car_speed_x
        # if car_y > 740 or car_y < 0:
        #     car_speed_y = -car_speed_y
        # pygame.time.delay(20)
        txt_surface = font.render(text, True, color1)
        txt_surface2 = font.render(text2, True, color2)
        button_surface = font.render('GO', True, button_color)



        width1 = max(100, txt_surface.get_width()+10)
        input_box1.w = width1
        width2 = max(100, txt_surface2.get_width()+10)
        input_box2.w = width2
        # width3 = max(100, button_surface.get_width())
        # button.w = width3


        # Blit the text.
        screen.blit(txt_surface, (input_box1.x+5, input_box1.y+5))
        screen.blit(txt_surface2, (input_box2.x+5, input_box2.y+5))
        screen.blit(button_surface, (button.x+5, button.y+5))
        # Blit the input_box rect.
        pygame.draw.rect(screen, color1, input_box1, 2)
        pygame.draw.rect(screen, color2, input_box2, 2)
        pygame.draw.rect(screen, button_color, button, 2)
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
