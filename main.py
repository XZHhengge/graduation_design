# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2020/10/8
# import globalVar
import threading
from pc.pygame_dir import pygame_display
from pc.camera_dir import color_track

if __name__ == '__main__':
    t2 = threading.Thread(target=color_track.main)
    t2.start()
    t1 = threading.Thread(target=pygame_display.creat)
    t1.start()

