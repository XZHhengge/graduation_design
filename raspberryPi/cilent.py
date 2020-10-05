# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2020/9/27
import socket
import sys

# 创建 socket 对象
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 获取本地主机名
host = '192.168.3.179'

# 设置端口号
port = 9999

# 连接服务，指定主机和端口
s.connect((host, port))

# 接收小于 1024 字节的数据
try:
    while 1:
        x = input("input")
        s.send(bytes(x, encoding='utf-8'))
finally:
    s.close()
