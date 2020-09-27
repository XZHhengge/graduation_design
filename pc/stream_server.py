# -*- coding:utf-8 -*-
__author__ = 'zhengwang'

import numpy as np
import cv2
import socket


class VideoStreamingTest(object):
    def __init__(self, host, port):

        self.server_socket = socket.socket()
        self.server_socket.bind((host, port))
        self.server_socket.listen(0)
        self.connection, self.client_address = self.server_socket.accept()
        self.connection = self.connection.makefile('rb')
        self.host_name = socket.gethostname()
        self.host_ip = socket.gethostbyname(self.host_name)
        self.streaming()

    def streaming(self):

        try:
            print("Host: ", self.host_name + ' ' + self.host_ip)
            print("Connection from: ", self.client_address)
            print("Streaming...")
            print("Press 'q' to exit")

            # need bytes here
            stream_bytes = b' '
            while True:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    new_screen = process_img(image)
                    cv2.imshow('window', new_screen)
                    cv2.imshow('image', image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.connection.close()
            self.server_socket.close()


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 3)
    except:
        pass


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)  # 灰度化
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)  # 边缘特征
    processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)  # 高斯模糊
    vertices = np.array([[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]], np.int32)
    processed_img = roi(processed_img, [vertices])  # 不规则ROI区域截取

    #                       edges
    lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 180, 20, 15)  # 霍夫直线检测
    draw_lines(processed_img, lines)  # 划线
    return processed_img


if __name__ == '__main__':
    # host, port
    # h, p = "192.168.43.76", 8001
    h, p = "192.168.3.55", 8001
    VideoStreamingTest(h, p)
