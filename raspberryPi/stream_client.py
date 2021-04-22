import socket
import cv2
import numpy
import time
import sys
import threading
import config
import path
from process_img import process_img2
from config import Car


def send_video():
    '''
    单个脚本client测试运行,发送树莓派摄像头视频流到主机
    :return:
    '''
    # 建立sock连接
    # address要连接的服务器IP地址和端口号
    address = ('192.168.3.55', 8004)
    # address = ('192.168.31.226', 8004)
    try:
        # 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
        # socket.AF_INET：服务器之间网络通信
        # socket.SOCK_STREAM：流式socket , for TCP
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 开启连接
        sock.connect(address)
    except socket.error as msg:
        print(msg)
        sys.exit(1)

    # 建立图像读取对象
    capture = cv2.VideoCapture(0)
    # 保存视频流
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # 读取一帧图像，读取成功:ret=1 frame=读取到的一帧图像；读取失败:ret=0
    ret, frame = capture.read()
    # frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 压缩参数，后面cv2.imencode将会用到，对于jpeg来说，15代表图像质量，越高代表图像质量越好为 0-100，默认95
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 15]
    t = threading.Thread(target=recv, args=(sock,))
    t.start()
    while ret:
    #     # 停止0.1S 防止发送过快服务的处理不过来，如果服务端的处理很多，那么应该加大这个值
        time.sleep(0.01)
        # cv2.imencode将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输
        # '.jpg'表示将图片按照jpg格式编码。
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        # 建立矩阵
        data = numpy.array(imgencode)
        # 将numpy矩阵转换成字符形式，以便在网络中传输
        stringData = data.tostring()

        # 先发送要发送的数据的长度
        # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
        sock.send(str.encode(str(len(stringData)).ljust(16)))  # 'length                         '
        # 发送数据
        sock.send(stringData)
        # 读取服务器返回值
        # receive = sock.recv(1024)
        # if len(receive): print(str(receive, encoding='utf-8'))
        # 读取下一帧图片
        ret, frame = capture.read()
    #     # process_img2(frame)
    #     # if cv2.waitKey(10) == 27:
    #     #     break
    #     # threading
    # sock.close()


def recv(conn):
# def recv():
        # address = ('192.168.3.55', 8004)
    # address = ('192.168.31.226', 8004)
    # try:
    #     # 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
    #     # socket.AF_INET：服务器之间网络通信
    #     # socket.SOCK_STREAM：流式socket , for TCP
    #     conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     # 开启连接
    #     conn.connect(address)
    # except socket.error as msg:
    #     print(msg)
    #     sys.exit(1)

    map_data = conn.recv(300)
    if map_data:
        # print(map_data)
        data_list = eval(str(map_data, encoding='utf-8'))
        print('stream_client-recv-data_list', data_list)
        (config.Map.X_LENGTH, config.Map.Y_LENGTH) = data_list[-1]
        del data_list[-1]
        config.Map.NODE_DICT = data_list[0]
        config.Map.EDGE_LIST = data_list[1]
        # print()
        while 1:
            length = recvall(conn, 20)
            if length:
                data = recvall(conn, int(length))
                data = str(data, encoding='utf-8')
                if data.isalpha():
                    config.Map.SOURCE, config.Map.TARGET = data[0], data[1]
                    config.Map.PATH = path.findShortPath(config.Map.NODE_DICT[data[0]], config.Map.NODE_DICT[data[1]])
                elif data[:4] == 'left':
                    config.Car.left_forward_v = int(data[4:])
                elif data[:5] == 'right':
                    config.Car.right_forward_v = int(data[5:])
                elif data == 'space1':
                    config.Car.STOP = True
                elif data == 'upup1':
                    config.Car.STOP = False
                elif data == 'turnl1':
                    config.Map.DYNAMIC_PATH.append('a')
                    config.Car.TURN_CORNER = True
                elif data == 'turnr1':
                    config.Map.DYNAMIC_PATH.append('d')
                    config.Car.TURN_CORNER = True
                else:
                    config.Map.WIM = eval(data)
                print(data)


def recvall(sock, count):
    buf = b''  # buf是一个byte类型
    while count:
        # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


if __name__ == '__main__':
    send_video()
