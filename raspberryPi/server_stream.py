import socket
import time
import cv2
import numpy
from pc.process_image import process1


conn = None


def ReceiveVideo():
    # IP地址'0.0.0.0'为等待客户端连接
    address = ('0.0.0.0', 8003)
    # 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
    # socket.AF_INET：服务器之间网络通信
    # socket.SOCK_STREAM：流式socket , for TCP
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 将套接字绑定到地址, 在AF_INET下,以元组（host,port）的形式表示地址.
    s.bind(address)
    # 开始监听TCP传入连接。参数指定在拒绝连接之前，操作系统可以挂起的最大连接数量。该值至少为1，大部分应用程序设为5就可以了。
    s.listen(5)

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

    def get_accept():
        global conn
        conn, addr = s.accept()
        print('connect from:' + str(addr))
    # 接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，可以用来接收和发送数据。addr是连接客户端的地址。
    # 没有连接则等待有连接
    get_accept()
    try:
        while 1:
            # start = time.time()  # 用于计算帧率信息
            length = recvall(conn, 16)  # 获得图片文件的长度,16代表获取长度
            if length:
                stringData = recvall(conn, int(length))  # 根据获得的文件长度，获取图片文件
                data = numpy.frombuffer(stringData, numpy.uint8)  # 将获取到的字符流数据转换成1维数组
                # decimg = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将数组解码成图像
                decimg = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)  # 将数组解码成图像
                # print(decimg.shape)
                cv2.startWindowThread()
                cv2.imshow("decimg", decimg)
                pro1 = process1.process_img(decimg)
                # cv2.imshow("process2", process1.process_img2(decimg))
                cv2.imshow('process1', pro1)  # 显示图像
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                cv2.destroyAllWindows()
                while 1:
                    get_accept()
                    if conn:
                        break

    finally:
        s.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ReceiveVideo()
