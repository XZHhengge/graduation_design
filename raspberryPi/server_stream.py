from time import sleep
from zlib import decompress
from threading import Thread
import socket
import cv2
# BUFFER_SIZE = 6
data = []


def show_image(image_bytes, image_size):

    global im
    # try:
    cv2.imshow("frame", image_bytes)


def recv_image():
    global receiving, im
    s = socket.socket()
    s.bind(('192.168.3.55', 8001))
    s.listen(1)

    while receiving:
        while receiving:
            chunk, _ = s.recv(640*480)

            if chunk == b'start':
                break
            elif chunk == b'close':
                sleep(0.1)
        else:
            break
        while receiving:
            chunk, _ = s.recv(480*640)
            if chunk.startwith(b'_over'):
                image_size = eval(chunk[5:])
                try:
                    image_data = decompress(b''.join(data))
                except:
                    data.clear()
                    break
                global thread_show
                thread_show = Thread(target=show_image, args=(image_data, image_size))
                thread_show.daemon =True
                thread_show.start()
                data.clear()
                break


if __name__ == '__main__':

    receiving = True
    recv_image()
    # thread_sender = Thread(target=recv_image)
    # thread_sender.daemon = True
    # thread_sender.start()




