# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2020/9/26
import numpy as np

num_storage = []


# num_dict.setdefault()


def get_correct_value(values: list, threshold):
    """
    误差消除
    :param values:
    :return:
    """
    # 求众数
    global num_storage
    slope = [(y / x) for x, y in values]
    while len(values) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            values.pop(idx)
        else:

            # return (np.mean())
            print(values[0])
            return values[0]


a = [[110.65892731508201, 136.39285945062494],
     [110.65892731508201, 136.68325384495066],
     [110.61980193041178, 135.24359200007046],
     [110.61980193041178, 135.52911557367233],
     [110.61980193041178, 411.9349770868692],
     [116.96062954594927, 262.2643353660793],
     [117.15551796724448, 136.6647598895376],
     [116.96062954594927, 98.5323404169034],
     [110.65892731508201, 168.64697046051145],
     [110.61980193041178, 95.03136530214799],
     [110.65892731508201, 135.52901268770765],
     [110.65892731508201, 414.59245333356085],
     ]

if __name__ == '__main__':
    get_correct_value(a, threshold=0.2)
