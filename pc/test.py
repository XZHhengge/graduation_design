# -*- coding:utf-8 -*-
# Author: cmzz
# @Time :2020/9/26

num_storage = []


# num_dict.setdefault()


def get_correct_value(value):
    """
    误差消除
    :param value:
    :return:
    """
    # 求众数
    global num_storage
    if len(num_storage) == 23:
        num_storage.sort()  # 升序
        while len(num_storage) > 3:
            del num_storage[0], num_storage[-1]  # 去掉一个最大值和最小值
            mean = np.mean(num_storage)  # 平均数
            center_num = num_storage[int(len(num_storage) / 2)]  # 中位数
            if abs(mean - center_num) > (mean * 0.1) and abs(mean - center_num) > (center_num * 0.1):  # 误差大于10%
                continue
            else:
                num_storage.clear()
                print("最终值为", (mean + center_num) / 2.0)
        # print(mean, nu)
        print("误差计算出错")

    else:
        # if isinstance(value, tuple):
        #     num_storage.append(value[0] + value[1])
        # else:
        num_storage.append(value)

a = [110.65892731508201,136.39285945062494,
110.65892731508201,136.68325384495066,
110.61980193041178,135.24359200007046,
110.61980193041178,135.52911557367233,
110.61980193041178,135.24359200007046,
116.96062954594927,262.2643353660793,
117.15551796724448,136.6647598895376,
116.96062954594927,129.49451005983894,
110.65892731508201,135.52901268770765,
110.61980193041178,136.68335586208082,
110.65892731508201,135.52901268770765,
110.65892731508201,135.52901268770765,
     ]
for i in range(0, 21, 2):
    # print(i)
    get_correct_value(a[i])