class Car:
    CAR_X = 0  # 小车的X轴位置
    CAR_Y = 0  # 小车的Y轴位置
    default_left_forward_v = 20  # 默认左轮初始速度
    left_forward_v = 15  # 小车左轮前进初始速度
    left_back_v = 0  # 小车左轮后退速度
    default_right_forward_v = 20  # 右轮前进速度
    right_forward_v = 15  # 右轮前进速度
    right_back_v = 0  # 右轮后退速度
    THRESHOLD_DISTANCE_OF_OBSTACLE = 20  # 与障碍物的刹车距离
    TURN_AROUND = False  # 是否掉头
    TURN_CORNER = False  # 是否转弯
    STOP = False  # 是否停止
    NEW = 1


class Map:
    G = None  # 终点
    X_LENGTH = 0  #
    Y_LENGTH = 0  #
    WIM = None  # 小车的位置
    PATH = None  # 行驶路径
    NODE_DICT = None  # 节点
    SOURCE = None  # 起点
    TARGET = None  # 终点
    FLAG = False  # 是否接收到行车指令
    EDGE_LIST = None  # 边list
    DYNAMIC_PATH = []  # 动态规划list


class Road:
    LINE_CLUSTER = {}  # 线的簇数
    IS_STRAIGHT = False  # 是否为直线
    IS_CROSS = False  # 是否为交叉路口
    DISTANCE = 0  # 距离


# 偏移阈值
DeviateThreshold = 20
