import networkx as nx
from config import Map
import math

#  构思如下，利用双目得到定位，计算出小车目前在那一条路径上，然后小车前进。


def findShortPath(source, target) -> list:
    '''
    利用networkx作路径规划
    :param source:'A'
    :param target:'F'
    :return:
    '''
    node_dict = Map.NODE_DICT
    Map.G = nx.DiGraph()
    for node in node_dict.values():
        Map.G.add_node(node)
    for i in Map.EDGE_LIST:  # EDGE_LIST: [((700, 0),(750, 700)), (B,C)]
        i_list = list(i)  # i: (A,B)
        i_list.append(math.sqrt(abs((i[0][0] - i[1][0]))**2 + abs((i[0][1] - i[1][1])**2)))
        Map.G.add_weighted_edges_from([tuple(i_list)])   # [(A,B,length)]
    path = nx.dijkstra_path(Map.G, source=source, target=target)
    # F.remove_edges_from([(11,12), (13,14)])
    get_control_way(path)
    return path


def get_control_way(path):
    '''
    获取路线上的左转、右转
    '''
    if Map.DYNAMIC_PATH:  # 重新规划路线
        Map.DYNAMIC_PATH.clear()
    flag = 0
    for i in range(len(path) - 1):
        x, y = path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]
        if flag > 0:  # 有前驱
            Map.DYNAMIC_PATH.append('a')
            Map.DYNAMIC_PATH.append('a')
            flag = 0
            #  左转，然后更新flag
        elif flag < 0:
            Map.DYNAMIC_PATH.append('d')
            Map.DYNAMIC_PATH.append('d')
            flag = 0
            #  右转， 然后更新flag
        else:  # 无前驱
            flag = x
    Map.DYNAMIC_PATH.reverse()
    print('path_get_control_way', Map.DYNAMIC_PATH)

