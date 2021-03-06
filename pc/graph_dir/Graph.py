# -*- coding:utf-8 -*-

import networkx as nx
from config import NODE_DICT, EDGE_LIST
from pc.camera_dir.color_track import get_2point_distance


def findShortPath(source, target) -> list:
    '''
    利用networkx作路径规划
    :param source:'A'
    :param target:'F'
    :return:
    '''
    G = nx.DiGraph()
    for node in NODE_DICT.values():
        G.add_node(node)
    for i in EDGE_LIST:
        i_list = list(i)
        i_list.append(get_2point_distance(i[0], i[1], shape=2))
        G.add_weighted_edges_from([tuple(i_list)])

    return nx.dijkstra_path(G, source=NODE_DICT[source], target=NODE_DICT[target])


if __name__ == '__main__':
    print(findShortPath('A', 'F'))

