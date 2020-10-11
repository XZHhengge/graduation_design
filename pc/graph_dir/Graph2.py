# -*- coding:utf-8 -*-

import networkx as nx
from config import NODE_LIST, EDGE_LIST
from pc.camera_dir.color_track import get_2point_distance


def findShortPath(source, target) -> list:
    G = nx.DiGraph()
    nodes = NODE_LIST
    for node in nodes:
        G.add_node(node)
    for i in EDGE_LIST:
        i_list = list(i)
        i_list.append(get_2point_distance(i[0], i[1], shape=2))
        G.add_weighted_edges_from([tuple(i_list)])

    return nx.dijkstra_path(G, source=source, target=target)


if __name__ == '__main__':
    print(findShortPath(NODE_LIST[0], NODE_LIST[-1]))

