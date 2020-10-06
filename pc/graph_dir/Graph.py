# -*- coding:utf-8 -*-
import numpy as np
import sys

"""
resource file https://mp.weixin.qq.com/s?__biz=MzUwOTg3NTQ4NQ==&mid=2247484940&idx=1&sn=92755969d5078e4ba26cc43bc042df0f&chksm=f90ac27ece7d4b685db5cd952ebd2c14db34620fb1827f612acc0403bdbef0a51ff69135a306&mpshare=1&scene=1&srcid=09288kapS03WXvVdmjl2nHus&sharer_sharetime=1601280126866&sharer_shareid=600806cd2c5c20d2c44b768d4d01168f&exportkey=AYjwEbv5LupbF%2F8Twr9ZilA%3D&pass_ticket=E%2FeWLM%2FIyWJ1mKjHyVYbknn2ay6rwc36VdA3TVGauKi42FwIXe1tAZ2rZB9hVF%2BL&wx_header=0#rd

A Python Class
A simple Python graph class, demonstrating the essential
facts and functionalities of graphs.
example:

graph = { "a" : [{"c": 2}],
          "b" : [{"c": 4}, {"e": 5}],
          "c" : [{"a": 2}, {"b": 5}, {"d": 6}, {"e": 4}],
          "d" : [{"c": 6}],
          "e" : [{"c": 4}, {"b": 6}],
          "f" : []
        }
"""

""" A Python Class
A simple Python graph class, demonstrating the essential 
facts and functionalities of graphs.
"""

MAXL = 99999


class Graph:
    def __init__(self, graph_dict=None):
        """ initializes a graph object
            If no dictionary or None is given,
            an empty dictionary will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.__graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = []

    def add_edge(self, edge, length):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!
        """
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1].append({vertex2: length})
        else:
            self.__graph_dict[vertex1] = [{vertex2: length}]

    def __generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one (a loop back to the vertex) or two
            vertices
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbour in self.__graph_dict[vertex]:
                for key, value in neighbour.items():
                    if {key, vertex} not in edges:
                        edges.append({vertex, key, value})
        return edges

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res

        """邻接节点(Adjacent Vertices)：如果两个Vertices存在一条连接Edge，则称它们是相邻接的。
        
        无向图中的Path: 无向图中的Path是一个点序列，序列中相邻的节点都是相邻接的。
        
        简单路径(Simple Path)：没有重复节点的Path称为Simple Path。
        """

    def find_path(self, start_vertex, end_vertex, path=None):
        """
        实现查找一条从开始顶点(Start Vertex)到结束顶点(End Vertex)的简单路径(Simple Path) 的算法。
        find a path from start_vertex to end_vertex in graph
        :param start_vertex:
        :param end_vertex:
        :return:
        """
        # path = [length, a, b, c, d]
        if path is None:
            path = [0]
        # if
        graph = self.__graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex not in graph:
            return None
        for vertexs in graph[start_vertex]:
            for vertex, length in vertexs.items():
                if vertex not in path:
                    extend_path = self.find_path(vertex, end_vertex, path)
                    if extend_path:
                        # print('x')
                        extend_path[0] += length
                        return extend_path
                # else:
                #     path[0] += length
        return None

    def find_all_paths(self, start_vertex, end_vertex, path=[]):
        """ find all paths from start_vertex to end_vertex in graph
        查找从开始顶点(Start Vertex)到结束顶点(End Vertex)的所有简单路径(Simple Path)的算法。
        """
        graph = self.__graph_dict
        if not path:
            path = [0]
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex not in graph:
            return []
        paths = []
        for vertexs in graph[start_vertex]:
            for vertex, length in vertexs.items():
                if vertex not in path:
                    extended_paths = self.find_all_paths(vertex, end_vertex, path)
                    for p in extended_paths:
                        p[0] += length
                        paths.append(p)
        return paths

    def extractPath(self, u, pred):
        path = []
        k = u
        path.append(k)
        while k in pred:
            path.append(pred[k])
            k = pred[k]

        path.reverse()

        return path

    def findShortestPath(self, start, end, path=[]):
        # Mark all the vertices as not visited
        closed = set()

        # Create a queue for BFS
        opened = []
        pred = {}

        # Mark the source node as visited and enqueue it
        opened.append(start)
        closed.add(start)

        while opened:
            u = opened.pop(0)
            if u == end:
                path = self.extractPath(u, pred)
                return path

            for i in self.__graph_dict[u]:
                for j, length in i.items():
                    if j not in closed:
                        opened.append(i)
                        pred[j] = u
                        closed.add(j)


def dijkstra(start, graph_struct, node):
    """
    function:dijkstra
    args:
        start 要计算的起始点
        graph_struct 带权有向图的结构
        node 图中节点个数
    return:
        dis 元素为-1时，表示没有路径。其余为距离
    """
    # n表示有N个点，m表示M条边，x表示求那个点到所有点的最短路径
    n, m, x = node, len(graph_struct), start
    max_int = sys.maxsize
    # max_int = 0
    weight = np.full([n + 1, n + 1], -1)
    dis = np.full(n + 1, max_int)
    # 初始化权重数组，自己到自己为0.其他为暂为-1
    for i in range(1, n + 1):
        weight[i][i] = 0

    for i in graph_struct:
        # 所有存在边的位置填上权重，没有关系的位置保持为-1，表示不可直接到达
        weight[i[0]][i[1]] = i[2]
        # 如果是与我们要求的x点相关的点，则也将x到i的权重填入dis列表中
        if i[0] == x:
            dis[i[1]] = i[2]

    # 程序走到这里，我们就有了权重数组 以及 dis数组（x到各个点的距离，如果没有边，则为max_int）
    # dis ： [max_int,  0,  max_int,  10,  max_int,  30,  100]， dis[0]不纳入计算，为了方便，我们只考虑index>=1的部分

    # 定义内部search函数，开始计算x到所有点的最短路径，最终更新到dis中
    def search(x, dis, weight, n):
        """
        function:search
        args:
            x 要求的点
            dis 距离数组
            weight 权重数组
            n 节点的个数
        return:dis
        """
        mark = np.full(n + 1, False)  # 创建一个mark数组，元素个数为n+1，[1,n]表示1-n点是否被当成最小值加过，已经加过为True，未被加过为False
        mark[x] = True  # 要求的点x，直接标记为加过
        dis[x] = 0  # 自己到自己的距离为0
        count = 1  # 当前已经加了几个点，当前只有x点，所以初始化为1

        # 开始循环，当count<=n时，说明还有点未被加过
        while count <= n:
            locate = 0  # locate记录计算出来马上要被加的点
            min = max_int  # 用于求最小值时比较用
            # 找到dis里面，还没有加过的位置(mark[idx]=False)里面数的最小值对应的index。
            # dis : [9223372036854775807 0 9223372036854775807 10 9223372036854775807 30 100]
            # mark : [False,True,False,False,False,False]，从中找出10的index为 3
            # 该for循环完毕后，min中的值就是最小值10
            for i in range(1, n + 1):
                if not mark[i] and dis[i] < min:
                    min = dis[i]
                    locate = i
            # 如果locate为0，则说明所有点都被加完了，直接退出循环
            if locate == 0: break
            # 如果locate不为0，说明找到了需要加的点，先对其进行标记
            mark[locate] = True
            # 加一个点count增1
            count += 1

            # 从我们找到的需要加的点locate（例如3）开始，看weight数组中他到各个点的距离
            for i in range(1, n + 1):
                # 如果某个点已经被加过了，我们就不计算locate到这个点的距离了
                # 如果locate到某个点的距离为-1，说明没有路，也不计算
                # 条件3：x到locate的距离（dis[locate]） + locate到点i的距离(weight[locate][i]) < x到i的距离 才能更新
                if not mark[i] and weight[locate][i] != -1 and (
                        dis[locate] + weight[locate][i] < dis[i]):
                    # 条件都满足，则计算，并更新dis中x-->i的距离
                    dis[i] = dis[locate] + weight[locate][i]

        return dis

    # 调用search开始计算x到各个点的距离，记录到dis数组中
    dis = search(x, dis, weight, n)

    # 打印dis数组
    for i in range(1, len(dis)):
        if dis[i] == max_int:
            dis[i] = -1
        print("%s点到%s点 %s" % (x, i, "的最短路径为%s" % dis[i] if dis[i] != max_int else '没有路'))

    # 返回
    return dis


if __name__ == '__main__':
    # 列举所有的边的权重，并写入weight列表
    weight_init = [(1, 3, 10), (1, 5, 30), (1, 6, 100), (2, 3, 5), (3, 4, 50), (4, 6, 10), (5, 6, 60), (5, 4, 20)]
    dis = dijkstra(1, weight_init, 6)


if __name__ == '__main__':
    # g = {"a": ["b", "c", "e"],
    #      "b": ["a", "d"],
    #      "c": ["a", "f"],
    #      "d": ["g", "e"],
    #      "e": ["f", "h", "d", "a"],
    #      "f": ["i", "c", "e"],
    #      "g": ["h", "d"],
    #      "h": ["g", "e", "i"],
    #      "i": ["h", "f"]
    #      }
    g = {"a": [{"c": 2}],
         "b": [{"c": 4}, {"e": 5}],
         "c": [{"a": 2}, {"b": 4}, {"d": 6}, {"e": 4}],
         "d": [{"c": 6}, {"f": 1}],
         "e": [{"c": 4}, {"b": 5}, {"f": 2}],
         "f": [{"d": 1}, {"e": 2}]
         }
    graph = Graph(g)
    print(graph.findShortestPath('a', 'f'))
    # graph.add_edge({"a", "z"}, 10)
    # print(graph.edges())
    # print(graph.edges())
    # print(graph.vertices())
    # print('The path from vertex "a" to vertex "b":')
    # path = graph.findShortestPath("a", "h")
    # print(path)
    #
    # print('All paths from vertex "a" to vertex "b":')
    # path = graph.find_all_paths("a", "h")
    # print(path)
