# -*- coding:utf-8 -*-
"""
resource file https://mp.weixin.qq.com/s?__biz=MzUwOTg3NTQ4NQ==&mid=2247484940&idx=1&sn=92755969d5078e4ba26cc43bc042df0f&chksm=f90ac27ece7d4b685db5cd952ebd2c14db34620fb1827f612acc0403bdbef0a51ff69135a306&mpshare=1&scene=1&srcid=09288kapS03WXvVdmjl2nHus&sharer_sharetime=1601280126866&sharer_shareid=600806cd2c5c20d2c44b768d4d01168f&exportkey=AYjwEbv5LupbF%2F8Twr9ZilA%3D&pass_ticket=E%2FeWLM%2FIyWJ1mKjHyVYbknn2ay6rwc36VdA3TVGauKi42FwIXe1tAZ2rZB9hVF%2BL&wx_header=0#rd

A Python Class
A simple Python graph class, demonstrating the essential
facts and functionalities of graphs.
example:

graph = { "a" : ["c"],
          "b" : ["c", "e"],
          "c" : ["a", "b", "d", "e"],
          "d" : ["c"],
          "e" : ["c", "b"],
          "f" : []
        }
"""


""" A Python Class
A simple Python graph class, demonstrating the essential 
facts and functionalities of graphs.
"""


MAXL=99999


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

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!
        """
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1].append(vertex2)
        else:
            self.__graph_dict[vertex1] = [vertex2]

    def __generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one (a loop back to the vertex) or two
            vertices
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbour in self.__graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
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
        if path is None:
            path = []
        graph = self.__graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex not in graph:
            return None
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extend_path = self.find_path(vertex, end_vertex, path)
                if extend_path:
                    return extend_path
        return None

    def find_all_paths(self, start_vertex, end_vertex, path=[]):
        """ find all paths from start_vertex to end_vertex in graph
        查找从开始顶点(Start Vertex)到结束顶点(End Vertex)的所有简单路径(Simple Path)的算法。
        """
        graph = self.__graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex not in graph:
            return []
        paths = []
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_paths = self.find_all_paths(vertex, end_vertex, path)
                for p in extended_paths:
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
                if i not in closed:
                    opened.append(i)
                    pred[i] = u
                    closed.add(i)


if __name__ == '__main__':
    g = {"a": ["b", "c", "e"],
         "b": ["a", "d"],
         "c": ["a", "f"],
         "d": ["g", "e"],
         "e": ["f", "h", "d", "a"],
         "f": ["i", "c", "e"],
         "g": ["h", "d"],
         "h": ["g", "e", "i"],
         "i": ["h", "f"]
         }

    graph = Graph(g)

    print('The path from vertex "a" to vertex "b":')
    path = graph.findShortestPath("a", "h")
    print(path)

    print('All paths from vertex "a" to vertex "b":')
    path = graph.find_all_paths("a", "h")
    print(path)
