# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import networkx as nx


class Graph_Matrix:
    """
    Adjacency Matrix
    """

    def __init__(self, vertices=[], matrix=[]):
        """

        :param vertices:a dict with vertex id and index of matrix , such as {vertex:index}
        :param matrix: a matrix
        """
        self.matrix = matrix
        self.edges_dict = {}  # {(tail, head):weight}
        self.edges_array = []  # (tail, head, weight)
        self.vertices = vertices
        self.num_edges = 0

        # if provide adjacency matrix then create the edges list
        if len(matrix) > 0:
            if len(vertices) != len(matrix):
                raise IndexError
            self.edges = self.getAllEdges()
            self.num_edges = len(self.edges)

        # if do not provide a adjacency matrix, but provide the vertices list, build a matrix with 0
        elif len(vertices) > 0:
            self.matrix = [[0 for col in range(len(vertices))] for row in range(len(vertices))]

        self.num_vertices = len(self.matrix)

    def isOutRange(self, x):
        try:
            if x >= self.num_vertices or x <= 0:
                raise IndexError
        except IndexError:
            print("节点下标出界")

    def isEmpty(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices == 0

    def add_vertex(self, key):
        if key not in self.vertices:
            self.vertices[key] = len(self.vertices) + 1

        # add a vertex mean add a row and a column
        # add a column for every row
        for i in range(self.getVerticesNumbers()):
            self.matrix[i].append(0)

        self.num_vertices += 1

        nRow = [0] * self.num_vertices
        self.matrix.append(nRow)

    def getVertex(self, key):
        pass

    def add_edges_from_list(self, edges_list):  # edges_list : [(tail, head, weight),()]
        for i in range(len(edges_list)):
            self.add_edge(edges_list[i][0], edges_list[i][1], edges_list[i][2], )

    def add_edge(self, tail, head, cost=0):
        # if self.vertices.index(tail) >= 0:
        #   self.addVertex(tail)
        if tail not in self.vertices:
            self.add_vertex(tail)
        # if self.vertices.index(head) >= 0:
        #   self.addVertex(head)
        if head not in self.vertices:
            self.add_vertex(head)

        # for directory matrix
        self.matrix[self.vertices.index(tail)][self.vertices.index(head)] = cost
        # for non-directory matrix
        # self.matrix[self.vertices.index(fromV)][self.vertices.index(toV)] = \
        #   self.matrix[self.vertices.index(toV)][self.vertices.index(fromV)] = cost

        self.edges_dict[(tail, head)] = cost
        self.edges_array.append((tail, head, cost))
        self.num_edges = len(self.edges_dict)

    def getEdges(self, V):
        pass

    def getVerticesNumbers(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices

    def getAllVertices(self):
        return self.vertices

    def getAllEdges(self):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if 0 < self.matrix[i][j] < float('inf'):
                    self.edges_dict[self.vertices[i], self.vertices[j]] = self.matrix[i][j]
                    self.edges_array.append([self.vertices[i], self.vertices[j], self.matrix[i][j]])

        return self.edges_array

    def __repr__(self):
        return str(''.join(str(i) for i in self.matrix))

    def to_do_vertex(self, i):
        print('vertex: %s' % (self.vertices[i]))

    def to_do_edge(self, w, k):
        print('edge tail: %s, edge head: %s, weight: %s' % (self.vertices[w], self.vertices[k], str(self.matrix[w][k])))


def create_directed_matrix(my_graph):
    nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    inf = float('inf')
    matrix = [[0, 2, 1, 3, 9, 4, inf, inf],  # a
              [inf, 0, 4, inf, 3, inf, inf, inf],  # b
              [inf, inf, 0, 8, inf, inf, inf, inf],  # c
              [inf, inf, inf, 0, 7, inf, inf, inf],  # d
              [inf, inf, inf, inf, 0, 5, inf, inf],  # e
              [inf, inf, 2, inf, inf, 0, 2, 2],  # f
              [inf, inf, inf, inf, inf, 1, 0, 6],  # g
              [inf, inf, inf, inf, inf, 9, 8, 0]]  # h

    my_graph = Graph_Matrix(nodes, matrix)
    print(my_graph)
    return my_graph


def create_directed_graph_from_edges():  # 有向
    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    edge_list = [('A', 'F', 9), ('A', 'B', 10), ('A', 'G', 15), ('B', 'F', 2),
                 ('G', 'F', 3), ('G', 'E', 12), ('G', 'C', 10), ('C', 'E', 1),
                 ('E', 'D', 7)]

    my_graph = Graph_Matrix(nodes)
    my_graph.add_edges_from_list(edge_list)
    # print(my_graph)
    return my_graph


def draw_undircted_graph(my_graph):  #
    '''
    画无
    :param my_graph:
    :return:
    '''
    G = nx.Graph()  # 建立一个空的无向图G
    for node in my_graph.vertices:
        G.add_node(str(node))
    for edge in my_graph.edges:
        G.add_edge(str(edge[0]), str(edge[1]))

    print("nodes:", G.nodes())  # 输出全部的节点： [1, 2, 3]
    print("edges:", G.edges())  # 输出全部的边：[(2, 3)]
    print("number of edges:", G.number_of_edges())  # 输出边的数量：1
    nx.draw(G, with_labels=True)
    plt.savefig("undirected_graph.png")
    plt.show()



def draw_directed_graph(my_graph):
    '''
    画有向图
    :param my_graph:
    :return:
    '''
    G = nx.DiGraph()  # 建立一个空的有向图G
    # G.
    for node in my_graph.vertices:
        G.add_node(str(node))
    G.add_weighted_edges_from(my_graph.edges_array)

    print("nodes:", G.nodes())  # 输出全部的节点
    print("edges:", G.edges())  # 输出全部的边
    print("number of edges:", G.number_of_edges())  # 输出边的数量
    print(nx.dijkstra_path(G, source='A', target='D'))
    nx.draw(G, with_labels=True)
    plt.savefig("directed_graph.png")
    plt.show()


if __name__ == '__main__':
    # g = Graph_Matrix()
    # created_graph = create_directed_graph_from_edges()
    G1 = nx.Graph()
    G2 = nx.DiGraph()
    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    edge_list = [('A', 'F', 9), ('A', 'B', 10), ('A', 'G', 15), ('B', 'F', 2),
                 ('G', 'F', 3), ('G', 'E', 12), ('G', 'C', 10), ('C', 'E', 1),
                 ('E', 'D', 7)]

    for node in nodes:
        G1.add_node(node)
        G2.add_node(node)
    for edge in edge_list:
        G1.add_weighted_edges_from([edge])
        G2.add_weighted_edges_from([edge])
    print(nx.dijkstra_path(G1, source='A', target='D'))
    print(nx.dijkstra_path(G2, source='A', target='D'))
    # my_graph = Graph_Matrix(nodes)
    # my_graph.add_edges_from_list(edge_list)
    # draw_directed_graph(my_graph)
    # draw_undircted_graph(created_graph)
