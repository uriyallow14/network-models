import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
import networkx


""" General Functions """


def build_graph(filename: str) -> networkx.Graph:
    """
    The function inputs a string – the name of the data set, e.g., "PartA1.csv"
    and returns a networx.Graph object will represent the nodes, the edges, and weights (when they exist)
    as they are at the data set.

    :param filename: the name of the file containing the data
    :return: a networx.Graph object
    """
    Graphtype = networkx.Graph()
    df = pd.read_csv(filename)
    if len(df.columns) > 2:
        weights_col = 'w'
    else:
        weights_col = None
    graph = networkx.from_pandas_edgelist(df, source='from', target='to', edge_attr=weights_col, create_using=Graphtype)
    return graph


""" Part A Functions """


def clustering_coefficient(graph: networkx.Graph) -> float:
    num_of_triangles = count_triangles(graph)
    num_of_all_triplets = 0.0
    for node in graph.nodes:
        neighbors_count = len(list(graph.adj[node]))
        num_of_all_triplets += (neighbors_count * (neighbors_count - 1) / 2)
    cc = num_of_triangles / num_of_all_triplets
    return cc


""" Part A Auxiliary Functions """


def calc_degree_histogram(graph: networkx.Graph) -> Dict:
    """
    The function inputs a networkx.Graph object and returns a dictionary representation of the nodes' degree histogram,
     i.e, if histogram[1] = 10 then there are 10 nodes whose degree = 1 (10 nodes have only 1 friend).

    :param graph: a networkx.Graph object
    :return: dictionary representation of the nodes' degree histogram
    """
    histogram = {}
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    for node in range(len(degree_sequence)):
        node_degree = degree_sequence[node]
        if node_degree in histogram:
            histogram[node_degree] += 1
        else:
            histogram[node_degree] = 1
    return histogram


def plot_degree_histogram(histogram: Dict, f: str):
    """
    The function inputs a dictionary representation of the degree histogram object and plots it.

    :param histogram: dictionary representation of the degree
    :param f: the name of the file containing the data
    """
    deg, cnt = zip(*histogram.items())
    plt.bar(deg, cnt, color="g")
    plt.title("Degree Histogram - " + f)
    plt.ylabel("Count")
    plt.xlabel("Degree")
    plt.show()


def count_triangles(graph: networkx.Graph) -> float:
    count_t = 0
    for edge_between_v_u in graph.edges:
        node_v = edge_between_v_u[0]
        node_u = edge_between_v_u[1]
        for node in graph.adj[node_v]:
            if graph.has_edge(node, node_u):
                count_t += 1
    return count_t


def is_not_triplet(my_list: List) -> bool:
    """
    :param my_list: list containing the neighbors of a given vertex
    :return: True if number of neighbors is bigger than 1, else False
    """
    if len(my_list) > 1:
        return False
    else:
        return True


def is_nodeA_equals_nodeC(vertex_a: str, vertex_b: str) -> bool:
    if vertex_a == vertex_b:
        return True
    else:
        return False


def is_closed_triplet(vertex_c_list: List, vertex_a: str) -> bool:
    if vertex_a in vertex_c_list:
        return True
    else:
        return False

