from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
from contextlib import contextmanager

import networkx as nx
import sys
import os
import sys, os


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def build_causal_skeleton(df, method='all'):
    methods = {
        'pc': build_PC,
        'fci': build_FCI,
        'ges': build_GES,
        'lingam': build_LiNGAM,
    }
    
    try:
        if method == 'all':
            result = {name: func(df) for name, func in methods.items()}
        elif method in methods:
            result = {method: methods[method](df)}
        else:
            raise ValueError(f'Invalid method. Available methods are: {list(methods.keys())}, all')
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
    return result

def build_PC(df):
    try:
        data_array = df.values
        CG = pc(data_array, node_names=df.columns.tolist(), show_progress=False)
        G = CG.G
        undirected_graph = nx.Graph()
        for edge in G.get_graph_edges():
            u = edge.get_node1().get_name()
            v = edge.get_node2().get_name()
            undirected_graph.add_edge(u, v)
    except Exception as e:
        return f"An error occurred in build_PC: {str(e)}"
    return undirected_graph

def build_FCI(df):
    try:
        data_array = df.values
        with suppress_stdout():
            G, _ = fci(data_array, verbose = False, show_progress = False)
        undirected_graph = nx.Graph()
        for edge in G.get_graph_edges():
            u = edge.get_node1().get_name()
            v = edge.get_node2().get_name()
            undirected_graph.add_edge(u, v)
        mapping = {"X{}".format(i+1): name for i, name in enumerate(df.columns.tolist())}
        undirected_graph = nx.relabel_nodes(undirected_graph, mapping)
    except Exception as e:
        return f"An error occurred in build_FCI: {str(e)}"
    return undirected_graph

def build_GES(df):
    try:
        data_array = df.values
        with suppress_stdout():
            Record = ges(data_array)
        G = Record['G']
        undirected_graph = nx.Graph()
        for edge in G.get_graph_edges():
            u = edge.get_node1().get_name()
            v = edge.get_node2().get_name()
            undirected_graph.add_edge(u, v)
        mapping = {"X{}".format(i+1): name for i, name in enumerate(df.columns.tolist())}
        undirected_graph = nx.relabel_nodes(undirected_graph, mapping)
    except Exception as e:
        return f"An error occurred in build_GES: {str(e)}"
    return undirected_graph

def build_LiNGAM(df):
    try:
        data_array = df.values
        model = lingam.ICALiNGAM(42, 700)
        model.fit(data_array)
        undirected_graph = nx.convert_matrix.from_numpy_array(model.adjacency_matrix_)
        mapping = {i: name for i, name in enumerate(df.columns.tolist())}
        undirected_graph = nx.relabel_nodes(undirected_graph, mapping)
    except Exception as e:
        return f"An error occurred in build_LiNGAM: {str(e)}"
    return undirected_graph
