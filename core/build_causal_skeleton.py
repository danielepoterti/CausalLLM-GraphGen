from contextlib import contextmanager

import networkx as nx
import os
import sys

from causallearn.search.ConstraintBased import FCI, PC
from causallearn.search.FCMBased import lingam
from causallearn.search.ScoreBased import GES


@contextmanager
def suppress_stdout():
    """Suppresses the standard output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def build_causal_skeleton(df, method='all'):
    """Builds causal skeleton using given method.

    Args:
        df: DataFrame to build the skeleton on.
        method: The method used to build the skeleton, default is 'all'.

    Returns:
        The result of the skeleton building.
    """
    methods = {
        'pc': build_pc,
        'fci': build_fci,
        'ges': build_ges,
        'lingam': build_lingam,
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


def build_pc(df):
    """Builds PC causal skeleton.

    Args:
        df: DataFrame to build the PC skeleton on.

    Returns:
        The result of the PC skeleton building.
    """
    try:
        data_array = df.values
        cg = PC.pc(data_array, node_names=df.columns.tolist(), show_progress=False)
        g = cg.G
        undirected_graph = nx.Graph()
        for edge in g.get_graph_edges():
            u = edge.get_node1().get_name()
            v = edge.get_node2().get_name()
            undirected_graph.add_edge(u, v)
    except Exception as e:
        return f"An error occurred in build_pc: {str(e)}"
    
    return undirected_graph


def build_fci(df):
    """Builds FCI causal skeleton.

    Args:
        df: DataFrame to build the FCI skeleton on.

    Returns:
        The result of the FCI skeleton building.
    """
    try:
        data_array = df.values
        with suppress_stdout():
            g, _ = FCI.fci(data_array, verbose=False, show_progress=False)
        undirected_graph = nx.Graph()
        for edge in g.get_graph_edges():
            u = edge.get_node1().get_name()
            v = edge.get_node2().get_name()
            undirected_graph.add_edge(u, v)
        mapping = {f"X{i+1}": name for i, name in enumerate(df.columns.tolist())}
        undirected_graph = nx.relabel_nodes(undirected_graph, mapping)
    except Exception as e:
        return f"An error occurred in build_fci: {str(e)}"
    
    return undirected_graph


def build_ges(df):
    """Builds GES causal skeleton.

    Args:
        df: DataFrame to build the GES skeleton on.

    Returns:
        The result of the GES skeleton building.
    """
    try:
        data_array = df.values
        with suppress_stdout():
            record = GES.ges(data_array)
        g = record['G']
        undirected_graph = nx.Graph()
        for edge in g.get_graph_edges():
            u = edge.get_node1().get_name()
            v = edge.get_node2().get_name()
            undirected_graph.add_edge(u, v)
        mapping = {f"X{i+1}": name for i, name in enumerate(df.columns.tolist())}
        undirected_graph = nx.relabel_nodes(undirected_graph, mapping)
    except Exception as e:
        return f"An error occurred in build_ges: {str(e)}"
    
    return undirected_graph


def build_lingam(df):
    """Builds LiNGAM causal skeleton.

    Args:
        df: DataFrame to build the LiNGAM skeleton on.

    Returns:
        The result of the LiNGAM skeleton building.
    """
    try:
        data_array = df.values
        model = lingam.ICALiNGAM(42, 700)
        model.fit(data_array)
        undirected_graph = nx.convert_matrix.from_numpy_array(model.adjacency_matrix_)
        mapping = {i: name for i, name in enumerate(df.columns.tolist())}
        undirected_graph = nx.relabel_nodes(undirected_graph, mapping)
    except Exception as e:
        return f"An error occurred in build_lingam: {str(e)}"
    
    return undirected_graph
