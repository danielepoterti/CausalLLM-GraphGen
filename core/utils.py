import matplotlib.pyplot as plt
import networkx as nx
import os


def draw_graph(graph, title="", output_dir=".", graph_type=""):
    """Draws the given graph.

    Args:
        graph: The graph to draw.
        title: Title of the graph, default is an empty string.
        output_dir: Directory to save the graph, default is current directory.
        graph_type: Type of the graph, default is an empty string.
    """
    plt.figure(figsize=(25,25))
    pos = nx.spring_layout(graph)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(graph, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(graph, pos, width=2)

    # labels
    nx.draw_networkx_labels(graph, pos, font_size=20, font_family="sans-serif")

    plt.title(title)
    
    # ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # save the figure to a PNG file in the output directory
    plt.savefig(os.path.join(output_dir, f"{title}_{graph_type}.png"))
    plt.close()


def draw_all_graphs(result, output_dir=".", graph_type=""):
    """Draws all the graphs in the result.

    Args:
        result: Dictionary of method and graph pairs.
        output_dir: Directory to save the graphs, default is current directory.
        graph_type: Type of the graphs, default is an empty string.
    """
    for method, graph in result.items():
        if isinstance(graph, nx.Graph):
            draw_graph(graph, method, output_dir, graph_type)
        else:
            print(f"Error drawing graph for method {method}: {graph}")
