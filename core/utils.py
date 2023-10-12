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
    pos = nx.spring_layout(graph, seed=42)  # positions for all nodes

    plt.figure(figsize=(8,6))

    nx.draw(graph, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_color='black', edgecolors='black', linewidths=1, width=2, edge_color='gray', connectionstyle='arc3,rad=0.1')
    
    plt.title('Causal Graph', fontdict={'size': 16})
    
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
