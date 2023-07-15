import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(G, title=""):
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, width=2)

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

    plt.title(title)
    plt.show()

def draw_all_graphs(result):
    for method, G in result.items():
        if isinstance(G, nx.Graph):
            draw_graph(G, method)
        else:
            print(f"Error drawing graph for method {method}: {G}")
