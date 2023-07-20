import matplotlib.pyplot as plt
import networkx as nx
import os

def draw_graph(G, title="", output_dir=".", type = ""):
    pos = nx.spring_layout(G,)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, width=2)

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

    plt.title(title)
    
    # ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # save the figure to a PNG file in the output directory
    plt.savefig(os.path.join(output_dir, f"{title}_{type}.png"))
    plt.close()

def draw_all_graphs(result, output_dir=".", type = ""):
    for method, G in result.items():
        if isinstance(G, nx.Graph):
            draw_graph(G, method, output_dir, type)
        else:
            print(f"Error drawing graph for method {method}: {G}")
