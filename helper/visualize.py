import networkx as nx
import matplotlib.pyplot as plt

# Function to build graph from the model
def build_graph(model):
    G = nx.DiGraph()

    previous_nodes = []

    for i in range(model.input.shape[1]):
        input_layer = model.input.name + "_" + str(i)
        G.add_node(input_layer, layer=0, type='input')

        previous_nodes.append(input_layer)

    for layer_num, layer in enumerate(model.layers, start=1):
        layer_name = layer.name
        current_nodes = [f"{layer_name}_{i}" for i in range(layer.units)]

        for prev_node in previous_nodes:
            for current_node in current_nodes:
                G.add_edge(prev_node, current_node)

        for i, node in enumerate(current_nodes):
            G.add_node(node, layer=layer_num, type='neuron', position=i)

        previous_nodes = current_nodes

    return G

# Build the graph

def visualize_graph(model):

    G = build_graph(model)

    plt.figure(figsize=(12, 6))

    pos = {}
    layer_nodes = {}

    # Assign positions to nodes for better visualization
    for node, data in G.nodes(data=True):
        layer = data['layer']
        if layer not in layer_nodes:
            layer_nodes[layer] = []
        layer_nodes[layer].append(node)

    for layer, nodes in layer_nodes.items():
        x = layer
        y_start = -(len(nodes) - 1) / 2
        for i, node in enumerate(nodes):
            pos[node] = (x, y_start + i)

    # Draw nodes with colors based on their type
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightgreen')

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='->', arrowsize=20)

    # Draw labels
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title("Neural Network Architecture")
    # plt.savefig(filename)
    plt.show()