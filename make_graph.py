from graph import Graph
from elements import Node, Edge, FloorMap

import matplotlib.pyplot as plt

num = 1

graph_old = Graph.load_graph(f'old/graph_{num}.pkl')
nodes = graph_old.Nodes
# Troublshoot - add the video_operations_3.py file from original code

print(graph_old.new_node_index, graph_old.no_of_floors, graph_old.Floor_map, graph_old.path_traversed)
print(nodes)


def run(code: int):
    if code == 0:
        graph = Graph()
        graph.add_floor_map(0)
        
        # Taking Nodes from old saved graphs to avoid marking nodes and connections
        # Ideally to mark do: graph.mark_nodes(0) and graph.make_connections(0)
        # print(graph.Nodes)
        # print(graph.new_node_index, graph.no_of_floors, graph.Floor_map, graph.path_traversed)
        graph.Nodes = nodes
        graph.new_node_index = graph_old.new_node_index
        # print(graph.Nodes)
        # print(graph.new_node_index, graph.no_of_floors, graph.Floor_map, graph.path_traversed)

        img = graph.print_graph_and_return(0)
        plt.imshow(img)
        graph.save_graph(f'graph_{num}.pkl')
        
    if code == 2:
        graph = Graph.load_graph(f'graph_{num}.pkl')
        graph.read_nodes(4)
        graph.read_edges(4)
        graph.save_graph(f'graph_{num}.pkl')


run(0)
run(2)
