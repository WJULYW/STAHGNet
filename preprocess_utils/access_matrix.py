import networkx
import os

import networkx as nx
import pandas as pd

def to_graph(relation, node):
    G=nx.Graph()
    for idx in range(len(relation)):
        G.add_edge(relation['from'][idx], relation['to'][idx])
    print(list(nx.connected_components(G)))


if __name__ == '__main__':
    data_name = 'PEMS07'

    if data_name == 'PeMSD4':
        file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/PeMSD4"))
        file_path = os.path.join(file_dir, 'PeMSD4.npz')
        kg_path = os.path.join(file_dir, 'PeMSD4.csv')
        num_of_vertices = 307
        node_idx = list(range(num_of_vertices))
    elif data_name == 'PeMSD8':
        file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/PeMSD8"))
        file_path = os.path.join(file_dir, 'PeMSD8.npz')
        kg_path = os.path.join(file_dir, 'PeMSD8.csv')
        num_of_vertices = 170
        node_idx = list(range(num_of_vertices))
    elif data_name == 'PEMS03':
        file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/PEMS03"))
        file_path = os.path.join(file_dir, 'pemsd3.npz')
        kg_path = os.path.join(file_dir, 'pemsd3.csv')
        # with open(os.path.join(file_dir, 'PEMS03.txt'), "r") as f:
        #     node_idx = f.readlines()
        # node_idx = [int(s) for s in node_idx]
        num_of_vertices = 358
        node_idx = list(range(num_of_vertices))
    elif data_name == 'PEMS07':
        file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/PEMS07"))
        file_path = os.path.join(file_dir, 'PEMS07.npz')
        kg_path = os.path.join(file_dir, 'PEMS07.csv')
        num_of_vertices = 883
        node_idx = list(range(num_of_vertices))

    temp_list = range(num_of_vertices)
    node_idx_dic = [node_idx[idx] for idx in temp_list]

    all_relation = pd.read_csv(kg_path)

    for i in temp_list:
        in_node = all_relation[all_relation['to'] == node_idx_dic[i]]['from']
        out_node = all_relation[all_relation['from'] == node_idx_dic[i]]['to']
        nodes = list(set(list(in_node) + list(out_node)))

    to_graph(all_relation, node_idx_dic)
